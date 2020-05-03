#!/usr/bin/env python3
# encoding: utf-8

"""
ARC's main module.
To run ARC through its API, first make an instance of the ARC class, then call the .execute() method. For example::

  arc0 = ARC(project='ArcDemo', arc_species_list=[spc0, spc1, spc2])
  arc0.execute()

Where ``spc0``, ``spc1``, and ``spc2`` in the above example are :ref:`ARCSpecies <species>` objects.
"""

import datetime
import logging
import os
import shutil
import time
import yaml
from distutils.spawn import find_executable
from IPython.display import display

from arkane.encorr.corr import assign_frequency_scale_factor
from rmgpy.reaction import Reaction
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.common import VERSION, format_level_of_theory_inputs, format_level_of_theory_for_logging, read_yaml_file, \
    time_lapse, check_ess_settings, initialize_log, log_footer, get_logger, save_yaml_file, initialize_job_types, \
    determine_model_chemistry_type
from arc.exceptions import InputError, SettingsError, SpeciesError
from arc.job.ssh import SSHClient
from arc.processor import process_arc_project
from arc.reaction import ARCReaction
from arc.scheduler import Scheduler
from arc.settings import arc_path, default_levels_of_theory, servers, valid_chars, default_job_types
from arc.species.species import ARCSpecies
from arc.utils.scale import determine_scaling_factors

try:
    from arc.settings import default_job_settings, global_ess_settings
except ImportError:
    global_ess_settings = None


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
        level_of_theory (str, optional): A shortcut representing either sp//geometry levels or a composite method.
                                         e.g., 'CBS-QB3', 'CCSD(T)-F12a/aug-cc-pVTZ//B3LYP/6-311++G(3df,3pd)'...
                                         Notice that this argument does NOT support levels with slashes in the name.
                                         e.g., 'ZINDO/2', 'DLPNO-MP2-F12/D'
                                         For these cases, use the dictionary-type job-specific level of theory arguments
                                         instead (e.g., ``opt_level``).
        composite_method (str, optional): Composite method.
        conformer_level (str or dict, optional): Level of theory for conformer searches.
        opt_level (str or dict, optional): Level of theory for geometry optimization.
        freq_level (str or dict, optional): Level of theory for frequency calculations.
        sp_level (str or dict, optional): Level of theory for single point calculations.
        scan_level (str or dict, optional): Level of theory for rotor scans.
        ts_guess_level (str or dict, optional): Level of theory for comparisons of TS guesses between different methods.
        irc_level (str or dict, optional): The level of theory to use for IRC calculations.
        orbitals_level (str or dict, optional): Level of theory for molecular orbitals calculations.
        use_bac (bool, optional): Whether or not to use bond additivity corrections for thermo calculations.
        job_types (dict, optional): A dictionary of job types to execute. Keys are job types, values are boolean.
        model_chemistry (str, optional): The model chemistry in Arkane for energy corrections (AE, BAC) and
                                         frequencies/ZPE scaling factor. Can usually be determined automatically.
        job_additional_options (dict, optional): Additional specifications to control the execution of a job.
        job_shortcut_keywords (dict, optional): Shortcut keyword specifications to control the execution of a job.
                                                keys are ESS, values are keywords
                                                e.g., {'gaussian': 'iop(99/33=1)'}
        T_min (tuple, optional): The minimum temperature for kinetics computations, e.g., (500, str('K')).
        T_max (tuple, optional): The maximum temperature for kinetics computations, e.g., (3000, str('K')).
        T_count (int, optional): The number of temperature points between ``T_min`` and ``T_max``.
        verbose (int, optional): The logging level to use.
        project_directory (str, optional): The path to the project directory.
        max_job_time (float, optional): The maximal allowed job time on the server in hours (can be fractional).
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
        n_confs (int, optional): The number of lowest force field conformers to consider.
        e_confs (float, optional): The energy threshold in kJ/mol above the lowest energy conformer below which
                                   force field conformers are considered.
        keep_checks (bool, optional): Whether to keep all Gaussian checkfiles when ARC terminates. True to keep,
                                      default is False.
        dont_gen_confs (list, optional): A list of species labels for which conformer generation should be avoided
                                         if xyz is given.
        compare_to_rmg (bool, optional): If ``True`` data calculated from the RMG-database will be calculated and
                                         included on the parity plot.
        solvation (dict, optional): This argument, if not ``None``, requests that a calculation be performed in the presence
                                    of a solvent by placing the solute in a cavity within the solvent reaction field.
                                    Keys are:
                                    - 'method' (optional values: 'pcm' (default), 'cpcm', 'dipole', 'ipcm', 'scipcm')
                                    -  'solvent' (values are strings of "known" solvents, see https://gaussian.com/scrf/,
                                                  default is "water")
        compute_thermo (bool, optional): Whether to compute thermodynamic properties for converged species.
        compute_rates (bool, optional): Whether to compute rate coefficients for converged reactions.
        compute_transport (bool, optional): Whether to compute transport properties for converged species.
        statmech_adapter (str, optional): The statmech software to use.

    Attributes:
        project (str): The project's name. Used for naming the working directory.
        project_directory (str): The path to the project directory.
        arc_species_list (list): A list of :ref:`ARCSpecies <species>` objects.
        arc_rxn_list (list): A list of :ref:`ARCReaction <reaction>` objects.
        level_of_theory (str): A shortcut representing either sp//geometry levels or a composite method,
        composite_method (str): A shortcut that represents a composite method.
        conformer_level (dict): Level of theory for conformer searches.
        opt_level (dict): Level of theory for geometry optimization.
        freq_level (dict): Level of theory for frequency calculations.
        sp_level (dict): Level of theory for single point calculations.
        scan_level (dict): Level of theory for rotor scans.
        ts_guess_level (dict): Level of theory for comparisons of TS guesses between different methods.
        irc_level (dict): The level of theory to use for IRC calculations.
        orbitals_level (dict): Level of theory for molecular orbitals calculations.
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
        job_additional_options (dict): Additional specifications to control the execution of a job.
        job_shortcut_keywords (dict): Shortcut keyword specifications to control the execution of a job.
        t0 (float): Initial time when the project was spawned.
        n_confs (int): The number of lowest force field conformers to consider.
        e_confs (float): The energy threshold in kJ/mol above the lowest energy conformer below which
                         force field conformers are considered.
        execution_time (str): Overall execution time.
        lib_long_desc (str): A multiline description of levels of theory for the outputted RMG libraries.
        running_jobs (dict): A dictionary of jobs submitted in a precious ARC instance, used for restarting ARC.
        T_min (tuple): The minimum temperature for kinetics computations, e.g., (500, str('K')).
        T_max (tuple): The maximum temperature for kinetics computations, e.g., (3000, str('K')).
        T_count (int): The number of temperature points between ``T_min`` and ``T_max``.
        max_job_time (float): The maximal allowed job time on the server in hours (can be fractional).
        rmg_database (RMGDatabase): The RMG database object.
        allow_nonisomorphic_2d (bool): Whether to optimize species even if they do not have a 3D conformer that is
                                       isomorphic to the 2D graph representation.
        memory (int): The total allocated job memory in GB (14 by default to be lower than 90% * 16 GB).
        job_types (dict): A dictionary of job types to execute. Keys are job types, values are boolean.
        specific_job_type (str): Specific job type to execute. Legal strings are job types (keys of job_types dict).
        bath_gas (str): A bath gas. Currently used in OneDMin to calc L-J parameters.
                        Allowed values are He, Ne, Ar, Kr, H2, N2, O2.
        keep_checks (bool): Whether to keep all Gaussian checkfiles when ARC terminates. True to keep, default is False.
        dont_gen_confs (list): A list of species labels for which conformer generation should be avoided
                               if xyz is given.
        compare_to_rmg (bool): If ``True`` data calculated from the RMG-database will be calculated and included on the
                               parity plot.
        solvent (dict): The solvent model and solvent to use.
        compute_thermo (bool): Whether to compute thermodynamic properties for converged species.
        compute_rates (bool): Whether to compute rate coefficients for converged reactions.
        compute_transport (bool): Whether to compute transport properties for converged species.
        statmech_adapter (str): The statmech software to use.
        fine_only (bool): If ``self.job_types['fine'] and not self.job_types['opt']`` ARC will not run optimization
                          jobs without fine=True
    """

    def __init__(self, input_dict=None, project=None, arc_species_list=None, arc_rxn_list=None, level_of_theory='',
                 conformer_level='', composite_method='', opt_level='', freq_level='', sp_level='', scan_level='',
                 ts_guess_level='', irc_level='', orbitals_level='', use_bac=True, job_types=None, model_chemistry='',
                 job_additional_options=None, job_shortcut_keywords=None, T_min=None, T_max=None, T_count=50,
                 verbose=logging.INFO, project_directory=None, max_job_time=None, allow_nonisomorphic_2d=False,
                 job_memory=None, ess_settings=None, bath_gas=None, adaptive_levels=None, freq_scale_factor=None,
                 calc_freq_factor=True, n_confs=10, e_confs=5, dont_gen_confs=None, keep_checks=False,
                 solvation=None, compare_to_rmg=True, compute_thermo=True, compute_rates=True, compute_transport=True,
                 specific_job_type='', statmech_adapter='Arkane'):
        self.__version__ = VERSION
        self.verbose = verbose
        self.output = dict()
        self.running_jobs = dict()
        self.lib_long_desc = ''
        self.unique_species_labels = list()
        self.rmg_database = rmgdb.make_rmg_database_object()
        self.max_job_time = max_job_time or default_job_settings.get('job_time_limit_hrs', 120)
        self.allow_nonisomorphic_2d = allow_nonisomorphic_2d
        self.memory = job_memory or default_job_settings.get('job_total_memory_gb', 14)
        self.ess_settings = dict()
        self.calc_freq_factor = calc_freq_factor
        self.keep_checks = keep_checks
        self.compare_to_rmg = compare_to_rmg

        if input_dict is None:
            if project is None:
                raise ValueError('A project name must be provided for a new project')
            self.project = project
            self.compute_thermo = compute_thermo
            self.compute_rates = compute_rates
            self.compute_transport = compute_transport
            self.statmech_adapter = statmech_adapter
            self.T_min = T_min
            self.T_max = T_max
            self.T_count = T_count
            self.specific_job_type = specific_job_type
            self.job_types = initialize_job_types(job_types, specific_job_type=self.specific_job_type)
            self.bath_gas = bath_gas
            self.solvation = solvation
            self.n_confs = n_confs
            self.e_confs = e_confs
            self.adaptive_levels = adaptive_levels
            self.project_directory = project_directory if project_directory is not None \
                else os.path.join(arc_path, 'Projects', self.project)
            if not os.path.exists(self.project_directory):
                os.makedirs(self.project_directory)
            initialize_log(log_file=os.path.join(self.project_directory, 'arc.log'), project=self.project,
                           project_directory=self.project_directory, verbose=self.verbose)
            self.dont_gen_confs = dont_gen_confs if dont_gen_confs is not None else list()
            self.t0 = time.time()  # init time
            self.execution_time = None
            self.job_additional_options = job_additional_options if job_additional_options is not None else dict()
            self.job_shortcut_keywords = job_shortcut_keywords if job_shortcut_keywords is not None else dict()
            if self.job_additional_options:
                logger.info(f'Use the following user-specified additional job options\n'
                            f'{yaml.dump(self.job_additional_options, default_flow_style=False)}')
            if self.job_shortcut_keywords:
                logger.info(f'Use the following user-specified additional job keywords\n'
                            f'{yaml.dump(self.job_shortcut_keywords, default_flow_style=False)}')
            self.use_bac = use_bac
            self.model_chemistry = model_chemistry
            self.freq_scale_factor = freq_scale_factor

            self.level_of_theory = level_of_theory
            self.composite_method = composite_method
            self.conformer_level = conformer_level
            self.opt_level = opt_level
            self.freq_level = freq_level
            self.sp_level = sp_level
            self.scan_level = scan_level
            self.ts_guess_level = ts_guess_level
            self.irc_level = irc_level
            self.orbitals_level = orbitals_level

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
            self.project_directory = project_directory if project_directory is not None \
                else os.path.abspath(os.path.dirname(input_dict))
            self.from_dict(input_dict=input_dict, project=project, project_directory=self.project_directory)
        if self.adaptive_levels is not None:
            logger.info('Using the following adaptive levels of theory:\n{0}'.format(self.adaptive_levels))
        if not self.ess_settings:
            # don't override self.ess_settings if determined from an input dictionary
            self.ess_settings = check_ess_settings(ess_settings or global_ess_settings)
        if not self.ess_settings:
            self.determine_ess_settings()

        # Determine if fine-only behavior is requested before determining chemistry for job types
        self.fine_only = False
        if self.job_types['fine'] and not self.job_types['opt']:
            self.fine_only = True
            self.job_types['opt'] = True  # Run the optimizations, self.fine_only will make sure that they are fine

        # execute regardless of new job or restart job
        self.determine_model_chemistry_for_job_types()  # all level of theory attributes should be dict after this call
        self.determine_model_chemistry()
        self.scheduler = None
        self.check_project_name()
        self.check_freq_scaling_factor()
        self.restart_dict = self.as_dict()

        if not self.job_types['fine'] and not determine_model_chemistry_type(self.opt_level['method']) == 'dft':
            logger.info('\n')
            logger.warning('Not using a fine DFT grid for geometry optimization jobs')
            logger.info('\n')

        # make a backup copy of the restart file if it exists (but don't save an updated one just yet)
        if os.path.isfile(os.path.join(self.project_directory, 'restart.yml')):
            if not os.path.isdir(os.path.join(self.project_directory, 'log_and_restart_archive')):
                os.mkdir(os.path.join(self.project_directory, 'log_and_restart_archive'))
            local_time = datetime.datetime.now().strftime("%H%M%S_%b%d_%Y")
            restart_backup_name = 'restart.old.' + local_time + '.yml'
            shutil.copy(os.path.join(self.project_directory, 'restart.yml'),
                        os.path.join(self.project_directory, 'log_and_restart_archive', restart_backup_name))

    def as_dict(self) -> dict:
        """
        A helper function for dumping this object as a dictionary in a YAML file for restarting ARC.
        """
        restart_dict = dict()
        restart_dict['project'] = self.project
        if not self.compute_thermo:
            restart_dict['compute_thermo'] = self.compute_thermo
        if not self.compute_rates:
            restart_dict['compute_rates'] = self.compute_rates
        if not self.compute_transport:
            restart_dict['compute_transport'] = self.compute_transport
        restart_dict['statmech_adapter'] = self.statmech_adapter
        if self.bath_gas is not None:
            restart_dict['bath_gas'] = self.bath_gas
        if self.solvation is not None:
            restart_dict['solvation'] = self.solvation
        if self.adaptive_levels is not None:
            restart_dict['adaptive_levels'] = self.adaptive_levels
        restart_dict['job_types'] = self.job_types
        restart_dict['use_bac'] = self.use_bac
        restart_dict['model_chemistry'] = self.model_chemistry

        # attributes related to job model chemistry specifications
        restart_dict['composite_method'] = self.composite_method
        restart_dict['conformer_level'] = self.conformer_level
        restart_dict['opt_level'] = self.opt_level
        restart_dict['freq_level'] = self.freq_level
        restart_dict['sp_level'] = self.sp_level
        restart_dict['scan_level'] = self.scan_level
        restart_dict['ts_guess_level'] = self.ts_guess_level
        restart_dict['irc_level'] = self.irc_level
        restart_dict['orbitals_level'] = self.orbitals_level

        # special treatment for level of theory to avoid conflict during restart
        check_if_empty = (self.composite_method, self.opt_level['method'], self.freq_level['method'],
                          self.sp_level['method'])
        if any(item != '' for item in check_if_empty):
            self.level_of_theory = ''
        restart_dict['level_of_theory'] = self.level_of_theory

        if self.job_additional_options:
            restart_dict['job_additional_options'] = self.job_additional_options
        if self.job_shortcut_keywords:
            restart_dict['job_shortcut_keywords'] = self.job_shortcut_keywords
        if self.freq_scale_factor is not None:
            restart_dict['freq_scale_factor'] = self.freq_scale_factor
        restart_dict['calc_freq_factor'] = self.calc_freq_factor
        if self.dont_gen_confs:
            restart_dict['dont_gen_confs'] = self.dont_gen_confs
        restart_dict['species'] = [spc.as_dict() for spc in self.arc_species_list]
        restart_dict['reactions'] = [rxn.as_dict() for rxn in self.arc_rxn_list]
        restart_dict['output'] = self.output  # if read from_dict then it has actual values
        restart_dict['running_jobs'] = self.running_jobs  # if read from_dict then it has actual values
        restart_dict['T_min'] = self.T_min
        restart_dict['T_max'] = self.T_max
        restart_dict['T_count'] = self.T_count
        restart_dict['max_job_time'] = self.max_job_time
        restart_dict['allow_nonisomorphic_2d'] = self.allow_nonisomorphic_2d
        restart_dict['ess_settings'] = self.ess_settings
        restart_dict['job_memory'] = self.memory
        restart_dict['n_confs'] = self.n_confs
        restart_dict['e_confs'] = self.e_confs
        restart_dict['specific_job_type'] = self.specific_job_type
        if self.keep_checks:
            restart_dict['keep_checks'] = self.keep_checks
        return restart_dict

    def from_dict(self, input_dict, project=None, project_directory=None):
        """
        A helper function for loading this object from a dictionary in a YAML file for restarting ARC.
        If `project` name and `ess_settings` are given as well to __init__, they will override the respective values
        in the restart dictionary.
        """
        if isinstance(input_dict, str):
            input_dict = read_yaml_file(path=input_dict, project_directory=self.project_directory)
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
        self.compute_thermo = input_dict['compute_thermo'] if 'compute_thermo' in input_dict else True
        self.compute_rates = input_dict['compute_rates'] if 'compute_rates' in input_dict else True
        self.compute_transport = input_dict['compute_transport'] if 'compute_transport' in input_dict else True
        self.statmech_adapter = input_dict['statmech_adapter'] if 'statmech_adapter' in input_dict else 'Arkane'
        self.verbose = input_dict['verbose'] if 'verbose' in input_dict else self.verbose
        self.max_job_time = input_dict['max_job_time'] if 'max_job_time' in input_dict else self.max_job_time
        self.memory = input_dict['job_memory'] if 'job_memory' in input_dict else self.memory
        self.bath_gas = input_dict['bath_gas'] if 'bath_gas' in input_dict else None
        self.solvation = input_dict['solvation'] if 'solvation' in input_dict else None
        self.n_confs = input_dict['n_confs'] if 'n_confs' in input_dict else 10
        self.e_confs = input_dict['e_confs'] if 'e_confs' in input_dict else 5  # kJ/mol
        self.adaptive_levels = input_dict['adaptive_levels'] if 'adaptive_levels' in input_dict else None
        self.keep_checks = input_dict['keep_checks'] if 'keep_checks' in input_dict else False
        self.allow_nonisomorphic_2d = input_dict['allow_nonisomorphic_2d'] \
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
                                    logger.debug(f'corrected path to {os.path.join(arc_path, val)}')
                                elif os.path.isfile(os.path.join(arc_path, 'Projects', val)):
                                    self.output[label]['paths'][key] = os.path.join(arc_path, 'Projects', val)
                                    logger.debug(f'corrected path to {os.path.join(arc_path, val)}')
                                else:
                                    raise SpeciesError(f'Could not find {key} output file for species {label}: {val}')
        self.running_jobs = input_dict['running_jobs'] if 'running_jobs' in input_dict else dict()
        logger.debug(f'output dictionary successfully parsed:\n{self.output}')
        self.T_min = input_dict['T_min'] if 'T_min' in input_dict else None
        self.T_max = input_dict['T_max'] if 'T_max' in input_dict else None
        self.T_count = input_dict['T_count'] if 'T_count' in input_dict else None
        self.job_additional_options = input_dict['job_additional_options'] if 'job_additional_options' \
                                                                              in input_dict else dict()
        self.job_shortcut_keywords = input_dict['job_shortcut_keywords'] if 'job_shortcut_keywords' \
                                                                            in input_dict else dict()
        self.specific_job_type = input_dict['specific_job_type'] if 'specific_job_type' in input_dict else None
        self.job_types = input_dict['job_types'] if 'job_types' in input_dict else default_job_types
        self.job_types = initialize_job_types(self.job_types, specific_job_type=self.specific_job_type)
        self.use_bac = input_dict['use_bac'] if 'use_bac' in input_dict else True
        self.calc_freq_factor = input_dict['calc_freq_factor'] if 'calc_freq_factor' in input_dict else True

        # attributes related to job model chemistry specifications
        self.level_of_theory = input_dict['level_of_theory'] if 'level_of_theory' in input_dict else ''
        self.composite_method = input_dict['composite_method'] if 'composite_method' in input_dict else ''
        self.conformer_level = input_dict['conformer_level'] if 'conformer_level' in input_dict else ''
        self.opt_level = input_dict['opt_level'] if 'opt_level' in input_dict else ''
        self.freq_level = input_dict['freq_level'] if 'freq_level' in input_dict else ''
        self.sp_level = input_dict['sp_level'] if 'sp_level' in input_dict else ''
        self.scan_level = input_dict['scan_level'] if 'scan_level' in input_dict else ''
        self.ts_guess_level = input_dict['ts_guess_level'] if 'ts_guess_level' in input_dict else ''
        self.irc_level = input_dict['irc_level'] if 'irc_level' in input_dict else ''
        self.orbitals_level = input_dict['orbitals_level'] if 'orbitals_level' in input_dict else ''

        ess_settings = input_dict['ess_settings'] if 'ess_settings' in input_dict else global_ess_settings
        self.ess_settings = check_ess_settings(ess_settings)
        self.dont_gen_confs = input_dict['dont_gen_confs'] if 'dont_gen_confs' in input_dict else list()
        self.model_chemistry = input_dict['model_chemistry'] if 'model_chemistry' in input_dict else ''
        if not self.job_types['fine']:
            logger.info('\n')
            logger.warning('Not using a fine grid for geometry optimization jobs')
            logger.info('\n')
        if not self.job_types['rotors']:
            logger.info('\n')
            logger.warning("Not running rotor scans. "
                           "This might compromise finding the best conformer, as dihedral angles won't be "
                           "corrected. Also, entropy won't be accurate.")
            logger.info('\n')

        if 'species' in input_dict:
            self.arc_species_list = [ARCSpecies(species_dict=spc_dict) for spc_dict in input_dict['species']]
            for spc in self.arc_species_list:
                if spc.rotors_dict is not None:
                    for rotor_num, rotor_dict in spc.rotors_dict.items():
                        if rotor_dict['scan_path'] and not os.path.isfile(rotor_dict['scan_path']) \
                                and rotor_dict['success']:
                            # try correcting relative paths
                            if os.path.isfile(os.path.join(arc_path, rotor_dict['scan_path'])):
                                spc.rotors_dict[rotor_num]['scan_path'] = os.path.join(arc_path, rotor_dict['scan_path'])
                            elif os.path.isfile(os.path.join(arc_path, 'Projects', rotor_dict['scan_path'])):
                                spc.rotors_dict[rotor_num]['scan_path'] = \
                                    os.path.join(arc_path, 'Projects', rotor_dict['scan_path'])
                            else:
                                raise SpeciesError(f'Could not find rotor scan output file for rotor {rotor_num} of '
                                                   f'species {spc.label}: {rotor_dict["scan_path"]}')
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
        logger.info(f'\n\nWriting input file to {path}')
        save_yaml_file(path=path, content=self.restart_dict)

    def execute(self) -> dict:
        """
        Execute ARC.

        Returns:
            dict: Status dictionary indicating which species converged successfully.
        """
        logger.info('\n')
        for species in self.arc_species_list:
            if not isinstance(species, ARCSpecies):
                raise ValueError(f'All species in arc_species_list must be ARCSpecies objects. Got {type(species)}')
            if species.is_ts:
                logger.info(f'Considering transition state: {species.label}')
            else:
                logger.info(f'Considering species: {species.label}')
                if species.mol is not None:
                    display(species.mol.copy(deep=True))
        logger.info('\n')
        for rxn in self.arc_rxn_list:
            if not isinstance(rxn, ARCReaction):
                raise ValueError(f'All reactions in arc_rxn_list must be ARCReaction objects. Got {type(rxn)}')
        self.scheduler = Scheduler(project=self.project, species_list=self.arc_species_list, rxn_list=self.arc_rxn_list,
                                   composite_method=self.composite_method, conformer_level=self.conformer_level,
                                   opt_level=self.opt_level, freq_level=self.freq_level, sp_level=self.sp_level,
                                   scan_level=self.scan_level, ts_guess_level=self.ts_guess_level,
                                   irc_level=self.irc_level, orbitals_level=self.orbitals_level,
                                   ess_settings=self.ess_settings, job_types=self.job_types, bath_gas=self.bath_gas,
                                   job_additional_options=self.job_additional_options, solvation=self.solvation,
                                   job_shortcut_keywords=self.job_shortcut_keywords, rmg_database=self.rmg_database,
                                   restart_dict=self.restart_dict, project_directory=self.project_directory,
                                   max_job_time=self.max_job_time, allow_nonisomorphic_2d=self.allow_nonisomorphic_2d,
                                   memory=self.memory, adaptive_levels=self.adaptive_levels,
                                   n_confs=self.n_confs, e_confs=self.e_confs, dont_gen_confs=self.dont_gen_confs,
                                   fine_only=self.fine_only)

        save_yaml_file(path=os.path.join(self.project_directory, 'output', 'status.yml'), content=self.scheduler.output)

        if not self.keep_checks:
            self.delete_check_files()

        self.save_project_info_file()

        sp_level = self.model_chemistry.split('//')[0] if '//' in self.model_chemistry else self.model_chemistry
        process_arc_project(statmech_adapter=self.statmech_adapter.lower(),
                            project=self.project,
                            project_directory=self.project_directory,
                            species_dict=self.scheduler.species_dict,
                            reactions=self.scheduler.rxn_list,
                            output_dict=self.scheduler.output,
                            use_bac=self.use_bac,
                            sp_level=sp_level,
                            freq_scale_factor=self.freq_scale_factor,
                            compute_thermo=self.compute_thermo,
                            compute_rates=self.compute_rates,
                            compute_transport=self.compute_transport,
                            T_min=self.T_min,
                            T_max=self.T_max,
                            T_count=self.T_count or 50,
                            lib_long_desc=self.lib_long_desc,
                            rmg_database=self.rmg_database,
                            compare_to_rmg=self.compare_to_rmg)

        status_dict = self.summary()
        log_footer(execution_time=self.execution_time)
        return status_dict

    def save_project_info_file(self):
        """
        Save a project info file.
        """
        self.execution_time = time_lapse(t0=self.t0)
        path = os.path.join(self.project_directory, f'{self.project}.info')
        if os.path.exists(path):
            os.remove(path)
        if self.job_types['fine']:
            fine_txt = '(using a fine grid)'
        else:
            fine_txt = '(NOT using a fine grid)'

        txt = ''
        txt += f'ARC v{self.__version__}\n'
        txt += f'ARC project {self.project}\n\nLevels of theory used:\n\n'
        txt += f'Conformers:       {format_level_of_theory_for_logging(self.conformer_level)}\n'
        txt += f'TS guesses:       {format_level_of_theory_for_logging(self.ts_guess_level)}\n'
        if self.composite_method:
            txt += f'Composite method: {self.composite_method} {fine_txt}\n'
            txt += f'Frequencies:      {format_level_of_theory_for_logging(self.freq_level)}\n'
        else:
            txt += f'Optimization:     {format_level_of_theory_for_logging(self.opt_level)} {fine_txt}\n'
            txt += f'Frequencies:      {format_level_of_theory_for_logging(self.freq_level)}\n'
            txt += f'Single point:     {format_level_of_theory_for_logging(self.sp_level)}\n'
        if self.solvation is not None:
            txt += f'Solvation model:  {self.solvation}\n'
        if 'rotors' in self.job_types:
            txt += f'Rotor scans:      {format_level_of_theory_for_logging(self.scan_level)}\n'
        else:
            txt += 'Not scanning rotors\n'
        if self.use_bac:
            txt += 'Using bond additivity corrections for thermo\n'
        else:
            txt += 'NOT using bond additivity corrections for thermo\n'
        if self.job_additional_options:
            txt += f'Using additional job options "{yaml.dump(self.job_additional_options, default_flow_style=False)}"'
        if self.job_shortcut_keywords:
            txt += f'Using additional job keywords "{yaml.dump(self.job_shortcut_keywords, default_flow_style=False)}"'
        txt += f'\nUsing the following ESS settings: {self.ess_settings}\n'
        txt += '\nConsidered the following species and TSs:\n'
        for species in self.arc_species_list:
            descriptor = 'TS' if species.is_ts else 'Species'
            failed = '' if self.scheduler.output[species.label]['convergence'] else ' (Failed!)'
            txt += f'{descriptor} {species.label}{failed} (run time: {species.run_time})\n'
        if self.arc_rxn_list:
            for rxn in self.arc_rxn_list:
                txt += f'Considered reaction: {rxn.label}\n'
        txt += f'\nOverall time since project initiation: {self.execution_time}'
        txt += '\n'

        with open(path, 'w') as f:
            f.write(str(txt))
        self.lib_long_desc = txt

    def summary(self) -> dict:
        """
        Report status and data of all species / reactions.

        Returns:
            dict: Status dictionary indicating which species converged successfully.
        """
        status_dict = {}
        logger.info(f'\n\n\nAll jobs terminated. Summary for project {self.project}:\n')
        for label, output in self.scheduler.output.items():
            if output['convergence']:
                status_dict[label] = True
                logger.info(f'Species {label} converged successfully\n')
            else:
                status_dict[label] = False
                job_type_status = {key: val for key, val in self.output[label]['job_types'].items()
                                   if key in self.job_types and self.job_types[key]}
                logger.info(f'  Species {label} failed with status:\n  {job_type_status}')
                keys = ['conformers', 'isomorphism', 'info']
                for key in keys:
                    if key in output and output[key]:
                        logger.info(output[key])
                if 'warnings' in output and output['warnings']:
                    logger.info(f'  and warnings: {output["warnings"]}')
                if 'errors' in output and output['errors']:
                    logger.info(f'  and errors: {output["errors"]}')
                logger.info('\n')
        return status_dict

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
                    'b3lyp/6-31g(d,p)']:
                logger.warning('No bond additivity corrections (BAC) are available in Arkane for "model chemistry"'
                               ' {0}. As a result, thermodynamic parameters are expected to be inaccurate. Make sure'
                               ' that atom energy corrections (AEC) were supplied or are available in Arkane to avoid'
                               ' error.'.format(self.model_chemistry))
        else:
            # model chemistry was not given, try to determine it from the sp_level and freq_level
            self.model_chemistry = ''
            if self.composite_method:
                self.model_chemistry = self.composite_method.lower()
            else:
                sp_level = self.sp_level['method'] + '/' + self.sp_level['basis']
                sp_level = sp_level.replace('f12a', 'f12').replace('f12b', 'f12').lower()
                freq_level = self.freq_level['method'] + '/' + self.freq_level['basis']
                freq_level = freq_level.replace('f12a', 'f12').replace('f12b', 'f12').lower()
                if sp_level in ['ccsd(t)-f12/cc-pvdz', 'ccsd(t)-f12/cc-pvtz', 'ccsd(t)-f12/cc-pvqz']:
                    logger.warning('Using model chemistry {0} based on sp level {1}.'.format(
                        sp_level + '-f12', sp_level))
                    sp_level += '-f12'
                if sp_level not in ['ccsd(t)-f12/cc-pvdz-f12', 'ccsd(t)-f12/cc-pvtz-f12', 'ccsd(t)-f12/cc-pvqz-f12',
                                    'b3lyp/cbsb7', 'b3lyp/6-311g(2d,d,p)', 'b3lyp/6-311+g(3df,2p)', 'b3lyp/6-31g(d,p)']\
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
                                      'fci/cc-pvtz', 'fci/cc-pvqz', 'bmk/cbsb7', 'bmk/6-311g(2d,d,p)',
                                      'b3lyp/6-31g(d,p)', 'b3lyp/6-311+g(3df,2p)', 'MRCI+Davidson/aug-cc-pV(T+d)Z']:
                    logger.warning('Could not determine a Model Chemistry to be used in Arkane, '
                                   'NOT calculating thermodata')
                    for spc in self.arc_species_list:
                        spc.compute_thermo = False
                self.model_chemistry = sp_level + '//' + freq_level
        if self.model_chemistry:
            logger.info(f'Using {self.model_chemistry} as a model chemistry in Arkane')

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

        for software in ['gaussian', 'molpro', 'onedmin', 'orca', 'qchem', 'terachem']:
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
                    logger.info(f'Found Gaussian: g03={g03}, g09={g09}, g16={g16}')
                self.ess_settings['gaussian'] = ['local']
            qchem = find_executable('qchem')
            if qchem:
                self.ess_settings['qchem'] = ['local']
            orca = find_executable('orca')
            if orca:
                self.ess_settings['orca'] = ['local']
            molpro = find_executable('molpro')
            if molpro:
                self.ess_settings['molpro'] = ['local']
            terachem = find_executable('terachem')
            if terachem:
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
                    logger.info(f'  Found Gaussian on {server}: g03={g03}, g09={g09}, g16={g16}')
                self.ess_settings['gaussian'].append(server)
            elif diagnostics:
                logger.info(f'  Did NOT find Gaussian on {server}')

            cmd = '. ~/.bashrc; which qchem'
            qchem = ssh.send_command_to_server(cmd)[0]
            if qchem:
                if diagnostics:
                    logger.info(f'  Found QChem on {server}')
                self.ess_settings['qchem'].append(server)
            elif diagnostics:
                logger.info(f'  Did NOT find QChem on {server}')

            cmd = '. ~/.bashrc; which orca'
            orca = ssh.send_command_to_server(cmd)[0]
            if orca:
                if diagnostics:
                    logger.info(f'  Found Orca on {server}')
                self.ess_settings['orca'].append(server)
            elif diagnostics:
                logger.info(f'  Did NOT find Orca on {server}')

            cmd = '. ~/.bashrc; which terachem'
            terachem = ssh.send_command_to_server(cmd)[0]
            if terachem:
                if diagnostics:
                    logging.info(f'  Found TeraChem on {server}')
                self.ess_settings['terachem'].append(server)
            elif diagnostics:
                logging.info(f'  Did NOT find TeraChem on {server}')

            cmd = '. .bashrc; which molpro'
            molpro = ssh.send_command_to_server(cmd)[0]
            if molpro:
                if diagnostics:
                    logger.info(f'  Found Molpro on {server}')
                self.ess_settings['molpro'].append(server)
            elif diagnostics:
                logger.info(f'  Did NOT find Molpro on {server}')
        if diagnostics:
            logger.info('\n\n')
        if 'gaussian' in self.ess_settings.keys():
            logger.info(f'Using Gaussian on {self.ess_settings["gaussian"]}')
        if 'qchem' in self.ess_settings.keys():
            logger.info(f'Using QChem on {self.ess_settings["qchem"]}')
        if 'orca' in self.ess_settings.keys():
            logger.info(f'Using Orca on {self.ess_settings["orca"]}')
        if 'molpro' in self.ess_settings.keys():
            logger.info(f'Using Molpro on {self.ess_settings["molpro"]}')
        logger.info('\n')

        if 'gaussian' not in self.ess_settings.keys() and 'qchem' not in self.ess_settings.keys() \
                and 'orca' not in self.ess_settings.keys() and 'molpro' not in self.ess_settings.keys()\
                and 'onedmin' not in self.ess_settings.keys() and not diagnostics:
            raise SettingsError('Could not find any ESS. Check your .bashrc definitions on the server.\n'
                                'Alternatively, you could pass a software-server dictionary to arc as `ess_settings`')
        elif diagnostics:
            logger.info(f'ESS diagnostics completed (elapsed time: {time_lapse(t0)})')

    def check_project_name(self):
        """
        Check the validity of the project name.
        """
        for char in self.project:
            if char not in valid_chars:
                raise InputError(f'A project name (used to naming folders) must contain only valid characters. '
                                 f'Got {char} in {self.project}.')
            if char == ' ':  # space IS a valid character for other purposes, but isn't valid in project names
                raise InputError(f'A project name (used to naming folders) must not contain spaces. Got {self.project}.')

    def check_freq_scaling_factor(self):
        """
        Check that the harmonic frequencies scaling factor is known,
        otherwise spawn a calculation for it if calc_freq_factor is set to True.
        """
        if self.freq_scale_factor is None:
            # the user did not specify a scaling factor, see if Arkane has it
            if not self.composite_method:
                level = self.freq_level['method'] + '/' + self.freq_level['basis']
            else:
                level = self.composite_method
            freq_scale_factor = assign_frequency_scale_factor(level)
            if freq_scale_factor != 1:
                # Arkane has this harmonic frequencies scaling factor (if not found, the factor is set to exactly 1)
                self.freq_scale_factor = freq_scale_factor
            else:
                logger.info(f'Could not determine the harmonic frequencies scaling factor for {level} from Arkane.')
                if self.calc_freq_factor:
                    logger.info("Calculating it using Truhlar's method:\n\n")
                    self.freq_scale_factor = determine_scaling_factors(
                        level, ess_settings=self.ess_settings, init_log=False)[0]
                else:
                    logger.info('Not calculating it, assuming a frequencies scaling factor of 1.')

    def delete_check_files(self):
        """
        Delete Gaussian and TeraChem checkfiles. They usually take up lots of space
        and are not needed after ARC terminates.
        Pass ``True`` to the ``keep_checks`` flag in ARC to avoid deleting check files.
        """
        logged = False
        calcs_path = os.path.join(self.project_directory, 'calcs')
        for (root, _, files) in os.walk(calcs_path):
            for file_ in files:
                if os.path.splitext(file_)[1] == '.chk' and os.path.isfile(os.path.join(root, file_)):
                    if not logged:
                        logger.info('\ndeleting all check files...\n')
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
                raise ValueError(f'Species label {arc_spc.label} is not unique')

    def add_hydrogen_for_bde(self):
        """
        Make sure ARC has a hydrogen species labeled as 'H' for the final processing of bde jobs (if not, create one).
        """
        if any([spc.bdes is not None for spc in self.arc_species_list]):
            for species in self.arc_species_list:
                if species.label == 'H':
                    if species.number_of_atoms == 1 and species.get_xyz(generate=True)['symbols'][0] == 'H':
                        break
                    else:
                        raise SpeciesError(f'A species with label "H" was defined, but does not seem to be '
                                           f'the hydrogen atom species. Cannot calculate bond dissociation energies.\n'
                                           f'Got the following species: {[spc.label for spc in self.arc_species_list]}')
            else:
                # no H species defined, make one
                h = ARCSpecies(label='H', smiles='[H]', compute_thermo=False, e0_only=True)
                self.arc_species_list.append(h)

    def determine_model_chemistry_for_job_types(self):
        """
        Determine method (e.g., b3lyp), basis set (e.g. def2-svp), and auxiliary basis set (e.g., def2-svp/c) for
        supported ARC job types (e.g., opt, freq, sp, scan, conformer, ts_guesses) from user inputs or defaults.
        """
        self._initialize_model_chemistry_inputs()
        # examples of expected job level formats after calling this initialization (other formats will raise an error)
        # level_of_theory = wb97xd/def2tzvp//b3lyp/def2svp or wb97xd/def2tzvp or ''
        # composite_method = cbs-qb3 or ''
        # job_level = '' or {'method': 'b3lyp', 'basis': 'def2-tzvp', 'auxiliary_basis': '', 'dispersion': ''}
        # job_level will NOT equal to {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        # level_of_theory and composite_method cannot be specified simultaneously
        # opt_level, freq_level, sp_level will overwrite level_of_theory if they are specified simultaneously

        if not self.conformer_level:
            self.conformer_level = default_levels_of_theory['conformer']
            default_flag = ' default'
        else:
            default_flag = ''
        self.conformer_level, level = format_level_of_theory_inputs(self.conformer_level)
        logger.info(f'Using{default_flag} level {format_level_of_theory_for_logging(level)} for refined conformer '
                    f'searches (after filtering via force fields).')

        if not self.ts_guess_level:
            self.ts_guess_level = default_levels_of_theory['ts_guesses']
            default_flag = ' default'
        else:
            default_flag = ''
        self.ts_guess_level, level = format_level_of_theory_inputs(self.ts_guess_level)
        logger.info(f'Using{default_flag} level {format_level_of_theory_for_logging(level)} for optimizing all TS '
                    f'initial guesses (before a refined TS search at a higher level.')

        if self.level_of_theory:
            # For composite methods self.level_of_theory is an empty string set by calling
            # initialize_model_chemistry_inputs.
            if '//' in self.level_of_theory:
                self.opt_level = format_level_of_theory_inputs(self.level_of_theory.split('//')[1])[0]
                self.freq_level = format_level_of_theory_inputs(self.level_of_theory.split('//')[1])[0]
                self.sp_level = format_level_of_theory_inputs(self.level_of_theory.split('//')[0])[0]
            else:
                # assume this is not a composite method, and the user meant to run opt, freq and sp at this level.
                self.opt_level = format_level_of_theory_inputs(self.level_of_theory)[0]
                self.freq_level = format_level_of_theory_inputs(self.level_of_theory)[0]
                self.sp_level = format_level_of_theory_inputs(self.level_of_theory)[0]

        if self.composite_method:
            logger.info(f'Using composite method {self.composite_method}')
            self.opt_level = format_level_of_theory_inputs('')[0]
            self.sp_level = format_level_of_theory_inputs('')[0]

            if not self.freq_level:
                self.freq_level = default_levels_of_theory['freq_for_composite']
                default_flag = ' default'
            else:
                default_flag = ''
            self.freq_level, level = format_level_of_theory_inputs(self.freq_level)
            logger.info(f'Using{default_flag} level {format_level_of_theory_for_logging(level)} for frequency '
                        f'calculations after composite jobs.')

            if self.job_types['rotors']:
                if not self.scan_level:
                    self.scan_level = default_levels_of_theory['scan_for_composite']
                    default_flag = ' default'
                else:
                    default_flag = ''
                self.scan_level, level = format_level_of_theory_inputs(self.scan_level)
                logger.info(f'Using{default_flag} level {format_level_of_theory_for_logging(level)} for rotor scans '
                            f'after composite jobs.')
            else:
                self.scan_level = format_level_of_theory_inputs('')[0]
                logger.warning("Not running rotor scans, since it was not requested by the user. This might compromise "
                               "finding the best conformer, as dihedral angles won't be corrected. "
                               "Also, entropy won't be accurate.")

            if self.job_types['irc']:
                if not self.irc_level:
                    self.irc_level = default_levels_of_theory['irc_for_composite']
                    default_flag = ' default'
                else:
                    default_flag = ''
                self.irc_level, level = format_level_of_theory_inputs(self.irc_level)
                logger.info(f'Using{default_flag} level {format_level_of_theory_for_logging(level)} for visualizing '
                            f'molecular orbitals after composite jobs.')
            else:
                self.irc_level = format_level_of_theory_inputs('')[0]
                logger.debug("Not running molecular orbitals visualization.")

            if self.job_types['orbitals']:
                if not self.orbitals_level:
                    self.orbitals_level = default_levels_of_theory['orbitals_for_composite']
                    default_flag = ' default'
                else:
                    default_flag = ''
                self.orbitals_level, level = format_level_of_theory_inputs(self.orbitals_level)
                logger.info(f'Using{default_flag} level {format_level_of_theory_for_logging(level)} for visualizing '
                            f'molecular orbitals after composite jobs.')
            else:
                self.orbitals_level = format_level_of_theory_inputs('')[0]
                logger.debug("Not running molecular orbitals visualization.")

        else:
            # this is not a composite method
            if self.job_types['opt']:
                if not self.opt_level:
                    self.opt_level = default_levels_of_theory['opt']
                    default_flag = ' default'
                else:
                    default_flag = ''
                self.opt_level, level = format_level_of_theory_inputs(self.opt_level)
                logger.info(f'Using{default_flag} level {format_level_of_theory_for_logging(level)} for geometry '
                            f'optimizations.')
            else:
                self.opt_level = format_level_of_theory_inputs('')[0]
                logger.warning("Not running geometry optimization, since it was not requested by the user.")

            if self.job_types['freq']:
                if not self.freq_level:
                    if self.opt_level:
                        self.freq_level = self.opt_level
                        info, default_flag = ' user-defined opt', ''
                    else:
                        self.freq_level = default_levels_of_theory['freq']
                        info, default_flag = '', ' default'
                else:
                    info, default_flag = '', ''
                self.freq_level, level = format_level_of_theory_inputs(self.freq_level)
                logger.info(f'Using{info}{default_flag} level {format_level_of_theory_for_logging(level)} for frequency'
                            f' calculations.')
            else:
                self.freq_level = format_level_of_theory_inputs('')[0]
                logger.warning("Not running frequency calculation, since it was not requested by the user.")

            if self.job_types['sp']:
                if not self.sp_level:
                    self.sp_level = default_levels_of_theory['sp']
                    default_flag = ' default'
                else:
                    default_flag = ''
                self.sp_level, level = format_level_of_theory_inputs(self.sp_level)
                logger.info(f'Using{default_flag} level {format_level_of_theory_for_logging(level)} for single point '
                            f'calculations.')
            else:
                self.sp_level = format_level_of_theory_inputs('')[0]
                logger.warning("Not running single point calculation, since it was not requested by the user.")

            if self.job_types['rotors']:
                if not self.scan_level:
                    if self.opt_level:
                        self.scan_level = self.opt_level
                        info, default_flag = ' user-defined opt', ''
                    else:
                        self.scan_level = default_levels_of_theory['scan']
                        info, default_flag = '', ' default'
                else:
                    info, default_flag = '', ''
                self.scan_level, level = format_level_of_theory_inputs(self.scan_level)
                logger.info(f'Using{info}{default_flag} level {format_level_of_theory_for_logging(level)} '
                            f'for rotor scans.')
            else:
                self.scan_level = format_level_of_theory_inputs('')[0]
                logger.warning("Not running rotor scans, since it was not requested by the user. This might compromise "
                               "finding the best conformer, as dihedral angles won't be corrected. "
                               "Also, entropy won't be accurate.")

            if self.job_types['irc']:
                if not self.irc_level:
                    self.irc_level = default_levels_of_theory['irc']
                    default_flag = ' default'
                else:
                    default_flag = ''
                self.irc_level, level = format_level_of_theory_inputs(self.irc_level)
                logger.info(f'Using{default_flag} level {format_level_of_theory_for_logging(level)} for '
                            f'IRC calculations.')
            else:
                self.irc_level = format_level_of_theory_inputs('')[0]
                logger.debug("Not running IRC computations, since it was not requested by the user.")

            if self.job_types['orbitals']:
                if not self.orbitals_level:
                    self.orbitals_level = default_levels_of_theory['orbitals']
                    default_flag = ' default'
                else:
                    default_flag = ''
                self.orbitals_level, level = format_level_of_theory_inputs(self.orbitals_level)
                logger.info(f'Using{default_flag} level {format_level_of_theory_for_logging(level)} for visualizing '
                            f'molecular orbitals.')
            else:
                self.orbitals_level = format_level_of_theory_inputs('')[0]
                logger.debug("Not running molecular orbitals visualization, since it was not requested by the user.")

    def _initialize_model_chemistry_inputs(self):
        """
        A helper function to initialize model chemistry inputs to expected default formats and identify illegal inputs
        for internal use in ARC.
        """
        # initialization (see comments below for expected outcomes)
        self._initialize_job_level_formats()
        # examples of expected job level formats after calling this initialization (other formats will raise an error)
        # job_level = '' or {'method': 'b3lyp', 'basis': 'def2-tzvp', 'auxiliary_basis': '', 'dispersion': ''}
        # job_level will NOT equal to {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}

        # level_of_theory and composite_method must be strings
        if not isinstance(self.level_of_theory, str):
            raise InputError(f'level_of_theory must be a string. Got {self.level_of_theory} which is a '
                             f'{type(self.level_of_theory)}.')
        if not isinstance(self.composite_method, str):
            raise InputError(f'composite_method must be a string. Got {self.composite_method} which is a '
                             f'{type(self.composite_method)}.')

        # level of theory and composite method can NOT be both specified
        if self.level_of_theory and self.composite_method:
            raise InputError(f'Specify either composite_method or level_of_theory but not both. Got \n'
                             f'level_of_theory: {self.level_of_theory} \n'
                             f'composite_method: {self.composite_method}')

        # check if level_of_theory specification is legal
        if self.level_of_theory:
            if ' ' in self.level_of_theory:
                raise InputError(f'level_of_theory seems wrong. It should not contain empty spaces.\n'
                                 f'Got: {self.level_of_theory} \n'
                                 f'See the documentation if you intended to include an auxiliary basis set or DFT '
                                 f'dispersion.')
            if self.level_of_theory.count('//') > 1:
                raise InputError(f'level_of_theory seems wrong. It should either be a composite method (like CBS-QB3) '
                                 f'or be of the form sp//geometry, e.g., CCSD(T)-F12/pvtz//wB97x-D3/6-311++g**.\n'
                                 f'Got: {self.level_of_theory}')
            if not all(item.count('/') < 2 for item in self.level_of_theory.split('//')):
                raise InputError(f'level_of_theory does not support method or basis set with multiple slashes.\n'
                                 f'Got: {self.level_of_theory} \n'
                                 f'Please specify opt_level, freq_level, sp_level directly in this case.\n'
                                 f'e.g., dlpno-mp2-f12/D/cc-pVDZ(fi/sf/fw)//b3lyp/G/def2svp is equivalent to \n'
                                 'opt_level = freq_level = {method: b3lyp/G, basis: cc-pVDZ(fi/sf/fw)} \n'
                                 'sp_level = {method: dlpno-mp2-f12/D, basis: cc-pVDZ(fi/sf/fw)} \n'
                                 'See the documentation if you intended to include an auxiliary basis set or DFT '
                                 'dispersion.')

            further_check = True
            standard_level_of_theory_format = True  # e.g., wb97xd/def2tzvp//b3lyp/def2svp
            if '/' not in self.level_of_theory:
                if determine_model_chemistry_type(self.level_of_theory) == 'composite':
                    self.composite_method = self.level_of_theory.lower()
                    logger.info(f'Given level_of_theory {self.level_of_theory} seems to be a composite method.')
                    # it will cause conflict with composite method at restart without reset level_of_theory here
                    self.level_of_theory = ''
                    further_check = False
                else:
                    raise InputError(f"Given level_of_theory {self.level_of_theory} is not in standard format (no "
                                     f"single slash '/' appears in the name) and seems not to be a composite method. "
                                     f"Please first check the spelling. For example, cbs-qb3 has a dash. If you "
                                     f"intended to specify a semi-empirical method such as AM1, please use the "
                                     f"dictionary-style job level format as described in the documentation.")
            elif '//' not in self.level_of_theory:
                # assume this is not a composite method, and the user meant to run opt, freq and sp at this level.
                logger.info(f'Given level_of_theory {self.level_of_theory} is not in standard format. Because '
                            f'no double slash (//) appear in the name, ARC assumes optimization, frequency, '
                            f'and single point calculation are all at this level.')
                standard_level_of_theory_format = False  # e.g., wb97xd/def2tzvp

            # Check if there are conflicts between level of theory and opt, freq, sp levels
            if further_check:
                if standard_level_of_theory_format:
                    sp_, opt_ = format_level_of_theory_inputs(self.level_of_theory.split('//')[0]), \
                                format_level_of_theory_inputs(self.level_of_theory.split('//')[1])
                else:
                    sp_ = opt_ = format_level_of_theory_inputs(self.level_of_theory)[0]

                if self.opt_level and self.opt_level != opt_:
                    raise InputError(f'It does not make sense to specify different opt_level and level_of_theory. '
                                     f'Please choose the correct one.\n'
                                     f'opt_level: {self.opt_level} \n'
                                     f'level_of_theory {self.level_of_theory}')

                if self.freq_level and self.freq_level != opt_:
                    raise InputError(f'It does not make sense to specify different freq_level and level_of_theory. '
                                     f'Please choose the correct one.\n'
                                     f'freq_level: {self.freq_level} \n'
                                     f'level_of_theory {self.level_of_theory}')

                if self.sp_level and self.sp_level != sp_:
                    raise InputError(f'It does not make sense to specify different sp_level and level_of_theory. '
                                     f'Please choose the correct one.\n'
                                     f'sp_level: {self.sp_level} \n'
                                     f'level_of_theory {self.level_of_theory}')

        # check if composite_method specification is legal
        if self.composite_method:
            self.composite_method = self.composite_method.lower()

    def _initialize_job_level_formats(self):
        """
        A helper function to initialize job level of theory inputs to expected default formats and identify illegal
        inputs for internal use in ARC.
        """
        # check if job level specification is legal
        empty_model_chemistry_input_dict = {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}

        # developer notice: hard code the following section is intended for code readability and maintainability
        # a less desired alternative is to make a tuple with str(job_types) as attribute names
        # then use a for loop with getattr() and setattr() methods to achieve the same goal as the following code
        # this alternative method can make code shorter, but is not good practice because if later the attribute names
        # are refactored, str(job_types) will not be updated automatically and might cause bug if overlooked.

        # for each job type, initialize self.job_type = '' for the following cases
        # self.job_type = None or {} or {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        # this initialization is important for determine_model_chemistry_for_job_types to function properly
        # check for conformer level specification
        if (not self.conformer_level) or (self.conformer_level == empty_model_chemistry_input_dict):
            self.conformer_level = ''
        else:
            try:
                self.conformer_level, _ = format_level_of_theory_inputs(self.conformer_level)
            except InputError as e:
                raise InputError(f'Specified model chemistry input dictionary for conformer seems wrong. {e!s}')

        # check for opt level specification
        if (not self.opt_level) or (self.opt_level == empty_model_chemistry_input_dict):
            self.opt_level = ''
        else:
            try:
                self.opt_level, _ = format_level_of_theory_inputs(self.opt_level)
            except InputError as e:
                raise InputError(f'Specified model chemistry input dictionary for optimization seems wrong. {e!s}')

        # check for freq level specification
        if (not self.freq_level) or (self.freq_level == empty_model_chemistry_input_dict):
            self.freq_level = ''
        else:
            try:
                self.freq_level, _ = format_level_of_theory_inputs(self.freq_level)
            except InputError as e:
                raise InputError(f'Specified model chemistry input dictionary for frequency seems wrong. {e!s}')

        # check for sp level specification
        if (not self.sp_level) or (self.sp_level == empty_model_chemistry_input_dict):
            self.sp_level = ''
        else:
            try:
                self.sp_level, _ = format_level_of_theory_inputs(self.sp_level)
            except InputError as e:
                raise InputError(f'Specified model chemistry input dictionary for single point seems wrong. {e!s}')

        # check for scan level specification
        if (not self.scan_level) or (self.scan_level == empty_model_chemistry_input_dict):
            self.scan_level = ''
        else:
            try:
                self.scan_level, _ = format_level_of_theory_inputs(self.scan_level)
            except InputError as e:
                raise InputError(f'Specified model chemistry input dictionary for rotor scan seems wrong. {e!s}')

        # check for ts guess level specification
        if (not self.ts_guess_level) or (self.ts_guess_level == empty_model_chemistry_input_dict):
            self.ts_guess_level = ''
        else:
            try:
                self.ts_guess_level, _ = format_level_of_theory_inputs(self.ts_guess_level)
            except InputError as e:
                raise InputError(f'Specified model chemistry input dictionary for optimizing TS guess seems wrong. '
                                 f'{e!s}')

        # check for orbital level specification
        if (not self.orbitals_level) or (self.orbitals_level == empty_model_chemistry_input_dict):
            self.orbitals_level = ''
        else:
            try:
                self.orbitals_level, _ = format_level_of_theory_inputs(self.orbitals_level)
            except InputError as e:
                raise InputError(f'Specified model chemistry input dictionary for visualizing molecular orbitals '
                                 f'seems wrong. {e!s}')
