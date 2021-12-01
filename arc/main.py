"""
ARC's main module.
To run ARC through its API, first make an instance of the ARC class, then call the .execute() method. For example::

  arc = ARC(project='ArcDemo', species=[spc0, spc1, spc2])
  arc.execute()

Where ``spc0``, ``spc1``, and ``spc2`` in the above example are :ref:`ARCSpecies <species>` objects.
"""

import datetime
import logging
import os
import shutil
import time
from distutils.spawn import find_executable
from enum import Enum
from IPython.display import display
from typing import Dict, List, Optional, Tuple, Union

from arkane.encorr.corr import assign_frequency_scale_factor
from rmgpy.reaction import Reaction
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.common import (VERSION,
                        ARC_PATH,
                        check_ess_settings,
                        get_logger,
                        globalize_path,
                        initialize_job_types,
                        initialize_log,
                        log_footer,
                        save_yaml_file,
                        time_lapse,
                        )
from arc.exceptions import InputError, SettingsError, SpeciesError
from arc.imports import settings
from arc.level import Level
from arc.job.ssh import SSHClient
from arc.processor import process_arc_project
from arc.reaction import ARCReaction
from arc.scheduler import Scheduler
from arc.species.species import ARCSpecies
from arc.utils.scale import determine_scaling_factors


logger = get_logger()

default_levels_of_theory, servers, valid_chars, default_job_types, default_job_settings, global_ess_settings = \
    settings['default_levels_of_theory'], settings['servers'], settings['valid_chars'], settings['default_job_types'], \
    settings['default_job_settings'], settings['global_ess_settings']


class StatmechEnum(str, Enum):
    """
    The supported statmech software adapters.
    The available adapters are a finite set.
    """
    arkane = 'arkane'
    # mesmer = 'mesmer'
    # mess = 'mess'


class ARC(object):
    """
    The main ARC class.

    Args:
        project (str, optional): The project's name. Used for naming the working directory.
        species (list, optional): Entries are :ref:`ARCSpecies <species>` objects.
        reactions (list, optional): A list of :ref:`ARCReaction <reaction>` objects.
        level_of_theory (str, optional): A shortcut representing either sp//geometry levels or a composite method.
                                         e.g., 'CBS-QB3', 'CCSD(T)-F12a/aug-cc-pVTZ//B3LYP/6-311++G(3df,3pd)'...
                                         Notice that this argument does NOT support levels with slashes in the name.
                                         e.g., 'ZINDO/2', 'DLPNO-MP2-F12/D'
                                         For these cases, use the dictionary-type job-specific level of theory arguments
                                         instead (e.g., ``opt_level``).
        composite_method (str, dict, Level, optional): Composite method.
        conformer_level (str, dict, Level, optional): Level of theory for conformer searches.
        opt_level (str, dict, Level, optional): Level of theory for geometry optimization.
        freq_level (str, dict, Level, optional): Level of theory for frequency calculations.
        sp_level (str, dict, Level, optional): Level of theory for single point calculations.
        scan_level (str, dict, Level, optional): Level of theory for rotor scans.
        ts_guess_level (str, dict, Level, optional): Level of theory for comparisons of TS guesses between different methods.
        irc_level (str, dict, Level, optional): The level of theory to use for IRC calculations.
        orbitals_level (str, dict, Level, optional): Level of theory for molecular orbitals calculations.
        bac_type (str, optional): The bond additivity correction type. 'p' for Petersson- or 'm' for Melius-type BAC.
                                  Default: 'p'. ``None`` to not use BAC.
        job_types (dict, optional): A dictionary of job types to execute. Keys are job types, values are boolean.
        arkane_level_of_theory (Union[dict, Level, str], optional):
            The Arkane level of theory to use for AEC and BAC.

            Note:
                This argument is a ``Level`` type, not a ``LevelOfTheory`` type.
                This argument will only affect AEC/BAC, not the frequency scaling factor.

        T_min (tuple, optional): The minimum temperature for kinetics computations, e.g., ``(500, 'K')``.
        T_max (tuple, optional): The maximum temperature for kinetics computations, e.g., ``(3000, 'K')``.
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
            the molecule. Keys are tuples of (min_num_atoms, max_num_atoms), values are dictionaries. Keys of the
            sub-dictionaries are tuples of job types, values are levels of theory (str, dict or Level).
            Job types not defined in adaptive levels will have non-adaptive (regular) levels.
            Example::

                adaptive_levels = {(1, 5):      {('opt', 'freq'): 'wb97xd/6-311+g(2d,2p)',
                                                 'sp': 'ccsd(t)-f12/aug-cc-pvtz-f12'},
                                   (6, 15):     {('opt', 'freq'): 'b3lyp/cbsb7',
                                                 'sp': 'dlpno-ccsd(t)/def2-tzvp'},
                                   (16, 30):    {('opt', 'freq'): 'b3lyp/6-31g(d,p)',
                                                 'sp': 'wb97xd/6-311+g(2d,2p)'},
                                   (31, 'inf'): {('opt', 'freq'): 'b3lyp/6-31g(d,p)',
                                                 'sp': 'b3lyp/6-311+g(d,p)'}}

        freq_scale_factor (float, optional): The harmonic frequencies scaling factor. Could be automatically determined
                                             if not available in Arkane and not provided by the user.
        calc_freq_factor (bool, optional): Whether to calculate the frequencies scaling factor using Truhlar's method if
                                           it was not given by the user and could not be determined by Arkane. True to
                                           calculate, False to use user input / Arkane's value / Arkane's default.
        n_confs (int, optional): The number of lowest force field conformers to consider.
        e_confs (float, optional): The energy threshold in kJ/mol above the lowest energy conformer below which
                                   force field conformers are considered.
        keep_checks (bool, optional): Whether to keep ESS checkfiles when ARC terminates. ``True`` to keep,
                                      default is ``False``.
        dont_gen_confs (list, optional): A list of species labels for which conformer generation should be avoided
                                         if xyz is given.
        compare_to_rmg (bool, optional): If ``True`` data calculated from the RMG-database will be calculated and
                                         included on the parity plot.
        compute_thermo (bool, optional): Whether to compute thermodynamic properties for converged species.
        compute_rates (bool, optional): Whether to compute rate coefficients for converged reactions.
        compute_transport (bool, optional): Whether to compute transport properties for converged species.
        specific_job_type (str, optional): A specific job type to execute.
                                           Legal strings are job types (keys of job_types dict).
        thermo_adapter (str, optional): The statmech software to use for thermodynamic property calculations.
                                        Default: 'Arkane'.
        kinetics_adapter (str, optional): The statmech software to use for kinetic rate coefficient calculations.
                                          Default: 'Arkane'.
        three_params (bool, optional): Compute rate coefficients using the modified three-parameter Arrhenius equation
                                       format (``True``, default) or classical two-parameter Arrhenius equation format
                                       (``False``).
        trsh_ess_jobs (bool, optional): Whether to attempt troubleshooting failed ESS jobs. Default is ``True``.
        output (dict, optional): Output dictionary with status and final QM file paths for all species.
                                 Only used for restarting.
        running_jobs (dict, optional): A dictionary of jobs submitted in a precious ARC instance, used for restarting.

    Attributes:
        project (str): The project's name. Used for naming the working directory.
        project_directory (str): The path to the project directory.
        species (list): A list of :ref:`ARCSpecies <species>` objects.
        reactions (list): A list of :ref:`ARCReaction <reaction>` objects.
        level_of_theory (str): A shortcut representing either sp//geometry levels or a composite method.
        composite_method (Level): Composite method.
        conformer_level (Level): Level of theory for conformer searches.
        opt_level (Level): Level of theory for geometry optimization.
        freq_level (Level): Level of theory for frequency calculations.
        sp_level (Level): Level of theory for single point calculations.
        scan_level (Level): Level of theory for rotor scans.
        ts_guess_level (Level): Level of theory for comparisons of TS guesses between different methods.
        irc_level (Level): The level of theory to use for IRC calculations.
        orbitals_level (Level): Level of theory for molecular orbitals calculations.
        adaptive_levels (dict): A dictionary of levels of theory for ranges of the number of heavy atoms in
            the molecule. Keys are tuples of (min_num_atoms, max_num_atoms), values are dictionaries. Keys of the
            sub-dictionaries are tuples of job types, values are levels of theory (str, dict or Level).
            Job types not defined in adaptive levels will have non-adaptive (regular) levels.
        output (dict): Output dictionary with status and final QM file paths for all species. Only used for restarting,
                       the actual object used is in the Scheduler class.
        bac_type (str): The bond additivity correction type. 'p' for Petersson- or 'm' for Melius-type BAC.
                        ``None`` to not use BAC.
        arkane_level_of_theory (Level): The Arkane level of theory to use for AEC and BAC.
        freq_scale_factor (float): The harmonic frequencies scaling factor. Could be automatically determined if not
                                   available in Arkane and not provided by the user.
        calc_freq_factor (bool): Whether to calculate the frequencies scaling factor using Truhlar's method if it was
                                 not given by the user and could not be determined by Arkane. True to calculate, False
                                 to use user input / Arkane's value / Arkane's default.
        ess_settings (dict): A dictionary of available ESS (keys) and a corresponding server list (values).
        t0 (float): Initial time when the project was spawned.
        n_confs (int): The number of lowest force field conformers to consider.
        e_confs (float): The energy threshold in kJ/mol above the lowest energy conformer below which
                         force field conformers are considered.
        execution_time (str): Overall execution time.
        lib_long_desc (str): A multiline description of levels of theory for the outputted RMG libraries.
        running_jobs (dict): A dictionary of jobs submitted in a precious ARC instance, used for restarting.
        T_min (tuple): The minimum temperature for kinetics computations, e.g., (500, 'K').
        T_max (tuple): The maximum temperature for kinetics computations, e.g., (3000, 'K').
        T_count (int): The number of temperature points between ``T_min`` and ``T_max``.
        max_job_time (float): The maximal allowed job time on the server in hours (can be fractional).
        rmg_database (RMGDatabase): The RMG database object.
        allow_nonisomorphic_2d (bool): Whether to optimize species even if they do not have a 3D conformer that is
                                       isomorphic to the 2D graph representation.
        memory (int): The total allocated job memory in GB (14 by default to be lower than 90% * 16 GB).
        job_types (dict): A dictionary of job types to execute. Keys are job types, values are boolean.
        specific_job_type (str): A specific job type to execute. Legal strings are job types (keys of job_types dict).
        bath_gas (str): A bath gas. Currently used in OneDMin to calc L-J parameters.
                        Allowed values are He, Ne, Ar, Kr, H2, N2, O2.
        keep_checks (bool): Whether to keep all Gaussian checkfiles when ARC terminates. True to keep, default is False.
        dont_gen_confs (list): A list of species labels for which conformer generation should be avoided
                               if xyz is given.
        compare_to_rmg (bool): If ``True`` data calculated from the RMG-database will be calculated and included on the
                               parity plot.
        compute_thermo (bool): Whether to compute thermodynamic properties for converged species.
        compute_rates (bool): Whether to compute rate coefficients for converged reactions.
        compute_transport (bool): Whether to compute transport properties for converged species.
        thermo_adapter (str): The statmech software to use for thermodynamic property calculations.
        kinetics_adapter (str): The statmech software to use for kinetic rate coefficient calculations.
        fine_only (bool): If ``self.job_types['fine'] and not self.job_types['opt']`` ARC will not run optimization
                          jobs without fine=True
        three_params (bool): Compute rate coefficients using the modified three-parameter Arrhenius equation
                             format (``True``) or classical two-parameter Arrhenius equation format (``False``).
        trsh_ess_jobs (bool): Whether to attempt troubleshooting failed ESS jobs. Default is ``True``.
    """

    def __init__(self,
                 adaptive_levels: Optional[dict] = None,
                 allow_nonisomorphic_2d: bool = False,
                 arkane_level_of_theory: Optional[Union[dict, Level, str]] = None,
                 bac_type: str = 'p',
                 bath_gas: Optional[str] = None,
                 calc_freq_factor: bool = True,
                 compare_to_rmg: bool = True,
                 composite_method: Optional[Union[str, dict, Level]] = None,
                 compute_rates: bool = True,
                 compute_thermo: bool = True,
                 compute_transport: bool = False,
                 conformer_level: Optional[Union[str, dict, Level]] = None,
                 dont_gen_confs: List[str] = None,
                 e_confs: float = 5.0,
                 ess_settings: Dict[str, Union[str, List[str]]] = None,
                 freq_level: Optional[Union[str, dict, Level]] = None,
                 freq_scale_factor: Optional[float] = None,
                 irc_level: Optional[Union[str, dict, Level]] = None,
                 keep_checks: bool = False,
                 kinetics_adapter: str = 'Arkane',
                 job_memory: Optional[int] = None,
                 job_types: Optional[Dict[str, bool]] = None,
                 level_of_theory: str = '',
                 max_job_time: Optional[float] = None,
                 n_confs: int = 10,
                 opt_level: Optional[Union[str, dict, Level]] = None,
                 orbitals_level: Optional[Union[str, dict, Level]] = None,
                 output: Optional[dict] = None,
                 project: Optional[str] = None,
                 project_directory: Optional[str] = None,
                 reactions: Optional[List[Union[ARCReaction, Reaction]]] = None,
                 running_jobs: Optional[dict] = None,
                 scan_level: Optional[Union[str, dict, Level]] = None,
                 sp_level: Optional[Union[str, dict, Level]] = None,
                 species: Optional[List[Union[ARCSpecies, Species]]] = None,
                 specific_job_type: str = '',
                 T_min: Optional[Tuple[float, str]] = None,
                 T_max: Optional[Tuple[float, str]] = None,
                 T_count: int = 50,
                 thermo_adapter: str = 'Arkane',
                 three_params: bool = True,
                 trsh_ess_jobs: bool = True,
                 ts_guess_level: Optional[Union[str, dict, Level]] = None,
                 verbose=logging.INFO,
                 ):

        if project is None:
            raise ValueError('A project name must be provided for a new project')
        self.project = project
        self.check_project_name()

        self.__version__ = VERSION
        self.verbose = verbose
        self.project_directory = project_directory if project_directory is not None \
            else os.path.join(ARC_PATH, 'Projects', self.project)
        if not os.path.exists(self.project_directory):
            os.makedirs(self.project_directory)
        self.output = output
        self.standardize_output_paths()  # depends on self.project_directory
        self.running_jobs = running_jobs or dict()
        self.lib_long_desc = ''
        self.unique_species_labels = list()
        self.rmg_database = rmgdb.make_rmg_database_object()
        self.max_job_time = max_job_time or default_job_settings.get('job_time_limit_hrs', 120)
        self.allow_nonisomorphic_2d = allow_nonisomorphic_2d
        self.memory = job_memory or default_job_settings.get('job_total_memory_gb', 14)
        self.calc_freq_factor = calc_freq_factor
        self.keep_checks = keep_checks
        self.compare_to_rmg = compare_to_rmg
        self.compute_thermo = compute_thermo
        self.compute_rates = compute_rates
        self.three_params = three_params
        self.trsh_ess_jobs = trsh_ess_jobs
        self.compute_transport = compute_transport
        self.thermo_adapter = StatmechEnum(thermo_adapter.lower()).value
        self.kinetics_adapter = StatmechEnum(kinetics_adapter.lower()).value
        self.T_min = T_min
        self.T_max = T_max
        self.T_count = T_count
        self.specific_job_type = specific_job_type
        self.job_types = job_types or default_job_types
        self.job_types = initialize_job_types(job_types, specific_job_type=self.specific_job_type)
        self.bath_gas = bath_gas
        self.n_confs = n_confs
        self.e_confs = e_confs
        self.adaptive_levels = process_adaptive_levels(adaptive_levels)
        initialize_log(log_file=os.path.join(self.project_directory, 'arc.log'), project=self.project,
                       project_directory=self.project_directory, verbose=self.verbose)
        self.dont_gen_confs = dont_gen_confs or list()
        self.t0 = time.time()  # init time
        self.execution_time = None
        self.bac_type = bac_type
        self.arkane_level_of_theory = Level(repr=arkane_level_of_theory) if arkane_level_of_theory is not None else None
        self.freq_scale_factor = freq_scale_factor

        # attributes related to level of theory specifications
        self.level_of_theory = level_of_theory
        self.composite_method = composite_method or None
        self.conformer_level = conformer_level or None
        self.opt_level = opt_level or None
        self.freq_level = freq_level or None
        self.sp_level = sp_level or None
        self.scan_level = scan_level or None
        self.ts_guess_level = ts_guess_level or None
        self.irc_level = irc_level or None
        self.orbitals_level = orbitals_level or None

        # species
        self.species = species or list()
        converted_species, indices_to_pop = list(), list()
        for i, spc in enumerate(self.species):
            if isinstance(spc, Species):
                # RMG Species
                if not spc.label:
                    raise InputError(f'Missing label on RMG Species object {spc}')
                indices_to_pop.append(i)
                arc_spc = ARCSpecies(is_ts=False, rmg_species=spc)  # assuming an RMG Species is not a TS
                converted_species.append(arc_spc)
            elif isinstance(spc, dict):
                # dict representation for ARCSpecies
                indices_to_pop.append(i)
                converted_species.append(ARCSpecies(species_dict=spc))
            elif not isinstance(spc, ARCSpecies):
                raise ValueError(f'A species should either be an RMG Species object, an ARCSpecies object, '
                                 f'or a dictionary representation of the later.\nGot: {type(spc)} for {spc}')
        for i in reversed(range(len(self.species))):  # pop from the end, so other indices won't change
            if i in indices_to_pop:
                self.species.pop(i)
        self.species.extend(converted_species)
        for spc in self.species:
            if spc.rotors_dict is not None:
                for rotor_num, rotor_dict in spc.rotors_dict.items():
                    if rotor_dict['scan_path'] and not os.path.isfile(rotor_dict['scan_path']) and rotor_dict['success']:
                        # try correcting relative paths
                        if os.path.isfile(os.path.join(ARC_PATH, rotor_dict['scan_path'])):
                            spc.rotors_dict[rotor_num]['scan_path'] = os.path.join(ARC_PATH, rotor_dict['scan_path'])
                        elif os.path.isfile(os.path.join(ARC_PATH, 'Projects', rotor_dict['scan_path'])):
                            spc.rotors_dict[rotor_num]['scan_path'] = \
                                os.path.join(ARC_PATH, 'Projects', rotor_dict['scan_path'])
                        else:
                            raise SpeciesError(f'Could not find rotor scan output file for rotor {rotor_num} of '
                                               f'species {spc.label}: {rotor_dict["scan_path"]}')
        if self.job_types['bde']:
            self.add_hydrogen_for_bde()
        self.determine_unique_species_labels()

        # reactions
        self.reactions = reactions or list()
        converted_reactions, indices_to_pop = list(), list()
        for i, rxn in enumerate(self.reactions):
            if isinstance(rxn, Reaction):
                # RMG Reaction
                if not rxn.reactants or not rxn.products:
                    raise InputError('Missing reactants and/or products in RMG Reaction object {0}'.format(rxn))
                indices_to_pop.append(i)
                arc_rxn = ARCReaction(rmg_reaction=rxn)
                converted_reactions.append(arc_rxn)
                for spc in rxn.reactants + rxn.products:
                    if not isinstance(spc, Species):
                        raise InputError(f'All reactants and products of an RMG Reaction have to be RMG Species '
                                         f'objects. Got: {type(spc)} in reaction {rxn}')
                    if not spc.label:
                        raise InputError(f'Missing label on RMG Species object {spc} in reaction {rxn}')
                    if spc.label not in self.unique_species_labels:
                        # Add species participating in an RMG Reaction to ``species`` if not already there
                        # We assume each species has a unique label
                        self.species.append(ARCSpecies(is_ts=False, rmg_species=spc))
                        self.unique_species_labels.append(spc.label)
            elif isinstance(rxn, dict):
                # dict representation for ARCReaction as in a YAML input file
                indices_to_pop.append(i)
                converted_reactions.append(ARCReaction(reaction_dict=rxn, species_list=self.species))
            elif not isinstance(rxn, ARCReaction):
                raise ValueError(f'A reaction should either be an `ARCReaction` object or an RMG `Reaction` object. '
                                 f'Got {type(rxn)} for {rxn}')
        for i in reversed(range(len(self.reactions))):  # pop from the end, so other indices won't change
            if i in indices_to_pop:
                self.reactions.pop(i)
        self.reactions.extend(converted_reactions)
        for rxn_index, arc_rxn in enumerate(self.reactions):
            arc_rxn.index = rxn_index

        if self.adaptive_levels is not None:
            logger.info(f'Using the following adaptive levels of theory:\n{self.adaptive_levels}')
        self.ess_settings = check_ess_settings(ess_settings or global_ess_settings)
        if not self.ess_settings:
            # Use the "radar" feature if ess_settings are still unavailable.
            self.determine_ess_settings()

        # Determine if fine-only behavior is requested before determining chemistry for job types.
        self.fine_only = False
        if self.job_types['fine'] and not self.job_types['opt']:
            self.fine_only = True
            self.job_types['opt'] = True  # Run the optimizations, self.fine_only will make sure that they are fine.

        self.set_levels_of_theory()  # All level of theories should be Level types after this call.
        if self.thermo_adapter == 'arkane':
            self.check_arkane_level_of_theory()

        if self.job_types['freq'] or self.composite_method is not None:
            self.check_freq_scaling_factor()

        if not self.trsh_ess_jobs:
            logger.info('\n')
            logger.warning('Not troubleshooting ESS jobs!')
            logger.info('\n')

        self.scheduler = None
        self.restart_dict = self.as_dict()
        self.backup_restart()

    def as_dict(self) -> dict:
        """
        A helper function for dumping this object as a dictionary in a YAML file for restarting ARC.
        """
        restart_dict = dict()
        if self.adaptive_levels is not None:
            restart_dict['adaptive_levels'] = {atom_range: {job_type: level.as_dict() for job_type, level in levels_dict}
                                               for atom_range, levels_dict in self.adaptive_levels.items()}
        restart_dict['allow_nonisomorphic_2d'] = self.allow_nonisomorphic_2d
        if self.arkane_level_of_theory is not None:
            restart_dict['arkane_level_of_theory'] = self.arkane_level_of_theory.as_dict() \
                if isinstance(self.arkane_level_of_theory, Level) else self.arkane_level_of_theory
        if self.bac_type != 'p':
            restart_dict['bac_type'] = self.bac_type
        if self.bath_gas is not None:
            restart_dict['bath_gas'] = self.bath_gas
        if self.calc_freq_factor:
            restart_dict['calc_freq_factor'] = self.calc_freq_factor
        if not self.compare_to_rmg:
            restart_dict['compare_to_rmg'] = self.compare_to_rmg
        if self.composite_method is not None:
            restart_dict['composite_method'] = self.composite_method.as_dict()
        if not self.compute_rates:
            restart_dict['compute_rates'] = self.compute_rates
        if not self.compute_thermo:
            restart_dict['compute_thermo'] = self.compute_thermo
        if not self.compute_transport:
            restart_dict['compute_transport'] = self.compute_transport
        if self.conformer_level is not None:
            restart_dict['conformer_level'] = self.conformer_level.as_dict()
        if self.dont_gen_confs:
            restart_dict['dont_gen_confs'] = self.dont_gen_confs
        restart_dict['e_confs'] = self.e_confs
        restart_dict['ess_settings'] = self.ess_settings
        if self.freq_level is not None:
            restart_dict['freq_level'] = self.freq_level.as_dict() \
                if not isinstance(self.freq_level, (dict, str)) else self.freq_level
        if self.freq_scale_factor is not None:
            restart_dict['freq_scale_factor'] = self.freq_scale_factor
        if self.irc_level is not None:
            restart_dict['irc_level'] = self.irc_level.as_dict() \
                if not isinstance(self.irc_level, (dict, str)) else self.irc_level
        if self.keep_checks:
            restart_dict['keep_checks'] = self.keep_checks
        restart_dict['kinetics_adapter'] = self.kinetics_adapter
        restart_dict['job_memory'] = self.memory
        restart_dict['job_types'] = self.job_types
        if self.level_of_theory:
            restart_dict['level_of_theory'] = self.level_of_theory
        restart_dict['max_job_time'] = self.max_job_time
        restart_dict['n_confs'] = self.n_confs
        if self.opt_level is not None:
            restart_dict['opt_level'] = self.opt_level.as_dict() \
                if not isinstance(self.opt_level, (dict, str)) else self.opt_level
        if self.orbitals_level is not None:
            restart_dict['orbitals_level'] = self.orbitals_level.as_dict() \
                if not isinstance(self.orbitals_level, (dict, str)) else self.orbitals_level
        restart_dict['output'] = self.output
        restart_dict['project'] = self.project
        restart_dict['reactions'] = [rxn.as_dict() for rxn in self.reactions]
        restart_dict['running_jobs'] = self.running_jobs
        if self.scan_level is not None:
            restart_dict['scan_level'] = self.scan_level.as_dict() \
                if not isinstance(self.scan_level, (dict, str)) else self.scan_level
        if self.sp_level is not None:
            restart_dict['sp_level'] = self.sp_level.as_dict() \
                if not isinstance(self.sp_level, (dict, str)) else self.sp_level
        restart_dict['species'] = [spc.as_dict() for spc in self.species]
        if self.specific_job_type:
            restart_dict['specific_job_type'] = self.specific_job_type
        restart_dict['T_min'] = self.T_min
        restart_dict['T_max'] = self.T_max
        restart_dict['T_count'] = self.T_count
        restart_dict['thermo_adapter'] = self.thermo_adapter
        if not self.three_params:
            restart_dict['three_params'] = self.three_params
        if not self.trsh_ess_jobs:
            restart_dict['trsh_ess_jobs'] = self.trsh_ess_jobs
        if self.ts_guess_level is not None:
            restart_dict['ts_guess_level'] = self.ts_guess_level.as_dict() \
                if not isinstance(self.ts_guess_level, (dict, str)) else self.ts_guess_level
        if self.verbose != logging.INFO:
            restart_dict['verbose'] = int(self.verbose)
        return restart_dict

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

        Returns: dict
            Status dictionary indicating which species converged successfully.
        """
        logger.info('\n')
        for species in self.species:
            if not isinstance(species, ARCSpecies):
                raise ValueError(f'All species must be ARCSpecies objects. Got {type(species)}')
            if species.is_ts:
                logger.info(f'Considering transition state: {species.label}')
            else:
                logger.info(f'Considering species: {species.label}')
                if species.mol is not None:
                    display(species.mol.copy(deep=True))
        logger.info('\n')
        for rxn in self.reactions:
            if not isinstance(rxn, ARCReaction):
                raise ValueError(f'All reactions be ARCReaction objects. Got {type(rxn)}')
        self.scheduler = Scheduler(project=self.project,
                                   species_list=self.species,
                                   rxn_list=self.reactions,
                                   composite_method=self.composite_method,
                                   conformer_level=self.conformer_level,
                                   opt_level=self.opt_level,
                                   freq_level=self.freq_level,
                                   sp_level=self.sp_level,
                                   scan_level=self.scan_level,
                                   ts_guess_level=self.ts_guess_level,
                                   irc_level=self.irc_level,
                                   orbitals_level=self.orbitals_level,
                                   ess_settings=self.ess_settings,
                                   job_types=self.job_types,
                                   bath_gas=self.bath_gas,
                                   rmg_database=self.rmg_database,
                                   restart_dict=self.restart_dict,
                                   project_directory=self.project_directory,
                                   max_job_time=self.max_job_time,
                                   allow_nonisomorphic_2d=self.allow_nonisomorphic_2d,
                                   memory=self.memory,
                                   adaptive_levels=self.adaptive_levels,
                                   n_confs=self.n_confs,
                                   e_confs=self.e_confs,
                                   dont_gen_confs=self.dont_gen_confs,
                                   trsh_ess_jobs=self.trsh_ess_jobs,
                                   fine_only=self.fine_only,
                                   )

        save_yaml_file(path=os.path.join(self.project_directory, 'output', 'status.yml'), content=self.scheduler.output)

        if not self.keep_checks:
            self.delete_check_files()

        self.save_project_info_file()

        process_arc_project(thermo_adapter=self.thermo_adapter.lower(),
                            kinetics_adapter=self.kinetics_adapter.lower(),
                            project=self.project,
                            project_directory=self.project_directory,
                            species_dict=self.scheduler.species_dict,
                            reactions=self.scheduler.rxn_list,
                            output_dict=self.scheduler.output,
                            bac_type=self.bac_type,
                            freq_scale_factor=self.freq_scale_factor,
                            compute_thermo=self.compute_thermo,
                            compute_rates=self.compute_rates,
                            compute_transport=self.compute_transport,
                            T_min=self.T_min,
                            T_max=self.T_max,
                            T_count=self.T_count or 50,
                            lib_long_desc=self.lib_long_desc,
                            rmg_database=self.rmg_database,
                            compare_to_rmg=self.compare_to_rmg,
                            three_params=self.three_params,
                            sp_level=self.arkane_level_of_theory,
                            )

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
        txt += f'Conformers:       {self.conformer_level}\n'
        txt += f'TS guesses:       {self.ts_guess_level}\n'
        if self.composite_method is not None:
            txt += f'Composite method: {self.composite_method} {fine_txt}\n'
            txt += f'Frequencies:      {self.freq_level}\n'
        else:
            txt += f'Optimization:     {self.opt_level} {fine_txt}\n'
            txt += f'Frequencies:      {self.freq_level}\n'
            txt += f'Single point:     {self.sp_level}\n'
        if 'rotors' in self.job_types:
            txt += f'Rotor scans:      {self.scan_level}\n'
        else:
            txt += 'Not scanning rotors\n'
        if self.bac_type is not None:
            txt += f'Using {self.bac_type}-type bond additivity corrections for thermo\n'
        else:
            txt += 'NOT using bond additivity corrections for thermo\n'
        txt += f'\nUsing the following ESS settings: {self.ess_settings}\n'
        txt += '\nConsidered the following species and TSs:\n'
        for species in self.species:
            descriptor = 'TS' if species.is_ts else 'Species'
            failed = '' if self.scheduler.output[species.label]['convergence'] else ' (Failed!)'
            txt += f'{descriptor} {species.label}{failed} (run time: {species.run_time})\n'
        if self.reactions:
            for rxn in self.reactions:
                txt += f'Considered reaction: {rxn.label}\n'
        txt += f'\nOverall time since project initiation: {self.execution_time}'
        txt += '\n'

        with open(path, 'w') as f:
            f.write(str(txt))
        self.lib_long_desc = txt

        # Save a YAML file to be used by T3.
        content = dict()
        content['species'], content['reactions'] = list(), list()
        path = os.path.join(self.project_directory, f'{self.project}_info.yml')
        if os.path.exists(path):
            os.remove(path)
        for species in self.species:
            if not species.is_ts:
                spc_dict = dict()
                spc_dict['label'] = species.label
                spc_dict['success'] = self.scheduler.output[species.label]['convergence']
                spc_dict['smiles'] = species.mol.copy(deep=True).to_smiles() if species.mol is not None else None
                spc_dict['adj'] = species.mol.copy(deep=True).to_adjacency_list() if species.mol is not None else None
                content['species'].append(spc_dict)
        for reaction in self.reactions:
            rxn_dict = dict()
            rxn_dict['label'] = reaction.label
            rxn_dict['success'] = self.scheduler.output[reaction.ts_species.label]['convergence']
            content['reactions'].append(rxn_dict)
        save_yaml_file(path=path, content=content)

    def summary(self) -> dict:
        """
        Report status and data of all species / reactions.

        Returns: dict
            Status dictionary indicating which species converged successfully.
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

    def determine_ess_settings(self, diagnostics=False):
        """
        Determine where each ESS is available, locally (in running on a server) and/or on remote servers.
        if `diagnostics` is True, this method will not raise errors, and will print its findings.
        """
        if self.ess_settings and not diagnostics:
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
            with SSHClient(server) as ssh:

                g03 = ssh.find_package('g03')
                g09 = ssh.find_package('g09')
                g16 = ssh.find_package('g16')
                if g03 or g09 or g16:
                    if diagnostics:
                        logger.info(f'  Found Gaussian on {server}: g03={g03}, g09={g09}, g16={g16}')
                    self.ess_settings['gaussian'].append(server)
                elif diagnostics:
                    logger.info(f'  Did NOT find Gaussian on {server}')

                qchem = ssh.find_package('qchem')
                if qchem:
                    if diagnostics:
                        logger.info(f'  Found QChem on {server}')
                    self.ess_settings['qchem'].append(server)
                elif diagnostics:
                    logger.info(f'  Did NOT find QChem on {server}')

                orca = ssh.find_package('orca')
                if orca:
                    if diagnostics:
                        logger.info(f'  Found Orca on {server}')
                    self.ess_settings['orca'].append(server)
                elif diagnostics:
                    logger.info(f'  Did NOT find Orca on {server}')

                terachem = ssh.find_package('terachem')
                if terachem:
                    if diagnostics:
                        logger.info(f'  Found TeraChem on {server}')
                    self.ess_settings['terachem'].append(server)
                elif diagnostics:
                    logger.info(f'  Did NOT find TeraChem on {server}')

                molpro = ssh.find_package('molpro')
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
                                 f'Got "{char}" in {self.project}.')

    def check_freq_scaling_factor(self):
        """
        Check that the harmonic frequencies scaling factor is known,
        otherwise, and if ``calc_freq_factor`` is set to ``True``, spawn a calculation for it using Truhlar's method.
        """
        if self.freq_scale_factor is None:
            # The user did not specify a scaling factor, see if Arkane has it.
            freq_level = self.composite_method if self.composite_method is not None \
                else self.freq_level if self.freq_level is not None else None
            if freq_level is not None:
                arkane_freq_lot = freq_level.to_arkane_level_of_theory(variant='freq')
                if arkane_freq_lot is not None:
                    # Arkane has this harmonic frequencies scaling factor.
                    self.freq_scale_factor = assign_frequency_scale_factor(level_of_theory=arkane_freq_lot)
                else:
                    logger.info(f'Could not determine the harmonic frequencies scaling factor for '
                                f'{arkane_freq_lot} from Arkane.')
                    if self.calc_freq_factor:
                        logger.info("Calculating it using Truhlar's method.")
                        logger.warning("This proceedure normally spawns QM jobs for various small species "
                                       "not directly asked for by the user.\n\n")
                        self.freq_scale_factor = determine_scaling_factors(levels=[freq_level],
                                                                           ess_settings=self.ess_settings,
                                                                           init_log=False)[0]
                    else:
                        logger.info('Not calculating it, assuming a frequencies scaling factor of 1.')
                        self.freq_scale_factor = 1

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
        for species in self.species:
            if species.label not in self.unique_species_labels:
                self.unique_species_labels.append(species.label)
            else:
                raise ValueError(f'Species label {species.label} is not unique')

    def add_hydrogen_for_bde(self):
        """
        Make sure ARC has a hydrogen species labeled as 'H' for the final processing of bde jobs (if not, create one).
        """
        if any([spc.bdes is not None for spc in self.species]):
            for species in self.species:
                if species.label == 'H':
                    if species.number_of_atoms == 1 and species.get_xyz(generate=True)['symbols'][0] == 'H':
                        break
                    else:
                        raise SpeciesError(f'A species with label "H" was defined, but does not seem to be '
                                           f'the hydrogen atom species. Cannot calculate bond dissociation energies.\n'
                                           f'Got the following species: {[spc.label for spc in self.species]}')
            else:
                # no H species defined, make one
                h = ARCSpecies(label='H', smiles='[H]', compute_thermo=False, e0_only=True)
                self.species.append(h)

    def set_levels_of_theory(self):
        """
        Set all levels of theory by job type to be :ref:`Level <level>` types.
        """
        self.process_level_of_theory()

        logger.info('\n\nUsing the following levels of theory:\n')

        if self.conformer_level is None:
            self.conformer_level = default_levels_of_theory['conformer']
            default_flag = ' (default)'
        else:
            default_flag = ''
        self.conformer_level = Level(repr=self.conformer_level)
        logger.info(f'Conformers:{default_flag} {self.conformer_level}')

        if self.reactions or any([spc.is_ts for spc in self.species]):
            if not self.ts_guess_level:
                self.ts_guess_level = default_levels_of_theory['ts_guesses']
                default_flag = ' (default)'
            else:
                default_flag = ''
            self.ts_guess_level = Level(repr=self.ts_guess_level)
            logger.info(f'TS guesses:{default_flag} {self.ts_guess_level}')

        if self.composite_method is not None:
            self.composite_method = Level(repr=self.composite_method)
            if self.composite_method.method_type != "composite":
                raise InputError(f'The composite method {self.composite_method} was not recognized by ARC.')
            logger.info(f'Composite method: {self.composite_method}')
            self.opt_level, self.sp_level = None, None

            if not self.freq_level:
                self.freq_level = default_levels_of_theory['freq_for_composite']
                default_flag = ' (composite default)'
            else:
                default_flag = ''
            self.freq_level = Level(repr=self.freq_level)
            logger.info(f'Frequencies:{default_flag} {self.freq_level}')

            if self.job_types['rotors']:
                if not self.scan_level:
                    self.scan_level = default_levels_of_theory['scan_for_composite']
                    default_flag = ' (composite default)'
                else:
                    default_flag = ''
                self.scan_level = Level(repr=self.scan_level)
                logger.info(f'Rotor scans:{default_flag} {self.scan_level}')
            else:
                logger.warning("Not performing rotor scans, since it was not requested by the user. This might "
                               "compromise finding the best conformer, as dihedral angles won't be corrected. "
                               "Also, the calculated thermodynamic properties and rate coefficients "
                               "will be less accurate.")

            if self.job_types['irc']:
                if not self.irc_level:
                    self.irc_level = default_levels_of_theory['irc_for_composite']
                    default_flag = ' (composite default)'
                else:
                    default_flag = ''
                self.irc_level = Level(repr=self.irc_level)
                logger.info(f'IRC:{default_flag} {self.irc_level}')

            if self.job_types['orbitals']:
                if not self.orbitals_level:
                    self.orbitals_level = default_levels_of_theory['orbitals_for_composite']
                    default_flag = ' (composite default)'
                else:
                    default_flag = ''
                self.orbitals_level = Level(repr=self.orbitals_level)
                logger.info(f'Orbitals:{default_flag} {self.orbitals_level}')

        else:
            # NOT a composite method
            if self.job_types['opt']:
                if not self.opt_level:
                    self.opt_level = default_levels_of_theory['opt']
                    default_flag = ' (default)'
                else:
                    default_flag = ''
                self.opt_level = Level(repr=self.opt_level)
                logger.info(f'Geometry optimization:{default_flag} {self.opt_level}')
            else:
                logger.warning("Not performing geometry optimization, since it was not requested by the user.")

            if self.job_types['freq']:
                if not self.freq_level:
                    if self.opt_level:
                        self.freq_level = self.opt_level
                        info = ' (user-defined opt)'
                    else:
                        self.freq_level = default_levels_of_theory['freq']
                        info = ' (default)'
                else:
                    info = ''
                self.freq_level = Level(repr=self.freq_level)
                logger.info(f'Frequencies:{info} {self.freq_level}')
            else:
                logger.warning("Not performing frequency calculation, since it was not requested by the user.")

            if self.job_types['sp']:
                if not self.sp_level:
                    self.sp_level = default_levels_of_theory['sp']
                    default_flag = ' (default)'
                else:
                    default_flag = ''
                self.sp_level = Level(repr=self.sp_level)
                logger.info(f'Energy:{default_flag} {self.sp_level}')
            else:
                logger.warning("Not performing single point energy calculation, since it was not requested by the user.")

            if self.job_types['rotors']:
                if not self.scan_level:
                    if self.opt_level:
                        self.scan_level = self.opt_level
                        info = ' (user-defined opt)'
                    else:
                        self.scan_level = default_levels_of_theory['scan']
                        info = ' (default)'
                else:
                    info = ''
                self.scan_level = Level(repr=self.scan_level)
                logger.info(f'Rotor scans:{info} {self.scan_level}')
            else:
                logger.warning("Not performing rotor scans, since it was not requested by the user. This might "
                               "compromise finding the best conformer, as dihedral angles won't be corrected. "
                               "Also, the calculated thermodynamic properties and rate coefficients "
                               "will be less accurate.")

            if self.job_types['irc']:
                if not self.irc_level:
                    self.irc_level = default_levels_of_theory['irc']
                    default_flag = ' (default)'
                else:
                    default_flag = ''
                self.irc_level = Level(repr=self.irc_level)
                logger.info(f'IRC:{default_flag} {self.irc_level}')
            else:
                logger.warning("Not running IRC computations, since it was not requested by the user.")

            if self.job_types['orbitals']:
                if not self.orbitals_level:
                    self.orbitals_level = default_levels_of_theory['orbitals']
                    default_flag = ' (default)'
                else:
                    default_flag = ''
                self.orbitals_level = Level(repr=self.orbitals_level)
                logger.info(f'Orbitals:{default_flag} {self.orbitals_level}')
            else:
                logger.debug("Not running molecular orbitals visualization, since it was not requested by the user.")

        if not self.job_types['fine'] and (self.opt_level is not None and self.opt_level.method_type == 'dft'
                                           or self.composite_method is not None):
            logger.info('\n')
            logger.warning('Not using a fine DFT grid for geometry optimization jobs')
            logger.info('\n')

    def process_level_of_theory(self):
        """
        Process the ``level_of_theory`` argument, and populate respective job level of theory arguments as needed.
        """
        if self.level_of_theory:
            if self.opt_level is not None:
                raise InputError(f'Got both level_of_theory and opt_level arguments. Choose the correct one:\n'
                                 f'level_of_theory: {self.level_of_theory}\n'
                                 f'opt_level: {self.opt_level}')
            if self.sp_level is not None:
                raise InputError(f'Got both level_of_theory and sp_level arguments. Choose the correct one:\n'
                                 f'level_of_theory: {self.level_of_theory}\n'
                                 f'sp_level: {self.sp_level}')
            if self.composite_method is not None:
                raise InputError(f'Got both level_of_theory and composite_method arguments. Choose the correct one:\n'
                                 f'level_of_theory: {self.level_of_theory}\n'
                                 f'composite_method: {self.composite_method}')
            if not isinstance(self.level_of_theory, str):
                raise InputError(f'level_of_theory must be a string.\n'
                                 f'Got {self.level_of_theory} which is a {type(self.level_of_theory)}.')
            if self.level_of_theory.count('//') > 1:
                raise InputError(f'level_of_theory seems wrong. It should either be a composite method (like CBS-QB3) '
                                 f'or be of the form sp//opt, e.g., CCSD(T)-F12/pvtz//wB97x-D3/6-311++g**.\n'
                                 f'Got: {self.level_of_theory}')
            if '//' in self.level_of_theory:
                sp, opt = self.level_of_theory.split('//')
            else:
                sp = opt = self.level_of_theory
            sp = Level(repr=sp)
            opt = Level(repr=opt)
            if sp.method_type == 'composite':
                self.composite_method = sp
            else:
                self.sp_level = sp
                self.opt_level = opt

        self.level_of_theory = ''  # Reset the level_of_theory argument to avoid conflicts upon restarting ARC.

    def check_arkane_level_of_theory(self):
        """
        Check that the level of theory has AEC in Arkane.
        """
        if self.arkane_level_of_theory is None:
            self.arkane_level_of_theory = self.composite_method if self.composite_method is not None \
                else self.sp_level if self.sp_level is not None else None
        if self.arkane_level_of_theory is not None:
            self.arkane_level_of_theory.to_arkane_level_of_theory(variant='AEC', raise_error=self.compute_thermo)
        else:
            logger.warning('Could not determine a level of theory to be used for Arkane!')

    def backup_restart(self):
        """
        Make a backup copy of the restart file if it exists (but don't save an updated one just yet)
        """
        if os.path.isfile(os.path.join(self.project_directory, 'restart.yml')):
            if not os.path.isdir(os.path.join(self.project_directory, 'log_and_restart_archive')):
                os.mkdir(os.path.join(self.project_directory, 'log_and_restart_archive'))
            local_time = datetime.datetime.now().strftime("%H%M%S_%b%d_%Y")
            restart_backup_name = 'restart.old.' + local_time + '.yml'
            shutil.copy(os.path.join(self.project_directory, 'restart.yml'),
                        os.path.join(self.project_directory, 'log_and_restart_archive', restart_backup_name))

    def standardize_output_paths(self):
        """
        Standardize the paths in the output dictionary.
        """
        if self.output is not None and self.output:
            for label, spc_output in self.output.items():
                if 'paths' in spc_output:
                    for key, val in spc_output['paths'].items():
                        if key in ['geo', 'freq', 'sp', 'composite']:
                            if val and not os.path.isfile(val):
                                # try correcting relative paths
                                if os.path.isfile(os.path.join(ARC_PATH, val)):
                                    self.output[label]['paths'][key] = os.path.join(ARC_PATH, val)
                                elif os.path.isfile(os.path.join(ARC_PATH, 'Projects', val)):
                                    self.output[label]['paths'][key] = os.path.join(ARC_PATH, 'Projects', val)
                                elif os.path.isfile(os.path.join(globalize_path(
                                        string=val, project_directory=self.project_directory))):
                                    self.output[label]['paths'][key] = globalize_path(
                                        string=val, project_directory=self.project_directory)
                                else:
                                    raise SpeciesError(f'Could not find {key} output file for species {label}: {val}')
            logger.debug(f'output dictionary successfully parsed:\n{self.output}')
        elif self.output is None:
            self.output = dict()


def process_adaptive_levels(adaptive_levels: Optional[dict]) -> Optional[dict]:
    """
    Process the ``adaptive_levels`` argument.

    Args:
        adaptive_levels (dict): The adaptive levels dictionary.

    Returns: dict
        The processed adaptive levels dictionary.
    """
    if adaptive_levels is None:
        return None
    processed = dict()
    if not isinstance(adaptive_levels, dict):
        raise InputError(f'The adaptive levels argument must be a dictionary, '
                         f'got {adaptive_levels} which is a {type(adaptive_levels)}')
    for atom_range, adaptive_level in adaptive_levels.items():
        if not isinstance(atom_range, tuple) \
                or not all([isinstance(a, int) or a == 'inf' for a in atom_range]) \
                or len(atom_range) != 2:
            raise InputError(f'Keys of the adaptive levels argument must be 2-length tuples of integers or an "inf" '
                             f'indicator, got {atom_range} which is a {type(atom_range)} in:\n{adaptive_levels}')
        if not isinstance(adaptive_level, dict):
            raise InputError(f'Each adaptive level in the adaptive levels argument must be a dictionary, '
                             f'got {adaptive_level} which is a {type(adaptive_level)} in:\n{adaptive_levels}')
        processed[atom_range] = dict()
        for sub_key, level in adaptive_level.items():
            new_sub_key = (sub_key,) if isinstance(sub_key, str) else sub_key
            if not isinstance(new_sub_key, tuple):
                raise InputError(f'Job types specifications in adaptive levels must be tuples, got {sub_key} '
                                 f'which is a {type(sub_key)} in:\n{adaptive_levels}')
            new_level = Level(repr=level)
            processed[atom_range][new_sub_key] = new_level
    atom_ranges = sorted(list(adaptive_levels.keys()), key=lambda x: x[0])
    for i, atom_range in enumerate(atom_ranges):
        if i and atom_ranges[i-1][1] + 1 != atom_ranges[i][0]:
            raise InputError(f'Atom ranges of adaptive levels must be consecutive. '
                             f'Got:\n{list(adaptive_levels.keys())}')
    if atom_ranges[-1][1] != 'inf':
        raise InputError(f'The last atom range must be "inf", got {atom_ranges[-1][1]} in {atom_ranges}')
    return processed
