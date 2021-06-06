"""
A module for scheduling jobs
Includes spawning, terminating, checking, and troubleshooting various jobs
"""

import datetime
import itertools
import logging
import os
import shutil
import time
from IPython.display import display
from typing import List, Optional, Tuple, Union

from arc import parser, plotter
from arc.common import (extermum_list,
                        get_angle_in_180_range,
                        get_logger,
                        save_yaml_file,
                        sort_two_lists_by_the_first,
                        )
from arc.exceptions import (InputError,
                            SanitizationError,
                            SchedulerError,
                            SpeciesError,
                            TrshError,
                            )
from arc.imports import settings
from arc.job.job import Job
from arc.job.local import check_running_jobs_ids
from arc.job.ssh import SSHClient
from arc.job.trsh import (scan_quality_check,
                          trsh_conformer_isomorphism,
                          trsh_ess_job,
                          trsh_negative_freq,
                          trsh_scan_job,
                          )
from arc.level import Level
from arc.species.species import (ARCSpecies,
                                 are_coords_compliant_with_graph,
                                 determine_rotor_symmetry,
                                 TSGuess)
from arc.species.converter import (check_isomorphism,
                                   compare_confs,
                                   molecules_from_xyz,
                                   str_to_xyz,
                                   xyz_to_coords_list,
                                   xyz_to_str,
                                   )
import arc.rmgdb as rmgdb
from arc.species.vectors import get_angle, calculate_dihedral_angle

logger = get_logger()


default_job_settings, default_job_types, rotor_scan_resolution = settings['default_job_settings'], \
                                                                 settings['default_job_types'], \
                                                                 settings['rotor_scan_resolution']


class Scheduler(object):
    """
    ARC's Scheduler class. Creates jobs, submits, checks status, troubleshoots.
    Each species in `species_list` has to have a unique label.

    Dictionary structures::

        job_dict = {label_1: {'conformers': {0: Job1,
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
                              <job_type>:   {job_name1: Job1,
                                             job_name2: Job2, ...},
                              ...
                              }
                    label_2: {...},
                    }

        output = {label_1: {'job_types': {job_type1: <status1>,  # boolean
                                          job_type2: <status2>,
                                         },
                            'paths': {'geo': <path to geometry optimization output file>,
                                      'freq': <path to freq output file>,
                                      'sp': <path to sp output file>,
                                      'composite': <path to composite output file>,
                                      'irc': [list of two IRC paths],
                                     },
                            'conformers': <comments>,
                            'isomorphism': <comments>,
                            'convergence': <status>,  # Optional[bool]
                            'restart': <comments>,
                            'info': <comments>,
                            'warnings': <comments>,
                            'errors': <comments>,
                           },
                 label_2: {...},
                 }

    Note:
        The rotor scan dicts are located under Species.rotors_dict

    Args:
        project (str): The project's name. Used for naming the working directory.
        ess_settings (dict): A dictionary of available ESS and a corresponding server list.
        species_list (list): Contains input :ref:`ARCSpecies <species>` objects (both wells and TSs).
        rxn_list (list): Contains input :ref:`ARCReaction <reaction>` objects.
        project_directory (str): Folder path for the project: the input file path or ARC/Projects/project-name.
        composite_method (str, optional): A composite method to use.
        conformer_level (Union[str, dict], optional): The level of theory to use for conformer comparisons.
        opt_level (Union[str, dict], optional): The level of theory to use for geometry optimizations.
        freq_level (Union[str, dict], optional): The level of theory to use for frequency calculations.
        sp_level (Union[str, dict], optional): The level of theory to use for single point energy calculations.
        scan_level (Union[str, dict], optional): The level of theory to use for torsion scans.
        ts_guess_level (Union[str, dict], optional): The level of theory to use for TS guess comparisons.
        irc_level (Union[str, dict], optional): The level of theory to use for IRC calculations.
        orbitals_level (Union[str, dict], optional): The level of theory to use for calculating MOs (for plotting).
        adaptive_levels (dict, optional): A dictionary of levels of theory for ranges of the number of heavy atoms
                                          in the species. Keys are tuples of (min_num_atoms, max_num_atoms),
                                          values are dictionaries with job type tuples as keys and levels of theory
                                          as values. 'inf' is accepted an max_num_atoms rmg_database
                                          (RMGDatabase, optional): The RMG database object.
        job_types (dict, optional): A dictionary of job types to execute. Keys are job types, values are boolean.
        bath_gas (str, optional): A bath gas. Currently used in OneDMin to calc L-J parameters.
                                  Allowed values are He, Ne, Ar, Kr, H2, N2, O2.
        restart_dict (dict, optional): A restart dictionary parsed from a YAML restart file.
        max_job_time (float, optional): The maximal allowed job time on the server in hours (can be fractional).
        allow_nonisomorphic_2d (bool, optional): Whether to optimize species even if they do not have a 3D conformer
                                                 that is isomorphic to the 2D graph representation.
        memory (float, optional): The total allocated job memory in GB (14 by default).
        testing (bool, optional): Used for internal ARC testing (generating the object w/o executing it).
        dont_gen_confs (list, optional): A list of species labels for which conformer jobs were loaded from a restart
                                         file, or user-requested. Additional conformer generation should be avoided.
        n_confs (int, optional): The number of lowest force field conformers to consider.
        e_confs (float, optional): The energy threshold in kJ/mol above the lowest energy conformer below which
                                   force field conformers are considered.
        fine_only (bool): If ``True`` ARC will not run optimization jobs without fine=True

    Attributes:
        project (str): The project's name. Used for naming the working directory.
        servers (list): A list of servers used for the present project.
        species_list (list): Contains input :ref:`ARCSpecies <species>` objects (both species and TSs).
        species_dict (dict): Keys are labels, values are :ref:`ARCSpecies <species>` objects.
        rxn_list (list): Contains input :ref:`ARCReaction <reaction>` objects.
        unique_species_labels (list): A list of species labels (checked for duplicates).
        job_dict (dict): A dictionary of all scheduled jobs. Keys are species / TS labels,
                         values are dictionaries where keys are job names (corresponding to
                         'running_jobs' if job is running) and values are the Job objects.
        running_jobs (dict): A dictionary of currently running jobs (a subset of `job_dict`).
                             Keys are species/TS label, values are lists of job names (e.g. 'conformer3', 'opt_a123').
        servers_jobs_ids (list): A list of relevant job IDs currently running on the server.
        output (dict): Output dictionary with status per job type and final QM file paths for all species.
        ess_settings (dict): A dictionary of available ESS and a corresponding server list.
        restart_dict (dict): A restart dictionary parsed from a YAML restart file.
        project_directory (str): Folder path for the project: the input file path or ARC/Projects/project-name.
        save_restart (bool): Whether to start saving a restart file. ``True`` only after all species are loaded
                             (otherwise saves a partial file and may cause loss of information).
        restart_path (str): Path to the `restart.yml` file to be saved.
        max_job_time (float): The maximal allowed job time on the server in hours (can be fractional).
        testing (bool): Used for internal ARC testing (generating the object w/o executing it).
        rmg_database (RMGDatabase): The RMG database object.
        allow_nonisomorphic_2d (bool): Whether to optimize species even if they do not have a 3D conformer that is
                                       isomorphic to the 2D graph representation.
        dont_gen_confs (list): A list of species labels for which conformer jobs were loaded from a restart file,
                               or user-requested. Additional conformer generation should be avoided for them.
        memory (float): The total allocated job memory in GB (14 by default).
        n_confs (int): The number of lowest force field conformers to consider.
        e_confs (float): The energy threshold in kJ/mol above the lowest energy conformer below which
                         force field conformers are considered.
        job_types (dict): A dictionary of job types to execute. Keys are job types, values are boolean.
        bath_gas (str): A bath gas. Currently used in OneDMin to calc L-J parameters.
                        Allowed values are He, Ne, Ar, Kr, H2, N2, O2.
        composite_method (str): A composite method to use.
        conformer_level (dict): The level of theory to use for conformer comparisons.
        opt_level (dict): The level of theory to use for geometry optimizations.
        freq_level (dict): The level of theory to use for frequency calculations.
        sp_level (dict): The level of theory to use for single point energy calculations.
        scan_level (dict): The level of theory to use for torsion scans.
        ts_guess_level (dict): The level of theory to use for TS guess comparisons.
        irc_level (dict): The level of theory to use for IRC calculations.
        orbitals_level (dict): The level of theory to use for calculating MOs (for plotting).
        adaptive_levels (dict): A dictionary of levels of theory for ranges of the number of heavy atoms
                                in the species. Keys are tuples of (min_num_atoms, max_num_atoms),
                                values are dictionaries with job type tuples as keys and levels of theory
                                as values. 'inf' is accepted an max_num_atoms rmg_database
                                (RMGDatabase, optional): The RMG database object.
        fine_only (bool): If ``True`` ARC will not run optimization jobs without fine=True
    """

    def __init__(self,
                 project: str,
                 ess_settings: dict,
                 species_list: list,
                 project_directory: str,
                 composite_method: Optional[Level] = None,
                 conformer_level: Optional[Level] = None,
                 opt_level: Optional[Level] = None,
                 freq_level: Optional[Level] = None,
                 sp_level: Optional[Level] = None,
                 scan_level: Optional[Level] = None,
                 ts_guess_level: Optional[Level] = None,
                 irc_level: Optional[Level] = None,
                 orbitals_level: Optional[Level] = None,
                 adaptive_levels: Optional[dict] = None,
                 rmg_database: Optional = None,
                 job_types: Optional[dict] = None,
                 rxn_list: Optional[list] = None,
                 bath_gas: Optional[str] = None,
                 restart_dict: Optional[dict] = None,
                 max_job_time: Optional[float] = None,
                 allow_nonisomorphic_2d: Optional[bool] = False,
                 memory: Optional[float] = None,
                 testing: Optional[bool] = False,
                 dont_gen_confs: Optional[list] = None,
                 n_confs: Optional[int] = 10,
                 e_confs: Optional[float] = 5,
                 fine_only: Optional[bool] = False,
                 ) -> None:

        self.project = project
        self.ess_settings = ess_settings
        self.species_list = species_list
        self.project_directory = project_directory

        self.rmg_database = rmg_database
        self.restart_dict = restart_dict
        self.rxn_list = rxn_list if rxn_list is not None else list()
        self.max_job_time = max_job_time or default_job_settings.get('job_time_limit_hrs', 120)
        self.job_dict = dict()
        self.servers_jobs_ids = list()
        self.running_jobs = dict()
        self.allow_nonisomorphic_2d = allow_nonisomorphic_2d
        self.testing = testing
        self.memory = memory or default_job_settings.get('job_total_memory_gb', 14)
        self.bath_gas = bath_gas
        self.adaptive_levels = adaptive_levels
        self.n_confs = n_confs
        self.e_confs = e_confs
        self.dont_gen_confs = dont_gen_confs or list()
        self.job_types = job_types if job_types is not None else default_job_types
        self.fine_only = fine_only
        self.output = dict()

        self.species_dict = dict()
        for species in self.species_list:
            self.species_dict[species.label] = species
        if self.restart_dict is not None:
            self.output = self.restart_dict['output']
            if 'running_jobs' in self.restart_dict:
                self.restore_running_jobs()
        self.initialize_output_dict()

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
        self.irc_level = irc_level
        self.orbitals_level = orbitals_level
        self.unique_species_labels = list()
        self.save_restart = False

        if len(self.rxn_list):
            rxn_info_path = self.make_reaction_labels_info_file()
            logger.info("\nLoading RMG's families...")
            rmgdb.load_families_only(self.rmg_database)
            for rxn in self.rxn_list:
                logger.info('\n\n')
                # update the ARCReaction object and generate an ARCSpecies object for its TS
                rxn.r_species, rxn.p_species = list(), list()
                for spc in self.species_list:
                    if spc.label in rxn.reactants:
                        rxn.r_species.append(spc)
                    if spc.label in rxn.products:
                        rxn.p_species.append(spc)
                rxn.rmg_reaction_from_arc_species()
                rxn.check_attributes()
                rxn.determine_family(rmg_database=self.rmg_database)
                family_text = ''
                if rxn.family is not None:
                    family_text = f'identified as belonging to RMG family {rxn.family.label}'
                    if rxn.family_own_reverse:
                        family_text += ", which is its own reverse"
                logger.info(f'Considering reaction: {rxn.label}')
                if family_text:
                    logger.info(f'({family_text})')
                if rxn.rmg_reaction is not None:
                    display(rxn.rmg_reaction.copy())
                rxn.determine_rxn_charge()
                rxn.determine_rxn_multiplicity()
                rxn.ts_label = rxn.ts_label if rxn.ts_label is not None else f'TS{rxn.index}'
                with open(rxn_info_path, 'a') as f:
                    f.write(f'{rxn.ts_label}: {rxn.label}')
                    if family_text:
                        family_text = f'\n({family_text})'
                        f.write(str(family_text))
                    f.write(str('\n\n'))
                if len(rxn.ts_xyz_guess) == 1 and 'user guess' not in rxn.ts_methods:
                    rxn.ts_methods.append('user guess')
                elif len(rxn.ts_xyz_guess) > 1 and all(['user guess' not in method for method in rxn.ts_methods]):
                    rxn.ts_methods.append(f'{len(rxn.ts_xyz_guess)} user guesses')
                if not any([spc.label == rxn.ts_label for spc in self.species_list]):
                    ts_species = ARCSpecies(
                        is_ts=True,
                        label=rxn.ts_label,
                        rxn_label=rxn.label,
                        multiplicity=rxn.multiplicity,
                        charge=rxn.charge,
                        compute_thermo=False,
                        ts_methods=rxn.ts_methods,
                        ts_number=rxn.index,
                        preserve_param_in_scan=rxn.preserve_param_in_scan,
                    )
                    ts_species.number_of_atoms = sum(reactant.number_of_atoms for reactant in rxn.r_species)
                    self.species_list.append(ts_species)
                    self.species_dict[ts_species.label] = ts_species
                    self.initialize_output_dict(ts_species.label)
                else:
                    # The TS species was already loaded from a restart dict or an Arkane YAML file
                    ts_species = None
                    for spc in self.species_list:
                        if spc.label == rxn.ts_label:
                            ts_species = spc
                            if ts_species.rxn_label is None:
                                ts_species.rxn_label = rxn.label
                            break
                if ts_species is None:
                    raise SchedulerError(f'Could not identify a TS species for {rxn}')
                rxn.ts_species = ts_species
                # Generate TSGuess objects for all methods, start with the user guesses
                for i, user_guess in enumerate(rxn.ts_xyz_guess):  # this is a list of guesses, could be empty
                    ts_species.ts_guesses.append(
                        TSGuess(
                            method=f'user guess {i}',
                            xyz=user_guess,
                            rmg_reaction=rxn.rmg_reaction,
                        )
                    )
                # now generate TSGuess objects for the other methods
                for method in rxn.ts_methods:
                    if 'user guess' not in method:
                        ts_guess = TSGuess(
                            method=method,
                            rmg_reaction=rxn.rmg_reaction,
                            arc_reaction=rxn,
                            project_dir=self.project_directory
                        )
                        ts_guess.execute_ts_guess_method()
                        ts_species.ts_guesses.append(ts_guess)
                rxn.check_atom_balance()
                rxn.check_done_opt_r_n_p()
            logger.info('\n\n')

        for species in self.species_list:
            if not isinstance(species, ARCSpecies):
                raise SpeciesError(f'Each species in `species_list` must be an ARCSpecies object. '
                                   f'Got type {type(species)} for {species.label}')
            if species.label in self.unique_species_labels:
                raise SpeciesError(f'Each species in `species_list` has to have a unique label. '
                                   f'Label of species {species.label} is not unique.')
            if species.mol is None and not species.is_ts:
                # we'll attempt to infer .mol for a TS after we attain xyz for it
                # for a non-TS, this attribute should be set by this point
                self.output[species.label]['errors'] = 'Could not infer a 2D graph (a .mol species attribute); '
            self.unique_species_labels.append(species.label)
            if self._does_output_dict_contain_info() and species.label in list(self.output.keys()):
                self.output[species.label]['restart'] += f'Restarted ARC at {datetime.datetime.now()}; '
            if species.label not in self.job_dict:
                self.job_dict[species.label] = dict()
            if species.yml_path is None:
                if self.job_types['rotors'] and not self.species_dict[species.label].number_of_rotors \
                        and self.species_dict[species.label].rotors_dict is not None:
                    # if species.rotors_dict is None, it means the species is marked to not spawn rotor scans
                    self.species_dict[species.label].determine_rotors()
                if not self.job_types['opt'] and self.species_dict[species.label].final_xyz is not None:
                    # opt wasn't asked for, and it's not needed, declare it as converged
                    self.output[species.label]['job_types']['opt'] = True
                if species.label not in self.running_jobs:
                    self.running_jobs[species.label] = list()  # initialize before running the first job
                if species.number_of_atoms == 1:
                    logger.debug(f'Species {species.label} is monoatomic')
                    if not self.species_dict[species.label].initial_xyz:
                        # generate a simple "Symbol   0.0   0.0   0.0" coords in a dictionary format
                        if self.species_dict[species.label].mol is not None:
                            symbol = self.species_dict[species.label].mol.atoms[0].symbol
                        else:
                            symbol = species.label
                            logger.warning(f'Could not determine element of monoatomic species {species.label}. '
                                           f'Assuming element is {symbol}')
                        monoatomic_xyz_dict = str_to_xyz(f'{symbol}   0.0   0.0   0.0')
                        self.species_dict[species.label].initial_xyz = monoatomic_xyz_dict
                        self.species_dict[species.label].final_xyz = monoatomic_xyz_dict
                    if not self.output[species.label]['job_types']['sp'] \
                            and not self.output[species.label]['job_types']['composite'] \
                            and 'sp' not in list(self.job_dict[species.label].keys()) \
                            and 'composite' not in list(self.job_dict[species.label].keys()):
                        # No need to run opt/freq jobs for a monoatomic species, only run sp (or composite if relevant)
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
                    if self.composite_method:
                        # composite-related restart
                        if not self.output[species.label]['job_types']['composite'] \
                                and 'composite' not in list(self.job_dict[species.label].keys()):
                            # doing composite; composite hasn't finished and is not running; spawn composite
                            self.run_composite_job(species.label)
                        elif 'composite' not in list(self.job_dict[species.label].keys()):
                            # composite is done; do other jobs
                            if not self.output[species.label]['job_types']['freq'] \
                                    and 'freq' not in list(self.job_dict[species.label].keys()) \
                                    and (self.species_dict[species.label].is_ts
                                         or self.species_dict[species.label].number_of_atoms > 1):
                                self.run_freq_job(species.label)
                            if self.job_types['rotors']:
                                self.run_scan_jobs(species.label)
                    else:
                        # non-composite-related restart
                        if ('opt' not in list(self.job_dict[species.label].keys()) and not self.job_types['fine']) or \
                                (self.job_types['fine'] and 'opt' not in list(self.job_dict[species.label].keys())
                                 and 'fine' not in list(self.job_dict[species.label].keys())):
                            # opt/fine isn't running
                            if not self.output[species.label]['paths']['geo']:
                                # opt/fine hasn't finished (and isn't running), so run it
                                self.run_opt_job(species.label, fine=self.fine_only)
                            else:
                                # opt/fine is done, check post-opt job types
                                if not self.output[species.label]['job_types']['freq'] \
                                        and 'freq' not in list(self.job_dict[species.label].keys()) \
                                        and (self.species_dict[species.label].is_ts
                                             or self.species_dict[species.label].number_of_atoms > 1):
                                    self.run_freq_job(species.label)
                                if not self.output[species.label]['job_types']['sp'] \
                                        and 'sp' not in list(self.job_dict[species.label].keys()):
                                    self.run_sp_job(species.label)
                                if self.job_types['rotors']:
                                    # some restart-related checks are performed within run_scan_jobs()
                                    self.run_scan_jobs(species.label)
            else:
                # Species is loaded from an Arkane YAML file (no need to execute any job)
                self.output[species.label]['convergence'] = True
                self.output[species.label]['info'] += 'Loaded from an Arkane YAML file; '
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
            if species.initial_xyz is None and species.final_xyz is None and species.conformers \
                    and any([e is not None for e in species.conformer_energies]):
                # the species has no xyz, but has conformers and at least one of the conformers has energy
                self.determine_most_stable_conformer(species.label)
                if species.initial_xyz is not None:
                    if self.composite_method:
                        self.run_composite_job(species.label)
                    else:
                        self.run_opt_job(species.label, fine=self.fine_only)
        self.run_conformer_jobs()
        while self.running_jobs != {}:  # loop while jobs are still running
            logger.debug(f'Currently running jobs:\n{self.running_jobs}')
            self.timer = True
            job_list = list()
            for label in self.unique_species_labels:
                if self.output[label]['convergence'] is False:
                    # skip unconverged species
                    if label in self.running_jobs:
                        del self.running_jobs[label]
                    continue
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
                                if 'conformer' in spec_jobs and spec_jobs != job_name:
                                    break
                            else:
                                # All conformer jobs terminated.
                                # Check isomorphism and run opt on most stable conformer geometry.
                                logger.info(f'\nConformer jobs for {label} successfully terminated.\n')
                                if self.species_dict[label].is_ts:
                                    self.determine_most_likely_ts_conformer(label)
                                else:
                                    self.determine_most_stable_conformer(label)  # also checks isomorphism
                                if self.species_dict[label].initial_xyz is not None:
                                    # if initial_xyz is None, then we're probably troubleshooting conformers, don't opt
                                    if not self.composite_method:
                                        self.run_opt_job(label, fine=self.fine_only)
                                    else:
                                        self.run_composite_job(label)
                            self.timer = False
                            break
                    elif 'opt' in job_name \
                            and self.job_dict[label]['opt'][job_name].job_id not in self.servers_jobs_ids:
                        # val is 'opt1', 'opt2', etc., or 'optfreq1', optfreq2', etc.
                        job = self.job_dict[label]['opt'][job_name]
                        successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                        if successful_server_termination:
                            success = self.parse_opt_geo(label=label, job=job)
                            if success:
                                self.spawn_post_opt_jobs(label=label, job_name=job_name)
                        self.timer = False
                        break
                    elif 'freq' in job_name \
                            and self.job_dict[label]['freq'][job_name].job_id not in self.servers_jobs_ids:
                        # this is NOT an 'optfreq' job
                        job = self.job_dict[label]['freq'][job_name]
                        successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                        if successful_server_termination:
                            self.check_freq_job(label=label, job=job)
                        self.timer = False
                        break
                    elif 'sp' in job_name \
                            and self.job_dict[label]['sp'][job_name].job_id not in self.servers_jobs_ids:
                        job = self.job_dict[label]['sp'][job_name]
                        successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                        if successful_server_termination:
                            self.check_sp_job(label=label, job=job)
                        self.timer = False
                        break
                    elif 'composite' in job_name \
                            and self.job_dict[label]['composite'][job_name].job_id not in self.servers_jobs_ids:
                        job = self.job_dict[label]['composite'][job_name]
                        successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                        if successful_server_termination:
                            success = self.parse_composite_geo(label=label, job=job)
                            if success:
                                self.spawn_post_opt_jobs(label=label, job_name=job_name)
                        self.timer = False
                        break
                    elif 'directed_scan' in job_name \
                            and self.job_dict[label]['directed_scan'][job_name].job_id not in self.servers_jobs_ids:
                        job = self.job_dict[label]['directed_scan'][job_name]
                        successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                        if successful_server_termination:
                            self.check_directed_scan_job(label=label, job=job)
                            if 'cont' in job.directed_scan_type and job.job_status[1]['status'] == 'done':
                                # this is a continuous restricted optimization, spawn the next job in the scan
                                xyz = parser.parse_xyz_from_file(job.local_path_to_output_file)
                                self.spawn_directed_scan_jobs(label=label, rotor_index=job.rotor_index, xyz=xyz)
                        if 'brute_force' in job.directed_scan_type:
                            # Just terminated a brute_force directed scan job.
                            # Are there additional jobs of the same type currently running for this species?
                            self.species_dict[label].rotors_dict[job.rotor_index]['number_of_running_jobs'] -= 1
                            if not self.species_dict[label].rotors_dict[job.rotor_index]['number_of_running_jobs']:
                                # All brute force scan jobs for these pivots terminated.
                                pivots = [scan[1:3] for scan in job.directed_scans]
                                logger.info(f'\nAll brute force directed scan jobs for species {label} between '
                                            f'pivots {pivots} successfully terminated.\n')
                                self.process_directed_scans(label, pivots=job.pivots)
                        self.timer = False
                        break
                    elif 'scan' in job_name and 'directed' not in job_name \
                            and self.job_dict[label]['scan'][job_name].job_id not in self.servers_jobs_ids:
                        job = self.job_dict[label]['scan'][job_name]
                        successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                        if successful_server_termination and job.directed_scans is None:
                            self.check_scan_job(label=label, job=job)
                        self.timer = False
                        break
                    elif 'irc' in job_name \
                            and self.job_dict[label]['irc'][job_name].job_id not in self.servers_jobs_ids:
                        job = self.job_dict[label]['irc'][job_name]
                        successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                        if successful_server_termination:
                            self.check_irc_job(label=label, job=job)
                        self.timer = False
                        break
                    elif 'orbitals' in job_name \
                            and self.job_dict[label]['orbitals'][job_name].job_id not in self.servers_jobs_ids:
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
                    elif 'onedmin' in job_name \
                            and self.job_dict[label]['onedmin'][job_name].job_id not in self.servers_jobs_ids:
                        job = self.job_dict[label]['onedmin'][job_name]
                        successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                        if successful_server_termination:
                            # copy the lennard_jones file to the species output folder (TS's don't have L-J data)
                            lj_output_path = os.path.join(self.project_directory, 'output', 'Species', label,
                                                          'lennard_jones.dat')
                            if os.path.isfile(job.local_path_to_lj_file):
                                shutil.copyfile(job.local_path_to_lj_file, lj_output_path)
                                self.output[label]['job_types']['onedmin'] = True
                                self.species_dict[label].set_transport_data(
                                    lj_path=os.path.join(self.project_directory, 'output', 'Species', label,
                                                         'lennard_jones.dat'),
                                    opt_path=self.output[label]['paths']['geo'], bath_gas=job.bath_gas,
                                    opt_level=self.opt_level)
                        self.timer = False
                        break

                if self.species_dict[label].is_ts and not self.species_dict[label].ts_conf_spawned \
                        and not any([tsg.success is None for tsg in self.species_dict[label].ts_guesses]):
                    # This is a TS Species for which conformers haven't been spawned, and all .success flags
                    # contain a values (whether ``True`` or ``False``)
                    # We're ready to spawn conformer jobs for this TS Species
                    # Todo: no need to wait for all TSGs before spawning the first opt jobs
                    self.run_ts_conformer_jobs(label=label)
                    self.species_dict[label].ts_conf_spawned = True

                if not len(job_list) and not (self.species_dict[label].is_ts
                                              and not self.species_dict[label].ts_conf_spawned):
                    self.check_all_done(label)
                    if not self.running_jobs[label]:
                        # delete the label only if it represents an empty dictionary
                        del self.running_jobs[label]

            if self.timer and len(job_list):
                time.sleep(30)  # wait 30 sec before bugging the servers again.
            t = time.time() - self.report_time
            if t > 3600 and self.running_jobs:
                self.report_time = time.time()
                logger.info(f'Currently running jobs:\n{self.running_jobs}')

        # After exiting the Scheduler while loop, append all YAML species not directly calculated to the species_dict:
        for spc in self.species_list:
            if spc.yml_path is not None:
                self.species_dict[spc.label] = spc

    def run_job(self,
                label: str,
                level_of_theory: Union[Level, dict, str],
                job_type: str,
                xyz: Optional[dict],
                fine: Optional[bool] = False,
                software: Optional[str] = None,
                shift: Optional[str] = '',
                trsh: Optional[str] = '',
                memory: Optional[int] = None,
                conformer: Optional[int] = -1,
                ess_trsh_methods: Optional[list] = None,
                scan: Optional[list] = None,
                pivots: Optional[list] = None,
                occ: Optional[int] = None,
                scan_trsh: Optional[str] = '',
                scan_res: Optional[int] = None,
                max_job_time: Optional[int] = None,
                radius: Optional[float] = None,
                directed_scan_type: Optional[str] = None,
                directed_scans: Optional[list] = None,
                directed_dihedrals: Optional[list] = None,
                rotor_index: Optional[int] = None,
                cpu_cores: Optional[int] = None,
                irc_direction: Optional[str] = None,
                ):
        """
        A helper function for running (all) jobs.

        Args:
            label (str): The species label.
            xyz (dict): The 3D coordinates for the species.
            level_of_theory (Level): The level of theory to use.
            job_type (str): The type of job to run.
            fine (bool, optional): Whether to run an optimization job with a fine grid. `True` to use fine.
            software (str, optional): An ESS software to use.
            shift (str, optional): A string representation alpha- and beta-spin orbitals shifts (molpro only).
            trsh (str, optional): A troubleshooting keyword to be used in input files.
            memory (int, optional): The total job allocated memory in GB.
            conformer (int, optional): Conformer number if optimizing conformers.
            ess_trsh_methods (list, optional): A list of troubleshooting methods already tried out for ESS convergence.
            scan (list, optional): A list representing atom labels for the dihedral scan
                                   (e.g., "2 1 3 5" as a string or [2, 1, 3, 5] as a list of integers).
            pivots (list, optional): The rotor scan pivots, if the job type is scan. Not used directly in these methods,
                                     but used to identify the rotor.
            occ (int, optional): The number of occupied orbitals (core + val) from a molpro CCSD sp calc.
            scan_trsh (str, optional): A troubleshooting method for rotor scans.
            scan_res (int, optional): The rotor scan resolution in degrees.
            max_job_time (int, optional): The maximal allowed job time on the server in hours.
            radius (float, optional): The species radius in Angstrom.
            directed_scan_type (str, optional): The type of the directed scan.
            directed_scans (list, optional): Entries are lists of four-atom dihedral scan indices to constrain.
            directed_dihedrals (list, optional): The dihedral angles of a directed scan job corresponding
                                                 to ``directed_scans``.
            rotor_index (int, optional): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
            cpu_cores (int, optional): The total number of cpu cores requested for a job.
            irc_direction (str, optional): The direction to run the IRC computation.
        """
        max_job_time = max_job_time or self.max_job_time  # if it's None, set to default
        ess_trsh_methods = ess_trsh_methods if ess_trsh_methods is not None else list()
        pivots = pivots if pivots is not None else list()
        species = self.species_dict[label]
        memory = memory if memory is not None else self.memory
        checkfile = self.species_dict[label].checkfile  # defaults to None
        if self.adaptive_levels is not None:
            level_of_theory = self.determine_adaptive_level(original_level_of_theory=level_of_theory, job_type=job_type,
                                                            heavy_atoms=self.species_dict[label].number_of_heavy_atoms)
        job = Job(project=self.project,
                  project_directory=self.project_directory,
                  species_name=label,
                  multiplicity=species.multiplicity,
                  job_type=job_type,
                  level=Level(repr=level_of_theory),
                  ess_settings=self.ess_settings,
                  xyz=xyz,
                  charge=species.charge,
                  conformer=conformer,
                  fine=fine,
                  shift=shift,
                  software=software,
                  is_ts=species.is_ts,
                  scan=scan,
                  pivots=pivots,
                  total_job_memory_gb=memory,
                  args={'keyword': {'trsh': trsh}, 'block': {}} if trsh else None,
                  scan_trsh=scan_trsh,
                  ess_trsh_methods=ess_trsh_methods,
                  bath_gas=self.bath_gas,
                  occ=occ,
                  max_job_time=max_job_time,
                  scan_res=scan_res,
                  checkfile=checkfile,
                  number_of_radicals=species.number_of_radicals,
                  radius=radius,
                  directed_scan_type=directed_scan_type,
                  directed_scans=directed_scans,
                  directed_dihedrals=directed_dihedrals,
                  rotor_index=rotor_index,
                  cpu_cores=cpu_cores,
                  irc_direction=irc_direction,
                  )
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
                self.running_jobs[label].append(f'conformer{conformer}')  # mark as a running job
                self.job_dict[label]['conformers'][conformer] = job  # save job object
                self.job_dict[label]['conformers'][conformer].run()  # run the job
            self.save_restart_dict()
            if job.server not in self.servers:
                self.servers.append(job.server)

    def end_job(self, job, label, job_name):
        """
        A helper function for checking job status, saving in csv file, and downloading output files.

        Args:
            job (Job): The job object.
            label (str): The species label.
            job_name (str): The job name from the running_jobs dict.

        Returns:
             bool: `True` if job terminated successfully on the server, `False` otherwise.
        """
        try:
            job.determine_job_status()  # also downloads output file
        except IOError:
            if job.job_type not in ['orbitals']:
                logger.warning(f'Tried to determine status of job {job.job_name}, but it seems like the job never ran. '
                               f'Re-running job.')
                self._run_a_job(job=job, label=label)
            if job_name in self.running_jobs[label]:
                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))

        if not os.path.exists(job.local_path_to_output_file):
            if 'restart_due_to_file_not_found' in job.ess_trsh_methods:
                job.job_status[0] = 'errored'
                job.job_status[1] = 'errored'
                logger.warning(f'Job {job.job_name} errored because for the second time ARC did not find the output '
                               f'file path {job.local_path_to_output_file}.')
            elif job.job_type not in ['orbitals']:
                job.ess_trsh_methods.append('restart_due_to_file_not_found')
                logger.warning(f'Did not find the output file of job {job.job_name} with path '
                               f'{job.local_path_to_output_file}. Maybe the job never ran. Re-running job.')
                self._run_a_job(job=job, label=label)
            if job_name in self.running_jobs[label]:
                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
            return False

        if job.job_status[0] != 'running' and job.job_status[1]['status'] != 'running':
            if job_name in self.running_jobs[label]:
                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
            self.timer = False
            job.write_completed_job_to_csv_file()
            logger.info(f'  Ending job {job_name} for {label} (run time: {job.run_time})')
            if job.job_status[0] != 'done':
                return False
            if job.software == 'gaussian' and os.path.isfile(os.path.join(job.local_path, 'check.chk')) \
                    and job.job_type in ['conformer', 'opt', 'optfreq', 'composite']:
                check_path = os.path.join(job.local_path, 'check.chk')
                if os.path.isfile(check_path):
                    if job.job_type != 'conformer':
                        self.species_dict[label].checkfile = check_path
            if job.job_type == 'scan' and job.directed_scan_type == 'ess':
                for rotors_dict in self.species_dict[label].rotors_dict.values():
                    if rotors_dict['pivots'] == job.pivots:
                        rotors_dict['scan_path'] = job.local_path_to_output_file
            self.save_restart_dict()
            return True

    def _run_a_job(self, job, label):
        """
        A helper function to run ARC job (used internally).

        Args:
            job (Job): The job object.
            label (str): The species label.
        """
        self.run_job(label=label,
                     xyz=job.xyz,
                     level_of_theory=job.level,
                     job_type=job.job_type,
                     fine=job.fine,
                     software=job.software,
                     shift=job.shift,
                     trsh=job.args['keyword']['trsh'] if 'trsh' in job.args['keyword'] else '',
                     memory=job.total_job_memory_gb,
                     conformer=job.conformer,
                     ess_trsh_methods=job.ess_trsh_methods,
                     scan=job.scan,
                     pivots=job.pivots,
                     occ=job.occ,
                     scan_trsh=job.scan_trsh,
                     scan_res=job.scan_res,
                     max_job_time=job.max_job_time,
                     radius=job.radius,
                     directed_scan_type=job.directed_scan_type,
                     directed_scans=job.directed_scans,
                     directed_dihedrals=job.directed_dihedrals,
                     rotor_index=job.rotor_index,
                     cpu_cores=job.cpu_cores,
                     irc_direction=job.irc_direction,
                     )

    def run_conformer_jobs(self, labels=None):
        """
        Select the most stable conformer for each species using molecular dynamics (force fields) and subsequently
        spawning opt jobs at the conformer level of theory, usually a reasonable yet cheap DFT, e.g., b97d3/6-31+g(d,p).
        The resulting conformer is saved in a string format xyz in the Species initial_xyz attribute.

        Args:
            labels (list): Labels of specific species to run conformer jobs for.
                           If ``None``, conformer jobs will be spawned for all species corresponding to labels in
                           self.unique_species_labels.
        """
        labels_to_consider = labels if labels is not None else self.unique_species_labels
        log_info_printed = False
        for label in labels_to_consider:
            if not self.species_dict[label].is_ts and not self.output[label]['job_types']['opt'] \
                    and 'opt' not in self.job_dict[label] and 'composite' not in self.job_dict[label] \
                    and all([e is None for e in self.species_dict[label].conformer_energies]) \
                    and self.species_dict[label].number_of_atoms > 1 and not self.output[label]['paths']['geo'] \
                    and self.species_dict[label].yml_path is None \
                    and (self.job_types['conformers'] and label not in self.dont_gen_confs
                         or self.species_dict[label].get_xyz(generate=False) is None):
                # This is not a TS, opt (/composite) did not converged nor running, and conformer energies were not set.
                # Also, either 'conformers' are set to True in job_types (and it's not in dont_gen_confs),
                # or they are set to False (or it's in dont_gen_confs), but the species has no 3D information.
                # Generate conformers.
                if not log_info_printed:
                    logger.info('\nStarting (non-TS) species conformational analysis...\n')
                    log_info_printed = True
                if self.species_dict[label].force_field == 'cheap':
                    # just embed in RDKit and use MMFF94s for opt and energies
                    if self.species_dict[label].initial_xyz is None:
                        self.species_dict[label].initial_xyz = self.species_dict[label].get_xyz()
                else:
                    # run the combinatorial method w/o fitting a force field
                    self.species_dict[label].generate_conformers(
                        n_confs=self.n_confs,
                        e_confs=self.e_confs,
                        plot_path=os.path.join(
                            self.project_directory, 'output', 'Species', label, 'geometry', 'conformers'))
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
        Spawn opt jobs at the ts_guesses level of theory for the TS guesses.

        Args:
            label (str): The TS species label.
        """
        plotter.save_conformers_file(project_directory=self.project_directory, label=label,
                                     xyzs=[tsg.initial_xyz for tsg in self.species_dict[label].ts_guesses],
                                     level_of_theory=self.ts_guess_level,
                                     multiplicity=self.species_dict[label].multiplicity,
                                     charge=self.species_dict[label].charge, is_ts=True,
                                     ts_methods=[tsg.method for tsg in self.species_dict[label].ts_guesses])
        successful_tsgs = [tsg for tsg in self.species_dict[label].ts_guesses if tsg.success]
        if len(successful_tsgs) > 1:
            self.job_dict[label]['conformers'] = dict()
            for i, tsg in enumerate(successful_tsgs):
                self.run_job(label=label, xyz=tsg.initial_xyz, level_of_theory=self.ts_guess_level,
                             job_type='conformer', conformer=i)
        elif len(successful_tsgs) == 1:
            if 'opt' not in self.job_dict[label] and 'composite' not in self.job_dict[label]:
                # proceed only if opt (/composite) not already spawned
                rxn = ''
                if self.species_dict[label].rxn_label is not None:
                    rxn = ' of reaction ' + self.species_dict[label].rxn_label
                logger.info(f'Only one TS guess is available for species {label}{rxn}, '
                            f'using it for geometry optimization')
                self.species_dict[label].initial_xyz = successful_tsgs[0].initial_xyz
                if not self.composite_method:
                    self.run_opt_job(label, fine=self.fine_only)
                else:
                    self.run_composite_job(label)
                self.species_dict[label].chosen_ts_method = self.species_dict[label].ts_guesses[0].method

    def run_opt_job(self, label, fine=False):
        """
        Spawn a geometry optimization job. The initial guess is taken from the `initial_xyz` attribute.

        Args:
            label (str): The species label.
            fine (bool): Whether or not a fine grid should be used during optimization
        """
        if 'opt' not in self.job_dict[label]:  # Check whether or not opt jobs have been spawned yet
            # we're spawning the first opt job for this species
            self.job_dict[label]['opt'] = dict()
        if self.species_dict[label].initial_xyz is None:
            raise SpeciesError(f'Cannot execute opt job for {label} without xyz (got None for Species.initial_xyz)')
        self.run_job(label=label, xyz=self.species_dict[label].initial_xyz, level_of_theory=self.opt_level,
                     job_type='opt', fine=fine)

    def run_composite_job(self, label):
        """
        Spawn a composite job (e.g., CBS-QB3) using 'final_xyz' for species ot TS 'label'.

        Args:
            label (str): The species label.
        """
        if not self.composite_method:
            raise SchedulerError(f'Cannot run {label} as a composite method without specifying a method.')
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

        Args:
            label (str): The species label.
        """
        if 'freq' not in self.job_dict[label]:  # Check whether or not freq jobs have been spawned yet
            # we're spawning the first freq job for this species
            self.job_dict[label]['freq'] = dict()
        if self.job_types['freq']:
            self.run_job(label=label, xyz=self.species_dict[label].get_xyz(generate=False),
                         level_of_theory=self.freq_level, job_type='freq')

    def run_sp_job(self,
                   label: str,
                   level: Optional[Level] = None,
                   ):
        """
        Spawn a single point job using 'final_xyz' for species ot TS 'label'.
        If the method is MRCI, first spawn a simple CCSD job, and use orbital determination to run the MRCI job.

        Args:
            label (str): The species label.
            level (Level): An alternative level of theory to run at. If ``None``, self.sp_level will be used.
        """
        level = level or self.sp_level

        # determine_occ(xyz=self.xyz, charge=self.charge)
        if level == self.opt_level and not self.composite_method \
                and 'paths' in self.output[label] and 'geo' in self.output[label]['paths'] \
                and self.output[label]['paths']['geo']:
            logger.info(f'Not running an sp job for {label} at {level} since the optimization was done at the '
                        f'same level of theory. Using the optimization output to parse the sp energy.')
            recent_opt_job_name, recent_opt_job = 'opt_a0', None
            if 'opt' in self.job_dict[label]:
                for opt_job_name, opt_job in self.job_dict[label]['opt'].items():
                    if int(opt_job_name.split('_a')[-1]) > int(recent_opt_job_name.split('_a')[-1]):
                        recent_opt_job_name, recent_opt_job = opt_job_name, opt_job
                self.post_sp_actions(label=label,
                                     sp_path=os.path.join(recent_opt_job.local_path, 'output.out'),
                                     level=level,
                                     )

            # If opt is not in the job dictionary, the likely explanation is this job has been restarted
            elif 'geo' in self.output[label]['paths']:  # Then just use this path directly
                self.post_sp_actions(label=label,
                                     sp_path=self.output[label]['paths']['geo'],
                                     level=level,
                                     )

            else:
                raise RuntimeError(f'Unable to set the path for the sp job for species {label}')

            return
        if 'sp' not in self.job_dict[label]:  # Check whether or not single point jobs have been spawned yet
            # we're spawning the first sp job for this species
            self.job_dict[label]['sp'] = dict()
        if self.composite_method:
            raise SchedulerError(f'run_sp_job() was called for {label} which has a composite method level of theory')
        if 'mrci' in level.method:
            if self.job_dict[label]['sp']:
                # Parse orbital information from the CCSD job, then run MRCI
                job0 = None
                job_name_0 = 0
                for job_name, job in self.job_dict[label]['sp']:
                    if int(job_name.split('_a')[-1]) > job_name_0:
                        job_name_0 = int(job_name.split('_a')[-1])
                        job0 = job
                with open(job0.local_path_to_output_file, 'r') as f:
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
                        raise SchedulerError(f'Could not determine number of core and valence orbitals from CCSD '
                                             f'sp calculation for {label}')
                occ = val + core  # the occupied orbitals are the core and valence orbitals
                self.run_job(label=label,
                             xyz=self.species_dict[label].get_xyz(generate=False),
                             level_of_theory='ccsd/vdz',
                             job_type='sp',
                             occ=occ)
            else:
                # MRCI was requested but no sp job ran for this species, run CCSD first
                logger.info(f'running a CCSD job for {label} before MRCI')
                self.run_job(label=label,
                             xyz=self.species_dict[label].get_xyz(generate=False),
                             level_of_theory='ccsd/vdz',
                             job_type='sp')
        if self.job_types['sp']:
            self.run_job(label=label,
                         xyz=self.species_dict[label].get_xyz(generate=False),
                         level_of_theory=level,
                         job_type='sp',
                         )

    def run_scan_jobs(self, label):
        """
        Spawn rotor scan jobs using 'final_xyz' for species (or TS).

        Args:
            label (str): The species label.
        """
        if self.job_types['rotors'] and isinstance(self.species_dict[label].rotors_dict, dict):
            for i, rotor in self.species_dict[label].rotors_dict.items():
                # Since this function applied in multiple cases, all cases are listed for debugging
                # [have not started] success = None, and scan_path = ''
                # [first time calculating] success = None, and scan_path = ''
                # [converged, good] success = True, and scan_path is file
                # [converged, invalidated] success = False, and scan_path is file
                # [previous converged, troubleshooting] success = None, and scan_path (previous scan) is file
                # [previous converged, lower conformer] success = None, and scan_path (previous scan) is file
                # [not a torsion] success = False, and scan_path = ''
                if rotor['success'] is not None:
                    if rotor['scan_path'] and not os.path.isfile(rotor['scan_path']):
                        # For some reason the output file does not exist
                        rotor['success'] = None
                    else:
                        continue
                scan = rotor['scan']
                if not isinstance(scan[0], list):
                    # check that a 1D rotors is not linear
                    coords = xyz_to_coords_list(self.species_dict[label].get_xyz())
                    v1 = [c1 - c2 for c1, c2 in zip(coords[scan[0] - 1], coords[scan[1] - 1])]
                    v2 = [c2 - c1 for c1, c2 in zip(coords[scan[1] - 1], coords[scan[2] - 1])]
                    v3 = [c1 - c2 for c1, c2 in zip(coords[scan[2] - 1], coords[scan[3] - 1])]
                    angle1, angle2 = get_angle(v1, v2, units='degs'), get_angle(v2, v3, units='degs')
                    if any([abs(angle - 180.0) < 0.15 for angle in [angle1, angle2]]):
                        # this is not a torsional mode, invalidate rotor
                        rotor['success'] = False
                        rotor['invalidation_reason'] = \
                            f'not a torsional mode (angles = {angle1:.2f}, {angle2:.2f} degrees)'
                        continue
                pivots = rotor['pivots']
                directed_scan_type = rotor['directed_scan_type'] if 'directed_scan_type' in rotor else ''

                if directed_scan_type:
                    # this is a directed scan
                    # check this job isn't already running on the server (from a restarted project)
                    if 'directed_scan' not in self.job_dict[label]:
                        # we're spawning the first brute force scan jobs for this species
                        self.job_dict[label]['directed_scan'] = dict()
                    for directed_scan_job in self.job_dict[label]['directed_scan'].values():
                        if directed_scan_job.pivots == pivots \
                                and directed_scan_job.job_name in self.running_jobs[label]:
                            break
                    else:
                        if 'cont' in directed_scan_type:
                            for directed_pivots in self.job_dict[label]['directed_scan'].keys():
                                if directed_pivots == pivots \
                                        and self.job_dict[label]['directed_scan'][directed_pivots]:
                                    # the previous job hasn't finished
                                    break
                            else:
                                self.spawn_directed_scan_jobs(label, rotor_index=i)
                        else:
                            self.spawn_directed_scan_jobs(label, rotor_index=i)
                else:
                    # this is a "normal" scan (not directed)
                    # check this job isn't already running on the server(from a restarted project)
                    if 'scan' not in self.job_dict[label]:
                        # we're spawning the first scan job for this species
                        self.job_dict[label]['scan'] = dict()
                    for scan_job in self.job_dict[label]['scan'].values():
                        if scan_job.pivots == pivots and scan_job.job_name in self.running_jobs[label]:
                            break
                    else:
                        self.run_job(label=label,
                                     xyz=self.species_dict[label].get_xyz(generate=False),
                                     level_of_theory=self.scan_level,
                                     job_type='scan',
                                     scan=scan,
                                     pivots=pivots)

    def run_irc_job(self, label, irc_direction='forward'):
        """
        Spawn an IRC job.

        Args:
            label (str): The species label.
            irc_direction (str): The IRC job direction, either 'forward' or 'reverse'.
        """
        self.run_job(label=label,
                     xyz=self.species_dict[label].get_xyz(generate=False),
                     level_of_theory=self.irc_level,
                     job_type='irc',
                     irc_direction=irc_direction)

    def run_orbitals_job(self, label):
        """
        Spawn orbitals job used for molecular orbital visualization.
        Currently supporting QChem for printing the orbitals, the output could be visualized using IQMol.

        Args:
            label (str): The species label.
        """
        self.run_job(label=label,
                     xyz=self.species_dict[label].get_xyz(generate=False),
                     level_of_theory=self.orbitals_level,
                     job_type='orbitals')

    def run_onedmin_job(self, label):
        """
        Spawn a lennard-jones calculation using OneDMin.

        Args:
            label (str): The species label.
        """
        if 'onedmin' not in self.ess_settings:
            logger.error('Cannot execute a Lennard Jones job without the OneDMin software')
        elif 'onedmin' not in self.job_dict[label]:
            self.run_job(label=label,
                         xyz=self.species_dict[label].get_xyz(generate=False),
                         job_type='onedmin',
                         level_of_theory='')

    def spawn_post_opt_jobs(self,
                            label: str,
                            job_name: str,
                            ):
        """
        Spawn additional jobs after opt has converged.

        Args:
            label (str): The species label.
            job_name (str): The opt job name (used for differentiating between ``opt`` and ``optfreq`` jobs).
        """
        composite = 'composite' in job_name  # Whether "post composite jobs" need to be spawned
        if not composite and self.composite_method:
            # This was originally a composite method, probably troubleshooted as 'opt'
            self.run_composite_job(label)
        elif composite and not self.composite_method:
            # This wasn't originally a composite method, probably troubleshooted as such
            self.run_opt_job(label, fine=self.fine_only)
        else:
            if self.job_types['irc'] and self.species_dict[label].is_ts:
                self.run_irc_job(label=label, irc_direction='forward')
                self.run_irc_job(label=label, irc_direction='reverse')
            if self.species_dict[label].number_of_atoms > 1:
                if 'freq' not in job_name:
                    # this is either an opt or a composite job, spawn freq
                    self.run_freq_job(label)
                elif not composite:
                    # this is an 'optfreq' job type, don't run freq
                    self.check_freq_job(label=label, job=self.job_dict[label]['optfreq'][job_name])
            if not composite:
                # don't use a solvation correction for this sp job if a solvation_scheme_level was specified
                self.run_sp_job(label)
            if self.species_dict[label].mol is None:
                # useful for TS species where xyz might not be given to perceive a .mol attribute,
                # and a user guess, if provided, cannot always be trusted
                self.species_dict[label].mol_from_xyz()
            if not self.species_dict[label].rotors_dict:
                self.species_dict[label].determine_rotors()
            self.run_scan_jobs(label)

        if composite and self.composite_method:
            self.post_sp_actions(label=label,
                                 sp_path=os.path.join(self.job_dict[label]['composite'][job_name].local_path,
                                                      'output.out'))

        if self.job_types['orbitals'] and 'orbitals' not in self.job_dict[label]:
            self.run_orbitals_job(label)

        if self.job_types['onedmin'] and not self.species_dict[label].is_ts:
            self.run_onedmin_job(label)

        if self.job_types['bde'] and self.species_dict[label].bdes is not None:
            bde_species_list = self.species_dict[label].scissors()
            for bde_species in bde_species_list:
                if bde_species.label != 'H':
                    # H is was added in main
                    logging.info(f'Creating the BDE species {bde_species.label} from the original species {label}')
                    self.species_list.append(bde_species)
                    self.species_dict[bde_species.label] = bde_species
                    self.unique_species_labels.append(bde_species.label)
                    self.initialize_output_dict(label=bde_species.label)
                    self.job_dict[bde_species.label] = dict()
                    self.running_jobs[bde_species.label] = list()
                    if bde_species.number_of_atoms == 1:
                        logger.debug(f'Species {bde_species.label} is monoatomic')
                        # No need to run opt/freq jobs for a monoatomic species, only run sp (or composite if relevant)
                        if self.composite_method:
                            self.run_composite_job(bde_species.label)
                        else:
                            self.run_sp_job(label=bde_species.label)
            # determine the lowest energy conformation of radicals generated in BDE calculations
            self.run_conformer_jobs(labels=[species.label for species in bde_species_list
                                            if species.number_of_atoms > 1])

    def spawn_directed_scan_jobs(self, label, rotor_index, xyz=None):
        """
        Spawn directed scan jobs.
        Directed scan types could be one of the following: 'brute_force_sp', 'brute_force_opt', 'cont_opt',
        'brute_force_sp_diagonal', 'brute_force_opt_diagonal', or 'cont_opt_diagonal'.
        Here we treat ``cont`` and ``brute_force`` separately, and also consider the ``diagonal`` keyword.
        The differentiation between ``sp`` and ``opt`` is done in the Job module.

        Args:
            label (str): The species label.
            rotor_index (int): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
            xyz (str, optional): The 3D coordinates for a continuous directed scan.

        Raises:
            InputError: If the species directed scan type has an unexpected value,
                        or if ``xyz`` wasn't given for a cont_opt job.
            SchedulerError: If the rotor scan resolution as defined in settings.py is illegal.
        """
        increment = rotor_scan_resolution
        if divmod(360, increment)[1]:
            raise SchedulerError(f'The directed scan got an illegal scan resolution of {increment}')
        scans = self.species_dict[label].rotors_dict[rotor_index]['scan']
        pivots = self.species_dict[label].rotors_dict[rotor_index]['pivots']
        directed_scan_type = self.species_dict[label].rotors_dict[rotor_index]['directed_scan_type']
        xyz = xyz or self.species_dict[label].get_xyz(generate=True)

        if 'cont' not in directed_scan_type and 'brute' not in directed_scan_type and 'ess' not in directed_scan_type:
            raise InputError(f'directed_scan_type must be either continuous or brute force, got: {directed_scan_type}')

        if 'ess' in directed_scan_type:
            # allow the ESS to control the scan
            self.run_job(label=label, xyz=xyz, level_of_theory=self.scan_level, job_type='scan',
                         directed_scan_type=directed_scan_type, directed_scans=scans, rotor_index=rotor_index,
                         pivots=pivots)

        elif 'brute' in directed_scan_type:
            # spawn jobs all at once
            dihedrals = dict()

            for scan in scans:
                original_dihedral = get_angle_in_180_range(calculate_dihedral_angle(coords=xyz['coords'],
                                                                                    torsion=scan,
                                                                                    index=1))
                dihedrals[tuple(scan)] = [get_angle_in_180_range(original_dihedral + i * increment) for i in
                                          range(int(360 / increment) + 1)]
            modified_xyz = xyz
            if 'diagonal' not in directed_scan_type:
                # increment dihedrals one by one (resulting in an ND scan)
                all_dihedral_combinations = list(itertools.product(*[dihedrals[tuple(scan)] for scan in scans]))
                for dihedral_tuple in all_dihedral_combinations:
                    for scan, dihedral in zip(scans, dihedral_tuple):
                        self.species_dict[label].set_dihedral(scan=scan, deg_abs=dihedral, count=False,
                                                              xyz=modified_xyz)
                        modified_xyz = self.species_dict[label].initial_xyz
                    self.species_dict[label].rotors_dict[rotor_index]['number_of_running_jobs'] += 1
                    self.run_job(label=label,
                                 xyz=modified_xyz,
                                 level_of_theory=self.scan_level,
                                 job_type='directed_scan',
                                 directed_scan_type=directed_scan_type,
                                 directed_scans=scans,
                                 directed_dihedrals=list(dihedral_tuple),
                                 rotor_index=rotor_index,
                                 pivots=pivots)
            else:
                # increment all dihedrals at once (resulting in a unique 1D scan along several changing dimensions)
                for i in range(len(dihedrals[tuple(scans[0])])):
                    for scan in scans:
                        dihedral = dihedrals[tuple(scan)][i]
                        self.species_dict[label].set_dihedral(scan=scan, deg_abs=dihedral, count=False,
                                                              xyz=modified_xyz)
                        modified_xyz = self.species_dict[label].initial_xyz
                    directed_dihedrals = [dihedrals[tuple(scan)][i] for scan in scans]
                    self.species_dict[label].rotors_dict[rotor_index]['number_of_running_jobs'] += 1
                    self.run_job(label=label,
                                 xyz=modified_xyz,
                                 level_of_theory=self.scan_level,
                                 job_type='directed_scan',
                                 directed_scan_type=directed_scan_type,
                                 directed_scans=scans,
                                 directed_dihedrals=directed_dihedrals,
                                 rotor_index=rotor_index,
                                 pivots=pivots,
                                 )

        elif 'cont' in directed_scan_type:
            # spawn jobs one by one
            if not len(self.species_dict[label].rotors_dict[rotor_index]['cont_indices']):
                self.species_dict[label].rotors_dict[rotor_index]['cont_indices'] = [0] * len(scans)
            if not len(self.species_dict[label].rotors_dict[rotor_index]['original_dihedrals']):
                self.species_dict[label].rotors_dict[rotor_index]['original_dihedrals'] = \
                    [f'{calculate_dihedral_angle(coords=xyz["coords"], torsion=scan, index=1):.2f}'
                     for scan in self.species_dict[label].rotors_dict[rotor_index]['scan']]  # stores as str for YAML
            rotor_dict = self.species_dict[label].rotors_dict[rotor_index]
            scans = rotor_dict['scan']
            pivots = rotor_dict['pivots']
            max_num = 360 / increment + 1  # dihedral angles per scan
            original_dihedrals = list()
            for dihedral in rotor_dict['original_dihedrals']:
                original_dihedrals.append(get_angle_in_180_range(dihedral))
            if not any(self.species_dict[label].rotors_dict[rotor_index]['cont_indices']):
                # this is the first call for this cont_opt directed rotor, spawn the first job w/o changing dihedrals
                self.run_job(label=label,
                             xyz=self.species_dict[label].final_xyz,
                             level_of_theory=self.scan_level,
                             job_type='directed_scan',
                             directed_scan_type=directed_scan_type,
                             directed_scans=scans,
                             directed_dihedrals=original_dihedrals,
                             rotor_index=rotor_index, pivots=pivots,
                             )
                self.species_dict[label].rotors_dict[rotor_index]['cont_indices'][0] += 1
                return
            else:
                # this is NOT the first call for this cont_opt directed rotor, check that ``xyz`` was given.
                if xyz is None:
                    # xyz is None only at the first time cont opt is spawned, where cont_index is [0, 0,... 0].
                    raise InputError('xyz argument must be given for a continuous scan job')
                # check whether this rotor is done
                if self.species_dict[label].rotors_dict[rotor_index]['cont_indices'][-1] == max_num - 1:  # 0-indexed
                    # no more counters to increment, all done!
                    logger.info(f'Completed all jobs for the continuous directed rotor scan for species {label} '
                                f'between pivots {pivots}')
                    self.process_directed_scans(label, pivots)
                    return

            modified_xyz = xyz
            dihedrals = list()
            for index, (original_dihedral, scan_) in enumerate(zip(original_dihedrals, scans)):
                dihedral = original_dihedral + \
                           self.species_dict[label].rotors_dict[rotor_index]['cont_indices'][index] * increment
                # change the original dihedral so we won't end up with two calcs for 180.0, but none for -180.0
                # (it only matters for plotting, the geometry is of course the same)
                dihedral = get_angle_in_180_range(dihedral)
                dihedrals.append(dihedral)
                # Only change the dihedrals in the xyz if this torsion corresponds to the current index,
                # or if this is a diagonal scan.
                # Species.set_dihedral() uses .final_xyz or the given xyz to modify the .initial_xyz
                # attribute to the desired dihedral.
                self.species_dict[label].set_dihedral(scan=scan_, deg_abs=dihedral, count=False, xyz=modified_xyz)
                modified_xyz = self.species_dict[label].initial_xyz
            self.run_job(label=label,
                         xyz=modified_xyz,
                         level_of_theory=self.scan_level,
                         job_type='directed_scan',
                         directed_scan_type=directed_scan_type,
                         directed_scans=scans,
                         directed_dihedrals=dihedrals,
                         rotor_index=rotor_index,
                         pivots=pivots,
                         )

            if 'diagonal' in directed_scan_type:
                # increment ALL counters for a diagonal scan
                self.species_dict[label].rotors_dict[rotor_index]['cont_indices'] = \
                    [self.species_dict[label].rotors_dict[rotor_index]['cont_indices'][0] + 1] * len(scans)
            else:
                # increment the counter sequentially (non-diagonal scan)
                for index in range(len(scans)):
                    if self.species_dict[label].rotors_dict[rotor_index]['cont_indices'][index] < max_num - 1:
                        self.species_dict[label].rotors_dict[rotor_index]['cont_indices'][index] += 1
                        break
                    elif (self.species_dict[label].rotors_dict[rotor_index]['cont_indices'][index] == max_num - 1
                          and index < len(scans) - 1):
                        self.species_dict[label].rotors_dict[rotor_index]['cont_indices'][index] = 0

    def process_directed_scans(self, label, pivots):
        """
        Process all directed rotors for a species and check the quality of the scan.

        rotors_dict structure (attribute of ARCSpecies)::

            rotors_dict: {1: {'pivots': ``list``,
                              'top': ``list``,
                              'scan': ``list``,
                              'number_of_running_jobs': ``int``,
                              'success': ``Optional[bool]``,
                              'invalidation_reason': ``str``,
                              'times_dihedral_set': ``int``,
                              'scan_path': <path to scan output file>,
                              'max_e': ``float``,  # in kJ/mol,
                              'symmetry': ``int``,
                              'dimensions': ``int``,
                              'original_dihedrals': ``list``,
                              'cont_indices': ``list``,
                              'directed_scan_type': ``str``,
                              'directed_scan': ``dict``,  # keys: tuples of dihedrals as strings,
                                                          # values: dicts of energy, xyz, is_isomorphic, trsh
                             }
                          2: {}, ...
                         }

        Args:
            label (str): The species label.
            pivots (list): The rotor pivots.
        """
        for rotor_dict_index in self.species_dict[label].rotors_dict.keys():
            rotor_dict = self.species_dict[label].rotors_dict[rotor_dict_index]  # avoid modifying the iterator
            if rotor_dict['pivots'] == pivots:
                # identified a directed scan (either continuous or brute force, they're treated the same here)
                dihedrals = [[float(dihedral) for dihedral in dihedral_string_tuple]
                             for dihedral_string_tuple in rotor_dict['directed_scan'].keys()]
                sorted_dihedrals = sorted(dihedrals)
                min_energy = extermum_list([directed_scan_dihedral['energy']
                                            for directed_scan_dihedral in rotor_dict['directed_scan'].values()],
                                           return_min=True)
                trshed_points = 0
                if rotor_dict['directed_scan_type'] == 'ess':
                    # parse the single output file
                    results = parser.parse_nd_scan_energies(path=rotor_dict['scan_path'])[0]
                else:
                    results = {'directed_scan_type': rotor_dict['directed_scan_type'],
                               'scans': rotor_dict['scan'],
                               'directed_scan': rotor_dict['directed_scan']}
                    for dihedral_list in sorted_dihedrals:
                        dihedrals_key = tuple(f'{dihedral:.2f}' for dihedral in dihedral_list)
                        dihedral_dict = results['directed_scan'][dihedrals_key]
                        if dihedral_dict['trsh']:
                            trshed_points += 1
                        if dihedral_dict['energy'] is not None:
                            dihedral_dict['energy'] -= min_energy  # set 0 at the minimal energy
                folder_name = 'rxns' if self.species_dict[label].is_ts else 'Species'
                rotor_yaml_file_path = os.path.join(self.project_directory, 'output', folder_name, label, 'rotors',
                                                    f'{pivots}_{rotor_dict["directed_scan_type"]}.yml')
                plotter.save_nd_rotor_yaml(results, path=rotor_yaml_file_path)
                self.species_dict[label].rotors_dict[rotor_dict_index]['scan_path'] = rotor_yaml_file_path
                if trshed_points:
                    logger.warning(f'Directed rotor scan for species {label} between pivots {rotor_dict["pivots"]} '
                                   f'had {trshed_points} points that required optimization troubleshooting.')
                rotor_path = os.path.join(self.project_directory, 'output', folder_name, label, 'rotors')
                if len(results['scans']) == 1:  # plot 1D rotor
                    plotter.plot_1d_rotor_scan(
                        results=results,
                        path=rotor_path,
                        scan=rotor_dict['scan'],
                        label=label,
                        original_dihedral=self.species_dict[label].rotors_dict[rotor_dict_index]['original_dihedrals'],
                    )
                elif len(results['scans']) == 2:  # plot 2D rotor
                    plotter.plot_2d_rotor_scan(results=results, path=rotor_path)
                else:
                    logger.debug('not plotting ND rotors with N > 2')

    def process_conformers(self, label):
        """
        Process the generated conformers and spawn DFT jobs at the conformer_level.
        If more than one conformer is available, they will be optimized at the DFT conformer_level.

        Args:
            label (str): The species label.
        """
        plotter.save_conformers_file(project_directory=self.project_directory,
                                     label=label,
                                     xyzs=self.species_dict[label].conformers,
                                     level_of_theory=self.conformer_level,
                                     multiplicity=self.species_dict[label].multiplicity,
                                     charge=self.species_dict[label].charge,
                                     is_ts=False,
                                     )  # before optimization
        self.species_dict[label].conformers_before_opt = tuple(self.species_dict[label].conformers)
        if self.species_dict[label].initial_xyz is None and self.species_dict[label].final_xyz is None \
                and not self.testing:
            if len(self.species_dict[label].conformers) > 1:
                self.job_dict[label]['conformers'] = dict()
                for i, xyz in enumerate(self.species_dict[label].conformers):
                    self.run_job(label=label, xyz=xyz, level_of_theory=self.conformer_level,
                                 job_type='conformer', conformer=i)
            elif len(self.species_dict[label].conformers) == 1:
                logger.info(f'Only one conformer is available for species {label}, using it as initial xyz.')
                self.species_dict[label].initial_xyz = self.species_dict[label].conformers[0]
                # check whether this conformer is isomorphic to the species 2D graph representation
                # (since this won't be checked in determine_most_stable_conformer, as there's only one)
                is_isomorphic, spawn_jobs = False, True
                try:
                    b_mol = molecules_from_xyz(self.species_dict[label].initial_xyz,
                                               multiplicity=self.species_dict[label].multiplicity,
                                               charge=self.species_dict[label].charge)[1]
                except SanitizationError:
                    b_mol = None
                    if self.allow_nonisomorphic_2d or self.species_dict[label].charge:
                        # we'll optimize the single conformer even if it is not isomorphic with the 2D graph
                        logger.error(f'The single conformer {label} could not be checked for isomorphism with the 2D '
                                     f'graph representation {self.species_dict[label].mol.copy(deep=True).to_smiles()}.'
                                     f' Optimizing this conformer anyway.')
                        if self.species_dict[label].charge:
                            logger.warning(f'Isomorphism check cannot be done for charged species {label}')
                        self.output[label]['conformers'] += 'Single conformer could not be checked for isomorphism; '
                        self.output[label]['job_types']['conformers'] = True
                        self.species_dict[label].conf_is_isomorphic, spawn_jobs = True, True
                    else:
                        logger.error(f'The only conformer for species {label} could not be checked for isomorphism '
                                     f'with the 2D graph representation '
                                     f'{self.species_dict[label].mol.copy(deep=True).to_smiles()}. NOT calculating '
                                     f'this species. To change this behaviour, pass `allow_nonisomorphic_2d = True`.')
                        self.species_dict[label].conf_is_isomorphic, spawn_jobs = False, False
                if b_mol is not None:
                    try:
                        is_isomorphic = check_isomorphism(self.species_dict[label].mol, b_mol)
                    except ValueError as e:
                        logger.error(f'Could not determine isomorphism for species {label}. '
                                     f'Got the following error:\n{e}')
                        if are_coords_compliant_with_graph(xyz=self.species_dict[label].initial_xyz, mol=b_mol):
                            # this is still considered isomorphic
                            self.species_dict[label].conf_is_isomorphic, spawn_jobs = True, True
                        else:
                            self.species_dict[label].conf_is_isomorphic, spawn_jobs = False, self.allow_nonisomorphic_2d
                    if is_isomorphic:
                        logger.info(f'The only conformer for species {label} was found to be isomorphic '
                                    f'with the 2D graph representation {b_mol.copy(deep=True).to_smiles()}\n')
                        self.output[label]['conformers'] += 'single conformer passed isomorphism check; '
                        self.output[label]['job_types']['conformers'] = True
                        self.species_dict[label].conf_is_isomorphic = True
                    else:
                        logger.error(f'The only conformer for species {label} is not isomorphic '
                                     f'with the 2D graph representation {b_mol.copy(deep=True).to_smiles()}\n')
                        self.species_dict[label].conf_is_isomorphic = False
                        if are_coords_compliant_with_graph(xyz=self.species_dict[label].initial_xyz, mol=b_mol):
                            logger.info('Using this conformer anyway (it is compliant with the 2D graph)')
                            spawn_jobs = True
                        elif self.allow_nonisomorphic_2d:
                            logger.info('Using this conformer anyway (allow_nonisomorphic_2d was set to True)')
                            spawn_jobs = True
                        else:
                            logger.info('Not using this conformer (to change this behavior, set allow_nonisomorphic_2d '
                                        'to True)')
                            spawn_jobs = False
                if spawn_jobs:
                    if not self.composite_method:
                        if self.job_types['opt']:
                            self.run_opt_job(label, self.fine_only)
                        else:
                            # opt wasn't requested, skip directly to additional relevant job types
                            if self.job_types['freq']:
                                self.run_freq_job(label)
                            if self.job_types['sp']:
                                self.run_sp_job(label)
                            if self.job_types['rotors']:
                                self.run_scan_jobs(label)
                            if self.job_types['onedmin']:
                                self.run_onedmin_job(label)
                            if self.job_types['orbitals']:
                                self.run_orbitals_job(label)
                    else:
                        self.run_composite_job(label)

    def parse_conformer(self, job, label, i):
        """
        Parse E0 (kJ/mol) from the conformer opt output file.
        For species, save it in the Species.conformer_energies attribute.
        Fot TSs, save it in the TSGuess.energy attribute, and also parse the geometry.

        Args:
            job (Job): The conformer job object.
            label (str): The TS species label.
            i (int): The conformer index.
        """
        if job.job_status[1]['status'] == 'done':
            xyz = parser.parse_geometry(path=job.local_path_to_output_file)
            energy = parser.parse_e_elect(path=job.local_path_to_output_file)
            if self.species_dict[label].is_ts:
                self.species_dict[label].ts_guesses[i].energy = energy
                self.species_dict[label].ts_guesses[i].opt_xyz = xyz
                self.species_dict[label].ts_guesses[i].index = i
                if energy is not None:
                    logger.debug(f'Energy for TSGuess {i} of {label} is {energy:.2f}')
                else:
                    logger.debug(f'Energy for TSGuess {i} of {label} is None')
            else:
                self.species_dict[label].conformer_energies[i] = energy
                self.species_dict[label].conformers[i] = xyz
                if energy is not None:
                    logger.debug(f'Energy for conformer {i} of {label} is {energy:.2f}')
                else:
                    logger.debug(f'Energy for conformer {i} of {label} is None')
        else:
            logger.warning(f'Conformer {i} for {label} did not converge.')

    def determine_most_stable_conformer(self, label):
        """
        Determine the most stable conformer for a species (which is not a TS).
        Also run an isomorphism check.
        Save the resulting xyz as `initial_xyz`.

        Args:
            label (str): The species label.
        """
        if self.species_dict[label].is_ts:
            raise SchedulerError('The determine_most_stable_conformer() method does not deal with transition '
                                 'state guesses.')
        if 'conformers' in self.job_dict[label] and all(e is None for e in self.species_dict[label].conformer_energies):
            logger.error(f'No conformer converged for species {label}! Trying to troubleshoot conformer jobs...')
            for i, job in self.job_dict[label]['conformers'].items():
                self.troubleshoot_ess(label, job, level_of_theory=job.level, conformer=job.conformer)
        else:
            conformer_xyz = None
            xyzs = list()
            if self.species_dict[label].conformer_energies:
                xyzs = self.species_dict[label].conformers
            else:
                for job in self.job_dict[label]['conformers'].values():
                    xyzs.append(parser.parse_xyz_from_file(path=job.local_path_to_output_file))
            xyzs_in_original_order = xyzs
            energies, xyzs = sort_two_lists_by_the_first(self.species_dict[label].conformer_energies, xyzs)
            plotter.save_conformers_file(project_directory=self.project_directory, label=label,
                                         xyzs=self.species_dict[label].conformers, level_of_theory=self.conformer_level,
                                         multiplicity=self.species_dict[label].multiplicity,
                                         charge=self.species_dict[label].charge, is_ts=False,
                                         energies=self.species_dict[label].conformer_energies)  # after optimization
            # Run isomorphism checks if a 2D representation is available
            if self.species_dict[label].mol is not None:
                for i, xyz in enumerate(xyzs):
                    try:
                        b_mol = molecules_from_xyz(xyz,
                                                   multiplicity=self.species_dict[label].multiplicity,
                                                   charge=self.species_dict[label].charge)[1]
                    except SanitizationError:
                        b_mol = None
                    if b_mol is not None:
                        try:
                            is_isomorphic = check_isomorphism(self.species_dict[label].mol, b_mol)
                        except ValueError as e:
                            if self.allow_nonisomorphic_2d or \
                                    are_coords_compliant_with_graph(xyz=xyz, mol=self.species_dict[label].mol):
                                logger.error(f'Could not determine isomorphism for charged species {label}. '
                                             f'Optimizing the most stable conformer anyway. Got the '
                                             f'following error:\n{e}')
                                conformer_xyz = xyz
                                break
                            else:
                                logger.error(f'Could not determine isomorphism for species {label}. Got the '
                                             f'following error:\n{e}')
                            break
                        if is_isomorphic or are_coords_compliant_with_graph(xyz=xyz, mol=self.species_dict[label].mol):
                            if i == 0:
                                b_mol_smiles = b_mol.copy(deep=True).to_smiles() \
                                    if b_mol is not None else '<no 2D structure available>'
                                logger.info(f'Most stable conformer for species {label} was found to be isomorphic '
                                            f'with the 2D graph representation {b_mol_smiles}\n')
                                conformer_xyz = xyz
                                if 'passed isomorphism check' not in self.output[label]['conformers']:
                                    self.output[label]['conformers'] += f'most stable conformer ({i}) passed ' \
                                                                        f'isomorphism check; '
                                self.species_dict[label].conf_is_isomorphic = True
                            else:
                                if energies[i] is not None:
                                    mol = molecules_from_xyz(xyzs[0],
                                                             multiplicity=self.species_dict[label].multiplicity,
                                                             charge=self.species_dict[label].charge)[1]
                                    smiles_1 = self.species_dict[label].mol.copy(deep=True).to_smiles() \
                                        if self.species_dict[label].mol is not None else '<no 2D structure available>'
                                    smiles_2 = mol.copy(deep=True).to_smiles() \
                                        if mol is not None else '<no 2D structure available>'
                                    logger.info(f'A conformer for species {label} was found to be isomorphic with the '
                                                f'2D graph representation {smiles_1}.\n'
                                                f'This conformer is {energies[i] - energies[0]:.2f} kJ/mol above the '
                                                f'most stable one corresponding to {smiles_2} (and is not isomorphic). '
                                                f'Using the isomorphic conformer for further geometry optimization.')
                                    self.output[label]['conformers'] += f'Conformer {i} was found to be the lowest ' \
                                                                        f'energy isomorphic conformer; '
                                conformer_xyz = xyz
                            if 'Conformers optimized and compared' not in self.output[label]['conformers']:
                                self.output[label]['conformers'] += \
                                    f'Conformers optimized and compared at {self.conformer_level.simple()}; '
                            break
                        else:
                            if i == 0:
                                self.output[label]['conformers'] += f'most stable conformer ({i}) did not ' \
                                                                    f'pass isomorphism check; '
                                self.species_dict[label].conf_is_isomorphic = False
                                smiles_1 = b_mol.copy(deep=True).to_smiles() \
                                    if b_mol is not None else '<no 2D structure available>'
                                smiles_2 = self.species_dict[label].mol.copy(deep=True).to_smiles() \
                                    if self.species_dict[label].mol is not None else '<no 2D structure available>'
                                logger.warning(f'Most stable conformer for species {label} with structure '
                                               f'{smiles_1} was found to be NON-isomorphic '
                                               f'with the 2D graph representation {smiles_2}. '
                                               f'Searching for a different conformer that is isomorphic...')
                else:
                    # all conformers for the species failed isomorphism test
                    smiles_list = list()
                    for xyz in xyzs:
                        try:
                            b_mol = molecules_from_xyz(xyz,
                                                       multiplicity=self.species_dict[label].multiplicity,
                                                       charge=self.species_dict[label].charge)[1]
                            smiles_list.append(b_mol.copy(deep=True).to_smiles())
                        except (SanitizationError, AttributeError):
                            smiles_list.append('Could not perceive molecule')
                    if self.allow_nonisomorphic_2d or self.species_dict[label].charge:
                        # we'll optimize the most stable conformer even if it is not isomorphic to the 2D graph
                        logger.error(f'No conformer for {label} was found to be isomorphic with the 2D graph '
                                     f'representation {self.species_dict[label].mol.copy(deep=True).to_smiles()} '
                                     f'(got: {smiles_list}).  Optimizing the most stable conformer anyway.')
                        self.output[label]['conformers'] += 'No conformer was found to be isomorphic with ' \
                                                            'the 2D graph representation; '
                        if self.species_dict[label].charge:
                            logger.warning(f'Isomorphism check cannot be done for charged species {label}')
                        conformer_xyz = xyzs[0]
                    else:
                        # troubleshoot when all conformers of a species failed isomorphic test
                        logger.warning(f'Isomorphism check for all conformers of species {label} failed at '
                                       f'{self.conformer_level.simple()}. '
                                       f'Attempting to troubleshoot using a different level.')
                        self.output[label]['conformers'] += \
                            f'Error: No conformer was found to be isomorphic with the 2D graph representation at ' \
                            f'{self.conformer_level.simple()}; '
                        self.troubleshoot_conformer_isomorphism(label=label)
            else:
                logger.warning(f'Could not run isomorphism check for species {label} due to missing 2D graph '
                               f'representation. Using the most stable conformer for further geometry optimization.')
                conformer_xyz = xyzs[0]
            if conformer_xyz is not None:
                self.species_dict[label].initial_xyz = conformer_xyz
                self.species_dict[label].most_stable_conformer = xyzs_in_original_order.index(conformer_xyz)
                logger.info(f'Conformer number {xyzs_in_original_order.index(conformer_xyz)} for species {label} is '
                            f'used for geometry optimization.')
                self.output[label]['job_types']['conformers'] = True

    def determine_most_likely_ts_conformer(self, label):
        """
        Determine the most likely TS conformer.
        Save the resulting xyz as `initial_xyz`.

        Args:
            label (str): The TS species label.
        """
        if not self.species_dict[label].is_ts:
            raise SchedulerError('determine_most_likely_ts_conformer() method only processes transition state guesses.')
        for tsg in self.species_dict[label].ts_guesses:
            if tsg.success:
                self.species_dict[label].successful_methods.append(tsg.method)
            else:
                self.species_dict[label].unsuccessful_methods.append(tsg.method)
        message = f'\nAll TS guesses for {label} terminated.'
        if self.species_dict[label].successful_methods and not self.species_dict[label].unsuccessful_methods:
            message += f'\n All methods were successful: {self.species_dict[label].successful_methods}'
        elif self.species_dict[label].successful_methods:
            message += f' Successful methods: {self.species_dict[label].successful_methods}'
        elif self.species_dict[label].yml_path is not None and self.species_dict[label].final_xyz is not None:
            message += ' Geometry parsed from YAML file.'
        else:
            message += ' No method has converged!'
            logger.error(f'No TS methods for {label} have converged!')
        if self.species_dict[label].unsuccessful_methods:
            message += f' Unsuccessful methods: {self.species_dict[label].unsuccessful_methods}'
        logger.info(message)
        logger.info('\n')

        if all(tsg.energy is None for tsg in self.species_dict[label].ts_guesses):
            logger.error(f'No guess converged for TS {label}!')
        else:
            # currently we take the most stable guess. We'll need to implement additional checks here:
            # - normal displacement mode of the imaginary frequency
            # - IRC isomorphism checks
            rxn_txt = '' if self.species_dict[label].rxn_label is None \
                else f' of reaction {self.species_dict[label].rxn_label}'
            logger.info(f'\n\nGeometry *guesses* of successful TS guesses for {label}{rxn_txt}:')
            e_min = extermum_list([tsg.energy for tsg in self.species_dict[label].ts_guesses], return_min=True)
            i_min = None
            for tsg in self.species_dict[label].ts_guesses:
                if tsg.energy is not None and tsg.energy == e_min:
                    i_min = tsg.index
            for tsg in self.species_dict[label].ts_guesses:
                if tsg.index == i_min:
                    self.species_dict[label].chosen_ts = i_min  # change this if selecting a better TS later
                    self.species_dict[label].chosen_ts_method = tsg.method  # change if selecting a better TS later
                    self.species_dict[label].initial_xyz = tsg.initial_xyz
                if tsg.success and tsg.energy is not None:  # guess method and ts_level opt were both successful
                    tsg.energy -= e_min
                    logger.info(f'TS guess {tsg.index} for {label}. Method: {tsg.method}, relative energy: '
                                f'{tsg.energy:.2f} kJ/mol, guess execution time: {tsg.execution_time}')
                    # for TSs, only use `draw_3d()`, not `show_sticks()` which gets connectivity wrong:
                    plotter.draw_structure(xyz=tsg.initial_xyz, method='draw_3d')
            if self.species_dict[label].chosen_ts is None:
                raise SpeciesError(f'Could not pair most stable conformer {i_min} of {label} to a respective '
                                   f'TS guess')
            plotter.save_conformers_file(project_directory=self.project_directory,
                                         label=label,
                                         xyzs=[tsg.opt_xyz for tsg in self.species_dict[label].ts_guesses],
                                         level_of_theory=self.ts_guess_level,
                                         multiplicity=self.species_dict[label].multiplicity,
                                         charge=self.species_dict[label].charge,
                                         is_ts=True,
                                         energies=[tsg.energy for tsg in self.species_dict[label].ts_guesses],
                                         ts_methods=[tsg.method for tsg in self.species_dict[label].ts_guesses],
                                         )

    def parse_composite_geo(self, label, job):
        """
        Check that a 'composite' job converged successfully, and parse the geometry into `final_xyz`.
        Also checks (QA) that no imaginary frequencies were assigned for stable species,
        and that exactly one imaginary frequency was assigned for a TS.
        Returns ``True`` if the job converged successfully, ``False`` otherwise and troubleshoots.

        Args:
            label (str): The species label.
            job (Job): The composite job object.
        """
        logger.debug(f'parsing composite geo for {job.job_name}')
        freq_ok = False
        if job.job_status[1]['status'] == 'done':
            self.species_dict[label].final_xyz = parser.parse_xyz_from_file(path=job.local_path_to_output_file)
            self.output[label]['job_types']['composite'] = True
            self.output[label]['job_types']['opt'] = True
            self.output[label]['job_types']['sp'] = True
            if self.job_types['fine']:
                self.output[label]['job_types']['fine'] = True  # all composite jobs are fine if fine was asked for
            self.output[label]['paths']['composite'] = os.path.join(job.local_path, 'output.out')
            self.species_dict[label].opt_level = self.composite_method.simple()
            rxn_str = ''
            if self.species_dict[label].is_ts:
                rxn_str = f' of reaction {self.species_dict[label].rxn_label}'
            logger.info(f'\nOptimized geometry for {label}{rxn_str} at {job.level.simple()}:\n'
                        f'{xyz_to_str(xyz_dict=self.species_dict[label].final_xyz)}\n')
            plotter.save_geo(species=self.species_dict[label], project_directory=self.project_directory)
            if not job.is_ts:
                plotter.draw_structure(species=self.species_dict[label],
                                       project_directory=self.project_directory)
            else:
                # for TSs, only use `draw_3d()`, not `show_sticks()` which gets connectivity wrong:
                plotter.draw_structure(species=self.species_dict[label],
                                       project_directory=self.project_directory,
                                       method='draw_3d')
            frequencies = parser.parse_frequencies(job.local_path_to_output_file, job.software)
            freq_ok = self.check_negative_freq(label=label, job=job, vibfreqs=frequencies)
            if freq_ok:
                # Update restart dictionary and save a restart file:
                self.save_restart_dict()
                success = True  # run freq / scan jobs on this optimized geometry
                if not self.species_dict[label].is_ts:
                    is_isomorphic = self.species_dict[label].check_xyz_isomorphism(
                        allow_nonisomorphic_2d=self.allow_nonisomorphic_2d)
                    if is_isomorphic:
                        self.output[label]['isomorphism'] += 'composite passed isomorphism check; '
                    else:
                        self.output[label]['isomorphism'] += 'composite did not pass isomorphism check; '
                    success &= is_isomorphic
                return success
            elif not self.species_dict[label].is_ts:
                self.troubleshoot_negative_freq(label=label, job=job)
        if job.job_status[1]['status'] != 'done' or not freq_ok:
            self.troubleshoot_ess(label=label,
                                  job=job,
                                  level_of_theory=job.level)
        return False  # return ``False``, so no freq / scan jobs are initiated for this unoptimized geometry

    def parse_opt_geo(self, label, job):
        """
        Check that an 'opt' or 'optfreq' job converged successfully, and parse the geometry into `final_xyz`.
        If the job is 'optfreq', also checks (QA) that no imaginary frequencies were assigned for stable species,
        and that exactly one imaginary frequency was assigned for a TS.
        Returns ``True`` if the job (or both jobs) converged successfully, ``False`` otherwise and troubleshoots opt.

        Args:
            label (str): The species label.
            job (Job): The optimization job object.
        """
        success = False
        logger.debug(f'parsing opt geo for {job.job_name}')
        if job.job_status[1]['status'] == 'done':
            self.species_dict[label].final_xyz = parser.parse_xyz_from_file(
                path=job.local_path_to_xyz or job.local_path_to_output_file)
            if not job.fine and self.job_types['fine'] and not job.software == 'molpro':
                # Run opt again using a finer grid.
                xyz = self.species_dict[label].final_xyz
                self.species_dict[label].initial_xyz = xyz  # save for troubleshooting, since trsh goes by initial
                self.run_job(label=label,
                             xyz=xyz,
                             level_of_theory=job.level,
                             job_type='opt',
                             fine=True)
            else:
                success = True
                if 'optfreq' in job.job_name:
                    self.check_freq_job(label, job)
                self.output[label]['job_types']['opt'] = True
                if self.job_types['fine']:
                    self.output[label]['job_types']['fine'] = True
                self.species_dict[label].opt_level = self.opt_level.simple()
                plotter.save_geo(species=self.species_dict[label], project_directory=self.project_directory)
                if self.species_dict[label].is_ts:
                    rxn_str = f' of reaction {self.species_dict[label].rxn_label}'
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
                                if reactant and \
                                        not any([reactant_xyz[0] == label for reactant_xyz in tsg.reactants_xyz]):
                                    # This species is a reactant of rxn,
                                    # and its geometry wasn't saved in the TSGuess objects
                                    tsg.reactants_xyz.append((label, self.species_dict[label].final_xyz))
                                if product and \
                                        not any([product_xyz[0] == label for product_xyz in tsg.products_xyz]):
                                    # This species is a product of rxn,
                                    # and its geometry wasn't saved in the TSGuess objects
                                    tsg.products_xyz.append((label, self.species_dict[label].final_xyz))
                logger.info(f'\nOptimized geometry for {label}{rxn_str} at {job.level.simple()}:\n'
                            f'{xyz_to_str(self.species_dict[label].final_xyz)}\n')
                self.save_restart_dict()
                self.output[label]['paths']['geo'] = job.local_path_to_output_file  # will be overwritten with freq
                if not self.species_dict[label].is_ts:
                    plotter.draw_structure(species=self.species_dict[label], project_directory=self.project_directory)
                    is_isomorphic = self.species_dict[label].check_xyz_isomorphism(
                        allow_nonisomorphic_2d=self.allow_nonisomorphic_2d)
                    if is_isomorphic:
                        if 'opt passed isomorphism check' not in self.output[label]['isomorphism']:
                            self.output[label]['isomorphism'] += 'opt passed isomorphism check; '
                    else:
                        self.output[label]['isomorphism'] += 'opt did not pass isomorphism check; '
                    success &= is_isomorphic
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

        Args:
            label (str): The species label.
            job (Job): The frequency job object.
        """
        if job.job_status[1]['status'] == 'done':
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
                        self.species_dict[label].transport_data.comment += \
                            str(f'\nPolarizability calculated at the {self.freq_level.simple()} level of theory')
                    else:
                        self.species_dict[label].transport_data.comment = \
                            str(f'Polarizability calculated at the {self.freq_level.simple()} level of theory')
        else:
            self.troubleshoot_ess(label=label,
                                  job=job,
                                  level_of_theory=job.level)

    def check_negative_freq(self, label, job, vibfreqs):
        """
        A helper function for determining the number of negative frequencies. Also logs appropriate errors.
        Returns ``True`` if the number of negative frequencies is as excepted, ``False`` otherwise.

        Args:
            label (str): The species label.
            job (Job): The optimization job object.
            vibfreqs (list): The vibrational frequencies.
        """
        neg_freqs = list()
        for freq in vibfreqs:
            if freq < 0:
                neg_freqs.append(freq)
        if self.species_dict[label].is_ts and len(neg_freqs) != 1:
            logger.error(f'TS {label} has {len(neg_freqs)} imaginary frequencies ({neg_freqs}), should have exactly 1.')
            if f'{len(neg_freqs)} imaginary freqs for' not in self.output[label]['warnings']:
                self.output[label]['warnings'] += f'Warning: {len(neg_freqs)} imaginary freqs for TS ({neg_freqs}); '
            return False
        elif not self.species_dict[label].is_ts and len(neg_freqs) != 0:
            logger.error(f'Species {label} has {len(neg_freqs)} imaginary frequencies ({neg_freqs}), '
                         f'should have exactly 0.')
            if f'{len(neg_freqs)} imaginary freq for' not in self.output[label]['warnings']:
                self.output[label]['warnings'] += f'Warning: {len(neg_freqs)} imaginary freq for stable species ' \
                                                  f'({neg_freqs}); '
            return False
        else:
            if self.species_dict[label].is_ts:
                logger.info(f'TS {label} has exactly one imaginary frequency: {neg_freqs[0]}')
                self.output[label]['info'] += f'Imaginary frequency: {neg_freqs[0]}; '
            self.output[label]['job_types']['freq'] = True
            self.output[label]['paths']['geo'] = job.local_path_to_output_file
            self.output[label]['paths']['freq'] = job.local_path_to_output_file
            if not self.testing:
                # Update restart dictionary and save the yaml restart file:
                self.save_restart_dict()
            return True

    def check_sp_job(self, label, job):
        """
        Check that a single point job converged successfully.

        Args:
            label (str): The species label.
            job (Job): The single point job object.
        """
        if 'mrci' in self.sp_level.method and 'mrci' not in job.level.method:
            # This is a CCSD job ran before MRCI. Spawn MRCI
            self.run_sp_job(label)
        elif job.job_status[1]['status'] == 'done':
            self.post_sp_actions(label,
                                 sp_path=os.path.join(job.local_path, 'output.out'),
                                 level=job.level,
                                 )
            # Update restart dictionary and save the yaml restart file:
            self.save_restart_dict()
            if self.species_dict[label].number_of_atoms == 1:
                # save the geometry from the sp job for monoatomic species for which no opt/freq jobs will be spawned
                self.output[label]['paths']['geo'] = job.local_path_to_output_file
        else:
            self.troubleshoot_ess(label=label,
                                  job=job,
                                  level_of_theory=job.level)

    def post_sp_actions(self,
                        label: str,
                        sp_path: str,
                        level: Optional[Level] = None,
                        ):
        """
        Perform post-sp actions.

        Args:
            label (str): The species label.
            sp_path (str): The path to 'output.out' for the single point job.
            level (Level, optional): The level of theory used for the sp job.
        """
        original_sp_path = self.output[label]['paths']['sp'] if 'sp' in self.output[label]['paths'] else None
        self.output[label]['paths']['sp'] = sp_path
        if self.sp_level is not None and 'ccsd' in self.sp_level.method:
            self.species_dict[label].t1 = parser.parse_t1(self.output[label]['paths']['sp'])
        zpe_scale_factor = 0.99 if (self.composite_method is not None and 'cbs-qb3' in self.composite_method.method) \
            else 1.0
        self.species_dict[label].e_elect = parser.parse_e_elect(self.output[label]['paths']['sp'],
                                                                zpe_scale_factor=zpe_scale_factor)
        if self.species_dict[label].t1 is not None:
            txt = ''
            if self.species_dict[label].t1 > 0.02:
                txt += ". Looks like it requires multireference treatment, I wouldn't trust it's calculated energy!"
            elif self.species_dict[label].t1 > 0.015:
                txt += ". It might have multireference characteristic."
            logger.info(f'Species {label} has a T1 diagnostic parameter of {self.species_dict[label].t1}{txt}')
            self.output[label]['info'] += f'T1 = {self.species_dict[label].t1}; '

        if self.sp_level is not None and self.sp_level.solvation_scheme_level is not None:
            # a complex solvation correction behavior was requested for the single-point energy value
            if not self.output[label]['job_types']['sp']:
                # this is the first "original" sp job, spawn two more at the sp_level.solvation_scheme_level level,
                # with and without solvation corrections
                solvation_sp_level = self.sp_level.solvation_scheme_level.copy()
                solvation_sp_level.solvation_method = self.sp_level.solvation_method
                solvation_sp_level.solvent = self.sp_level.solvent
                self.run_sp_job(label=label, level=solvation_sp_level)
                self.run_sp_job(label=label, level=self.sp_level.solvation_scheme_level)
            else:
                # this is one of the additional sp jobs spawned by the above previously
                if level is not None and level.solvation_method is not None:
                    self.output[label]['paths']['sp_sol'] = sp_path
                else:
                    self.output[label]['paths']['sp_no_sol'] = sp_path
                self.output[label]['paths']['sp'] = original_sp_path  # restore the original path

        # set *at the end* to differentiate between sp jobs when using complex solvation corrections
        self.output[label]['job_types']['sp'] = True

    def check_irc_job(self, label, job):
        """
        Check an IRC job and perform post-job tasks.

        Todo:
            Need to check isomorphism

        Args:
            label (str): The species label.
            job (Job): The single point job object.
        """
        self.output[label]['paths']['irc'].append(os.path.join(job.local_path, 'output.out'))
        if len(self.output[label]['paths']['irc']) == 2:
            self.output[label]['job_types']['irc'] = True
            plotter.save_irc_traj_animation(irc_f_path=self.output[label]['paths']['irc'][0],
                                            irc_r_path=self.output[label]['paths']['irc'][1],
                                            out_path=os.path.join(self.project_directory, 'output',
                                                                  'rxns', label, 'irc_traj.gjf'))

    def check_scan_job(self, label: str, job: Job) -> None:
        """
        Check that a rotor scan job converged successfully. Also checks (QA) whether the scan is relatively "smooth",
        and whether the optimized geometry indeed represents the minimum energy conformer.
        Recommends whether or not to use this rotor using the 'successful_rotors' and 'unsuccessful_rotors' attributes.
        rotors_dict structure (attribute of ARCSpecies)::

            rotors_dict: {1: {'pivots': ``list``,
                              'top': ``list``,
                              'scan': ``list``,
                              'number_of_running_jobs': ``int``,
                              'success': ``Optional[bool]``,
                              'invalidation_reason': ``str``,
                              'times_dihedral_set': ``int``,
                              'scan_path': <path to scan output file>,
                              'max_e': ``float``,  # in kJ/mol,
                              'symmetry': ``int``,
                              'trsh_methods': ``list``,
                              'dimensions': ``int``,
                              'original_dihedrals': ``list``,
                              'cont_indices': ``list``,
                              'directed_scan_type': ``str``,
                              'directed_scan': ``dict``,  # keys: tuples of dihedrals as strings,
                                                          # values: dicts of energy, xyz, is_isomorphic, trsh
                             }
                          2: {}, ...
                         }

        Args:
            label (str): The species label.
            job (Job): The rotor scan job object.
        """
        # If the job has not converged, troubleshoot ESS
        # Besides, according to the experience, 'Internal coordinate error' cannot be handled by
        # troubleshoot_ess() for scan jobs. It is usually related to bond or angle changes which
        # causes the internal coordinates mess up in the middway of the scan. It can be resolved
        # by conformer based scan troubleshooting method and its energies are readable.
        if job.job_status[1]['status'] != 'done' \
                and job.job_status[1]['error'] != 'Internal coordinate error':
            self.troubleshoot_ess(label=label,
                                  job=job,
                                  level_of_theory=job.level)
            return None
        # Otherwise, check the scan job quality
        invalidate, actions, energies = False, list(), list()
        for i in range(self.species_dict[label].number_of_rotors):
            if self.species_dict[label].rotors_dict[i]['pivots'] == job.pivots:
                # Read energy profile (in kJ/mol), it may be used in the troubleshooting
                energies, angles = parser.parse_1d_scan_energies(path=job.local_path_to_output_file)
                self.species_dict[label].rotors_dict[i]['original_dihedrals'] = \
                    [calculate_dihedral_angle(coords=job.xyz, torsion=job.scan, index=1, units='degs')]
                if energies is None:
                    invalidate = True
                    invalidation_reason = 'Could not read energies'
                    message = f'Energies from rotor scan of {label} between pivots {job.pivots} could not ' \
                              f'be read. Invalidating rotor.'
                    logger.error(message)
                    break
                trajectory = parser.parse_1d_scan_coords(path=job.local_path_to_output_file) \
                    if self.species_dict[label].is_ts else None
                invalidate, invalidation_reason, message, actions = scan_quality_check(
                    label=label,
                    pivots=job.pivots,
                    energies=energies,
                    scan_res=job.scan_res,
                    used_methods=self.species_dict[label].rotors_dict[i]['trsh_methods'],
                    log_file=job.local_path_to_output_file,
                    species=self.species_dict[label],
                    preserve_params=self.species_dict[label].preserve_param_in_scan,
                    trajectory=trajectory,
                    original_xyz=self.species_dict[label].final_xyz,
                )

                if len(actions):
                    # the rotor scan is problematic, troubleshooting is required
                    logger.info(f'Trying to troubleshoot rotor {job.pivots} of {label} ...')
                    # Try to troubleshoot the rotor. sometimes, troubleshooting cannot yield solutions
                    # actions from scan_quality_check() is not the actual actions applied,
                    # they will be post-processed by trsh_scan_job. If troubleshooting fails,
                    # The actual actions will be an empty list, indicating invalid rotor.
                    trsh_success, actions = self.troubleshoot_scan_job(job=job, methods=actions)
                    if not trsh_success:
                        # Detailed reasons are logged in the troubleshoot_scan_job()
                        invalidation_reason += ' But unable to propose troubleshooting methods.'
                    else:
                        # record actions, only if the method is valid
                        self.species_dict[label].rotors_dict[i]['trsh_methods'].append(actions)

                if not invalidate:
                    # the rotor scan is good, calculate the symmetry number
                    self.species_dict[label].rotors_dict[i]['success'] = True
                    self.species_dict[label].rotors_dict[i]['symmetry'] = determine_rotor_symmetry(
                        label=label, pivots=self.species_dict[label].rotors_dict[i]['pivots'],
                        rotor_path=job.local_path_to_output_file)[0]
                    logger.info('Rotor scan {scan} between pivots {pivots} for {label} has symmetry {symmetry}'.format(
                        scan=self.species_dict[label].rotors_dict[i]['scan'],
                        pivots=self.species_dict[label].rotors_dict[i]['pivots'],
                        label=label, symmetry=self.species_dict[label].rotors_dict[i]['symmetry']))
                break
        else:
            raise SchedulerError(f'Could not match rotor with pivots {job.pivots} in species {label}')

        # This is a bad rotor scan
        if invalidate:
            self.species_dict[label].rotors_dict[i]['success'] = None if len(actions) else False

        # Better to save the path and invalidation reason for debugging and tracking the file
        # if ``success`` is None, it means that the job is being troubleshooted
        self.species_dict[label].rotors_dict[i]['scan_path'] = job.local_path_to_output_file
        self.species_dict[label].rotors_dict[i]['invalidation_reason'] += invalidation_reason

        # If energies were obtained, draw the scan curve
        if energies is not None and len(energies):
            folder_name = 'rxns' if job.is_ts else 'Species'
            rotor_path = os.path.join(self.project_directory, 'output', folder_name, job.species_name, 'rotors')
            plotter.plot_1d_rotor_scan(angles=angles,
                                       energies=energies,
                                       path=rotor_path,
                                       scan=job.scan,
                                       comment=message,
                                       label=label,
                                       original_dihedral=self.species_dict[label].rotors_dict[i]['original_dihedrals'],
                                       )

        # Save the restart dictionary
        self.save_restart_dict()

    def check_directed_scan(self, label, pivots, scan, energies):
        """
        Checks (QA) whether the directed scan is relatively "smooth",
        and whether the optimized geometry indeed represents the minimum energy conformer.
        Recommends whether or not to use this rotor using the 'successful_rotors' and 'unsuccessful_rotors' attributes.
        This method differs from check_directed_scan_job(), since here we consider the entire scan.

        Args:
            label (str): The species label.
            pivots (list): The rotor pivots.
            scan (list): The four atoms defining the dihedral.
            energies (list): The rotor scan energies in kJ/mol.

        Todo:
            - Not used!!
            - adjust to ND, merge with check_directed_scan_job (this one isn't being called)
        """
        # If the job has not converged, troubleshoot
        invalidate, invalidation_reason, message, actions = scan_quality_check(label=label,
                                                                               pivots=pivots,
                                                                               energies=energies)
        if actions:
            # the rotor scan is problematic, troubleshooting is required
            if 'change conformer' in actions:
                # a lower conformation was found
                deg_increment = actions[1]
                self.species_dict[label].set_dihedral(scan=scan, deg_increment=deg_increment)
                is_isomorphic = self.species_dict[label].check_xyz_isomorphism(
                    allow_nonisomorphic_2d=self.allow_nonisomorphic_2d,
                    xyz=self.species_dict[label].initial_xyz)
                if is_isomorphic:
                    self.delete_all_species_jobs(label)
                    # Remove all completed rotor calculation information
                    for rotor_dict in self.species_dict[label].rotors_dict.values():
                        # don't initialize all parameters, e.g., `times_dihedral_set` needs to remain as is
                        rotor_dict['scan_path'] = ''
                        rotor_dict['invalidation_reason'] = ''
                        rotor_dict['success'] = None
                        rotor_dict.pop('symmetry', None)
                    # re-run opt (or composite) on the new initial_xyz with the desired dihedral
                    if not self.composite_method:
                        self.run_opt_job(label, fine=self.fine_only)
                    else:
                        self.run_composite_job(label)
                else:
                    # The conformer is wrong, and changing the dihedral resulted in a non-isomorphic species.
                    self.output[label]['errors'] += f'A lower conformer was found for {label} via a torsion mode, ' \
                                                    f'but it is not isomorphic with the 2D graph representation ' \
                                                    f'{self.species_dict[label].mol.copy(deep=True).to_smiles()}. ' \
                                                    f'Not calculating this species.'
                    self.output[label]['conformers'] += 'Unconverged'
                    self.output[label]['convergence'] = False
            else:
                logger.error(f'Directed scan for species {label} for pivots {pivots} failed with: '
                             f'{invalidation_reason}. Currently rotor troubleshooting methods do not apply for '
                             f'directed scans. Not troubleshooting rotor.')
                for rotor_dict in self.species_dict[label].rotors_dict.values():
                    if rotor_dict['pivots'] == pivots:
                        rotor_dict['scan_path'] = ''
                        rotor_dict['invalidation_reason'] = invalidation_reason
                        rotor_dict['success'] = False

        else:
            # the rotor scan is good, calculate the symmetry number
            for rotor_dict in self.species_dict[label].rotors_dict.values():
                if rotor_dict['pivots'] == pivots:
                    if not invalidate:
                        rotor_dict['success'] = True
                        rotor_dict['symmetry'] = determine_rotor_symmetry(label=label,
                                                                          pivots=pivots,
                                                                          energies=energies)[0]
                        logger.info(f'Rotor scan {scan} between pivots {pivots} for {label} has symmetry '
                                    f'{rotor_dict["symmetry"]}')
                    else:
                        rotor_dict['success'] = False

        # Save the restart dictionary
        self.save_restart_dict()

    def check_directed_scan_job(self, label, job):
        """
        Check that a directed scan job for a specific dihedral angle converged successfully, otherwise troubleshoot.

        rotors_dict structure (attribute of ARCSpecies)::

            rotors_dict: {1: {'pivots': ``list``,
                              'top': ``list``,
                              'scan': ``list``,
                              'number_of_running_jobs': ``int``,
                              'success': ``bool``,
                              'invalidation_reason': ``str``,
                              'times_dihedral_set': ``int``,
                              'scan_path': <path to scan output file>,
                              'max_e': ``float``,  # in kJ/mol,
                              'symmetry': ``int``,
                              'dimensions': ``int``,
                              'original_dihedrals': ``list``,
                              'cont_indices': ``list``,
                              'directed_scan_type': ``str``,
                              'directed_scan': ``dict``,  # keys: tuples of dihedrals as strings,
                                                          # values: dicts of energy, xyz, is_isomorphic, trsh
                             }
                          2: {}, ...
                         }

        Args:
            label (str): The species label.
            job (Job): The rotor scan job object.
        """
        if job.job_status[1]['status'] == 'done':
            xyz = parser.parse_geometry(path=job.local_path_to_output_file)
            is_isomorphic = self.species_dict[label].check_xyz_isomorphism(xyz=xyz, verbose=False)
            for rotor_dict in self.species_dict[label].rotors_dict.values():
                if rotor_dict['pivots'] == job.pivots:
                    key = tuple(f'{dihedral:.2f}' for dihedral in job.directed_dihedrals)
                    rotor_dict['directed_scan'][key] = {'energy': parser.parse_e_elect(
                        path=job.local_path_to_output_file),
                        'xyz': xyz,
                        'is_isomorphic': is_isomorphic,
                        'trsh': job.ess_trsh_methods,
                    }
        else:
            self.troubleshoot_ess(label=label,
                                  job=job,
                                  level_of_theory=self.scan_level)

    def check_all_done(self, label):
        """
        Check that we have all required data for the species/TS.

        Args:
            label (str): The species label.
        """
        all_converged = True
        for job_type, spawn_job_type in self.job_types.items():
            if spawn_job_type and not self.output[label]['job_types'][job_type] \
                    and not ((self.species_dict[label].is_ts and job_type in ['scan', 'conformers'])
                             or (self.species_dict[label].number_of_atoms == 1
                                 and job_type in ['conformers', 'opt', 'fine', 'freq', 'rotors', 'bde'])
                             or job_type == 'bde' and self.species_dict[label].bdes is None
                             or job_type == 'conformers'):
                logger.debug(f'Species {label} did not converge')
                all_converged = False
                break
        if all_converged:
            self.output[label]['convergence'] = True
            if self.species_dict[label].is_ts:
                self.species_dict[label].make_ts_report()
                logger.info(self.species_dict[label].ts_report + '\n')
            zero_delta = datetime.timedelta(0)
            conf_time = extermum_list([job.run_time for job in self.job_dict[label]['conformers'].values()],
                                      return_min=False) \
                if 'conformers' in self.job_dict[label] else zero_delta
            opt_time = sum_time_delta([job.run_time for job in self.job_dict[label]['opt'].values()]) \
                if 'opt' in self.job_dict[label] else zero_delta
            comp_time = sum_time_delta([job.run_time for job in self.job_dict[label]['composite'].values()]) \
                if 'composite' in self.job_dict[label] else zero_delta
            other_time = extermum_list([sum_time_delta([job.run_time for job in job_dictionary.values()])
                                        for job_type, job_dictionary in self.job_dict[label].items()
                                        if job_type not in ['conformers', 'opt', 'composite']], return_min=False) \
                if any([job_type not in ['conformers', 'opt', 'composite']
                        for job_type in self.job_dict[label].keys()]) else zero_delta
            self.species_dict[label].run_time = self.species_dict[label].run_time \
                                                or (conf_time or zero_delta) + (opt_time or zero_delta) \
                                                + (comp_time or zero_delta) + (other_time or zero_delta)
            logger.info(f'\nAll jobs for species {label} successfully converged. '
                        f'Run time: {self.species_dict[label].run_time}')
        else:
            job_type_status = {key: val for key, val in self.output[label]['job_types'].items()
                               if key in self.job_types and self.job_types[key]}
            logger.error(f'Species {label} did not converge. Job type status is: {job_type_status}')
        # Update restart dictionary and save the yaml restart file:
        self.save_restart_dict()

    def get_servers_jobs_ids(self):
        """
        Check status on all active servers, return a list of relevant running job IDs
        """
        self.servers_jobs_ids = list()
        for server in self.servers:
            if server != 'local':
                with SSHClient(server) as ssh:
                    self.servers_jobs_ids.extend(ssh.check_running_jobs_ids())
            else:
                self.servers_jobs_ids.extend(check_running_jobs_ids())

    def troubleshoot_negative_freq(self, label, job):
        """
        Troubleshooting cases where non-TS species have negative frequencies.
        Run newly generated conformers.

        Args:
            label (str): The species label.
            job (Job): The frequency job object.
        """
        current_neg_freqs_trshed, confs, output_errors, output_warnings = trsh_negative_freq(
            label=label, log_file=job.local_path_to_output_file,
            neg_freqs_trshed=self.species_dict[label].neg_freqs_trshed, job_types=self.job_types)
        self.species_dict[label].neg_freqs_trshed.extend(current_neg_freqs_trshed)
        for output_error in output_errors:
            self.output[label]['errors'] += output_error
            if 'Invalidating species' in output_error:
                logger.info(f'Deleting all currently running jobs for species {label}...')
                self.delete_all_species_jobs(label)
                self.output[label]['convergence'] = False
        for output_warning in output_warnings:
            self.output[label]['warnings'] += output_warning
        if len(confs):
            logger.info(f'Deleting all currently running jobs for species {label} before troubleshooting for '
                        f'negative frequency...')
            self.delete_all_species_jobs(label)
            self.species_dict[label].conformers = confs
            self.species_dict[label].conformer_energies = [None] * len(confs)
            self.job_dict[label]['conformers'] = dict()  # initialize the conformer job dictionary
            for i, xyz in enumerate(self.species_dict[label].conformers):
                self.run_job(label=label, xyz=xyz, level_of_theory=self.conformer_level, job_type='conformer',
                             conformer=i)

    def troubleshoot_scan_job(self,
                              job: Job,
                              methods: Optional[dict] = None,
                              ) -> Tuple[bool, dict]:
        """
        Troubleshooting rotor scans
        Using the following methods:
        1. freeze: freezing specific internal coordinates or all torsions other than the scan's pivots
        2. inc_res: increasing the scan resolution.
        3. change conformer: changing to a conformer with a lower energy

        Args:
            job (Job): The scan Job object.
            methods (dict): The troubleshooting method/s to try::

                {'freeze': <a list of problematic internal coordinates>,
                 'inc_res': ``None``,
                 'change conformer': <a xyz dict>}

        Returns: Tuple[bool, dict]:
            - ``True`` if the troubleshooting is valid.
            - The actions are actual applied in the troubleshooting.
        """
        label = job.species_name
        trsh_success = False
        actual_actions = dict()  # If troubleshooting fails, there will be no action
        # Read used troubleshooting methods
        for rotor in self.species_dict[label].rotors_dict.values():
            if rotor['scan'] == job.scan:
                used_trsh_methods = rotor['trsh_methods']
                break

        # A lower conformation was found
        if 'change conformer' in methods:
            # We will delete all of the jobs no matter we can successfully change to the conformer.
            # If succeed, we have to cancel jobs to avoid conflicts
            # If not succeed, we are in a situation that we find a lower conformer, but either
            # this is a incorrect conformer or we have applied this troubleshooting before, but it
            # didn't yield a good result.
            self.delete_all_species_jobs(label)

            new_xyz = methods['change conformer']
            # Check if the same conformer is used in previous troubleshooting
            for used_trsh_method in used_trsh_methods:
                if 'change conformer' in used_trsh_method \
                        and compare_confs(new_xyz, used_trsh_method['change conformer']):
                    # Find we have used this conformer for troubleshooting. Invalid the troubleshooting.
                    logger.error(f'The change conformer method for {label} is invalid. '
                                 f'ARC will not change to the same conformer twice.')
                    break
            else:
                # If the conformer is not used, check isomorphism
                is_isomorphic = self.species_dict[label].check_xyz_isomorphism(
                    allow_nonisomorphic_2d=self.allow_nonisomorphic_2d,
                    xyz=new_xyz)
                if is_isomorphic:
                    self.species_dict[label].final_xyz = new_xyz
                    # Remove all completed rotor calculation information
                    for rotor in self.species_dict[label].rotors_dict.values():
                        # don't initialize all parameters, e.g., `times_dihedral_set` needs to remain as is
                        rotor['scan_path'] = ''
                        rotor['invalidation_reason'] = ''
                        rotor['success'] = None
                        rotor.pop('symmetry', None)
                        if rotor['scan'] == job.scan:
                            rotor['times_dihedral_set'] += 1
                        # We can save the change conformer trsh info, but other trsh methods like
                        # freezing or increasing scan resolution can be cleaned, otherwise, they may
                        # not be troubleshot
                        rotor['trsh_methods'] = [trsh_method for trsh_method in rotor['trsh_methods']
                                                 if 'change conformer' in trsh_method]
                    # re-run opt (or composite) on the new initial_xyz with the desired dihedral
                    if not self.composite_method:
                        self.run_opt_job(label)
                    else:
                        self.run_composite_job(label)
                    trsh_success = True
                    actual_actions = methods
                    return trsh_success, actual_actions

            # The conformer is wrong, or we are in a loop changing to the same conformers again
            self.output[label]['errors'] += \
                f'A lower conformer was found for {label} via a torsion mode, ' \
                f'but it is not isomorphic with the 2D graph representation ' \
                f'{self.species_dict[label].mol.copy(deep=True).to_smiles()}. ' \
                f'Not calculating this species.'
            self.output[label]['conformers'] += 'Unconverged'
            self.output[label]['convergence'] = False
        else:
            # Freezing or increasing scan resolution
            scan_list = [rotor_dict['scan'] for rotor_dict in
                         self.species_dict[label].rotors_dict.values()]
            try:
                scan_trsh, scan_res = trsh_scan_job(label=label,
                                                    scan_res=job.scan_res,
                                                    scan=job.scan,
                                                    scan_list=scan_list,
                                                    methods=methods,
                                                    log_file=job.local_path_to_output_file,
                                                    )
            except TrshError as e:
                logger.error(f'Troubleshooting of the rotor scan on {job.scan} for '
                             f'{label} failed. Got: {e}\nJob info:\n{job}')
            except InputError as e:
                logger.debug(f'Got invalid input for trsh_scan_job: {e}\nJob info:\n{job}')
            else:
                if scan_trsh or job.scan_res != scan_res \
                        and {'scan_trsh': scan_trsh, 'scan_res': scan_res} not in used_trsh_methods:
                    # Valid troubleshooting method for freezing or increasing resolution
                    trsh_success = True
                    actual_actions = {'scan_trsh': scan_trsh, 'scan_res': scan_res}
                    self.run_job(label=label,
                                 xyz=job.xyz,
                                 level_of_theory=job.level,
                                 job_type='scan',
                                 scan=job.scan,
                                 pivots=job.pivots,
                                 scan_trsh=scan_trsh,
                                 scan_res=scan_res)
        return trsh_success, actual_actions

    def troubleshoot_opt_jobs(self, label):
        """
        We're troubleshooting for opt jobs.
        First check for server status and troubleshoot if needed. Then check for ESS status and troubleshoot
        if needed. Finally, check whether or not the last job had fine=True, add if it didn't run with fine.

        Args:
            label (str): The species label.
        """
        previous_job_num, latest_job_num = -1, -1
        job = None
        for job_name in self.job_dict[label]['opt'].keys():  # get latest Job object for the species / TS
            job_name_int = int(job_name[5:])
            if job_name_int > latest_job_num:
                previous_job_num = latest_job_num
                latest_job_num = job_name_int
                job = self.job_dict[label]['opt'][job_name]
        if job.job_status[0] == 'done':
            if job.job_status[1]['status'] == 'done':
                if job.fine:
                    # run_opt_job should not be called if all looks good...
                    logger.error(f'opt job for {label} seems right, yet "run_opt_job" was called.')
                    raise SchedulerError(f'opt job for {label} seems right, yet "run_opt_job" was called.')
                else:
                    # Run opt again using a finer grid.
                    self.parse_opt_geo(label=label, job=job)
                    xyz = self.species_dict[label].final_xyz
                    self.species_dict[label].initial_xyz = xyz  # save for troubleshooting, since trsh goes by initial
                    self.run_job(label=label, xyz=xyz, level_of_theory=self.opt_level, job_type='opt', fine=True)
            else:
                trsh_opt = True
                # job passed on the server, but failed in ESS calculation
                if previous_job_num >= 0 and job.fine:
                    previous_job = self.job_dict[label]['opt']['opt_a' + str(previous_job_num)]
                    if not previous_job.fine and previous_job.job_status[0] == 'done' \
                            and previous_job.job_status[1]['status'] == 'done':
                        # The present job with a fine grid failed in the ESS calculation.
                        # A *previous* job without a fine grid terminated successfully on the server and ESS.
                        # So use the xyz determined w/o the fine grid, and output an error message to alert users.
                        logger.error(f'Optimization job for {label} with a fine grid terminated successfully '
                                     f'on the server, but crashed during calculation. NOT running with fine '
                                     f'grid again.')
                        self.parse_opt_geo(label=label, job=previous_job)
                        trsh_opt = False
                if trsh_opt:
                    self.troubleshoot_ess(label=label,
                                          job=job,
                                          level_of_theory=self.opt_level)
        else:
            job.troubleshoot_server()

    def troubleshoot_ess(self,
                         label: str,
                         job: Job,
                         level_of_theory: Union[Level, dict, str],
                         conformer: int = -1):
        """
        Troubleshoot issues related to the electronic structure software, such as conversion.

        Args:
            label (str): The species label.
            job (Job): The job object to troubleshoot.
            level_of_theory (Level, dict, str): The level of theory to use.
            conformer (int, optional): The conformer index.
        """
        level_of_theory = Level(repr=level_of_theory)
        logger.info('\n')
        logger.warning(f'Troubleshooting {label} job {job.job_name} which failed with status: '
                       f'"{job.job_status[1]["status"]},"\n'
                       f'with keywords: {job.job_status[1]["keywords"]}\n'
                       f'in {job.software}.\n'
                       f'The error "{job.job_status[1]["error"]}" was derived from the following line in the log '
                       f'file:\n"{job.job_status[1]["line"]}".')
        if conformer != -1:
            xyz = self.species_dict[label].conformers[conformer]
        else:
            xyz = self.species_dict[label].final_xyz or self.species_dict[label].initial_xyz

        if 'Unknown' in job.job_status[1]['keywords'] and 'change_node' not in job.ess_trsh_methods:
            job.ess_trsh_methods.append('change_node')
            job.troubleshoot_server()
            if job.job_name not in self.running_jobs[label]:
                self.running_jobs[label].append(job.job_name)  # mark as a running job
        if job.software == 'gaussian':
            if self.species_dict[label].checkfile is None:
                self.species_dict[label].checkfile = job.checkfile
        # determine if the species is a hydrogen (or its isotope) atom
        is_h = self.species_dict[label].number_of_atoms == 1 and \
               self.species_dict[label].mol.atoms[0].element.symbol in ['H', 'D', 'T']
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, \
            software, job_type, fine, trsh_keyword, memory, shift, cpu_cores, dont_rerun = \
            trsh_ess_job(label=label,
                         level_of_theory=level_of_theory,
                         server=job.server,
                         job_status=job.job_status[1],
                         is_h=is_h,
                         job_type=job.job_type,
                         num_heavy_atoms=self.species_dict[label].number_of_heavy_atoms,
                         software=job.software,
                         fine=job.fine,
                         memory_gb=job.total_job_memory_gb,
                         cpu_cores=job.cpu_cores,
                         ess_trsh_methods=job.ess_trsh_methods,
                         available_ess=list(self.ess_settings.keys()),
                         )
        for output_error in output_errors:
            self.output[label]['errors'] += output_error
        if remove_checkfile:
            self.species_dict[label].checkfile = None
        job.ess_trsh_methods = ess_trsh_methods

        if not dont_rerun:
            self.run_job(label=label,
                         xyz=xyz,
                         level_of_theory=level_of_theory,
                         software=software,
                         memory=memory,
                         job_type=job_type,
                         fine=fine,
                         ess_trsh_methods=ess_trsh_methods,
                         trsh=trsh_keyword,
                         conformer=conformer,
                         scan=job.scan,
                         pivots=job.pivots,
                         scan_res=job.scan_res,
                         shift=shift,
                         directed_dihedrals=job.directed_dihedrals,
                         directed_scans=job.directed_scans,
                         directed_scan_type=job.directed_scan_type,
                         rotor_index=job.rotor_index,
                         cpu_cores=cpu_cores,
                         )
        self.save_restart_dict()

    def troubleshoot_conformer_isomorphism(self, label: str):
        """
        Troubleshoot conformer optimization for a species that failed isomorphic test in
        ``determine_most_stable_conformer``.

        Args:
            label (str): The species label.
        """
        if self.species_dict[label].is_ts:
            raise SchedulerError('The troubleshoot_conformer_isomorphism() method does not yet deal with TSs.')

        num_of_conformers = len(self.species_dict[label].conformers)
        if not num_of_conformers:
            raise SchedulerError('The troubleshoot_conformer_isomorphism() method got zero conformers.')

        # use the first conformer of a species to determine applicable troubleshooting method
        job = self.job_dict[label]['conformers'][0]

        level_of_theory = trsh_conformer_isomorphism(software=job.software, ess_trsh_methods=job.ess_trsh_methods)

        if level_of_theory is None:
            logger.error(f'ARC has attempted all built-in conformer isomorphism troubleshoot methods for species '
                         f'{label}. No conformer for this species was found to be isomorphic with the 2D graph '
                         f'representation {self.species_dict[label].mol.copy(deep=True).to_smiles()}. '
                         f'NOT optimizing this species.')
            self.output[label]['conformers'] += 'Error: No conformer was found to be isomorphic with the 2D' \
                                                ' graph representation!; '
        else:
            logger.info(f'Troubleshooting conformer job in {job.software} using {level_of_theory} for species {label}')

            # rerun conformer job at higher level for all conformers
            for conformer in range(0, num_of_conformers):

                # initial xyz before troubleshooting
                xyz = self.species_dict[label].conformers_before_opt[conformer]

                job = self.job_dict[label]['conformers'][conformer]
                if 'Conformers: ' + level_of_theory not in job.ess_trsh_methods:
                    job.ess_trsh_methods.append('Conformers: ' + level_of_theory)

                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type='conformer', ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)

    def delete_all_species_jobs(self, label: str):
        """
        Delete all jobs of a species/TS.

        Args:
            label (str): The species label.
        """
        logger.debug(f'Deleting all jobs for species {label}')
        for job_dict in self.job_dict[label].values():
            for job_name, job in job_dict.items():
                if job_name in self.running_jobs[label]:
                    logger.info(f'Deleted job {job_name}')
                    job.delete()
        self.running_jobs[label] = list()

    def restore_running_jobs(self):
        """
        Make Job objects for jobs which were running in the previous session.
        Important for the restart feature so long jobs won't be ran twice.
        """
        jobs = self.restart_dict['running_jobs']
        if not jobs or not any([job for job in jobs.values()]):
            del self.restart_dict['running_jobs']
            self.running_jobs = dict()
            logger.debug('It seems that there are no running jobs specified in the ARC restart file. '
                         'Assuming all jobs have finished.')
        else:
            for spc_label in jobs.keys():
                if spc_label not in self.running_jobs:
                    self.running_jobs[spc_label] = list()
                for job_description in jobs[spc_label]:
                    if 'conformer' not in job_description or job_description['conformer'] < 0:
                        self.running_jobs[spc_label].append(job_description['job_name'])
                    else:
                        self.running_jobs[spc_label].append(f'conformer{job_description["conformer"]}')
                    for species in self.species_list:
                        if species.label == spc_label:
                            break
                    else:
                        raise SchedulerError(f'Could not find species {spc_label} in the restart file')
                    job = Job(**job_description)
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
                    content += f'\n{spc_label}: '
                    for job_type in self.job_dict[spc_label].keys():
                        for job_name in self.job_dict[spc_label][job_type].keys():
                            if job_type != 'conformers':
                                content += job_name + ', '
                            else:
                                content += self.job_dict[spc_label][job_type][job_name].job_name \
                                           + f' (conformer{job_name}), '
                content += '\n\n'
                logger.info(content)

    def save_restart_dict(self):
        """
        Update the restart_dict and save the restart.yml file.
        """
        if self.save_restart and self.restart_dict is not None:
            logger.debug('Creating a restart file...')
            self.restart_dict['output'] = self.output
            self.restart_dict['species'] = [spc.as_dict() for spc in self.species_dict.values()]
            self.restart_dict['running_jobs'] = dict()
            for spc in self.species_dict.values():
                if spc.label in self.running_jobs:
                    self.restart_dict['running_jobs'][spc.label] = \
                        [self.job_dict[spc.label][job_name.rsplit('_', 1)[0]][job_name].as_dict()
                         for job_name in self.running_jobs[spc.label] if 'conformer' not in job_name] \
                        + [self.job_dict[spc.label]['conformers'][int(job_name.split('mer')[1])].as_dict()
                           for job_name in self.running_jobs[spc.label] if 'conformer' in job_name]
            logger.debug(f'Dumping restart dictionary:\n{self.restart_dict}')
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

    def determine_adaptive_level(self,
                                 original_level_of_theory: Level,
                                 job_type: str,
                                 heavy_atoms: int,
                                 ) -> Level:
        """
        Determine the level of theory to be used according to the job type and number of heavy atoms.
        self.adaptive_levels is a dictionary of levels of theory for ranges of the number of heavy atoms in the
        species. Keys are tuples of (min_num_atoms, max_num_atoms), values are dictionaries with job type tuples
        as keys and levels of theory as values. 'inf' is accepted an max_num_atoms.

        Args:
            original_level_of_theory (Level): The level of theory for non-sp/opt/freq job types.
            job_type (str): The job type for which the level of theory is determined.
            heavy_atoms (int): The number of heavy atoms in the species.
        """
        for atom_range, adaptive_level in self.adaptive_levels.items():
            if atom_range[1] == 'inf' and heavy_atoms >= atom_range[0] or atom_range[0] <= heavy_atoms <= atom_range[1]:
                break
        else:
            raise SchedulerError(f'Could not determine adaptive level of theory for {heavy_atoms} heavy atoms using '
                                 f'the following adaptive levels:\n{self.adaptive_levels}')
        for job_type_tuple, level in adaptive_level.items():
            if job_type in job_type_tuple:
                return level
        # for any other job type use the original level of theory regardless of the number of heavy atoms
        return original_level_of_theory

    def initialize_output_dict(self, label: Optional[str] = None):
        """
        Initialize self.output.
        Do not initialize keys that will contain paths ('geo', 'freq', 'sp', 'composite'),
        their existence indicate the job was terminated for restarting purposes.
        If ``label`` is not ``None``, will initialize for a specific species, otherwise will initialize for all species.

        Args:
            label (str, optional): A species label.
        """
        if label is not None or not self._does_output_dict_contain_info():
            for species in self.species_list:
                if label is None or (label is not None and species.label == label):
                    if species.label not in self.output:
                        self.output[species.label] = dict()
                    if 'paths' not in self.output[species.label]:
                        self.output[species.label]['paths'] = dict()
                    path_keys = ['geo', 'freq', 'sp', 'composite']
                    for key in path_keys:
                        if key not in self.output[species.label]['paths']:
                            self.output[species.label]['paths'][key] = ''
                    if 'irc' not in self.output[species.label]['paths'] and species.is_ts:
                        self.output[species.label]['paths']['irc'] = list()
                    if 'job_types' not in self.output[species.label]:
                        self.output[species.label]['job_types'] = dict()
                    for job_type in list(set(self.job_types.keys())) + ['opt', 'freq', 'sp', 'composite', 'onedmin']:
                        if job_type in ['rotors', 'bde', 'irc']:
                            # rotors could be invalidated due to many reasons,
                            # also could be falsely identified in a species that has no torsional modes.
                            self.output[species.label]['job_types'][job_type] = True
                        else:
                            self.output[species.label]['job_types'][job_type] = False
                    keys = ['conformers', 'isomorphism', 'convergence', 'restart', 'errors', 'warnings', 'info']
                    for key in keys:
                        if key not in self.output[species.label]:
                            if key == 'convergence':
                                self.output[species.label][key] = None
                            else:
                                self.output[species.label][key] = ''

    def _does_output_dict_contain_info(self):
        """
        Determine whether self.output contains any information other than the initialized structure.

        Returns:
            bool: Whether self.output contains any information, `True` if it does.
        """
        for species_output_dict in self.output.values():
            for key0, val0 in species_output_dict.items():
                if key0 in ['paths', 'job_types']:
                    for key1, val1 in species_output_dict[key0].items():
                        if val1 and key1 not in ['rotors', 'bde']:
                            return True
                else:
                    if val0:
                        return True
        return False


def sum_time_delta(timedelta_list: List[datetime.timedelta]) -> datetime.timedelta:
    """
    A helper function for summing datetime.timedelta objects.

    Args:
        timedelta_list (list): Time delta's to sum.

    Returns:
        datetime.timedelta: The timedelta sum.
    """
    result = datetime.timedelta(0)
    for timedelta in timedelta_list:
        if timedelta is not None:
            result += timedelta
    return result
