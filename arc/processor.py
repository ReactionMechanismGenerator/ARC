"""
Processor module for computing thermodynamic properties and rate coefficients using statistical mechanics.
"""

import os
import shutil
from typing import Optional, Type

from rmgpy.data.rmg import RMGDatabase

import arc.plotter as plotter
import arc.rmgdb as rmgdb
from arc.common import get_logger
from arc.level import Level
from arc.statmech.factory import statmech_factory


logger = get_logger()


def process_arc_project(thermo_adapter: str,
                        kinetics_adapter: str,
                        project: str,
                        project_directory: str,
                        species_dict: dict,
                        reactions: list,
                        output_dict: dict,
                        bac_type: Optional[str] = None,
                        sp_level: Optional[Level] = None,
                        freq_scale_factor: float = 1.0,
                        compute_thermo: bool = True,
                        compute_rates: bool = True,
                        compute_transport: bool = False,
                        T_min: tuple = None,
                        T_max: tuple = None,
                        T_count: int = 50,
                        lib_long_desc: str = '',
                        rmg_database: Optional[RMGDatabase] = None,
                        compare_to_rmg: bool = True,
                        three_params: bool = True,
                        ) -> None:
    """
    Process an ARC project, generate thermo and rate coefficients using statistical mechanics (statmech).

    Args:
        thermo_adapter (str): The software to use for calculating thermodynamic data.
        kinetics_adapter (str): The software to use for calculating rate coefficients.
        project (str): The ARC project name.
        project_directory (str): The path to the ARC project directory.
        species_dict (dict): Keys are labels, values are ARCSpecies objects.
        reactions (list): Entries are ARCReaction objects.
        output_dict (dict): Keys are labels, values are output file paths.
                            See Scheduler for a description of this dictionary.
        bac_type (str, optional): The bond additivity correction type. 'p' for Petersson- or 'm' for Melius-type BAC.
                                  ``None`` to not use BAC.
        sp_level (Level, optional): The level of theory used for energy corrections.
        freq_scale_factor (float, optional): The harmonic frequencies scaling factor.
        compute_thermo (bool, optional): Whether to compute thermodynamic properties for the provided species.
        compute_rates (bool, optional): Whether to compute high pressure limit rate coefficients.
        compute_transport (bool, optional): Whether to compute transport properties.
        T_min (tuple, optional): The minimum temperature for kinetics computations, e.g., (500, 'K').
        T_max (tuple, optional): The maximum temperature for kinetics computations, e.g., (3000, 'K').
        T_count (int, optional): The number of temperature points between ``T_min`` and ``T_max``.
        lib_long_desc (str, optional): A multiline description of levels of theory for the resulting RMG libraries.
        rmg_database (RMGDatabase, optional): The RMG database object.
        compare_to_rmg (bool, optional): If ``True``, ARC's calculations will be compared against estimations
                                         from RMG's database.
        three_params (bool, optional): Compute rate coefficients using the modified three-parameter Arrhenius equation
                                       format (``True``, default) or classical two-parameter Arrhenius equation format
                                       (``False``).
    """
    T_min = T_min or (300, 'K')
    T_max = T_max or (3000, 'K')
    if isinstance(T_min, (int, float)):
        T_min = (T_min, 'K')
    if isinstance(T_max, (int, float)):
        T_max = (T_max, 'K')
    T_count = T_count or 50

    species_for_thermo_lib, unconverged_species = list(), list()
    rxns_for_kinetics_lib, unconverged_rxns = list(), list()
    species_for_transport_lib = list()
    bde_report = dict()

    output_directory = os.path.join(project_directory, 'output')
    libraries_path = os.path.join(output_directory, 'RMG libraries')
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    # 1. Rates
    if compute_rates:
        for reaction in reactions:
            if reaction.ts_species.ts_guesses_exhausted:
                continue
            species_converged = True
            considered_labels = list()  # Species labels considered in this reaction.
            if output_dict[reaction.ts_label]['convergence']:
                for species in reaction.r_species + reaction.p_species:
                    if species.label in considered_labels:
                        # Consider cases where the same species appears in a reaction both as a reactant
                        # and as a product (e.g., H2O that catalyzes a reaction).
                        continue
                    considered_labels.append(species.label)
                    if output_dict[species.label]['convergence']:
                        statmech_adapter = statmech_factory(statmech_adapter_label=kinetics_adapter,
                                                            output_directory=output_directory,
                                                            output_dict=output_dict,
                                                            bac_type=None,
                                                            sp_level=sp_level,
                                                            freq_scale_factor=freq_scale_factor,
                                                            species=species,
                                                            )
                        statmech_adapter.compute_thermo(kinetics_flag=True)
                    else:
                        logger.error(f'Species {species.label} did not converge, cannot compute a rate coefficient '
                                     f'for {reaction.label}')
                        unconverged_species.append(species)
                        species_converged = False
                if species_converged:
                    statmech_adapter = statmech_factory(statmech_adapter_label=kinetics_adapter,
                                                        output_directory=output_directory,
                                                        output_dict=output_dict,
                                                        bac_type=None,
                                                        sp_level=sp_level,
                                                        freq_scale_factor=freq_scale_factor,
                                                        reaction=reaction,
                                                        species_dict=species_dict,
                                                        T_min=T_min,
                                                        T_max=T_max,
                                                        T_count=T_count,
                                                        three_params=three_params,
                                                        )
                    statmech_adapter.compute_high_p_rate_coefficient()
                    if reaction.kinetics is not None:
                        rxns_for_kinetics_lib.append(reaction)
                    else:
                        unconverged_rxns.append(reaction)
            else:
                unconverged_rxns.append(reaction)
        if rxns_for_kinetics_lib:
            plotter.save_kinetics_lib(rxn_list=rxns_for_kinetics_lib,
                                      path=libraries_path,
                                      name=project,
                                      lib_long_desc=lib_long_desc)

    # 2. Thermo
    if compute_thermo:
        for species in species_dict.values():
            if (species.compute_thermo or species.e0_only) and output_dict[species.label]['convergence']:
                statmech_adapter = statmech_factory(statmech_adapter_label=thermo_adapter,
                                                    output_directory=output_directory,
                                                    output_dict=output_dict,
                                                    bac_type=bac_type,
                                                    sp_level=sp_level,
                                                    freq_scale_factor=freq_scale_factor,
                                                    species=species,
                                                    )
                statmech_adapter.compute_thermo(kinetics_flag=False, e0_only=species.e0_only)
                if species.thermo is not None:
                    species_for_thermo_lib.append(species)
                elif not species.e0_only and species not in unconverged_species:
                    unconverged_species.append(species)
            elif species.compute_thermo and not output_dict[species.label]['convergence'] \
                    and species not in unconverged_species:
                unconverged_species.append(species)
        if species_for_thermo_lib:
            plotter.save_thermo_lib(species_list=species_for_thermo_lib,
                                    path=libraries_path,
                                    name=project,
                                    lib_long_desc=lib_long_desc)

    # 3. Transport
    if compute_transport:
        for species in species_dict.values():
            if output_dict[species.label]['job_types']['onedmin'] and output_dict[species.label]['convergence']:
                pass
                # todo
        if species_for_transport_lib:
            plotter.save_transport_lib(species_list=species_for_thermo_lib,
                                       path=libraries_path,
                                       name=project,
                                       lib_long_desc=lib_long_desc)

    # 4. BDE
    for species in species_dict.values():
        # looping again to make sure all relevant Species.e0 attributes were set
        if species.bdes is not None:
            bde_report[species.label] = process_bdes(label=species.label, species_dict=species_dict)
    if bde_report:
        bde_path = os.path.join(project_directory, 'output', 'BDE_report.txt')
        plotter.log_bde_report(path=bde_path, bde_report=bde_report, spc_dict=species_dict)

    # Comparisons
    if compare_to_rmg:
        try:
            load_rmg_database(rmg_database=rmg_database, species_dict=species_dict, output_dict=output_dict)
        except Exception as e:
            logger.error(f'Could not load the RMG database! Got:\n{e}')
        else:
            compare_thermo(species_for_thermo_lib=species_for_thermo_lib,
                           rmg_database=rmg_database,
                           output_directory=output_directory)
            compare_rates(rxns_for_kinetics_lib, rmg_database,
                          output_directory=output_directory,
                          T_min=T_min,
                          T_max=T_max,
                          T_count=T_count)
            compare_transport(species_for_transport_lib, rmg_database,
                              output_directory=output_directory)

    write_unconverged_log(unconverged_species=unconverged_species,
                          unconverged_rxns=unconverged_rxns,
                          log_file_path=os.path.join(output_directory, 'unconverged_species.log'))
    clean_output_directory(project_directory)


def compare_thermo(species_for_thermo_lib: list,
                   rmg_database: Type[RMGDatabase],
                   output_directory: str,
                   ) -> None:
    """
    Compare the calculates thermo with RMG's estimations.

    Args:
        species_for_thermo_lib (list): Species for which thermochemical properties were computed.
        rmg_database (RMGDatabase, optional): The RMG database object.
        output_directory (str): The path to the project's output folder.
    """
    species_to_compare = list()  # species for which thermo was both calculated and estimated.
    for species in species_for_thermo_lib:
        try:
            species.rmg_thermo = rmg_database.thermo.get_thermo_data(species.rmg_species)
        except Exception as e:
            logger.info(f'Could not estimate thermo for species {species.label} using RMG, possibly due to a missing '
                        f'2D structure. Not including this species in the parity plots. Got:\n{e}')
        else:
            species_to_compare.append(species)
    if species_to_compare:
        plotter.draw_thermo_parity_plots(species_list=species_to_compare,
                                         path=output_directory)


def compare_rates(rxns_for_kinetics_lib: list,
                  rmg_database: Type[RMGDatabase],
                  output_directory: str,
                  T_min: tuple = None,
                  T_max: tuple = None,
                  T_count: int = 50,
                  ) -> None:
    """
    Compare the calculates thermo with RMG's estimations.

    Args:
        rxns_for_kinetics_lib (list): Reactions for which rate coefficients were computed.
        rmg_database (RMGDatabase, optional): The RMG database object.
        output_directory (str): The path to the project's output folder.
        T_min (tuple, optional): The minimum temperature for kinetics computations, e.g., (500, 'K').
        T_max (tuple, optional): The maximum temperature for kinetics computations, e.g., (3000, 'K').
        T_count (int, optional): The number of temperature points between ``T_min`` and ``T_max``.
    """
    reactions_to_compare = list()  # reactions for which a rate was both calculated and estimated.
    for reaction in rxns_for_kinetics_lib:
        try:
            reaction.rmg_reactions = rmgdb.determine_rmg_kinetics(rmgdb=rmg_database,
                                                                  reaction=reaction.rmg_reaction,
                                                                  dh_rxn298=reaction.dh_rxn298)
        except Exception as e:
            logger.info(f'Could not estimate a rate for reaction {reaction.label} using RMG. '
                        f'Not generating a comparison plot for this reaction.\nGot: {e}')
        else:
            reactions_to_compare.append(reaction)
    if reactions_to_compare:
        plotter.draw_kinetics_plots(reactions_to_compare,
                                    T_min=T_min,
                                    T_max=T_max,
                                    T_count=T_count,
                                    path=output_directory)


def compare_transport(species_for_transport_lib: list,
                      rmg_database: Type[RMGDatabase],
                      output_directory: str,
                      ) -> None:
    """
    Compare the calculates transport data with RMG's estimations.

    Args:
        species_for_transport_lib (list): Species for which thermochemical properties were computed.
        rmg_database (RMGDatabase, optional): The RMG database object.
        output_directory (str): The path to the project's output folder.
    """
    pass
    # todo


def load_rmg_database(rmg_database: Optional[Type[RMGDatabase]],
                      species_dict: dict,
                      output_dict: dict,
                      ) -> None:
    """
    Load the RMG database.

    Args:
        rmg_database (RMGDatabase, optional): The RMG database object.
        species_dict (dict): Keys are labels, values are ARCSpecies objects.
        output_dict (dict): Keys are labels, values are output file paths.
                            See Scheduler for a description of this dictionary.
    """
    load_thermo_libs, load_kinetic_libs = False, False
    if any([species.is_ts and output_dict[species.label]['convergence'] for species in species_dict.values()]):
        load_kinetic_libs = True
    if any([species.compute_thermo and output_dict[species.label]['convergence']
            for species in species_dict.values()]):
        load_thermo_libs = True
    if rmg_database is not None and (load_kinetic_libs or load_thermo_libs):
        rmgdb.load_rmg_database(rmgdb=rmg_database,
                                load_thermo_libs=load_thermo_libs,
                                load_kinetic_libs=load_kinetic_libs)


def process_bdes(label: str,
                 species_dict: dict,
                 ) -> dict:
    """
    Process bond dissociation energies for a single parent species represented by `label`.

    Args:
        label (str): The species label.
        species_dict (dict): Keys are labels, values are ARCSpecies objects.

    Returns:
        dict: The BDE report for a single species. Keys are pivots, values are energies in kJ/mol.
    """
    source = species_dict[label]
    bde_report = dict()
    if source.e0 is None:
        logger.error(f'Cannot calculate BDEs without E0 for {label}. Make sure freq and sp jobs ran successfully '
                     f'for this species.')
        return bde_report
    for bde_indices in source.bdes:
        found_a_label = False
        # Index 0 of the tuple:
        if source.mol.atoms[bde_indices[0] - 1].is_hydrogen():
            e1 = species_dict['H'].e0
        else:
            bde_label = f'{label}_BDE_{bde_indices[0]}_{bde_indices[1]}_A'
            if bde_label not in species_dict.keys():
                logger.error(f'Could not find BDE species {bde_label} for generating a BDE report for {label}. '
                             f'Not generating a BDE report for this species.')
                return dict()
            found_a_label = True
            e1 = species_dict[bde_label].e0
        # Index 1 of the tuple:
        if source.mol.atoms[bde_indices[1] - 1].is_hydrogen():
            e2 = species_dict['H'].e0
        else:
            letter = 'B' if found_a_label else 'A'
            bde_label = f'{label}_BDE_{bde_indices[0]}_{bde_indices[1]}_{letter}'
            if bde_label not in species_dict.keys():
                logger.error(f'Could not find BDE species {bde_label} for generating a BDE report for {label}. '
                             f'Not generating a BDE report for this species.')
                return dict()
            e2 = species_dict[bde_label].e0
        if e1 is not None and e2 is not None:
            bde_report[bde_indices] = e1 + e2 - source.e0  # products - reactant
        else:
            bde_report[bde_indices] = 'N/A'
            logger.error(f'Could not calculate BDE for {label} between atoms '
                         f'{bde_indices[0]} ({source.mol.atoms[bde_indices[0] - 1].element.symbol}) '
                         f'and {bde_indices[1]} ({source.mol.atoms[bde_indices[1] - 1].element.symbol})')
    return bde_report


def write_unconverged_log(unconverged_species: list,
                          unconverged_rxns: list,
                          log_file_path: str,
                          ) -> None:
    """
    Write a log file of unconverged species and reactions.

    Args:
        unconverged_species (list): List of unconverged species to report.
        unconverged_rxns (list): List of unconverged reactions to report.
        log_file_path (str): The path to the log file to write.
    """
    if unconverged_species or unconverged_rxns:
        with open(log_file_path, 'w') as f:
            if unconverged_species:
                f.write('Species:')
                for species in unconverged_species:
                    f.write(species.label)
                    if species.is_ts:
                        f.write(f' rxn: {species.rxn_label}')
                    elif species.mol is not None:
                        f.write(f' SMILES: {species.mol.copy(deep=True).to_smiles()}')
                    f.write('\n')
            if unconverged_rxns:
                f.write('Reactions:')
                for reaction in unconverged_rxns:
                    f.write(reaction.label)


def clean_output_directory(project_directory: str) -> None:
    """
    A helper function to organize the output directory.
        - remove redundant rotor.txt files (from kinetics jobs)
        - move remaining rotor files to the rotor directory
        - move the Arkane YAML file from the `species` directory to the base directory, and delete `species`

    Args:
        project_directory (str): The path to the ARC project directory.
    """
    for base_folder in ['Species', 'rxns']:
        base_path = os.path.join(project_directory, 'output', base_folder)
        dir_names = list()
        for (_, dirs, _) in os.walk(base_path):
            dir_names.extend(dirs)
            break  # don't continue to explore subdirectories
        for species_label in dir_names:
            species_path = os.path.join(base_path, species_label)
            file_names = list()
            for (_, _, files) in os.walk(species_path):
                file_names.extend(files)
                break  # don't continue to explore subdirectories
            if any(['rotor' in file_name for file_name in file_names]) \
                    and not os.path.exists(os.path.join(species_path, 'rotors')):
                os.makedirs(os.path.join(species_path, 'rotors'))
            for file_name in file_names:
                if '_rotor' in file_name:  # move to the rotor directory
                    shutil.move(src=os.path.join(species_path, file_name),
                                dst=os.path.join(species_path, 'rotors', file_name))
