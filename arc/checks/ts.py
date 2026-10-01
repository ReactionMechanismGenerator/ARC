"""
A module for checking the quality of TS-related calculations, contains helper functions for Scheduler.
"""

from itertools import product
import os

import networkx as nx
import numpy as np
from typing import TYPE_CHECKING

from arc.parser import parser
from arc.checks.common import (IRC_START_GEOMETRY_TOLERANCE,
                               TS_ATOM_MAP_METHOD,
                               get_index_of_abs_largest_neg_freq,
                               is_ts_check_exempt,
                               record_ts_check_warning,
                               )
from arc.checks.nmd import DEFAULT_AMPLITUDE, analyze_ts_normal_mode_displacement
from arc.common import (ARC_PATH,
                        convert_list_index_0_to_1,
                        extremum_list,
                        get_logger,
                        get_bonds_from_dmat,
                        read_yaml_file,
                        sum_list_entries,
                        )
from arc.imports import settings
from arc.species.converter import check_isomorphism, check_xyz_dict, kabsch, xyz_from_data, xyz_to_dmat
from arc.species.perceive import perceive_molecule_from_xyz
from arc.statmech.factory import statmech_factory

if TYPE_CHECKING:
    from arc.job.adapter import JobAdapter
    from arc.level import Level
    from arc.molecule.molecule import Molecule
    from arc.species.species import ARCSpecies, TSGuess
    from arc.reaction import ARCReaction

logger = get_logger()

MAX_IRC_FRAGMENTS_FOR_CHARGE_SEARCH = 4
LOWEST_MAJOR_TS_FREQ, HIGHEST_MAJOR_TS_FREQ = settings['LOWEST_MAJOR_TS_FREQ'], settings['HIGHEST_MAJOR_TS_FREQ']


def check_ts(reaction: ARCReaction,
             job: JobAdapter | None = None,
             checks: list[str] | None = None,
             rxn_zone_atom_indices: list[int] | None = None,
             species_dict: dict | None = None,
             project_directory: str | None = None,
             kinetics_adapter: str | None = None,
             output: dict | None = None,
             sp_level: Level | None = None,
             freq_scale_factor: float = 1.0,
             skip_nmd: bool = False,
             verbose: bool = True,
             ):
    """
    Check the TS in terms of energy, normal mode displacement, and IRC.
    Populates the ``TS.ts_checks`` dictionary.
    Note that the 'freq' check is done in Scheduler.check_negative_freq() and not here,
    and that the 'IRC' check is done in check_irc_species_and_rxn() and not here.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        job (JobAdapter, optional): The frequency job object instance.
        checks (list[str], optional): Specific checks to run. Optional values: 'energy', 'NMD', 'IRC', 'rotors'.
        rxn_zone_atom_indices (list[int], optional): The 0-indices of atoms identified by the normal mode displacement
                                                     as the reaction zone. Automatically determined if not given.
        species_dict (dict, optional): The Scheduler species dictionary.
        project_directory (str, optional): The path to ARC's project directory.
        kinetics_adapter (str, optional): The statmech software to use for kinetic rate coefficient calculations.
        output (dict, optional): The Scheduler output dictionary.
        sp_level (Level, optional): The single-point energy level of theory.
        freq_scale_factor (float, optional): The frequency scaling factor.
        skip_nmd (bool, optional): Whether to skip the normal mode displacement check.
        verbose (bool, optional): Whether to print logging messages.
    """
    checks = checks or list()
    for entry in checks:
        if entry not in ['energy', 'NMD', 'IRC', 'rotors']:
            raise ValueError(f"Requested checks could be 'energy', 'IRC', 'NMD', or 'rotors', got:\n{checks}")

    if 'energy' in checks:
        if not reaction.ts_species.ts_checks['E0']:
            rxn_copy = compute_rxn_e0(reaction=reaction,
                                      species_dict=species_dict,
                                      project_directory=project_directory,
                                      kinetics_adapter=kinetics_adapter,
                                      output=output,
                                      sp_level=sp_level,
                                      freq_scale_factor=freq_scale_factor)
            if rxn_copy is not None:
                reaction.copy_e0_values(rxn_copy)
                if rxn_copy.ts_species.e0 is not None:
                    reaction.ts_species.e0 = rxn_copy.ts_species.e0
                    reaction.ts_species.e0_atom_corrections_applied = rxn_copy.ts_species.e0_atom_corrections_applied
                    reaction.ts_species.e0_bond_corrections_applied = rxn_copy.ts_species.e0_bond_corrections_applied
                    reaction.ts_species.e0_aec_yml_sha256 = rxn_copy.ts_species.e0_aec_yml_sha256
                check_rxn_e0(reaction=rxn_copy, verbose=verbose)
                reaction.ts_species.ts_checks['E0'] = rxn_copy.ts_species.ts_checks['E0']
            else:
                reaction.ts_species.ts_checks['E0'] = None
                logger.warning(f'Could not compute E0 values under the same settings for all species of reaction '
                               f'{reaction.label}, e.g., because a freq calculation is not done, a species E0 is '
                               f'taken from a yml file, or the statmech inputs are missing. '
                               f'The E0 check of TS {reaction.ts_species.label} is left undetermined.')
        if reaction.ts_species.ts_checks['E0'] is None and not reaction.ts_species.ts_checks['e_elect']:
            check_rxn_e_elect(reaction=reaction, verbose=verbose, check_e0=False)

    if 'NMD' in checks and not reaction.ts_species.ts_checks['NMD']:
        check_normal_mode_displacement(reaction, job=job)
        if skip_nmd and reaction.ts_species.ts_checks['NMD'] is False:
            logger.warning(f'Skipping failed normal mode displacement check for TS {reaction.ts_species.label}')
            reaction.ts_species.ts_checks['NMD'] = True
            reaction.ts_species.nmd_record['forced'] = True

    if 'rotors' in checks or (ts_passed_checks(species=reaction.ts_species, exemptions=['E0', 'warnings'])
                              and job is not None):
        invalidate_rotors_with_both_pivots_in_a_reactive_zone(reaction, job,
                                                              rxn_zone_atom_indices=rxn_zone_atom_indices)


def ts_passed_checks(species: ARCSpecies,
                     exemptions: list[str] | None = None,
                     verbose: bool = False,
                     ) -> bool:
    """
    Check whether the TS species passes all checks other than ones specified in ``exemptions``.

    The 'IRC' and 'NMD' checks are three-valued: ``True`` means the check was performed and passed,
    ``False`` means it was performed and positively failed, and ``None`` means it was not performed
    (IRC jobs were not requested; the ESS reported no normal mode displacements; the TS atom order does
    not follow the reactant order). Only a ``False`` verdict is considered a failure for those two, so
    that a check which could not run does not silently withhold the actions a passing TS receives.

    A failed 'e_elect' check is excused once 'E0' passed, as determined by
    ``checks.common.is_ts_check_exempt()``.

    Args:
        species (ARCSpecies): The TS species.
        exemptions (list[str], optional): Keys of the TS.ts_checks dict to pass.
        verbose (bool, optional): Whether to log findings.

    Returns:
        bool: Whether the TS species passed all checks.
    """
    exemptions = exemptions or list()
    for check, value in species.ts_checks.items():
        if check in exemptions or (check in ['IRC', 'NMD'] and value is None):
            continue
        if not value and not is_ts_check_exempt(check, species.ts_checks):
            if verbose:
                logger.warning(f'TS {species.label} did not pass the all checks, status is:\n{species.ts_checks}')
            return False
    return True


def check_rxn_e_elect(reaction: ARCReaction,
                      verbose: bool = True,
                      check_e0: bool = True,
                      ) -> None:
    """
    Check that the TS electronic energy is above both reactant and product wells in a ``reaction``.
    Sets the respective energy parameter 'e_elect' in the ``TS.ts_checks`` dictionary.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        verbose (bool, optional): Whether to print logging messages.
        check_e0 (bool, optional): Whether to run the E0 check first, and skip this check if the E0 check passed.
    """
    if check_e0:
        check_rxn_e0(reaction=reaction, verbose=verbose)
        if reaction.ts_species.ts_checks['E0']:
            return
    r_ee = sum_list_entries([r.e_elect for r in reaction.r_species],
                            multipliers=[reaction.get_species_count(species=r, well=0) for r in reaction.r_species])
    p_ee = sum_list_entries([p.e_elect for p in reaction.p_species],
                            multipliers=[reaction.get_species_count(species=p, well=1) for p in reaction.p_species])
    ts_ee = reaction.ts_species.e_elect
    if verbose:
        report_ts_and_wells_energy(r_e=r_ee, p_e=p_ee, ts_e=ts_ee, rxn_label=reaction.label,
                                   ts_label=reaction.ts_label, chosen_ts=reaction.ts_species.chosen_ts,
                                   energy_type='electronic energy')
    if all([val is not None for val in [r_ee, p_ee, ts_ee]]):
        if ts_ee > r_ee + 1.0 and ts_ee > p_ee + 1.0:
            reaction.ts_species.ts_checks['e_elect'] = True
            return
        if verbose:
            logger.error(f'TS of reaction {reaction.label} has a lower electronic energy value than expected.')
        reaction.ts_species.ts_checks['e_elect'] = False
        return
    if verbose:
        logger.info('\n')
        logger.warning(f"Could not get electronic energy for all species in reaction {reaction.label}.\n")
    reaction.ts_species.ts_checks['e_elect'] = None
    record_ts_check_warning(species=reaction.ts_species,
                            warning='Could not determine TS e_elect relative to the wells; ')


def compute_rxn_e0(reaction: ARCReaction,
                   species_dict: dict,
                   project_directory: str,
                   kinetics_adapter: str,
                   output: dict,
                   sp_level: Level,
                   freq_scale_factor: float = 1.0,
                   ) -> ARCReaction | None:
    """
    Checking the E0 values between wells and a TS in a ``reaction`` using ZPE from statmech.
    This function computes the E0 values of the wells and of the TS in a single statmech run, without bond
    additivity corrections, and populates them in a copy of the given reaction instance.
    E0 values already set on the given reaction are ignored and left unchanged.
    The E0 of a species with a ``yml_path`` is taken from that file as is.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        species_dict (dict): The Scheduler species dictionary.
        project_directory (str): The path to ARC's project directory.
        kinetics_adapter (str): The statmech software to use for kinetic rate coefficient calculations.
        output (dict): The Scheduler output dictionary.
        sp_level (Level): The single-point energy level of theory.
        freq_scale_factor (float, optional): The frequency scaling factor.

    Returns:
        ARCReaction | None: A copy of the reaction object with E0 values populated.
    """
    if any(val is None for val in [species_dict, project_directory, kinetics_adapter,
                                   output, sp_level, freq_scale_factor]):
        return None
    for spc in reaction.r_species + reaction.p_species + [reaction.ts_species]:
        folder = 'rxns' if species_dict[spc.label].is_ts else 'Species'
        freq_path = os.path.join(project_directory, 'output', folder, spc.label, 'geometry', 'freq.out')
        if not spc.yml_path and not os.path.isfile(freq_path) and not species_dict[spc.label].is_monoatomic():
            return None
    rxn_copy = reaction.copy()
    species_list = rxn_copy.r_species + rxn_copy.p_species + [rxn_copy.ts_species]
    unique_species = dict()
    for species in species_list:
        species.e0 = None
        species.e0_atom_corrections_applied = None
        species.e0_bond_corrections_applied = None
        species.e0_aec_yml_sha256 = None
        unique_species.setdefault(species.label, species)
        if species.yml_path:
            logger.warning(f'The E0 of {species.label} is taken from its yml file as is, '
                           f'and may include bond additivity corrections.')
    statmech_adapter = statmech_factory(statmech_adapter_label=kinetics_adapter,
                                        output_directory=os.path.join(project_directory, 'output'),
                                        calcs_directory=os.path.join(project_directory, 'calcs'),
                                        output_dict=output,
                                        species=list(unique_species.values()),
                                        bac_type=None,
                                        sp_level=sp_level,
                                        freq_scale_factor=freq_scale_factor,
                                        )
    statmech_adapter.compute_thermo(e0_only=True, skip_rotors=True)
    for species in species_list:
        species.e0 = unique_species[species.label].e0
        species.e0_atom_corrections_applied = unique_species[species.label].e0_atom_corrections_applied
        species.e0_bond_corrections_applied = unique_species[species.label].e0_bond_corrections_applied
        species.e0_aec_yml_sha256 = unique_species[species.label].e0_aec_yml_sha256
    return rxn_copy


def check_rxn_e0(reaction: ARCReaction,
                 verbose: bool = True,
                 ):
    """
    Check the E0 values between wells and a TS in a ``reaction``, assuming that E0 values are available.

    Args:
        reaction (ARCReaction): The reaction to consider.
        verbose (bool, optional): Whether to print logging messages.
    """
    if reaction.ts_species.ts_checks['E0']:
        return
    r_e0 = sum_list_entries([r.e0 for r in reaction.r_species],
                            multipliers=[reaction.get_species_count(species=r, well=0) for r in reaction.r_species])
    p_e0 = sum_list_entries([p.e0 for p in reaction.p_species],
                            multipliers=[reaction.get_species_count(species=p, well=1) for p in reaction.p_species])
    ts_e0 = reaction.ts_species.e0
    if any(e0 is None for e0 in [r_e0, p_e0, ts_e0]):
        reaction.ts_species.ts_checks['E0'] = None
    else:
        if verbose:
            report_ts_and_wells_energy(r_e=r_e0, p_e=p_e0, ts_e=ts_e0, rxn_label=reaction.label, ts_label=reaction.ts_label,
                                       chosen_ts=reaction.ts_species.chosen_ts, energy_type='E0')
        if r_e0 >= ts_e0 or p_e0 >= ts_e0:
            reaction.ts_species.ts_checks['E0'] = False
        if r_e0 + 1 >= ts_e0 or p_e0 + 1 >= ts_e0:
            logger.warning(f'The TS energy of {reaction.ts_label} is less than 1 kJ/mol above one of the wells '
                           f'of {reaction.label}, skipping this TS')
            reaction.ts_species.ts_checks['E0'] = False
        else:
            reaction.ts_species.ts_checks['E0'] = True


def report_ts_and_wells_energy(r_e: float,
                               p_e: float,
                               ts_e: float,
                               rxn_label: str,
                               ts_label: str,
                               chosen_ts: int,
                               energy_type: str = 'electronic energy',
                               ):
    """
    Report the relative R/TS/P energies.

    Args:
        r_e (float): The reactant energy.
        p_e (float): The product energy.
        ts_e (float): The TS energy.
        rxn_label (str): The reaction label.
        ts_label (str): The TS label.
        chosen_ts (int): The TSG number.
        energy_type (str): The energy type: 'electronic energy' or 'E0'.
    """
    if all([val is not None for val in [r_e, p_e, ts_e]]):
        min_e = extremum_list(lst=[r_e, p_e, ts_e], return_min=True)
        r_text = f'{r_e - min_e:.2f} kJ/mol'
        ts_text = f'{ts_e - min_e:.2f} kJ/mol'
        p_text = f'{p_e - min_e:.2f} kJ/mol'
        logger.info(
            f'\nReaction {rxn_label} (TS {ts_label}, TSG {chosen_ts}) has the following {energy_type} values:\n'
            f'Reactants: {r_text}\n'
            f'TS: {ts_text}\n'
            f'Products: {p_text}')


def check_normal_mode_displacement(reaction: ARCReaction,
                                   job: JobAdapter | None,
                                   amplitude: float | list | None = None,
                                   ) -> None:
    """
    Check the normal mode displacement by identifying bonds that break and form
    and comparing them to the expected RMG template, if available.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        job (JobAdapter): The frequency job object instance.
        amplitude (float | list | None): The amplitude of the normal mode displacement motion to check.
                                        If a list, all possible results are returned.
                                        ``None`` selects ``DEFAULT_AMPLITUDE``; any other value is passed
                                        through as given, including ``0`` and an empty list, which both
                                        leave no amplitude to probe.
    """
    amplitude = DEFAULT_AMPLITUDE if amplitude is None else amplitude
    reaction.ts_species.nmd_record = dict()
    reaction.ts_species.ts_checks['NMD'] = analyze_ts_normal_mode_displacement(reaction=reaction,
                                                                               job=job,
                                                                               amplitude=amplitude,
                                                                               )


def determine_changing_bond(bond: tuple[int, ...],
                            dmat_bonds_1: list[tuple[int, int]],
                            dmat_bonds_2: list[tuple[int, int]],
                            ) -> str | None:
    """
    Determine whether a bond breaks or forms in a TS.
    Note that ``bond`` and all bond entries in `dmat_bonds_1/2`` must be already sorted from small to large indices.

    Args:
        bond (tuple[int]): The atom indices describing the bond.
        dmat_bonds_1 (list[tuple[int, int]]): The bonds perceived from dmat_1.
        dmat_bonds_2 (list[tuple[int, int]]): The bonds perceived from dmat_2.

    Returns:
        bool | None:
            'forming' if the bond indeed forms between ``dmat_1`` and ``dmat_2``, 'breaking' if it indeed breaks,
            ``None`` if it does not change significantly.
    """
    if len(bond) != 2 or any(not isinstance(entry, int) for entry in bond):
        raise ValueError(f'Expected a bond to be represented by a list of length 2 with int entries, got {bond} '
                         f'of length {len(bond) if isinstance(bond, list) else None} with {type(bond[0]), type(bond[1])}')
    if bond not in dmat_bonds_1 and bond in dmat_bonds_2:
        return 'forming'
    if bond in dmat_bonds_1 and bond not in dmat_bonds_2:
        return 'breaking'
    return None


def invalidate_rotors_with_both_pivots_in_a_reactive_zone(reaction: ARCReaction,
                                                          job: JobAdapter,
                                                          rxn_zone_atom_indices: list[int] | None = None,
                                                          ):
    """
    Invalidate rotors in which both pivots are included in the reactive zone.

    Args:
        reaction (ARCReaction): The respective reaction object instance.
        job (JobAdapter): The frequency job object instance.
        rxn_zone_atom_indices (list[int], optional): The 0-indices of atoms identified by the normal mode displacement
                                                     as the reaction zone. Automatically determined if not given.
    """
    rxn_zone_atom_indices = rxn_zone_atom_indices or get_rxn_zone_atom_indices(reaction, job)
    if not rxn_zone_atom_indices:
        logger.warning(f'No reaction zone could be determined for TS {reaction.ts_species.label}, so no rotor of '
                       f'it was invalidated for having both pivots in the reaction zone. A rotor along the '
                       f'reaction coordinate, if there is one, is treated as an ordinary hindered rotor.')
        return
    if not reaction.ts_species.rotors_dict:
        reaction.ts_species.determine_rotors()
    rxn_zone_atom_indices_1 = convert_list_index_0_to_1(rxn_zone_atom_indices)
    for key, rotor in reaction.ts_species.rotors_dict.items():
        if rotor['pivots'][0] in rxn_zone_atom_indices_1 and rotor['pivots'][1] in rxn_zone_atom_indices_1:
            rotor['success'] = False
            if 'pivTS' not in rotor['invalidation_reason']:
                rotor['invalidation_reason'] += 'Pivots participate in the TS reaction zone (code: pivTS). '
                logger.info(f"\nNot considering rotor {key} with pivots {rotor['pivots']} in TS {reaction.ts_species.label}\n")


def get_rxn_zone_atom_indices(reaction: ARCReaction,
                              job: JobAdapter,
                              ) -> list[int]:
    """
    Get the reaction zone atom indices by parsing normal mode displacement.

    Args:
        reaction (ARCReaction): The respective reaction object instance.
        job (JobAdapter): The frequency job object instance.

    Returns:
        list[int]: The indices of the atoms participating in the reaction.
                   The indices are 0-indexed and sorted in an increasing order.
                   An empty list is returned when the output file yields no normal mode displacements,
                   since the reaction zone is derived from them alone.
    """
    parsed_modes = parser.get_normal_mode_displacement(log_file_path=job.local_path_to_output_file,
                                                       label=reaction.ts_species.label)
    if parsed_modes is None:
        return list()
    freqs, normal_mode_disp = parsed_modes
    normal_mode_disp_rms = get_rms_from_normal_mode_disp(normal_mode_disp, freqs, reaction=reaction)
    num_of_atoms = get_expected_num_atoms_with_largest_normal_mode_disp(normal_mode_disp_rms=normal_mode_disp_rms,
                                                                        ts_guesses=reaction.ts_species.ts_guesses,
                                                                        reaction=reaction) \
                   + round(reaction.ts_species.number_of_atoms ** 0.25)  # Peripheral atoms might get in the way
    indices = sorted(range(len(normal_mode_disp_rms)), key=lambda i: normal_mode_disp_rms[i], reverse=True)[:num_of_atoms]
    return indices


def get_rms_from_normal_mode_disp(normal_mode_disp: np.ndarray,
                                  freqs: np.ndarray,
                                  reaction: ARCReaction | None = None,
                                  ) -> list[float]:
    """
    Get the root mean squares of the normal mode displacements.
    Use atom mass weights if ``reaction`` is given.

    Args:
        normal_mode_disp (np.ndarray): The normal mode displacement array.
        freqs (np.ndarray): Entries are frequency values.
        reaction (ARCReaction): The respective reaction object instance.

    Returns:
        list[float]: The RMS of the normal mode displacements.
    """
    mode_index = get_index_of_abs_largest_neg_freq(freqs)
    nmd = normal_mode_disp[mode_index]
    masses = reaction.get_element_mass() if reaction is not None else [1] * len(nmd)
    rms = list()
    for i, entry in enumerate(nmd):
        rms.append(((entry[0] ** 2 + entry[1] ** 2 + entry[2] ** 2) ** 0.5) * masses[i] ** 0.55)
    return rms


def get_expected_num_atoms_with_largest_normal_mode_disp(normal_mode_disp_rms: list[float],
                                                         ts_guesses: list[TSGuess],
                                                         reaction: ARCReaction | None = None,
                                                         ) -> int:
    """
    Get the number of atoms that are expected to have the largest normal mode displacement for the TS
    (considering all families). This is a wrapper for ``get_rxn_normal_mode_disp_atom_number()``.
    It is theoretically possible that TSGuesses of the same species will belong to different families.

    Args:
        normal_mode_disp_rms (list[float]): The RMS of the normal mode displacements.
        ts_guesses (list[TSGuess]): The TSGuess objects of a TS species.
        reaction (ARCReaction): The respective reaction object instance.

    Returns:
        int: The number of atoms to consider that have a significant motions in the normal mode displacement.
    """
    num_of_atoms = reaction.get_number_of_atoms_in_reaction_zone() if reaction is not None else None
    if num_of_atoms is not None:
        return num_of_atoms
    families = list(set([tsg.family for tsg in ts_guesses]))
    num_of_atoms = max([get_rxn_normal_mode_disp_atom_number(rxn_family=family,
                                                             reaction=reaction,
                                                             rms_list=normal_mode_disp_rms,
                                                             )
                        for family in families])
    return num_of_atoms


def get_rxn_normal_mode_disp_atom_number(rxn_family: str | None = None,
                                         reaction: ARCReaction | None = None,
                                         rms_list: list[float] | None = None,
                                         ) -> int:
    """
    Get the number of atoms expected to have the largest normal mode displacement per family.
    If ``rms_list`` is given, also include atoms with an RMS value close to the lowest RMS still considered.

    Args:
        rxn_family (str, optional): The reaction family label.
        reaction (ARCReaction, optional): The reaction object instance.
        rms_list (list[float], optional): The root mean squares of the normal mode displacements.

    Raises:
        TypeError: If ``rms_list`` is not ``None`` and is either not a list or does not contain floats.

    Returns:
        int: The respective number of atoms.
    """
    if rxn_family is None and reaction is None:
        raise ValueError('Either `rxn_family` or `reaction` must be given.')
    default = 3
    if rms_list is not None \
            and (not isinstance(rms_list, list) or not all(isinstance(entry, float) for entry in rms_list)):
        raise TypeError(f'rms_list must be a non empty list, got {rms_list} of type {type(rms_list)}.')
    if reaction is not None and reaction.family is None:
        logger.warning(f'Cannot deduce a reaction family for {reaction}, assuming {default} atoms in the reaction zone.')
        return default
    content = read_yaml_file(os.path.join(ARC_PATH, 'data', 'rxn_normal_mode_disp.yml'))
    number_by_family = content.get(rxn_family or reaction.family, default)
    if rms_list is None or not len(rms_list):
        return number_by_family
    entry = None
    rms_list = rms_list.copy()
    for i in range(number_by_family):
        entry = max(rms_list)
        rms_list.pop(rms_list.index(entry))
    if entry is not None:
        for rms in rms_list:
            if (entry - rms) / entry < 0.12:
                number_by_family += 1
    return number_by_family


def check_irc_species_and_rxn(xyz_1: dict,
                              xyz_2: dict,
                              rxn: ARCReaction | None,
                              endpoint_labels: tuple[str | None, str | None] | None = None,
                              irc_log_paths: list[str] | None = None,
                              endpoint_log_paths: list[str] | None = None,
                              ):
    """
    Check that the two species that result from optimizing the outputs of two IRC runs
    correspond to the desired reactants and products of the corresponding reaction.

    Uses molecular graph isomorphism (including bond orders and resonance structures)
    when molecule perception succeeds for both endpoints. Falls back to distance-matrix-based
    bond-list comparison if perception fails for either endpoint or if the expected
    reactant/product ``Molecule`` objects are unavailable.

    Sets ``rxn.ts_species.ts_checks['IRC']`` to ``True`` if the endpoints match the reactants
    and products, to ``False`` only if a comparison was actually performed and the endpoints
    did not match, and leaves it as ``None`` (unknown) if no comparison could be performed
    (e.g., the expected connectivity of the reaction is unavailable). ``False`` means
    "checked and failed", never "could not check".

    A negative isomorphism result alone does not set ``False``: it is treated as inconclusive
    and the bond-list comparison decides. The verdict remains ``None`` if that comparison
    cannot be carried out.

    The bond-list comparison perceives bonds without assuming that an endpoint is a single
    connected molecule, so a dissociated fragment stays dissociated instead of being bonded
    to its nearest neighbour regardless of distance.

    Args:
        xyz_1 (dict): The coordinates of IRC species 1.
        xyz_2 (dict): The coordinates of IRC species 2.
        rxn (ARCReaction): The corresponding reaction object instance.
        endpoint_labels (tuple[str | None, str | None], optional): The labels of the species whose geometries are
                                                                   ``xyz_1`` and ``xyz_2``.
        irc_log_paths (list[str], optional): The IRC logs, in the order of the endpoints. Their starting geometries
                                             must be the TS geometry for a TS atom map to be recorded.
        endpoint_log_paths (list[str], optional): The optimization logs of the two endpoint species, in the order of
                                                  ``xyz_1`` and ``xyz_2``, which must start from the last geometry of
                                                  the IRC log of the same index and end at ``xyz_1`` and ``xyz_2``.

    When the isomorphism comparison establishes the verdict, ``rxn.ts_species.irc_participant_mapping`` records which
    atoms of each endpoint geometry belong to which participant species (see ``_get_irc_participant_mapping``);
    otherwise it is ``None``, and it is also ``None`` when building it fails, which leaves the verdict as decided.
    On the same path ``rxn.ts_species.ts_atom_map`` records the TS atom of every reactant and product atom of the
    reaction (see ``get_ts_atom_map``), and ``rxn.ts_species.ts_atom_map_unavailable_reason`` says why it could not;
    the bond-list fallback records the reason ``'irc_fallback_path'``. Neither affects the verdict or the atom map.
    """
    if rxn is None:
        return None
    rxn.ts_species.ts_checks['IRC'] = None
    rxn.ts_species.irc_participant_mapping = None
    rxn.ts_species.ts_atom_map = None
    rxn.ts_species.ts_atom_map_unavailable_reason = None
    xyz_1, xyz_2 = check_xyz_dict(xyz_1), check_xyz_dict(xyz_2)

    # Primary check: molecular graph isomorphism
    reactants, products = rxn.get_reactants_and_products(return_copies=True)
    r_mols = [r.mol for r in reactants]
    p_mols = [p.mol for p in products]

    if not any(m is None for m in r_mols + p_mols):
        charge = rxn.charge or 0
        frags_1 = _perceive_irc_fragments(xyz_1, charge=charge)
        frags_2 = _perceive_irc_fragments(xyz_2, charge=charge)
        if frags_1 is not None and frags_2 is not None:
            assign_1_r = _assign_fragments_to_species(frags_1, r_mols)
            assign_2_p = _assign_fragments_to_species(frags_2, p_mols) if assign_1_r is not None else None
            if assign_1_r is not None and assign_2_p is not None:
                reactant_side, product_side = (1, assign_1_r), (2, assign_2_p)
            else:
                assign_1_p = _assign_fragments_to_species(frags_1, p_mols)
                assign_2_r = _assign_fragments_to_species(frags_2, r_mols) if assign_1_p is not None else None
                reactant_side, product_side = ((2, assign_2_r), (1, assign_1_p)) if assign_2_r is not None \
                    else (None, None)
            if reactant_side is not None and product_side is not None:
                mapping, order_inputs = None, None
                ts_xyz = rxn.ts_species.get_xyz(generate=False)
                has_ts = isinstance(ts_xyz, dict) and bool(ts_xyz.get('symbols'))
                try:
                    start_reason = get_irc_start_geometry_reason(ts_xyz, irc_log_paths) if has_ts else 'no_ts'
                    chain_reason = get_irc_endpoint_chain_reason({1: xyz_1, 2: xyz_2},
                                                                 endpoint_log_paths,
                                                                 irc_log_paths) if has_ts else None
                    order_inputs = (start_reason, chain_reason, _assign_fragments_to_species(r_mols, p_mols) is None)
                except Exception as e:
                    logger.warning(f'Could not compare the IRC geometries of {rxn.ts_species.label} with its '
                                   f'TS, got:\n{e.__class__.__name__}: {e}\n'
                                   f'The IRC verdict is unaffected.')
                if order_inputs is not None:
                    start_reason, chain_reason, sides_distinguishable = order_inputs
                    geometry_reason = start_reason or chain_reason
                    elements_agree = has_ts and tuple(xyz_1['symbols']) == tuple(ts_xyz['symbols']) \
                        == tuple(xyz_2['symbols'])
                    atom_order_matches_ts = None if not has_ts \
                        else False if (geometry_reason in ('irc_start_geometry_differs',
                                                           'irc_endpoint_geometry_differs')
                                       or not elements_agree) \
                        else True if geometry_reason is None else None
                    try:
                        endpoint_atoms = {1: _get_irc_fragment_atom_indices(xyz_1),
                                          2: _get_irc_fragment_atom_indices(xyz_2)}
                        mapping = _get_irc_participant_mapping(
                            reactants=reactants,
                            products=products,
                            reactant_side=(reactant_side[0], endpoint_atoms[reactant_side[0]], reactant_side[1]),
                            product_side=(product_side[0], endpoint_atoms[product_side[0]], product_side[1]),
                            endpoint_labels=endpoint_labels,
                            sides_distinguishable=sides_distinguishable,
                            atom_order_matches_ts=atom_order_matches_ts,
                        )
                    except Exception as e:
                        logger.warning(f'Could not build the IRC participant mapping of {rxn.ts_species.label}, '
                                       f'got:\n{e.__class__.__name__}: {e}\n'
                                       f'The IRC verdict is unaffected.')
                        mapping = None
                    try:
                        rxn.ts_species.ts_atom_map, rxn.ts_species.ts_atom_map_unavailable_reason = get_ts_atom_map(
                            rxn=rxn,
                            reactants=reactants,
                            endpoint_xyzs={1: xyz_1, 2: xyz_2},
                            endpoint_fragments={1: frags_1, 2: frags_2},
                            reactant_endpoint=reactant_side[0],
                            ts_xyz=ts_xyz,
                            irc_geometry_reason=geometry_reason,
                            sides_distinguishable=sides_distinguishable,
                        )
                    except Exception as e:
                        logger.warning(f'Could not build the TS atom map of {rxn.ts_species.label}, '
                                       f'got:\n{e.__class__.__name__}: {e}\n'
                                       f'The IRC verdict is unaffected.')
                        rxn.ts_species.ts_atom_map = None
                        rxn.ts_species.ts_atom_map_unavailable_reason = 'computation_failed'
                else:
                    rxn.ts_species.ts_atom_map_unavailable_reason = 'computation_failed'
                rxn.ts_species.ts_checks['IRC'] = True
                rxn.ts_species.irc_participant_mapping = mapping
                return
            logger.debug('IRC isomorphism check failed, falling back to bond-list comparison.')
        else:
            logger.debug('IRC molecule perception failed for one or both endpoints, '
                         'falling back to bond-list comparison.')

    # Fallback: bond-list connectivity comparison
    rxn.ts_species.ts_atom_map_unavailable_reason = 'irc_fallback_path'
    try:
        r_bonds, p_bonds = rxn.get_bonds()
    except Exception as e:
        logger.warning(f'Could not get the reaction bonds of {rxn} for the IRC fallback check, '
                       f'got:\n{e.__class__.__name__}: {e}\n'
                       f'The IRC check of {rxn.ts_species.label} is therefore left undetermined.')
        return
    dmat_1, dmat_2 = xyz_to_dmat(xyz_1), xyz_to_dmat(xyz_2)
    dmat_bonds_1 = get_bonds_from_dmat(dmat=dmat_1, elements=xyz_1['symbols'], n_fragments=0)
    dmat_bonds_2 = get_bonds_from_dmat(dmat=dmat_2, elements=xyz_2['symbols'], n_fragments=0)
    rxn_symbols = [atom.element.symbol for spc in reactants for atom in spc.mol.atoms]
    bonds_1 = {tuple(sorted(bond)) for bond in dmat_bonds_1}
    bonds_2 = {tuple(sorted(bond)) for bond in dmat_bonds_2}
    reaction_graph = _get_condensed_graph_of_reaction(rxn_symbols, r_bonds, p_bonds)
    if bonds_1 != bonds_2 and any(
            _find_cgr_isomorphism(reaction_graph,
                                  _get_condensed_graph_of_reaction(list(xyz_1['symbols']), r_bonds_ts, p_bonds_ts))
            is not None
            for r_bonds_ts, p_bonds_ts in ((bonds_1, bonds_2), (bonds_2, bonds_1))):
        rxn.ts_species.ts_checks['IRC'] = True
    else:
        rxn.ts_species.ts_checks['IRC'] = False


def _get_irc_fragment_atom_indices(xyz: dict) -> list[list[int]]:
    """
    Get the atoms of each connected component of an IRC endpoint geometry, from its distance-matrix-based bond list.

    Args:
        xyz (dict): The Cartesian coordinates of the IRC endpoint.

    Returns:
        list[list[int]]: The ascending 0-based atom indices of each component, the components ordered by their
                         lowest atom index.
    """
    symbols = xyz['symbols']
    n_atoms = len(symbols)

    dmat = xyz_to_dmat(xyz)
    # Pass n_fragments != 1 to skip the heavy-atom bridging heuristic in get_bonds_from_dmat.
    bonds = get_bonds_from_dmat(dmat=dmat, elements=symbols, n_fragments=0)

    adj = {i: set() for i in range(n_atoms)}
    for a, b in bonds:
        adj[a].add(b)
        adj[b].add(a)

    visited = set()
    fragment_indices = []
    for start in range(n_atoms):
        if start in visited:
            continue
        component = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        fragment_indices.append(sorted(component))
    return fragment_indices


def _perceive_irc_fragments(xyz: dict,
                            charge: int = 0,
                            ) -> list[Molecule] | None:
    """
    Perceive individual molecular fragments from an IRC endpoint geometry.

    Detects connected components from the distance-matrix-based bond list,
    then perceives fragments as ``Molecule`` objects. For multi-fragment systems,
    charge distribution across fragments is handled by brute-force search over
    charge splits that sum to the total charge, preferring minimal total absolute charge.

    Args:
        xyz (dict): The Cartesian coordinates of the IRC endpoint.
        charge (int): The net charge of the full system.

    Returns:
        list[Molecule] | None: A list of perceived ``Molecule`` objects (one per fragment, in the order of
                               ``_get_irc_fragment_atom_indices``), or ``None`` if perception fails for any fragment.
    """
    symbols = xyz['symbols']
    coords = xyz['coords']
    fragment_indices = _get_irc_fragment_atom_indices(xyz)

    n_frags = len(fragment_indices)
    frag_xyzs = []
    for frag_idx in fragment_indices:
        frag_symbols = tuple(symbols[i] for i in frag_idx)
        frag_coords = tuple(coords[i] for i in frag_idx)
        frag_xyzs.append(xyz_from_data(coords=frag_coords, symbols=frag_symbols))

    if n_frags == 1:
        mol = perceive_molecule_from_xyz(frag_xyzs[0], charge=charge, n_fragments=1)
        return [mol] if mol is not None else None

    # Prefer splits that minimize the total absolute charge (e.g., 0,0 over +1,-1).
    if n_frags > MAX_IRC_FRAGMENTS_FOR_CHARGE_SEARCH:
        return None
    max_abs_charge = max(2, abs(charge) + 1)
    charge_range = range(-max_abs_charge, max_abs_charge + 1)
    best_mols = None
    best_sep = float('inf')
    for charges in product(charge_range, repeat=n_frags):
        if sum(charges) != charge:
            continue
        sep = sum(abs(c) for c in charges)
        if sep >= best_sep:
            continue
        mols = []
        ok = True
        for frag_xyz, frag_charge in zip(frag_xyzs, charges):
            mol = perceive_molecule_from_xyz(frag_xyz, charge=frag_charge, n_fragments=1)
            if mol is None or mol.get_net_charge() != frag_charge:
                ok = False
                break
            mols.append(mol)
        if ok:
            best_mols, best_sep = mols, sep
            if sep == 0:
                break
    return best_mols


def _assign_fragments_to_species(fragments: list[Molecule],
                                 expected_mols: list[Molecule],
                                 ) -> list[int] | None:
    """
    Find a one-to-one matching between perceived molecular fragments and expected species via graph isomorphism.
    Handles multi-species reactions (e.g., A + B) using backtracking with pruning. When several matchings exist
    (e.g., a species occurring twice), the first one found is returned.

    Args:
        fragments (list[Molecule]): Perceived fragment molecules from an IRC endpoint.
        expected_mols (list[Molecule]): Expected species molecules from the reaction.

    Returns:
        list[int] | None: Entry ``i`` is the index in ``expected_mols`` matched to ``fragments[i]``,
                          or ``None`` if no valid one-to-one isomorphic matching exists.
    """
    n = len(fragments)
    if n != len(expected_mols):
        return None
    if n == 0:
        return list()
    frag_formulas = sorted(frag.get_formula() for frag in fragments)
    expected_formulas = sorted(mol.get_formula() for mol in expected_mols)
    if frag_formulas != expected_formulas:
        return None
    if n == 1:
        return [0] if check_isomorphism(fragments[0], expected_mols[0]) else None
    iso_matrix = [[check_isomorphism(fragments[i], expected_mols[j]) for j in range(n)] for i in range(n)]
    used = [False] * n
    assignment = [-1] * n

    def _backtrack(i: int) -> bool:
        if i == n:
            return True
        for j in range(n):
            if not used[j] and iso_matrix[i][j]:
                used[j] = True
                assignment[i] = j
                if _backtrack(i + 1):
                    return True
                used[j] = False
        return False

    return assignment if _backtrack(0) else None


def _get_irc_participant_mapping(reactants: list,
                                 products: list,
                                 reactant_side: tuple[int, list[list[int]], list[int]],
                                 product_side: tuple[int, list[list[int]], list[int]],
                                 endpoint_labels: tuple[str | None, str | None] | None = None,
                                 sides_distinguishable: bool = True,
                                 atom_order_matches_ts: bool | None = None,
                                 ) -> dict:
    """
    Describe which atoms of each IRC endpoint geometry belong to which participant species.

    Args:
        reactants (list[ARCSpecies]): The reactants, repeated species included (``get_reactants_and_products``).
        products (list[ARCSpecies]): The products, repeated species included.
        reactant_side (tuple): ``(endpoint, fragment atom indices, assignment)`` of the endpoint matched to the
                               reactants, where ``endpoint`` is ``1`` or ``2`` for the first or second geometry given
                               to the check.
        product_side (tuple): The same for the endpoint matched to the products.
        endpoint_labels (tuple[str | None, str | None], optional): The labels of the species of the first and second
                                                                   endpoint geometries.
        sides_distinguishable (bool, optional): Whether the reactants and the products are not graph-isomorphic to
                                                each other. When they are (e.g., ``CH3 + CH4 <=> CH4 + CH3``), which
                                                endpoint is called the reactants is a convention.
        atom_order_matches_ts (bool | None, optional): Whether every IRC started from a geometry congruent with the TS
                                                        geometry, atom for atom (the IRC start geometry check);
                                                        ``None`` if that could not be checked.

    Returns:
        dict: ``{'reactants': side, 'products': side, 'sides_distinguishable': bool,
              'atom_order_matches_ts': bool | None}``, where each side is
              ``{'endpoint': 1 | 2, 'endpoint_label': str | None, 'participants': [...]}`` and each participant is
              ``{'label': str, 'position': int, 'occurrence': int, 'atom_indices': list[int]}``. Participants are
              listed in the order of ``reactants`` / ``products``; ``position`` is 1-based there and ``occurrence``
              is the 1-based count of that label among the participants up to it. ``atom_indices`` are ascending
              0-based atom indices into the geometry of the endpoint (not of the TS). Which fragment of the endpoint
              is which occurrence of a repeated species is arbitrary, and the order of the atoms within a participant
              is not matched to the species' own atom order: only atom-set membership is recorded.
    """
    mapping = dict()
    for well, (endpoint, fragment_indices, assignment), species in (('reactants', reactant_side, reactants),
                                                                     ('products', product_side, products)):
        fragment_by_species = {species_index: fragment_index
                               for fragment_index, species_index in enumerate(assignment)}
        seen = dict()
        participants = list()
        for position, spc in enumerate(species, start=1):
            seen[spc.label] = seen.get(spc.label, 0) + 1
            participants.append({'label': spc.label,
                                 'position': position,
                                 'occurrence': seen[spc.label],
                                 'atom_indices': list(fragment_indices[fragment_by_species[position - 1]]),
                                 })
        mapping[well] = {'endpoint': endpoint,
                         'endpoint_label': endpoint_labels[endpoint - 1] if endpoint_labels is not None else None,
                         'participants': participants,
                         }
    mapping['sides_distinguishable'] = sides_distinguishable
    mapping['atom_order_matches_ts'] = atom_order_matches_ts
    return mapping


def get_ts_atom_map(rxn: ARCReaction,
                    reactants: list[ARCSpecies],
                    endpoint_xyzs: dict[int, dict],
                    endpoint_fragments: dict[int, list[Molecule]],
                    reactant_endpoint: int,
                    ts_xyz: dict | None,
                    irc_geometry_reason: str | None,
                    sides_distinguishable: bool = True,
                    ) -> tuple[dict | None, str | None]:
    """
    Get the TS atom of every reactant atom and of every product atom of a reaction whose IRC endpoints were matched
    to its reactants and products, from a label-preserving isomorphism of two condensed graphs of reaction (CGRs).

    The reactant-indexed CGR has the concatenated reactant atoms (``get_reactants_and_products`` order, repeats
    expanded, the order ``rxn.atom_map`` counts in) as nodes labelled by element, and an edge for every bond of the
    reactants or of the products (``rxn.get_bonds()``, both already in reactant atom indices), labelled by whether it
    is in the reactants and whether it is in the products. The TS-indexed CGR has the TS atoms as nodes and, as edges,
    the connectivity of the reactant endpoint and of the product endpoint, taken from the ``Molecule`` fragments the
    IRC verdict perceived (``_get_endpoint_bonds_from_fragments``), labelled the same way. Only connectivity and
    element are used, never bond orders.

    Atom ``i`` of an endpoint is taken to be atom ``i`` of the TS. That rests on ARC starting each IRC from the TS
    geometry, which is verified from the first geometry of each IRC log, and the endpoint geometries on those
    IRC logs through the endpoint optimization logs (``irc_geometry_reason``, from ``get_irc_start_geometry_reason``
    and ``get_irc_endpoint_chain_reason``), and on the ESS keeping the atom order of its input within a log and ARC's
    parsing and writing keeping it too. Before the CGRs are compared, the connectivity of each endpoint must be
    isomorphic to the bond graph of its side. An isomorphism ``t`` of the first CGR onto the second states that
    reactant atom ``i`` is TS atom ``t[i]``, and product atom ``j`` is TS atom ``t[atom_map.index(j)]``. The map is
    a constitutional (2D) correspondence. The identity is preferred when it is a solution; otherwise the first
    isomorphism the matcher finds is taken, so symmetry-equivalent atoms, diastereotopic ones included, follow a
    deterministic convention that is not canonical across networkx versions, and the number of equivalent maps is
    not stated. When the reactants and the products are isomorphic to each other (``sides_distinguishable`` is
    ``False``), the endpoints are also tried in the opposite roles. Neither ``rxn.atom_map`` nor anything else is
    altered.

    Args:
        rxn (ARCReaction): The reaction, with its atom map already set (it is never computed here).
        reactants (list[ARCSpecies]): The reactants, repeated species included (``get_reactants_and_products``).
        endpoint_xyzs (dict[int, dict]): The two IRC endpoint geometries, keyed ``1`` and ``2``.
        endpoint_fragments (dict[int, list[Molecule]]): The fragments the IRC verdict perceived on each endpoint.
        reactant_endpoint (int): The endpoint (``1`` or ``2``) that was matched to the reactants.
        ts_xyz (dict | None): The TS geometry.
        irc_geometry_reason (str | None): ``None`` when every IRC log starts from the TS geometry and the endpoint
                                          geometries follow from the IRC logs, otherwise the reason from
                                          ``get_irc_start_geometry_reason`` or ``get_irc_endpoint_chain_reason``.
        sides_distinguishable (bool, optional): Whether the reactants and the products are not isomorphic.

    Returns:
        tuple[dict | None, str | None]: The map ``{'ts_label', 'reactants', 'products', 'method',
                                        'reactant_endpoint', 'ts_atom_order_follows_reactants'}``, where ``reactants``
                                        and ``products`` hold 0-based TS atom indices and ``reactant_endpoint`` is the
                                        endpoint that served as the reactants, or ``None`` and one of
                                        ``TS_ATOM_MAP_UNAVAILABLE_REASONS``.
    """
    if not isinstance(ts_xyz, dict) or not ts_xyz.get('symbols'):
        return None, 'no_ts'
    atom_map = getattr(rxn, '_atom_map', None)
    n_atoms = len(atom_map) if isinstance(atom_map, (list, tuple)) else 0
    if not n_atoms or sorted(atom_map) != list(range(n_atoms)):
        return None, 'no_atom_map'
    try:
        r_bonds, p_bonds = rxn.get_bonds()
    except Exception as e:
        logger.warning(f'Could not get the bonds of {rxn} to map the atoms of {rxn.ts_species.label}, '
                       f'got:\n{e.__class__.__name__}: {e}')
        return None, 'no_atom_map'
    r_symbols = [atom.element.symbol for spc in reactants for atom in spc.mol.atoms]
    ts_symbols = list(ts_xyz['symbols'])
    if len(r_symbols) != n_atoms or len(ts_symbols) != n_atoms:
        return None, 'atom_map_contradicts_ts'
    if irc_geometry_reason is not None:
        return None, irc_geometry_reason
    endpoint_bonds = dict()
    for endpoint in (1, 2):
        bonds = _get_endpoint_bonds_from_fragments(endpoint_xyzs[endpoint], endpoint_fragments[endpoint])
        if bonds is None or list(endpoint_xyzs[endpoint]['symbols']) != ts_symbols:
            logger.warning(f'The fragments the IRC verdict perceived on endpoint {endpoint} of '
                           f'{rxn.ts_species.label} cannot be tied to its geometry and to the TS atoms, '
                           f'so no TS atom map is recorded.')
            return None, 'endpoint_perception_mismatch'
        endpoint_bonds[endpoint] = bonds
    reactant_graph = _get_condensed_graph_of_reaction(r_symbols, r_bonds, p_bonds)
    other_endpoint = 2 if reactant_endpoint == 1 else 1
    orientations = [(reactant_endpoint, other_endpoint)]
    if not sides_distinguishable:
        orientations.append((other_endpoint, reactant_endpoint))
    endpoints_perceived = False
    for r_endpoint, p_endpoint in orientations:
        r_endpoint_bonds, p_endpoint_bonds = endpoint_bonds[r_endpoint], endpoint_bonds[p_endpoint]
        if not (_are_bond_graphs_isomorphic(r_symbols, r_bonds, ts_symbols, r_endpoint_bonds)
                and _are_bond_graphs_isomorphic(r_symbols, p_bonds, ts_symbols, p_endpoint_bonds)):
            continue
        endpoints_perceived = True
        isomorphism = _find_cgr_isomorphism(reactant_graph,
                                            _get_condensed_graph_of_reaction(ts_symbols,
                                                                             r_endpoint_bonds,
                                                                             p_endpoint_bonds))
        if isomorphism is None:
            continue
        product_atom_to_reactant_atom = {product_atom: reactant_atom
                                         for reactant_atom, product_atom in enumerate(atom_map)}
        return {'ts_label': rxn.ts_species.label,
                'reactants': isomorphism,
                'products': [isomorphism[product_atom_to_reactant_atom[j]] for j in range(n_atoms)],
                'method': TS_ATOM_MAP_METHOD,
                'reactant_endpoint': r_endpoint,
                'ts_atom_order_follows_reactants': isomorphism == list(range(n_atoms)),
                }, None
    if not endpoints_perceived:
        logger.warning(f'The connectivity the IRC verdict perceived on the endpoints of {rxn.ts_species.label} is '
                       f'not that of the species of {rxn}, so no TS atom map is recorded.')
        return None, 'endpoint_perception_mismatch'
    logger.warning(f'The atom map of {rxn} contradicts the bonds the IRC endpoints of {rxn.ts_species.label} '
                   f'form and break, so no TS atom map is recorded. The atom map is left as it is.')
    return None, 'atom_map_contradicts_ts'


def are_geometries_congruent(reference_xyz: dict,
                             xyz: dict,
                             tolerance: float = IRC_START_GEOMETRY_TOLERANCE,
                             ) -> bool:
    """
    Check whether a geometry is congruent with a reference geometry, atom for atom.

    The two geometries must have the same element sequence. The isotopes of the reference are then given to the
    geometry (a parsed log states none, and the center of mass that ``kabsch`` rotates about is mass-weighted), and
    they are superimposed by ``kabsch`` on index-matched atoms, with no permutation of the atoms, by a proper rotation
    and a translation only (so a mirror image is not accepted). The root of the summed squared deviations it returns
    must not exceed ``tolerance``.

    Args:
        reference_xyz (dict): The reference geometry.
        xyz (dict): The geometry to compare.
        tolerance (float, optional): The largest root of the summed squared deviations in Angstrom.

    Returns:
        bool: Whether the geometry is congruent with the reference.
    """
    if tuple(reference_xyz['symbols']) != tuple(xyz['symbols']) or len(reference_xyz['coords']) != len(xyz['coords']):
        return False
    if reference_xyz.get('isotopes') is not None:
        xyz = dict(xyz, isotopes=reference_xyz['isotopes'])
    return bool(kabsch(reference_xyz, xyz) <= tolerance)


def get_irc_start_geometry_reason(ts_xyz: dict,
                                  irc_log_paths: list[str] | None,
                                  ) -> str | None:
    """
    Verify, from the logs, that every IRC was started from the TS geometry in the TS atom order, which is what makes
    atom ``i`` of an IRC endpoint atom ``i`` of the TS. The first geometry of each IRC log (``parse_irc_start_geometry``)
    is compared with the TS geometry atom for atom, without permuting atoms (see
    ``is_irc_start_geometry_the_ts_geometry``).

    Args:
        ts_xyz (dict): The TS geometry.
        irc_log_paths (list[str], optional): The IRC logs.

    Returns:
        str | None: ``None`` when every log starts from the TS geometry, ``'irc_start_geometry_unavailable'`` when
                    there is no log or the starting geometry of one cannot be read, and
                    ``'irc_start_geometry_differs'`` when one differs from the TS geometry.
    """
    paths = [path for path in (irc_log_paths or list()) if path]
    if not paths or len(paths) != len(irc_log_paths):
        return 'irc_start_geometry_unavailable'
    start_xyzs = list()
    for path in paths:
        try:
            start_xyz = parser.parse_irc_start_geometry(log_file_path=path) if os.path.isfile(path) else None
        except Exception as e:
            logger.warning(f'Could not read the starting geometry of the IRC log {path}, '
                           f'got:\n{e.__class__.__name__}: {e}')
            start_xyz = None
        if start_xyz is None:
            logger.warning(f'Could not read the starting geometry of the IRC log {path}, '
                           f'so no TS atom map is recorded.')
            return 'irc_start_geometry_unavailable'
        start_xyzs.append(start_xyz)
    for path, start_xyz in zip(paths, start_xyzs):
        if not are_geometries_congruent(ts_xyz, start_xyz):
            logger.warning(f'The IRC log {path} does not start from the TS geometry, '
                           f'so no TS atom map is recorded.')
            return 'irc_start_geometry_differs'
    return None


def get_irc_endpoint_chain_reason(endpoint_xyzs: dict[int, dict],
                                  endpoint_log_paths: list[str] | None,
                                  irc_log_paths: list[str] | None,
                                  ) -> str | None:
    """
    Verify, from the logs, that each IRC endpoint geometry follows from the IRC log of the same index, which is what
    keeps endpoint atom ``i`` the atom ``i`` that the IRC started from. ARC optimizes the geometry parsed from the
    last point of an IRC log (the IRC log of index ``k`` is the one whose geometry the species of endpoint ``k + 1``
    was created from) and passes the final geometry of that optimization to the IRC check.

    For each endpoint the optimization log's first geometry (``parse_irc_start_geometry``) must be congruent with
    the last geometry of the IRC log (``are_geometries_congruent``), and the endpoint geometry must be the final
    geometry of the optimization log, exactly.

    Args:
        endpoint_xyzs (dict[int, dict]): The two IRC endpoint geometries, keyed ``1`` and ``2``.
        endpoint_log_paths (list[str], optional): The optimization logs of the two endpoint species, in order.
        irc_log_paths (list[str], optional): The IRC logs, in the order of the endpoints.

    Returns:
        str | None: ``None`` when the chain holds, ``'irc_start_geometry_unavailable'`` when a log is missing or a
                    geometry cannot be read, and ``'irc_endpoint_geometry_differs'`` when a link of the chain breaks.
    """
    if not endpoint_log_paths or not irc_log_paths or len(endpoint_log_paths) != 2 or len(irc_log_paths) != 2 \
            or not all(endpoint_log_paths) or not all(irc_log_paths):
        return 'irc_start_geometry_unavailable'
    for k in (0, 1):
        endpoint_xyz = endpoint_xyzs[k + 1]
        try:
            paths_exist = os.path.isfile(endpoint_log_paths[k]) and os.path.isfile(irc_log_paths[k])
            final_xyz = parser.parse_geometry(log_file_path=endpoint_log_paths[k]) if paths_exist else None
            opt_start_xyz = parser.parse_irc_start_geometry(log_file_path=endpoint_log_paths[k]) \
                if paths_exist else None
            irc_last_xyz = parser.parse_geometry(log_file_path=irc_log_paths[k]) if paths_exist else None
        except Exception as e:
            logger.warning(f'Could not read the geometries of the logs of IRC endpoint {k + 1}, '
                           f'got:\n{e.__class__.__name__}: {e}')
            final_xyz = opt_start_xyz = irc_last_xyz = None
        if final_xyz is None or opt_start_xyz is None or irc_last_xyz is None:
            logger.warning(f'Could not read the geometries of the logs of IRC endpoint {k + 1}, '
                           f'so no TS atom map is recorded.')
            return 'irc_start_geometry_unavailable'
        if tuple(final_xyz['symbols']) != tuple(endpoint_xyz['symbols']) \
                or not np.array_equal(np.array(final_xyz['coords'], dtype=float),
                                      np.array(endpoint_xyz['coords'], dtype=float)):
            logger.warning(f'IRC endpoint {k + 1} is not the final geometry of its optimization log, '
                           f'so no TS atom map is recorded.')
            return 'irc_endpoint_geometry_differs'
        if not are_geometries_congruent(irc_last_xyz, opt_start_xyz):
            logger.warning(f'The optimization of IRC endpoint {k + 1} did not start from the last geometry of its '
                           f'IRC log, so no TS atom map is recorded.')
            return 'irc_endpoint_geometry_differs'
    return None


def _are_bond_graphs_isomorphic(symbols_1: list[str],
                                bonds_1: list[tuple[int, int]],
                                symbols_2: list[str],
                                bonds_2: list[tuple[int, int]],
                                ) -> bool:
    """
    Check whether two bond graphs, with the atoms labelled by element, are isomorphic.

    Args:
        symbols_1 (list[str]): The element of every atom of the first graph.
        bonds_1 (list[tuple[int, int]]): The bonds of the first graph.
        symbols_2 (list[str]): The element of every atom of the second graph.
        bonds_2 (list[tuple[int, int]]): The bonds of the second graph.

    Returns:
        bool: ``True`` if they are isomorphic.
    """
    graphs = list()
    for symbols, bonds in ((symbols_1, bonds_1), (symbols_2, bonds_2)):
        graph = nx.Graph()
        graph.add_nodes_from((index, {'element': symbol}) for index, symbol in enumerate(symbols))
        graph.add_edges_from(bonds)
        graphs.append(graph)
    return nx.is_isomorphic(graphs[0], graphs[1], node_match=lambda a, b: a['element'] == b['element'])


def _get_endpoint_bonds_from_fragments(xyz: dict,
                                       fragments: list[Molecule],
                                       ) -> list[tuple[int, int]] | None:
    """
    Get the connectivity of an IRC endpoint from the ``Molecule`` fragments the IRC verdict perceived on it, in the
    atom indices of the endpoint geometry. Bond orders are ignored.

    A perception may rebuild and reorder a ``Molecule``, so the atoms of a fragment are tied to the endpoint atoms by
    the exact equality of their coordinates, with no tolerance: every fragment atom must carry coordinates, equal to
    those of exactly one endpoint atom of the same element, and the fragment atoms together must cover every
    endpoint atom once.

    Args:
        xyz (dict): The Cartesian coordinates of the IRC endpoint.
        fragments (list[Molecule]): The perceived fragments.

    Returns:
        list[tuple[int, int]] | None: The bonds, each a sorted pair of 0-based endpoint atom indices, or ``None`` when
                                      the fragment atoms cannot be tied to the endpoint atoms one to one.
    """
    atoms_at = dict()
    for index, coords in enumerate(xyz['coords']):
        atoms_at.setdefault(tuple(float(c) for c in coords), list()).append(index)
    bonds, tied = list(), set()
    for mol in fragments:
        index_of = dict()
        for atom in mol.atoms:
            coords = getattr(atom, 'coords', None)
            if coords is None or len(coords) != 3:
                return None
            candidates = atoms_at.get(tuple(float(c) for c in coords), list())
            if len(candidates) != 1 or xyz['symbols'][candidates[0]] != atom.element.symbol \
                    or candidates[0] in tied:
                return None
            tied.add(candidates[0])
            index_of[id(atom)] = candidates[0]
        for atom in mol.atoms:
            for neighbor in atom.edges:
                if id(neighbor) not in index_of:
                    return None
                if index_of[id(atom)] < index_of[id(neighbor)]:
                    bonds.append((index_of[id(atom)], index_of[id(neighbor)]))
    return bonds if len(tied) == len(xyz['symbols']) else None


def _get_condensed_graph_of_reaction(symbols: list[str],
                                     bonds_a: list[tuple[int, int]],
                                     bonds_b: list[tuple[int, int]],
                                     ) -> nx.Graph:
    """
    Build the condensed graph of a reaction: the atoms as nodes labelled by element, and an edge for every bond that is
    in either list, labelled ``(in bonds_a, in bonds_b)``.

    Args:
        symbols (list[str]): The element symbol of every atom.
        bonds_a (list[tuple[int, int]]): The bonds of the first state (the reactants).
        bonds_b (list[tuple[int, int]]): The bonds of the second state (the products).

    Returns:
        nx.Graph: The graph, with the node attribute ``element`` and the edge attribute ``kind``.
    """
    set_a = {tuple(sorted(bond)) for bond in bonds_a}
    set_b = {tuple(sorted(bond)) for bond in bonds_b}
    graph = nx.Graph()
    graph.add_nodes_from((index, {'element': symbol}) for index, symbol in enumerate(symbols))
    for bond in sorted(set_a | set_b):
        graph.add_edge(*bond, kind=(bond in set_a, bond in set_b))
    return graph


def _find_cgr_isomorphism(graph_1: nx.Graph, graph_2: nx.Graph) -> list[int] | None:
    """
    Find a label-preserving isomorphism between two condensed graphs of reaction, preferring the identity.
    Only the first isomorphism is looked for; the automorphisms are never enumerated or counted.

    Args:
        graph_1 (nx.Graph): The graph whose nodes are mapped.
        graph_2 (nx.Graph): The graph they are mapped onto.

    Returns:
        list[int] | None: Entry ``i`` is the node of ``graph_2`` that node ``i`` of ``graph_1`` maps to,
                          or ``None`` if the graphs are not isomorphic.
    """
    n_atoms = graph_1.number_of_nodes()
    if n_atoms != graph_2.number_of_nodes() or graph_1.number_of_edges() != graph_2.number_of_edges():
        return None

    def _node_match(attributes_1: dict, attributes_2: dict) -> bool:
        return attributes_1['element'] == attributes_2['element']

    def _edge_match(attributes_1: dict, attributes_2: dict) -> bool:
        return attributes_1['kind'] == attributes_2['kind']

    if all(_node_match(graph_1.nodes[i], graph_2.nodes[i]) for i in range(n_atoms)) \
            and all(graph_2.has_edge(u, v) and _edge_match(data, graph_2.edges[u, v])
                    for u, v, data in graph_1.edges(data=True)):
        return list(range(n_atoms))
    matcher = nx.algorithms.isomorphism.GraphMatcher(graph_1, graph_2,
                                                     node_match=_node_match, edge_match=_edge_match)
    mapping = next(matcher.isomorphisms_iter(), None)
    return [mapping[i] for i in range(n_atoms)] if mapping is not None else None


def check_imaginary_frequencies(imaginary_freqs: list[float] | None) -> bool:
    """
    Check that the number of imaginary frequencies make sense.
    Theoretically, a TS should only have one "large" imaginary frequency,
    however additional imaginary frequency are allowed if they are very small in magnitude.
    This method does not consider the normal mode displacement check.

    Args:
        imaginary_freqs (list[float]): The imaginary frequencies of the TS guess after optimization.

    Returns:
        bool: Whether the imaginary frequencies make sense.
    """
    if imaginary_freqs is None:
        # Freqs haven't been calculated for this TS guess, do consider it as an optional candidate.
        return True
    if len(imaginary_freqs) == 0:
        return False
    if len(imaginary_freqs) == 1 \
            and LOWEST_MAJOR_TS_FREQ < abs(imaginary_freqs[0]) < HIGHEST_MAJOR_TS_FREQ:
        return True
    else:
        return len([im_freq for im_freq in imaginary_freqs if LOWEST_MAJOR_TS_FREQ < abs(im_freq) < HIGHEST_MAJOR_TS_FREQ]) == 1
