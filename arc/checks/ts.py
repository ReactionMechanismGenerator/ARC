"""
A module for checking the quality of TS-related calculations, contains helper functions for Scheduler.
"""

import logging
import os

import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from rmgpy.exceptions import ActionError

import arc.rmgdb as rmgdb
from arc import parser
from arc.common import (ARC_PATH,
                        convert_list_index_0_to_1,
                        extremum_list,
                        get_logger,
                        read_yaml_file,
                        )
from arc.species.mapping import find_equivalent_atoms_in_reactants
from arc.statmech.factory import statmech_factory

if TYPE_CHECKING:
    from arc.job.adapter import JobAdapter
    from arc.level import Level
    from arc.species.species import ARCSpecies, TSGuess
    from arc.reaction import ARCReaction

logger = get_logger()


def check_ts(reaction: 'ARCReaction',
             verbose: bool = True,
             job: Optional['JobAdapter'] = None,
             checks: Optional[List[str]] = None,
             ):
    """
    Check the TS in terms of energy, normal mode displacement, and IRC.
    Populates the ``TS.ts_checks`` dictionary.
    Note that the 'freq' check is done in Scheduler.check_negative_freq() and not here.

    Todo:
        check IRC
        add tests

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        verbose (bool, optional): Whether to print logging messages.
        parameter (str, optional): The energy parameter to consider ('E0' or 'e_elect').
        job (JobAdapter, optional): The frequency job object instance.
        checks (List[str], optional): Specific checks to run. Optional values: 'energy', 'freq', 'IRC', 'rotors'.
    """
    checks = checks or list()
    for entry in checks:
        if entry not in ['energy', 'freq', 'IRC', 'rotors']:
            raise ValueError(f"Requested checks could be 'energy', 'freq', 'IRC', or 'rotors', got:\n{checks}")

    if 'energy' in checks or not reaction.ts_species.ts_checks['e_elect']:
        check_ts_energy(reaction=reaction, verbose=verbose)

    if 'freq' in checks or (not reaction.ts_species.ts_checks['normal_mode_displacement'] and job is not None):
        check_normal_mode_displacement(reaction, job=job)

    # if 'IRC' in checks or (not self.ts_species.ts_checks['IRC'] and IRC_wells is not None):
    #     self.check_irc()

    if 'rotors' in checks or (ts_passed_all_checks(species=reaction.ts_species, exemptions=['E0', 'warnings', 'IRC'])
                              and job is not None):
        invalidate_rotors_with_both_pivots_in_a_reactive_zone(reaction, job)


def ts_passed_all_checks(species: 'ARCSpecies',
                         exemptions: Optional[List[str]] = None,
                         verbose: bool = False,
                         ) -> bool:
    """
    Check whether the TS species passes all checks other than ones specified in ``exemptions``.

    Args:
        species (ARCSpecies): The TS species.
        exemptions (List[str], optional): Keys of the TS.ts_checks dict to pass.
        verbose (bool, optional): Whether to log findings.

    Returns:
        bool: Whether the TS species passed all checks.
    """
    exemptions = exemptions or list()
    for check, value in species.ts_checks.items():
        if check not in exemptions and not value and not (check == 'e_elect' and species.ts_checks['E0']):
            if verbose:
                logger.warning(f'TS {species.label} did not pass the all checks, status is:\n{species.ts_checks}')
            return False
    return True


def check_ts_energy(reaction: 'ARCReaction',
                    verbose: bool = True,
                    ) -> None:
    """
    Check that the TS electronic energy is above both reactant and product wells.
    Sets the respective energy parameter 'e_elect' in the ``TS.ts_checks`` dictionary.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        verbose (bool, optional): Whether to print logging messages.
    """
    r_e_elect = None if any([spc.e_elect is None for spc in reaction.r_species]) \
        else sum(spc.e_elect * reaction.get_species_count(species=spc, well=0) for spc in reaction.r_species)
    p_e_elect = None if any([spc.e_elect is None for spc in reaction.p_species]) \
        else sum(spc.e_elect * reaction.get_species_count(species=spc, well=1) for spc in reaction.p_species)
    ts_e_elect = reaction.ts_species.e_elect
    min_e = extremum_list([r_e_elect, p_e_elect, ts_e_elect], return_min=True)

    if any([val is not None for val in [r_e_elect, p_e_elect, ts_e_elect]]):
        if verbose:
            r_text = f'{r_e_elect - min_e:.2f} kJ/mol' if r_e_elect is not None else 'None'
            ts_text = f'{ts_e_elect - min_e:.2f} kJ/mol' if ts_e_elect is not None else 'None'
            p_text = f'{p_e_elect - min_e:.2f} kJ/mol' if p_e_elect is not None else 'None'
            logger.info(f'\nReaction {reaction.label} (TS {reaction.ts_label}) has the following path electronic energy:\n'
                        f'Reactants: {r_text}\n'
                        f'TS: {ts_text}\n'
                        f'Products: {p_text}')

        if all([val is not None for val in [r_e_elect, p_e_elect, ts_e_elect]]):
            # We have all params, we can make a quantitative decision.
            if ts_e_elect > r_e_elect and ts_e_elect > p_e_elect:
                # TS is above both wells.
                reaction.ts_species.ts_checks['e_elect'] = True
                return
            # TS is not above both wells.
            if verbose:
                logger.error(f'TS of reaction {reaction.label} has a lower electronic energy value than expected.')
                reaction.ts_species.ts_checks['e_elect'] = False
                return
    # We don't have any params (they are all ``None``)
    if verbose:
        logger.info('\n')
        logger.error(f"Could not get electronic energy of all species in reaction {reaction.label}. Cannot check TS.\n")
    # We don't really know, assume ``True``
    reaction.ts_species.ts_checks['e_elect'] = True
    reaction.ts_species.ts_checks['warnings'] += 'Could not determine TS energy relative to the wells; '


def check_rxn_e0(reaction: 'ARCReaction',
                 species_dict: dict,
                 project_directory: str,
                 kinetics_adapter: str,
                 output: dict,
                 sp_level: 'Level',
                 freq_scale_factor: float = 1.0,
                 ):
    """
    A helper function for checking a reaction E0 between wells and a TS using ZPE from statmech.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        species_dict (dict): The Scheduler species dictionary.
        project_directory (str): The path to ARC's project directory.
        kinetics_adapter (str): The statmech software to use for kinetic rate coefficient calculations.
        output (dict): The Scheduler output dictionary.
        sp_level (Level): The single-point level of theory.
        freq_scale_factor (float, optional): The frequency scaling factor.
    """
    for spc_label in reaction.reactants + reaction.products + [reaction.ts_label]:
        folder = 'rxns' if species_dict[spc_label].is_ts else 'Species'
        freq_path = os.path.join(project_directory, 'output', folder, spc_label, 'geometry', 'freq.out')
        if not os.path.isfile(freq_path):
            return None
    considered_labels = list()
    for species in reaction.r_species + reaction.p_species:
        if species.label in considered_labels:
            continue
        considered_labels.append(species.label)
        statmech_adapter = statmech_factory(statmech_adapter_label=kinetics_adapter,
                                            output_directory=os.path.join(project_directory, 'output'),
                                            output_dict=output,
                                            bac_type=None,
                                            sp_level=sp_level,
                                            freq_scale_factor=freq_scale_factor,
                                            species=species,
                                            )
        statmech_adapter.compute_thermo(kinetics_flag=True,
                                        e0_only=True,
                                        skip_rotors=True,
                                        )
    statmech_adapter = statmech_factory(statmech_adapter_label=kinetics_adapter,
                                        output_directory=os.path.join(project_directory, 'output'),
                                        output_dict=output,
                                        bac_type=None,
                                        sp_level=sp_level,
                                        freq_scale_factor=freq_scale_factor,
                                        reaction=reaction,
                                        species_dict=species_dict,
                                        T_min=(500, 'K'),
                                        T_max=(1000, 'K'),
                                        T_count=6,
                                        )
    statmech_adapter.compute_high_p_rate_coefficient(skip_rotors=True,
                                                     estimate_dh_rxn=True,
                                                     verbose=True,
                                                     )
    if reaction.kinetics is None:
        if reaction.ts_species.ts_guesses_exhausted:
            return False
        return True  # Switch TS.
    reaction.ts_species.ts_checks['E0'] = True
    reaction.kinetics = None
    return False  # Don't switch TS.


def check_normal_mode_displacement(reaction: 'ARCReaction',
                                   job: 'JobAdapter',
                                   rxn_zone_atom_indices: Optional[List[int]] = None,
                                   ):
    """
    Check the normal mode displacement by making sure that the atom indices derived from the major motion
    (the major normal mode displacement) fit the expected RMG reaction template.
    Note that RMG does not differentiate well between hydrogen atoms since it thinks in 2D,
    and therefore we must consider symmetry/degeneracy as facilitated by find_equivalent_atoms_in_reactants().

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        job (JobAdapter): The frequency job object instance.
        rxn_zone_atom_indices (List[int], optional): The 0-indices of atoms identified by the normal displacement
                                                     mode as the reaction zone. Automatically determined if not given.
    """
    if job is None:
        return

    if reaction.family is None:
        rmgdb.determine_family(reaction)

    rxn_zone_atom_indices = rxn_zone_atom_indices or get_rxn_zone_atom_indices(reaction, job)
    reaction.ts_species.ts_checks['normal_mode_displacement'] = False
    rmg_rxn = reaction.rmg_reaction.copy()

    try:
        reaction.family.add_atom_labels_for_reaction(reaction=rmg_rxn, output_with_resonance=False, save_order=True)
    except (ActionError, AttributeError, ValueError):
        reaction.ts_species.ts_checks['warnings'] += 'Could not determine atom labels from RMG, ' \
                                                     'cannot check normal mode displacement; '
        reaction.ts_species.ts_checks['normal_mode_displacement'] = True
    else:
        equivalent_indices = find_equivalent_atoms_in_reactants(arc_reaction=reaction)
        found_positions = list()
        for rxn_zone_atom_index in rxn_zone_atom_indices:
            atom_found = False
            for i, entry in enumerate(equivalent_indices):
                if rxn_zone_atom_index in entry and i not in found_positions:
                    atom_found = True
                    found_positions.append(i)
                    break
            if not atom_found:
                break
        else:
            reaction.ts_species.ts_checks['normal_mode_displacement'] = True


def invalidate_rotors_with_both_pivots_in_a_reactive_zone(reaction: 'ARCReaction',
                                                          job: 'JobAdapter',
                                                          rxn_zone_atom_indices: Optional[List[int]] = None,
                                                          ):
    """
    Invalidate rotors in which both pivots are included in the reactive zone.

    Args:
        reaction (ARCReaction): The respective reaction object instance.
        job (JobAdapter): The frequency job object instance.
        rxn_zone_atom_indices (List[int], optional): The 0-indices of atoms identified by the normal displacement
                                                     mode as the reaction zone. Automatically determined if not given.
    """
    rxn_zone_atom_indices = rxn_zone_atom_indices or get_rxn_zone_atom_indices(reaction, job)
    if not reaction.ts_species.rotors_dict:
        reaction.ts_species.determine_rotors()
    rxn_zone_atom_indices_1 = convert_list_index_0_to_1(rxn_zone_atom_indices)
    for key, rotor in reaction.ts_species.rotors_dict.items():
        if rotor['pivots'][0] in rxn_zone_atom_indices_1 and rotor['pivots'][1] in rxn_zone_atom_indices_1:
            rotor['success'] = False
            if 'pivTS' not in rotor['invalidation_reason']:
                rotor['invalidation_reason'] += 'Pivots participate in the TS reaction zone (code: pivTS). '
                logging.info(f"\nNot considering rotor {key} with pivots {rotor['pivots']} in TS {reaction.ts_species.label}\n")


def get_rxn_zone_atom_indices(reaction: 'ARCReaction',
                              job: 'JobAdapter',
                              ) -> List[int]:
    """
    Get the reaction zone atom indices by parsing normal mode displacement.

    Args:
        reaction (ARCReaction): The respective reaction object instance.
        job (JobAdapter): The frequency job object instance.

    Returns:
        List[int]: The indices of the atoms participating in the reaction.
                   The indices are 0-indexed and sorted in an increasing order.
    """
    h_abstractor_index = None
    if reaction.family is not None and reaction.family.label == 'H_Abstraction' \
            and any(spc.number_of_heavy_atoms == 0 for spc in reaction.r_species + reaction.p_species):
        for i, reactant in enumerate(reaction.r_species):
            if reactant.number_of_heavy_atoms == 0 and reactant.number_of_atoms == 1:
                h_abstractor_index = 0 if i == 0 else reaction.r_species[0].number_of_atoms
        if h_abstractor_index is None:
            for i, product in enumerate(reaction.p_species):
                if product.number_of_heavy_atoms == 0 and product.number_of_atoms == 1:
                    h_abstractor_index = reaction.atom_map.index(0) if i == 0 \
                        else reaction.atom_map.index(reaction.p_species[0].number_of_atoms)
    freqs, normal_mode_disp = parser.parse_normal_mode_displacement(path=job.local_path_to_output_file,
                                                                    raise_error=False)
    normal_disp_mode_rms = get_rms_from_normal_mode_disp(normal_mode_disp, freqs)
    element_weighted_normal_disp_mode_rms = list()
    r_i, r_atoms_i = 0, 0
    for i in range(len(normal_disp_mode_rms)):
        if i >= sum([len(reaction.r_species[j].mol.atoms) for j in range(r_i + 1)]):
            r_i += 1
            r_atoms_i = 0
        entry = reaction.r_species[r_i].mol.atoms[r_atoms_i].element.mass \
            if h_abstractor_index is None or i != h_abstractor_index else 1
        entry *= normal_disp_mode_rms[i]
        element_weighted_normal_disp_mode_rms.append(entry)
        r_atoms_i += 1
    num_of_atoms = get_expected_num_atoms_with_largest_normal_mode_disp(normal_disp_mode_rms=normal_disp_mode_rms,
                                                                        ts_guesses=reaction.ts_species.ts_guesses,
                                                                        reaction=reaction,
                                                                        )
    return sorted(range(len(element_weighted_normal_disp_mode_rms)),
                  key=lambda i: element_weighted_normal_disp_mode_rms[i], reverse=True)[:num_of_atoms]


def get_rms_from_normal_mode_disp(normal_mode_disp: np.ndarray,
                                  freqs: np.ndarray,
                                  ) -> List[float]:
    """
    Get the root mean squares of the normal displacement modes.

    Args:
        normal_mode_disp (np.ndarray): The normal displacement modes array.
        freqs (np.ndarray): Entries are frequency values.

    Returns:
        List[float]: The RMS of the normal displacement modes.
    """
    rms = list()
    mode_index = get_index_of_abs_largest_neg_freq(freqs)
    nmd = normal_mode_disp[mode_index]
    for entry in nmd:
        rms.append((entry[0] ** 2 + entry[1] ** 2 + entry[2] ** 2) ** 0.5)
    return rms


def get_index_of_abs_largest_neg_freq(freqs: np.ndarray) -> Optional[int]:
    """
    Get the index of the |largest| negative frequency.

    Args:
        freqs (np.ndarray): Entries are frequency values.

    Returns:
        Optional[int]: The 0-index of the largest absolute negative frequency.
    """
    if not len(freqs) or all(freq > 0 for freq in freqs):
        return None
    return list(freqs).index(min(freqs))


def get_expected_num_atoms_with_largest_normal_mode_disp(normal_disp_mode_rms: List[float],
                                                         ts_guesses: List['TSGuess'],
                                                         reaction: Optional['ARCReaction'] = None,
                                                         ) -> int:
    """
    Get the number of atoms that are expected to have the largest normal mode displacement for the TS
    (considering all families). This is a wrapper for ``get_rxn_normal_mode_disp_atom_number()``.
    It is theoretically possible that TSGuesses of the same species will belong to different families.

    Args:
        normal_disp_mode_rms (List[float]): The RMS of the normal displacement modes.
        ts_guesses (List[TSGuess]): The TSGuess objects of a TS species.
        reaction (ARCReaction): The respective reaction object instance.

    Returns:
        int: The number of atoms to consider that have a significant motions in the normal mode displacement.
    """
    families = list(set([tsg.family for tsg in ts_guesses]))
    num_of_atoms = max([get_rxn_normal_mode_disp_atom_number(rxn_family=family,
                                                             reaction=reaction,
                                                             rms_list=normal_disp_mode_rms,
                                                             )
                        for family in families])
    return num_of_atoms


def get_rxn_normal_mode_disp_atom_number(rxn_family: Optional[str] = None,
                                         reaction: Optional['ARCReaction'] = None,
                                         rms_list: Optional[List[float]] = None,
                                         ) -> int:
    """
    Get the number of atoms expected to have the largest normal mode displacement per family.
    If ``rms_list`` is given, also include atoms with an rms value close to the lowest rms still considered.

    Args:
        rxn_family (str, optional): The reaction family label.
        reaction (ARCReaction, optional): The reaction object instance.
        rms_list (List[float], optional): The root mean squares of the normal mode displacements.

    Raises:
        TypeError: If ``rms_list`` is not ``None`` and is either not a list or does not contain floats.

    Returns:
        int: The respective number of atoms.
    """
    default = 3
    if rms_list is not None \
            and (not isinstance(rms_list, list) or not all(isinstance(entry, float) for entry in rms_list)):
        raise TypeError(f'rms_list must be a non empty list, got {rms_list} of type {type(rms_list)}.')
    family = rxn_family
    if family is None and reaction is not None and reaction.family is not None:
        family = reaction.family.label
    if family is None:
        logger.warning(f'Cannot deduce a reaction family for {reaction}, assuming {default} atoms in the reaction zone.')
        return default
    content = read_yaml_file(os.path.join(ARC_PATH, 'data', 'rxn_normal_mode_disp.yml'))
    number_by_family = content.get(rxn_family, default)
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
