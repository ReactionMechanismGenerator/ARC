"""
A module for checking the quality of TS-related calculations, contains helper functions for Scheduler.
"""

import logging
import os

import numpy as np
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import arc.rmgdb as rmgdb
from arc import parser
from arc.common import (ARC_PATH,
                        convert_list_index_0_to_1,
                        extremum_list,
                        get_logger,
                        get_bonds_from_dmat,
                        read_yaml_file,
                        sum_list_entries,
                        )
from arc.imports import settings
from arc.species.converter import check_xyz_dict, displace_xyz, xyz_to_dmat
from arc.mapping.engine import (get_atom_indices_of_labeled_atoms_in_an_rmg_reaction,
                                get_rmg_reactions_from_arc_reaction,
                                )
from arc.statmech.factory import statmech_factory

if TYPE_CHECKING:
    from arc.job.adapter import JobAdapter
    from arc.level import Level
    from arc.species.species import ARCSpecies, TSGuess
    from arc.reaction import ARCReaction

logger = get_logger()

LOWEST_MAJOR_TS_FREQ, HIGHEST_MAJOR_TS_FREQ = settings['LOWEST_MAJOR_TS_FREQ'], settings['HIGHEST_MAJOR_TS_FREQ']


def check_ts(reaction: 'ARCReaction',
             job: Optional['JobAdapter'] = None,
             checks: Optional[List[str]] = None,
             rxn_zone_atom_indices: Optional[List[int]] = None,
             species_dict: Optional[dict] = None,
             project_directory: Optional[str] = None,
             kinetics_adapter: Optional[str] = None,
             output: Optional[dict] = None,
             sp_level: Optional['Level'] = None,
             freq_scale_factor: float = 1.0,
             verbose: bool = True,
             ):
    """
    Check the TS in terms of energy, normal mode displacement, and IRC.
    Populates the ``TS.ts_checks`` dictionary.
    Note that the 'freq' check is done in Scheduler.check_negative_freq() and not here.

    Todo:
        check IRC

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        job (JobAdapter, optional): The frequency job object instance.
        checks (List[str], optional): Specific checks to run. Optional values: 'energy', 'freq', 'IRC', 'rotors'.
        rxn_zone_atom_indices (List[int], optional): The 0-indices of atoms identified by the normal mode displacement
                                                     as the reaction zone. Automatically determined if not given.
        species_dict (dict, optional): The Scheduler species dictionary.
        project_directory (str, optional): The path to ARC's project directory.
        kinetics_adapter (str, optional): The statmech software to use for kinetic rate coefficient calculations.
        output (dict, optional): The Scheduler output dictionary.
        sp_level (Level, optional): The single-point energy level of theory.
        freq_scale_factor (float, optional): The frequency scaling factor.
        verbose (bool, optional): Whether to print logging messages.
    """
    checks = checks or list()
    for entry in checks:
        if entry not in ['energy', 'freq', 'IRC', 'rotors']:
            raise ValueError(f"Requested checks could be 'energy', 'freq', 'IRC', or 'rotors', got:\n{checks}")

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
        check_rxn_e0(reaction=reaction, verbose=verbose)
        if reaction.ts_species.ts_checks['E0'] is None and not reaction.ts_species.ts_checks['e_elect']:
            check_rxn_e_elect(reaction=reaction, verbose=verbose)

    if 'freq' in checks or (not reaction.ts_species.ts_checks['normal_mode_displacement'] and job is not None):
        try:
            check_normal_mode_displacement(reaction, job=job)
        except (ValueError, KeyError) as e:
            logger.error(f'Could not check normal mode displacement, got: \n{e}')
            reaction.ts_species.ts_checks['normal_mode_displacement'] = True

    if 'rotors' in checks or (ts_passed_all_checks(species=reaction.ts_species, exemptions=['E0', 'warnings', 'IRC'])
                              and job is not None):
        invalidate_rotors_with_both_pivots_in_a_reactive_zone(reaction, job,
                                                              rxn_zone_atom_indices=rxn_zone_atom_indices)


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


def check_rxn_e_elect(reaction: 'ARCReaction',
                      verbose: bool = True,
                      ) -> None:
    """
    Check that the TS electronic energy is above both reactant and product wells in a ``reaction``.
    Sets the respective energy parameter 'e_elect' in the ``TS.ts_checks`` dictionary.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        verbose (bool, optional): Whether to print logging messages.
    """
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
    if 'Could not determine TS e_elect relative to the wells; ' not in reaction.ts_species.ts_checks['warnings']:
        reaction.ts_species.ts_checks['warnings'] += 'Could not determine TS e_elect relative to the wells; '


def compute_rxn_e0(reaction: 'ARCReaction',
                   species_dict: dict,
                   project_directory: str,
                   kinetics_adapter: str,
                   output: dict,
                   sp_level: 'Level',
                   freq_scale_factor: float = 1.0,
                   ) -> Optional['ARCReaction']:
    """
    Checking the E0 values between wells and a TS in a ``reaction`` using ZPE from statmech.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        species_dict (dict): The Scheduler species dictionary.
        project_directory (str): The path to ARC's project directory.
        kinetics_adapter (str): The statmech software to use for kinetic rate coefficient calculations.
        output (dict): The Scheduler output dictionary.
        sp_level (Level): The single-point energy level of theory.
        freq_scale_factor (float, optional): The frequency scaling factor.

    Returns:
        Optional['ARCReaction']: A copy of the reaction object with E0 values populated.
    """
    if any(val is None for val in [species_dict, project_directory, kinetics_adapter,
                                   output, sp_level, freq_scale_factor]):
        return None
    for spc in reaction.r_species + reaction.p_species + [reaction.ts_species]:
        folder = 'rxns' if species_dict[spc.label].is_ts else 'Species'
        freq_path = os.path.join(project_directory, 'output', folder, spc.label, 'geometry', 'freq.out')
        if not spc.yml_path and not os.path.isfile(freq_path) and not species_dict[spc.label].is_monoatomic():
            return None
    considered_labels = list()
    rxn_copy = reaction.copy()
    for species in rxn_copy.r_species + rxn_copy.p_species + [rxn_copy.ts_species]:
        if species.label in considered_labels or species.e0:
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
    return rxn_copy


def check_rxn_e0(reaction: 'ARCReaction',
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
            logger.warnign('TS energy gas relative to one fo the wells is lower than 1 kJ/mol, skipping this TS')
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


def check_normal_mode_displacement(reaction: 'ARCReaction',
                                   job: Optional['JobAdapter'],
                                   amplitudes: Optional[Union[float, List[float]]] = None,
                                   ):
    """
    Check the normal mode displacement by identifying bonds that break and form
    and comparing them to the expected RMG template, if available.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        job (JobAdapter): The frequency job object instance.
        amplitudes (Union[float, List[float]], optional): The factor(s) multiplication for the displacement.
    """
    if job is None:
        return
    if reaction.family is None:
        rmgdb.determine_family(reaction)
    amplitudes = amplitudes or [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    amplitudes = [amplitudes] if isinstance(amplitudes, float) else amplitudes
    reaction.ts_species.ts_checks['normal_mode_displacement'] = False
    rmg_reactions = get_rmg_reactions_from_arc_reaction(arc_reaction=reaction) or list()
    freqs, normal_modes_disp = parser.parse_normal_mode_displacement(path=job.local_path_to_output_file, raise_error=False)
    if not len(normal_modes_disp):
        return
    largest_neg_freq_idx = get_index_of_abs_largest_neg_freq(freqs)
    bond_lone_hs = any(len(spc.mol.atoms) == 2 and spc.mol.atoms[0].element.symbol == 'H'
                       and spc.mol.atoms[0].element.symbol == 'H' for spc in reaction.r_species + reaction.p_species)
    # bond_lone_hs = False
    xyz = parser.parse_xyz_from_file(job.local_path_to_output_file)
    if not xyz['coords']:
        xyz = reaction.ts_species.get_xyz()

    done = False
    for amplitude in amplitudes:
        xyz_1, xyz_2 = displace_xyz(xyz=xyz, displacement=normal_modes_disp[largest_neg_freq_idx], amplitude=amplitude)
        dmat_1, dmat_2 = xyz_to_dmat(xyz_1), xyz_to_dmat(xyz_2)
        dmat_bonds_1 = get_bonds_from_dmat(dmat=dmat_1,
                                           elements=xyz_1['symbols'],
                                           tolerance=1.5,
                                           bond_lone_hydrogens=bond_lone_hs)
        dmat_bonds_2 = get_bonds_from_dmat(dmat=dmat_2,
                                           elements=xyz_2['symbols'],
                                           tolerance=1.5,
                                           bond_lone_hydrogens=bond_lone_hs)
        got_expected_changing_bonds = False
        for i, rmg_reaction in enumerate(rmg_reactions):
            r_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=reaction,
                                                                                rmg_reaction=rmg_reaction)[0]
            if r_label_dict is None:
                continue
            expected_breaking_bonds, expected_forming_bonds = reaction.get_expected_changing_bonds(r_label_dict=r_label_dict)
            if expected_breaking_bonds is None or expected_forming_bonds is None:
                continue
            got_expected_changing_bonds = True
            breaking = [determine_changing_bond(bond, dmat_bonds_1, dmat_bonds_2) for bond in expected_breaking_bonds]
            forming = [determine_changing_bond(bond, dmat_bonds_1, dmat_bonds_2) for bond in expected_forming_bonds]
            if len(breaking) and len(forming) \
                    and not any(entry is None for entry in breaking) and not any(entry is None for entry in forming) \
                    and all(entry == breaking[0] for entry in breaking) and all(entry == forming[0] for entry in forming) \
                    and breaking[0] != forming[0]:
                reaction.ts_species.ts_checks['normal_mode_displacement'] = True
                done = True
                break
        if not got_expected_changing_bonds and not reaction.ts_species.ts_checks['normal_mode_displacement']:
            reaction.ts_species.ts_checks['warnings'] += 'Could not compare normal mode displacement to expected ' \
                                                         'breaking/forming bonds due to a missing RMG template; '
            reaction.ts_species.ts_checks['normal_mode_displacement'] = True
            break
        if not len(rmg_reactions):
            # Just check that some bonds break/form, and that this is not a torsional saddle point.
            warning = f'Cannot check normal mode displacement for reaction {reaction} since a corresponding ' \
                      f'RMG template could not be generated'
            logger.warning(warning)
            reaction.ts_species.ts_checks['warnings'] += warning + '; '
            if any(bond not in dmat_bonds_2 for bond in dmat_bonds_1) \
                    or any(bond not in dmat_bonds_1 for bond in dmat_bonds_2):
                reaction.ts_species.ts_checks['normal_mode_displacement'] = True
                break
        if done:
            break


def determine_changing_bond(bond: Tuple[int, ...],
                            dmat_bonds_1: List[Tuple[int, int]],
                            dmat_bonds_2: List[Tuple[int, int]],
                            ) -> Optional[str]:
    """
    Determine whether a bond breaks or forms in a TS.
    Note that ``bond`` and all bond entries in `dmat_bonds_1/2`` must be already sorted from small to large indices.

    Args:
        bond (Tuple[int]): The atom indices describing the bond.
        dmat_bonds_1 (List[Tuple[int, int]]): The bonds perceived from dmat_1.
        dmat_bonds_2 (List[Tuple[int, int]]): The bonds perceived from dmat_2.

    Returns:
        Optional[bool]:
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


def invalidate_rotors_with_both_pivots_in_a_reactive_zone(reaction: 'ARCReaction',
                                                          job: 'JobAdapter',
                                                          rxn_zone_atom_indices: Optional[List[int]] = None,
                                                          ):
    """
    Invalidate rotors in which both pivots are included in the reactive zone.

    Args:
        reaction (ARCReaction): The respective reaction object instance.
        job (JobAdapter): The frequency job object instance.
        rxn_zone_atom_indices (List[int], optional): The 0-indices of atoms identified by the normal mode displacement
                                                     as the reaction zone. Automatically determined if not given.
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
    freqs, normal_mode_disp = parser.parse_normal_mode_displacement(path=job.local_path_to_output_file,
                                                                    raise_error=False)
    normal_mode_disp_rms = get_rms_from_normal_mode_disp(normal_mode_disp, freqs, reaction=reaction)
    num_of_atoms = get_expected_num_atoms_with_largest_normal_mode_disp(normal_mode_disp_rms=normal_mode_disp_rms,
                                                                        ts_guesses=reaction.ts_species.ts_guesses,
                                                                        reaction=reaction) \
                   + round(reaction.ts_species.number_of_atoms ** 0.25)  # Peripheral atoms might get in the way
    indices = sorted(range(len(normal_mode_disp_rms)), key=lambda i: normal_mode_disp_rms[i], reverse=True)[:num_of_atoms]
    return indices


def get_rms_from_normal_mode_disp(normal_mode_disp: np.ndarray,
                                  freqs: np.ndarray,
                                  reaction: Optional['ARCReaction'] = None,
                                  ) -> List[float]:
    """
    Get the root mean squares of the normal mode displacements.
    Use atom mass weights if ``reaction`` is given.

    Args:
        normal_mode_disp (np.ndarray): The normal mode displacement array.
        freqs (np.ndarray): Entries are frequency values.
        reaction (ARCReaction): The respective reaction object instance.

    Returns:
        List[float]: The RMS of the normal mode displacements.
    """
    mode_index = get_index_of_abs_largest_neg_freq(freqs)
    nmd = normal_mode_disp[mode_index]
    masses = reaction.get_element_mass() if reaction is not None else [1] * len(nmd)
    rms = list()
    for i, entry in enumerate(nmd):
        rms.append(((entry[0] ** 2 + entry[1] ** 2 + entry[2] ** 2) ** 0.5) * masses[i] ** 0.55)
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


def get_expected_num_atoms_with_largest_normal_mode_disp(normal_mode_disp_rms: List[float],
                                                         ts_guesses: List['TSGuess'],
                                                         reaction: Optional['ARCReaction'] = None,
                                                         ) -> int:
    """
    Get the number of atoms that are expected to have the largest normal mode displacement for the TS
    (considering all families). This is a wrapper for ``get_rxn_normal_mode_disp_atom_number()``.
    It is theoretically possible that TSGuesses of the same species will belong to different families.

    Args:
        normal_mode_disp_rms (List[float]): The RMS of the normal mode displacements.
        ts_guesses (List[TSGuess]): The TSGuess objects of a TS species.
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


def get_rxn_normal_mode_disp_atom_number(rxn_family: Optional[str] = None,
                                         reaction: Optional['ARCReaction'] = None,
                                         rms_list: Optional[List[float]] = None,
                                         ) -> int:
    """
    Get the number of atoms expected to have the largest normal mode displacement per family.
    If ``rms_list`` is given, also include atoms with an RMS value close to the lowest RMS still considered.

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


def check_irc_species_and_rxn(xyz_1: dict,
                              xyz_2: dict,
                              rxn: Optional['ARCReaction'],
                              ):
    """
    Check that the two species that result from optimizing the outputs of two IRC runs
    correspond to the desired reactants and products of the corresponding reaction.

    Args:
        xyz_1 (dict): The coordinates of IRS species 1.
        xyz_2 (dict): The coordinates of IRS species 2.
        rxn (ARCReaction): The corresponding reaction object instance.
    """
    if rxn is None:
        return None
    rxn.ts_species.ts_checks['IRC'] = False
    xyz_1, xyz_2 = check_xyz_dict(xyz_1), check_xyz_dict(xyz_2)
    dmat_1, dmat_2 = xyz_to_dmat(xyz_1), xyz_to_dmat(xyz_2)
    dmat_bonds_1 = get_bonds_from_dmat(dmat=dmat_1,
                                       elements=xyz_1['symbols'],
                                       tolerance=1.5,
                                       bond_lone_hydrogens=False,
                                       )
    dmat_bonds_2 = get_bonds_from_dmat(dmat=dmat_2,
                                       elements=xyz_2['symbols'],
                                       tolerance=1.5,
                                       bond_lone_hydrogens=False,
                                       )
    r_bonds, p_bonds = rxn.get_bonds()
    if _check_equal_bonds_list(dmat_bonds_1, r_bonds) and _check_equal_bonds_list(dmat_bonds_2, p_bonds) \
            or _check_equal_bonds_list(dmat_bonds_2, r_bonds) and _check_equal_bonds_list(dmat_bonds_1, p_bonds):
        rxn.ts_species.ts_checks['IRC'] = True


def _check_equal_bonds_list(bonds_1: List[Tuple[int, int]],
                            bonds_2: List[Tuple[int, int]],
                            ) -> bool:
    """
    Check whether two lists of bonds are equal.

    Args:
        bonds_1 (List[Tuple[int, int]]): List 1 of bonds.
        bonds_2 (List[Tuple[int, int]]): List 2 of bonds.

    Returns:
        bool: Whether the two lists of bonds are equal.
    """
    if len(bonds_1) != len(bonds_2):
        return False
    if all(bond in bonds_2 for bond in bonds_1):
        return True
    return False


def check_imaginary_frequencies(imaginary_freqs: Optional[List[float]]) -> bool:
    """
    Check that the number of imaginary frequencies make sense.
    Theoretically, a TS should only have one "large" imaginary frequency,
    however additional imaginary frequency are allowed if they are very small in magnitude.
    This method does not consider the normal mode displacement check.

    Args:
        imaginary_freqs (List[float]): The imaginary frequencies of the TS guess after optimization.

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
