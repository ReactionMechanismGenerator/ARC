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
                        )
from arc.imports import settings
from arc.species.converter import displace_xyz, xyz_to_dmat
from arc.species.mapping import (get_atom_indices_of_labeled_atoms_in_an_rmg_reaction,
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
             verbose: bool = True,
             job: Optional['JobAdapter'] = None,
             checks: Optional[List[str]] = None,
             rxn_zone_atom_indices: Optional[List[int]] = None,
             ):
    """
    Check the TS in terms of energy, normal mode displacement, and IRC.
    Populates the ``TS.ts_checks`` dictionary.
    Note that the 'freq' check is done in Scheduler.check_negative_freq() and not here.

    Todo:
        check IRC

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        verbose (bool, optional): Whether to print logging messages.
        job (JobAdapter, optional): The frequency job object instance.
        checks (List[str], optional): Specific checks to run. Optional values: 'energy', 'freq', 'IRC', 'rotors'.
        rxn_zone_atom_indices (List[int], optional): The 0-indices of atoms identified by the normal mode displacement
                                                     as the reaction zone. Automatically determined if not given.
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


def check_ts_energy(reaction: 'ARCReaction',
                    verbose: bool = True,
                    ) -> None:
    """
    Check that the TS electronic energy is above both reactant and product wells in a ``reaction``.
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
            logger.info(
                f'\nReaction {reaction.label} (TS {reaction.ts_label}) has the following path electronic energy:\n'
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
    # We don't really know.
    reaction.ts_species.ts_checks['e_elect'] = None
    reaction.ts_species.ts_checks['warnings'] += 'Could not determine TS e_elect relative to the wells; '


def check_rxn_e0(reaction: 'ARCReaction',
                 species_dict: dict,
                 project_directory: str,
                 kinetics_adapter: str,
                 output: dict,
                 sp_level: 'Level',
                 freq_scale_factor: float = 1.0,
                 ) -> Optional[bool]:
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
        Optional[bool]: Whether the test failed and the scheduler should switch to a different TS guess,
                        ``None`` if the test couldn't be performed.
    """
    for spc_label in reaction.reactants + reaction.products + [reaction.ts_label]:
        folder = 'rxns' if species_dict[spc_label].is_ts else 'Species'
        freq_path = os.path.join(project_directory, 'output', folder, spc_label, 'geometry', 'freq.out')
        if not os.path.isfile(freq_path) and not species_dict[spc_label].is_monoatomic():
            return None
    considered_labels = list()
    rxn_copy = reaction.copy()
    for species in rxn_copy.r_species + rxn_copy.p_species:
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
    if rxn_copy.kinetics is None:
        if rxn_copy.ts_species.ts_guesses_exhausted:
            return False
        return True  # Switch TS.
    reaction.ts_species.ts_checks['E0'] = True
    reaction.kinetics = None
    return False  # Don't switch TS.


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
            if not any(entry is None for entry in breaking) and not any(entry is None for entry in forming) \
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
    Use atom mass weights if `reaction` is given.

    Args:
        normal_mode_disp (np.ndarray): The normal mode displacement array.
        freqs (np.ndarray): Entries are frequency values.
        reaction (ARCReaction): The respective reaction object instance.

    Returns:
        List[float]: The RMS of the normal mode displacements.
    """
    rms, masses = list(), list()
    mode_index = get_index_of_abs_largest_neg_freq(freqs)
    nmd = normal_mode_disp[mode_index]
    if reaction is not None:
        for reactant in reaction.r_species:
            for atom in reactant.mol.atoms:
                masses.append(atom.element.mass)
    else:
        masses = [1] * len(nmd)
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


def check_imaginary_frequencies(imaginary_freqs: Optional[List[float]]) -> bool:
    """
    Check that the number of imaginary frequencies make sense.
    Theoretically a TS should only have one imaginary frequency,
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
        # Freqs have been calculated, and there are no imaginary frequencies.
        return False
    if len(imaginary_freqs) == 1 \
            and LOWEST_MAJOR_TS_FREQ < abs(imaginary_freqs[0]) < HIGHEST_MAJOR_TS_FREQ:
        # Freqs have been calculated, and there is only one imaginary frequency within the right range.
        return True
    else:
        # Freqs have been calculated, and there are several imaginary frequencies.
        num_major_freqs = 0
        for im_freq in imaginary_freqs:
            if 25 < abs(im_freq) < 10000:
                num_major_freqs += 1
        if num_major_freqs == 1:
            # Only one major imaginary frequency.
            return True
        return False
