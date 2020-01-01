#!/usr/bin/env python3
# encoding: utf-8

"""
A module for (non-TS) species conformer generation

Note:
    variables that contain atom indices such as torsions and tops are 1-indexed,
    while atoms in Molecules are 0-indexed.

Todo:
    * Consider boat-chair conformers (https://en.wikipedia.org/wiki/Cyclohexane_conformation)
    * finally, consider h-bonds
    * Does it take the scan energy into account when generating combinations??
    * The secretary problem - incorporate for stochastic searching
    * What's the confirmed bottleneck?

conformers is a list of dictionaries, each with the following keys::

    {'xyz': <dict>,
     'index': <int>,
     'FF energy': <float>,
     'source': <str>,
     'torsion_dihedrals': {<torsion tuple 0>: angle 0,
                           <torsion tuple 1>: angle 1,
     }


Module workflow::

    generate_conformers
        generate_force_field_conformers
            get_force_field_energies, rdkit_force_field or mix_rdkit_and_openbabel_force_field, determine_dihedrals
        deduce_new_conformers
            get_torsion_angles, determine_torsion_symmetry, determine_torsion_sampling_points,
            change_dihedrals_and_force_field_it
        get_lowest_confs

"""

import copy
import logging
import sys
import time
from itertools import product

import openbabel as ob
import pybel as pyb
from rdkit import Chem
from rdkit.Chem.rdchem import EditableMol as RDMol

import rmgpy.molecule.group as gr
from rmgpy.exceptions import ILPSolutionError, ResonanceError
from rmgpy.molecule.converter import to_ob_mol
from rmgpy.molecule.molecule import Atom, Bond, Molecule
from rmgpy.molecule.element import C as C_ELEMENT, H as H_ELEMENT, F as F_ELEMENT, Cl as Cl_ELEMENT, I as I_ELEMENT

from arc.common import logger, calculate_dihedral_angle
from arc.exceptions import ConformerError, InputError
import arc.plotter
from arc.species import converter
from arc.species import vectors


# The number of conformers to generate per range of heavy atoms in the molecule
# (will be increased if there are chiral centers)
CONFS_VS_HEAVY_ATOMS = {(0, 3): 75,
                        (4, 9): 500,
                        (10, 29): 1000,
                        (30, 59): 2500,
                        (60, 99): 5000,
                        (100, 'inf'): 7500,
                        }

# The number of conformers to generate per range of potential torsions in the molecule
# (will be increased if there are chiral centers)
CONFS_VS_TORSIONS = {(0, 1): 75,
                     (2, 5): 500,
                     (5, 19): 1000,
                     (20, 34): 2500,
                     (35, 49): 5000,
                     (50, 'inf'): 7500,
                     }

# The resolution (in degrees) for scanning smeared wells
SMEARED_SCAN_RESOLUTIONS = 30.0

# The number of conformers to return. Will be iteratively checked for consistency. The rest will be written to a file.
NUM_CONFS_TO_RETURN = 5

# An energy threshold (in kJ/mol) above which wells in a torsion will not be considered (rel. to the most stable well)
DE_THRESHOLD = 5.

# The gap (in degrees) that defines different wells
WELL_GAP = 20

# The maximum number of times to iteratively search for the lowest conformer
MAX_COMBINATION_ITERATIONS = 25

# A threshold below which all combinations will be generated. Above it just samples of the entire search space.
COMBINATION_THRESHOLD = 1000


def generate_conformers(mol_list, label, xyzs=None, torsions=None, tops=None, charge=0, multiplicity=None,
                        num_confs=None, num_confs_to_return=None, de_threshold=None, smeared_scan_res=None,
                        combination_threshold=None, force_field='MMFF94s', max_combination_iterations=None,
                        diastereomers=None, return_all_conformers=False, plot_path=None, print_logs=True):
    """
    Generate conformers for (non-TS) species starting from a list of RMG Molecules.
    (resonance structures are assumed to have already been generated and included in the molecule list)

    Args:
        mol_list (list or Molecule): Molecule objects to consider (or Molecule, resonance structures will be generated).
        label (str): The species' label.
        xyzs (list), optional: A list of user guess xyzs that will also be taken into account, each in a dict format.
        torsions (list, optional): A list of all possible torsions in the molecule. Will be determined if not given.
        tops (list, optional): A list of tops corresponding to torsions. Will be determined if not given.
        charge (int, optional): The species charge. Used to perceive a molecule from xyz.
        multiplicity (int, optional): The species multiplicity. Used to perceive a molecule from xyz.
        num_confs (int, optional): The number of conformers to generate. Determined automatically if not given.
        num_confs_to_return (int, optional): The number of conformers to return.
        well_tolerance (float, optional): The required precision (in degrees) around which to center a well's mean.
        de_threshold (float, optional): Energy threshold (in kJ/mol) above which wells will not be considered.
        smeared_scan_res (float, optional): The resolution (in degrees) for scanning smeared wells.
        combination_threshold (int, optional): A threshold below which all combinations will be generated.
        force_field (str, optional): The type of force field to use (MMFF94, MMFF94s, UFF, GAFF, fit).
                                     'fit' will first run MMFF94, than fit a custom Amber FF to the species.
        max_combination_iterations (int, optional): The maximum number of times to iteratively search
                                                    for the lowest conformer.
        diastereomers (list, optional): Entries are xyz's in a dictionary format or conformer structures
                                        representing specific diastereomers to keep.
        return_all_conformers (bool, optional): Whether to return the full conformers list of conformer dictionaries
                                                In addition to the lowest conformers list. Tru to return it.
        plot_path (str, optional): A folder path in which the plot will be saved.
                                   If None, the plot will not be shown (nor saved).
        print_logs (bool, optional): Whether define a logger so logs are also printed to stdout.
                                     Useful when run outside of ARC. True to print.

    Returns:
        list: Lowest conformers (number of entries is num_confs_to_return times the number of enantiomer combinations)

    Raises:
        ConformerError: If something goes wrong.
        TypeError: If xyzs has entries of a wrong type.
    """
    if len(mol_list[0].atoms) == 1:
        # this is a mono-atomic species
        element_symbol = mol_list[0].atoms[0].element.symbol
        confs = [{'xyz': {'symbols': (element_symbol,),
                          'isotopes': (converter.get_most_common_isotope_for_element(element_symbol),),
                          'coords': ((0.0, 0.0, 0.0),)},
                  'zmat': {'symbols': (element_symbol,), 'coords': ((None, None, None),), 'vars': {}, 'map': {0: 0}},
                  'index': 0,
                  'FF energy': 0.0,
                  'chirality': None,
                  'source': 'mono atomic species',
                  'torsion_dihedrals': None,
                   }]
        if not return_all_conformers:
            return confs
        else:
            return confs, confs
    if xyzs is not None and any([not isinstance(xyz, dict) for xyz in xyzs]):
        raise TypeError("xyz entries of xyzs must be dictionaries, e.g.:\n\n"
                        "{{'symbols': ('O', 'C', 'H', 'H'),\n'isotopes': (16, 12, 1, 1),\n"
                        "'coords': ((0.0, 0.0, 0.678514),\n           (0.0, 0.0, -0.532672),\n"
                        "           (0.0, 0.935797, -1.116041),\n           (0.0, -0.935797, -1.116041))}}\n\n"
                        "Got {0}".format([type(xyz) for xyz in xyzs]))
    if print_logs:
        initialize_log()
    t0 = time.time()
    logger.info('Generating conformers for {0}'.format(label))

    num_confs_to_return = num_confs_to_return or NUM_CONFS_TO_RETURN
    max_combination_iterations = max_combination_iterations or MAX_COMBINATION_ITERATIONS
    combination_threshold = combination_threshold or COMBINATION_THRESHOLD

    if isinstance(mol_list, Molecule):
        # try generating resonance structures, but strictly keep atom order
        success = False
        try:
            new_mol_list = mol_list.copy(deep=True).generate_resonance_structures(keep_isomorphic=False,
                                                                                  filter_structures=True)
            success = converter.order_atoms_in_mol_list(ref_mol=mol_list[0].copy(deep=True), mol_list=new_mol_list)
        except (ValueError, ILPSolutionError, ResonanceError) as e:
            logger.warning(f'Could not generate resonance structures for species {label}. Got: {e}')
        if success:
            mol_list = new_mol_list
        else:
            mol_list = [mol_list]
    if not isinstance(mol_list, list):
        logger.error('The `mol_list` argument must be a list, got {0}'.format(type(mol_list)))
        return None
    for mol in mol_list:
        if not isinstance(mol, Molecule):
            raise ConformerError('Each entry in the `mol_list` argument must be an RMG Molecule object, '
                                 'got {0}'.format(type(mol)))
    mol_list = [update_mol(mol) for mol in mol_list]

    if torsions is None or tops is None:
        torsions, tops = determine_rotors(mol_list)
    conformers = generate_force_field_conformers(
        mol_list=mol_list, label=label, xyzs=xyzs, torsion_num=len(torsions), charge=charge, multiplicity=multiplicity,
        num_confs=num_confs, force_field=force_field)

    conformers = determine_dihedrals(conformers, torsions)

    new_conformers = deduce_new_conformers(
        label, conformers, torsions, tops, mol_list, smeared_scan_res, plot_path=plot_path,
        combination_threshold=combination_threshold, force_field=force_field,
        max_combination_iterations=max_combination_iterations, diastereomers=diastereomers, de_threshold=de_threshold)

    new_conformers = determine_chirality(conformers=new_conformers, label=label, mol=mol_list[0])

    num_confs_to_return = min(num_confs_to_return, len(new_conformers))  # don't return more than we have
    lowest_confs = get_lowest_confs(label, new_conformers, n=num_confs_to_return)

    lowest_confs.sort(key=lambda x: x['FF energy'], reverse=False)  # sort by output confs from lowest to highest energy
    indices_to_pop = list()
    for i, conf in enumerate(lowest_confs):
        if i and compare_xyz(conf['xyz'], lowest_confs[i-1]['xyz']):
            indices_to_pop.append(i)
    for i in reversed(range(len(lowest_confs))):  # pop from the end, so other indices won't change
        if i in indices_to_pop:
            lowest_confs.pop(i)

    execution_time = time.time() - t0
    t, s = divmod(execution_time, 60)
    t, m = divmod(t, 60)
    d, h = divmod(t, 24)
    days = '{0} days and '.format(int(d)) if d else ''
    time_str = '{days}{hrs:02d}:{min:02d}:{sec:02d}'.format(days=days, hrs=int(h), min=int(m), sec=int(s))
    if execution_time > 10:
        logger.info('Conformer execution time using {0}: {1}'.format(force_field, time_str))

    if not return_all_conformers:
        return lowest_confs
    else:
        return lowest_confs, conformers


def deduce_new_conformers(label, conformers, torsions, tops, mol_list, smeared_scan_res=None, plot_path=None,
                          combination_threshold=1000, force_field='MMFF94s', max_combination_iterations=25,
                          diastereomers=None, de_threshold=None):
    """
    By knowing the existing torsion wells, get the geometries of all important conformers.
    Validate that atoms don't collide in the generated conformers (don't consider ones where they do).

    Args:
        label (str): The species' label.
        conformers (list): Entries are conformer dictionaries.
        torsions (list): A list of all possible torsion angles in the molecule, each torsion angles list is sorted.
        tops (list): A list of tops corresponding to torsions.
        mol_list (list): A list of RMG Molecule objects.
        smeared_scan_res (float, optional): The resolution (in degrees) for scanning smeared wells.
        plot_path (str, optional): A folder path in which the plot will be saved.
                                   If None, the plot will not be shown (nor saved).
        combination_threshold (int, optional): A threshold below which all combinations will be generated.
        force_field (str, optional): The type of force field to use.
        max_combination_iterations (int, optional): The max num of times to iteratively search for the lowest conformer.
        diastereomers (list, optional): Entries are xyz's in a dictionary format or conformer structures
                                        representing specific diastereomers to keep.
        de_threshold (float, optional): An energy threshold (in kJ/mol) above which wells in a torsion
                                        will not be considered.

    Returns:
        list: The deduced conformers.
    """
    smeared_scan_res = smeared_scan_res or SMEARED_SCAN_RESOLUTIONS
    if not any(['torsion_dihedrals' in conformer for conformer in conformers]):
        conformers = determine_dihedrals(conformers, torsions)
    torsion_angles = get_torsion_angles(label, conformers, torsions)  # get all wells per torsion
    mol = mol_list[0]

    symmetries = dict()
    for torsion, top in zip(torsions, tops):
        # identify symmetric torsions so we don't bother considering them in the conformational combinations
        symmetry = determine_torsion_symmetry(label, top, mol_list, torsion_angles[tuple(torsion)])
        symmetries[tuple(torsion)] = symmetry
    logger.debug('Identified {0} symmetric wells for {1}'.format(len([s for s in symmetries.values() if s > 1]), label))

    torsions_sampling_points, wells_dict = dict(), dict()
    for tor, tor_angles in torsion_angles.items():
        torsions_sampling_points[tor], wells_dict[tor] = \
            determine_torsion_sampling_points(label, tor_angles, smeared_scan_res=smeared_scan_res,
                                              symmetry=symmetries[tor])

    if plot_path is not None:
        arc.plotter.plot_torsion_angles(torsion_angles, torsions_sampling_points, wells_dict=wells_dict,
                                        plot_path=plot_path)

    hypothetical_num_comb = 1
    for points in torsions_sampling_points.values():
        hypothetical_num_comb *= len(points)
    number_of_chiral_centers = get_number_of_chiral_centers(label, mol, conformer=conformers[0],
                                                            just_get_the_number=True)
    hypothetical_num_comb *= 2 ** number_of_chiral_centers
    if hypothetical_num_comb > 1000:
        hypothetical_num_comb_str = '{0:.2E}'.format(hypothetical_num_comb)
    else:
        hypothetical_num_comb_str = str(hypothetical_num_comb)
    logger.info(f'\nHypothetical number of conformer combinations for {label}: {hypothetical_num_comb_str}')

    # split torsions_sampling_points into two lists, use combinations only for those with multiple sampling points
    single_tors, multiple_tors, single_sampling_point, multiple_sampling_points = list(), list(), list(), list()
    multiple_sampling_points_dict = dict()  # used for plotting an energy "scan"
    for tor, points in torsions_sampling_points.items():
        if len(points) == 1:
            single_tors.append(tor)
            single_sampling_point.append((points[0]))
        else:
            multiple_sampling_points_dict[tor] = points
            multiple_tors.append(tor)
            multiple_sampling_points.append(points)

    diastereomeric_conformers = get_lowest_diastereomers(label=label, mol=mol, conformers=conformers,
                                                         diastereomers=diastereomers)
    new_conformers = list()
    for diastereomeric_conformer in diastereomeric_conformers:
        # set symmetric (single well) torsions to the mean of the well
        if 'chirality' in diastereomeric_conformer and diastereomeric_conformer['chirality'] != dict():
            logger.info(f"Considering diastereomer {diastereomeric_conformer['chirality']}")
        base_xyz = diastereomeric_conformer['xyz']  # base_xyz is modified within the loop below
        for torsion, dihedral in zip(single_tors, single_sampling_point):
            torsion_0_indexed = [tor - 1 for tor in torsion]
            conf, rd_mol = converter.rdkit_conf_from_mol(mol, base_xyz)
            if conf is not None:
                base_xyz = converter.set_rdkit_dihedrals(conf, rd_mol, torsion_0_indexed, deg_abs=dihedral)

        new_conformers.extend(generate_conformer_combinations(
            label=label, mol=mol_list[0], base_xyz=base_xyz, hypothetical_num_comb=hypothetical_num_comb,
            multiple_tors=multiple_tors, multiple_sampling_points=multiple_sampling_points,
            combination_threshold=combination_threshold, len_conformers=len(conformers), force_field=force_field,
            max_combination_iterations=max_combination_iterations, plot_path=plot_path, torsion_angles=torsion_angles,
            multiple_sampling_points_dict=multiple_sampling_points_dict, wells_dict=wells_dict,
            de_threshold=de_threshold))

    if plot_path is not None:
        lowest_conf = get_lowest_confs(label=label, confs=new_conformers, n=1)[0]
        lowest_conf = determine_chirality([lowest_conf], label, mol, force=False)[0]
        diastereomer = f" (diastereomer: {lowest_conf['chirality']})" if 'chirality' in lowest_conf \
                                                                         and lowest_conf['chirality'] else ''
        logger.info(f'Lowest force field conformer for {label}{diastereomer}:')
        logger.info(converter.xyz_to_str(lowest_conf['xyz']))
        arc.plotter.draw_structure(xyz=lowest_conf['xyz'])

    return new_conformers


def generate_conformer_combinations(label, mol, base_xyz, hypothetical_num_comb, multiple_tors,
                                    multiple_sampling_points, combination_threshold=1000, len_conformers=-1,
                                    force_field='MMFF94s', max_combination_iterations=25, plot_path=None,
                                    torsion_angles=None, multiple_sampling_points_dict=None, wells_dict=None,
                                    de_threshold=None):
    """
    Call either conformers_combinations_by_lowest_conformer() or generate_all_combinations(),
    according to the hypothetical_num_comb.

    Args:
        label (str): The species' label.
        mol (Molecule): The RMG molecule with the connectivity information.
        base_xyz (dict): The base 3D geometry to be changed.
        hypothetical_num_comb (int): The number of combinations that could be generated by changing dihedrals,
                                     considering symmetry but not considering atom collisions.
        combination_threshold (int, optional): A threshold below which all combinations will be generated.
        multiple_tors (list): Entries are torsion tuples of non-symmetric torsions.
        multiple_sampling_points (list): Entries are lists of dihedral angles (sampling points), respectively correspond
                                         to torsions in multiple_tors.
        len_conformers (int, optional): The length of the existing conformers list (for consecutive numbering).
        de_threshold (float, optional): An energy threshold (in kJ/mol) above which wells in a torsion
                                        will not be considered.
        force_field (str, optional): The type of force field to use.
        max_combination_iterations (int, optional): The max num of times to iteratively search for the lowest conformer.
        torsion_angles (dict, optional): The torsion angles. Keys are torsion tuples, values are lists of all
                                         corresponding angles from conformers.
        multiple_sampling_points_dict (dict, optional): Keys are torsion tuples, values are respective sampling points.
        wells_dict (dict, optional): Keys are torsion tuples, values are well dictionaries.
        plot_path (str, optional): A folder path in which the plot will be saved.
                                            If None, the plot will not be shown (nor saved).

    Returns:
        list: New conformer combinations, entries are conformer dictionaries.
    """
    de_threshold = de_threshold or DE_THRESHOLD
    if hypothetical_num_comb > combination_threshold:
        # don't generate all combinations, there are simply too many
        # iteratively modify the lowest conformer until it converges.
        logger.debug('hypothetical_num_comb for {0} is > {1}'.format(label, combination_threshold))
        new_conformers = conformers_combinations_by_lowest_conformer(
            label, mol=mol, base_xyz=base_xyz, multiple_tors=multiple_tors,
            multiple_sampling_points=multiple_sampling_points, len_conformers=len_conformers, force_field=force_field,
            plot_path=plot_path, de_threshold=de_threshold, max_combination_iterations=max_combination_iterations,
            torsion_angles=torsion_angles, multiple_sampling_points_dict=multiple_sampling_points_dict,
            wells_dict=wells_dict)
    else:
        # just generate all combinations and get their FF energies
        logger.debug('hypothetical_num_comb for {0} is < {1}'.format(label, combination_threshold))
        new_conformers = generate_all_combinations(label, mol, base_xyz, multiple_tors, multiple_sampling_points,
                                                   len_conformers=len_conformers, force_field=force_field,
                                                   torsions=list(torsion_angles.keys()))
    return new_conformers


def conformers_combinations_by_lowest_conformer(label, mol, base_xyz, multiple_tors, multiple_sampling_points,
                                                len_conformers=-1, force_field='MMFF94s', max_combination_iterations=25,
                                                torsion_angles=None, multiple_sampling_points_dict=None,
                                                wells_dict=None, de_threshold=None, plot_path=False):
    """
    Iteratively modify dihedrals in the lowest conformer (each iteration deduces a new lowest conformer),
    until convergence.

    Args:
        label (str): The species' label.
        mol (Molecule): The RMG molecule with the connectivity information.
        base_xyz (dict): The base 3D geometry to be changed.
        multiple_tors (list): Entries are torsion tuples of non-symmetric torsions.
        multiple_sampling_points (list): Entries are lists of dihedral angles (sampling points), respectively correspond
                                         to torsions in multiple_tors.
        len_conformers (int, optional): The length of the existing conformers list (for consecutive numbering).
        de_threshold (float, optional): An energy threshold (in kJ/mol) above which wells in a torsion
                                        will not be considered.
        force_field (str, optional): The type of force field to use.
        max_combination_iterations (int, optional): The max num of times to iteratively search for the lowest conformer.
        torsion_angles (dict, optional): The torsion angles. Keys are torsion tuples, values are lists of all
                                         corresponding angles from conformers.
        multiple_sampling_points_dict (dict, optional): Keys are torsion tuples, values are respective sampling points.
        wells_dict (dict, optional): Keys are torsion tuples, values are well dictionaries.
        plot_path (str, optional): A folder path in which the plot will be saved.
                                            If None, the plot will not be shown (nor saved).

    Returns:
        list: New conformer combinations, entries are conformer dictionaries.
    """
    base_energy = get_force_field_energies(label, mol, num_confs=None, xyz=base_xyz,
                                           force_field=force_field, optimize=True)[1][0]
    new_conformers = list()  # will be returned
    lowest_conf_i = None
    for i in range(max_combination_iterations):
        newest_conformers_dict, newest_conformer_list = dict(), list()  # conformers from the current iteration
        for tor, sampling_points in zip(multiple_tors, multiple_sampling_points):
            xyzs, energies = change_dihedrals_and_force_field_it(label, mol, xyz=base_xyz, torsions=[tor],
                                                                 new_dihedrals=[[sp] for sp in sampling_points],
                                                                 force_field=force_field, optimize=False)
            newest_conformers_dict[tor] = list()  # keys are torsions for plotting
            for xyz, energy, dihedral in zip(xyzs, energies, sampling_points):
                exists = False
                for conf in new_conformers + newest_conformer_list:
                    if compare_xyz(xyz, conf['xyz']):
                        exists = True
                        break
                if xyz is not None:
                    conformer = {'index': len_conformers + len(new_conformers) + len(newest_conformer_list),
                                 'xyz': xyz,
                                 'FF energy': round(energy, 3),
                                 'source': 'Changing dihedrals on most stable conformer, iteration {0}'.format(i),
                                 'torsion': tor,
                                 'dihedral': round(dihedral, 2)}
                    newest_conformers_dict[tor].append(conformer)
                    if not exists:
                        newest_conformer_list.append(conformer)
                else:
                    # if xyz is None, atoms have collided
                    logger.debug('\n\natoms colliding in {0} for torsion {1} and dihedral {2}:'.format(
                        label, tor, dihedral))
                    logger.debug(xyz)
                    logger.debug('\n\n')
        new_conformers.extend(newest_conformer_list)
        if not newest_conformer_list:
            newest_conformer_list = [lowest_conf_i]
        if force_field != 'gromacs':
            lowest_conf_i = get_lowest_confs(label, newest_conformer_list, n=1)[0]
            if lowest_conf_i['FF energy'] == base_energy \
                    and compare_xyz(lowest_conf_i['xyz'], base_xyz):
                break
            elif lowest_conf_i['FF energy'] < base_energy:
                base_energy = lowest_conf_i['FF energy']
    if plot_path is not None:
        logger.info(converter.xyz_to_str(lowest_conf_i['xyz']))
        arc.plotter.show_sticks(lowest_conf_i['xyz'])
        num_comb = arc.plotter.plot_torsion_angles(torsion_angles, multiple_sampling_points_dict,
                                                   wells_dict=wells_dict, e_conformers=newest_conformers_dict,
                                                   de_threshold=de_threshold, plot_path=plot_path)
        if num_comb is not None:
            if num_comb > 1000:
                num_comb_str = '{0:.2E}'.format(num_comb)
            else:
                num_comb_str = str(num_comb)
            logger.info('Number of conformer combinations for {0} after reduction: {1}'.format(label, num_comb_str))
    if de_threshold is not None:
        min_e = min([conf['FF energy'] for conf in new_conformers])
        new_conformers = [conf for conf in new_conformers if conf['FF energy'] - min_e < de_threshold]
    return new_conformers


def generate_all_combinations(label, mol, base_xyz, multiple_tors, multiple_sampling_points, len_conformers=-1,
                              torsions=None, force_field='MMFF94s'):
    """
    Generate all combinations of torsion wells from a base conformer.

    Args:
        label (str): The species' label.
        mol (Molecule): The RMG molecule with the connectivity information.
        base_xyz (dict): The base 3D geometry to be changed.
        multiple_tors (list): Entries are torsion tuples of non-symmetric torsions.
        multiple_sampling_points (list): Entries are lists of dihedral angles (sampling points), respectively correspond
                                         to torsions in multiple_tors.
        len_conformers (int, optional): The length of the existing conformers list (for consecutive numbering).
        force_field (str, optional): The type of force field to use.
        torsions (list, optional): A list of all possible torsions in the molecule. Will be determined if not given.

    Returns:
        list: New conformer combinations, entries are conformer dictionaries.
    """
    # generate sampling points combinations
    product_combinations = list(product(*multiple_sampling_points))
    new_conformers = list()  # will be returned

    if multiple_tors:
        xyzs, energies = change_dihedrals_and_force_field_it(label, mol, xyz=base_xyz, torsions=multiple_tors,
                                                             new_dihedrals=product_combinations, optimize=True,
                                                             force_field=force_field)
        for xyz, energy in zip(xyzs, energies):
            if xyz is not None:
                new_conformers.append({'index': len_conformers + len(new_conformers),
                                       'xyz': xyz,
                                       'FF energy': energy,
                                       'source': 'Generated all combinations from scan map'})
    else:
        # no multiple torsions (all torsions are symmetric or no torsions in the molecule), this is a trivial case
        energy = get_force_field_energies(label, mol, num_confs=None, xyz=base_xyz, force_field=force_field,
                                          optimize=True)[1][0]
        new_conformers.append({'index': len_conformers + len(new_conformers),
                               'xyz': base_xyz,
                               'FF energy': energy,
                               'source': 'Generated all combinations from scan map (trivial case)'})
    if torsions is None:
        torsions = determine_rotors([mol])
    new_conformers = determine_dihedrals(new_conformers, torsions)
    return new_conformers


def generate_force_field_conformers(label, mol_list, torsion_num, charge, multiplicity, xyzs=None, num_confs=None,
                                    force_field='MMFF94s'):
    """
    Generate conformers using RDKit and Open Babel and optimize them using a force field
    Also consider user guesses in `xyzs`

    Args:
        label (str): The species' label.
        mol_list (list): Entries are Molecule objects representing resonance structures of a chemical species.
        xyzs (list, optional): Entries are xyz coordinates in dict format, given as initial guesses.
        torsion_num (int): The number of torsions identified in the molecule.
        charge (int): The net charge of the species.
        multiplicity (int): The species spin multiplicity.
        num_confs (int, optional): The number of conformers to generate.
        force_field (str, optional): The type of force field to use.

    Returns:
        list: Entries are conformer dictionaries.

    Raises:
        ConformerError: If xyzs is given and it is not a list, or its entries are not strings.
    """
    conformers = list()
    number_of_heavy_atoms = len([atom for atom in mol_list[0].atoms if atom.is_non_hydrogen()])
    if num_confs is None:
        num_confs, num_chiral_centers = determine_number_of_conformers_to_generate(
            label=label, heavy_atoms=number_of_heavy_atoms, torsion_num=torsion_num, mol=mol_list[0],
            xyz=xyzs[0] if xyzs is not None else None)
    else:
        num_chiral_centers = ''
    chiral_centers = '' if not num_chiral_centers else f', {num_chiral_centers} chiral centers,'
    logger.info(f'Species {label} has {number_of_heavy_atoms} heavy atoms{chiral_centers} and {torsion_num} torsions. '
                f'Using {num_confs} random conformers.')
    for mol in mol_list:
        ff_xyzs, ff_energies = list(), list()
        try:
            ff_xyzs, ff_energies = get_force_field_energies(label, mol, num_confs=num_confs, force_field=force_field)
        except ValueError as e:
            logger.warning('Could not generate conformers for {0}, failed with: {1}'.format(label, e))
        if ff_xyzs:
            for xyz, energy in zip(ff_xyzs, ff_energies):
                conformers.append({'xyz': xyz,
                                   'index': len(conformers),
                                   'FF energy': energy,
                                   'source': force_field})
    # User guesses
    if xyzs is not None and xyzs:
        if not isinstance(xyzs, list):
            raise ConformerError('The xyzs argument must be a list, got {0}'.format(type(xyzs)))
        for xyz in xyzs:
            if not isinstance(xyz, dict):
                raise ConformerError('Each entry in xyzs must be a dictionary, got {0}'.format(type(xyz)))
            s_mol, b_mol = converter.molecules_from_xyz(xyz, multiplicity=multiplicity, charge=charge)
            conformers.append({'xyz': xyz,
                               'index': len(conformers),
                               'FF energy': get_force_field_energies(label, mol=b_mol or s_mol, xyz=xyz,
                                                                     optimize=True, force_field=force_field)[1][0],
                               'source': 'User Guess'})
    return conformers


def change_dihedrals_and_force_field_it(label, mol, xyz, torsions, new_dihedrals, optimize=True, force_field='MMFF94s'):
    """
    Change dihedrals of specified torsions according to the new dihedrals specified, and get FF energies.

    Example::

        torsions = [(1, 2, 3, 4), (9, 4, 7, 1)]
        new_dihedrals = [[90, 120], [90, 300], [180, 270], [30, 270]]

    This will calculate the energy of the original conformer (defined using `xyz`).
    We iterate through new_dihedrals. The torsions are set accordingly and the energy and xyz of the newly
    generated conformer are kept.

    We assume that each list entry in new_dihedrals is of the length of the torsions list (2 in the example).

    Args:
        label (str): The species' label.
        mol (Molecule): The RMG molecule with the connectivity information.
        xyz (dict): The base 3D geometry to be changed.
        torsions (list): Entries are torsion tuples for which the dihedral will be changed relative to xyz.
        new_dihedrals (list): Entries are same size lists of dihedral angles (floats) corresponding to the torsions.
        optimize (bool, optional): Whether to optimize the coordinates using FF. True to optimize.
        force_field (str, optional): The type of force field to use.

    Returns:
        list: The conformer FF energies corresponding to the list of dihedrals.
    Returns:
        list: The conformer xyz geometries corresponding to the list of dihedrals.
    """
    if isinstance(xyz, str):
        xyz = converter.str_to_xyz(xyz)

    if torsions is None or new_dihedrals is None:
        xyz, energy = get_force_field_energies(label, mol=mol, xyz=xyz, optimize=True, force_field=force_field)
        return xyz, energy

    xyzs, energies = list(), list()
    # make sure new_dihedrals is a list of lists (or tuples):
    if isinstance(new_dihedrals, (int, float)):
        new_dihedrals = [[new_dihedrals]]
    if isinstance(new_dihedrals, list) and not isinstance(new_dihedrals[0], (list, tuple)):
        new_dihedrals = [new_dihedrals]

    for dihedrals in new_dihedrals:
        xyz_dihedrals = xyz
        for torsion, dihedral in zip(torsions, dihedrals):
            conf, rd_mol = converter.rdkit_conf_from_mol(mol, xyz_dihedrals)
            if conf is not None:
                torsion_0_indexed = [tor -1 for tor in torsion]
                xyz_dihedrals = converter.set_rdkit_dihedrals(conf, rd_mol, torsion_0_indexed, deg_abs=dihedral)
        if force_field != 'gromacs':
            xyz_, energy = get_force_field_energies(label, mol=mol, xyz=xyz_dihedrals, optimize=True,
                                                    force_field=force_field)
            if energy and xyz_:
                energies.append(energy[0])
                if optimize:
                    xyzs.append(xyz_[0])
                else:
                    xyzs.append(xyz_dihedrals)
        else:
            energies.append(None)
            xyzs.append(xyz_dihedrals)
    return xyzs, energies


def determine_rotors(mol_list):
    """
    Determine possible unique rotors in the species to be treated as hindered rotors.

    Args:
        mol_list (list): Localized structures (Molecule objects) by which all rotors will be determined.

    Returns:
        list: A list of indices of scan pivots.
    Returns:
        list: A list of indices of top atoms (including one of the pivotal atoms) corresponding to the torsions.
    """
    torsions, tops = list(), list()
    for mol in mol_list:
        rotors = find_internal_rotors(mol)
        for new_rotor in rotors:
            for existing_torsion in torsions:
                if existing_torsion == new_rotor['scan']:
                    break
            else:
                torsions.append(new_rotor['scan'])
                tops.append(new_rotor['top'])
    return torsions, tops


def determine_number_of_conformers_to_generate(label, heavy_atoms, torsion_num, mol=None, xyz=None, minimalist=False):
    """
    Determine the number of conformers to generate using molecular mechanics

    Args:
        label (str): The species' label.
        heavy_atoms (int): The number of heavy atoms in the molecule.
        torsion_num (int): The number of potential torsions in the molecule.
        mol (Molecule, optional): The RMG Molecule object.
        xyz (dict, optional): The xyz coordinates.
        minimalist (bool, optional): Whether to return a small number of conformers, useful when this is just a guess
                                     before fitting a force field. True to be minimalistic.

    Returns:
        int: The number of conformers to generate.
    Returns:
        int: The number of chiral centers.

    Raises:
        ConformerError: If the number of conformers to generate cannot be determined.
    """
    if isinstance(torsion_num, list):
        torsion_num = len(torsion_num)

    for heavy_range, num_confs_1 in CONFS_VS_HEAVY_ATOMS.items():
        if heavy_range[1] == 'inf' and heavy_atoms >= heavy_range[0]:
            break
        elif heavy_range[1] >= heavy_atoms >= heavy_range[0]:
            break
    else:
        raise ConformerError('Could not determine the number of conformers to generate according to the number '
                             'of heavy atoms ({heavy}) in {label}. The CONFS_VS_HEAVY_ATOMS dictionary might be '
                             'corrupt, got:\n {d}'.format(heavy=heavy_atoms, label=label, d=CONFS_VS_HEAVY_ATOMS))

    for torsion_range, num_confs_2 in CONFS_VS_TORSIONS.items():
        if torsion_range[1] == 'inf' and torsion_num >= torsion_range[0]:
            break
        elif torsion_range[1] >= torsion_num >= torsion_range[0]:
            break
    else:
        raise ConformerError('Could not determine the number of conformers to generate according to the number '
                             'of torsions ({torsion_num}) in {label}. The CONFS_VS_TORSIONS dictionary might be '
                             'corrupt, got:\n {d}'.format(torsion_num=torsion_num, label=label, d=CONFS_VS_TORSIONS))

    if minimalist:
        num_confs = min(num_confs_1, num_confs_2, 250)
    else:
        num_confs = max(num_confs_1, num_confs_2)

    # increase the number of conformers if there are more than two chiral centers
    num_chiral_centers = 0
    if mol is None and xyz is not None:
        mol = converter.molecules_from_xyz(xyz)[1]
    if mol is not None and xyz is None:
        xyzs = get_force_field_energies(label, mol, num_confs=1)[0]
        xyz = xyzs[0] if len(xyzs) else None
    if mol is not None and xyz is not None:
        num_chiral_centers = get_number_of_chiral_centers(label, mol, xyz=xyz, just_get_the_number=True)
    if num_chiral_centers > 2:
        num_confs = int(num_confs * num_chiral_centers)

    return num_confs, num_chiral_centers


def determine_dihedrals(conformers, torsions):
    """
    For each conformer in `conformers` determine the respective dihedrals.

    Args:
        conformers (list): Entries are conformer dictionaries.
        torsions (list): All possible torsions in the molecule.

    Returns:
        list: Entries are conformer dictionaries.
    """
    for conformer in conformers:
        if isinstance(conformer['xyz'], str):
            xyz = converter.str_to_xyz(conformer['xyz'])
        else:
            xyz = conformer['xyz']
        if 'torsion_dihedrals' not in conformer or not conformer['torsion_dihedrals']:
            conformer['torsion_dihedrals'] = dict()
            for torsion in torsions:
                angle = calculate_dihedral_angle(coords=xyz['coords'], torsion=torsion)
                conformer['torsion_dihedrals'][tuple(torsion)] = angle
    return conformers


def determine_torsion_sampling_points(label, torsion_angles, smeared_scan_res=None, symmetry=1):
    """
    Determine how many points to consider in each well of a torsion for conformer combinations.

    Args:
        label (str): The species' label.
        torsion_angles (list): Well angles in the torsion.
        smeared_scan_res (float, optional): The resolution (in degrees) for scanning smeared wells.
        symmetry (int, optional): The torsion symmetry number.

    Returns:
        list: Sampling points for the torsion.
    Returns:
        list: Each entry is a well dictionary with the keys
        ``start_idx``, ``end_idx``, ``start_angle``, ``end_angle``, ``angles``.
    """
    smeared_scan_res = smeared_scan_res or SMEARED_SCAN_RESOLUTIONS
    sampling_points = list()
    wells = get_wells(label, torsion_angles, blank=20)
    for i, well in enumerate(wells):
        width = abs(well['end_angle'] - well['start_angle'])
        mean = sum(well['angles']) / len(well['angles'])
        if width <= 2 * smeared_scan_res:
            sampling_points.append(mean)
        else:
            num = int(width / smeared_scan_res)
            padding = abs(mean - well['start_angle'] - ((num - 1) * smeared_scan_res) / 2)
            sampling_points.extend([padding + well['angles'][0] + smeared_scan_res * j for j in range(int(num))])
        if symmetry > 1 and i == len(wells) / symmetry - 1:
            break
    return sampling_points, wells


def determine_torsion_symmetry(label, top1, mol_list, torsion_scan):
    """
    Check whether a torsion is symmetric.

    If a torsion well is "well defined" and not smeared, it could be symmetric.
    Check the groups attached to the rotor pivots to determine whether it is indeed symmetric
    We don't care about the actual rotor symmetry number here, since we plan to just use the first well
    (they're all the same).

    Args:
        label (str): The species' label.
        top1 (list): A list of atom indices on one side of the torsion, including the pivotal atom.
        mol_list (list): A list of molecules.
        torsion_scan (list): The angles corresponding to this torsion from all conformers.

    Returns:
        int: The rotor symmetry number.
    """
    symmetry = 1
    check_tops = [1, 1]  # flags for checking top1 and top2
    mol = mol_list[0]
    top2 = [i + 1 for i in range(len(mol.atoms)) if i + 1 not in top1]
    for j, top in enumerate([top1, top2]):
        # A quick bypass for methyl rotors which are too common:
        if len(top) == 4 and mol.atoms[top[0] - 1].is_carbon() \
                and all([mol.atoms[top[i] - 1].is_hydrogen() for i in range(1, 4)]):
            symmetry *= 3
            check_tops[j] = 0
        # A quick bypass for benzene rings:
        elif len(top) == 11 and sum([mol.atoms[top[i] - 1].is_carbon() for i in range(11)]) == 6 \
                and sum([mol.atoms[top[i] - 1].is_hydrogen() for i in range(11)]) == 5:
            symmetry *= 2
            check_tops[j] = 0
    # treat the torsion list as cyclic, search for at least two blank parts of at least 60 degrees each
    # if the means of all data parts of the scan are uniformly scattered, the torsion might be symmetric
    wells = get_wells(label=label, angles=torsion_scan, blank=60)

    distances, well_widths = list(), list()
    for i in range(len(wells)):
        well_widths.append(abs(wells[i]['end_angle'] - wells[i]['start_angle']))
        if i > 0:
            distances.append(int(round(abs(wells[i]['start_angle'] - wells[i - 1]['end_angle'])) / 10) * 10)
    mean_well_width = sum(well_widths) / len(well_widths)

    if len(wells) in [1, 2, 3, 4, 6, 9] and all([distance == distances[0] for distance in distances]) \
            and all([abs(width - mean_well_width) / mean_well_width < determine_well_width_tolerance(mean_well_width)
                     for width in well_widths]):
        # All well distances and widths are equal. The torsion scan might be symmetric, check the groups
        for j, top in enumerate([top1, top2]):
            if check_tops[j]:
                groups, grp_idx, groups_indices = list(), list(), list()
                for atom in mol.atoms[top[0] - 1].edges.keys():
                    if mol.vertices.index(atom) + 1 in top:
                        atom_indices = determine_top_group_indices(
                            mol=mol, atom1=mol.atoms[top[0] - 1], atom2=atom, index=0)[0]
                        groups.append(to_group(mol, atom_indices))
                        grp_idx.append(atom_indices)
                        groups_indices.append([g + 1 for g in atom_indices])
                # hard-coding for NO2/NS2 groups, since the two O or S atoms have different atom types in each localized
                # structure, hence are not isomorphic
                if len(top) == 3 and mol.atoms[top[0] - 1].atomtype.label == 'N5dc' \
                        and (all([mol.atoms[top[k] - 1].atomtype.label in ['O2d', 'O0sc'] for k in [1, 2]])
                             or all([mol.atoms[top[k] - 1].atomtype.label in ['S2d', 'S0sc'] for k in [1, 2]])):
                    symmetry *= 2
                # all other groups:
                elif not mol.atoms[top[0] - 1].lone_pairs > 0 and not mol.atoms[top[0] - 1].radical_electrons > 0 \
                        and all([groups[0].is_isomorphic(group, save_order=True) for group in groups[1:]]):
                    symmetry *= len(groups)
    return symmetry


def determine_well_width_tolerance(mean_width):
    """
    Determine the tolerance by which well widths are determined to be nearly equal.

    Fitted to a polynomial trend line for the following data of (mean, tolerance) pairs::

        (100, 0.11), (60, 0.13), (50, 0.15), (25, 0.25), (5, 0.50), (1, 0.59)

    Args:
        mean_width (float): The mean well width in degrees.

    Returns:
        float: The tolerance.
    """
    if mean_width > 100:
        return 0.1
    tol = -1.695e-10 * mean_width ** 5 + 6.209e-8 * mean_width ** 4 - 8.855e-6 * mean_width ** 3 \
        + 6.446e-4 * mean_width ** 2 - 2.610e-2 * mean_width + 0.6155
    return tol


def get_lowest_confs(label, confs, n=1, energy='FF energy'):
    """
    Get the most stable conformer

    Args:
        label (str): The species' label.
        confs (list): Entries are either conformer dictionaries or a length two list of xyz coordinates and energy.
        n (int): Number of lowest conformers to return.
        energy (str, optional): The energy attribute to search by. Currently only 'FF energy' is supported.

    Returns:
        list: Conformer dictionaries.
    """
    if not confs or confs is None:
        raise ConformerError('get_lowest_confs() got no conformers for {0}'.format(label))
    if isinstance(confs[0], list):
        conformer_list = list()
        for entry in confs:
            if entry[1] is not None:
                conformer_list.append({'xyz': entry[0], energy: entry[1]})
    elif isinstance(confs[0], dict):
        conformer_list = [conformer for conformer in confs if energy in conformer and conformer[energy] is not None]
    else:
        raise ConformerError("confs could either be a list of dictionaries or a list of lists. "
                             "Got a list of {0}'s for {1}".format(type(confs[0]), label))
    conformer_list.sort(key=lambda conformer: conformer[energy], reverse=False)
    n_lowest_confs = [conformer_list[0]]
    index = 1
    while n - 1 and index < len(conformer_list):
        if not compare_xyz(n_lowest_confs[-1]['xyz'], conformer_list[index]['xyz']):
            n_lowest_confs.append(conformer_list[index])
            n -= 1
        index += 1
    return n_lowest_confs


def get_torsion_angles(label, conformers, torsions):
    """
    Populate each torsion pivots with all available angles from the generated conformers

    Args:
        label (str): The species' label.
        conformers (list): The conformers from which to extract the angles.
        torsions (list): The torsions to consider.

    Returns:
        dict: The torsion angles. Keys are torsion tuples, values are lists of all corresponding angles from conformers.
    """
    torsion_angles = dict()
    if len(conformers) and not any(['torsion_dihedrals' in conformer for conformer in conformers]):
        raise ConformerError(f'Could not determine dihedral torsion angles for {label}. '
                             f'Consider calling `determine_dihedrals()` first.')
    for conformer in conformers:
        if 'torsion_dihedrals' in conformer and conformer['torsion_dihedrals']:
            for torsion in torsions:
                if tuple(torsion) not in torsion_angles:
                    torsion_angles[tuple(torsion)] = list()
                torsion_angles[tuple(torsion)].append(conformer['torsion_dihedrals'][tuple(torsion)])
    for tor in torsion_angles.keys():
        torsion_angles[tor].sort()
    return torsion_angles


def get_force_field_energies(label, mol, num_confs=None, xyz=None, force_field='MMFF94s',  optimize=True):
    """
    Determine force field energies using RDKit.
    If num_confs is given, random 3D geometries will be generated. If xyz is given, it will be directly used instead.
    The coordinates are returned in the order of atoms in mol.

    Args:
        label (str): The species' label.
        mol (Molecule): The RMG molecule object with connectivity and bond order information.
        num_confs (int, optional): The number of random 3D conformations to generate.
        xyz (dict, optional): The 3D coordinates guess.
        force_field (str, optional): The type of force field to use.
        optimize (bool, optional): Whether to first optimize the conformer using FF. True to optimize.

    Returns:
        list: Entries are xyz coordinates, each in a dict format.
    Returns:
        list: Entries are the FF energies (in kJ/mol).

    Raises:
        ConformerError: If conformers could not be generated.
    """
    xyzs, energies = list(), list()
    if force_field.lower() in ['mmff94', 'mmff94s', 'uff']:
        rd_mol = embed_rdkit(label, mol, num_confs=num_confs, xyz=xyz)
        xyzs, energies = rdkit_force_field(label, rd_mol, mol=mol, force_field=force_field, optimize=optimize)
    if not len(xyzs) and force_field.lower() in ['gaff', 'mmff94', 'mmff94s', 'uff', 'ghemical']:
        xyzs, energies = mix_rdkit_and_openbabel_force_field(label, mol, num_confs=num_confs, xyz=xyz,
                                                             force_field=force_field)
    if not len(xyzs):
        if force_field.lower() not in ['mmff94', 'mmff94s', 'uff', 'gaff', 'ghemical']:
            raise ConformerError(f'Unrecognized force field for {label}. Should be either MMFF94, MMFF94s, UFF, '
                                 f'Ghemical, or GAFF. Got: {force_field}.')
        raise ConformerError(f'Could not generate conformers for species {label}.')
    return xyzs, energies


def mix_rdkit_and_openbabel_force_field(label, mol, num_confs=None, xyz=None, force_field='GAFF'):
    """
    Optimize conformers using a force field (GAFF, MMFF94s, MMFF94, UFF, Ghemical)
    Use RDKit to generate the random conformers (open babel isn't good enough),
    but use open babel to optimize them (RDKit doesn't have GAFF)

    Args:
        label (str): The species' label.
        mol (Molecule, optional): The RMG molecule object with connectivity and bond order information.
        num_confs (int, optional): The number of random 3D conformations to generate.
        xyz (string or list, optional): The 3D coordinates in either a string or an array format.
        force_field (str, optional): The type of force field to use.

    Returns:
        list: Entries are optimized xyz's in a list format.
    Returns:
        list: Entries are float numbers representing the energies in kJ/mol.
    """
    xyzs, energies = list(), list()
    rd_mol = embed_rdkit(label, mol, num_confs=num_confs, xyz=xyz)
    unoptimized_xyzs = list()
    for i in range(rd_mol.GetNumConformers()):
        conf, xyz = rd_mol.GetConformer(i), list()
        for j in range(conf.GetNumAtoms()):
            pt = conf.GetAtomPosition(j)
            xyz.append([pt.x, pt.y, pt.z])
        xyz = [xyz[j] for j, _ in enumerate(xyz)]  # reorder
        unoptimized_xyzs.append(xyz)

    if not len(unoptimized_xyzs):
        # use OB as the fall back method
        logger.warning(f'Using OpenBable instead of RDKit as a fall back method to generate conformers for {label}. '
                       f'This is often slower, and prohibits ARC from using all features of the conformers module.')
        xyzs, energies = openbabel_force_field(label, mol, num_confs, force_field=force_field)

    else:
        for xyz in unoptimized_xyzs:
            xyzs_, energies_ = openbabel_force_field(label, mol, num_confs, xyz=xyz, force_field=force_field)
            xyzs.extend(xyzs_)
            energies.extend(energies_)
    return xyzs, energies


def openbabel_force_field(label, mol, num_confs=None, xyz=None, force_field='GAFF', method='diverse'):
    """
    Optimize conformers using a force field (GAFF, MMFF94s, MMFF94, UFF, Ghemical)

    Args:
        label (str): The species' label.
        mol (Molecule, optional): The RMG molecule object with connectivity and bond order information.
        num_confs (int, optional): The number of random 3D conformations to generate.
        xyz (dict, optional): The 3D coordinates.
        force_field (str, optional): The type of force field to use.
        method (str, optional): The conformer searching method to use in open babel.
                                         For method description, see http://openbabel.org/dev-api/group__conformer.shtml

    Returns:
        list: Entries are optimized xyz's in a list format.
    Returns:
        list: Entries are float numbers representing the energies in kJ/mol.
    """
    xyzs, energies = list(), list()
    ff = ob.OBForceField.FindForceField(force_field)

    if xyz is not None:
        # generate an open babel molecule
        obmol = ob.OBMol()
        atoms = mol.vertices
        ob_atom_ids = dict()  # dictionary of OB atom IDs
        for i, atom in enumerate(atoms):
            a = obmol.NewAtom()
            a.SetAtomicNum(atom.number)
            a.SetVector(xyz['coords'][i][0], xyz['coords'][i][1], xyz['coords'][i][2])
            if atom.element.isotope != -1:
                a.SetIsotope(atom.element.isotope)
            a.SetFormalCharge(atom.charge)
            ob_atom_ids[atom] = a.GetId()
        orders = {1: 1, 2: 2, 3: 3, 4: 4, 1.5: 5}
        for atom1 in mol.vertices:
            for atom2, bond in atom1.edges.items():
                if bond.is_hydrogen_bond():
                    continue
                index1 = atoms.index(atom1)
                index2 = atoms.index(atom2)
                if index1 < index2:
                    obmol.AddBond(index1 + 1, index2 + 1, orders[bond.order])

        # optimize
        ff.Setup(obmol)
        ff.SetLogLevel(0)
        ff.SetVDWCutOff(6.0)  # The VDW cut-off distance (default=6.0)
        ff.SetElectrostaticCutOff(10.0)  # The Electrostatic cut-off distance (default=10.0)
        ff.SetUpdateFrequency(10)  # The frequency to update the non-bonded pairs (default=10)
        ff.EnableCutOff(False)  # Use cut-off (default=don't use cut-off)
        # ff.SetLineSearchType('Newton2Num')
        ff.SteepestDescentInitialize()  # ConjugateGradientsInitialize
        v = 1
        while v:
            v = ff.SteepestDescentTakeNSteps(1)  # ConjugateGradientsTakeNSteps
            if ff.DetectExplosion():
                raise ConformerError('Force field {0} exploded with method {1} for {2}'.format(
                    force_field, 'SteepestDescent', label))
        ff.GetCoordinates(obmol)

    elif num_confs is not None:
        obmol, ob_atom_ids = to_ob_mol(mol, return_mapping=True)
        pybmol = pyb.Molecule(obmol)
        pybmol.make3D()
        obmol = pybmol.OBMol
        ff.Setup(obmol)

        if method.lower() == 'weighted':
            ff.WeightedRotorSearch(num_confs, 2000)
        elif method.lower() == 'random':
            ff.RandomRotorSearch(num_confs, 2000)
        elif method.lower() == 'diverse':
            rmsd_cutoff = 0.5
            energy_cutoff = 50.
            confab_verbose = False
            ff.DiverseConfGen(rmsd_cutoff, num_confs, energy_cutoff, confab_verbose)
        elif method.lower() == 'systematic':
            ff.SystematicRotorSearch(num_confs)
        else:
            raise ConformerError('Could not identify method {0} for {1}'.format(method, label))
    else:
        raise ConformerError('Either num_confs or xyz should be given for {0}'.format(label))

    ff.GetConformers(obmol)
    obconversion = ob.OBConversion()
    obconversion.SetOutFormat('xyz')

    for i in range(obmol.NumConformers()):
        obmol.SetConformer(i)
        ff.Setup(obmol)
        xyz_str = '\n'.join(obconversion.WriteString(obmol).splitlines()[2:])
        xyz_dict = converter.str_to_xyz(xyz_str)
        # reorder:
        xyz_dict['coords'] = tuple(xyz_dict['coords'][ob_atom_ids[mol.atoms[j]]]
                                   for j in range(len(xyz_dict['coords'])))
        xyzs.append(xyz_dict)
        energies.append(ff.Energy())
    return xyzs, energies


def embed_rdkit(label, mol, num_confs=None, xyz=None):
    """
    Generate unoptimized conformers in RDKit. If ``xyz`` is not given, random conformers will be generated.

    Args:
        label (str): The species' label.
        mol (RMG Molecule or RDKit RDMol): The molecule object with connectivity and bond order information.
        num_confs (int, optional): The number of random 3D conformations to generate.
        xyz (dict, optional): The 3D coordinates.

    Returns:
        RDMol: An RDKIt molecule with embedded conformers.
    """
    if num_confs is None and xyz is None:
        raise ConformerError('Either num_confs or xyz must be set when calling embed_rdkit() for {0}'.format(label))
    if isinstance(mol, RDMol):
        rd_mol = mol
    elif isinstance(mol, Molecule):
        rd_mol = converter.to_rdkit_mol(mol=mol, remove_h=False)
    else:
        raise ConformerError('Argument mol can be either an RMG Molecule or an RDKit RDMol object. '
                             'Got {0} for {1}'.format(type(mol), label))
    if num_confs is not None:
        Chem.AllChem.EmbedMultipleConfs(rd_mol, numConfs=num_confs, randomSeed=1, enforceChirality=True)
        # Chem.AllChem.EmbedMultipleConfs(rd_mol, numConfs=num_confs, randomSeed=15, enforceChirality=False)
    elif xyz is not None:
        rd_conf = Chem.Conformer(rd_mol.GetNumAtoms())
        for i in range(rd_mol.GetNumAtoms()):
            rd_conf.SetAtomPosition(i, xyz['coords'][i])
        rd_mol.AddConformer(rd_conf)
    return rd_mol


def read_rdkit_embedded_conformers(label, rd_mol, i=None, rd_index_map=None):
    """
    Read coordinates from RDKit conformers.

    Args:
        label (str): The species' label.
        rd_mol (RDKit RDMol): The RDKit molecule with embedded conformers to optimize.
        i (int, optional): The conformer index from rd_mol to read. If None, all will be read,
        rd_index_map (list, optional): An atom map dictionary to reorder the xyz. Requires mol to not be None.

    Returns:
        list: entries are xyz coordinate dicts.
    """
    xyzs = list()
    if i is None:
        # read all conformers:
        for i in range(rd_mol.GetNumConformers()):
            xyzs.append(read_rdkit_embedded_conformer_i(rd_mol, i, rd_index_map=rd_index_map))
    elif isinstance(i, int) and i < rd_mol.GetNumConformers():
        # read only conformer i:
        xyzs.append(read_rdkit_embedded_conformer_i(rd_mol, i, rd_index_map=rd_index_map))
    else:
        raise ConformerError('Cannot read conformer number "{0}" out of {1} RDKit conformers for {2}'.format(
            i, rd_mol.GetNumConformers(), label))
    return xyzs


def read_rdkit_embedded_conformer_i(rd_mol, i, rd_index_map=None):
    """
    Read coordinates from RDKit conformers.

    Args:
        rd_mol (RDKit RDMol): The RDKit molecule with embedded conformers to optimize.
        i (int): The conformer index from rd_mol to read.
        rd_index_map (list, optional): An atom map dictionary to reorder the xyz.
                                       Keys are rdkit atom indices, values are RMG mol atom indices

    Returns:
        dict: xyz coordinates.
    """
    conf = rd_mol.GetConformer(i)
    coords = list()
    for j in range(conf.GetNumAtoms()):
        pt = conf.GetAtomPosition(j)
        coords.append((pt.x, pt.y, pt.z))
    symbols = [rd_atom.GetSymbol() for rd_atom in rd_mol.GetAtoms()]
    if rd_index_map is not None:
        # reorder
        coords = [coords[rd_index_map[j]] for j in range(len(coords))]
        symbols = [symbols[rd_index_map[j]] for j in range(len(symbols))]
    xyz_dict = converter.xyz_from_data(coords=coords, symbols=symbols)
    return xyz_dict


def rdkit_force_field(label, rd_mol, mol=None, force_field='MMFF94s', optimize=True):
    """
    Optimize RDKit conformers using a force field (MMFF94 or MMFF94s are recommended).
    Fallback to Open Babel if RDKit fails.

    Args:
        label (str): The species' label.
        rd_mol (RDKit RDMol): The RDKit molecule with embedded conformers to optimize.
        mol (Molecule, optional): The RMG molecule object with connectivity and bond order information.
        force_field (str, optional): The type of force field to use.
        optimize (bool, optional): Whether to first optimize the conformer using FF. True to optimize.

    Returns:
        list: Entries are optimized xyz's in a dictionary format.
    Returns:
        list: Entries are float numbers representing the energies.
    """
    xyzs, energies = list(), list()
    for i in range(rd_mol.GetNumConformers()):
        if optimize:
            v, j = 1, 0
            while v and j < 200:
                v = Chem.AllChem.MMFFOptimizeMolecule(rd_mol, mmffVariant=force_field, confId=i,
                                                      maxIters=500, ignoreInterfragInteractions=False)
                j += 1
        mol_properties = Chem.AllChem.MMFFGetMoleculeProperties(rd_mol, mmffVariant=force_field)
        if mol_properties is not None:
            ff = Chem.AllChem.MMFFGetMoleculeForceField(rd_mol, mol_properties, confId=i)
            if optimize:
                energies.append(ff.CalcEnergy())
            xyzs.append(read_rdkit_embedded_conformer_i(rd_mol, i))
    if not len(xyzs):
        # RDKit failed, try Open Babel
        energies = list()
        xyzs = read_rdkit_embedded_conformers(label, rd_mol)
        for xyz in xyzs:
            energies.append(openbabel_force_field(label, mol, xyz=xyz, force_field=force_field)[1][0])
    return xyzs, energies


def get_wells(label, angles, blank=20):
    """
    Determine the distinct wells from a list of angles.

    Args:
        label (str): The species' label.
        angles (list): The angles in the torsion.
        blank (int, optional): The blank space between wells.

    Returns:
        list: Entry are well dicts with keys: ``start_idx``, ``end_idx``, ``start_angle``, ``end_angle``, ``angles``.
    """
    if not angles:
        raise ConformerError('Cannot determine wells without angles for {0}'.format(label))
    new_angles = angles
    if angles[0] < -180 + blank and angles[-1] > 180 - blank:
        # relocate the first chunk of data to the end, the well seems to include the  +180/-180 degrees point
        for i, angle in enumerate(angles):
            if i > 0 and abs(angle - angles[i - 1]) > blank:
                part2 = angles[:i]
                for j, _ in enumerate(part2):
                    part2[j] += 360
                new_angles = angles[i:] + part2
                break
    wells = list()
    new_well = True
    for i in range(len(new_angles) - 1):
        if new_well:
            wells.append({'start_idx': i,
                          'end_idx': None,
                          'start_angle': new_angles[i],
                          'end_angle': None,
                          'angles': list()})
            new_well = False
        wells[-1]['angles'].append(new_angles[i])
        if abs(new_angles[i + 1] - new_angles[i]) > blank:
            # This is the last point in this well
            wells[-1]['end_idx'] = i
            wells[-1]['end_angle'] = new_angles[i]
            new_well = True
    if len(wells):
        wells[-1]['end_idx'] = len(new_angles) - 1
        wells[-1]['end_angle'] = new_angles[-1]
        wells[-1]['angles'].append(new_angles[-1])
    return wells


def check_atom_collisions(xyz):
    """
    Check whether atoms are too close to each other.

    Args:
        xyz (dict): The 3D geometry.

    Returns:
         bool: True if they are colliding, False otherwise.

    Todo:
        - Consider atomic radius for the different elements
    """
    coords = xyz['coords']
    symbols = xyz['symbols']
    if symbols == ('H', 'H'):
        # hard-code for H2:
        if sum((coords[0][k] - coords[1][k]) ** 2 for k in range(3)) ** 0.5 < 0.5:
            return True
    for i, coord1 in enumerate(coords):
        if i < len(coords) - 1:
            for coord2 in coords[i+1:]:
                if sum((coord1[k] - coord2[k]) ** 2 for k in range(3)) ** 0.5 < 0.9:
                    return True
    return False


def check_special_non_rotor_cases(mol, top1, top2):
    """
    Check whether one of the tops correspond to a special case which does not have a torsional mode.
    Checking for ``R-[C,N]#[N,[CH],[C]]`` groups, such as: in cyano groups (`R-C#N``),
    C#C groups (``R-C#CH`` or ``R-C#[C]``), and azide groups: (``R-N#N``).

    Args:
        mol (Molecule): The RMG molecule.
        top1 (list): Entries are atom indices (1-indexed) on one side of the torsion, inc. one of the pivotal atoms.
        top2 (list): Entries are atom indices (1-indexed) on the other side of the torsion, inc. the other pivotal atom.

    Returns:
        bool: ``True`` if this is indeed a special case which should **not** be treated as a torsional mode.
    """
    for top in [top1, top2]:
        if mol.atoms[top[0] - 1].atomtype.label in ['Ct', 'N3t', 'N5tc'] \
                and mol.atoms[top[1] - 1].atomtype.label in ['Ct', 'N3t'] and \
                (len(top) == 2 or (len(top) == 3 and mol.atoms[top[2] - 1].is_hydrogen())):
            return True
    return False


def determine_top_group_indices(mol, atom1, atom2, index=1):
    """
    Determine the indices of a "top group" in a molecule.
    The top is defined as all atoms connected to atom2, including atom2, excluding the direction of atom1.
    Two ``atom_list_to_explore`` are used so the list the loop iterates through isn't changed within the loop.

    Args:
        mol (Molecule): The Molecule object to explore.
        atom1 (Atom): The pivotal atom in mol.
        atom2 (Atom): The beginning of the top relative to atom1 in mol.
        index (bool, optional): Whether to return 1-index or 0-index conventions. 1 for 1-index.

    Returns:
        list: The indices of the atoms in the top (either 0-index or 1-index, as requested).
    Returns:
        bool: Whether the top has heavy atoms (is not just a hydrogen atom). True if it has heavy atoms.
    """
    top = list()
    explored_atom_list, atom_list_to_explore1, atom_list_to_explore2 = [atom1], [atom2], []
    while len(atom_list_to_explore1 + atom_list_to_explore2):
        for atom3 in atom_list_to_explore1:
            top.append(mol.vertices.index(atom3) + index)
            for atom4 in atom3.edges.keys():
                if atom4.is_hydrogen():
                    # append H w/o further exploring
                    top.append(mol.vertices.index(atom4) + index)
                elif atom4 not in explored_atom_list and atom4 not in atom_list_to_explore2:
                    atom_list_to_explore2.append(atom4)  # explore it further
            explored_atom_list.append(atom3)  # mark as explored
        atom_list_to_explore1, atom_list_to_explore2 = atom_list_to_explore2, []
    return top, not atom2.is_hydrogen()


def find_internal_rotors(mol):
    """
    Locates the sets of indices corresponding to every internal rotor (1-indexed).

    Args:
        mol (Molecule): The molecule for which rotors will be determined

    Returns:
        list: Entries are rotor dictionaries with the four-atom scan coordinates, the pivots, and the smallest top.
    """
    rotors = []
    for atom1 in mol.vertices:
        if atom1.is_non_hydrogen():
            for atom2, bond in atom1.edges.items():
                if atom2.is_non_hydrogen() and mol.vertices.index(atom1) < mol.vertices.index(atom2) \
                        and (bond.is_single() or bond.is_hydrogen_bond()) and not mol.is_bond_in_cycle(bond):
                    if len(atom1.edges) > 1 and len(atom2.edges) > 1:  # none of the pivotal atoms are terminal
                        rotor = dict()
                        # pivots:
                        rotor['pivots'] = [mol.vertices.index(atom1) + 1, mol.vertices.index(atom2) + 1]
                        # top:
                        top1, top1_has_heavy_atoms = determine_top_group_indices(mol, atom2, atom1)
                        top2, top2_has_heavy_atoms = determine_top_group_indices(mol, atom1, atom2)
                        non_rotor = check_special_non_rotor_cases(mol, top1, top2)
                        if non_rotor:
                            continue
                        if top1_has_heavy_atoms and not top2_has_heavy_atoms:
                            rotor['top'] = top2
                        elif top2_has_heavy_atoms and not top1_has_heavy_atoms:
                            rotor['top'] = top1
                        else:
                            rotor['top'] = top1 if len(top1) <= len(top2) else top2
                        # scan:
                        rotor['scan'] = []
                        heavy_atoms = []
                        hydrogens = []
                        for atom3 in atom1.edges.keys():
                            if atom3.is_hydrogen():
                                hydrogens.append(mol.vertices.index(atom3))
                            elif atom3 is not atom2:
                                heavy_atoms.append(mol.vertices.index(atom3))
                        smallest_index = len(mol.vertices)
                        if len(heavy_atoms):
                            for i in heavy_atoms:
                                if i < smallest_index:
                                    smallest_index = i
                        else:
                            for i in hydrogens:
                                if i < smallest_index:
                                    smallest_index = i
                        rotor['scan'].append(smallest_index + 1)
                        rotor['scan'].extend([mol.vertices.index(atom1) + 1, mol.vertices.index(atom2) + 1])
                        heavy_atoms = []
                        hydrogens = []
                        for atom3 in atom2.edges.keys():
                            if atom3.is_hydrogen():
                                hydrogens.append(mol.vertices.index(atom3))
                            elif atom3 is not atom1:
                                heavy_atoms.append(mol.vertices.index(atom3))
                        smallest_index = len(mol.vertices)
                        if len(heavy_atoms):
                            for i in heavy_atoms:
                                if i < smallest_index:
                                    smallest_index = i
                        else:
                            for i in hydrogens:
                                if i < smallest_index:
                                    smallest_index = i
                        rotor['scan'].append(smallest_index + 1)
                        rotor['number_of_running_jobs'] = 0
                        rotor['success'] = None
                        rotor['invalidation_reason'] = ''
                        rotor['times_dihedral_set'] = 0
                        rotor['trsh_methods'] = list()
                        rotor['scan_path'] = ''
                        rotor['directed_scan_type'] = ''
                        rotor['directed_scan'] = dict()
                        rotor['dimensions'] = 1
                        rotor['original_dihedrals'] = list()
                        rotor['cont_indices'] = list()
                        rotors.append(rotor)
    return rotors


def to_group(mol, atom_indices):
    """
    This method converts a defined part of a Molecule into a Group.

    Args:
        mol (Molecule): The base molecule.
        atom_indices (list): 0-indexed atom indices corresponding to atoms in mol to be included in the group.

    Returns:
        Group: A group consisting of the desired atoms in mol.
    """
    # Create GroupAtom object for each atom in the molecule
    group_atoms = list()
    index_map = dict()  # keys are Molecule atom indices, values are Group atom indices
    for i, atom_index in enumerate(atom_indices):
        atom = mol.atoms[atom_index]
        group_atoms.append(gr.GroupAtom(atomtype=[atom.atomtype], radical_electrons=[atom.radical_electrons],
                                        charge=[atom.charge], lone_pairs=[atom.lone_pairs]))
        index_map[atom_index] = i
    group = gr.Group(atoms=group_atoms, multiplicity=[mol.multiplicity])
    for atom in mol.atoms:
        # Create a GroupBond for each bond between desired atoms in the molecule
        if mol.atoms.index(atom) in atom_indices:
            for bonded_atom, bond in atom.edges.items():
                if mol.atoms.index(bonded_atom) in atom_indices:
                    group.add_bond(gr.GroupBond(atom1=group_atoms[index_map[mol.atoms.index(atom)]],
                                                atom2=group_atoms[index_map[mol.atoms.index(bonded_atom)]],
                                                order=[bond.order]))
    group.update()
    return group


def update_mol(mol):
    """
    Update atom types, multiplicity, and atom charges in the molecule.

    Args:
        mol (Molecule): The molecule to update.

    Returns:
        Molecule: the updated molecule.
    """
    for atom in mol.atoms:
        atom.update_charge()
    mol.update_atomtypes(log_species=False)
    mol.update_multiplicity()
    mol.identify_ring_membership()
    return mol


def compare_xyz(xyz1, xyz2, precision=0.1):
    """
    Compare coordinates of two conformers of the same species. Not checking isotopes.

    Args:
        xyz1 (dict): Coordinates of conformer 1 in either string or array format.
        xyz2 (dict): Coordinates of conformer 2 in either string or array format.
        precision (float, optional): The allowed difference threshold between coordinates, in Angstroms.

    Returns:
        bool: Whether the coordinates represent the same conformer within the given ``precision``, ``True`` if they do.

    Raises:
        InputError: If ``xyz1`` and ``xyz2`` are of wrong type or have different elements (not considering isotopes).
    """
    if not all(isinstance(xyz, dict) for xyz in [xyz1, xyz2]):
        raise InputError('xyz1 and xyz2 must be dictionaries, got {0} and {1}, respectively'.format(
                              type(xyz1), type(xyz2)))
    if xyz1['symbols'] != xyz2['symbols']:
        raise IndexError('xyz1 and xyz2 have different elements, cannot compare coordinates. '
                         'Got:\n{0}\nand:\n{1}'.format(xyz1['symbols'], xyz2['symbols']))
    for coord1, coord2 in zip(xyz1['coords'], xyz2['coords']):
        for entry1, entry2 in zip(coord1, coord2):
            if abs(entry1 - entry2) > precision:
                return False
    return True


def translate_groups(label, mol, xyz, pivot):
    """
    Exchange between two groups in a molecule. The groups cannot share a ring with the pivotal atom.
    The function does not change the atom order, just the coordinates of atoms.
    If the pivotal atom has exactly one lone pair, consider it as well as a dummy atom in translations.

    Args:
        label (str): The species' label.
        mol (Molecule): The 2D graph representation of the molecule.
        xyz (dict): A string-format 3d coordinates of the molecule with the same atom order as in mol.
        pivot (int): The 0-index of the pivotal atom around which groups are to be translated.

    Returns:
        dict: The translated coordinates.
    """
    mol.identify_ring_membership()  # populates the Atom.props['inRing'] attribute
    atom1 = mol.atoms[pivot]
    lp = atom1.lone_pairs
    if lp > 1:
        logger.warning('Cannot translate groups for {0} if the pivotal atom has more than one '
                       'lone electron pair'.format(label))
        return xyz
    groups, translate, dont_translate = list(), list(), list()
    for atom2 in mol.atoms[pivot].edges.keys():
        top = determine_top_group_indices(mol, atom1, atom2, index=0)[0]
        groups.append({'atom': atom2, 'protons': sum([mol.atoms[i].number for i in top])})  # a dict per top
        if 'inRing' in atom1.props and atom1.props['inRing'] and 'inRing' in atom2.props and atom2.props['inRing']:
            # check whether atom1 and atom2 belong to the same ring
            sssr = mol.get_deterministic_sssr()
            for ring in sssr:
                if atom1 in ring and atom2 in ring:
                    dont_translate.append(atom2)
                    break
    groups.sort(key=lambda x: x['protons'], reverse=False)  # sort by the size (sum of atomic numbers)
    i = 0
    while len(translate) < 2 - lp and i < len(groups):
        if groups[i]['atom'] not in dont_translate:
            translate.append(groups[i])
        i += 1
    if len(translate) == 1 and lp:
        vector = vectors.get_lp_vector(label, mol=mol, xyz=xyz, pivot=pivot)
        new_xyz = translate_group(mol=mol, xyz=xyz, pivot=pivot,
                                  anchor=mol.atoms.index(translate[0]['atom']), vector=vector)
    elif len(translate) == 2 and not lp:
        vector = vectors.get_vector(pivot=pivot, anchor=mol.atoms.index(translate[1]['atom']), xyz=xyz)
        new_xyz = translate_group(mol=mol, xyz=xyz, pivot=pivot,
                                  anchor=mol.atoms.index(translate[0]['atom']), vector=vector)
        # keep original xyz:
        vector = vectors.get_vector(pivot=pivot, anchor=mol.atoms.index(translate[0]['atom']), xyz=xyz)
        new_xyz = translate_group(mol=mol, xyz=new_xyz, pivot=pivot,
                                  anchor=mol.atoms.index(translate[1]['atom']), vector=vector)
    else:
        if lp:
            raise ConformerError('The number of groups to translate is {0}, expected 1 (with a lone pair) '
                                 'for {1}.'.format(len(translate), label))
        else:
            raise ConformerError('The number of groups to translate is {0}, expected 2 for {1}.'.format(
                len(translate), label))
    return new_xyz


def translate_group(mol, xyz, pivot, anchor, vector):
    """
    Translate a group (a set of atoms from the pivot towards the anchor and onwards) by changing its
    pivot -> anchor vector to the desired new vector. Keep the relative distances between the group's atoms constant,
    as well as the distance between the anchor and the vector atoms.

    Args:
        mol (Molecule): The 2D graph representation of the molecule.
        xyz (dict): The 3D coordinates of the molecule with the same atom order as in mol.
        pivot (int): The 0-index of the pivotal atom around which groups are to be translated.
        anchor (int): The 0-index of an anchor atom. The group is defined from the pivot atom to the anchor atom,
                      including all other atoms in the molecule connected to the anchor. The pivot and anchor
                      atoms should not have another path connecting them such as a ring.
        vector (list): The new vector by which the group will be translated.

    Returns:
        dict: The translated coordinates.
    """
    # v1 = unit_vector([-vector[0], -vector[1], -vector[2]])  # reverse the direction to get the correct angle
    v1 = vectors.unit_vector(vector)
    v2 = vectors.unit_vector(vectors.get_vector(pivot=pivot, anchor=anchor, xyz=xyz))
    normal = vectors.get_normal(v2, v1)
    theta = vectors.get_angle(v1, v2)
    # print(theta * 180 / math.pi)  # print theta in degrees when troubleshooting
    # All atoms within the group will be rotated around the same normal vector by theta:
    group = determine_top_group_indices(mol=mol, atom1=mol.atoms[pivot], atom2=mol.atoms[anchor], index=0)[0]
    coords = converter.xyz_to_coords_list(xyz)
    for i in group:
        coords[i] = vectors.rotate_vector(point_a=coords[pivot], point_b=coords[i], normal=normal, theta=theta)
    new_xyz = converter.xyz_from_data(coords=coords, symbols=xyz['symbols'], isotopes=xyz['isotopes'])
    return new_xyz


def get_number_of_chiral_centers(label, mol, conformer=None, xyz=None, just_get_the_number=True):
    """
    Determine the number of chiral centers by type. Either ``conformer`` or ``xyz`` must be given.

    Args:
        label (str): The species label.
        mol (Molecule): The RMG Molecule object.
        conformer (dict, optional): A conformer dictionary.
        xyz (dict, optional): The xyz coordinates.
        just_get_the_number (bool, optional): Return the number of chiral centers regardless of their type.

    Returns:
        dict, int : Keys are types of chiral sites ('C' for carbon, 'N' for nitrogen, 'D' for double bond),
                    values are the number of chiral centers of each type. If ``just_get_the_number`` is ``True``,
                    just returns the number of chiral centers (integer).

    Raises:
        InputError: If neither ``conformer`` nor ``xyz`` were given.
    """
    if conformer is None and xyz is None:
        raise InputError('Must get either conformer or xyz.')
    if conformer is None:
        conformer = {'xyz': xyz}
    conformer = determine_chirality(conformers=[conformer], label=label, mol=mol)[0]
    result = {'C': 0, 'N': 0, 'D': 0}
    for symbol in conformer['chirality'].values():
        if symbol in ['R', 'S']:
            result['C'] += 1
        elif symbol in ['NR', 'NS']:
            result['N'] += 1
        elif symbol in ['E', 'Z']:
            result['D'] += 1
        else:
            raise ConformerError(f"Chiral symbols must be either `R`, `S`, `NR`, `NS`, `E`, `Z`, got: {symbol}.")
    if just_get_the_number:
        return sum([val for val in result.values()])
    return result


def get_lowest_diastereomers(label, mol, conformers, diastereomers=None):
    """
    Get the 2^(n-1) diastereomers with the lowest energy (where n is the number of chiral centers in the molecule).
    We exclude enantiomers (mirror images where ALL chiral centers invert).
    If a specific diasteromer is given (in an xyz dict form), then only the lowest conformer with the same chirality
    will be returned.
    * untested *

    Args:
        label (str): The species' label.
        mol (Molecule): The 2D graph representation of the molecule.
        conformers (list): Entries are conformer dictionaries.
        diastereomers (list, optional): Entries are xyz's in a dictionary format or conformer structures
                                        representing specific diastereomers to keep.

    Returns:
        list: Entries are lowest energy diastereomeric conformer dictionaries to consider.

    Raises:
        ConformerError: If diastereomers is not None and is of wrong type,
                        or if conformers with the requested chirality combination could not be generated.
    """
    # assign chirality properties to all conformers
    conformers = determine_chirality(conformers, label, mol)
    # initialize the enantiomeric dictionary (includes enantiomers and diastereomers)
    # keys are chiral combinations, values are lowest conformers
    enantiomers_dict = dict()
    for conformer in conformers:
        if conformer['FF energy'] is not None:
            chirality_tuple = chirality_dict_to_tuple(conformer['chirality'])
            if chirality_tuple not in list(enantiomers_dict.keys()):
                # this is a new enantiomer, consider it
                enantiomers_dict[chirality_tuple] = conformer
            elif conformer['FF energy'] < enantiomers_dict[chirality_tuple]['FF energy']:
                # found a lower energy conformer with the same chirality, replace
                enantiomers_dict[chirality_tuple] = conformer
    if diastereomers is None:
        # no specific diastereomers were requested
        pruned_enantiomers_dict = prune_enantiomers_dict(label, enantiomers_dict)
    else:
        if isinstance(diastereomers, list):
            # make sure entries are conformers, convert if needed
            modified_diastereomers = list()
            for diastereomer in diastereomers:
                if isinstance(diastereomer, str) or isinstance(diastereomer, dict) and 'coords' in diastereomer:
                    # we'll also accept string format xyz
                    modified_diastereomers.append({'xyz': converter.check_xyz_dict(diastereomer)})
                elif isinstance(diastereomer, dict) and 'xyz' in diastereomer:
                    modified_diastereomers.append(diastereomer)
                else:
                    raise ConformerError(f'diastereomers entries must be either xyz or conformer dictionaries, '
                                         f'got {type(diastereomer)} for {label}')
            diastereomer_confs = [{'xyz': converter.check_xyz_dict(diastereomer)} for diastereomer in diastereomers]
            diastereomer_confs = determine_chirality(diastereomer_confs, label, mol)
        else:
            raise ConformerError(f'diastereomers must be a list of xyz coordinates, got: {type(diastereomers)}')
        chirality_tuples = [chirality_dict_to_tuple(conformer['chirality']) for conformer in diastereomer_confs]
        new_enantiomers_dict = dict()
        for chirality_tuple, conformer in enantiomers_dict.items():
            if chirality_tuple in chirality_tuples:
                new_enantiomers_dict[chirality_tuple] = conformer
        if not new_enantiomers_dict:
            raise ConformerError(f'Could not generate conformers with chirality combination:\n{chirality_tuples}')
        pruned_enantiomers_dict = prune_enantiomers_dict(label, new_enantiomers_dict)
        if len(list(pruned_enantiomers_dict.keys())) and list(pruned_enantiomers_dict.keys())[0] != tuple():
            logger.info(f'Considering the following enantiomeric combinations for {label}:\n'
                        f'{list(pruned_enantiomers_dict.keys())}')
    return list(pruned_enantiomers_dict.values())


def prune_enantiomers_dict(label, enantiomers_dict):
    """
    A helper function for screening out enantiomers from the enantiomers_dict, leaving only diastereomers
    (so removing all exact mirror images). Note that double bond chiralities 'E' and 'Z' are not mirror images of each
    other, and are not pruned out.

    Args:
        label (str): The species' label.
        enantiomers_dict (dict): Keys are chirality tuples, values are conformer structures.

    Returns:
        dict: The pruned enantiomers_dict.
    """
    pruned_enantiomers_dict = dict()
    for chirality_tuples, conformer in enantiomers_dict.items():
        inversed_chirality_tuples = tuple([(chirality_tuple[0], inverse_chirality_symbol(chirality_tuple[1]))
                                           for chirality_tuple in chirality_tuples])
        if chirality_tuples not in list(pruned_enantiomers_dict.keys()) \
                and inversed_chirality_tuples not in list(pruned_enantiomers_dict.keys()):
            # this combination (or its exact mirror image) was not considered yet
            if inversed_chirality_tuples in list(enantiomers_dict.keys()):
                # the mirror image exists, check which has a lower energy
                inversed_conformer = enantiomers_dict[inversed_chirality_tuples]
                if inversed_conformer['FF energy'] is None and conformer['FF energy'] is None:
                    logger.warning(f'Could not get energies of enantiomers {chirality_tuples} '
                                   f'nor its mirror image {inversed_chirality_tuples} for species {label}')
                    continue
                elif inversed_conformer['FF energy'] is None:
                    pruned_enantiomers_dict[chirality_tuples] = conformer
                elif conformer['FF energy'] is None:
                    pruned_enantiomers_dict[inversed_chirality_tuples] = inversed_conformer
                elif conformer['FF energy'] <= inversed_conformer['FF energy']:
                    pruned_enantiomers_dict[chirality_tuples] = conformer
                else:
                    pruned_enantiomers_dict[inversed_chirality_tuples] = inversed_conformer
            else:
                # the mirror image does not exist
                pruned_enantiomers_dict[chirality_tuples] = conformer
    return pruned_enantiomers_dict


def inverse_chirality_symbol(symbol):
    """
    Inverses a chirality symbol, e.g., the 'R' character to 'S', or 'NS' to 'NR'.
    Note that chiral double bonds ('E' and 'Z') must not be inversed (they are not mirror images of each other).

    Args:
        symbol (str): The chirality symbol.

    Returns:
        str: The inverse chirality symbol.

    Raises:
        InputError: If ``symbol`` could not be recognized.
    """
    inversion_dict = {'R': 'S', 'S': 'R', 'NR': 'NS', 'NS': 'NR', 'E': 'E', 'Z': 'Z'}
    if symbol not in list(inversion_dict.keys()):
        raise InputError(f"Recognized chirality symbols are 'R', 'S', 'NR', 'NS', 'E', and 'Z', got {symbol}.")
    return inversion_dict[symbol]


def chirality_dict_to_tuple(chirality_dict):
    """
    A helper function for using the chirality dictionary of a conformer as a key in the enantiomers_dict
    by converting it to a tuple deterministically.

    Args:
        chirality_dict (dict): The chirality dictionary of a conformer.

    Returns:
        tuple: A deterministic tuple representation of the chirality dictionary.

    Raises:
        ConformerError: If the chirality values are wrong.
    """
    # extract carbon sites (values are either 'R' or 'S'), nitrogen sites (values are either 'NR' or 'NS')
    # and chiral double bonds (values are either 'E' or 'Z')
    c_sites, n_sites, bonds, result = list(), list(), list(), list()
    for site, chirality in chirality_dict.items():
        if chirality in ['R', 'S']:
            c_sites.append((site, chirality))
        elif chirality in ['NR', 'NS']:
            n_sites.append((site, chirality))
        elif chirality in ['E', 'Z']:
            bond_site = site if site[0] < site[1] else (site[1], site[0])
            bonds.append((bond_site, chirality))
        else:
            raise ConformerError(f'Chiralities could either be R, S, NR, NS, E, or Z. Got: {chirality}.')
    # sort the lists
    c_sites.sort(key=lambda entry: entry[0])
    n_sites.sort(key=lambda entry: entry[0])
    bonds.sort(key=lambda entry: entry[0])
    # combine by order
    for entry in c_sites + n_sites + bonds:
        result.append(entry)
    return tuple(result)


def determine_chirality(conformers, label, mol, force=False):
    """
    Determines the Cahn–Ingold–Prelog (CIP) chirality (R or S) of atoms in the conformer,
    as well as the CIP chirality of double bonds (E or Z).

    Args:
        conformers (list): Entries are conformer dictionaries.
        label (str): The species' label.
        mol (RMG Molecule or RDKit RDMol): The molecule object with connectivity and bond order information.
        force (bool, optional): Whether to override data, ``True`` to override, default is ``False``.

    Returns:
        list: Conformer dictionaries with updated with 'chirality'. ``conformer['chirality']`` is a dictionary.
              Keys are either a 1-length tuple of atom indices (for chiral atom centers) or a 2-length tuple of atom
              indices (for chiral double bonds), values are either 'R' or 'S' for chiral atom centers
              (or 'NR' or 'NS' for chiral nitrogen centers), or 'E' or 'Z' for chiral double bonds.
              All atom indices are 0-indexed.
    """
    chiral_nitrogen_centers = identify_chiral_nitrogen_centers(mol)
    new_mol, elements_to_insert = replace_n_with_c_in_mol(mol, chiral_nitrogen_centers)
    for conformer in conformers:
        if 'chirality' not in conformer:
            # keys are either 1-length atom indices (for chiral atom centers)
            # or 2-length atom indices (for chiral double bonds)
            # values are either 'R', 'S', 'NR', 'NS', 'E', or 'Z'
            conformer['chirality'] = dict()
        elif conformer['chirality'] != dict() and not force:
            # don't override data
            continue
        new_xyz = replace_n_with_c_in_xyz(label, mol, conformer['xyz'], chiral_nitrogen_centers, elements_to_insert)
        rd_mol = embed_rdkit(label, new_mol, xyz=new_xyz)
        Chem.rdmolops.AssignStereochemistryFrom3D(rd_mol, 0)
        for i, rd_atom in enumerate(rd_mol.GetAtoms()):
            rd_atom_props_dict = rd_atom.GetPropsAsDict()
            if '_CIPCode' in list(rd_atom_props_dict.keys()):
                if mol.atoms[i].is_nitrogen():
                    # this is a nitrogen site in the original molecule, mark accordingly
                    conformer['chirality'][(i,)] = 'N' + rd_atom_props_dict['_CIPCode']
                else:
                    conformer['chirality'][(i,)] = rd_atom_props_dict['_CIPCode']
        for rd_bond in rd_mol.GetBonds():
            stereo = str(rd_bond.GetStereo())
            if stereo in ['STEREOE', 'STEREOZ']:
                # possible values are 'STEREOANY', 'STEREOCIS', 'STEREOE', 'STEREONONE', 'STEREOTRANS', and 'STEREOZ'
                rd_atoms = [rd_bond.GetBeginAtomIdx(), rd_bond.GetEndAtomIdx()]  # indices of atoms bonded by this bond
                conformer['chirality'][tuple(rd_atom for rd_atom in rd_atoms)] = stereo[-1]
    return conformers


def identify_chiral_nitrogen_centers(mol):
    """
    Identify the atom indices corresponding to a chiral nitrogen centers in a molecule (umbrella modes).

    Args:
        mol (Molecule): The molecule to be analyzed.

    Returns:
        list: Atom numbers (0-indexed) representing chiral nitrogen centers in the molecule (umbrella modes).

    Raises:
        TypeError: If ``mol`` is of wrong type.
    """
    if not isinstance(mol, Molecule):
        raise TypeError(f'mol must be a Molecule instance, got: {type(mol)}')
    chiral_nitrogen_centers = list()
    for atom1 in mol.atoms:
        if atom1.is_nitrogen() and atom1.lone_pairs == 1 and atom1.radical_electrons == 0 \
                and (len(list(atom1.edges.keys())) == 3
                     or (atom1.radical_electrons == 1 and len(list(atom1.edges.keys())) == 2)):
            groups, tops, top_element_counts = list(), list(), list()
            for atom2 in atom1.edges.keys():
                top = determine_top_group_indices(mol, atom1, atom2, index=0)[0]
                tops.append(top)
                top_element_counts.append(get_top_element_count(mol, top))
                groups.append(to_group(mol, top))
            if (top_element_counts[0] != top_element_counts[1] and top_element_counts[1] != top_element_counts[2]) \
                    or all([not groups[0].is_isomorphic(group, save_order=True) for group in groups[1:]] +
                           [not groups[-1].is_isomorphic(group, save_order=True) for group in groups[:-1]]):
                # if we can say that TWO groups, each separately considered, isn't isomorphic to the others,
                # then this nitrogen has all different groups.
                chiral_nitrogen_centers.append(mol.atoms.index(atom1))
    return chiral_nitrogen_centers


def replace_n_with_c_in_mol(mol, chiral_nitrogen_centers):
    """
    Replace nitrogen atoms (pre-identified as chiral centers) with carbon atoms, replacing the lone electron pair
    (assuming just one exists) with a hydrogen or a halogen atom, preserving any radical electrons on the nitrogen atom.

    Args:
        mol (Molecule): The molecule to be analyzed.
        chiral_nitrogen_centers (list): The 0-index of chiral (umbrella mode) nitrogen atoms in the molecule.

    Returns:
        Molecule: A copy of the molecule with replaced N atoms.
    Returns:
        list: Elements inserted in addition to the C atom, ordered as in ``chiral_nitrogen_centers``.

    Raises:
        ConformerError: If any of the atoms indicated by ``chiral_nitrogen_centers`` could not be a chiral nitrogen atom
    """
    new_mol = mol.copy(deep=True)
    inserted_elements = list()
    for n_index in chiral_nitrogen_centers:
        if not mol.atoms[n_index].is_nitrogen():
            raise ConformerError(f'Cannot replace a nitrogen atom index {n_index} if it is not a nitrogen element.')
        if mol.atoms[n_index].lone_pairs != 1:
            raise ConformerError(f'Cannot replace a nitrogen atom index {n_index} with number of lone pairs '
                                 f'different than one (got: {mol.atoms[n_index].lone_pairs}).')
        if mol.atoms[n_index].radical_electrons > 1:
            raise ConformerError(f'Cannot replace a nitrogen atom index {n_index} if it has more than one radical '
                                 f'electrons (got: {mol.atoms[n_index].radical_electrons}).')
        if any([not bond.is_single() for bond in mol.atoms[n_index].edges.values()]):
            raise ConformerError(f'Cannot replace a nitrogen atom index {n_index} if not all of its bonds are single '
                                 f'(got: {[bond.order for bond in mol.atoms[n_index].edges.values()]}).')
        new_c_atom = Atom(element=C_ELEMENT, radical_electrons=mol.atoms[n_index].radical_electrons,
                        charge=mol.atoms[n_index].charge, lone_pairs=0, id=mol.atoms[n_index].id)
        new_c_atom.edges = dict()
        for atom2 in mol.atoms[n_index].edges.keys():
            # delete bonds from all other atoms connected to the atom represented by n_index
            del new_mol.atoms[mol.atoms.index(atom2)].edges[new_mol.atoms[n_index]]
        new_mol.vertices[n_index] = new_c_atom
        h, f, cl = False, False, False  # mark hydrogen, fluorine, and chlorine neighbors of the original atom
        for atom2 in mol.atoms[n_index].edges.keys():
            new_mol.add_bond(Bond(atom1=new_c_atom, atom2=new_mol.atoms[mol.atoms.index(atom2)], order=1))
            if atom2.is_hydrogen():
                h = True
            elif atom2.is_fluorine():
                f = True
            elif atom2.is_chlorine():
                cl = True
        if not h:
            additional_element = H_ELEMENT
            inserted_elements.append('H')
        elif not f:
            additional_element = F_ELEMENT
            inserted_elements.append('F')
        elif not cl:
            additional_element = Cl_ELEMENT
            inserted_elements.append('Cl')
        else:
            # this can only happen if the molecule is NHFCl (ammonia substituted with one F and one Cl), use iodine
            additional_element = I_ELEMENT
            inserted_elements.append('I')
        new_atom = Atom(element=additional_element, radical_electrons=0, charge=0,
                        lone_pairs=0 if additional_element.number == 1 else 3)
        new_atom.edges = dict()
        # new_mol.add_atom(new_atom)

        new_mol.vertices.append(new_atom)
        new_bond = Bond(atom1=new_c_atom, atom2=new_atom, order=1)
        new_mol.add_bond(new_bond)
    return new_mol, inserted_elements


def replace_n_with_c_in_xyz(label, mol, xyz, chiral_nitrogen_centers, elements_to_insert):
    """
    Replace nitrogen atoms (pre-identified as chiral centers) with carbon atoms, replacing the lone electron pair
    (assuming just one exists) with a hydrogen or a halogen atom.

    Args:
        label (str): The species label.
        mol (Molecule): The respective molecule object.
        xyz (dict): The 3D coordinates to process.
        chiral_nitrogen_centers (list): The 0-index of chiral (umbrella mode) nitrogen atoms in the molecule.
        elements_to_insert (list): The element (H/F/Cl/I) to insert in addition to C per nitrogen center.

    Returns:
        dict: The coordinates with replaced N atoms.
    """
    symbols = list(copy.copy(xyz['symbols']))
    isotopes = list(copy.copy(xyz['isotopes'])) if 'isotopes' in xyz else None
    coords = converter.xyz_to_coords_list(xyz)
    for n_index, element_to_insert in zip(chiral_nitrogen_centers, elements_to_insert):
        symbols[n_index] = 'C'
        if isotopes is not None:
            isotopes[n_index] = 12
        if element_to_insert == 'H':
            symbol, isotope, distance = 'H', 1, 1.1
        elif element_to_insert == 'F':
            symbol, isotope, distance = 'F', 19, 2.0
        elif element_to_insert == 'Cl':
            symbol, isotope, distance = 'Cl', 35, 1.77
        elif element_to_insert == 'I':
            symbol, isotope, distance = 'I', 127, 2.14
        else:
            raise ConformerError(f'Element to insert must be either H, F, Cl, or I. Got: {element_to_insert}')
        symbols.append(symbol)
        if isotopes is not None:
            isotopes.append(isotope)
        lp_vector = vectors.set_vector_length(vectors.get_lp_vector(label, mol, xyz, n_index), distance)
        lp_vector[0] += coords[n_index][0]
        lp_vector[1] += coords[n_index][1]
        lp_vector[2] += coords[n_index][2]
        coords.append(lp_vector)
    new_xyz = converter.xyz_from_data(coords=coords, symbols=symbols, isotopes=isotopes)
    return new_xyz


def get_top_element_count(mol, top):
    """
    Returns the element count for the molecule considering only the atom indices in ``top``.

    Args:
        mol (Molecule): The molecule to consider.
        top (list): The atom indices to consider.

    Returns:
        dict: The element count, keys are tuples of (element symbol, isotope number), values are counts.
    """
    if not isinstance(top, list):
        top = list(top)
    element_count = {}
    for i, atom in enumerate(mol.atoms):
        if i in top:
            key = (atom.element.symbol, atom.element.isotope)
            if key in element_count:
                element_count[key] += 1
            else:
                element_count[key] = 1
    return element_count


def initialize_log(verbose=logging.INFO):
    """
    Set up a simple logger for stdout printing (not saving into as log file).

    Args:
        verbose (int, optional): Specify the amount of log text seen.
    """
    logger.setLevel(verbose)
    logger.propagate = False

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
