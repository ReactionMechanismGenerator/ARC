#!/usr/bin/env python
# encoding: utf-8

"""
Determine scaling factors for a given list of levels of theory

Based on DOI: 10.1016/j.cpc.2016.09.004
Adapted by Duminda Ranasinghe and Alon Grinberg Dana
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import time

from arkane.statmech import determine_qm_software

from arc.common import get_logger
from arc.settings import arc_path
from arc.scheduler import Scheduler
from arc.arc_exceptions import InputError
from arc.species.species import ARCSpecies
from arc.common import check_ess_settings, time_lapse, initialize_log

try:
    from arc.settings import global_ess_settings
except ImportError:
    global_ess_settings = None

##################################################################

logger = get_logger()


HEADER = 'FREQ: A PROGRAM FOR OPTIMIZING SCALE FACTORS (Version 1)\n'\
         '                 written by                 \n'\
         'Haoyu S. Yu, Lucas J. Fiedler, I.M. Alecu, and Donald G. Truhlar\n'\
         'Department of Chemistry and Supercomputing Institute\n'\
         'University of Minnesota, Minnesota 55455-0431\n'\
         'CITATIONS:\n'\
         '1. I.M., Alecu, J. Zheng, Y. Zhao, D.G. Truhlar, J. Chem. Theory Comput. 2010, 6, 9, 2872-2887,\n'\
         '   DOI: 10.1021/ct100326h\n'\
         '2. H.S. Yu, L.J. Fiedler, I.M. Alecu,, D.G. Truhlar, Computer Physics Communications 2017, 210, 132-138,\n'\
         '   DOI: 10.1016/j.cpc.2016.09.004\n\n'


def determine_scaling_factors(levels_of_theory, ess_settings=None, init_log=True):
    """
    Determine the zero-point energy, harmonic frequencies, and fundamental frequencies scaling factors
    for a given frequencies level of theory.

    Args:
        levels_of_theory (list, str, unicode): A list of frequencies levels of theory
                                               for which scaling factors are determined.
                                               A string can also be passed for just one level of theory.
        ess_settings (dict, optional): A dictionary of available ESS (keys) and a corresponding server list (values).
        init_log (bool, optional): Whether to initialize the logger. True to initialize.
                                   Should be True when called as a stand alone, and False when called within ARC.

    Returns:
        str, unicode: The modified level of theory
    """
    if init_log:
        initialize_log(log_file='scaling_factor.log', project='Scaling Factors')

    if isinstance(levels_of_theory, (str, unicode)):
        levels_of_theory = [levels_of_theory]
    if not isinstance(levels_of_theory, list):
        raise InputError('levels_of_theory must be a list (or a string if only one level is desired). Got: {0}'.format(
            type(levels_of_theory)))
    t0 = time.time()

    print('\n\n\n')
    print(HEADER)
    print('\n\nstarting ARC...\n')

    job_types = {'conformers': False, 'opt': True, 'fine': True, 'freq': True, 'sp': False, '1d_rotors': False,
                 'orbitals': False, 'onedmin': False}  # only run opt (fine) and freq

    lambda_zpes, zpe_dicts, times = list(), list(), list()
    for level_of_theory in levels_of_theory:
        t1 = time.time()
        print('\nComputing scaling factors at the {0} level of theory...\n\n'.format(level_of_theory))
        renamed_level = rename_level(level_of_theory)
        project = 'scaling_' + renamed_level
        project_directory = os.path.join(arc_path, 'Projects', 'scaling_factors', project)

        species_list = get_species_list()

        if '//' in level_of_theory:
            raise InputError('Level of theory should either be a composite method or in a method/basis-set format. '
                             'Got {0}'.format(level_of_theory))
        if '/' not in level_of_theory:  # assume this is a composite method
            freq_level = ''
            composite_method = level_of_theory.lower()
            job_types['freq'] = False
        else:
            freq_level = level_of_theory.lower()
            composite_method = ''
            job_types['freq'] = True

        ess_settings = check_ess_settings(ess_settings or global_ess_settings)

        Scheduler(project=project, project_directory=project_directory, species_list=species_list,
                  composite_method=composite_method, opt_level=freq_level, freq_level=freq_level,
                  ess_settings=ess_settings, job_types=job_types, allow_nonisomorphic_2d=True)

        zpe_dict = dict()
        for spc in species_list:
            try:
                zpe = get_zpe(os.path.join(project_directory, 'output', 'Species', spc.label, 'geometry', 'freq.out'))
            except Exception:
                zpe = None
            zpe_dict[spc.label] = zpe
        zpe_dicts.append(zpe_dict)

        lambda_zpes.append(calculate_truhlar_scaling_factors(zpe_dict, level_of_theory))
        times.append(time_lapse(t1))

    summarize_results(lambda_zpes, levels_of_theory, zpe_dicts, times, time_lapse(t0))
    print('\n\n\n')
    print(HEADER)

    harmonic_freq_scaling_factors = [lambda_zpe * 1.014 for lambda_zpe in lambda_zpes]
    return harmonic_freq_scaling_factors


def calculate_truhlar_scaling_factors(zpe_dict, level_of_theory):
    """
    Calculate the scaling factors using Truhlar's method:

    FREQ: A PROGRAM FOR OPTIMIZING SCALE FACTORS (Version 1)
    written by Haoyu S. Yu, Lucas J. Fiedler, I.M. Alecu, and Donald G. Truhlar
    Department of Chemistry and Supercomputing Institute
    University of Minnesota, Minnesota 55455-0431

    Citations:
        1. I.M., Alecu, J. Zheng, Y. Zhao, D.G. Truhlar, J. Chem. Theory Comput. 2010, 6, 9, 2872-2887
           DOI: 10.1021/ct100326h
        2. H.S. Yu, L.J. Fiedler, I.M. Alecu,, D.G. Truhlar, Computer Physics Communications 2017, 210, 132-138
           DOI: 10.1016/j.cpc.2016.09.004

    Args:
        zpe_dict (dict): The calculated vibrational zero-point energies at the requested level of theory.
                         Keys are species labels, values are floats representing the ZPE in J/mol.
        level_of_theory (str, unicode): The frequencies level of theory.

    Returns:
        float: The scale factor for the vibrational zero-point energy (lambda ZPE) as defined in reference [2].
    """
    unconverged = [key for key, val in zpe_dict.items() if val is None]
    if len(unconverged):
        print('\n\nWarning: Not all species in the standard set have converged at the {0} level of theory!\n'
              'Unconverged species: {1}\n\n'.format(level_of_theory, unconverged))
    else:
        print('\n\nAll species in the standard set have converged at the {0} level of theory\n\n\n'.format(
            level_of_theory))

    # Experimental ZPE values converted from kcal/mol to J/mol, as reported in reference [2]:
    exp_zpe_dict = {'C2H2': 16.490 * 4184,
                    'CH4': 27.710 * 4184,
                    'CO2': 7.3 * 4184,
                    'CO': 3.0929144 * 4184,
                    'F2': 1.302 * 4184,
                    'CH2O': 16.1 * 4184,
                    'H2O': 13.26 * 4184,
                    'H2': 6.231 * 4184,
                    'HCN': 10.000 * 4184,
                    'HF': 5.864 * 4184,
                    'N2O': 6.770 * 4184,
                    'N2': 3.3618 * 4184,
                    'NH3':21.200 * 4184,
                    'OH': 5.2915 * 4184,
                    'Cl2': 0.7983 * 4184}

    numerator, denominator = 0.0, 0.0  # numerator and denominator in eq. 5 of reference [2]

    for label, zpe in zpe_dict.items():
        numerator += zpe * exp_zpe_dict[label] if zpe is not None else 0
        if zpe is not None:
            denominator += zpe ** 2.0
        else:
            logger.error('ZPE of species {0} could not be determined!'.format(label))
    lambda_zpe = numerator / denominator  # lambda_zpe on the left side of eq. 5 of [2]

    return lambda_zpe


def summarize_results(lambda_zpes, levels_of_theory, zpe_dicts, times, overall_time, base_path=None):
    """
    Print and save the results to file.

    Args:
        lambda_zpes (list): The scale factors for the vibrational zero-point energy, entries are floats.
        levels_of_theory (list): The frequencies levels of theory.
        zpe_dicts (list): Entries are The calculated vibrational zero-point energies at the requested level of theory.
                          Keys are species labels, values are floats representing the ZPE in J/mol.
        times (list): Entries are string-format of the calculation execution times.
        overall_time (str, unicode): A string-format of the overall calculation execution time.
        base_path (str, unicode, optional): The path to the scaling factors base folder.
    """
    base_path = base_path or os.path.join(arc_path, 'Projects', 'scaling_factors')
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    i, success = 0, False
    while not success:
        info_file_path = os.path.join(base_path, 'scaling_factors_' + str(i) + '.info')
        if os.path.isfile(info_file_path):
            i += 1
        else:
            success = True

    with open(info_file_path, 'w') as f:
        f.write(str(HEADER))
        arkane_text = '\n\n\nYou may copy-paste the following harmonic frequencies scaling factor/s to Arkane\n' \
                      '(paste in the `freq_dict` under assign_frequency_scale_factor() in arkane/statmech.py):\n'
        arkane_formats = list()
        harmonic_freq_scaling_factors = list()
        for lambda_zpe, level_of_theory, zpe_dict, execution_time\
                in zip(lambda_zpes, levels_of_theory, zpe_dicts, times):
            harmonic_freq_scaling_factor = lambda_zpe * 1.014
            fundamental_freq_scaling_factor = lambda_zpe * 0.974
            harmonic_freq_scaling_factors.append(fundamental_freq_scaling_factor)
            unconverged = [key for key, val in zpe_dict.items() if val is None]

            text = '\n\nLevel of theory: {0}\n'.format(level_of_theory)
            if unconverged:
                text += 'The following species from the standard set did not converge at this level:\n {0}\n'.format(
                    unconverged)
            text += 'Scale Factor for Zero-Point Energies     = {0:.3f}\n'.format(lambda_zpe)
            text += 'Scale Factor for Harmonic Frequencies    = {0:.3f}\n'.format(harmonic_freq_scaling_factor)
            text += 'Scale Factor for Fundamental Frequencies = {0:.3f}\n'.format(fundamental_freq_scaling_factor)
            text += '(execution time: {0})\n'.format(execution_time)
            print(text)
            f.write(str(text))
            arkane_formats.append("                 '{0}': {1:.3f},  # [4]\n".format(level_of_theory,
                                                                            harmonic_freq_scaling_factor))
        print(arkane_text)
        f.write(arkane_text)
        for arkane_format in arkane_formats:
            print(arkane_format)
            f.write(str(arkane_format))
        overall_time_text = '\n\nScaling factors calculation for {0} levels of theory completed' \
                            ' (elapsed time: {1}).\n'.format(len(levels_of_theory), overall_time)
        print(overall_time_text)
        f.write(str(overall_time_text))


def get_species_list():
    """
    Generates the standardized species list.

    Returns:
        list: The standardized species list initialized with xyz.
    """
    c2h2_xyz = """     C                  0.00000000    0.00000000    0.00000000
     C                  0.00000000    0.00000000    1.20314200
     H                  0.00000000   -0.00000000    2.26574700
     H                 -0.00000000   -0.00000000   -1.06260500"""
    ch4_xyz = """     C                  0.00000000    0.00000000    0.00000000
     H                  0.00000000    0.00000000    1.08744517
     H                  1.02525314    0.00000000   -0.36248173
     H                 -0.51262658    0.88789525   -0.36248173
     H                 -0.51262658   -0.88789525   -0.36248173"""
    co2_xyz = """     C                  0.00000000    0.00000000    0.00000000
     O                  0.00000000    0.00000000    1.15948460
     O                  0.00000000    0.00000000   -1.15948460"""
    co_xyz = """     O                  0.00000000    0.00000000    0.00000000
     C                  0.00000000    0.00000000    1.12960815"""
    f2_xyz = """     F                  0.00000000    0.00000000    0.00000000
     F                  0.00000000    0.00000000    1.39520410"""
    ch2o_xyz = """     O                  0.00000000    0.00000000    0.67462200
     C                  0.00000000    0.00000000   -0.52970700
     H                  0.00000000    0.93548800   -1.10936700
     H                  0.00000000   -0.93548800   -1.10936700"""
    h2o_xyz = """     O                  0.00000000    0.00000000    0.00000000
     H                  0.00000000    0.00000000    0.95691441
     H                  0.92636305    0.00000000   -0.23986808"""
    h2_xyz = """     H                  0.00000000    0.00000000    0.00000000
     H                  0.00000000    0.00000000    0.74187646"""
    hcn_xyz = """     C                  0.00000000    0.00000000   -0.50036500
     N                  0.00000000    0.00000000    0.65264000
     H                  0.00000000    0.00000000   -1.56629100"""
    hf_xyz = """     F                  0.00000000    0.00000000    0.00000000
     H                  0.00000000    0.00000000    0.91538107"""
    n2o_xyz = """     N                  0.00000000    0.00000000    0.00000000
     N                  0.00000000    0.00000000    1.12056262
     O                  0.00000000    0.00000000    2.30761092"""
    n2_xyz = """     N                  0.00000000    0.00000000    0.00000000
     N                  0.00000000    0.00000000    1.09710935"""
    nh3_xyz = """     N                  0.00000000    0.00000000    0.11289000
     H                  0.00000000    0.93802400   -0.26340900
     H                  0.81235300   -0.46901200   -0.26340900
     H                 -0.81235300   -0.46901200   -0.26340900"""
    oh_xyz = """     O                  0.00000000    0.00000000    0.00000000
     H                  0.00000000    0.00000000    0.96700000"""
    cl2_xyz = """     Cl                 0.00000000    0.00000000    0.00000000
     Cl                 0.00000000    0.00000000    1.10000000"""

    c2h2 = ARCSpecies(label='C2H2', smiles=str('C#C'), multiplicity=1, charge=0)
    c2h2.initial_xyz = c2h2_xyz

    ch4 = ARCSpecies(label='CH4', smiles=str('C'), multiplicity=1, charge=0)
    ch4.initial_xyz = ch4_xyz

    co2 = ARCSpecies(label='CO2', smiles=str('O=C=O'), multiplicity=1, charge=0)
    co2.initial_xyz = co2_xyz

    co = ARCSpecies(label='CO', smiles=str('[C-]#[O+]'), multiplicity=1, charge=0)
    co.initial_xyz = co_xyz

    f2 = ARCSpecies(label='F2', smiles=str('[F][F]'), multiplicity=1, charge=0)
    f2.initial_xyz = f2_xyz

    ch2o = ARCSpecies(label='CH2O', smiles=str('C=O'), multiplicity=1, charge=0)
    ch2o.initial_xyz = ch2o_xyz

    h2o = ARCSpecies(label='H2O', smiles=str('O'), multiplicity=1, charge=0)
    h2o.initial_xyz = h2o_xyz

    h2 = ARCSpecies(label='H2', smiles=str('[H][H]'), multiplicity=1, charge=0)
    h2.initial_xyz = h2_xyz

    hcn = ARCSpecies(label='HCN', smiles=str('C#N'), multiplicity=1, charge=0)
    hcn.initial_xyz = hcn_xyz

    hf = ARCSpecies(label='HF', smiles=str('F'), multiplicity=1, charge=0)
    hf.initial_xyz = hf_xyz

    n2o = ARCSpecies(label='N2O', smiles=str('[N-]=[N+]=O'), multiplicity=1, charge=0)
    n2o.initial_xyz = n2o_xyz

    n2 = ARCSpecies(label='N2', smiles=str('N#N'), multiplicity=1, charge=0)
    n2.initial_xyz = n2_xyz

    nh3 = ARCSpecies(label='NH3', smiles=str('N'), multiplicity=1, charge=0)
    nh3.initial_xyz = nh3_xyz

    oh = ARCSpecies(label='OH', smiles=str('[OH]'), multiplicity=2, charge=0)
    oh.initial_xyz = oh_xyz

    cl2 = ARCSpecies(label='Cl2', smiles=str('[Cl][Cl]'), multiplicity=1, charge=0)
    cl2.initial_xyz = cl2_xyz

    species_list = [c2h2, ch4, co2, co, f2, ch2o, h2o, h2, hcn, hf, n2o, n2, nh3, oh, cl2]

    return species_list


def get_zpe(path):
    """
    Determine the calculated ZPE from a frequency output file"

    Args:
        path (str, unicode): The path to a frequency calculation output file.

    Returns:
        float: The calculated zero point energy in J/mol.
    """
    statmech_log = determine_qm_software(path)
    zpe = statmech_log.loadZeroPointEnergy()
    return zpe


def rename_level(level):
    """
    Rename the level of theory so it can be used for folder names.

    Args:
        level (str, unicode): The level of theory to be renamed.

    Returns:
        str, unicode: The renamed level of theory
    """
    level=level.replace('/', '_')
    level=level.replace('*', 's')
    level=level.replace('+', 'p')
    level=level.replace('(', 'b')
    level=level.replace(')', 'b')
    level=level.replace(')', 'b')
    level=level.replace(', ', 'c')
    return level
