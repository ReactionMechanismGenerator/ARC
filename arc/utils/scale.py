"""
Determine scaling factors for a given list of levels of theory

Based on DOI: 10.1016/j.cpc.2016.09.004
Adapted by Duminda Ranasinghe and Alon Grinberg Dana
"""

import os
import time
from typing import List, Optional, Union
import shutil

from arc.common import (ARC_PATH,
                        check_ess_settings,
                        get_logger,
                        initialize_job_types,
                        initialize_log,
                        time_lapse,
                        )
from arc.level import Level
from arc.parser import parse_zpe
from arc.scheduler import Scheduler
from arc.species.species import ARCSpecies

try:
    from arc.settings import global_ess_settings
except ImportError:
    global_ess_settings = None


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


def determine_scaling_factors(levels: List[Union[Level, dict, str]],
                              ess_settings: Optional[dict] = None,
                              init_log: Optional[bool] = True,
                              ) -> list:
    """
    Determine the zero-point energy, harmonic frequencies, and fundamental frequencies scaling factors
    for a given frequencies level of theory.

    Args:
        levels (list): A list of frequencies levels of theory for which scaling factors are determined.
                       Entries are either Level instances, dictionaries, or simple string representations.
                       If a single entry is given, it will be converted to a list.
        ess_settings (dict, optional): A dictionary of available ESS (keys) and a corresponding server list (values).
        init_log (bool, optional): Whether to initialize the logger. ``True`` to initialize.
                                   Should be ``True`` when called as a standalone, but ``False`` when called within ARC.

    Returns:
        list: The determined frequency scaling factors.
    """
    if init_log:
        initialize_log(log_file='scaling_factor.log', project='Scaling Factors')

    if not isinstance(levels, (list, tuple)):
        levels = [levels]
    levels = [Level(repr=level) if not isinstance(level, Level) else level for level in levels]

    t0 = time.time()

    logger.info('\n\n\n')
    logger.info(HEADER)
    logger.info('\n\nstarting ARC...\n')

    # only run opt (fine) and freq
    job_types = initialize_job_types(dict())  # get the defaults, so no job type is missing
    job_types = {job_type: False for job_type in job_types.keys()}
    job_types['opt'], job_types['fine'], job_types['freq'] = True, True, True

    lambda_zpes, zpe_dicts, times = list(), list(), list()
    for level in levels:
        t1 = time.time()
        logger.info(f'\nComputing scaling factors at the {level} level of theory...\n\n')
        renamed_level = rename_level(str(level))
        project = 'scaling_' + renamed_level
        project_directory = os.path.join(ARC_PATH, 'Projects', 'scaling_factors', project)
        logger.info(f'\nRunning in: {project_directory}\n')
        if os.path.isdir(project_directory):
            shutil.rmtree(project_directory)

        species_list = get_species_list()

        if level.method_type == 'composite':
            freq_level = None
            composite_method = level
            job_types['freq'] = False
        else:
            freq_level = level
            composite_method = None

        ess_settings = check_ess_settings(ess_settings or global_ess_settings)

        Scheduler(project=project, project_directory=project_directory, species_list=species_list,
                  composite_method=composite_method, opt_level=freq_level, freq_level=freq_level, sp_level=freq_level,
                  ess_settings=ess_settings, job_types=job_types, allow_nonisomorphic_2d=True)

        zpe_dict = dict()
        for spc in species_list:
            zpe_dict[spc.label] = parse_zpe(os.path.join(project_directory, 'output', 'Species', spc.label,
                                                         'geometry', 'freq.out')) * 1000  # convert to J/mol
        zpe_dicts.append(zpe_dict)

        lambda_zpes.append(calculate_truhlar_scaling_factors(zpe_dict=zpe_dict, level=str(level)))
        times.append(time_lapse(t1))

    summarize_results(lambda_zpes=lambda_zpes,
                      levels=levels,
                      zpe_dicts=zpe_dicts,
                      times=times,
                      overall_time=time_lapse(t0))
    logger.info('\n\n\n')
    logger.info(HEADER)

    harmonic_freq_scaling_factors = [lambda_zpe * 1.014 for lambda_zpe in lambda_zpes]
    return harmonic_freq_scaling_factors


def calculate_truhlar_scaling_factors(zpe_dict: dict,
                                      level: str,
                                      ) -> float:
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
        level (str): A string representation of the frequencies level of theory.

    Returns:
        float: The scale factor for the vibrational zero-point energy (lambda ZPE) as defined in reference [2].
    """
    unconverged = [key for key, val in zpe_dict.items() if val is None]
    if len(unconverged):
        logger.info(f'\n\nWarning: Not all species in the standard set have converged at the {level} '
                    f'level of theory!\nUnconverged species: {unconverged}\n\n')
    else:
        logger.info(f'\n\nAll species in the standard set have converged at the {level} level of theory\n\n\n')

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
#                    'NH3': 21.200 * 4184,
                    'OH': 5.2915 * 4184,
                    'Cl2': 0.7983 * 4184}

    numerator, denominator = 0.0, 0.0  # numerator and denominator in eq. 5 of reference [2]

    for label, zpe in zpe_dict.items():
        numerator += zpe * exp_zpe_dict[label] if zpe is not None else 0
        if zpe is not None:
            denominator += zpe ** 2.0
        else:
            logger.error('ZPE of species {label} could not be determined!')
    lambda_zpe = numerator / denominator  # lambda_zpe on the left side of eq. 5 of [2]

    return lambda_zpe


def summarize_results(lambda_zpes: list,
                      levels: List[Union[Level, dict, str]],
                      zpe_dicts: list,
                      times: list,
                      overall_time: str,
                      base_path: Optional[str] = None,
                      ) -> None:
    """
    Print and save the results to file.

    Args:
        lambda_zpes (list): The scale factors for the vibrational zero-point energy, entries are floats.
        levels (list): Entries are the frequency levels of theory.
        zpe_dicts (list): Entries are The calculated vibrational zero-point energies at the requested level of theory.
                          Keys are species labels, values are floats representing the ZPE in J/mol.
        times (list): Entries are string-format of the calculation execution times.
        overall_time (str): A string-format of the overall calculation execution time.
        base_path (str, optional): The path to the scaling factors base folder.
    """
    base_path = base_path or os.path.join(ARC_PATH, 'Projects', 'scaling_factors')
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
        f.write(HEADER)
        database_text = '\n\n\nYou may copy-paste the following harmonic frequency scaling factor(s) to ' \
                        'the RMG-database repository\n' \
                        '(under the `freq_dict` in RMG-database/input/quantum_corrections/data.py):\n'
        database_formats = list()
        harmonic_freq_scaling_factors = list()
        for lambda_zpe, level, zpe_dict, execution_time\
                in zip(lambda_zpes, levels, zpe_dicts, times):
            harmonic_freq_scaling_factor = lambda_zpe * 1.014
            fundamental_freq_scaling_factor = lambda_zpe * 0.974
            harmonic_freq_scaling_factors.append(fundamental_freq_scaling_factor)
            unconverged = [key for key, val in zpe_dict.items() if val is None]
            arkane_level = level.to_arkane_level_of_theory().simple()
            arkane_level_str = f"LevelOfTheory(method='{level.method}'"
            if arkane_level.basis is not None:
                arkane_level_str += f",basis='{level.basis}'"
            if arkane_level.software is not None:
                arkane_level_str += f",software='{level.software}'"
            arkane_level_str += f")"

            text = f'\n\nLevel of theory: {level}\n'
            if unconverged:
                text += f'The following species from the standard set did not converge at this level:\n {unconverged}\n'
            text += f'Scale Factor for Zero-Point Energies     = {lambda_zpe:.3f}\n'
            text += f'Scale Factor for Harmonic Frequencies    = {harmonic_freq_scaling_factor:.3f}\n'
            text += f'Scale Factor for Fundamental Frequencies = {fundamental_freq_scaling_factor:.3f}\n'
            text += f'(execution time: {execution_time})\n'
            logger.info(text)
            f.write(text)
            database_formats.append(f"""             "{arkane_level_str}": {harmonic_freq_scaling_factor:.3f},  # [4]\n""")
        logger.info(database_text)
        f.write(database_text)
        for database_format in database_formats:
            logger.info(database_format)
            f.write(database_format)
        overall_time_text = f'\n\nScaling factors calculation for {len(levels)} levels of theory completed ' \
                            f'(elapsed time: {overall_time}).\n'
        logger.info(overall_time_text)
        f.write(overall_time_text)


def get_species_list() -> list:
    """
    Generates the standardized species list.

    Returns:
        list: The standardized species list initialized with xyz.
    """
    c2h2_xyz = {'symbols': ('C', 'C', 'H', 'H'), 'isotopes': (12, 12, 1, 1),
                'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 1.203142), (0.0, -0.0, 2.265747), (-0.0, -0.0, -1.062605))}
    ch4_xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'), 'isotopes': (12, 1, 1, 1, 1),
               'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 1.08744517), (1.02525314, 0.0, -0.36248173),
                          (-0.51262658, 0.88789525, -0.36248173), (-0.51262658, -0.88789525, -0.36248173))}
    co2_xyz = {'symbols': ('C', 'O', 'O'), 'isotopes': (12, 16, 16),
               'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 1.1594846), (0.0, 0.0, -1.1594846))}
    co_xyz = {'symbols': ('O', 'C'), 'isotopes': (16, 12), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 1.12960815))}
    f2_xyz = {'symbols': ('F', 'F'), 'isotopes': (19, 19), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 1.3952041))}
    ch2o_xyz = {'symbols': ('O', 'C', 'H', 'H'), 'isotopes': (16, 12, 1, 1),
                'coords': ((0.0, 0.0, 0.674622), (0.0, 0.0, -0.529707),
                           (0.0, 0.935488, -1.109367), (0.0, -0.935488, -1.109367))}
    h2o_xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
               'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.95691441), (0.92636305, 0.0, -0.23986808))}
    h2_xyz = {'symbols': ('H', 'H'), 'isotopes': (1, 1), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.74187646))}
    hcn_xyz = {'symbols': ('C', 'N', 'H'), 'isotopes': (12, 14, 1),
               'coords': ((0.0, 0.0, -0.500365), (0.0, 0.0, 0.65264), (0.0, 0.0, -1.566291))}
    hf_xyz = {'symbols': ('F', 'H'), 'isotopes': (19, 1), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.91538107))}
    n2o_xyz = {'symbols': ('N', 'N', 'O'), 'isotopes': (14, 14, 16),
               'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 1.12056262), (0.0, 0.0, 2.30761092))}
    n2_xyz = {'symbols': ('N', 'N'), 'isotopes': (14, 14), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 1.09710935))}
    # nh3_xyz = {'symbols': ('N', 'H', 'H', 'H'), 'isotopes': (14, 1, 1, 1),
    #            'coords': ((0.0, 0.0, 0.11289), (0.0, 0.938024, -0.263409),
    #                       (0.812353, -0.469012, -0.263409), (-0.812353, -0.469012, -0.263409))}
    oh_xyz = {'symbols': ('O', 'H'), 'isotopes': (16, 1), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.967))}
    cl2_xyz = {'symbols': ('Cl', 'Cl'), 'isotopes': (35, 35), 'coords': ((0.0, 0.0, 0.995), (0.0, 0.0, -0.995))}

    c2h2 = ARCSpecies(label='C2H2', smiles='C#C', multiplicity=1, charge=0)
    c2h2.initial_xyz = c2h2_xyz

    ch4 = ARCSpecies(label='CH4', smiles='C', multiplicity=1, charge=0)
    ch4.initial_xyz = ch4_xyz

    co2 = ARCSpecies(label='CO2', smiles='O=C=O', multiplicity=1, charge=0)
    co2.initial_xyz = co2_xyz

    co = ARCSpecies(label='CO', smiles='[C-]#[O+]', multiplicity=1, charge=0)
    co.initial_xyz = co_xyz

    f2 = ARCSpecies(label='F2', smiles='[F][F]', multiplicity=1, charge=0)
    f2.initial_xyz = f2_xyz

    ch2o = ARCSpecies(label='CH2O', smiles='C=O', multiplicity=1, charge=0)
    ch2o.initial_xyz = ch2o_xyz

    h2o = ARCSpecies(label='H2O', smiles='O', multiplicity=1, charge=0)
    h2o.initial_xyz = h2o_xyz

    h2 = ARCSpecies(label='H2', smiles='[H][H]', multiplicity=1, charge=0)
    h2.initial_xyz = h2_xyz

    hcn = ARCSpecies(label='HCN', smiles='C#N', multiplicity=1, charge=0)
    hcn.initial_xyz = hcn_xyz

    hf = ARCSpecies(label='HF', smiles='F', multiplicity=1, charge=0)
    hf.initial_xyz = hf_xyz

    n2o = ARCSpecies(label='N2O', smiles='[N-]=[N+]=O', multiplicity=1, charge=0)
    n2o.initial_xyz = n2o_xyz

    n2 = ARCSpecies(label='N2', smiles='N#N', multiplicity=1, charge=0)
    n2.initial_xyz = n2_xyz

    # nh3 = ARCSpecies(label='NH3', smiles='N', multiplicity=1, charge=0)
    # nh3.initial_xyz = nh3_xyz

    oh = ARCSpecies(label='OH', smiles='[OH]', multiplicity=2, charge=0)
    oh.initial_xyz = oh_xyz

    cl2 = ARCSpecies(label='Cl2', smiles='[Cl][Cl]', multiplicity=1, charge=0)
    cl2.initial_xyz = cl2_xyz

    species_list = [c2h2, ch4, co2, co, f2, ch2o, h2o, h2, hcn, hf, n2o, n2, oh, cl2]  # removed nh3

    return species_list


def rename_level(level: str) -> str:
    """
    Rename the level of theory so it can be used for folder names.

    Args:
        level (str): The level of theory to be renamed.

    Returns:
        str: The renamed level of theory
    """
    level = level.replace('/', '_')
    level = level.replace('*', 's')
    level = level.replace('+', 'p')
    level = level.replace('(', 'b')
    level = level.replace(')', 'b')
    level = level.replace(')', 'b')
    level = level.replace(' ', '_')
    level = level.replace(':', '..')
    return level
