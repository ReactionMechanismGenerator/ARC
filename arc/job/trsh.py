#!/usr/bin/env python3
# encoding: utf-8

"""
The ARC troubleshooting ("trsh") module
"""

import os

import numpy as np

from arc.common import get_logger, determine_ess
from arc.exceptions import SpeciesError, TrshError
from arc.job.ssh import SSHClient
from arc.settings import servers, delete_command, list_available_nodes_command, submit_filename, \
    inconsistency_ab, inconsistency_az, maximum_barrier, rotor_scan_resolution
from arc.species.converter import xyz_from_data, xyz_to_coords_list
from arc.species.species import determine_rotor_symmetry
from arc.parser import parse_normal_displacement_modes, parse_xyz_from_file


logger = get_logger()


def determine_ess_status(output_path, species_label, job_type, software=None):
    """
    Determine the reason that caused an ESS job to crash, assign error keywords for troubleshooting.

    Args:
        output_path (str): The path to the ESS output file.
        species_label (str): The species label.
        job_type (str): The job type (e.g., 'opt, 'freq', 'ts', 'sp').
        software (str, optional): The ESS software.

    Returns:
        status (str): The status. Either 'done' or 'errored'.
    Returns:
        keywords (list): The standardized error keywords.
    Returns:
        error (str): A description of the error.
    Returns:
        line (str): The parsed line from the ESS output file indicating the error.
    """
    if software is None:
        software = determine_ess(log_file=output_path)

    keywords, error, = list(), ''
    with open(output_path, 'r') as f:
        lines = f.readlines()

        if len(lines) < 5:
            return 'errored', ['NoOutput'], 'Log file could not be read', ''

        if software == 'gaussian':
            for line in lines[-1:-20:-1]:
                if 'Normal termination' in line:
                    return 'done', list(), '', ''
            for i, line in enumerate(lines[::-1]):
                if 'termination' in line:
                    if 'l9999.exe' in line or 'link 9999' in line:
                        keywords = ['Unconverged', 'GL9999']  # GL stand for Gaussian Link
                        error = 'Unconverged'
                    elif 'l101.exe' in line:
                        keywords = ['InputError', 'GL101']
                        error = 'The blank line after the coordinate section is missing, ' \
                                'or charge/multiplicity was not specified correctly.'
                    elif 'l103.exe' in line:
                        keywords = ['InternalCoordinateError', 'GL103']
                        error = 'Internal coordinate error'
                    elif 'l108.exe' in line:
                        keywords = ['InputError', 'GL108']
                        error = 'There are two blank lines between z-matrix and ' \
                                'the variables, expected only one.'
                    elif 'l202.exe' in line:
                        keywords = ['OptOrientation', 'GL202']
                        error = 'During the optimization process, either the standard ' \
                                'orientation or the point group of the molecule has changed.'
                    elif 'l301.exe' in line:
                        keywords = ['GL301']
                    elif 'l401.exe' in line:
                        keywords = ['GL401']
                    elif 'l502.exe' in line:
                        keywords = ['SCF', 'GL502']
                        error = 'Unconverged SCF.'
                    elif 'l716.exe' in line:
                        keywords = ['ZMat', 'GL716']
                        error = 'Angle in z-matrix outside the allowed range 0 < x < 180.'
                    elif 'l906.exe' in line:
                        keywords = ['MP2', 'GL906']
                        error = 'The MP2 calculation has failed. It may be related to pseudopotential. ' \
                                'Basis sets (CEP-121G*) that are used with polarization functions, ' \
                                'where no polarization functions actually exist.'
                    elif 'l913.exe' in line:
                        keywords = ['MaxOptCycles', 'GL913']
                        error = 'Maximum optimization cycles reached.'
                    if any([keyword in ['GL301', 'GL401'] for keyword in keywords]):
                        additional_info = lines[len(lines) - i - 2]
                        if 'No data on chk file' in additional_info \
                                or 'Basis set data is not on the checkpoint file' in additional_info:
                            keywords = ['CheckFile']
                            error = additional_info.rstrip()
                        elif 'GL301' in keywords:
                            if 'Atomic number out of range for' in lines[len(lines) - i - 2]:
                                keywords.append('BasisSet')
                                error = f'The basis set {lines[len(lines) - i - 2].split()[6]} ' \
                                        f'is not appropriate for the this chemistry.'
                            else:
                                keywords.append('InputError')
                                error = 'Either charge, multiplicity, or basis set was not ' \
                                        'specified correctly. Alternatively, a specified atom does not match any ' \
                                        'standard atomic symbol.'
                        elif 'GL401' in keywords:
                            keywords.append('BasisSet')
                            error = 'The projection from the old to the new basis set has failed.'
                elif 'Erroneous write' in line or 'Write error in NtrExt1' in line:
                    keywords = ['DiskSpace']
                    error = 'Ran out of disk space.'
                    line = ''
                elif 'NtrErr' in line:
                    keywords = ['CheckFile']
                    error = 'An operation on the check file was specified, but a .chk was not found or is incomplete.'
                    line = ''
                elif 'malloc failed' in line or 'galloc' in line:
                    keywords = ['Memory']
                    error = 'Memory allocation failed (did you ask for too much?)'
                    line = ''
                elif 'PGFIO/stdio: No such file or directory' in line:
                    keywords = ['Scratch']
                    error = 'Wrongly specified the scratch directory. Correct the "GAUSS_SCRDIR" ' \
                            'variable in the submit script, it should point to an existing directory. ' \
                            'Make sure to add "mkdir -p $GAUSS_SCRDIR" to your submit script.'
                    line = ''
                if 'a syntax error was detected' in line.lower():
                    keywords = ['Syntax']
                    error = 'There was a syntax error in the Gaussian input file. Check your Gaussian input file ' \
                            'template under arc/job/inputs.py. Alternatively, perhaps the level of theory is not ' \
                            'supported by Gaussian in the format it was given.'
                    line = ''
                if keywords:
                    break
            error = error if error else 'Gaussian job terminated for an unknown reason. ' \
                                        'It is possible there was a server node failure.'
            keywords = keywords if keywords else ['Unknown']
            return 'errored', keywords, error, line

        elif software == 'qchem':
            done = False
            for line in lines[::-1]:
                if 'Thank you very much for using Q-Chem' in line:
                    done = True
                    # if this is an opt job, we must also check that the max num of cycles hasn't been reached,
                    # so don't break yet
                    if 'opt' not in job_type and 'conformer' not in job_type and 'ts' not in job_type:
                        break
                elif 'SCF failed' in line:
                    keywords = ['SCF']
                    error = 'SCF failed'
                    break
                elif 'error' in line and 'DIIS' not in line:
                    # these are **normal** lines that we should not capture:
                    # "SCF converges when DIIS error is below 1.0E-08", or
                    # "Cycle       Energy         DIIS Error"
                    keywords = ['SCF', 'DIIS']
                    error = 'SCF failed'
                    break
                elif 'Invalid charge/multiplicity combination' in line:
                    raise SpeciesError('The multiplicity and charge combination for species {0} are wrong.'.format(
                        species_label))
                if 'opt' in job_type or 'conformer' in job_type or 'ts' in job_type:
                    if 'MAXIMUM OPTIMIZATION CYCLES REACHED' in line:
                        keywords = ['MaxOptCycles']
                        error = 'Maximum optimization cycles reached.'
                        break
                    elif 'OPTIMIZATION CONVERGED' in line and done:  # `done` should already be assigned
                        done = True
                        break
            if done:
                return 'done', keywords, '', ''
            error = error if error else 'QChem job terminated for an unknown reason.'
            keywords = keywords if keywords else ['Unknown']
            return 'errored', keywords, error, line

        elif software == 'molpro':
            for line in lines[::-1]:
                if 'molpro calculation terminated' in line.lower() \
                        or 'variable memory released' in line.lower():
                    return 'done', list(), '', ''
                elif 'No convergence' in line:
                    keywords = ['Unconverged']
                    error = 'Unconverged'
                    break
                elif 'A further' in line and 'Mwords of memory are needed' in line and 'Increase memory to' in line:
                    # e.g.: `A further 246.03 Mwords of memory are needed for the triples to run.
                    # Increase memory to 996.31 Mwords.` (w/o the line break)
                    keywords = ['Memory']
                    error = 'Additional memory required: {0} MW'.format(line.split()[2])
                    break
                elif 'insufficient memory available - require' in line:
                    # e.g.: `insufficient memory available - require              228765625  have
                    #        62928590
                    #        the request was for real words`
                    # add_mem = (float(line.split()[-2]) - float(prev_line.split()[0])) / 1e6
                    keywords = ['Memory']
                    error = 'Additional memory required: {0} MW'.format(float(line.split()[-2]) / 1e6)
                    break
                elif 'Basis library exhausted' in line:
                    # e.g.:
                    # ` SETTING BASIS          =    6-311G**
                    #
                    #
                    #  Using spherical harmonics
                    #
                    #  LIBRARY EXHAUSTED
                    #   Searching for I  S 6-311G
                    #   Library contains the following bases:
                    #  ? Error
                    #  ? Basis library exhausted
                    #  ? The problem occurs in Binput`
                    keywords = ['BasisSet']
                    basis_set = None
                    for line0 in lines[::-1]:
                        if 'SETTING BASIS' in line0:
                            basis_set = line0.split()[-1]
                    error = f'Unrecognized basis set {basis_set}'
                    break
                elif 'the problem occurs' in line:
                    keywords = ['Unknown']
                    error = 'Unknown'
                    break
            error = error if error else 'Molpro job terminated for an unknown reason.'
            keywords = keywords if keywords else ['Unknown']
            if keywords:
                return 'errored', keywords, error, line
            return 'done', list(), '', ''


def trsh_negative_freq(label, log_file, neg_freqs_trshed=None, job_types=None):
    """
    Troubleshooting cases where non-TS species have negative frequencies.
    We take +/-1.1 displacements, generating several new initial geometries.

    Args:
        label (str): The species label.
        log_file (str): The frequency job log file.
        neg_freqs_trshed (list, optional): A list of negative frequencies the species was troubleshooted for.
        job_types (list, optional): The job types used for ARC, e.g., ['opt', 'rotors'].

    Todo:
        * get all torsions of the molecule (if weren't already generated),
          identify atom/s with largest displacements (top 2)
          determine torsions with unique PIVOTS where these atoms are in the "scan" and "top" but not pivotal
          generate a 360 scan using 30 deg increments and append all 12 results as conformers
          (consider rotor symmetry to append less conformers?)

    Returns:
        current_neg_freqs_trshed (list): The current troubleshooted negative frequencies.
    Returns:
        conformers (list): The new conformers to try optimizing.
    Returns:
        output_errors (list): Errors to report.
    Returns:
        output_warnings (list): Warnings to report.

    Raises:
        TrshError: If a negative frequency could not be determined.
    """
    neg_freqs_trshed = neg_freqs_trshed if neg_freqs_trshed is not None else list()
    job_types = job_types if job_types is not None else ['rotors']
    output_errors, output_warnings, conformers, current_neg_freqs_trshed = list(), list(), list(), list()
    factor = 1.1
    try:
        freqs, normal_disp_modes = parse_normal_displacement_modes(path=log_file)
    except NotImplementedError as e:
        logger.error(f'Could not troubleshoot negative frequency for species {label}, got:\n{e}')
        return [], [], output_errors, []
    if len(neg_freqs_trshed) > 10:
        logger.error('Species {0} was troubleshooted for negative frequencies too many times.'.format(label))
        if 'rotors' not in job_types:
            logger.error('The rotor scans feature is turned off, '
                         'cannot troubleshoot geometry using dihedral modifications.')
            output_warnings.append('rotors = False; ')
        logger.error('Invalidating species.')
        output_errors.append('Error: Encountered negative frequencies too many times; ')
    else:
        neg_freqs_idx = list()  # store indices w.r.t. vibfreqs
        largest_neg_freq_idx = 0  # index in vibfreqs
        for i, freq in enumerate(freqs):
            if freq < 0:
                neg_freqs_idx.append(i)
                if freqs[i] < freqs[largest_neg_freq_idx]:
                    largest_neg_freq_idx = i
            else:
                # assuming frequencies are ordered, break after the first positive freq encountered
                break
        if freqs[largest_neg_freq_idx] >= 0 or len(neg_freqs_idx) == 0:
            raise TrshError('Could not determine a negative frequency for species {0} '
                            'while troubleshooting for it.'.format(label))
        if len(neg_freqs_idx) == 1 and not len(neg_freqs_trshed):
            # species has one negative frequency, and has not been troubleshooted for it before
            logger.info('Species {0} has a negative frequency ({1}). Perturbing its geometry using the respective '
                        'vibrational displacements'.format(label, freqs[largest_neg_freq_idx]))
            neg_freqs_idx = [largest_neg_freq_idx]  # indices of the negative frequencies to troubleshoot for
        elif len(neg_freqs_idx) == 1 and any([np.allclose(freqs[0], vf, rtol=1e-04, atol=1e-02)
                                              for vf in neg_freqs_trshed]):
            # species has one negative frequency, and has been troubleshooted for it before
            factor = 1 + 0.1 * (len(neg_freqs_trshed) + 1)
            logger.info('Species {0} has a negative frequency ({1}) for the {2} time. Perturbing its geometry using '
                        'the respective vibrational displacements, this time using a larger factor (x {3})'.format(
                         label, freqs[largest_neg_freq_idx], len(neg_freqs_trshed), factor))
            neg_freqs_idx = [largest_neg_freq_idx]  # indices of the negative frequencies to troubleshoot for
        elif len(neg_freqs_idx) > 1 and not any([np.allclose(freqs[0], vf, rtol=1e-04, atol=1e-02)
                                                 for vf in neg_freqs_trshed]):
            # species has more than one negative frequency, and has not been troubleshooted for it before
            logger.info('Species {0} has {1} negative frequencies. Perturbing its geometry using the vibrational '
                        'displacements of its largest negative frequency, {2}'.format(label, len(neg_freqs_idx),
                                                                                      freqs[largest_neg_freq_idx]))
            neg_freqs_idx = [largest_neg_freq_idx]  # indices of the negative frequencies to troubleshoot for
        elif len(neg_freqs_idx) > 1 and any([np.allclose(freqs[0], vf, rtol=1e-04, atol=1e-02)
                                             for vf in neg_freqs_trshed]):
            # species has more than one negative frequency, and has been troubleshooted for it before
            logger.info('Species {0} has {1} negative frequencies. Perturbing its geometry using the vibrational'
                        ' displacements of ALL negative frequencies'.format(label, len(neg_freqs_idx)))
        current_neg_freqs_trshed = [round(freqs[i], 2) for i in neg_freqs_idx]  # record trshed negative freqs
        xyz = parse_xyz_from_file(log_file)
        coords = np.array(xyz_to_coords_list(xyz), np.float64)
        for neg_freq_idx in neg_freqs_idx:
            displacement = normal_disp_modes[neg_freq_idx]
            coords1 = coords + factor * displacement
            coords2 = coords - factor * displacement
            conformers.append(xyz_from_data(coords=coords1, symbols=xyz['symbols']))
            conformers.append(xyz_from_data(coords=coords2, symbols=xyz['symbols']))
    return current_neg_freqs_trshed, conformers, output_errors, output_warnings


def trsh_scan_job(label, scan_res, scan, species_scan_lists, methods):
    """
    Troubleshooting rotor scans
    Using the following methods: freezing all dihedrals other than the scan's pivots for this job,
    or increasing the scan resolution.

    Args:
        label (str): The species label.
        scan_res (int): The scan resolution in degrees.
        scan (list): The four atom indices representing the torsion to be troubleshooted.
        species_scan_lists (list): Entries are lists of four atom indices each representing a torsion.
        methods (list): The troubleshooting method/s to try. Accepted values: 'freeze' and/or 'inc_res'.

    Returns:
        scan_trsh (str): The scan troubleshooting keywords to be appended to the Gaussian input file.
    Returns:
        scan_res (int): The new scan resolution in degrees.

    Raises:
        TrshError: If troubleshooted dihedral is not found.
    """
    if methods is None:
        raise TrshError('Expected to get a list of methods, got None.')
    scan_trsh = ''
    if 'freeze' in methods:
        if scan not in species_scan_lists:
            raise TrshError('Could not find the dihedral to troubleshoot for in the scan list of species '
                            '{0}'.format(label))
        species_scan_lists.pop(species_scan_lists.index(scan))
        if len(species_scan_lists):
            scan_trsh = '\n'
            for scan in species_scan_lists:
                scan_trsh += 'D ' + ''.join([str(num) + ' ' for num in scan]) + 'F\n'
    if 'inc_res' in methods:
        scan_res = min(4, int(scan_res / 2))
        # make sure mod(360, scan res) is 0:
        if scan_res not in [4, 2, 1]:
            scan_res = min([4, 2, 1], key=lambda x: abs(x - scan_res))
    return scan_trsh, scan_res


def trsh_ess_job(label, level_of_theory, server, job_status, job_type, software, fine, memory_gb, num_heavy_atoms,
                 cpu_cores, ess_trsh_methods, available_ess=None):
    """
    Troubleshoot issues related to the electronic structure software, such as convergence.

    Args:
        label (str): The species label.
        level_of_theory (str): The level of theory to use.
        server (str): The server used for this job.
        job_status (dict): The ESS job status dictionary with standardized error keywords
                           as generated using the `determine_ess_status` function.
        job_type (str): The original job type.
        software (str, optional): The ESS software.
        fine (bool): Whether the job used an ultrafine grid, `True` if it did.
        memory_gb (float): The memory in GB used for the job.
        cpu_cores (int): The total number of cpu cores requested for a job.
        ess_trsh_methods (list, optional): The troubleshooting methods tried for this job.
        available_ess (list, optional): Entries are string representations of available ESS.
        num_heavy_atoms (int): Number of heavy atoms in a molecule.

    Todo:
        * Change server to one that has the same ESS if running out of disk space.

    Returns:
        output_errors (list): Errors to report.
    Returns:
        ess_trsh_methods (list): The updated troubleshooting methods tried for this job.
    Returns:
        remove_checkfile (bool): Whether to remove the checkfile from the job, `True` to remove.
    Returns:
        level_of_theory (str): The new level of theory to use.
    Returns:
        software (str, optional): The new ESS software to use.
    Returns:
        job_type (str): The new job type to use.
    Returns:
        fine (bool): whether the new job should use a fine grid, `True` if it should.
    Returns:
        trsh_keyword (str): The troubleshooting keyword to use.
    Returns:
        memory (float): The new memory in GB to use for the job.
    Returns:
        shift (str): The shift to use (only in Molpro).
    Returns:
        cpus (int): The total number of cpu cores requested for a job.
    Returns:
        couldnt_trsh (bool): Whether a troubleshooting solution was found. `True` if it was not found.
    """
    output_errors = list()
    remove_checkfile, couldnt_trsh = False, False
    trsh_keyword, shift = '', ''
    memory = memory_gb

    if 'DiskSpace' in job_status['keywords']:
        output_errors.append(f'Error: Could not troubleshoot {job_type} for {label}! '
                             f'The job ran out of disc space on {server}; ')
        logger.error('Could not troubleshoot {job_type} for {label}! The job ran out of disc space on '
                     '{server}'.format(job_type=job_type, label=label, server=server))
        couldnt_trsh = True
    elif 'BasisSet' in job_status['keywords']\
            and ('Unrecognized basis set' in job_status['error']
                 or 'is not appropriate for the this chemistry' in job_status['error']):
        output_errors.append(f'Error: Could not recognize basis set {job_status["error"].split()[-1]} in {software}; ')
        couldnt_trsh = True

    elif software == 'gaussian':
        if 'CheckFile' in job_status['keywords'] and 'checkfie=None' not in ess_trsh_methods:
            # The checkfile doesn't match the new basis set, remove it and rerun the job.
            logger.info('Troubleshooting {type} job in {software} for {label} that failed with '
                        '"Basis set data is not on the checkpoint file" by removing the checkfile.'.format(
                         type=job_type, software=software, label=label))
            ess_trsh_methods.append('checkfie=None')
            remove_checkfile = True
        elif 'InternalCoordinateError' in job_status['keywords'] \
                and 'cartesian' not in ess_trsh_methods and job_type == 'opt':
            # try both cartesian and nosymm
            logger.info('Troubleshooting {type} job in {software} for {label} using opt=cartesian with '
                        'nosyym'.format(type=job_type, software=software, label=label))
            ess_trsh_methods.append('cartesian')
            trsh_keyword = 'opt=(cartesian,nosymm)'
        elif 'Unconverged' in job_status['keywords'] and 'fine' not in ess_trsh_methods and not fine:
            # try a fine grid for SCF and integral
            logger.info('Troubleshooting {type} job in {software} for {label} using a fine grid'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('fine')
            fine = True
        elif 'SCF' in job_status['keywords'] and 'scf=(qc,nosymm)' not in ess_trsh_methods:
            # try both qc and nosymm
            logger.info('Troubleshooting {type} job in {software} for {label} using scf=(qc,nosymm)'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('scf=(qc,nosymm)')
            trsh_keyword = 'scf=(qc,nosymm)'
        elif 'SCF' in job_status['keywords'] and 'scf=(NDump=30)' not in ess_trsh_methods:
            # Allows dynamic dumping for up to N SCF iterations (slower conversion)
            logger.info('Troubleshooting {type} job in {software} for {label} using scf=(NDump=30)'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('scf=(NDump=30)')
            trsh_keyword = 'scf=(NDump=30)'
        elif 'SCF' in job_status['keywords'] and 'scf=NoDIIS' not in ess_trsh_methods:
            # Switching off Pulay's Direct Inversion
            logger.info('Troubleshooting {type} job in {software} for {label} using scf=NoDIIS'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('scf=NoDIIS')
            trsh_keyword = 'scf=NoDIIS'
        elif 'SCF' in job_status['keywords'] and 'scf=nosymm' not in ess_trsh_methods:
            # try running w/o considering symmetry
            logger.info('Troubleshooting {type} job in {software} for {label} using scf=nosymm'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('scf=nosymm')
            trsh_keyword = 'scf=nosymm'
        elif 'int=(Acc2E=14)' not in ess_trsh_methods:  # does not work in g03
            # Change integral accuracy (skip everything up to 1E-14 instead of 1E-12)
            logger.info('Troubleshooting {type} job in {software} for {label} using int=(Acc2E=14)'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('int=(Acc2E=14)')
            trsh_keyword = 'int=(Acc2E=14)'
        # suggest spwaning a cbs-qb3 job if there are not many heavy atoms
        elif 'cbs-qb3' not in ess_trsh_methods and level_of_theory != 'cbs-qb3' \
                and 'scan' not in job_type and num_heavy_atoms <= 10:
            # try running CBS-QB3, which is relatively robust.
            logger.info('Troubleshooting {type} job in {software} for {label} using CBS-QB3'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('cbs-qb3')
            level_of_theory = 'cbs-qb3'
            job_type = 'composite'
        elif 'Memory' in job_status['keywords'] and 'memory' not in ess_trsh_methods:
            # Increase memory allocation
            max_mem = servers[server].get('memory', 128)  # Node memory in GB, defaults to 128 if not specified
            memory = min(memory_gb * 2, max_mem * 0.9)
            logger.info('Troubleshooting {type} job in {software} for {label} using more memory: {mem} GB instead of '
                        '{old} GB'.format(type=job_type, software=software, mem=memory, old=memory_gb,
                                          label=label))
            ess_trsh_methods.append('memory')
        elif level_of_theory != 'cbs-qb3' and 'scf=(qc,nosymm) & CBS-QB3' not in ess_trsh_methods:
            # try both qc and nosymm with CBS-QB3
            logger.info('Troubleshooting {type} job in {software} for {label} using scf=(qc,nosymm) with '
                        'CBS-QB3'.format(type=job_type, software=software, label=label))
            ess_trsh_methods.append('scf=(qc,nosymm) & CBS-QB3')
            level_of_theory = 'cbs-qb3'
            trsh_keyword = 'scf=(qc,nosymm)'
        elif 'qchem' not in ess_trsh_methods and job_type != 'composite' and \
                (available_ess is None or 'qchem' in [ess.lower() for ess in available_ess]):
            # Try QChem
            logger.info('Troubleshooting {type} job using qchem instead of {software} for {label}'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('qchem')
            software = 'qchem'
        elif 'molpro' not in ess_trsh_methods and job_type not in ['composite', 'scan'] \
                and (available_ess is None or 'molpro' in [ess.lower() for ess in available_ess]):
            # Try molpro
            logger.info('Troubleshooting {type} job using molpro instead of {software} for {label}'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('molpro')
            software = 'molpro'
        else:
            couldnt_trsh = True

    elif software == 'qchem':
        if 'MaxOptCycles' in job_status['keywords'] and 'max_cycles' not in ess_trsh_methods:
            # this is a common error, increase max cycles and continue running from last geometry
            logger.info('Troubleshooting {type} job in {software} for {label} using max_cycles'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('max_cycles')
            trsh_keyword = '\n   GEOM_OPT_MAX_CYCLES 250'  # default is 50
        elif 'SCF' in job_status['keywords'] and 'DIIS_GDM' not in ess_trsh_methods:
            # change the SCF algorithm and increase max SCF cycles
            logger.info('Troubleshooting {type} job in {software} for {label} using the DIIS_GDM SCF algorithm'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('DIIS_GDM')
            trsh_keyword = '\n   SCF_ALGORITHM DIIS_GDM\n   MAX_SCF_CYCLES 1000'  # default is 50
        elif 'SYM_IGNORE' not in ess_trsh_methods:  # symmetry - look in manual, no symm if fails
            # change the SCF algorithm and increase max SCF cycles
            logger.info('Troubleshooting {type} job in {software} for {label} using SYM_IGNORE as well as the '
                        'DIIS_GDM SCF algorithm'.format(type=job_type, software=software, label=label))
            ess_trsh_methods.append('SYM_IGNORE')
            trsh_keyword = '\n   SCF_ALGORITHM DIIS_GDM\n   MAX_SCF_CYCLES 250\n   SYM_IGNORE     True'
        elif 'wB97X-D3/def2-TZVP' not in ess_trsh_methods:
            logger.info('Troubleshooting {type} job in {software} for {label} using wB97X-D3/def2-TZVP'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('wB97X-D3/def2-TZVP')
            # try converging with wB97X-D3/def2-TZVP
            level_of_theory = 'wb97x-d3/def2-tzvp'
        elif 'b3lyp/6-311++g(d,p)' not in ess_trsh_methods:
            logger.info('Troubleshooting {type} job in {software} for {label} using b3lyp/6-311++g(d,p)'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('b3lyp/6-311++g(d,p)')
            # try converging with B3LYP
            level_of_theory = 'b3lyp/6-311++g(d,p)'
        elif 'gaussian' not in ess_trsh_methods \
                and (available_ess is None or 'gaussian' in [ess.lower() for ess in available_ess]):
            # Try Gaussian
            logger.info('Troubleshooting {type} job using gaussian instead of {software} for {label}'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('gaussian')
            software = 'gaussian'
        elif 'molpro' not in ess_trsh_methods and job_type != 'scan' \
                and (available_ess is None or 'molpro' in [ess.lower() for ess in available_ess]):
            # Try molpro
            logger.info('Troubleshooting {type} job using molpro instead of {software} for {label}'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('molpro')
            software = 'molpro'
        else:
            couldnt_trsh = True

    elif 'molpro' in software:
        if 'Memory' in job_status['keywords']:
            # Increase memory allocation.
            # molpro gives something like `'errored: additional memory (mW) required: 996.31'`.
            # job_status standardizes the format to be:  `'Additional memory required: {0} MW'`
            # The number is the ADDITIONAL memory required in GB
            ess_trsh_methods.append('memory')
            add_mem = float(job_status['error'].split()[-2])  # parse Molpro's requirement in MW
            add_mem = int(np.ceil(add_mem / 100.0)) * 100  # round up to the next hundred
            memory = memory_gb + add_mem / 128. + 5  # convert MW to GB, add 5 extra GB (be conservative)
            logger.info('Troubleshooting {type} job in {software} for {label} using memory: {mem} GB instead of '
                        '{old} GB'.format(type=job_type, software=software, mem=memory, old=memory_gb,
                                          label=label))
        elif 'shift' not in ess_trsh_methods:
            # Try adding a level shift for alpha- and beta-spin orbitals
            # Applying large negative level shifts like {rhf; shift,-1.0,-0.5}
            # will often stabilize convergence at the expense of making it somewhat slower.
            logger.info('Troubleshooting {type} job in {software} for {label} using shift'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('shift')
            shift = 'shift,-1.0,-0.5;'
        elif 'vdz' not in ess_trsh_methods:
            # degrade the basis set
            logger.info('Troubleshooting {type} job in {software} for {label} using vdz'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('vdz')
            trsh_keyword = 'vdz'
        elif 'vdz & shift' not in ess_trsh_methods:
            # try adding a level shift for alpha- and beta-spin orbitals
            logger.info('Troubleshooting {type} job in {software} for {label} using vdz'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('vdz & shift')
            shift = 'shift,-1.0,-0.5;'
            trsh_keyword = 'vdz'
        elif 'memory' not in ess_trsh_methods:
            # Increase memory allocation, also run with a shift
            ess_trsh_methods.append('memory')
            memory = servers[server]['memory']  # set memory to the value of an entire node (in GB)
            logger.info('Troubleshooting {type} job in {software} for {label} using memory: {mem} GB instead of '
                        '{old} GB'.format(type=job_type, software=software, mem=memory, old=memory_gb,
                                          label=label))
            shift = 'shift,-1.0,-0.5;'
        elif 'gaussian' not in ess_trsh_methods\
                and (available_ess is None or 'gaussian' in [ess.lower() for ess in available_ess]):
            # Try Gaussian
            logger.info('Troubleshooting {type} job using gaussian instead of {software} for {label}'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('gaussian')
            software = 'gaussian'
        elif 'qchem' not in ess_trsh_methods\
                and (available_ess is None or 'qchem' in [ess.lower() for ess in available_ess]):
            # Try QChem
            logger.info('Troubleshooting {type} job using qchem instead of {software} for {label}'.format(
                type=job_type, software=software, label=label))
            ess_trsh_methods.append('qchem')
            software = 'qchem'
        else:
            couldnt_trsh = True

    if couldnt_trsh:
        logger.error('Could not troubleshoot geometry optimization for {label}! '
                     'Tried troubleshooting with the following methods: {methods}'.format(
                      label=label, methods=ess_trsh_methods))
        output_errors.append('Error: Could not troubleshoot {job_type} for {label}! '
                             'Tried troubleshooting with the following methods: {methods}; '.format(
                              job_type=job_type, label=label, methods=ess_trsh_methods))
    return output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, \
        trsh_keyword, memory, shift, cpu_cores, couldnt_trsh


def trsh_conformer_isomorphism(software, ess_trsh_methods=None):
    """
    Troubleshoot conformer optimization for a species that failed isomorphic test in
    `determine_most_stable_conformer` by specifying a "good" level of theory.

    Args:
        software (str): The ESS used.
        ess_trsh_methods (list, optional): The troubleshooting methods tried for this job.

    Returns:
        level_of_theory (str): Tte level of theory to troubleshoot at.

    Raises:
        TrshError: If the requested ``ess_trsh_methods`` is not supported.
    """
    ess_trsh_methods = ess_trsh_methods if ess_trsh_methods is not None else list()
    if software == 'gaussian':
        conformer_trsh_methods = ['wb97xd/def2TZVP', 'apfd/def2TZVP']
    elif software == 'qchem':
        conformer_trsh_methods = ['wb97x-d3/def2-TZVP']
    else:
        raise TrshError('The troubleshoot_conformer_isomorphism() method is not implemented for {0}.'.format(software))

    level_of_theory = None
    for method in conformer_trsh_methods:
        if 'conformer ' + method in ess_trsh_methods:
            continue
        ess_trsh_methods.append('conformer ' + method)
        level_of_theory = method
        break
    return level_of_theory


def trsh_job_on_server(server, job_name, job_id, job_server_status, remote_path, server_nodes=None):
    """
    Troubleshoot server errors.

    Args:
        server (str): The server name.
        job_name (str): The job's name (e.g., 'opt_a103').
        job_id (str): The job's ID on the server.
        job_server_status (str): The job server status (either 'initializing', 'running', 'errored', or 'done').
        remote_path (str): The remote path to the job folder.
        server_nodes (list, optional): The nodes already tried on this server for this jobs.

    Returns:
        str: The new node on the server (or None).
    Returns:
        bool: Whether to re-run the job, `True` to rerun.
    """
    server_nodes = server_nodes if server_nodes is not None else list()
    if job_server_status != 'done':
        logger.error('Job {name} has server status "{stat}" on {server}.'.format(
            name=job_name, stat=job_server_status, server=server))

    if servers[server]['cluster_soft'].lower() == 'oge':

        logger.error('Troubleshooting by changing node.')
        ssh = SSHClient(server)
        ssh.send_command_to_server(command=delete_command[servers[server]['cluster_soft']] + ' ' + str(job_id))
        # find available nodes
        stdout = ssh.send_command_to_server(command=list_available_nodes_command[servers[server]['cluster_soft']])[0]
        for line in stdout:
            node = line.split()[0].split('.')[0].split('node')[1]
            if servers[server]['cluster_soft'] == 'OGE' and '0/0/8' in line and node not in server_nodes:
                server_nodes.append(node)
                break
        else:
            logger.error('Could not find an available node on the server {0}'.format(server))
            # TODO: continue troubleshooting; if all else fails, put the job to sleep,
            #       and try again searching for a node
            return None, False

        # modify the submit file
        content = ssh.read_remote_file(remote_path=remote_path,
                                       filename=submit_filename[servers[server]['cluster_soft']])
        for i, line in enumerate(content):
            if '#$ -l h=node' in line:
                content[i] = '#$ -l h=node{0}.cluster'.format(node)
                break
        else:
            content.insert(7, '#$ -l h=node{0}.cluster'.format(node))
        content = ''.join(content)  # convert list into a single string, not to upset paramiko
        # resubmit
        ssh.upload_file(remote_file_path=os.path.join(remote_path,
                        submit_filename[servers[server]['cluster_soft']]), file_string=content)
        return node, True

    elif servers[server]['cluster_soft'].lower() == 'slurm':
        # TODO: change node on Slurm
        logger.error('Re-submitting job {0} on {1}'.format(job_name, server))
        # delete current server run
        ssh = SSHClient(server)
        ssh.send_command_to_server(command=delete_command[servers[server]['cluster_soft']] + ' ' + str(job_id))
        return None, True


def scan_quality_check(label, pivots, energies, scan_res=rotor_scan_resolution, used_methods=None):
    """
    Checks the scan's quality:
    - Whether the initial and final points are consistent
    - whether it is relatively "smooth"
    - whether the optimized geometry indeed represents the minimum energy conformer
    - whether the barrier height is reasonable
    Recommends whether or not to use this rotor using the 'successful_rotors' and 'unsuccessful_rotors' attributes.

    Args:
        label (str): The species label.
        pivots (list): The rotor pivots.
        energies (list): The scan energies in kJ/mol.
        scan_res (float, optional): The scan resolution in degrees.
        used_methods (list, optional): Troubleshooting methods already tried out.

    Returns:
        invalidate (bool): Whether to invalidate this rotor, ``True`` to invalidate.
    Returns:
        invalidation_reason (str): Reason for invalidating this rotor.
    Returns:
        message (str): Error or warning message.
    Returns:
        actions (list): Troubleshooting methods to apply, including conformational changes.
    """
    message, invalidation_reason = '', ''
    invalidate = False
    actions = list()
    used_methods = used_methods or list()
    energies = np.array(energies, np.float64)

    # 1. Check rotor scan curve
    # 1.1. Check consistency between initial and final points
    if abs(energies[-1] - energies[0]) > inconsistency_az:
        # initial and final points differ by more than `inconsistency_az` kJ/mol.
        # seems like this rotor broke the conformer. Invalidate
        invalidate = True
        invalidation_reason = f'initial and final points are inconsistent by more than {inconsistency_az:.2f} kJ/mol'
        message = f'Rotor scan of {label} between pivots {pivots} is inconsistent by more ' \
                  f'than {inconsistency_az:.2f} kJ/mol between initial and final positions. ' \
                  f'Invalidating rotor.\nenergies[0] = {energies[0]}, energies[-1] = {energies[-1]}'
        logger.error(message)
        actions = ['inc_res', 'freeze']
        return invalidate, invalidation_reason, message, actions

    # 1.2. Check consistency between consecutive points
    for j in range(len(energies) - 1):
        if abs(energies[j] - energies[j + 1]) > inconsistency_ab * np.max(energies):
            # Two consecutive points on the scan differ by more than `inconsistency_ab` kJ/mol.
            # This is a serious inconsistency. Invalidate
            invalidate = True
            invalidation_reason = 'Two consecutive points are inconsistent by more than ' \
                                  '{0:.2f} kJ/mol'.format(inconsistency_ab * max(energies))
            message = 'Rotor scan of {label} between pivots {pivots} is inconsistent ' \
                      'by more than {incons_ab:.2f} kJ/mol between two consecutive ' \
                      'points. Invalidating rotor.'.format(
                       label=label, pivots=pivots, incons_ab=inconsistency_ab * max(energies))
            logger.error(message)
            if ['inc_res'] not in used_methods:
                actions = ['inc_res']
            elif ['inc_res', 'freeze'] not in used_methods:
                actions = ['inc_res', 'freeze']
            return invalidate, invalidation_reason, message, actions

    # 2. Check conformation:
    energy_diff = energies[0] - np.min(energies)
    if energy_diff >= 2 or energy_diff > 0.5 * (max(energies) - min(energies)):
        invalidate = True
        invalidation_reason = 'Another conformer for {0} exists which is {1:.2f} kJ/mol lower.'.format(
                               label, energy_diff)
        message = 'Species {label} is not oriented correctly around pivots {pivots}, ' \
                  'searching for a better conformation...'.format(label=label, pivots=pivots)
        logger.info(message)
        # Find the rotation dihedral in degrees to the closest minimum:
        min_index = np.argmin(energies)
        deg_increment = min_index * scan_res
        actions = ['change conformer', deg_increment]
        return invalidate, invalidation_reason, message, actions

    # 3. Check the barrier height
    if (np.max(energies) - np.min(energies)) > maximum_barrier:
        # The barrier for the internal rotation is higher than `maximum_barrier`
        num_wells = determine_rotor_symmetry(label=label, pivots=pivots, rotor_path='', energies=energies,
                                             return_num_wells=True)[-1]
        if num_wells == 1:
            invalidate = True
            invalidation_reason = 'The rotor scan has a barrier of {0} kJ/mol, which is higher than the maximal ' \
                                  'barrier for rotation ({1} kJ/mol)'.format(
                                   np.max(energies) - np.min(energies), maximum_barrier)
            message = 'Rotor scan of {label} between pivots {pivots} has a barrier ' \
                      'larger than {max_barrier:.2f} kJ/mol. Invalidating rotor.'.format(
                       label=label, pivots=pivots, max_barrier=maximum_barrier)
            logger.warning(message)
            return invalidate, invalidation_reason, message, actions
        else:
            logger.warning(f'The maximal barrier for rotor {pivots} of {label} is '
                           f'{(np.max(energies) - np.min(energies))} kJ/mol, which is higher than the set threshold '
                           f'of {maximum_barrier} kJ/mol. Since this mode when treated as torsion has {num_wells}, '
                           f'this mode is not invalidated: treating it as a vibrational mode will be less accurate than'
                           f'the a hindered rotor treatment, since the entropy contribution from the population of '
                           f'this species at the higher wells will not be taken into account. NOT invalidating this '
                           f'torsional mode.')

    return invalidate, invalidation_reason, message, actions
