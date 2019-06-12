#!/usr/bin/env python
# encoding: utf-8
"""
The ARC Job module
"""


from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import csv
import logging
import shutil
import datetime

from arc.settings import arc_path, servers, submit_filename, delete_command, t_max_format,\
    input_filename, output_filename, rotor_scan_resolution, list_available_nodes_command, levels_ess
from arc.job.submit import submit_scripts
from arc.job.inputs import input_files
from arc.job.ssh import SSHClient
from arc.job.local import get_last_modified_time, submit_job, delete_job, execute_command, check_job_status,\
    rename_output
from arc.plotter import save_geo
from arc.arc_exceptions import JobError, SpeciesError

##################################################################


class Job(object):
    """
    ARC Job class. The attributes are:

    ================ =================== ===============================================================================
    Attribute        Type                Description
    ================ =================== ===============================================================================
    `project`         ``str``            The project's name. Used for naming the directory.
    `ess_settings`    ``dict``           A dictionary of available ESS and a corresponding server list
    `species_name`    ``str``            The species/TS name. Used for naming the directory.
    `charge`          ``int``            The species net charge. Default is 0
    `multiplicity`    ``int``            The species multiplicity.
    `number_of_radicals` ``int``         The number of radicals (inputted by the user, ARC won't attempt to determine
                                           it). Defaults to None. Important, e.g., if a Species is a bi-rad singlet,
                                           in which case the job should be unrestricted, but the multiplicity does not
                                           have the required information to make that decision (r vs. u)
    `spin`            ``int``            The spin. automatically derived from the multiplicity
    `xyz`             ``str``            The xyz geometry. Used for the calculation
    `n_atoms`         ``int``            The number of atoms in self.xyz
    `conformer`       ``int``            Conformer number if optimizing conformers
    `is_ts`           ``bool``           Whether this species represents a transition structure
    `level_of_theory` ``str``            Level of theory, e.g. 'CBS-QB3', 'CCSD(T)-F12a/aug-cc-pVTZ',
                                           'B3LYP/6-311++G(3df,3pd)'...
    `job_type`        ``str``            The job's type
    `scan`            ``list``           A list representing atom labels for the dihedral scan
                                           (e.g., "2 1 3 5" as a string or [2, 1, 3, 5] as a list of integers)
    `pivots`          ``list``           The rotor scan pivots, if the job type is scan. Not used directly in these
                                           methods, but used to identify the rotor.
    `scan_res`        ``int``            The rotor scan resolution in degrees
    `software`        ``str``            The electronic structure software to be used
    `server_nodes`    ``list``           A list of nodes this job was submitted to (for troubleshooting)
    `memory`          ``int``            The allocated memory (1500 MB by default)
    `method`          ``str``            The calculation method (e.g., 'B3LYP', 'CCSD(T)', 'CBS-QB3'...)
    `basis_set`       ``str``            The basis set (e.g., '6-311++G(d,p)', 'aug-cc-pVTZ'...)
    `fine`            ``bool``           Whether to use fine geometry optimization parameters
    `shift`           ``str``            A string representation alpha- and beta-spin orbitals shifts (molpro only)
    `comments`        ``str``            Job comments (archived, not used)
    `initial_time`    ``datetime``       The date-time this job was initiated. Determined automatically
    `final_time`      ``datetime``       The date-time this job was initiated. Determined automatically
    `run_time`        ``timedelta``      Job execution time. Determined automatically
    `job_status`      ``list``           The job's server and ESS statuses. Determined automatically
    `job_server_name` ``str``            Job's name on the server (e.g., 'a103'). Determined automatically
    `job_name`        ``str``            Job's name for internal usage (e.g., 'opt_a103'). Determined automatically
    `job_id`          ``int``            The job's ID determined by the server.
    `local_path`      ``str``            Local path to job's folder. Determined automatically
    `local_path_to_output_file` ``str``  The local path to the output.out file
    `local_path_to_orbitals_file` ``str``  The local path to the orbitals.fchk file (only for orbitals jobs)
    `local_path_to_check_file` ``str``   The local path to the Gaussian check file of the current job (downloaded)
    `local_path_to_lj_file`  ``str``     The local path to the lennard_jones data file (from OneDMin)
    `checkfile`       ``str``            The path to a previous Gaussian checkfile to be used in the current job
    `remote_path`     ``str``            Remote path to job's folder. Determined automatically
    `submit`          ``str``            The submit script. Created automatically
    `input`           ``str``            The input file. Created automatically
    `server`          ``str``            Server's name. Determined automatically
    'trsh'            ''str''            A troubleshooting handle to be appended to input files
    'ess_trsh_methods' ``list``          A list of troubleshooting methods already tried out for ESS convergence
    `initial_trsh`    ``dict``           Troubleshooting methods to try by default. Keys are ESS software,
                                           values are trshs
    `scan_trsh`       ``str``            A troubleshooting method for rotor scans
    `occ`             ``int``            The number of occupied orbitals (core + val) from a molpro CCSD sp calc
    `project_directory` ``str``          The path to the project directory
    `max_job_time`    ``int``            The maximal allowed job time on the server in hours
    `bath_gas`        ``str``            A bath gas. Currently used in OneDMin to calc L-J parameters.
                                           Allowed values are He, Ne, Ar, Kr, H2, N2, O2
    ================ =================== ===============================================================================

    self.job_status:
    The job server status is in job.job_status[0] and can be either 'initializing' / 'running' / 'errored' / 'done'
    The job ess (electronic structure software calculation) status is in  job.job_status[0] and can be
    either `initializing` / `running` / `errored: {error type / message}` / `unconverged` / `done`
    """
    def __init__(self, project, ess_settings, species_name, xyz, job_type, level_of_theory, multiplicity,
                 project_directory, charge=0, conformer=-1, fine=False, shift='', software=None, is_ts=False, scan='',
                 pivots=None, memory=15000, comments='', trsh='', scan_trsh='', ess_trsh_methods=None, bath_gas=None,
                 initial_trsh=None, job_num=None, job_server_name=None, job_name=None, job_id=None, server=None,
                 initial_time=None, occ=None, max_job_time=120, scan_res=None, checkfile=None, number_of_radicals=None,
                 testing=False):
        self.project = project
        self.ess_settings = ess_settings
        self.initial_time = initial_time
        self.final_time = None
        self.run_time = None
        self.species_name = species_name
        self.job_num = job_num if job_num is not None else -1
        self.charge = charge
        self.multiplicity = multiplicity
        self.spin = self.multiplicity - 1
        self.number_of_radicals = number_of_radicals
        self.xyz = xyz
        if self.xyz is None:
            raise ValueError('{0} Job of species {1} got None for xyz'.format(job_type, self.species_name))
        self.n_atoms = self.xyz.count('\n')
        self.conformer = conformer
        self.is_ts = is_ts
        self.ess_trsh_methods = ess_trsh_methods if ess_trsh_methods is not None else list()
        self.trsh = trsh
        self.initial_trsh = initial_trsh if initial_trsh is not None else dict()
        self.scan_trsh = scan_trsh
        self.scan_res = scan_res if scan_res is not None else rotor_scan_resolution
        self.max_job_time = max_job_time
        self.bath_gas = bath_gas
        self.testing = testing
        job_types = ['conformer', 'opt', 'freq', 'optfreq', 'sp', 'composite', 'scan', 'gsm', 'irc', 'ts_guess',
                     'orbitals', 'onedmin']
        # the 'conformer' job type is identical to 'opt', but we differentiate them to be identifiable in Scheduler
        if job_type not in job_types:
            raise ValueError("Job type {0} not understood. Must be one of the following:\n{1}".format(
                job_type, job_types))
        self.job_type = job_type
        if self.job_num < 0:
            self._set_job_number()
        self.job_server_name = job_server_name if job_server_name is not None else 'a' + str(self.job_num)
        self.job_name = job_name if job_name is not None else self.job_type + '_' + self.job_server_name

        # determine level of theory and software to use:
        self.level_of_theory = level_of_theory.lower()
        self.software = software
        self.method, self.basis_set = '', ''
        if '/' in self.level_of_theory:
            self.method, self.basis_set = self.level_of_theory.split('/')
        else:  # this is a composite job
            self.method, self.basis_set = self.level_of_theory, ''
        if self.software is not None:
            self.software = self.software.lower()
        else:
            if job_type == 'onedmin':
                if 'onedmin' not in self.ess_settings.keys():
                    raise JobError('Could not find the OneDMin software to compute Lennard-Jones parameters.\n'
                                   'ess_settings is:\n{0}'.format(self.ess_settings))
                self.software = 'onedmin'
                if self.bath_gas is None:
                    logging.info('Setting bath gas for Lennard-Jones calculation to N2 for species {0}'.format(
                        self.species_name))
                    self.bath_gas = 'N2'
                elif self.bath_gas not in ['He', 'Ne', 'Ar', 'Kr', 'H2', 'N2', 'O2']:
                    raise JobError('Bath gas for OneDMin should be one of the following:\n'
                                   'He, Ne, Ar, Kr, H2, N2, O2.\nGot: {0}'.format(self.bath_gas))
            elif job_type == 'orbitals':
                # currently we only have a script to print orbitals on QChem,
                # could/should definitely be elaborated to additional ESS
                if 'qchem' not in self.ess_settings.keys():
                    logging.debug('Could not find the QChem software to compute molecular orbitals.\n'
                                  'ess_settings is:\n{0}'.format(self.ess_settings))
                    self.software = None
                else:
                    self.software = 'qchem'
            elif job_type == 'composite':
                if 'gaussian' not in self.ess_settings.keys():
                    raise JobError('Could not find Gaussian to run the composite method {0}.\n'
                                   'ess_settings is:\n{1}'.format(self.method, self.ess_settings))
                self.software = 'gaussian'
            else:
                # use the levels_ess dictionary from settings.py:
                for ess, phrase_list in levels_ess.items():
                    for phrase in phrase_list:
                        if phrase in self.level_of_theory:
                            self.software = ess.lower()
            if self.software is None:
                # otherwise, deduce which software to use base on hard coded heuristics
                if job_type in ['conformer', 'opt', 'freq', 'optfreq', 'sp']:
                    if 'b2' in self.method or 'dsd' in self.method or 'pw2' in self.method:
                        # this is a double-hybrid (MP2) DFT method, use Gaussian
                        if 'gaussian' not in self.ess_settings.keys():
                            raise JobError('Could not find Gaussian to run the double-hybrid method {0}.\n'
                                           'ess_settings is:\n{1}'.format(self.method, self.ess_settings))
                        self.software = 'gaussian'
                    if 'ccs' in self.method or 'cis' in self.method or 'pv' in self.basis_set:
                        if 'molpro' in self.ess_settings.keys():
                            self.software = 'molpro'
                        elif 'gaussian' in self.ess_settings.keys():
                            self.software = 'gaussian'
                        elif 'qchem' in self.ess_settings.keys():
                            self.software = 'qchem'
                    elif 'b3lyp' in self.method:
                        if 'gaussian' in self.ess_settings.keys():
                            self.software = 'gaussian'
                        elif 'qchem' in self.ess_settings.keys():
                            self.software = 'qchem'
                        elif 'terachem' in self.ess_settings.keys():
                            self.software = 'terachem'
                        elif 'molpro' in self.ess_settings.keys():
                            self.software = 'molpro'
                    elif 'wb97xd' in self.method:
                        if 'gaussian' in self.ess_settings.keys():
                            self.software = 'gaussian'
                        elif 'terachem' in self.ess_settings.keys():
                            self.software = 'terachem'
                    elif 'wb97x-d3' in self.method:
                        if 'qchem' not in self.ess_settings.keys():
                            raise JobError('Could not find QChem to run {0}/{1}'.format(self.method, self.basis_set))
                        self.software = 'qchem'
                    elif 'wb97x' in self.method:
                        if 'terachem' in self.ess_settings.keys():
                            self.software = 'terachem'
                        elif 'gaussian' in self.ess_settings.keys():
                            self.software = 'gaussian'
                        elif 'qchem' in self.ess_settings.keys():
                            self.software = 'qchem'
                    elif 'wb97' in self.method or 'def2' in self.basis_set:
                        if 'terachem' in self.ess_settings.keys():
                            self.software = 'terachem'
                        elif 'qchem' in self.ess_settings.keys():
                            self.software = 'qchem'
                    elif 'b97' in self.method or 'def2' in self.basis_set:
                        if 'gaussian' in self.ess_settings.keys():
                            self.software = 'gaussian'
                        elif 'qchem' in self.ess_settings.keys():
                            self.software = 'qchem'
                    elif 'm062x' in self.method:  # without dash
                        if 'gaussian' not in self.ess_settings.keys():
                            raise JobError('Could not find Gaussian to run {0}/{1}'.format(self.method, self.basis_set))
                        self.software = 'gaussian'
                    elif 'm06-2x' in self.method:  # with dash
                        if 'qchem' not in self.ess_settings.keys():
                            raise JobError('Could not find QChem to run {0}/{1}'.format(self.method, self.basis_set))
                        self.software = 'qchem'
                    elif self.method in ['pbe', 'wpbe', 'pbe0']:
                        if 'terachem' in self.ess_settings.keys():
                            self.software = 'terachem'
                        elif 'qchem' in self.ess_settings.keys():
                            self.software = 'qchem'
                    elif self.method in ['svwn', 'bhandhlyp', 'b3p86', 'b3p86', 'b3lyp5', 'b3pw91', 'pw91', 'revpbe',
                                         'revpbe0', 'wpbeh', 'bop', 'mubop', 'camb3lyp', '', '', '']:
                        if 'terachem' in self.ess_settings.keys():
                            self.software = 'terachem'
                    elif '1pbe' in self.method or '2pbe' in self.method or '2pbe' in self.method:
                        if 'gaussian' in self.ess_settings.keys():
                            self.software = 'gaussian'
                elif job_type == 'scan':
                    if 'wb97xd' in self.method:
                        if 'gaussian' not in self.ess_settings.keys():
                            raise JobError('Could not find Gaussian to run {0}/{1}'.format(self.method, self.basis_set))
                        self.software = 'gaussian'
                    elif 'wb97x-d3' in self.method:
                        if 'qchem' not in self.ess_settings.keys():
                            raise JobError('Could not find QChem to run {0}/{1}'.format(self.method, self.basis_set))
                        self.software = 'qchem'
                    elif 'b3lyp' in self.method:
                        if 'gaussian' in self.ess_settings.keys():
                            self.software = 'gaussian'
                        elif 'qchem' in self.ess_settings.keys():
                            self.software = 'qchem'
                    elif 'b97' in self.method or 'def2' in self.basis_set:
                        if 'gaussian' in self.ess_settings.keys():
                            self.software = 'gaussian'
                        elif 'qchem' in self.ess_settings.keys():
                            self.software = 'qchem'
                    elif 'm06-2x' in self.method:  # with dash
                        if 'qchem' not in self.ess_settings.keys():
                            raise JobError('Could not find QChem to run {0}/{1}'.format(self.method, self.basis_set))
                        self.software = 'qchem'
                    elif 'm062x' in self.method:  # without dash
                        if 'gaussian' not in self.ess_settings.keys():
                            raise JobError('Could not find Gaussian to run {0}/{1}'.format(self.method, self.basis_set))
                        self.software = 'gaussian'
                    else:
                        if 'gaussian' in self.ess_settings.keys():
                            self.software = 'gaussian'
                        elif 'terachem' in self.ess_settings.keys():
                            self.software = 'terachem'
                        elif 'qchem' in self.ess_settings.keys():
                            self.software = 'qchem'
                        elif 'molpro' in self.ess_settings.keys():
                            self.software = 'molpro'
                elif job_type in ['gsm', 'irc']:
                    if 'gaussian' not in self.ess_settings.keys():
                        raise JobError('Could not find Gaussian to run {0}'.format(job_type))
        if self.software is None:
            # if still no software was determined, just try by order, if exists: Gaussian > QChem > Molpro
            logging.error('job_num: {0}'.format(self.job_num))
            logging.error('ess_trsh_methods: {0}'.format(self.ess_trsh_methods))
            logging.error('trsh: {0}'.format(self.trsh))
            logging.error('job_type: {0}'.format(self.job_type))
            logging.error('job_name: {0}'.format(self.job_name))
            logging.error('level_of_theory: {0}'.format(self.level_of_theory))
            logging.error('software: {0}'.format(self.software))
            logging.error('method: {0}'.format(self.method))
            logging.error('basis_set: {0}'.format(self.basis_set))
            logging.error('Could not determine software for job {0}'.format(self.job_name))
            if 'gaussian' in self.ess_settings.keys():
                logging.error('Setting it to Gaussian')
                self.software = 'gaussian'
            elif 'orca' in self.ess_settings.keys():
                logging.error('Setting it to Orca')
                self.software = 'orca'
            elif 'qchem' in self.ess_settings.keys():
                logging.error('Setting it to QChem')
                self.software = 'qchem'
            elif 'molpro' in self.ess_settings.keys():
                logging.error('Setting it to Molpro')
                self.software = 'molpro'
            elif 'terachem' in self.ess_settings.keys():
                logging.error('Setting it to TeraChem')
                self.software = 'terachem'

        self.server = server if server is not None else self.ess_settings[self.software][0]

        if self.software == 'molpro':
            # molpro's memory is in MW, 1500 MW should be enough as an initial general memory requirement assessment
            memory /= 10
        self.memory = memory

        self.fine = fine
        self.shift = shift
        self.occ = occ
        self.job_status = ['initializing', 'initializing']
        self.job_id = job_id if job_id is not None else 0
        self.comments = comments

        self.scan = scan
        self.pivots = list() if pivots is None else pivots

        conformer_folder = '' if self.conformer < 0 else os.path.join('conformers', str(self.conformer))
        folder_name = 'TSs' if self.is_ts else 'Species'
        self.project_directory = project_directory
        self.local_path = os.path.join(self.project_directory, 'calcs', folder_name,
                                       self.species_name, conformer_folder, self.job_name)
        self.local_path_to_output_file = os.path.join(self.local_path, 'output.out')
        self.local_path_to_orbitals_file = os.path.join(self.local_path, 'orbitals.fchk')
        self.local_path_to_lj_file = os.path.join(self.local_path, 'lj.dat')
        self.local_path_to_check_file = os.path.join(self.local_path, 'check.chk')
        self.checkfile = checkfile
        # parentheses don't play well in folder names:
        species_name_for_remote_path = self.species_name.replace('(', '_').replace(')', '_')
        self.remote_path = os.path.join('runs', 'ARC_Projects', self.project,
                                        species_name_for_remote_path, conformer_folder, self.job_name)
        self.submit = ''
        self.input = ''
        self.server_nodes = list()
        if job_num is None:
            # this checks jon_num and not self.job_num on purpose
            # if job_num was given, then don't save as initiated jobs, this is a restarted job
            self._write_initiated_job_to_csv_file()

    def as_dict(self):
        """A helper function for dumping this object as a dictionary in a YAML file for restarting ARC"""
        job_dict = dict()
        job_dict['initial_time'] = self.initial_time
        job_dict['final_time'] = self.final_time
        if self.run_time is not None:
            job_dict['run_time'] = self.run_time.total_seconds()
        job_dict['max_job_time'] = self.max_job_time
        job_dict['project_directory'] = self.project_directory
        job_dict['job_num'] = self.job_num
        job_dict['server'] = self.server
        job_dict['ess_trsh_methods'] = self.ess_trsh_methods
        job_dict['trsh'] = self.trsh
        job_dict['initial_trsh'] = self.initial_trsh
        job_dict['job_type'] = self.job_type
        job_dict['job_server_name'] = self.job_server_name
        job_dict['job_name'] = self.job_name
        job_dict['level_of_theory'] = self.level_of_theory
        job_dict['xyz'] = self.xyz
        job_dict['fine'] = self.fine
        job_dict['shift'] = self.shift
        job_dict['memory'] = self.memory
        job_dict['software'] = self.software
        job_dict['occ'] = self.occ
        job_dict['job_status'] = self.job_status
        if self.conformer >= 0:
            job_dict['conformer'] = self.conformer
        job_dict['job_id'] = self.job_id
        job_dict['comments'] = self.comments
        job_dict['scan'] = self.scan
        job_dict['pivots'] = self.pivots
        job_dict['scan_res'] = self.scan_res
        job_dict['scan_trsh'] = self.scan_trsh
        return job_dict

    def _set_job_number(self):
        """
        Used as the entry number in the database, as well as the job name on the server
        """
        csv_path = os.path.join(arc_path, 'initiated_jobs.csv')
        if not os.path.isfile(csv_path):
            # check file, make index file and write headers if file doesn't exists
            with open(csv_path, 'wb') as f:
                writer = csv.writer(f, dialect='excel')
                row = ['job_num', 'project', 'species_name', 'conformer', 'is_ts', 'charge', 'multiplicity', 'job_type',
                       'job_name', 'job_id', 'server', 'software', 'memory', 'method', 'basis_set', 'comments']
                writer.writerow(row)
        with open(csv_path, 'rb') as f:
            reader = csv.reader(f, dialect='excel')
            job_num = 0
            for _ in reader:
                job_num += 1
                if job_num == 100000:
                    job_num = 0
            self.job_num = job_num

    def _write_initiated_job_to_csv_file(self):
        """
        Write an initiated ARCJob into the initiated_jobs.csv file.
        """
        csv_path = os.path.join(arc_path, 'initiated_jobs.csv')
        if self.conformer < 0:  # this is not a conformer search job
            conformer = '-'
        else:
            conformer = str(self.conformer)
        with open(csv_path, 'ab') as f:
            writer = csv.writer(f, dialect='excel')
            row = [self.job_num, self.project, self.species_name, conformer, self.is_ts, self.charge,
                   self.multiplicity, self.job_type, self.job_name, self.job_id, self.server, self.software,
                   self.memory, self.method, self.basis_set, self.comments]
            writer.writerow(row)

    def write_completed_job_to_csv_file(self):
        """
        Write a completed ARCJob into the completed_jobs.csv file.
        """
        if self.job_status != ['done', 'done']:
            self.determine_job_status()
        csv_path = os.path.join(arc_path, 'completed_jobs.csv')
        if not os.path.isfile(csv_path):
            # check file, make index file and write headers if file doesn't exists
            with open(csv_path, 'wb') as f:
                writer = csv.writer(f, dialect='excel')
                row = ['job_num', 'project', 'species_name', 'conformer', 'is_ts', 'charge', 'multiplicity', 'job_type',
                       'job_name', 'job_id', 'server', 'software', 'memory', 'method', 'basis_set', 'initial_time',
                       'final_time', 'run_time', 'job_status_(server)', 'job_status_(ESS)',
                       'ESS troubleshooting methods used', 'comments']
                writer.writerow(row)
        csv_path = os.path.join(arc_path, 'completed_jobs.csv')
        if self.conformer < 0:  # this is not a conformer search job
            conformer = '-'
        else:
            conformer = str(self.conformer)
        with open(csv_path, 'ab') as f:
            writer = csv.writer(f, dialect='excel')
            job_type = self.job_type
            if self.fine:
                job_type += ' (fine)'
            row = [self.job_num, self.project, self.species_name, conformer, self.is_ts, self.charge,
                   self.multiplicity, job_type, self.job_name, self.job_id, self.server, self.software,
                   self.memory, self.method, self.basis_set, self.initial_time, self.final_time, self.run_time,
                   self.job_status[0], self.job_status[1], self.ess_trsh_methods, self.comments]
            writer.writerow(row)

    def write_submit_script(self):
        """Write the Job's submit script"""
        un = servers[self.server]['un']  # user name
        if self.max_job_time > 9999 or self.max_job_time <= 0:
            logging.debug('Setting max_job_time to 120 hours')
            self.max_job_time = 120
        if t_max_format[servers[self.server]['cluster_soft']] == 'days':
            # e.g., 5-0:00:00
            d, h = divmod(self.max_job_time, 24)
            t_max = '{0}-{1}:00:00'.format(d, h)
        elif t_max_format[servers[self.server]['cluster_soft']] == 'hours':
            # e.g., 120:00:00
            t_max = '{0}:00:00'.format(self.max_job_time)
        else:
            raise JobError('Could not determine format for maximal job time.\n Format is determined by {0}, but '
                           'got {1} for {2}'.format(t_max_format, servers[self.server]['cluster_soft'], self.server))
        cpus = servers[self.server]['cpus'] if 'cpus' in servers[self.server] else 8
        architecture = ''
        if self.server.lower() == 'pharos':
            # here we're hard-coding ARC for Pharos, a Green Group server
            # If your server has different node architectures, implement something similar
            if cpus <= 8:
                architecture = '\n#$ -l harpertown'
            else:
                architecture = '\n#$ -l magnycours'
        try:
            self.submit = submit_scripts[self.server][self.software.lower()].format(
                name=self.job_server_name, un=un, t_max=t_max, mem_cpu=int(self.memory / cpus), cpus=cpus,
                architecture=architecture)
        except KeyError:
            logging.error('Could not find submit script for server {0}, make sure your submit scripts '
                          '(under arc/job/submit.py) are updated with the servers defined.'.format(self.server))
            raise
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        with open(os.path.join(self.local_path, submit_filename[servers[self.server]['cluster_soft']]), 'wb') as f:
            f.write(self.submit)
        if self.server != 'local' and not self.testing:
            self._upload_submit_file()

    def write_input_file(self):
        """
        Write a software-specific job-specific input file.
        Saves the file locally and also uploads it to the server.
        """
        if self.initial_trsh and not self.trsh:
            # use the default trshs defined by the user in the initial_trsh dictionary
            if self.software in self.initial_trsh:
                self.trsh = self.initial_trsh[self.software]

        self.input = input_files[self.software]

        slash = ''
        if self.software == 'gaussian' and '/' in self.level_of_theory:
            slash = '/'

        dispersion = ''
        if self.software == 'terachem':
            if self.method[-2:] == 'd2':
                dispersion = 'd2'
                self.method = self.method[:-2]
            elif self.method[-2:] == 'd3':
                dispersion = 'd3'
                self.method = self.method[:-2]
            elif self.method[-1:] == 'd':
                dispersion = 'yes'
                self.method = self.method[:-2]
            else:
                dispersion = 'no'

        if (self.multiplicity > 1 and '/' in self.level_of_theory) or self.number_of_radicals > 1:
            # don't add 'u' to composite jobs. Do add 'u' for bi-rad singlets if `number_of_radicals` > 1
            if self.number_of_radicals > 1:
                logging.info('Using an unrestricted method for species {0} which has {1} radicals and '
                             'multiplicity {2}'.format(self.species_name, self.number_of_radicals, self.multiplicity))
            if self.software == 'qchem':
                restricted = 'True'  # In QChem this attribute is "unrestricted"
            else:
                restricted = 'u'
        else:
            if self.software == 'qchem':
                restricted = 'False'  # In QChem this attribute is "unrestricted"
            else:
                restricted = ''

        job_type_1, job_type_2, fine = '', '', ''

        # 'vdz' troubleshooting in molpro:
        if self.software == 'molpro' and self.trsh == 'vdz':
            self.trsh = ''
            self.input = """***,name
memory,{memory},m;
geometry={{angstrom;
{xyz}
}}

basis=cc-pVDZ
int;
{{hf;wf,spin={spin},charge={charge};}}
{restricted}{method};

basis={basis}
int;
{{hf;{shift}
maxit,1000;
wf,spin={spin},charge={charge};}}

{restricted}{method};
{job_type_1}
{job_type_2}
---;"""

        if self.job_type in ['conformer', 'opt']:
            if self.software == 'gaussian':
                if self.is_ts:
                    job_type_1 = 'opt=(ts, calcfc, noeigentest, maxstep=5)'
                else:
                    job_type_1 = 'opt=(calcfc, noeigentest)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
                if self.fine:
                    # Note that the Acc2E argument is not available in Gaussian03
                    fine = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                    if self.is_ts:
                        job_type_1 = 'opt=(ts, calcfc, noeigentest, tight, maxstep=5)'
                    else:
                        job_type_1 = 'opt=(calcfc, noeigentest, tight)'
            elif self.software == 'qchem':
                if self.is_ts:
                    job_type_1 = 'ts'
                else:
                    job_type_1 = 'opt'
                if self.fine:
                    fine = '\n   GEOM_OPT_TOL_GRADIENT 15\n   GEOM_OPT_TOL_DISPLACEMENT 60\n   GEOM_OPT_TOL_ENERGY 5'
                    if 'b' in self.level_of_theory:
                        # Try to capture DFT levels (containing the letter 'b'?), and det a fine DFT grid
                        # See 4.4.5.2 Standard Quadrature Grids, S in
                        # http://www.q-chem.com/qchem-website/manual/qchem50_manual/sect-DFT.html
                        fine += '\n   XC_GRID 3'
            elif self.software == 'molpro':
                if self.is_ts:
                    job_type_1 = "\noptg, root=2, method=qsd, readhess, savexyz='geometry.xyz'"
                else:
                    job_type_1 = "\noptg, savexyz='geometry.xyz'"
            elif self.software == 'terachem':
                if self.is_ts:
                    job_type_1 = 'ts'
                else:
                    job_type_1 = 'minimize'
                if self.fine:
                    fine = '4\ndynamicgrid yes'  # corresponds to ~60,000 grid points/atom
                else:
                    fine = '1'  # default, corresponds to ~800 grid points/atom

        elif self.job_type == 'orbitals' and self.software == 'qchem':
            if self.is_ts:
                job_type_1 = 'ts'
            else:
                job_type_1 = 'opt'
            if 'PRINT_ORBITALS' not in self.trsh:
                self.trsh += '\n   NBO           TRUE\n   RUN_NBO6      TRUE\n   ' \
                             'PRINT_ORBITALS  TRUE\n   GUI           2'

        elif self.job_type == 'freq':
            if self.software == 'gaussian':
                job_type_2 = 'freq iop(7/33=1) scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
            elif self.software == 'qchem':
                job_type_1 = 'freq'
            elif self.software == 'molpro':
                job_type_1 = '\n{frequencies;\nthermo;\nprint,HESSIAN,thermo;}'
            elif self.software == 'terachem':
                job_type_1 = 'frequencies'

        elif self.job_type == 'optfreq':
            if self.software == 'gaussian':
                if self.is_ts:
                    job_type_1 = 'opt=(ts, calcfc, noeigentest, maxstep=5)'
                else:
                    job_type_1 = 'opt=(calcfc, noeigentest)'
                job_type_2 = 'freq iop(7/33=1)'
                if self.fine:
                    fine = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                    if self.is_ts:
                        job_type_1 = 'opt=(ts, calcfc, noeigentest, tight, maxstep=5)'
                    else:
                        job_type_1 = 'opt=(calcfc, noeigentest, tight)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
            elif self.software == 'qchem':
                self.input += """@@@

$molecule
   read
$end

$rem
   JOBTYPE       {job_type_2}
   METHOD        {method}
   UNRESTRICTED  {restricted}
   BASIS         {basis}
   SCF_GUESS     read
$end

"""
                if self.is_ts:
                    job_type_1 = 'ts'
                else:
                    job_type_1 = 'opt'
                job_type_2 = 'freq'
                if self.fine:
                    fine = '\n   GEOM_OPT_TOL_GRADIENT 15\n   GEOM_OPT_TOL_DISPLACEMENT 60\n   GEOM_OPT_TOL_ENERGY 5'
            elif self.software == 'molpro':
                if self.is_ts:
                    job_type_1 = "\noptg,root=2,method=qsd,readhess,savexyz='geometry.xyz'"
                else:
                    job_type_1 = "\noptg,savexyz='geometry.xyz"
                job_type_2 = '\n{frequencies;\nthermo;\nprint,HESSIAN,thermo;}'

        if self.job_type == 'sp':
            if self.software == 'gaussian':
                job_type_1 = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
            elif self.software == 'qchem':
                job_type_1 = 'sp'
            elif self.software == 'molpro':
                pass
            elif self.software == 'terachem':
                job_type_1 = 'energy'

        if self.job_type == 'composite':
            if self.software == 'gaussian':
                if self.fine:
                    fine = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                if self.is_ts:
                    job_type_1 = 'opt=(ts, calcfc, noeigentest, tight, maxstep=5)'
                else:
                    if self.level_of_theory in ['rocbs-qb3']:
                        # No analytic 2nd derivatives (FC) for these methods
                        job_type_1 = 'opt=(noeigentest, tight)'
                    else:
                        job_type_1 = 'opt=(calcfc, noeigentest, tight)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
            else:
                raise JobError('Currently composite methods are only supported in gaussian')

        if self.job_type == 'scan':
            if self.software == 'gaussian':
                if self.is_ts:
                    job_type_1 = 'opt=(ts, modredundant, calcfc, noeigentest, maxStep=5) scf=(tight, direct)' \
                                 ' integral=(grid=ultrafine, Acc2E=12)'
                else:
                    job_type_1 = 'opt=(modredundant, calcfc, noeigentest, maxStep=5) scf=(tight, direct)' \
                                 ' integral=(grid=ultrafine, Acc2E=12)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
                scan_string = ''.join([str(num) + ' ' for num in self.scan])
                if not divmod(360, self.scan_res):
                    raise JobError('Scan job got an illegal rotor scan resolution of {0}'.format(self.scan_res))
                scan_string = 'D ' + scan_string + 'S ' + str(int(360 / self.scan_res)) + ' ' +\
                              '{0:10}'.format(float(self.scan_res))
            else:
                raise ValueError('Currently rotor scan is only supported in gaussian. Got: {0} using the {1} level of'
                                 ' theory'.format(self.software, self.method + '/' + self.basis_set))
        else:
            scan_string = ''

        if self.software == 'gaussian' and not self.trsh:
            if self.level_of_theory[:2] == 'ro':
                self.trsh = 'use=L506'
            else:
                # xqc will do qc (quadratic convergence) if the job fails w/o it, so use by default
                self.trsh = 'scf=xqc'

        if self.software == 'terachem':
            # TeraChem requires an additional xyz file
            # note: the xyz filename must correspond to the xyz filename specified in TeraChem's input file
            save_geo(xyz_coordinates=self.xyz, path=self.local_path, filename='coord', format_='xyz')

        if self.job_type == 'irc':  # TODO
            pass

        if self.job_type == 'gsm':  # TODO
            pass

        if 'mrci' in self.method:
            if self.software != 'molpro':
                raise JobError('Can only run MRCI on Molpro, not {0}'.format(self.software))
            if self.occ > 16:
                raise JobError('Will not execute an MRCI calculation with more than 16 occupied orbitals.'
                               'Selective occ, closed, core, frozen keyword still not implemented.')
            else:
                try:
                    self.input = input_files['mrci'].format(memory=self.memory, xyz=self.xyz, basis=self.basis_set,
                                                            spin=self.spin, charge=self.charge, trsh=self.trsh)
                except KeyError:
                    logging.error('Could not interpret all input file keys in\n{0}'.format(self.input))
                    raise
        else:
            try:
                cpus = servers[self.server]['cpus'] if 'cpus' in servers[self.server] else 8
                self.input = self.input.format(memory=self.memory, method=self.method, slash=slash, bath=self.bath_gas,
                                               basis=self.basis_set, charge=self.charge, multiplicity=self.multiplicity,
                                               spin=self.spin, xyz=self.xyz, job_type_1=job_type_1, cpus=cpus,
                                               job_type_2=job_type_2, scan=scan_string, restricted=restricted,
                                               fine=fine, shift=self.shift, trsh=self.trsh, scan_trsh=self.scan_trsh,
                                               dispersion=dispersion)
            except KeyError:
                logging.error('Could not interpret all input file keys in\n{0}'.format(self.input))
                raise
        if not self.testing:
            if not os.path.exists(self.local_path):
                os.makedirs(self.local_path)
            with open(os.path.join(self.local_path, input_filename[self.software]), 'w') as f:
                f.write(self.input)
            if self.server != 'local':
                self._upload_input_file()
            else:
                self.initial_time = get_last_modified_time(
                    file_path=os.path.join(self.local_path, input_filename[self.software]))
            if self.checkfile is not None and os.path.isfile(self.checkfile):
                self._upload_check_file(local_check_file_path=self.checkfile)

    def _upload_submit_file(self):
        ssh = SSHClient(self.server)
        ssh.send_command_to_server(command='mkdir -p {0}'.format(self.remote_path))
        remote_file_path = os.path.join(self.remote_path, submit_filename[servers[self.server]['cluster_soft']])
        ssh.upload_file(remote_file_path=remote_file_path, file_string=self.submit)

    def _upload_input_file(self):
        ssh = SSHClient(self.server)
        ssh.send_command_to_server(command='mkdir -p {0}'.format(self.remote_path))
        remote_file_path = os.path.join(self.remote_path, input_filename[self.software])
        ssh.upload_file(remote_file_path=remote_file_path, file_string=self.input)
        if self.software == 'onedmin':
            # also create and upload a geometry file
            local_geo_path = os.path.join(self.local_path, 'geo.xyz')
            remote_geo_path = os.path.join(self.remote_path, 'geo.xyz')
            with open(local_geo_path, 'w') as f:
                f.write(self.xyz)
            ssh.upload_file(remote_file_path=remote_geo_path, local_file_path=local_geo_path)
            # also create and upload the molpro directives
            remote_mx_path = os.path.join(self.remote_path, 'm.x')
            ssh.upload_file(remote_file_path=remote_mx_path, file_string=input_files['onedmin.molpro.x'])
            remote_qcmol_path = os.path.join(self.remote_path, 'qc.mol')
            ssh.upload_file(remote_file_path=remote_qcmol_path, file_string=input_files['onedmin.qc.mol'])
            # make the m.x file executable
            ssh.send_command_to_server(command='chmod +x m.x', remote_path=self.remote_path)
        if self.software == 'terachem':
            local_geo_path = os.path.join(self.local_path, 'coord.xyz')
            remote_geo_path = os.path.join(self.remote_path, 'coord.xyz')
            ssh.upload_file(remote_file_path=remote_geo_path, local_file_path=local_geo_path)
        self.initial_time = ssh.get_last_modified_time(remote_file_path=remote_file_path)

    def _upload_check_file(self, local_check_file_path=None):
        if self.server != 'local':
            ssh = SSHClient(self.server)
            remote_check_file_path = os.path.join(self.remote_path, 'check.chk')
            local_check_file_path = os.path.join(self.local_path, 'check.chk') if remote_check_file_path is None\
                else local_check_file_path
            if os.path.isfile(local_check_file_path) and self.software.lower() == 'gaussian':
                ssh.upload_file(remote_file_path=remote_check_file_path, local_file_path=local_check_file_path)
                logging.debug('uploading checkpoint file for {0}'.format(self.job_name))
        else:
            # running locally, just copy the check file to the job folder
            new_check_file_path = os.path.join(self.local_path, 'check.chk')
            shutil.copyfile(local_check_file_path, new_check_file_path)

    def _download_output_file(self):
        """Download ESS output, orbitals check file, and the Gaussian check file, if relevant"""
        ssh = SSHClient(self.server)

        # download output file
        remote_file_path = os.path.join(self.remote_path, output_filename[self.software])
        ssh.download_file(remote_file_path=remote_file_path, local_file_path=self.local_path_to_output_file)
        if not os.path.isfile(self.local_path_to_output_file):
            raise JobError('output file for {0} was not downloaded properly'.format(self.job_name))
        self.final_time = ssh.get_last_modified_time(remote_file_path=remote_file_path)

        # download orbitals FChk file
        if self.job_type == 'orbitals':
            remote_file_path = os.path.join(self.remote_path, 'input.FChk')
            ssh.download_file(remote_file_path=remote_file_path, local_file_path=self.local_path_to_orbitals_file)
            if not os.path.isfile(self.local_path_to_orbitals_file):
                logging.warning('Orbitals FChk file for {0} was not downloaded properly '
                                '(this is not the Gaussian formatted check file...)'.format(self.job_name))

        # download Gaussian check file
        if self.software.lower() == 'gaussian':
            remote_check_file_path = os.path.join(self.remote_path, 'check.chk')
            ssh.download_file(remote_file_path=remote_check_file_path, local_file_path=self.local_path_to_check_file)
            if not os.path.isfile(self.local_path_to_check_file):
                logging.warning('Gaussian check file for {0} was not downloaded properly'.format(self.job_name))

        # download Lennard_Jones data file
        if self.software.lower() == 'onedmin':
            remote_lj_file_path = os.path.join(self.remote_path, 'lj.dat')
            ssh.download_file(remote_file_path=remote_lj_file_path, local_file_path=self.local_path_to_lj_file)
            if not os.path.isfile(self.local_path_to_lj_file):
                logging.warning('Lennard-Jones data file for {0} was not downloaded properly'.format(self.job_name))

        # download molpro log file (in addition to the output file)
        if self.software.lower() == 'molpro':
            remote_log_file_path = os.path.join(self.remote_path, 'input.log')
            local_log_file_path = os.path.join(self.local_path, 'output.log')
            ssh.download_file(remote_file_path=remote_log_file_path, local_file_path=local_log_file_path)
            if not os.path.isfile(local_log_file_path):
                logging.warning('Could not download Molpro log file for {0} '
                                '(this is not the output file)'.format(self.job_name))

    def run(self):
        """Execute the Job"""
        if self.fine:
            logging.info('Running job {name} for {label} (fine opt)'.format(name=self.job_name,
                                                                            label=self.species_name))
        elif self.pivots:
            logging.info('Running job {name} for {label} (pivots: {pivots})'.format(name=self.job_name,
                                                                                    label=self.species_name,
                                                                                    pivots=self.pivots))
        else:
            logging.info('Running job {name} for {label}'.format(name=self.job_name, label=self.species_name))
        logging.debug('writing submit script...')
        self.write_submit_script()
        logging.debug('writing input file...')
        self.write_input_file()
        if self.server != 'local':
            ssh = SSHClient(self.server)
            logging.debug('submitting job...')
            # submit_job returns job server status and job server id
            try:
                self.job_status[0], self.job_id = ssh.submit_job(remote_path=self.remote_path)
            except IndexError:
                # if the connection broke, the files might not have been uploaded correctly
                self.write_submit_script()
                self.write_input_file()
                self.job_status[0], self.job_id = ssh.submit_job(remote_path=self.remote_path)
        else:
            # running locally
            self.job_status[0], self.job_id = submit_job(path=self.local_path)

    def delete(self):
        """Delete a running Job"""
        logging.debug('Deleting job {name} for {label}'.format(name=self.job_name, label=self.species_name))
        if self.server != 'local':
            ssh = SSHClient(self.server)
            logging.debug('deleting job on {0}...'.format(self.server))
            ssh.delete_job(self.job_id)
        else:
            logging.debug('deleting job locally...')
            delete_job(job_id=self.job_id)

    def determine_job_status(self):
        """Determine the Job's status"""
        if self.job_status[0] == 'errored':
            return
        server_status = self._check_job_server_status()
        ess_status = ''
        if server_status == 'done':
            try:
                ess_status = self._check_job_ess_status()  # also downloads output file
            except IOError:
                logging.error('Got an IOError when trying to download output file for job {0}.'.format(self.job_name))
                content = self._get_additional_job_info()
                if content:
                    logging.info('Got the following information from the server:')
                    logging.info(content)
                    for line in content.splitlines():
                        # example:
                        # slurmstepd: *** JOB 7752164 CANCELLED AT 2019-03-27T00:30:50 DUE TO TIME LIMIT on node096 ***
                        if 'cancelled' in line.lower() and 'due to time limit' in line.lower():
                            logging.warning('Looks like the job was cancelled on {0} due to time limit. '
                                            'Got: {1}'.format(self.server, line))
                            new_max_job_time = self.max_job_time - 24 if self.max_job_time > 25 else 1
                            logging.warning('Setting max job time to {0} (was {1})'.format(new_max_job_time,
                                                                                           self.max_job_time))
                            self.max_job_time = new_max_job_time
                raise
        elif server_status == 'running':
            ess_status = 'running'
        self.job_status = [server_status, ess_status]

    def _get_additional_job_info(self):
        """
        Download the additional information of stdout and stderr from the server
        """
        lines1, lines2 = list(), list()
        content = ''
        cluster_soft = servers[self.server]['cluster_soft'].lower()
        if cluster_soft in ['oge', 'sge']:
            local_file_path1 = os.path.join(self.local_path, 'out.txt')
            local_file_path2 = os.path.join(self.local_path, 'err.txt')
            if self.server != 'local':
                ssh = SSHClient(self.server)
                remote_file_path = os.path.join(self.remote_path, 'out.txt')
                try:
                    ssh.download_file(remote_file_path=remote_file_path, local_file_path=local_file_path1)
                except (TypeError, IOError) as e:
                    logging.warning('Got the following error when trying to download out.txt for {0}:'.format(
                        self.job_name))
                    logging.warning(e.message)
                remote_file_path = os.path.join(self.remote_path, 'err.txt')
                try:
                    ssh.download_file(remote_file_path=remote_file_path, local_file_path=local_file_path2)
                except (TypeError, IOError) as e:
                    logging.warning('Got the following error when trying to download err.txt for {0}:'.format(
                        self.job_name))
                    logging.warning(e.message)
            if os.path.isfile(local_file_path1):
                with open(local_file_path1, 'r') as f:
                    lines1 = f.readlines()
            if os.path.isfile(local_file_path2):
                with open(local_file_path2, 'r') as f:
                    lines2 = f.readlines()
            content += ''.join([line for line in lines1])
            content += '\n'
            content += ''.join([line for line in lines2])
        elif cluster_soft == 'slurm':
            if self.server != 'local':
                ssh = SSHClient(self.server)
                response = ssh.send_command_to_server(command='ls -alF', remote_path=self.remote_path)
            else:
                response = execute_command('ls -alF {0}'.format(self.local_path))
            files = list()
            for line in response[0][0].splitlines():
                files.append(line.split()[-1])
            for file_name in files:
                if 'slurm' in file_name and '.out' in file_name:
                    local_file_path = os.path.join(self.local_path, file_name)
                    if self.server != 'local':
                        remote_file_path = os.path.join(self.remote_path, file_name)
                        try:
                            ssh.download_file(remote_file_path=remote_file_path, local_file_path=local_file_path)
                        except (TypeError, IOError) as e:
                            logging.warning('Got the following error when trying to download {0} for {1}:'.format(
                                file_name, self.job_name))
                            logging.warning(e.message)
                    if os.path.isfile(local_file_path):
                        with open(local_file_path, 'r') as f:
                            lines1 = f.readlines()
                    content += ''.join([line for line in lines1])
                    content += '\n'
        return content

    def _check_job_server_status(self):
        """
        Possible statuses: `initializing`, `running`, `errored on node xx`, `done`
        """
        if self.server != 'local':
            ssh = SSHClient(self.server)
            return ssh.check_job_status(self.job_id)
        else:
            return check_job_status(self.job_id)

    def _check_job_ess_status(self):
        """
        Check the status of the job ran by the electronic structure software (ESS)
        Possible statuses: `initializing`, `running`, `errored: {error type / message}`, `unconverged`, `done`
        """
        if self.server != 'local':
            if os.path.exists(self.local_path_to_output_file):
                os.remove(self.local_path_to_output_file)
            if os.path.exists(self.local_path_to_orbitals_file):
                os.remove(self.local_path_to_orbitals_file)
            if os.path.exists(self.local_path_to_check_file):
                os.remove(self.local_path_to_check_file)
            self._download_output_file()  # also downloads the Gaussian check file and orbital file if exist
        else:
            # If running locally, just rename the output file to "output.out" for consistency between software
            if self.final_time is None:
                self.final_time = get_last_modified_time(
                    file_path=os.path.join(self.local_path, output_filename[self.software]))
            rename_output(local_file_path=self.local_path_to_output_file, software=self.software)
        self.determine_run_time()
        with open(self.local_path_to_output_file, 'r') as f:
            lines = f.readlines()

            if self.software == 'gaussian':
                for line in lines[-1:-20:-1]:
                    if 'Normal termination of Gaussian' in line:
                        break
                else:
                    for line in lines[::-1]:
                        if 'Error' in line or 'NtrErr' in line or 'Erroneous' in line or 'malloc' in line\
                                or 'galloc' in line:
                            reason = ''
                            if 'l9999.exe' in line or 'l103.exe' in line:
                                return 'unconverged'
                            elif 'l502.exe' in line:
                                return 'unconverged SCF'
                            elif 'l103.exe' in line:
                                return 'l103 internal coordinate error'
                            elif 'Erroneous write' in line or 'Write error in NtrExt1' in line:
                                reason = 'Ran out of disk space.'
                            elif 'l716.exe' in line:
                                reason = 'Angle in z-matrix outside the allowed range 0 < x < 180.'
                            elif 'l301.exe' in line:
                                reason = 'Input Error. Either charge, multiplicity, or basis set was not specified ' \
                                         'correctly. Or, an atom specified does not match any standard atomic symbol.'
                            elif 'NtrErr Called from FileIO' in line:
                                reason = 'Operation on .chk file was specified, but .chk was not found.'
                            elif 'l101.exe' in line:
                                reason = 'Input Error. The blank line after the coordinate section is missing, ' \
                                         'or charge/multiplicity was not specified correctly.'
                            elif 'l202.exe' in line:
                                reason = 'During the optimization process, either the standard orientation ' \
                                         'or the point group of the molecule has changed.'
                            elif 'l401.exe' in line:
                                reason = 'The projection from the old to the new basis set has failed.'
                            elif 'malloc failed' in line or 'galloc' in line:
                                reason = 'Memory allocation failed (did you ask for too much?)'
                            elif 'A SYNTAX ERROR WAS DETECTED' in line:
                                reason = 'Check .inp carefully for syntax errors in keywords.'
                            return 'errored: {0}; {1}'.format(line, reason)
                    return 'errored: Unknown reason'
                return 'done'

            elif self.software == 'qchem':
                done = False
                error_message = ''
                for line in lines[::-1]:
                    if 'Thank you very much for using Q-Chem' in line:
                        done = True
                    elif 'SCF failed' in line:
                        return 'errored: {0}'.format(line)
                    elif 'error' in line and 'DIIS' not in line:
                        # these are *normal* lines: "SCF converges when DIIS error is below 1.0E-08", or
                        # "Cycle       Energy         DIIS Error"
                        error_message = line
                    elif 'Invalid charge/multiplicity combination' in line:
                        raise SpeciesError('The multiplicity and charge combination for species {0} are wrong.'.format(
                            self.species_name))
                    if 'opt' in self.job_type or 'conformer' in self.job_type or 'ts' in self.job_type:
                        if 'MAXIMUM OPTIMIZATION CYCLES REACHED' in line:
                            return 'errored: unconverged, max opt cycles reached'
                        elif 'OPTIMIZATION CONVERGED' in line and done:  # `done` should already be assigned
                            return 'done'
                if done:
                    return 'done'
                else:
                    if error_message:
                        return 'errored: ' + error_message
                    else:
                        return 'errored: Unknown reason'

            elif self.software == 'molpro':
                for line in lines[::-1]:
                    if 'molpro calculation terminated' in line.lower()\
                            or 'variable memory released' in line.lower():
                        return 'done'
                    elif 'No convergence' in line:
                        return 'unconverged'
                    elif 'A further' in line and 'Mwords of memory are needed' in line and 'Increase memory to' in line:
                        # e.g.: `A further 246.03 Mwords of memory are needed for the triples to run.
                        # Increase memory to 996.31 Mwords.` (w/o the line break)
                        return 'errored: additional memory (mW) required: {0}'.format(line.split()[2])
                    elif 'insufficient memory available - require' in line:
                        # e.g.: `insufficient memory available - require              228765625  have
                        #        62928590
                        #        the request was for real words`
                        # add_mem = (float(line.split()[-2]) - float(prev_line.split()[0])) / 1e6
                        return 'errored: additional memory (mW) required: {0}'.format(float(line.split()[-2]) / 1e6)
                for line in lines[::-1]:
                    if 'the problem occurs' in line:
                        return 'errored: ' + line
                return 'errored: Unknown reason'

            elif self.software == 'terachem':
                for line in lines[::-1]:
                    if 'Job finished:' in line:
                        return 'done'
                    elif 'incorrect method' in line.lower():
                        return 'errored: incorrect method'
                    elif 'error: ' in line.lower():
                        # e.g.: "ERROR: Closed shell calculations can't have spin multiplicity 0."
                        return 'errored: {0}'.format(line[7:])
                    elif 'unable to open file: ' in line.lower():
                        # e.g.: "Unable to open file /<..path..>/TeraChem/basis/6-311++g[d,p]"
                        if 'basis' in line:
                            return 'errored: Could not find basis set {0} in TeraChem'.format(
                                line.split('/')[-1].replace('[', '(').replace(']', ')'))
                        else:
                            return 'errored: {0}'.format(line)
                return 'errored: Unknown reason'

        if self.software == 'onedmin':
            with open(self.local_path_to_lj_file, 'r') as f:
                lines = f.readlines()
                score = 0
                for line in lines:
                    if 'LennardJones' in line and len(line.split()) == 1:
                        score += 1
                    elif 'Epsilons[1/cm]' in line and len(line.split()) == 3:
                        score += 1
                    elif 'Sigmas[angstrom]' in line and len(line.split()) == 3:
                        score += 1
                    elif 'End' in line and len(line.split()) == 1:
                        score += 1
                if score == 4:
                    return 'done'
                else:
                    return 'errored: Unknown reason'

    def troubleshoot_server(self):
        """Troubleshoot server errors"""
        if servers[self.server]['cluster_soft'].lower() == 'oge':
            # delete present server run
            logging.error('Job {name} has server status "{stat}" on {server}. Troubleshooting by changing node.'.
                          format(name=self.job_name, stat=self.job_status[0], server=self.server))
            ssh = SSHClient(self.server)
            ssh.send_command_to_server(command=delete_command[servers[self.server]['cluster_soft']] +
                                       ' ' + str(self.job_id))
            # find available nodes
            stdout, _ = ssh.send_command_to_server(
                command=list_available_nodes_command[servers[self.server]['cluster_soft']])
            for line in stdout:
                node = line.split()[0].split('.')[0].split('node')[1]
                if servers[self.server]['cluster_soft'] == 'OGE' and '0/0/8' in line \
                        and node not in self.server_nodes:
                    self.server_nodes.append(node)
                    break
            else:
                logging.error('Could not find an available node on the server')
                # TODO: continue troubleshooting; if all else fails, put job to sleep
                #  and try again searching for a node
                return
            # modify submit file
            content = ssh.read_remote_file(remote_path=self.remote_path,
                                           filename=submit_filename[servers[self.server]['cluster_soft']])
            for i, line in enumerate(content):
                if '#$ -l h=node' in line:
                    content[i] = '#$ -l h=node{0}.cluster'.format(node)
                    break
            else:
                content.insert(7, '#$ -l h=node{0}.cluster'.format(node))
            content = ''.join(content)  # convert list into a single string, not to upset paramiko
            # resubmit
            ssh.upload_file(remote_file_path=os.path.join(self.remote_path,
                            submit_filename[servers[self.server]['cluster_soft']]), file_string=content)
            self.run()
        elif servers[self.server]['cluster_soft'].lower() == 'slurm':
            # TODO: change node on Slurm
            # delete present server run
            if self.job_status[0] != 'done':
                logging.error('Job {name} has server status "{stat}" on {server}. Re-running job.'.format(
                    name=self.job_name, stat=self.job_status[0], server=self.server))
            if self.server != 'local':
                ssh = SSHClient(self.server)
                ssh.send_command_to_server(command=delete_command[servers[self.server]['cluster_soft']] +
                                           ' ' + str(self.job_id))
            # resubmit
            self.run()

    def determine_run_time(self):
        """
        Determine the run time
        Round to seconds
        """
        if self.initial_time is not None and self.final_time is not None:
            time_delta = self.final_time - self.initial_time
            remainder = time_delta.microseconds > 5e5
            self.run_time = datetime.timedelta(seconds=time_delta.seconds + remainder)
