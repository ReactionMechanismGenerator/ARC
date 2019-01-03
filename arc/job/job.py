#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import datetime
import csv
import logging

from arc.settings import arc_path, servers, submit_filename, delete_command,\
    input_filename, output_filename, rotor_scan_resolution, list_available_nodes_command
from arc.job.submit import submit_scripts
from arc.job.inputs import input_files
from arc.job.ssh import SSH_Client
from arc.exceptions import JobError, SpeciesError

##################################################################


class Job(object):
    """
    ARC Job class. The attributes are:

    ================ =================== ===============================================================================
    Attribute        Type                Description
    ================ =================== ===============================================================================
    `project`         ``str``            The project's name. Used for naming the directory.
    `settings`        ``dict``           A dictionary of available servers and software
    `species_name`    ``str``            The species/TS name. Used for naming the directory.
    `charge`          ``int``            The species net charge. Default is 0
    `multiplicity`    ``int``            The species multiplicity.
    `spin`            ``int``            The spin. automatically derived from the multiplicity
    `xyz`             ``str``            The xyz geometry. Used for the calculation
    `n_atoms`         ``int``            The number of atoms in self.xyz
    `conformer`       ``int``            Conformer number if optimizing conformers
    `is_ts`           ``bool``           Whether this species represents a transition structure
    `level_of_theory` ``str``            Level of theory, e.g. 'CBS-QB3', 'CCSD(T)-F12a/aug-cc-pVTZ',
                                           'B3LYP/6-311++G(3df,3pd)'...
    `job_type`         ``str``           The job's type
    `scan`             ``list``          A list representing atom labels for the dihedral scan
                                           (e.g., "2 1 3 5" as a string or [2, 1, 3, 5] as a list of integers)
    `pivots`           ``list``          The rotor scan pivots, if the job type is scan. Not used directly in these
                                           methods, but used to identify the rotor.
    `software`         ``str``           The electronic structure software to be used
    `server_nodes`     ``list``          A list of nodes this job was submitted to (for troubleshooting)
    `memory`           ``int``           The allocated memory (1000 mb by default)
    `method`           ``str``           The calculation method (e.g., 'B3LYP', 'CCSD(T)', 'CBS-QB3'...)
    `basis_set`        ``str``           The basis set (e.g., '6-311++G(d,p)', 'aug-cc-pVTZ'...)
    `fine`             ``bool``          Whether to use fine geometry optimization parameters
    `shift`            ``str``           A string representation alpha- and beta-spin orbitals shifts (molpro only)
    `comments`         ``str``           Job comments (archived, not used)
    `date_time`        ``datetime``      The date-time this job was initiated. Determined automatically
    `run_time`         ``str``           Runtime. Determined automatically
    `job_status`       ``list``          The job's server and ESS statuses. Determined automatically
    `job_server_name`  ``str``           Job's name on the server (e.g., 'a103'). Determined automatically
    `job_name`         ``str``           Job's name for interal usage (e.g., 'opt_a103'). Determined automatically
    `job_id`           ``int``           The job's ID determined by the server.
    `local_path`       ``str``           Local path to job's folder. Determined automatically
    'local_path_to_output_file' ``str``  The local path to the output.out file
    `remote_path`      ``str``           Remote path to job's folder. Determined automatically
    `submit`           ``str``           The submit script. Created automatically
    `input`            ``str``           The input file. Created automatically
    `server`           ``str``           Server's name. Determined automatically
    'trsh'             ''str''           A troubleshooting handle to be appended to input files
    'ess_trsh_methods' ``list``          A list of troubleshooting methods already tried out for ESS convergence
    `occ`              ``int``           The number of occupied orbitals (core + val) from a molpro CCSD sp calc
    ================ =================== ===============================================================================

    self.job_status:
    The job server status is in job.job_status[0] and can be either 'initializing' / 'running' / 'errored' / 'done'
    The job ess (electronic structure software calculation) status is in  job.job_status[0] and can be
    either `initializing` / `running` / `errored: {error type / message}` / `unconverged` / `done`
    """
    def __init__(self, project, settings, species_name, xyz, job_type, level_of_theory, multiplicity, charge=0,
                 conformer=-1, fine=False, shift='', software=None, is_ts=False, scan='', pivots=None, memory=1500,
                 comments='', trsh='', ess_trsh_methods=None, occ=None):
        self.project = project
        self.settings=settings
        self.date_time = datetime.datetime.now()
        self.species_name = species_name
        self.job_num = -1
        self.charge = charge
        self.multiplicity = multiplicity
        self.spin = self.multiplicity - 1
        self.xyz = xyz
        self.n_atoms = self.xyz.count('\n')
        self.conformer = conformer
        self.is_ts = is_ts
        self.ess_trsh_methods = ess_trsh_methods if ess_trsh_methods is not None else list()
        self.trsh = trsh
        job_types = ['conformer', 'opt', 'freq', 'optfreq', 'sp', 'composite', 'scan', 'gsm', 'irc']
        # the 'conformer' job type is identical to 'opt', but we differentiate them to be identifiable in Scheduler
        if job_type not in job_types:
            raise ValueError("Job type {0} not understood. Must be on of the following: {1}".format(
                job_type, job_types))
        self.job_type = job_type
        self._set_job_number()
        self.job_server_name = 'a' + str(self.job_num)
        self.job_name = self.job_type + '_' + self.job_server_name

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
            if job_type == 'composite':
                if not self.settings['gaussian']:
                    raise JobError('Could not find the Gaussian software to run the composite method {0}'.format(
                        self.method))
                self.software = 'gaussian'
            elif job_type in ['conformer', 'opt', 'freq', 'optfreq', 'sp']:
                if 'ccs' in self.method or 'cis' in self.method or 'pv' in self.basis_set:
                    if self.settings['molpro']:
                        self.software = 'molpro'
                    elif self.settings['gaussian']:
                        self.software = 'gaussian'
                    elif self.settings['qchem']:
                        self.software = 'qchem'
                elif 'b3lyp' in self.method:
                    if self.settings['gaussian']:
                        self.software = 'gaussian'
                    elif self.settings['qchem']:
                        self.software = 'qchem'
                    elif self.settings['molpro']:
                        self.software = 'molpro'
                elif 'b97' in self.method or 'm06-2x' in self.method or 'def2' in self.basis_set:
                    if not self.settings['qchem']:
                        raise JobError('Could not find the QChem software to run {0}/{1}'.format(
                            self.method, self.basis_set))
                    self.software = 'qchem'
            elif job_type == 'scan':
                if 'b97' in self.method or 'm06-2x' in self.method or 'def2' in self.basis_set:
                    if not self.settings['qchem']:
                        raise JobError('Could not find the QChem software to run {0}/{1}'.format(
                            self.method, self.basis_set))
                    self.software = 'qchem'
                else:
                    if self.settings['gaussian']:
                        self.software = 'gaussian'
                    else:
                        self.software = 'qchem'
            elif job_type in ['gsm', 'irc']:
                if not self.settings['gaussian']:
                    raise JobError('Could not find the Gaussian software to run {0}'.format(job_type))
        if self.software is None:
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
            if self.settings['gaussian']:
                logging.error('Setting it to gaussian')
                self.software = 'gaussian'
            elif self.settings['qchem']:
                logging.error('Setting it to qchem')
                self.software = 'qchem'
            elif self.settings['molpro']:
                logging.error('Setting it to molpro')
                self.software = 'molpro'

        if self.settings['ssh']:
            self.server = self.settings[self.software]
        else:
            self.server = None

        if self.software == 'molpro':
            # molpro's memory is in MW, 500 should be enough
            memory /= 2
        self.memory = memory

        self.fine = fine
        self.shift = shift
        self.occ = occ
        self.run_time = ''
        self.job_status = ['initializing', 'initializing']
        self.job_id = 0
        self.comments = comments

        self.scan = scan
        self.pivots = list() if pivots is None else pivots

        conformer_folder = '' if self.conformer < 0 else os.path.join('conformers',str(self.conformer))
        self.local_path = os.path.join(arc_path, 'Projects', self.project,
                                       self.species_name, conformer_folder, self.job_name)
        self.local_path_to_output_file = os.path.join(self.local_path, 'output.out')
        # parentheses don't play well in folder names:
        species_name_for_remote_path = self.species_name.replace('(', '_').replace(')','_')
        self.remote_path = os.path.join('runs', 'ARC_Projects', self.project,
                                        species_name_for_remote_path, conformer_folder, self.job_name)
        self.submit = ''
        self.input = ''
        self.server_nodes = list()
        self._write_initiated_job_to_csv_file()

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
                       'job_name', 'job_id', 'server', 'software', 'memory', 'method', 'basis_set', 'date_time',
                       'comments']
                writer.writerow(row)
        with open(csv_path, 'rb') as f:
            reader = csv.reader(f, dialect='excel')
            job_num = 0
            for row in reader:
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
        else: conformer = str(self.conformer)
        with open(csv_path, 'ab') as f:
            writer = csv.writer(f, dialect='excel')
            row = [self.job_num, self.project, self.species_name, conformer, self.is_ts, self.charge,
                   self.multiplicity, self.job_type, self.job_name, self.job_id, self.server, self.software,
                   self.memory, self.method, self.basis_set, self.date_time, self.comments]
            writer.writerow(row)

    def write_completed_job_to_csv_file(self):
        """
        Write a completed ARCJob into the completed_jobs.csv file.
        """
        self.determine_job_status()
        csv_path = os.path.join(arc_path, 'completed_jobs.csv')
        if not os.path.isfile(csv_path):
            # check file, make index file and write headers if file doesn't exists
            with open(csv_path, 'wb') as f:
                writer = csv.writer(f, dialect='excel')
                row = ['job_num', 'project', 'species_name', 'conformer', 'is_ts', 'charge', 'multiplicity', 'job_type',
                       'job_name', 'job_id', 'server', 'software', 'memory', 'method', 'basis_set', 'date_time',
                       'run_time', 'job_status_(server)', 'job_status_(ESS)', 'ESS troubleshooting methods used',
                       'comments']
                writer.writerow(row)
        csv_path = os.path.join(arc_path, 'completed_jobs.csv')
        if self.conformer < 0:  # this is not a conformer search job
            conformer = '-'
        else: conformer = str(self.conformer)
        with open(csv_path, 'ab') as f:
            writer = csv.writer(f, dialect='excel')
            job_type = self.job_type
            if self.fine:
                job_type += ' (fine)'
            row = [self.job_num, self.project, self.species_name, conformer, self.is_ts, self.charge,
                   self.multiplicity, job_type, self.job_name, self.job_id, self.server, self.software,
                   self.memory, self.method, self.basis_set, self.date_time, self.run_time, self.job_status[0],
                   self.job_status[1], self.ess_trsh_methods, self.comments]
            writer.writerow(row)

    def write_submit_script(self):
        un = servers[self.server]['un']  # user name
        self.submit = submit_scripts[self.software].format(name=self.job_server_name, un=un)
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        with open(os.path.join(self.local_path, submit_filename[servers[self.server]['cluster_soft']]), 'wb') as f:
            f.write(self.submit)
        if self.settings['ssh']:
            self._upload_submit_file()

    def write_input_file(self):
        """
        Write a software-specific job-specific input file.
        Saves the file locally and also uploads it to the server.
        """

        self.input = input_files[self.software]

        slash = ''
        if self.software == 'gaussian' and not self.job_type == 'composite':
            slash = '/'

        if self.multiplicity > 1 and '/' in self.level_of_theory:  # only applies for non-composite jobs
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

        if self.job_type in ['conformer', 'opt', 'optfreq']:
            if self.software == 'gaussian':
                if self.is_ts:
                    job_type_1 = 'opt=(calcfc,ts,noeigen)'
                else:
                    job_type_1 = 'opt=calcfc'
                if self.fine:
                    fine = 'scf=(tight,direct) int=finegrid'
            elif self.software == 'qchem':
                if self.is_ts:
                    job_type_1 = 'ts'
                else:
                    job_type_1 = 'opt'
                if self.fine:
                    fine = '\n   GEOM_OPT_TOL_GRADIENT 15\n   GEOM_OPT_TOL_DISPLACEMENT 60\n   GEOM_OPT_TOL_ENERGY 5'
            elif self.software == 'molpro':
                if self.is_ts:
                    job_type_1 = "\noptg,root=2,method=qsd,readhess,savexyz='geometry.xyz'"
                else:
                    job_type_1 = "\noptg,savexyz='geometry.xyz'"

        elif self.job_type == 'freq':
            if self.software == 'gaussian':
                job_type_2 = 'freq iop(7/33=1)'
            elif self.software == 'qchem':
                job_type_1 = 'freq'
            elif self.software == 'molpro':
                job_type_1 = '\n{frequencies;\nthermo;\nprint,HESSIAN,thermo;}'

        elif self.job_type == 'optfreq':
            if self.software == 'gaussian':
                if self.is_ts:
                    job_type_1 = 'opt=(calcfc,ts,noeigen)'
                else:
                    job_type_1 = 'opt=calcfc'
                job_type_2 = 'freq iop(7/33=1)'
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
                pass
            elif self.software == 'qchem':
                job_type_1 = 'sp'
            elif self.software == 'molpro':
                pass

        if self.job_type == 'composite':
            if self.software == 'gaussian':
                if self.fine:
                    fine = 'scf=(tight,direct) int=finegrid'
                if self.is_ts:
                    job_type_1 = 'opt=(ts,noeigentest,calcfc)'
                else:
                    pass  # no need to specify anything else for a basic composite method run
            else:
                raise JobError('Currently composite methods are only supported in gaussian')

        if self.job_type == 'scan':
            if self.software == 'gaussian':
                job_type_1 = 'opt=modredundant'
                scan_string = ''.join([str(num) + ' ' for num in self.scan])
                if not divmod(360, rotor_scan_resolution):
                    raise JobError('Scan job got an illegal rotor scan resolution of {0}'.format(rotor_scan_resolution))
                scan_string = 'D ' + scan_string + 'S ' + str(int(360 / rotor_scan_resolution)) + ' ' +\
                              '{0:10}'.format(float(rotor_scan_resolution))
            else:
                raise ValueError('Currently rotor scan is only supported in gaussian')
        else:
            scan_string = ''

        if self.job_type == 'irc':  # TODO
            pass

        if self.job_type == 'gsm':  # TODO
            pass

        if 'mrci' in self.method:
            if self.software != 'molpro':
                raise JobError('Can only run MRCI on Molpro, not {0}'.format(self.software))
            if self.occ > 16:
                raise JobError('Will not excecute an MRCI calculation with more than 16 occupied orbitals.'
                               'Selective occ, closed, core, frozen keyword still not implemented.')
            else:
                try:
                    self.input = input_files['mrci'].format(memory=self.memory, xyz=self.xyz, basis=self.basis_set,
                                                            spin=self.spin, charge=self.charge, trsh=self.trsh)
                except KeyError as e:
                    logging.error('Could not interpret all input file keys in\n{0}'.format(self.input))
                    raise e
        else:
            try:
                self.input = self.input.format(memory=self.memory, method=self.method, slash=slash,
                                               basis=self.basis_set, charge=self.charge, multiplicity=self.multiplicity,
                                               spin=self.spin, xyz=self.xyz, job_type_1=job_type_1,
                                               job_type_2=job_type_2, scan=scan_string, restricted=restricted, fine=fine,
                                               shift=self.shift, trsh=self.trsh)
            except KeyError as e:
                logging.error('Could not interpret all input file keys in\n{0}'.format(self.input))
                raise e
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        with open(os.path.join(self.local_path, input_filename[self.software]), 'wb') as f:
            f.write(self.input)
        if self.settings['ssh']:
            self._upload_input_file()

    def _upload_submit_file(self):
        ssh = SSH_Client(self.server)
        ssh.send_command_to_server(command='mkdir -p {0}'.format(self.remote_path))
        remote_file_path = os.path.join(self.remote_path, submit_filename[servers[self.server]['cluster_soft']])
        ssh.upload_file(remote_file_path=remote_file_path, file_string=self.submit)

    def _upload_input_file(self):
        ssh = SSH_Client(self.server)
        ssh.send_command_to_server(command='mkdir -p {0}'.format(self.remote_path))
        remote_file_path = os.path.join(self.remote_path, input_filename[self.software])
        ssh.upload_file(remote_file_path=remote_file_path, file_string=self.input)

    def _download_output_file(self):
        ssh = SSH_Client(self.server)
        remote_file_path = os.path.join(self.remote_path, output_filename[self.software])
        local_file_path = os.path.join(self.local_path, 'output.out')
        ssh.download_file(remote_file_path=remote_file_path, local_file_path=local_file_path)

    def run(self):
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
        if self.settings['ssh']:
            ssh = SSH_Client(self.server)
            logging.debug('submitting job...')
            # submit_job returns job server status and job server id
            self.job_status[0], self.job_id = ssh.submit_job(remote_path=self.remote_path)

    def delete(self):
        logging.debug('Deleting job {name} for {label}'.format(name=self.job_name, label=self.species_name))
        if self.settings['ssh']:
            ssh = SSH_Client(self.server)
            logging.debug('deleting job...')
            ssh.delete_job(self.job_id)

    def determine_job_status(self):
        if self.job_status[0] == 'errored':
            return
        server_status = self._check_job_server_status()
        ess_status = ''
        if server_status == 'done':
            try:
                ess_status = self._check_job_ess_status()
            except IOError:
                logging.error('Got an IOError when trying to download output file for job {0}.'.format(self.job_name))
                raise
        elif server_status == 'running':
            ess_status = 'running'
        self.job_status = [server_status, ess_status]

    def _check_job_server_status(self):
        """
        Possible statuses: `initializing`, `running`, `errored on node xx`, `done`
        """
        if self.settings['ssh']:
            ssh = SSH_Client(self.server)
            return ssh.check_job_status(self.job_id)

    def _check_job_ess_status(self):
        """
        Check the status of the job ran by the electronic structure software (ESS)
        Possible statuses: `initializing`, `running`, `errored: {error type / message}`, `unconverged`, `done`
        """
        output_path = os.path.join(self.local_path, 'output.out')
        if os.path.exists(output_path):
            os.remove(output_path)
        if self.settings['ssh']:
            self._download_output_file()
        with open(output_path, 'rb') as f:
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
                            if 'l502.exe' in line:
                                return 'unconverged SCF'
                            if 'Erroneous write' in line or 'Write error in NtrExt1' in line:
                                reason = 'Ran out of disk space.'
                            if 'l716.exe' in line:
                                reason = 'Angle in z-matrix outside the allowed range 0 < x < 180.'
                            if 'l301.exe' in line:
                                reason = 'Input Error. Either charge, multiplicity, or basis set was not specified ' \
                                         'correctly. Or, an atom specified does not match any standard atomic symbol.'
                            if 'NtrErr Called from FileIO' in line:
                                reason = 'Operation on .chk file was specified, but .chk was not found.'
                            if 'l101.exe' in line:
                                reason = 'Input Error. The blank line after the coordinate section is missing, ' \
                                         'or charge/multiplicity was not specified correctly.'
                            if 'l202.exe' in line:
                                reason = 'During the optimization process, either the standard orientation ' \
                                         'or the point group of the molecule has changed.'
                            if 'l401.exe' in line:
                                reason = 'The projection from the old to the new basis set has failed.'
                            if 'malloc failed' in line or 'galloc' in line:
                                reason = 'Memory allocation failed (did you ask for too much?)'
                            if 'A SYNTAX ERROR WAS DETECTED' in line:
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
                    elif 'opt' in self.job_type or 'conformer' in self.job_type or 'ts' in self.job_type:
                            if 'MAXIMUM OPTIMIZATION CYCLES REACHED' in line:
                                return 'errored: unconverged, max opt cycles reached'
                            elif 'OPTIMIZATION CONVERGED' in line and done:  # `done` should already be assigned
                                return 'done'
                    elif 'SCF failed' in line:
                        return 'errored: {0}'.format(line)
                    elif 'error' in line and 'DIIS' not in line:
                        # these are *normal* lines: "SCF converges when DIIS error is below 1.0E-08", or
                        # "Cycle       Energy         DIIS Error"
                        error_message = line
                    elif 'Invalid charge/multiplicity combination' in line:
                        raise SpeciesError('The multiplicity and charge combination for species {0} are wrong.'.format(
                            self.species_name))
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
                    if 'No convergence' in line:
                        return 'unconverged'
                for line in lines[::-1]:
                    if 'the problem occurs' in line:
                        return 'errored: ' + line
                return 'errored: Unknown reason'

    def troubleshoot_server(self):
        # TODO: troubleshoot node on RMG
        # delete present server run
        logging.error('Job {name} has server status {stat} on {server}. Troubleshooting by changing node.'.format(
            name=self.job_name, stat=self.job_status[0], server=self.server))
        if self.settings['ssh']:
            ssh = SSH_Client(self.server)
            ssh.send_command_to_server(command=delete_command[servers[self.server]['cluster_soft']] +
                                       ' ' + str(self.job_id))
            # find available nodes
            stdout, _ = ssh.send_command_to_server(
                command=list_available_nodes_command[servers[self.server]['cluster_soft']])
            for line in stdout:
                node = line.split()[0].split('.')[0].split('node')[1]
                if servers[self.server]['cluster_soft'] == 'OGE' and '0/0/8' in line and node not in self.server_nodes:
                    self.server_nodes.append(node)
                    break
            else:
                logging.error('Cold not find an available node on the server')  # TODO: continue troubleshooting; if all else fails, put job to sleep for x min and try again searching for a node
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
            content = ''.join(content)  # convert list into a single string, not to upset paramico
            # resubmit
            ssh.upload_file(remote_file_path=os.path.join(self.remote_path,
                            submit_filename[servers[self.server]['cluster_soft']]), file_string=content)
            self.run()

# TODO: irc, gsm input files
