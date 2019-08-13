#!/usr/bin/env python
# encoding: utf-8
"""
The ARC Job module
"""


from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import csv
import shutil
import datetime

from arc.common import get_logger
from arc.settings import arc_path, servers, submit_filename, delete_command, t_max_format,\
    input_filename, output_filename, rotor_scan_resolution, list_available_nodes_command, levels_ess
from arc.job.submit import submit_scripts
from arc.job.inputs import input_files
from arc.job.ssh import SSHClient
from arc.job.local import get_last_modified_time, submit_job, delete_job, execute_command, check_job_status,\
    rename_output
from arc.arc_exceptions import JobError, SpeciesError

##################################################################

logger = get_logger()


class Job(object):
    """
    ARC's Job class.

    Args:
        project (str): The project's name. Used for naming the directory.
        project_directory (str): The path to the project directory.
        ess_settings (dict): A dictionary of available ESS and a corresponding server list.
        species_name (str): The species/TS name. Used for naming the directory.
        xyz (str): The xyz geometry. Used for the calculation.
        job_type (str): The job's type.
        level_of_theory (str): Level of theory, e.g. 'CBS-QB3', 'CCSD(T)-F12a/aug-cc-pVTZ', 'B3LYP/6-311++G(3df,3pd)'...
        multiplicity (int): The species multiplicity.
        charge (int, optional): The species net charge. Default is 0.
        conformer (int, optional): Conformer number if optimizing conformers.
        fine (bool, optional): Whether to use fine geometry optimization parameters.
        shift (str, optional): A string representation alpha- and beta-spin orbitals shifts (molpro only).
        software (str, optional): The electronic structure software to be used.
        is_ts (bool, optional): Whether this species represents a transition structure.
        scan (list, optional): A list representing atom labels for the dihedral scan
                              (e.g., "2 1 3 5" as a string or [2, 1, 3, 5] as a list of integers).
        pivots (list, optional): The rotor scan pivots, if the job type is scan. Not used directly in these methods,
                                 but used to identify the rotor.
        memory (int, optional): The total job allocated memory in GB.
        comments (str, optional): Job comments (archived, not used).
        trsh (str, optional): A troubleshooting keyword to be used in input files.
        scan_trsh (str, optional): A troubleshooting method for rotor scans.
        ess_trsh_methods (list, optional): A list of troubleshooting methods already tried out for ESS convergence.
        bath_gas (str, optional): A bath gas. Currently used in OneDMin to calc L-J parameters.
                                  Allowed values are He, Ne, Ar, Kr, H2, N2, O2
        initial_trsh (dict, optional): Troubleshooting methods to try by default. Keys are ESS software,
                                       values are trshs.
        job_num (int, optional): Used as the entry number in the database, as well as the job name on the server.
        job_server_name (str, optional): Job's name on the server (e.g., 'a103').
        job_name (str, optional): Job's name for internal usage (e.g., 'opt_a103').
        job_id (int, optional): The job's ID determined by the server.
        server (str, optional): Server's name.
        initial_time (datetime, optional): The date-time this job was initiated.
        occ (int, optional): The number of occupied orbitals (core + val) from a molpro CCSD sp calc.
        max_job_time (int, optional): The maximal allowed job time on the server in hours.
        scan_res (int, optional): The rotor scan resolution in degrees.
        checkfile (str, optional): The path to a previous Gaussian checkfile to be used in the current job.
        number_of_radicals (int, optional): The number of radicals (inputted by the user, ARC won't attempt to
                                            determine it). Defaults to None. Important, e.g., if a Species is a bi-rad
                                            singlet, in which case the job should be unrestricted with
                                            multiplicity = 1.
        conformers (str, optional): A path to the YAML file conformer coordinates for a Gromacs MD job.
        radius (float, optional): The species radius in Angstrom.
        testing (bool, optional): Whether the object is generated for testing purposes, True if it is.

    Attributes:
        project (str): The project's name. Used for naming the directory.
        ess_settings (dict): A dictionary of available ESS and a corresponding server list.
        species_name (str): The species/TS name. Used for naming the directory.
        charge (int): The species net charge. Default is 0.
        multiplicity (int): The species multiplicity.
        number_of_radicals (int): The number of radicals (inputted by the user, ARC won't attempt to determine it).
                                  Defaults to None. Important, e.g., if a Species is a bi-rad singlet, in which case
                                  the job should be unrestricted with multiplicity = 1.
        spin (int): The spin. automatically derived from the multiplicity.
        xyz (str): The xyz geometry. Used for the calculation.
        radius (float): The species radius in Angstrom.
        n_atoms (int): The number of atoms in self.xyz.
        conformer (int): Conformer number if optimizing conformers.
        conformers (str): A path to the YAML file conformer coordinates for a Gromacs MD job.
        is_ts (bool): Whether this species represents a transition structure.
        level_of_theory (str): Level of theory, e.g. 'CBS-QB3', 'CCSD(T)-F12a/aug-cc-pVTZ', 'B3LYP/6-311++G(3df,3pd)'...
        job_type (str): The job's type.
        scan (list): A list representing atom labels for the dihedral scan
                    (e.g., "2 1 3 5" as a string or [2, 1, 3, 5] as a list of integers).
        pivots (list): The rotor scan pivots, if the job type is scan. Not used directly in these methods,
                       but used to identify the rotor.
        scan_res (int): The rotor scan resolution in degrees.
        software (str): The electronic structure software to be used.
        server_nodes (list): A list of nodes this job was submitted to (for troubleshooting).
        memory_gb (int): The total job allocated memory in GB (14 by default).
        memory (int): The total job allocated memory in appropriate units per ESS.
        method (str): The calculation method (e.g., 'B3LYP', 'CCSD(T)', 'CBS-QB3'...).
        basis_set (str): The basis set (e.g., '6-311++G(d,p)', 'aug-cc-pVTZ'...).
        fine (bool): Whether to use fine geometry optimization parameters.
        shift (str): A string representation alpha- and beta-spin orbitals shifts (molpro only).
        comments (str): Job comments (archived, not used).
        initial_time (datetime): The date-time this job was initiated.
        final_time (datetime): The date-time this job was initiated.
        run_time (timedelta): Job execution time.
        job_status (list): The job's server and ESS statuses.
                           The job server status is in job.job_status[0] and can be either 'initializing' / 'running'
                           / 'errored' / 'done'. The job ess (electronic structure software calculation) status is in
                           job.job_status[1] and can be either `initializing` / `running` / `errored:
                           {error type / message}` / `unconverged` / `done`.
        job_server_name (str): Job's name on the server (e.g., 'a103').
        job_name (str): Job's name for internal usage (e.g., 'opt_a103').
        job_id (int): The job's ID determined by the server.
        job_num (int): Used as the entry number in the database, as well as the job name on the server.
        local_path (str): Local path to job's folder.
        local_path_to_output_file (str): The local path to the output.out file.
        local_path_to_orbitals_file (str): The local path to the orbitals.fchk file (only for orbitals jobs).
        local_path_to_check_file (str): The local path to the Gaussian check file of the current job (downloaded).
        local_path_to_lj_file (str): The local path to the lennard_jones data file (from OneDMin).
        checkfile (str): The path to a previous Gaussian checkfile to be used in the current job.
        remote_path (str): Remote path to job's folder.
        submit (str): The submit script. Created automatically.
        input (str): The input file. Created automatically.
        server (str): Server's name.
        trsh (str): A troubleshooting keyword to be used in input files.
        ess_trsh_methods (list): A list of troubleshooting methods already tried out for ESS convergence.
        initial_trsh (dict): Troubleshooting methods to try by default. Keys are ESS software, values are trshs.
        scan_trsh (str): A troubleshooting method for rotor scans.
        occ (int): The number of occupied orbitals (core + val) from a molpro CCSD sp calc.
        project_directory (str): The path to the project directory.
        max_job_time (int): The maximal allowed job time on the server in hours.
        bath_gas (str): A bath gas. Currently used in OneDMin to calc L-J parameters.
                        Allowed values are He, Ne, Ar, Kr, H2, N2, O2

    """
    def __init__(self, project, ess_settings, species_name, xyz, job_type, level_of_theory, multiplicity,
                 project_directory, charge=0, conformer=-1, fine=False, shift='', software=None, is_ts=False, scan=None,
                 pivots=None, memory=14, comments='', trsh='', scan_trsh='', ess_trsh_methods=None, bath_gas=None,
                 initial_trsh=None, job_num=None, job_server_name=None, job_name=None, job_id=None, server=None,
                 initial_time=None, occ=None, max_job_time=120, scan_res=None, checkfile=None, number_of_radicals=None,
                 conformers=None, radius=None, testing=False):
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
        self.n_atoms = self.xyz.count('\n') if xyz is not None else None
        self.radius = radius
        self.conformer = conformer
        self.conformers = conformers
        self.is_ts = is_ts
        self.ess_trsh_methods = ess_trsh_methods if ess_trsh_methods is not None else list()
        self.trsh = trsh
        self.initial_trsh = initial_trsh if initial_trsh is not None else dict()
        self.scan_trsh = scan_trsh
        self.scan_res = scan_res if scan_res is not None else rotor_scan_resolution
        self.scan = scan
        self.pivots = list() if pivots is None else pivots
        self.max_job_time = max_job_time
        self.bath_gas = bath_gas
        self.testing = testing
        self.fine = fine
        self.shift = shift
        self.occ = occ
        self.job_status = ['initializing', 'initializing']
        self.job_id = job_id if job_id is not None else 0
        self.comments = comments
        self.project_directory = project_directory
        self.checkfile = checkfile
        self.submit = ''
        self.input = ''
        self.server_nodes = list()
        job_types = ['conformer', 'opt', 'freq', 'optfreq', 'sp', 'composite', 'scan', 'gsm', 'irc', 'ts_guess',
                     'orbitals', 'onedmin', 'ff_param_fit', 'gromacs']  # allowed job types
        # the 'conformer' job type is identical to 'opt', but we differentiate them to be identifiable in Scheduler
        if job_type not in job_types:
            raise ValueError("Job type {0} not understood. Must be one of the following:\n{1}".format(
                job_type, job_types))
        self.job_type = job_type

        if self.xyz is None and not self.job_type == 'gromacs':
            raise ValueError('{0} Job of species {1} got None for xyz'.format(self.job_type, self.species_name))
        if self.conformers is None and self.job_type == 'gromacs':
            raise ValueError('{0} Job of species {1} got None for conformers'.format(self.job_type, self.species_name))

        if self.job_num < 0:
            self._set_job_number()
        self.job_server_name = job_server_name if job_server_name is not None else 'a' + str(self.job_num)
        self.job_name = job_name if job_name is not None else self.job_type + '_' + self.job_server_name

        # determine level of theory and software to use:
        self.level_of_theory = level_of_theory.lower()
        self.software = software
        self.method, self.basis_set = '', ''
        if '/' in self.level_of_theory:
            splits = self.level_of_theory.split('/')
            self.method = splits[0]
            self.basis_set = '/'.join(splits[1:])  # there are two '/' symbols in a ff_param_fit job's l.o.t, keep both
        else:  # this is a composite job
            self.method, self.basis_set = self.level_of_theory, ''

        if self.software is not None:
            self.software = self.software.lower()
        else:
            self.deduce_software()
        self.server = server if server is not None else self.ess_settings[self.software][0]
        self.mem_per_cpu, self.cpus, self.memory_gb, self.memory = None, None, None, None
        self.set_cpu_and_mem(memory=memory)

        self.set_file_paths()

        if job_num is None:
            # this checks job_num and not self.job_num on purpose
            # if job_num was given, then don't save as initiated jobs, this is a restarted job
            self._write_initiated_job_to_csv_file()

    def as_dict(self):
        """
        A helper function for dumping this object as a dictionary in a YAML file for restarting ARC.
        """
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
        job_dict['memory'] = self.memory_gb
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
        Used as the entry number in the database, as well as the job name on the server.
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
        """
        Write the Job's submit script.
        """
        un = servers[self.server]['un']  # user name
        size = int(self.radius * 4) if self.radius is not None else None
        if self.max_job_time > 9999 or self.max_job_time <= 0:
            logger.debug('Setting max_job_time to 120 hours')
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
        architecture = ''
        if self.server.lower() == 'pharos':
            # here we're hard-coding ARC for Pharos, a Green Group server
            # If your server has different node architectures, implement something similar
            if self.cpus <= 8:
                architecture = '\n#$ -l harpertown'
            else:
                architecture = '\n#$ -l magnycours'
        try:
            self.submit = submit_scripts[self.server][self.software.lower()].format(
                name=self.job_server_name, un=un, t_max=t_max, mem_per_cpu=int(self.mem_per_cpu), cpus=self.cpus,
                architecture=architecture, size=size)
        except KeyError:
            submit_scripts_for_printing = dict()
            for server, values in submit_scripts.items():
                submit_scripts_for_printing[server] = list()
                for software in values.keys():
                    submit_scripts_for_printing[server].append(software)
            logger.error('Could not find submit script for server {0} and software {1}. Make sure your submit scripts '
                         '(in arc/job/submit.py) are updated with the servers and software defined in arc/settings.py\n'
                         'Alternatively, It is possible that you defined parameters in curly braces (e.g., {{PARAM}}) '
                         'in your submit script/s. To avoid error, replace them with double curly braces (e.g., '
                         '{{{{PARAM}}}} instead of {{PARAM}}.\nIdentified the following submit scripts:\n{2}'.format(
                          self.server, self.software.lower(), submit_scripts_for_printing))
            raise
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        with open(os.path.join(self.local_path, submit_filename[servers[self.server]['cluster_soft']]), 'wb') as f:
            f.write(self.submit)
        if self.server != 'local' and not self.testing:
            self._upload_submit_file()

    def write_input_file(self):
        """
        Write a software-specific, job-specific input file.
        Save the file locally and also upload it to the server.
        """
        if self.initial_trsh and not self.trsh:
            # use the default trshs defined by the user in the initial_trsh dictionary
            if self.software in self.initial_trsh:
                self.trsh = self.initial_trsh[self.software]

        self.input = input_files.get(self.software, None)

        slash = ''
        if self.software == 'gaussian' and '/' in self.level_of_theory:
            slash = '/'

        if (self.multiplicity > 1 and '/' in self.level_of_theory) or self.number_of_radicals > 1:
            # don't add 'u' to composite jobs. Do add 'u' for bi-rad singlets if `number_of_radicals` > 1
            if self.number_of_radicals > 1:
                logger.info('Using an unrestricted method for species {0} which has {1} radicals and '
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
                    job_type_1 = 'opt'
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
                        job_type_1 = 'opt=(tight)'
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

        elif self.job_type == 'orbitals' and self.software == 'qchem':
            if self.is_ts:
                job_type_1 = 'ts'
            else:
                job_type_1 = 'opt'
            if 'PRINT_ORBITALS' not in self.trsh:
                self.trsh += '\n   NBO           TRUE\n   RUN_NBO6      TRUE\n   ' \
                             'PRINT_ORBITALS  TRUE\n   GUI           2'

        elif self.job_type == 'ff_param_fit' and self.software == 'gaussian':
            job_type_1, job_type_2 = 'opt', 'freq'
            self.input += """

--Link1--
%chk=check.chk
%mem={memory}mb
%NProcShared={cpus}

# HF/6-31G(d) SCF=Tight Pop=MK IOp(6/33=2,6/41=10,6/42=17) scf(maxcyc=500) guess=read geom=check Maxdisk=2GB

name

{charge} {multiplicity}


"""

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
                    logger.error('Could not interpret all input file keys in\n{0}'.format(self.input))
                    raise
        else:
            try:
                self.input = self.input.format(memory=int(self.memory), method=self.method, slash=slash,
                                               basis=self.basis_set, charge=self.charge, cpus=self.cpus,
                                               multiplicity=self.multiplicity, spin=self.spin, xyz=self.xyz,
                                               job_type_1=job_type_1, job_type_2=job_type_2, scan=scan_string,
                                               restricted=restricted, fine=fine, shift=self.shift, trsh=self.trsh,
                                               scan_trsh=self.scan_trsh, bath=self.bath_gas) \
                    if self.input is not None else None
            except KeyError:
                logger.error('Could not interpret all input file keys in\n{0}'.format(self.input))
                raise
        if not self.testing:
            if not os.path.exists(self.local_path):
                os.makedirs(self.local_path)
            if self.input is not None:
                with open(os.path.join(self.local_path, input_filename[self.software]), 'w') as f:
                    f.write(self.input)
            if self.server != 'local':
                self._upload_input_file()
            else:
                self.initial_time = get_last_modified_time(
                    file_path=os.path.join(self.local_path, submit_filename[servers[self.server]['cluster_soft']]))
                # copy additional input files to local running directory
                for up_file in self.additional_files_to_upload:
                    if up_file['source'] == 'path':
                        source_path = up_file['local']
                        destination_path = os.path.join(self.local_path, up_file['name'])
                        shutil.copyfile(source_path, destination_path)
                    elif up_file['source'] == 'input_files':
                        with open(os.path.join(self.local_path, up_file['name']), 'w') as f:
                            f.write(str(input_files[up_file['local']]))

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
        if self.input is not None:
            remote_file_path = os.path.join(self.remote_path, input_filename[self.software])
            ssh.upload_file(remote_file_path=remote_file_path, file_string=self.input)
        for up_file in self.additional_files_to_upload:
            if up_file['source'] == 'path':
                local_file_path = up_file['local']
            elif up_file['source'] == 'input_files':
                local_file_path = input_files[up_file['local']]
            else:
                raise JobError('Unclear file source for {0}. Should either be "path" of "input_files", '
                               'got: {1}'.format(up_file['name'], up_file['source']))
            ssh.upload_file(remote_file_path=up_file['remote'], local_file_path=local_file_path)
            if up_file['make_x']:
                ssh.send_command_to_server(command='chmod +x {0}'.format(up_file['name']), remote_path=self.remote_path)
        self.initial_time = ssh.get_last_modified_time(
            remote_file_path=os.path.join(self.remote_path, submit_filename[servers[self.server]['cluster_soft']]))

    def _upload_check_file(self, local_check_file_path=None):
        if self.server != 'local':
            ssh = SSHClient(self.server)
            remote_check_file_path = os.path.join(self.remote_path, 'check.chk')
            local_check_file_path = os.path.join(self.local_path, 'check.chk') if remote_check_file_path is None\
                else local_check_file_path
            if os.path.isfile(local_check_file_path) and self.software.lower() == 'gaussian':
                ssh.upload_file(remote_file_path=remote_check_file_path, local_file_path=local_check_file_path)
                logger.debug('uploading checkpoint file for {0}'.format(self.job_name))
        else:
            # running locally, just copy the check file to the job folder
            new_check_file_path = os.path.join(self.local_path, 'check.chk')
            shutil.copyfile(local_check_file_path, new_check_file_path)

    def _download_output_file(self):
        """
        Download ESS output, orbitals check file, and the Gaussian check file, if relevant.
        """
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
                logger.warning('Orbitals FChk file for {0} was not downloaded properly '
                               '(this is not the Gaussian formatted check file...)'.format(self.job_name))

        # download Gaussian check file
        if self.software.lower() == 'gaussian':
            remote_check_file_path = os.path.join(self.remote_path, 'check.chk')
            ssh.download_file(remote_file_path=remote_check_file_path, local_file_path=self.local_path_to_check_file)
            if not os.path.isfile(self.local_path_to_check_file):
                logger.warning('Gaussian check file for {0} was not downloaded properly'.format(self.job_name))

        # download Lennard_Jones data file
        if self.software.lower() == 'onedmin':
            remote_lj_file_path = os.path.join(self.remote_path, 'lj.dat')
            ssh.download_file(remote_file_path=remote_lj_file_path, local_file_path=self.local_path_to_lj_file)
            if not os.path.isfile(self.local_path_to_lj_file):
                logger.warning('Lennard-Jones data file for {0} was not downloaded properly'.format(self.job_name))

        # download molpro log file (in addition to the output file)
        if self.software.lower() == 'molpro':
            remote_log_file_path = os.path.join(self.remote_path, 'input.log')
            local_log_file_path = os.path.join(self.local_path, 'output.log')
            ssh.download_file(remote_file_path=remote_log_file_path, local_file_path=local_log_file_path)
            if not os.path.isfile(local_log_file_path):
                logger.warning('Could not download Molpro log file for {0} '
                               '(this is not the output file)'.format(self.job_name))

    def run(self):
        """
        Execute the Job.
        """
        if self.fine:
            logger.info('Running job {name} for {label} (fine opt)'.format(name=self.job_name,
                                                                           label=self.species_name))
        elif self.pivots:
            logger.info('Running job {name} for {label} (pivots: {pivots})'.format(name=self.job_name,
                                                                                   label=self.species_name,
                                                                                   pivots=self.pivots))
        else:
            logger.info('Running job {name} for {label}'.format(name=self.job_name, label=self.species_name))
        logger.debug('writing submit script...')
        self.write_submit_script()
        logger.debug('writing input file...')
        self.write_input_file()
        if self.server != 'local':
            ssh = SSHClient(self.server)
            logger.debug('submitting job...')
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
        """
        Delete a running Job.
        """
        logger.debug('Deleting job {name} for {label}'.format(name=self.job_name, label=self.species_name))
        if self.server != 'local':
            ssh = SSHClient(self.server)
            logger.debug('deleting job on {0}...'.format(self.server))
            ssh.delete_job(self.job_id)
        else:
            logger.debug('deleting job locally...')
            delete_job(job_id=self.job_id)

    def determine_job_status(self):
        """
        Determine the Job's status. Updates self.job_status.
        """
        if self.job_status[0] == 'errored':
            return
        server_status = self._check_job_server_status()
        ess_status = ''
        if server_status == 'done':
            try:
                ess_status = self._check_job_ess_status()  # also downloads output file
            except IOError:
                logger.error('Got an IOError when trying to download output file for job {0}.'.format(self.job_name))
                content = self._get_additional_job_info()
                if content:
                    logger.info('Got the following information from the server:')
                    logger.info(content)
                    for line in content.splitlines():
                        # example:
                        # slurmstepd: *** JOB 7752164 CANCELLED AT 2019-03-27T00:30:50 DUE TO TIME LIMIT on node096 ***
                        if 'cancelled' in line.lower() and 'due to time limit' in line.lower():
                            logger.warning('Looks like the job was cancelled on {0} due to time limit. '
                                           'Got: {1}'.format(self.server, line))
                            new_max_job_time = self.max_job_time - 24 if self.max_job_time > 25 else 1
                            logger.warning('Setting max job time to {0} (was {1})'.format(new_max_job_time,
                                                                                          self.max_job_time))
                            self.max_job_time = new_max_job_time
                raise
        elif server_status == 'running':
            ess_status = 'running'
        self.job_status = [server_status, ess_status]

    def _get_additional_job_info(self):
        """
        Download the additional information of stdout and stderr from the server.
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
                    logger.warning('Got the following error when trying to download out.txt for {0}:'.format(
                        self.job_name))
                    logger.warning(e.message)
                remote_file_path = os.path.join(self.remote_path, 'err.txt')
                try:
                    ssh.download_file(remote_file_path=remote_file_path, local_file_path=local_file_path2)
                except (TypeError, IOError) as e:
                    logger.warning('Got the following error when trying to download err.txt for {0}:'.format(
                        self.job_name))
                    logger.warning(e.message)
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
                            logger.warning('Got the following error when trying to download {0} for {1}:'.format(
                                file_name, self.job_name))
                            logger.warning(e.message)
                    if os.path.isfile(local_file_path):
                        with open(local_file_path, 'r') as f:
                            lines1 = f.readlines()
                    content += ''.join([line for line in lines1])
                    content += '\n'
        return content

    def _check_job_server_status(self):
        """
        Possible statuses: `initializing`, `running`, `errored on node xx`, `done`.
        """
        if self.server != 'local':
            ssh = SSHClient(self.server)
            return ssh.check_job_status(self.job_id)
        else:
            return check_job_status(self.job_id)

    def _check_job_ess_status(self):
        """
        Check the status of the job ran by the electronic structure software (ESS).
        Possible statuses: `initializing`, `running`, `errored: {error type / message}`, `unconverged`, `done`.
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
                    for i, line in enumerate(lines[::-1]):
                        if 'Error' in line or 'NtrErr' in line or 'Erroneous' in line or 'malloc' in line \
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
                                reason = 'l301'
                            elif 'NtrErr Called from FileIO' in line:
                                reason = 'Operation on .chk file was specified, but .chk was not found.'
                            elif 'l101.exe' in line:
                                reason = 'Input Error. The blank line after the coordinate section is missing, ' \
                                         'or charge/multiplicity was not specified correctly.'
                            elif 'l202.exe' in line:
                                reason = 'During the optimization process, either the standard orientation ' \
                                         'or the point group of the molecule has changed.'
                            elif 'l401.exe' in line:
                                reason = 'l401'
                            elif 'malloc failed' in line or 'galloc' in line:
                                reason = 'Memory allocation failed (did you ask for too much?)'
                            elif 'A SYNTAX ERROR WAS DETECTED' in line:
                                reason = 'Check .inp carefully for syntax errors in keywords.'

                            if reason in ['l301', 'l401']:
                                additional_info = lines[len(lines) - i - 2]
                                if 'No data on chk file' in additional_info \
                                        or 'Basis set data is not on the checkpoint file' in additional_info:
                                    reason += ' check file problematic'
                                elif reason == 'l301':
                                    reason += 'Input Error. Either charge, multiplicity, or basis set was not ' \
                                             'specified correctly. Or, an atom specified does not match any standard ' \
                                              'atomic symbol.'
                                elif reason == 'l401':
                                    reason += ' "The projection from the old to the new basis set has failed."'

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
        """
        Troubleshoot server errors.
        """
        if servers[self.server]['cluster_soft'].lower() == 'oge':
            # delete present server run
            logger.error('Job {name} has server status "{stat}" on {server}. Troubleshooting by changing node.'.
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
                logger.error('Could not find an available node on the server')
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
                logger.error('Job {name} has server status "{stat}" on {server}. Re-running job.'.format(
                    name=self.job_name, stat=self.job_status[0], server=self.server))
            ssh = SSHClient(self.server)
            ssh.send_command_to_server(command=delete_command[servers[self.server]['cluster_soft']] +
                                       ' ' + str(self.job_id))
            # resubmit
            self.run()

    def determine_run_time(self):
        """
        Determine the run time. Update self.run_time and round to seconds.
        """
        if self.initial_time is not None and self.final_time is not None:
            time_delta = self.final_time - self.initial_time
            remainder = time_delta.microseconds > 5e5
            self.run_time = datetime.timedelta(seconds=time_delta.seconds + remainder)

    def deduce_software(self):
        """
        Deduce the software to be used based on heuristics.
        """
        if self.job_type == 'onedmin':
            if 'onedmin' not in self.ess_settings.keys():
                raise JobError('Could not find the OneDMin software to compute Lennard-Jones parameters.\n'
                               'ess_settings is:\n{0}'.format(self.ess_settings))
            self.software = 'onedmin'
            if self.bath_gas is None:
                logger.info('Setting bath gas for Lennard-Jones calculation to N2 for species {0}'.format(
                    self.species_name))
                self.bath_gas = 'N2'
            elif self.bath_gas not in ['He', 'Ne', 'Ar', 'Kr', 'H2', 'N2', 'O2']:
                raise JobError('Bath gas for OneDMin should be one of the following:\n'
                               'He, Ne, Ar, Kr, H2, N2, O2.\nGot: {0}'.format(self.bath_gas))
        elif self.job_type == 'gromacs':
            if 'gromacs' not in self.ess_settings.keys():
                raise JobError('Could not find the Gromacs software to run the MD job {0}.\n'
                               'ess_settings is:\n{1}'.format(self.method, self.ess_settings))
            self.software = 'gromacs'
        elif self.job_type == 'orbitals':
            # currently we only have a script to print orbitals on QChem,
            # could/should definitely be elaborated to additional ESS
            if 'qchem' not in self.ess_settings.keys():
                logger.debug('Could not find the QChem software to compute molecular orbitals.\n'
                             'ess_settings is:\n{0}'.format(self.ess_settings))
                self.software = None
            else:
                self.software = 'qchem'
        elif self.job_type == 'composite':
            if 'gaussian' not in self.ess_settings.keys():
                raise JobError('Could not find the Gaussian software to run the composite method {0}.\n'
                               'ess_settings is:\n{1}'.format(self.method, self.ess_settings))
            self.software = 'gaussian'
        elif self.job_type == 'ff_param_fit':
            if 'gaussian' not in self.ess_settings.keys():
                raise JobError('Could not find Gaussian to fit force field parameters.\n'
                               'ess_settings is:\n{0}'.format(self.ess_settings))
            self.software = 'gaussian'
        else:
            # First check the levels_ess dictionary from settings.py:
            for ess, phrase_list in levels_ess.items():
                for phrase in phrase_list:
                    if phrase in self.level_of_theory:
                        self.software = ess.lower()
            if self.software is None:
                if self.job_type in ['conformer', 'opt', 'freq', 'optfreq', 'sp']:
                    if 'b2' in self.method or 'dsd' in self.method or 'pw2' in self.method:
                        # this is a double-hybrid (MP2) DFT method, use Gaussian
                        if 'gaussian' not in self.ess_settings.keys():
                            raise JobError('Could not find the Gaussian software to run the double-hybrid method {0}.\n'
                                           'ess_settings is:\n{1}'.format(self.method, self.ess_settings))
                        self.software = 'gaussian'
                    elif 'ccs' in self.method or 'cis' in self.method:
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
                        elif 'molpro' in self.ess_settings.keys():
                            self.software = 'molpro'
                    elif 'wb97xd' in self.method:
                        if 'gaussian' not in self.ess_settings.keys():
                            raise JobError('Could not find the Gaussian software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'gaussian'
                    elif 'wb97x-d3' in self.method:
                        if 'qchem' not in self.ess_settings.keys():
                            raise JobError('Could not find the QChem software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'qchem'
                    elif 'b97' in self.method or 'def2' in self.basis_set:
                        if 'gaussian' in self.ess_settings.keys():
                            self.software = 'gaussian'
                        elif 'qchem' in self.ess_settings.keys():
                            self.software = 'qchem'
                    elif 'm062x' in self.method:  # without dash
                        if 'gaussian' not in self.ess_settings.keys():
                            raise JobError('Could not find the Gaussian software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'gaussian'
                    elif 'm06-2x' in self.method:  # with dash
                        if 'qchem' not in self.ess_settings.keys():
                            raise JobError('Could not find the QChem software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'qchem'
                elif self.job_type == 'scan':
                    if 'wb97xd' in self.method:
                        if 'gaussian' not in self.ess_settings.keys():
                            raise JobError('Could not find the Gaussian software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'gaussian'
                    elif 'wb97x-d3' in self.method:
                        if 'qchem' not in self.ess_settings.keys():
                            raise JobError('Could not find the QChem software to run {0}/{1}'.format(
                                self.method, self.basis_set))
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
                            raise JobError('Could not find the QChem software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'qchem'
                    elif 'm062x' in self.method:  # without dash
                        if 'gaussian' not in self.ess_settings.keys():
                            raise JobError('Could not find the Gaussian software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'gaussian'
                    elif 'pv' in self.basis_set:
                        if 'molpro' in self.ess_settings.keys():
                            self.software = 'molpro'
                        elif 'gaussian' in self.ess_settings.keys():
                            self.software = 'gaussian'
                        elif 'qchem' in self.ess_settings.keys():
                            self.software = 'qchem'
                elif self.job_type in ['gsm', 'irc']:
                    if 'gaussian' not in self.ess_settings.keys():
                        raise JobError('Could not find the Gaussian software to run {0}'.format(self.job_type))
                    self.software = 'gaussian'
            if self.software is None:
                # if still no software was determined, just try by order, if exists
                logger.error('job_num: {0}'.format(self.job_num))
                logger.error('ess_trsh_methods: {0}'.format(self.ess_trsh_methods))
                logger.error('trsh: {0}'.format(self.trsh))
                logger.error('job_type: {0}'.format(self.job_type))
                logger.error('job_name: {0}'.format(self.job_name))
                logger.error('level_of_theory: {0}'.format(self.level_of_theory))
                logger.error('software: {0}'.format(self.software))
                logger.error('method: {0}'.format(self.method))
                logger.error('basis_set: {0}'.format(self.basis_set))
                logger.error('Could not determine software for job {0}'.format(self.job_name))
                if 'gaussian' in self.ess_settings.keys():
                    logger.error('Setting it to Gaussian')
                    self.software = 'gaussian'
                elif 'orca' in self.ess_settings.keys():
                    logger.error('Setting it to Orca')
                    self.software = 'orca'
                elif 'qchem' in self.ess_settings.keys():
                    logger.error('Setting it to QChem')
                    self.software = 'qchem'
                elif 'molpro' in self.ess_settings.keys():
                    logger.error('Setting it to Molpro')
                    self.software = 'molpro'

    def set_cpu_and_mem(self, memory):
        """
        Set the number of cpu's and the job's memory.
        self.memory is the actual memory allocated to the ESS.
        self.mem_per_cpu is the cluster software allocated memory.
        (self.mem_per_cpu should be slightly larger than self.memory when considering all cpus).
        """
        self.cpus = servers[self.server].get('cpus', 8)  # set to 8 by default
        max_mem = servers[self.server].get('memory', None)  # max memory per node
        if max_mem is not None and memory > max_mem * 0.9:
            logger.warning('The memory for job {0} using {1} ({2} GB) exceeds 90% of the the maximum node memory on '
                           '{3}. Setting it to 90% * {4} GB.'.format(self.job_name, self.software,
                                                                     memory, self.server, max_mem))
            memory = 0.9 * max_mem
            self.mem_per_cpu = memory * 1024 * 1.05 / self.cpus
        else:
            self.mem_per_cpu = memory * 1024 * 1.1 / self.cpus  # The `#SBATCH --mem-per-cpu` directive is in MB

        self.memory_gb = memory  # store the memory in GB for troubleshooting (when re-running the job)
        if self.software == 'molpro':
            # Molpro's memory is per cpu and in MW (mega word; 1 MW ~= 8 MB; 1 GB = 128 MW)
            self.memory = memory * 128 / self.cpus
        if self.software == 'terachem':
            # TeraChem's memory is per cpu and in MW (mega word; 1 MW ~= 8 MB; 1 GB = 128 MW)
            self.memory = memory * 128 / self.cpus
        elif self.software == 'gaussian':
            # Gaussian's memory is in MB, total for all cpus
            self.memory = memory * 1000
        elif self.software == 'orca':
            # Orca's memory is per cpu and in MB
            self.memory = memory * 1000 / self.cpus
        elif self.software == 'qchem':
            # QChem manages its memory automatically, for now ARC will not intervene
            # see http://www.q-chem.com/qchem-website/manual/qchem44_manual/CCparallel.html
            self.memory = memory  # dummy
        elif self.software == 'gromacs':
            # not managing memory for Gromacs
            self.memory = memory  # dummy

    def set_file_paths(self):
        """
        Set local and remote job file paths.
        """
        conformer_folder = '' if self.conformer < 0 else os.path.join('conformers', str(self.conformer))
        folder_name = 'TSs' if self.is_ts else 'Species'
        self.local_path = os.path.join(self.project_directory, 'calcs', folder_name,
                                       self.species_name, conformer_folder, self.job_name)
        self.local_path_to_output_file = os.path.join(self.local_path, 'output.out')
        self.local_path_to_orbitals_file = os.path.join(self.local_path, 'orbitals.fchk')
        self.local_path_to_lj_file = os.path.join(self.local_path, 'lj.dat')
        self.local_path_to_check_file = os.path.join(self.local_path, 'check.chk')

        # parentheses don't play well in folder names:
        species_name_for_remote_path = self.species_name.replace('(', '_').replace(')', '_')
        self.remote_path = os.path.join('runs', 'ARC_Projects', self.project,
                                        species_name_for_remote_path, conformer_folder, self.job_name)

        self.additional_files_to_upload = list()
        # self.additional_files_to_upload is a list of dictionaries, each with the following keys:
        # 'name', 'source', 'local', and 'remote'.
        # If 'source' = 'path', then the value in 'local' is treated as a file path.
        # If 'source' = 'input_files', then the value in 'local' will be taken from the respective entry in inputs.py
        # If 'make_x' is True, the file will be made executable.
        if self.job_type == 'onedmin':
            if self.testing and not os.path.isdir(self.local_path):
                os.makedirs(self.local_path)
            with open(os.path.join(self.local_path, 'geo.xyz'), 'w') as f:
                f.write(self.xyz)
            self.additional_files_to_upload.append({'name': 'geo', 'source': 'path', 'make_x': False,
                                                    'local': os.path.join(self.local_path, 'geo.xyz'),
                                                    'remote': os.path.join(self.remote_path, 'geo.xyz')})
            # make the m.x file executable
            self.additional_files_to_upload.append({'name': 'm.x', 'source': 'input_files', 'make_x': True,
                                                    'local': 'onedmin.molpro.x',
                                                    'remote': os.path.join(self.remote_path, 'm.x')})
            self.additional_files_to_upload.append({'name': 'qc.mol', 'source': 'input_files', 'make_x': False,
                                                    'local': 'onedmin.qc.mol',
                                                    'remote': os.path.join(self.remote_path, 'qc.mol')})
        if self.job_type == 'gromacs':
            self.additional_files_to_upload.append({'name': 'gaussian.out', 'source': 'path', 'make_x': False,
                                                    'local': os.path.join(self.project_directory, 'calcs', 'Species',
                                                                          self.species_name, 'ff_param_fit',
                                                                          'gaussian.out'),
                                                    'remote': os.path.join(self.remote_path, 'gaussian.out')})
            self.additional_files_to_upload.append({'name': 'coords.yml', 'source': 'path', 'make_x': False,
                                                    'local': self.conformers,
                                                    'remote': os.path.join(self.remote_path, 'coords.yml')})
            self.additional_files_to_upload.append({'name': 'acpype.py', 'source': 'path', 'make_x': False,
                                                    'local': os.path.join(arc_path, 'arc', 'scripts', 'conformers',
                                                                          'acpype.py'),
                                                    'remote': os.path.join(self.remote_path, 'acpype.py')})
            self.additional_files_to_upload.append({'name': 'mdconf.py', 'source': 'path', 'make_x': False,
                                                    'local': os.path.join(arc_path, 'arc', 'scripts', 'conformers',
                                                                          'mdconf.py'),
                                                    'remote': os.path.join(self.remote_path, 'mdconf.py')})
            self.additional_files_to_upload.append({'name': 'M00.tleap', 'source': 'path', 'make_x': False,
                                                    'local': os.path.join(arc_path, 'arc', 'scripts', 'conformers',
                                                                          'M00.tleap'),
                                                    'remote': os.path.join(self.remote_path, 'M00.tleap')})
            self.additional_files_to_upload.append({'name': 'mdp.mdp', 'source': 'path', 'make_x': False,
                                                    'local': os.path.join(arc_path, 'arc', 'scripts', 'conformers',
                                                                          'mdp.mdp'),
                                                    'remote': os.path.join(self.remote_path, 'mdp.mdp')})
