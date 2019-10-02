#!/usr/bin/env python3
# encoding: utf-8

"""
The ARC Job module
"""

import csv
import datetime
import os
import shutil

from arc.exceptions import JobError, InputError
from arc.common import get_logger, calculate_dihedral_angle
from arc.job.inputs import input_files
from arc.job.local import get_last_modified_time, submit_job, delete_job, execute_command, check_job_status, \
    rename_output
from arc.job.submit import submit_scripts
from arc.job.ssh import SSHClient
from arc.job.trsh import determine_ess_status, trsh_job_on_server
from arc.settings import arc_path, servers, submit_filename, t_max_format, input_filename, output_filename, \
    rotor_scan_resolution, levels_ess
from arc.species.converter import xyz_to_str, str_to_xyz, check_xyz_dict


logger = get_logger()


class Job(object):
    """
    ARC's Job class.

    Args:
        project (str): The project's name. Used for naming the directory.
        project_directory (str): The path to the project directory.
        ess_settings (dict): A dictionary of available ESS and a corresponding server list.
        species_name (str): The species/TS name. Used for naming the directory.
        xyz (dict): The xyz geometry. Used for the calculation.
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
        directed_scans (list): Entries are lists of four-atom dihedral scan indices to constrain during a directed scan.
        directed_dihedrals (list): The dihedral angles of a directed scan job corresponding to ``directed_scans``.
        directed_scan_type (str): The type of the directed scan.
        rotor_index (int): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        job_dict (dict, optional): A dictionary to create this object from (used when restarting ARC).
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
        xyz (dict): The xyz geometry. Used for the calculation.
        radius (float): The species radius in Angstrom.
        n_atoms (int): The number of atoms in self.xyz.
        conformer (int): Conformer number if optimizing conformers.
        conformers (str): A path to the YAML file conformer coordinates for a Gromacs MD job.
        is_ts (bool): Whether this species represents a transition structure.
        level_of_theory (str): Level of theory, e.g. 'CBS-QB3', 'CCSD(T)-F12a/aug-cc-pVTZ', 'B3LYP/6-311++G(3df,3pd)'...
        job_type (str): The job's type.
        scan (list): A list representing atom labels for the dihedral scan (e.g., [2, 1, 3, 5]).
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
                           / 'errored' / 'done'. The job ESS status is in job.job_status[1] is a dictionary of
                           {'status': str, 'keywords': list, 'error': str, 'line': str}.
                           The values of 'status' can be either `initializing`, `running`, `errored`, `unconverged`,
                           or `done`. If the status is 'errored', then standardized error keywords, the error
                           description and the identified error line from the ESS log file will be given as well.
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
                        Allowed values are He, Ne, Ar, Kr, H2, N2, O2.
        directed_scans (list): Entries are lists of four-atom dihedral scan indices to constrain during a directed scan.
        directed_dihedrals (list): The dihedral angles of a directed scan job corresponding to ``directed_scans``.
        directed_scan_type (str): The type of the directed scan.
        rotor_index (int): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
    """
    def __init__(self, project='', ess_settings=None, species_name='', xyz=None, job_type='', level_of_theory='',
                 multiplicity=None, project_directory='', charge=0, conformer=-1, fine=False, shift='', software=None,
                 is_ts=False, scan=None, pivots=None, memory=14, comments='', trsh='', scan_trsh='', job_dict=None,
                 ess_trsh_methods=None, bath_gas=None, initial_trsh=None, job_num=None, job_server_name=None,
                 job_name=None, job_id=None, server=None, initial_time=None, occ=None, max_job_time=120, scan_res=None,
                 checkfile=None, number_of_radicals=None, conformers=None, radius=None, directed_scan_type=None,
                 directed_scans=None, directed_dihedrals=None, rotor_index=None, testing=False):
        if job_dict is not None:
            self.from_dict(job_dict)
        else:
            if not project:
                raise InputError('project must be specified')
            if not species_name:
                raise InputError('species_name must be specified')
            if not job_type:
                raise InputError('job_type must be specified')
            if not level_of_theory:
                raise InputError('level_of_theory must be specified')
            if ess_settings is None:
                raise InputError('ess_settings must be specified')
            if multiplicity is None:
                raise InputError('multiplicity must be specified')
            self.project = project
            self.project_directory = project_directory
            self.species_name = species_name
            self.initial_time = initial_time
            self.final_time = None
            self.run_time = None
            self.ess_settings = ess_settings
            self.job_num = job_num if job_num is not None else -1
            self.charge = charge
            self.multiplicity = multiplicity
            self.number_of_radicals = number_of_radicals
            self.xyz = check_xyz_dict(xyz)
            self.radius = radius
            self.directed_scan_type = directed_scan_type
            self.rotor_index = rotor_index
            self.directed_scans = directed_scans
            self.directed_dihedrals = directed_dihedrals
            self.conformer = conformer
            self.conformers = conformers
            self.is_ts = is_ts
            self.ess_trsh_methods = ess_trsh_methods if ess_trsh_methods is not None else list()
            self.trsh = trsh
            self.initial_trsh = initial_trsh if initial_trsh is not None else dict()
            self.scan_trsh = scan_trsh
            self.scan_res = scan_res if scan_res is not None else rotor_scan_resolution
            self.scan = scan
            self.pivots = pivots if pivots is not None else list()
            self.max_job_time = max_job_time
            self.bath_gas = bath_gas
            self.testing = testing
            self.fine = fine
            self.shift = shift
            self.occ = occ
            self.job_status = ['initializing', {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}]
            self.job_id = job_id if job_id is not None else 0
            self.comments = comments
            self.checkfile = checkfile
            self.server_nodes = list()
            self.job_type = job_type
            self.job_server_name = job_server_name
            self.job_name = job_name
            self.level_of_theory = level_of_theory.lower()
            self.server = server
            self.software = software
        # allowed job types:
        job_types = ['conformer', 'opt', 'freq', 'optfreq', 'sp', 'composite', 'bde', 'scan', 'directed_scan',
                     'gsm', 'irc', 'ts_guess', 'orbitals', 'onedmin', 'ff_param_fit', 'gromacs']
        if self.job_type not in job_types:
            raise ValueError('Job type {0} not understood. Must be one of the following:\n{1}'.format(
                self.job_type, job_types))
        if self.xyz is None and not self.job_type == 'gromacs':
            raise InputError('{0} Job of species {1} got None for xyz'.format(self.job_type, self.species_name))
        if self.job_type == 'gromacs' and self.conformers is None:
            raise InputError('{0} Job of species {1} got None for conformers'.format(self.job_type, self.species_name))
        if self.job_type == 'directed_scan' and (self.directed_dihedrals is None or self.directed_scans is None
                                                 or self.directed_scan_type is None):
            raise InputError('Must have the directed_dihedrals, directed_scans, and directed_scan_type attributes '
                             'for a directed scan job. Got {0}, {1}, {2}, respectively.'.format(
                              self.directed_dihedrals, self.directed_scans, self.directed_scan_type))
        if self.job_type == 'directed_scan' and self.rotor_index is None:
            raise InputError('Must have the rotor_index argument for a directed scan job.')

        if self.job_num < 0:
            self._set_job_number()
        self.job_server_name = self.job_server_name if self.job_server_name is not None else 'a' + str(self.job_num)
        self.job_name = self.job_name if self.job_name is not None else self.job_type + '_' + self.job_server_name

        # determine the level of theory and software to use:
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
        if self.server is None:  # might have been set in from_dict()
            self.server = server if server is not None else self.ess_settings[self.software][0]

        self.spin = self.multiplicity - 1
        self.n_atoms = len(self.xyz['symbols']) if self.xyz is not None else None
        self.submit = ''
        self.input = ''
        self.mem_per_cpu, self.cpus, self.memory_gb, self.memory = None, None, None, None
        self.set_cpu_and_mem(memory=memory)
        self.determine_run_time()

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
        job_dict['project'] = self.project
        job_dict['project_directory'] = self.project_directory
        job_dict['species_name'] = self.species_name
        job_dict['ess_settings'] = self.ess_settings
        job_dict['max_job_time'] = self.max_job_time
        job_dict['job_num'] = self.job_num
        job_dict['server'] = self.server
        job_dict['job_type'] = self.job_type
        job_dict['job_server_name'] = self.job_server_name
        job_dict['job_name'] = self.job_name
        job_dict['level_of_theory'] = self.level_of_theory
        job_dict['xyz'] = xyz_to_str(self.xyz)
        job_dict['fine'] = self.fine
        job_dict['job_status'] = self.job_status
        job_dict['memory'] = self.memory_gb
        job_dict['job_id'] = self.job_id
        job_dict['scan_res'] = self.scan_res
        job_dict['is_ts'] = self.is_ts
        job_dict['multiplicity'] = self.multiplicity
        if self.initial_time is not None:
            job_dict['initial_time'] = self.initial_time.strftime('%Y-%m-%d %H:%M:%S')
        if self.final_time is not None:
            job_dict['final_time'] = self.final_time.strftime('%Y-%m-%d %H:%M:%S')
        if self.server_nodes:
            job_dict['server_nodes'] = self.server_nodes
        if self.number_of_radicals is not None:
            job_dict['number_of_radicals'] = self.number_of_radicals
        if self.ess_trsh_methods:
            job_dict['ess_trsh_methods'] = self.ess_trsh_methods
        if self.trsh:
            job_dict['trsh'] = self.trsh
        if self.initial_trsh:
            job_dict['initial_trsh'] = self.initial_trsh
        if self.shift:
            job_dict['shift'] = self.shift
        if self.software is not None:
            job_dict['software'] = self.software
        if self.occ is not None:
            job_dict['occ'] = self.occ
        if self.conformer >= 0:
            job_dict['conformer'] = self.conformer
        if self.comments:
            job_dict['comments'] = self.comments
        if self.scan is not None:
            job_dict['scan'] = self.scan
        if self.pivots:
            job_dict['pivots'] = self.pivots
        if self.scan_trsh:
            job_dict['scan_trsh'] = self.scan_trsh
        if self.directed_dihedrals is not None:
            job_dict['directed_dihedrals'] = ['{0:.2f}'.format(dihedral) for dihedral in self.directed_dihedrals]
        if self.directed_scans is not None:
            job_dict['directed_scans'] = self.directed_scans
        if self.directed_scan_type is not None:
            job_dict['directed_scan_type'] = self.directed_scan_type
        if self.rotor_index is not None:
            job_dict['rotor_index'] = self.rotor_index
        if self.bath_gas is not None:
            job_dict['bath_gas'] = self.bath_gas
        if self.checkfile is not None:
            job_dict['checkfile'] = self.checkfile
        if self.conformers is not None:
            job_dict['conformers'] = self.conformers
        if self.radius is not None:
            job_dict['radius'] = self.radius
        return job_dict

    def from_dict(self, job_dict):
        """
        A helper function for loading this object from a dictionary in a YAML file for restarting ARC
        """
        # mandatory attributes:
        self.project = job_dict['project']
        self.project_directory = job_dict['project_directory']
        self.initial_time = datetime.datetime.strptime(job_dict['initial_time'], '%Y-%m-%d %H:%M:%S') \
            if 'initial_time' in job_dict else None
        self.final_time = datetime.datetime.strptime(job_dict['final_time'], '%Y-%m-%d %H:%M:%S') \
            if 'final_time' in job_dict else None
        self.ess_settings = job_dict['ess_settings']
        self.species_name = job_dict['species_name']
        self.job_type = job_dict['job_type']
        self.level_of_theory = job_dict['level_of_theory'].lower()
        self.multiplicity = job_dict['multiplicity']
        # optional attributes:
        self.xyz = str_to_xyz(job_dict['xyz']) if 'xyz' in job_dict else None
        self.server_nodes = job_dict['server_nodes'] if 'server_nodes' in job_dict else list()
        self.job_status = job_dict['job_status'] if 'job_status' in job_dict \
            else ['initializing', {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}]
        self.charge = job_dict['charge'] if 'charge' in job_dict else 0
        self.conformer = job_dict['conformer'] if 'conformer' in job_dict else -1
        self.fine = job_dict['fine'] if 'fine' in job_dict else False
        self.shift = job_dict['shift'] if 'shift' in job_dict else ''
        self.software = job_dict['software'] if 'software' in job_dict else None
        self.is_ts = job_dict['is_ts'] if 'is_ts' in job_dict else False
        self.scan = job_dict['scan'] if 'scan' in job_dict else None
        self.pivots = job_dict['pivots'] if 'pivots' in job_dict else list()
        self.memory = job_dict['memory'] if 'memory' in job_dict else 14
        self.comments = job_dict['comments'] if 'comments' in job_dict else ''
        self.trsh = job_dict['trsh'] if 'trsh' in job_dict else ''
        self.scan_trsh = job_dict['scan_trsh'] if 'scan_trsh' in job_dict else ''
        self.initial_trsh = job_dict['initial_trsh'] if 'initial_trsh' in job_dict else dict()
        self.ess_trsh_methods = job_dict['ess_trsh_methods'] if 'ess_trsh_methods' in job_dict else list()
        self.bath_gas = job_dict['bath_gas'] if 'bath_gas' in job_dict else None
        self.job_num = job_dict['job_num'] if 'job_num' in job_dict else -1
        self.job_server_name = job_dict['job_server_name'] if 'job_server_name' in job_dict else None
        self.job_name = job_dict['job_name'] if 'job_name' in job_dict else None
        self.job_id = job_dict['job_id'] if 'job_id' in job_dict else 0
        self.server = job_dict['server'] if 'server' in job_dict else None
        self.occ = job_dict['occ'] if 'occ' in job_dict else None
        self.max_job_time = job_dict['max_job_time'] if 'max_job_time' in job_dict else 120
        self.scan_res = job_dict['scan_res'] if 'scan_res' in job_dict else rotor_scan_resolution
        self.checkfile = job_dict['checkfile'] if 'checkfile' in job_dict else None
        self.number_of_radicals = job_dict['number_of_radicals'] if 'number_of_radicals' in job_dict else None
        self.conformers = job_dict['conformers'] if 'conformers' in job_dict else None
        self.radius = job_dict['radius'] if 'radius' in job_dict else None
        self.directed_dihedrals = [float(dihedral) for dihedral in job_dict['directed_dihedrals']] \
            if 'directed_dihedrals' in job_dict else None
        self.directed_scans = job_dict['directed_scans'] if 'directed_scans' in job_dict else None
        self.directed_scan_type = job_dict['directed_scan_type'] if 'directed_scan_type' in job_dict else None
        self.rotor_index = job_dict['rotor_index'] if 'rotor_index' in job_dict else None

    def _set_job_number(self):
        """
        Used as the entry number in the database, as well as the job name on the server.
        """
        csv_path = os.path.join(arc_path, 'initiated_jobs.csv')
        if not os.path.isfile(csv_path):
            # check file, make index file and write headers if file doesn't exists
            with open(csv_path, 'w') as f:
                writer = csv.writer(f, dialect='excel')
                row = ['job_num', 'project', 'species_name', 'conformer', 'is_ts', 'charge', 'multiplicity', 'job_type',
                       'job_name', 'job_id', 'server', 'software', 'memory', 'method', 'basis_set', 'comments']
                writer.writerow(row)
        with open(csv_path, 'r') as f:
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
        with open(csv_path, 'a') as f:
            writer = csv.writer(f, dialect='excel')
            row = [self.job_num, self.project, self.species_name, conformer, self.is_ts, self.charge,
                   self.multiplicity, self.job_type, self.job_name, self.job_id, self.server, self.software,
                   self.memory, self.method, self.basis_set, self.comments]
            writer.writerow(row)

    def write_completed_job_to_csv_file(self):
        """
        Write a completed ARCJob into the completed_jobs.csv file.
        """
        if self.job_status[0] != 'done' or self.job_status[1]['status'] != 'done':
            self.determine_job_status()
        csv_path = os.path.join(arc_path, 'completed_jobs.csv')
        if not os.path.isfile(csv_path):
            # check file, make index file and write headers if file doesn't exists
            with open(csv_path, 'w') as f:
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
        with open(csv_path, 'a') as f:
            writer = csv.writer(f, dialect='excel')
            job_type = self.job_type
            if self.fine:
                job_type += ' (fine)'
            row = [self.job_num, self.project, self.species_name, conformer, self.is_ts, self.charge,
                   self.multiplicity, job_type, self.job_name, self.job_id, self.server, self.software,
                   self.memory, self.method, self.basis_set, self.initial_time, self.final_time, self.run_time,
                   self.job_status[0], self.job_status[1]['status'], self.ess_trsh_methods, self.comments]
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
        if not os.path.isdir(self.local_path):
            os.makedirs(self.local_path)
        with open(os.path.join(self.local_path, submit_filename[servers[self.server]['cluster_soft']]), 'w') as f:
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

        slash, scan_string, constraint = '', '', ''
        if self.software == 'gaussian' and '/' in self.level_of_theory:
            slash = '/'

        if (self.multiplicity > 1 and '/' in self.level_of_theory) \
                or (self.number_of_radicals is not None and self.number_of_radicals > 1):
            # don't add 'u' to composite jobs. Do add 'u' for bi-rad singlets if `number_of_radicals` > 1
            if self.number_of_radicals is not None and self.number_of_radicals > 1:
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
                if self.fine:
                    # Note that the Acc2E argument is not available in Gaussian03
                    fine = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                    if self.is_ts:
                        job_type_1 = 'opt=(ts, calcfc, noeigentest, tight, maxstep=5)'
                    else:
                        job_type_1 = 'opt=(tight)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
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
            if not divmod(360, self.scan_res):
                raise JobError('Scan job got an illegal rotor scan resolution of {0}'.format(self.scan_res))
            scan = ' '.join([str(num) for num in self.scan])
            if self.software == 'gaussian':
                if self.is_ts:
                    job_type_1 = 'opt=(ts, modredundant, calcfc, noeigentest, maxStep=5) scf=(tight, direct) ' \
                                 'integral=(grid=ultrafine, Acc2E=12)'
                else:
                    job_type_1 = 'opt=(modredundant, calcfc, noeigentest, maxStep=5) scf=(tight, direct) ' \
                                 'integral=(grid=ultrafine, Acc2E=12)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
                scan_string = 'D {scan} S {steps} {increment:.1f}'.format(scan=scan,
                                                                          steps=str(int(360 / self.scan_res)),
                                                                          increment=float(self.scan_res))
            elif self.software == 'qchem':
                if self.is_ts:
                    job_type_1 = 'ts'
                else:
                    job_type_1 = 'opt'
                dihedral1 = int(calculate_dihedral_angle(coords=self.xyz['coords'], torsion=self.scan))
                dihedral2 = dihedral1 - self.scan_res
                if dihedral2 < -180:
                    dihedral2 += 360
                scan_string = """
$scan
    tors {scan} {dihedral1} {dihedral2} {increment}
$end
""".format(scan=scan, dihedral1=0, dihedral2=0, increment=self.scan_res)
            else:
                raise ValueError('Currently rotor scan is only supported in Gaussian and QChem. Got: {0} using the '
                                 '{1} level of theory'.format(self.software, self.method + '/' + self.basis_set))

        elif self.job_type == 'directed_scan':
            # this is either a constrained opt job or an sp job (depends on self.directed_scan_type).
            # If opt, the dihedrals in self.directed_dihedral are constrained.
            if self.software == 'gaussian':
                if 'sp' in self.directed_scan_type:
                    job_type_1 = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                else:
                    if self.is_ts:
                        job_type_1 = 'opt=(ts, modredundant, calcfc, noeigentest, maxStep=5) scf=(tight, direct)' \
                                     ' integral=(grid=ultrafine, Acc2E=12)'
                    else:
                        job_type_1 = 'opt=(modredundant, calcfc, noeigentest, maxStep=5) scf=(tight, direct)' \
                                     ' integral=(grid=ultrafine, Acc2E=12)'
                    for directed_scan, directed_dihedral in zip(self.directed_scans, self.directed_dihedrals):
                        scan_atoms = ' '.join([str(num) for num in directed_scan])
                        scan_string += 'D {scan} ={dihedral} B\nD {scan} F\n'.format(
                            scan=scan_atoms, dihedral='{0:.2f}'.format(directed_dihedral))
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
            elif self.software == 'qchem':
                # following https://manual.q-chem.com/5.2/Ch10.S3.SS4.html
                if 'sp' in self.directed_scan_type:
                    job_type_1 = 'sp'
                else:
                    if self.is_ts:
                        job_type_1 = 'ts'
                    else:
                        job_type_1 = 'opt'
                    constraint = '\n    CONSTRAINT\n'
                    for directed_scan, directed_dihedral in zip(self.directed_scans, self.directed_dihedrals):
                        scan_atoms = ' '.join([str(num) for num in directed_scan])
                        constraint += '        tors {scan} {dihedral}\n'.format(
                            scan=scan_atoms, dihedral='{0:.2f}'.format(directed_dihedral))
                    constraint += '    ENDCONSTRAINT\n'
            else:
                raise ValueError('Currently directed rotor scans are only supported in Gaussian and QChem. '
                                 'Got: {0} using the {1} level of theory'.format(
                                  self.software, self.method + '/' + self.basis_set))

        if self.software == 'gaussian' and not self.trsh:
            if self.level_of_theory[:2] == 'ro':
                self.trsh = 'use=L506'
            else:
                # xqc will do qc (quadratic convergence) if the job fails w/o it, so use by default
                self.trsh = 'scf=xqc'

        if self.job_type == 'irc':  # TODO
            if self.fine:
                # Note that the Acc2E argument is not available in Gaussian03
                fine = 'scf=(direct) integral=(grid=ultrafine, Acc2E=12)'
            job_type_1 = 'irc=(CalcAll,forward,maxpoints=50,stepsize=7)'  # also run reverse; trsh: stepsize=20
            if self.checkfile is not None:
                job_type_1 += ' guess=read'
            else:
                job_type_1 += ' guess=mix'

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
                    self.input = input_files['mrci'].format(memory=self.memory, xyz=xyz_to_str(self.xyz),
                                                            basis=self.basis_set, spin=self.spin, charge=self.charge,
                                                            trsh=self.trsh)
                except KeyError:
                    logger.error('Could not interpret all input file keys in\n{0}'.format(self.input))
                    raise
        else:
            try:
                self.input = self.input.format(memory=int(self.memory), method=self.method, slash=slash,
                                               basis=self.basis_set, charge=self.charge, cpus=self.cpus,
                                               multiplicity=self.multiplicity, spin=self.spin, xyz=xyz_to_str(self.xyz),
                                               job_type_1=job_type_1, job_type_2=job_type_2, scan=scan_string,
                                               restricted=restricted, fine=fine, shift=self.shift, trsh=self.trsh,
                                               scan_trsh=self.scan_trsh, bath=self.bath_gas, constraint=constraint) \
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
                            f.write(input_files[up_file['local']])

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
            logger.info('Running job {name} for {label} (fine opt)'.format(
                name=self.job_name, label=self.species_name))
        elif self.directed_dihedrals is not None and self.directed_scans is not None:
            dihedrals = ['{0:.2f}'.format(dihedral) for dihedral in self.directed_dihedrals]
            logger.info('Running job {name} for {label} (pivots: {pivots}, dihedrals: {dihedrals})'.format(
                name=self.job_name, label=self.species_name, pivots=self.directed_scans, dihedrals=dihedrals))
        elif self.pivots:
            logger.info('Running job {name} for {label} (pivots: {pivots})'.format(
                name=self.job_name, label=self.species_name, pivots=self.pivots))
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

        Raises:
            IOError: If the output file and any additional server information cannot be found.
        """
        if self.job_status[0] == 'errored':
            return
        self.job_status[0] = self._check_job_server_status()
        if self.job_status[0] == 'done':
            try:
                self._check_job_ess_status()  # populates self.job_status[1], and downloads the output file
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
                            self.job_status[1]['status'] = 'errored'
                            self.job_status[1]['keywords'] = ['ServerTimeLimit']
                            self.job_status[1]['error'] = 'Job cancelled by the server since it reached the maximal ' \
                                                          'time limit.'
                            self.job_status[1]['line'] = ''
                raise
        elif self.job_status[0] == 'running':
            self.job_status[1]['status'] = 'running'

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
                    logger.warning(e)
                remote_file_path = os.path.join(self.remote_path, 'err.txt')
                try:
                    ssh.download_file(remote_file_path=remote_file_path, local_file_path=local_file_path2)
                except (TypeError, IOError) as e:
                    logger.warning('Got the following error when trying to download err.txt for {0}:'.format(
                        self.job_name))
                    logger.warning(e)
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
            for line in response[0]:
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
                            logger.warning(e)
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
        status, keywords, error, line = determine_ess_status(output_path=self.local_path_to_output_file,
                                                             species_label=self.species_name, job_type=self.job_type,
                                                             software=self.software)
        self.job_status[1]['status'] = status
        self.job_status[1]['keywords'] = keywords
        self.job_status[1]['error'] = error
        self.job_status[1]['line'] = line.rstrip()

    def troubleshoot_server(self):
        """
        Troubleshoot server errors.
        """
        node, run_job = trsh_job_on_server(server=self.server, job_name=self.job_name, job_id=self.job_id,
                                           job_server_status=self.job_status[0], remote_path=self.remote_path,
                                           server_nodes=self.server_nodes)
        if node is not None:
            self.server_nodes.append(node)
        if run_job:
            # resubmit job
            self.run()

    def determine_run_time(self):
        """
        Determine the run time. Update self.run_time and round to seconds.
        """
        if self.initial_time is not None and self.final_time is not None:
            time_delta = self.final_time - self.initial_time
            remainder = time_delta.microseconds > 5e5
            self.run_time = datetime.timedelta(seconds=time_delta.seconds + remainder)
        else:
            self.run_time = None

    def deduce_software(self):
        """
        Deduce the software to be used based on heuristics.
        """
        esss = self.ess_settings.keys()
        if self.job_type == 'onedmin':
            if 'onedmin' not in esss:
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
            if 'gromacs' not in esss:
                raise JobError('Could not find the Gromacs software to run the MD job {0}.\n'
                               'ess_settings is:\n{1}'.format(self.method, self.ess_settings))
            self.software = 'gromacs'
        elif self.job_type == 'orbitals':
            # currently we only have a script to print orbitals on QChem,
            # could/should definitely be elaborated to additional ESS
            if 'qchem' not in esss:
                logger.debug('Could not find the QChem software to compute molecular orbitals.\n'
                             'ess_settings is:\n{0}'.format(self.ess_settings))
                self.software = None
            else:
                self.software = 'qchem'
        elif self.job_type == 'composite':
            if 'gaussian' not in esss:
                raise JobError('Could not find the Gaussian software to run the composite method {0}.\n'
                               'ess_settings is:\n{1}'.format(self.method, self.ess_settings))
            self.software = 'gaussian'
        elif self.job_type == 'ff_param_fit':
            if 'gaussian' not in esss:
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
                if self.job_type in ['conformer', 'opt', 'freq', 'optfreq', 'sp',
                                     'directed_scan']:
                    if 'b2' in self.method or 'dsd' in self.method or 'pw2' in self.method:
                        # this is a double-hybrid (MP2) DFT method, use Gaussian
                        if 'gaussian' not in esss:
                            raise JobError('Could not find the Gaussian software to run the double-hybrid method {0}.\n'
                                           'ess_settings is:\n{1}'.format(self.method, self.ess_settings))
                        self.software = 'gaussian'
                    elif 'ccs' in self.method or 'cis' in self.method:
                        if 'molpro' in esss:
                            self.software = 'molpro'
                        elif 'gaussian' in esss:
                            self.software = 'gaussian'
                        elif 'qchem' in esss:
                            self.software = 'qchem'
                    elif 'b3lyp' in self.method:
                        if 'gaussian' in esss:
                            self.software = 'gaussian'
                        elif 'qchem' in esss:
                            self.software = 'qchem'
                        elif 'molpro' in esss:
                            self.software = 'molpro'
                    elif 'wb97xd' in self.method:
                        if 'gaussian' not in esss:
                            raise JobError('Could not find the Gaussian software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'gaussian'
                    elif 'wb97x-d3' in self.method or 'wb97m' in self.method:
                        if 'qchem' not in esss:
                            raise JobError('Could not find the QChem software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'qchem'
                    elif 'b97' in self.method or 'def2' in self.basis_set:
                        if 'gaussian' in esss:
                            self.software = 'gaussian'
                        elif 'qchem' in esss:
                            self.software = 'qchem'
                    elif 'm062x' in self.method:  # without dash
                        if 'gaussian' not in esss:
                            raise JobError('Could not find the Gaussian software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'gaussian'
                    elif 'm06-2x' in self.method:  # with dash
                        if 'qchem' not in esss:
                            raise JobError('Could not find the QChem software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'qchem'
                elif self.job_type == 'scan':
                    if 'wb97xd' in self.method:
                        if 'gaussian' not in esss:
                            raise JobError('Could not find the Gaussian software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'gaussian'
                    elif 'wb97x-d3' in self.method:
                        if 'qchem' not in esss:
                            raise JobError('Could not find the QChem software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'qchem'
                    elif 'b3lyp' in self.method:
                        if 'gaussian' in esss:
                            self.software = 'gaussian'
                        elif 'qchem' in esss:
                            self.software = 'qchem'
                    elif 'b97' in self.method or 'def2' in self.basis_set:
                        if 'gaussian' in esss:
                            self.software = 'gaussian'
                        elif 'qchem' in esss:
                            self.software = 'qchem'
                    elif 'm06-2x' in self.method:  # with dash
                        if 'qchem' not in esss:
                            raise JobError('Could not find the QChem software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'qchem'
                    elif 'm062x' in self.method:  # without dash
                        if 'gaussian' not in esss:
                            raise JobError('Could not find the Gaussian software to run {0}/{1}'.format(
                                self.method, self.basis_set))
                        self.software = 'gaussian'
                    elif 'pv' in self.basis_set:
                        if 'molpro' in esss:
                            self.software = 'molpro'
                        elif 'gaussian' in esss:
                            self.software = 'gaussian'
                        elif 'qchem' in esss:
                            self.software = 'qchem'
                elif self.job_type in ['gsm', 'irc']:
                    if 'gaussian' not in esss:
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
                if 'gaussian' in esss:
                    logger.error('Setting it to Gaussian')
                    self.software = 'gaussian'
                elif 'orca' in esss:
                    logger.error('Setting it to Orca')
                    self.software = 'orca'
                elif 'qchem' in esss:
                    logger.error('Setting it to QChem')
                    self.software = 'qchem'
                elif 'molpro' in esss:
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
                f.write(xyz_to_str(self.xyz))
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
