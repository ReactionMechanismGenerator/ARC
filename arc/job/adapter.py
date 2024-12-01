"""
A module for the abstract JobAdapter class

A job execution type could be either "incore" (run on the calling machine) or "queue" (submitted to a cluster)
A job could (independently) consist a single process (e.g., optimizing a single well)
or have multiple processes (an array) executed by several "workers" on a server.
A single process job could run "incore" or be submitted to the queue.
A multiprocess job could be executed "incore" (taking no advantage of potential server parallelization)
or be submitted to the queue, in which case a powerful "pipe" (similar to job arrays) will be created.
"""

import csv
import datetime
import itertools
import math
import os
import shutil
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from arc.common import ARC_PATH, get_logger, read_yaml_file, save_yaml_file, torsions_to_scans, convert_to_hours
from arc.exceptions import JobError
from arc.imports import local_arc_path, pipe_submit, settings, submit_scripts
from arc.job.local import (change_mode,
                           check_job_status,
                           delete_job,
                           get_last_modified_time,
                           rename_output,
                           submit_job,
                           )
from arc.job.trsh import trsh_job_on_server, trsh_job_queue
from arc.job.ssh import SSHClient
from arc.job.trsh import determine_ess_status
from arc.species.converter import xyz_to_str
from arc.species.vectors import calculate_dihedral_angle

if TYPE_CHECKING:
    from arc.species import ARCSpecies


logger = get_logger()

default_job_settings, servers, submit_filenames, t_max_format, input_filenames, output_filenames, workers_coeff = \
    settings['default_job_settings'], settings['servers'], settings['submit_filenames'], settings['t_max_format'], \
    settings['input_filenames'], settings['output_filenames'], settings['workers_coeff']

constraint_type_dict = {2: 'B', 3: 'A', 4: 'D'}


class JobEnum(str, Enum):
    """
    The supported job software adapters.
    The available adapters are a finite set.

    Consider adding the following adapters:
        - ESS:
            - cosmo
            - PySCF (https://pyscf.org/user/geomopt.html)
            - TS opt via pysisyphus (https://pysisyphus.readthedocs.io/en/dev/tsoptimization.html)
            - onedmin
            - openbabel
            - rdkit
            - terachem
            - AIMNet (https://github.com/aiqm/aimnet)
            - turbomol
            - xtb
        - TS search:
            - gsm   # Double ended growing string method (DE-GSM), [10.1021/ct400319w, 10.1063/1.4804162]
            - pygsm  # Double ended growing string method (DE-GSM): pyGSM: https://github.com/ZimmermanGroup/molecularGSM/wiki
            - neb_ase  # NEB in ASE: https://www.scm.com/doc/Tutorials/ADF/Transition_State_with_ASE.html, https://wiki.fysik.dtu.dk/ase/ase/neb.html, https://wiki.fysik.dtu.dk/ase/tutorials/neb/idpp.html#idpp-tutorial., ASE autoNEB: https://wiki.fysik.dtu.dk/ase/dev/_modules/ase/autoneb.html
            - neb_terachem  # NEB in TeraChem, [10.1063/1.1329672, 10.1063/1.1323224]
            - neb_gpr  # NEB GPR:  https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.156001
            - neb_terpolation  # NEBTERPOLATION using MD to find TSs: https://pubs.acs.org/doi/full/10.1021/acs.jctc.5b00830?src=recsys
            - qst2  # Synchronous Transit-Guided Quasi-Newton (STQN) implemented in Gaussian, https://onlinelibrary.wiley.com/doi/epdf/10.1002/ijch.199300051
            - readuct  # ReaDuct: https://doi.org/10.1021/acs.jctc.8b00169
            - copenhagen  # Copenhagen: https://chemrxiv.org/articles/Fast_and_Automatic_Estimation_of_Transition_State_Structures_Using_Tight_Binding_Quantum_Chemical_Calculations/12600443/1
            - fsm  # FSM in QChem: http://www.q-chem.com/qchem-website/manual/qchem43_manual/sect-approx_hess.html, 10.1021/acs.jctc.5b00407, 10.1063/1.3664901
            - gan  # Generative adversarial networks, https://doi.org/10.1063/5.0055094
    """
    # ESS
    cfour = 'cfour'
    gaussian = 'gaussian'
    mockter = 'mockter'
    molpro = 'molpro'
    orca = 'orca'
    psi4 = 'psi4'
    qchem = 'qchem'
    terachem = 'terachem'
    torchani = 'torchani'
    xtb = 'xtb'
    openbabel = 'openbabel'

    # TS search methods
    autotst = 'autotst'  # AutoTST, 10.1021/acs.jpca.7b07361, 10.26434/chemrxiv.13277870.v2
    heuristics = 'heuristics'  # ARC's heuristics
    kinbot = 'kinbot'  # KinBot, 10.1016/j.cpc.2019.106947
    gcn = 'gcn'  # Graph neural network for isomerization, https://doi.org/10.1021/acs.jpclett.0c00500
    user = 'user'  # user guesses
    xtb_gsm = 'xtb_gsm'   # Double ended growing string method (DE-GSM), [10.1021/ct400319w, 10.1063/1.4804162] via xTB


class JobTypeEnum(str, Enum):
    """
    The supported job types.
    The available jon types are a finite set.
    """
    composite = 'composite'
    conf_opt = 'conf_opt'  # conformer optimization (not generation)
    conf_sp = 'conf_sp'  # conformer single point
    freq = 'freq'
    gen_confs = 'gen_confs'  # conformer generation
    irc = 'irc'
    onedmin = 'onedmin'
    opt = 'opt'
    optfreq = 'optfreq'
    orbitals = 'orbitals'
    scan = 'scan'
    directed_scan = 'directed_scan'
    sp = 'sp'
    tsg = 'tsg'  # TS search (TS guess)


class JobExecutionTypeEnum(str, Enum):
    """
    The supported job execution types.
    The available job execution types are a finite set.
    """
    incore = 'incore'
    queue = 'queue'
    pipe = 'pipe'


class DataPoint(object):
    """
    A class for representing a data point dictionary (a single job) per species for the HDF5 file.

    Args:
        job_types (List[str]): The job types to be executed in sequence.
        label (str): The species label.
        level (dict): The level of theory, a Level.dict() representation.
        xyz_1 (dict): The cartesian coordinates to consider.
        args (dict, str, optional): Methods (including troubleshooting) to be used in input files.
        bath_gas (str, optional): A bath gas. Currently only used in OneDMin to calculate L-J parameters.
        charge (int): The species (or TS) charge.
        constraints (List[Tuple[List[int], float]], optional): Optimization constraint.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
        dihedrals (List[float], optional): The dihedral angels corresponding to self.torsions.
        fine (bool, optional): Whether to use fine geometry optimization parameters. Default: ``False``.
        irc_direction (str, optional): The direction of the IRC job (`forward` or `reverse`).
        multiplicity (int): The species (or TS) multiplicity.
        torsions (List[List[int]], optional): The 0-indexed atom indices of the torsion(s).
        xyz_2 (dict, optional): Additional cartesian coordinates to consider in double-ended TS search methods.
    """

    def __init__(self,
                 job_types: List[str],
                 label: str,
                 level: dict,
                 xyz_1: dict,
                 args: Optional[Union[dict, str]] = None,
                 bath_gas: Optional[str] = None,
                 charge: int = 0,
                 constraints: Optional[List[Tuple[List[int], float]]] = None,
                 cpu_cores: Optional[str] = None,
                 dihedrals: Optional[List[float]] = None,
                 fine: bool = False,
                 irc_direction: Optional[str] = None,
                 multiplicity: int = 1,
                 torsions: Optional[List[List[int]]] = None,
                 xyz_2: Optional[dict] = None,
                 ):
        self.job_types = job_types
        self.label = label
        self.level = level
        self.xyz_1 = xyz_1

        self.args = args
        self.bath_gas = bath_gas
        self.charge = charge
        self.constraints = constraints
        self.cpu_cores = cpu_cores
        self.dihedrals = dihedrals
        self.fine = fine
        self.irc_direction = irc_direction
        self.multiplicity = multiplicity
        self.torsions = torsions
        self.xyz_2 = xyz_2

        self.status = 0

        # initialize outputs
        self.electronic_energy = None
        self.error = None
        self.frequencies = None
        self.xyz_out = None

    def as_dict(self):
        """
        A dictionary representation of the object, not storing default or trivial data.

        Returns: dict
            The dictionary representation.
        """
        result = {'job_types': self.job_types,
                  'label': self.label,
                  'level': self.level,
                  'xyz_1': self.xyz_1,
                  'status': self.status,
                  'electronic_energy': self.electronic_energy,
                  'error': self.error,
                  'frequencies': self.frequencies,
                  'xyz_out': self.xyz_out,
                  }
        if self.args is not None:
            result['args'] = self.args
        if self.bath_gas is not None:
            result['bath_gas'] = self.bath_gas
        if self.charge != 0:
            result['charge'] = self.charge
        if self.constraints is not None:
            result['constraints'] = self.constraints
        if self.cpu_cores is not None:
            result['cpu_cores'] = self.cpu_cores
        if self.fine:
            result['fine'] = self.fine
        if self.irc_direction is not None:
            result['irc_direction'] = self.irc_direction
        if self.multiplicity != 1:
            result['multiplicity'] = self.multiplicity
        if self.xyz_2 is not None:
            result['xyz_2'] = self.xyz_2
        return result


class JobAdapter(ABC):
    """
    An abstract class for job adapters.
    """

    @abstractmethod
    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.
        """
        pass

    @abstractmethod
    def set_files(self) -> None:
        """
        Set files to be uploaded and downloaded. Writes the files if needed.
        Modifies the self.files_to_upload and self.files_to_download attributes.

        self.files_to_download is a list of remote paths.

        self.files_to_upload is a list of dictionaries, each with the following keys:
        ``'name'``, ``'source'``, ``'make_x'``, ``'local'``, and ``'remote'``.
        If ``'source'`` = ``'path'``, then the value in ``'local'`` is treated as a file path.
        Else if ``'source'`` = ``'input_files'``, then the value in ``'local'`` will be taken
        from the respective entry in inputs.py
        If ``'make_x'`` is ``True``, the file will be made executable.
        """
        pass

    @abstractmethod
    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        pass

    @abstractmethod
    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        pass

    @abstractmethod
    def execute_incore(self):
        """
        Execute a job incore.
        """
        pass

    @abstractmethod
    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        pass

    def execute(self):
        """
        Execute a job.
        The execution type could be 'incore', 'queue', or 'pipe'.

        An 'incore' execution type assumes a single job (if more are given, only the first one will be executed),
        and executes the job in the same CPU process as ARC (i.e., Python waits for the response).

        A 'queue' execution type assumes a single job (if more are given, only the first one will be executed),
        and submits that single job to the server queue. The server could be either remote (accessed via SSH) or local.

        A 'pipe' execution type assumes an array of jobs and submits several ARC instances (workers)
        with an HDF5 file that contains specific directions.
        The output is returned within the HDF5 file.
        The new ARC instance, representing a single worker, will run all of its jobs incore.
        """
        self.upload_files()
        execution_type = JobExecutionTypeEnum(self.execution_type)
        if execution_type == JobExecutionTypeEnum.incore:
            self.initial_time = datetime.datetime.now()
            self.job_status[0] = 'running'
            self.execute_incore()
            self.job_status[0] = 'done'
            self.job_status[1]['status'] = 'done'
            self.final_time = datetime.datetime.now()
            self.determine_run_time()
        elif execution_type == JobExecutionTypeEnum.queue:
            self.execute_queue()
        elif execution_type == JobExecutionTypeEnum.pipe:
            # Todo:
            #   - Check that the HDF5 file is available, else raise an error.
            #   - Submit ARC workers with the HDF5 file.
            self.execute_queue()  # This is temporary until pipe is fully functional.
        if not self.restarted:
            self._write_initiated_job_to_csv_file()

    def legacy_queue_execution(self):
        """
        Execute a job to the server's queue.
        The server could be either "local" or remote.
        """
        self._log_job_execution()
        # Submit to queue, differentiate between local (same machine using its queue) and remote servers.
        if self.server != 'local':
            with SSHClient(self.server) as ssh:
                self.job_status[0], self.job_id = ssh.submit_job(remote_path=self.remote_path)
        else:
            # submit to the local queue
            self.job_status[0], self.job_id = submit_job(path=self.local_path)

    def set_job_shell_file_to_upload(self) -> dict:
        """
        The HTCondor cluster software does not allow, to the best of our understanding,
        to include sell commands in its submit script. As a result, often an additional
        ``job.sh`` file is required. This method generalizes such cases for all cluster software
        and will search within ARC's ``submit_scripts`` dictionary for a respective server and a
        respective ESS to see whether an additional ``<ess>_job`` key is available, where ``<ess>`` is
        the actual ESS name, e.g., ``gaussian_job``.
        This file will be uploaded to the server simply as ``job.sh`` and will be set as executable.

        Returns:
            dict: A file representation.
        """
        file_name, script_key = 'job.sh', f'{self.job_adapter}_job'
        if self.server in submit_scripts.keys() and script_key in submit_scripts[self.server].keys():
            file_content = submit_scripts[self.server][script_key].format(un='$USER', cpus=self.cpu_cores)
            with open(os.path.join(self.local_path, file_name), 'w') as f:
                f.write(file_content)
                if self.server == 'local':
                    change_mode(mode='+x', file_name=file_name, path=self.local_path)
            return self.get_file_property_dictionary(file_name=file_name, make_x=True)

    def determine_job_array_parameters(self):
        """
        Determine the number of processes to use in a job array
        and whether to iterate by conformers, species, reactions, or scan constraints.

        Explaining "workers" vs. "processes":
        A pipe job may have, e.g., 1000 individual processes to compute.
        ARC will allocate, e.g., 8 workers, to simultaneously get processes (one by one) from the HDF5 bank
        and execute them. On average, each worker in this example executes 125 jobs.
        """
        if self.execution_type == 'incore' or self.run_multi_species:
            return None
        if len(self.job_types) > 1:
            self.iterate_by.append('job_types')

        for job_type in self.job_types:
            if self.species is not None:
                if len(self.species) > 1:
                    self.iterate_by.append('species')
                if job_type == 'conf_opt':
                    if self.species is not None and sum(len(species.conformers) for species in self.species) > 10:
                        self.iterate_by.append('conf_opt')
                        self.number_of_processes += sum([len(species.conformers) for species in self.species])
                for species in self.species:
                    if job_type in ['sp', 'opt', 'freq', 'optfreq', 'composite', 'ornitals', 'onedmin', 'irc']:
                        self.number_of_processes += 1
                    # elif job_type == 'scan' and rotor_dict['directed_scan_type'] != 'ess':  # Todo: implement directed scans
                    elif job_type == 'scan' and len(species.rotors_dict.keys()) > 1000:  # Todo: Modify when pipe is implemented
                        self.iterate_by.append('scan')
                        scan_points_per_dimension = 360.0 / self.scan_res
                        for rotor_dict in species.rotors_dict.values():
                            if rotor_dict['directed_scan_type'] == 'ess':
                                self.number_of_processes += 1
                            elif 'cont_opt' in rotor_dict['directed_scan_type']:
                                # A single calculation per species for a continuous scan, either diagonal or not.
                                self.number_of_processes += 1
                            elif 'brute_force' in rotor_dict['directed_scan_type']:
                                if 'diagonal' in rotor_dict['directed_scan_type']:
                                    self.number_of_processes += scan_points_per_dimension
                                else:
                                    self.number_of_processes += \
                                        sum([scan_points_per_dimension ** len(rotor_dict['scan'])])

            elif self.reactions is not None:
                if len(self.reactions) > 1:
                    self.iterate_by.append('reactions')
                self.number_of_processes += len(self.reactions)

        if self.number_of_processes > self.incore_capacity:
            self.execution_type = 'pipe'
            self._determine_workers()
            self.write_hdf5()

    def _determine_workers(self):
        """
        Determine the number of workers to use in a pipe job.
        """
        if self.workers is None:
            if self.number_of_processes <= workers_coeff['max_one']:
                self.workers = 1
            elif self.number_of_processes <= workers_coeff['max_two']:
                self.workers = 2
            else:
                self.workers = min(round(workers_coeff['A'] * self.number_of_processes ** workers_coeff['b']),
                                   workers_coeff['cap'])

    def write_hdf5(self):
        """
        Write the HDF5 data file for job arrays.
        Each data point is a dictionary representation of the DataPoint class.
        Note: Each data point will always run "incore". A job array is created once the pipe is submitted to the queue
        (rather than running the pipe "incore", taking no advantage of the server's potential for parallelization).
        """
        if self.iterate_by:
            data = dict()
            if 'reactions' in self.iterate_by:
                for reaction in self.reactions:
                    data[reaction.index] = list()
                    data[reaction.index].append(DataPoint(charge=reaction.charge,
                                                          job_types=[self.job_type],
                                                          label=reaction.label,
                                                          level=self.level.as_dict(),
                                                          multiplicity=reaction.multiplicity,
                                                          xyz_1=reaction.get_reactants_xyz(),
                                                          xyz_2=reaction.get_products_xyz(),
                                                          constraints=self.constraints,
                                                          ).as_dict())
            else:
                for species in self.species:
                    data[species.label] = list()
                    if 'conf_opt' in self.iterate_by:
                        for conformer in species.conformers:
                            data[species.label].append(DataPoint(charge=species.charge,
                                                                 job_types=['opt'],
                                                                 label=species.label,
                                                                 level=self.level.as_dict(),
                                                                 multiplicity=species.multiplicity,
                                                                 xyz_1=conformer,
                                                                 ).as_dict())
                    elif 'scan' in self.iterate_by:
                        data[species.label].extend(self.generate_scan_points(species=species))
                    elif 'species' in self.iterate_by:
                        data[species.label].append(DataPoint(charge=species.charge,
                                                             job_types=[self.job_type],
                                                             label=species.label,
                                                             level=self.level.as_dict(),
                                                             multiplicity=species.multiplicity,
                                                             xyz_1=species.get_xyz(),
                                                             constraints=self.constraints,
                                                             ).as_dict())

            df = pd.json_normalize(data)
            df.to_hdf(os.path.join(self.local_path, 'data.hdf5'), key='df', mode='w')

    def write_submit_script(self) -> None:
        """
        Write a submit script to execute the job.
        """
        if self.server is None:
            return
        if self.max_job_time > 9999 or self.max_job_time <= 0:
            self.max_job_time = 120
        architecture = ''
        if self.server.lower() == 'pharos':
            # here we're hard-coding ARC for Pharos, a Green Group server at MIT.
            # If your server has different node architectures, implement something similar here.
            architecture = '\n#$ -l harpertown' if self.cpu_cores <= 8 else '\n#$ -l magnycours'

        # Extract the 'queue' dictionary from servers[self.server], defaulting to an empty dictionary if 'queue' is not a key
        default_queue, _ = next(iter(servers[self.server].get('queues', {}).items()), (None, None))
        if default_queue and default_queue not in self.attempted_queues:
            self.attempted_queues.append(default_queue)

        submit_script = submit_scripts[self.server][self.job_adapter] if self.workers is None \
            else pipe_submit[self.server]

        queue = self.queue if self.queue is not None else default_queue

        format_params = {
            "name": self.job_server_name,
            "un": servers[self.server]['un'],
            "queue": queue,
            "t_max": self.format_max_job_time(time_format=t_max_format[servers[self.server]['cluster_soft']]),
            "memory": int(self.submit_script_memory) if isinstance(self.submit_script_memory, (int, float)) else self.submit_script_memory,
            "cpus": self.cpu_cores,
            "architecture": architecture,
            "max_task_num": self.workers,
            "arc_path": ARC_PATH,
            "hdf5_path": os.path.join(self.remote_path, 'data.hdf5'),
            "pwd": self.remote_path if self.server.lower() != 'local' else self.local_path,
        }

        if queue is None:
            logger.debug(f'Queue not defined for server {self.server}. '
                         f'Assuming the queue name is defined in your submit.py script.')
            del format_params['queue']

        try:
            submit_script = submit_script.format(**format_params)
        except KeyError:
            if self.workers is None:
                submit_scripts_for_printing = {server: [software for software in values.keys()]
                                               for server, values in submit_scripts.items()}
                pipe = ''
            else:
                submit_scripts_for_printing = {server for server, values in pipe_submit.keys()}
                pipe = ' pipe'
            logger.error(f'Could not find{pipe} submit script for server {self.server} and software {self.job_adapter}.'
                         f'\nMake sure your submit scripts (under arc/job/submit.py) are updated with the servers '
                         f'and software defined in arc/settings.py\n'
                         f'Alternatively, It is possible that you defined parameters in curly braces (e.g., {{PARAM}}) '
                         f'in your submit script/s. To avoid error, replace them with double curly braces (e.g., '
                         f'{{{{PARAM}}}} instead of {{PARAM}}.\nIdentified the following submit scripts:\n'
                         f'{submit_scripts_for_printing}')
            raise

        with open(os.path.join(self.local_path, submit_filenames[servers[self.server]['cluster_soft']]), 'w') as f:
            f.write(submit_script)

    def set_file_paths(self):
        """
        Set local and remote job file paths.
        """
        folder_name = 'TS_guesses' if self.reactions is not None else 'TSs' if self.species[0].is_ts else 'Species'
        if self.run_multi_species == False:
            self.local_path = os.path.join(self.project_directory, 'calcs', folder_name, self.species_label, self.job_name)
        else:
            self.local_path = os.path.join(self.project_directory, 'calcs', folder_name, self.species[0].multi_species, self.job_name)
        self.local_path_to_output_file = os.path.join(self.local_path, settings['output_filenames'][self.job_adapter]) \
            if self.job_adapter in settings['output_filenames'] else 'output.out'
        self.local_path_to_orbitals_file = os.path.join(self.local_path, 'orbitals.fchk')
        self.local_path_to_check_file = os.path.join(self.local_path, 'check.chk')
        self.local_path_to_hess_file = os.path.join(self.local_path, 'input.hess')
        self.local_path_to_xyz = None

        if not os.path.isdir(self.local_path):
            os.makedirs(self.local_path)

        if self.server is not None:
            # Parentheses don't play well in folder names:
            species_name_remote = self.species_label if isinstance(self.species_label, str) else self.species[0].multi_species
            species_name_remote = species_name_remote.replace('(', '_').replace(')', '_')
            path = servers[self.server].get('path', '').lower()
            path = os.path.join(path, servers[self.server]['un']) if path else ''
            self.remote_path = os.path.join(path, 'runs', 'ARC_Projects', self.project,
                                            species_name_remote, self.job_name)

        self.set_additional_file_paths()

    def upload_files(self):
        """
        Upload the relevant files for the job.
        """
        if not self.testing:
            if self.execution_type != 'incore' and self.server != 'local':
                # If the job execution type is incore, then no need to upload any files.
                # Also, even if the job is submitted to the que, no need to upload files if the server is local.
                with SSHClient(self.server) as ssh:
                    for up_file in self.files_to_upload:
                        logger.debug(f"Uploading {up_file['file_name']} source {up_file['source']} to {self.server}")
                        if up_file['source'] == 'path':
                            ssh.upload_file(remote_file_path=up_file['remote'], local_file_path=up_file['local'])
                        elif up_file['source'] == 'input_files':
                            ssh.upload_file(remote_file_path=up_file['remote'], file_string=up_file['local'])
                        else:
                            raise ValueError(f"Unclear file source for {up_file['file_name']}. Should either be 'path' or "
                                             f"'input_files', got: {up_file['source']}")
                        if up_file['make_x']:
                            ssh.change_mode(mode='+x', file_name=up_file['file_name'], remote_path=self.remote_path)
            else:
                # running locally, just copy the check file, if exists, to the job folder
                for up_file in self.files_to_upload:
                    if up_file['file_name'] == 'check.chk':
                        try:
                            shutil.copyfile(src=up_file['local'], dst=os.path.join(self.local_path, 'check.chk'))
                        except shutil.SameFileError:
                            pass
            self.initial_time = datetime.datetime.now()

    def download_files(self):
        """
        Download the relevant files.
        """
        if not self.testing:
            if self.execution_type != 'incore' and self.server != 'local':
                # If the job execution type is incore, then no need to download any files.
                # Also, even if the job is submitted to the que, no need to download files if the server is local.
                with SSHClient(self.server) as ssh:
                    for dl_file in self.files_to_download:
                        ssh.download_file(remote_file_path=dl_file['remote'], local_file_path=dl_file['local'])
                    self.set_initial_and_final_times(ssh=ssh)
            elif self.server == 'local':
                self.set_initial_and_final_times()
        self.final_time = self.final_time or datetime.datetime.now()

    def set_initial_and_final_times(self, ssh: Optional[SSHClient] = None):
        """
        Set the end time of the job.

        Args:
            ssh (SSHClient, optional): The SSHClient object instance.
        """
        # filenames_dict = input_filenames if initial else output_filenames
        # touched_file_name = 'initial_time' if initial else 'final_time'
        initial_time, final_time = None, None
        if ssh is not None:
            initial_time = ssh.get_last_modified_time(remote_file_path_1=os.path.join(self.remote_path,
                                                                                      'initial_time'))
            for dl_file in self.files_to_download:
                if dl_file['file_name'] == output_filenames[self.job_adapter]:
                    final_time = ssh.get_last_modified_time(
                        remote_file_path_1=os.path.join(self.remote_path, 'final_time'),
                        remote_file_path_2=dl_file['remote'],
                    )
                    break
        elif self.server == 'local':
            initial_time = get_last_modified_time(file_path_1=os.path.join(self.local_path, 'initial_time'))
            final_time = get_last_modified_time(
                file_path_1=os.path.join(self.local_path, 'final_time'),
                file_path_2=os.path.join(self.local_path, output_filenames[self.job_adapter]),
            )
        self.initial_time = initial_time or self.initial_time
        self.final_time = final_time or datetime.datetime.now()

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

    def _set_job_number(self):
        """
        Used as the entry number in the database, as well as the job name on the server.
        Also sets other attributes: job_name, job_server_id.
        This is not an abstract method and should not be overwritten.
        """
        # 1. Set job_number.
        local_arc_path_ = local_arc_path if os.path.isdir(local_arc_path) else ARC_PATH
        csv_path = os.path.join(local_arc_path_, 'initiated_jobs.csv')
        if os.path.isfile(csv_path):
            # Check that this is the updated version.
            with open(csv_path, 'r') as f:
                d_reader = csv.DictReader(f)
                headers = d_reader.fieldnames
            if 'comments' in headers:
                os.remove(csv_path)
        if not os.path.isfile(csv_path):
            with open(csv_path, 'w') as f:
                writer = csv.writer(f, dialect='excel')
                row = ['job_num', 'project', 'species_label', 'job_type', 'is_ts', 'charge', 'multiplicity',
                       'job_name', 'job_id', 'server', 'job_adapter', 'memory', 'level']
                writer.writerow(row)
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, dialect='excel')
            job_num = 0
            for _ in reader:
                job_num += 1
                if job_num == 100000:
                    job_num = 0
            self.job_num = job_num
        # 2. Set other related attributes job_name and job_server_name.
        self.job_server_name = self.job_server_name or 'a' + str(self.job_num)
        if self.conformer is not None and self.job_name is None:
            self.job_name = f'{self.job_type}_{self.conformer}'
        elif self.tsg is not None and (self.job_name is None or 'tsg_a' in self.job_name):
            if self.job_name is not None:
                logger.warning(f'Replacing job name {self.job_name} with tsg{self.conformer}')
            self.job_name = f'tsg{self.tsg}'
        elif self.job_name is None:
            self.job_name = f'{self.job_type}_{self.job_server_name}'
        # 3. Set job_id.
        self.job_id = self.job_num if self.execution_type == 'incore' and self.job_id is None else self.job_id

    def _write_initiated_job_to_csv_file(self):
        """
        Write an initiated ARC job into the initiated_jobs.csv file.
        """
        if not self.testing:
            local_arc_path_ = local_arc_path if os.path.isdir(local_arc_path) else ARC_PATH
            csv_path = os.path.join(local_arc_path_, 'initiated_jobs.csv')
            with open(csv_path, 'a') as f:
                writer = csv.writer(f, dialect='excel')
                row = [self.job_num, self.project, self.species_label, self.job_type, self.is_ts, self.charge,
                       self.multiplicity, self.job_name, self.job_id, self.server, self.job_adapter,
                       self.job_memory_gb, str(self.level)]
                writer.writerow(row)

    def write_completed_job_to_csv_file(self):
        """
        Write a completed ARCJob into the completed_jobs.csv file.
        """
        if not self.testing:
            if self.job_status[0] != 'done' or self.job_status[1]['status'] != 'done':
                self.determine_job_status()
            local_arc_path_ = local_arc_path if os.path.isdir(local_arc_path) else ARC_PATH
            csv_path = os.path.join(local_arc_path_, 'completed_jobs.csv')
            if os.path.isfile(csv_path):
                # check that this is the updated version
                with open(csv_path, 'r') as f:
                    d_reader = csv.DictReader(f)
                    headers = d_reader.fieldnames
                if 'comments' in headers:
                    os.remove(csv_path)
            if not os.path.isfile(csv_path):
                # Check file, make index file and write headers if file doesn't exist.
                with open(csv_path, 'w') as f:
                    writer = csv.writer(f, dialect='excel')
                    row = ['job_num', 'project', 'species_label', 'job_type', 'is_ts', 'charge', 'multiplicity',
                           'job_name', 'job_id', 'server', 'job_adapter', 'memory', 'level', 'initial_time',
                           'final_time', 'run_time', 'job_status_(server)', 'job_status_(ESS)',
                           'ESS troubleshooting methods used']
                    writer.writerow(row)
            csv_path = os.path.join(local_arc_path_, 'completed_jobs.csv')
            with open(csv_path, 'a') as f:
                writer = csv.writer(f, dialect='excel')
                job_type = self.job_type
                if self.fine:
                    job_type += ' (fine)'
                row = [self.job_num, self.project, self.species_label, job_type, self.is_ts, self.charge,
                       self.multiplicity, self.job_name, self.job_id, self.server, self.job_adapter,
                       self.job_memory_gb, str(self.level), self.initial_time, self.final_time,
                       self.run_time, self.job_status[0], self.job_status[1]['status'], self.ess_trsh_methods]
                writer.writerow(row)

    def set_cpu_and_mem(self):
        """
        Set cpu and memory based on ESS and cluster software.
        This is not an abstract method and should not be overwritten.
        """
        max_cpu = servers[self.server].get('cpus', None) if self.server is not None else 8  # Max cpus per node on server
        # Set to 8 if user did not specify cpu in settings and in ARC input file.
        job_cpu_cores = default_job_settings.get('job_cpu_cores', 8)
        if max_cpu is not None and job_cpu_cores > max_cpu:
            job_cpu_cores = max_cpu
        self.cpu_cores = self.cpu_cores or job_cpu_cores
        max_mem = servers[self.server].get('memory', None) if self.server is not None else 32.0  # Max memory per node in GB.
        job_max_server_node_memory_allocation = default_job_settings.get('job_max_server_node_memory_allocation', 0.95)
        if max_mem is not None and self.job_memory_gb > max_mem * job_max_server_node_memory_allocation:
            logger.warning(f'The memory for job {self.job_name} using {self.job_adapter} ({self.job_memory_gb} GB) '
                           f'exceeds {100 * job_max_server_node_memory_allocation}% of the the maximum node memory on '
                           f'{self.server}. Setting it to {job_max_server_node_memory_allocation * max_mem:.2f} GB.')
            self.job_memory_gb = job_max_server_node_memory_allocation * max_mem
            total_submit_script_memory = self.job_memory_gb * 1024 * 1.05  # MB
            self.job_status[1]['keywords'].append('max_total_job_memory')  # Useful info when troubleshooting.
        else:
            total_submit_script_memory = self.job_memory_gb * 1024 * 1.1  # MB
        # Determine amount of memory in submit script based on cluster job scheduling system.
        cluster_software = servers[self.server].get('cluster_soft').lower() if self.server is not None else None
        if cluster_software in ['oge', 'sge', 'htcondor']:
            # In SGE, "-l h_vmem=5000M" specifies the memory for all cores to be 5000 MB.
            self.submit_script_memory = math.ceil(total_submit_script_memory)  # in MB
        if cluster_software in ['pbs']:
            # In PBS, "#PBS -l select=1:ncpus=8:mem=12000000" specifies the memory for all cores to be 12 MB.
            self.submit_script_memory = math.ceil(total_submit_script_memory) * 1E6  # in Bytes
        elif cluster_software in ['slurm']:
            # In Slurm, "#SBATCH --mem-per-cpu=2000" specifies the memory **per cpu/thread** to be 2000 MB.
            self.submit_script_memory = math.ceil(total_submit_script_memory / self.cpu_cores)  # in MB
        self.set_input_file_memory()

    def as_dict(self) -> dict:
        """
        A helper function for dumping this object as a dictionary, used for saving in the restart file.
        """
        job_dict = dict()
        job_dict['job_adapter'] = self.job_adapter
        job_dict['project'] = self.project
        job_dict['project_directory'] = self.project_directory
        job_dict['job_type'] = self.job_type
        job_dict['args'] = self.args
        if self.bath_gas is not None:
            job_dict['bath_gas'] = self.bath_gas
        if self.checkfile is not None:
            job_dict['checkfile'] = self.checkfile
        if self.conformer is not None:
            job_dict['conformer'] = self.conformer
        if self.constraints is not None:
            job_dict['constraints'] = self.constraints
        job_dict['cpu_cores'] = self.cpu_cores
        if self.dihedrals is not None:
            job_dict['dihedrals'] = self.dihedrals
        job_dict['ess_settings'] = self.ess_settings
        if isinstance(self.ess_trsh_methods, list) and self.ess_trsh_methods:
            job_dict['ess_trsh_methods'] = self.ess_trsh_methods
        job_dict['execution_type'] = self.execution_type
        if self.fine:
            job_dict['fine'] = self.fine
        if self.initial_time is not None:
            job_dict['initial_time'] = str(self.initial_time)
        if self.irc_direction is not None:
            job_dict['irc_direction'] = self.irc_direction
        job_dict['job_id'] = self.job_id
        job_dict['job_memory_gb'] = float(self.job_memory_gb)
        job_dict['job_name'] = self.job_name
        job_dict['job_num'] = self.job_num
        job_dict['job_server_name'] = self.job_server_name
        job_dict['job_status'] = self.job_status
        job_dict['level'] = self.level.as_dict() if self.level is not None else None
        job_dict['max_job_time'] = self.max_job_time
        if self.reactions is not None:
            job_dict['reaction_indices'] = [reaction.index for reaction in self.reactions]
        if self.rotor_index is not None:
            job_dict['rotor_index'] = self.rotor_index
        if self.server is not None:
            job_dict['server'] = self.server
        if isinstance(self.server_nodes, dict) and self.server_nodes:
            job_dict['server_nodes'] = self.server_nodes
        if self.species is not None:
            job_dict['species_labels'] = [species.label for species in self.species]
        if self.torsions is not None:
            job_dict['torsions'] = self.torsions
        if self.tsg is not None:
            job_dict['tsg'] = self.tsg
        if self.xyz is not None:
            job_dict['xyz'] = self.xyz
        return job_dict

    def format_max_job_time(self, time_format: str) -> str:
        """
        Convert the max_job_time attribute into a format supported by the server cluster software.

        Args:
            time_format (str): Either ``'days'`` (e.g., 5-0:00:00) or ``'hours'`` (e.g., 120:00:00).

        Returns: str
            The formatted maximum job time string.
        """
        t_delta = datetime.timedelta(hours=self.max_job_time)
        if time_format == 'days':
            # e.g., 5-0:00:00
            t_max = f'{t_delta.days}-{datetime.timedelta(seconds=t_delta.seconds)}'
        elif time_format == 'hours':
            # e.g., 120:00:00
            h, s = divmod(t_delta.seconds, 3600)
            h += t_delta.days * 24
            t_max = f"{h}:{':'.join(str(datetime.timedelta(seconds=s)).split(':')[1:])}"
        else:
            raise JobError(f"Could not determine a format for maximal job time.\n Format is determined by "
                           f"{t_max_format}, but got {servers[self.server]['cluster_soft']} for {self.server}")
        return t_max

    def delete(self):
        """
        Delete a running Job.
        """
        logger.debug(f'Deleting job {self.job_name} for {self.species_label}')
        if self.server != 'local':
            logger.debug(f'deleting job on {self.server}...')
            with SSHClient(self.server) as ssh:
                ssh.delete_job(self.job_id)
        else:
            logger.debug('deleting job locally...')
            delete_job(job_id=self.job_id)

    def determine_job_status(self):
        """
        Determine the Job's status. Updates self.job_status.
        """
        cluster_soft = servers[self.server]['cluster_soft'].lower()
        if self.job_status[0] == 'errored':
            return
        self.job_status[0] = self._check_job_server_status() if self.execution_type != 'incore' else 'done'
        if self.job_status[0] == 'done':
            try:
                self._check_job_ess_status()  # Populates self.job_status[1], and downloads the output file.
            except IOError:
                logger.error(f'Got an IOError when trying to download output file for job {self.job_name}.')
            self._get_additional_job_info()
            if self.additional_job_info:
                logger.debug(f'Got the following information from the server:\n{self.additional_job_info}')
                for line in self.additional_job_info.splitlines():
                    # Example:
                    # slurmstepd: *** JOB 7752164 CANCELLED AT 2019-03-27T00:30:50 DUE TO TIME LIMIT on node096 ***
                    if 'cancelled' in line and 'due to time limit' in line:
                        logger.warning(f'Looks like the job was cancelled on {self.server} due to time limit. '
                                       f'Got: {line}')
                        new_max_job_time = self.max_job_time - 24 if self.max_job_time > 25 else 1
                        logger.warning(f'Setting max job time to {new_max_job_time} (was {self.max_job_time})')
                        self.max_job_time = new_max_job_time
                        self.job_status[1]['status'] = 'errored'
                        self.job_status[1]['keywords'] = ['ServerTimeLimit']
                        self.job_status[1]['error'] = 'Job cancelled by the server since it reached the maximal ' \
                                                      'time limit.'
                        self.job_status[1]['line'] = line
                        break
                    elif 'job killed' in line and 'exceeded limit' in line and cluster_soft == 'pbs':
                        # =>> PBS: job killed: walltime 10837 exceeded limit 10800
                        logger.warning(f'Looks like the job was killed on {self.server} due to time limit. '
                                       f'Got: {line}')
                        time_limit = int(line.split('limit')[1].split()[0])/3600
                        new_max_job_time = self.max_job_time - time_limit if self.max_job_time > time_limit else 1
                        logger.warning(f'Setting max job time to {new_max_job_time} (was {time_limit})')
                        self.max_job_time = new_max_job_time
                        self.job_status[1]['status'] = 'errored'
                        self.job_status[1]['keywords'] = ['ServerTimeLimit']
                        self.job_status[1]['error'] = 'Job killed by the server since it reached the maximal ' \
                                                      'time limit.'
                        self.job_status[1]['line'] = line
                        break
        elif self.job_status[0] == 'running':
            self.job_status[1]['status'] = 'running'

    def _get_additional_job_info(self):
        """
        Download the additional information of stdout and stderr from the server.
        stdout and stderr are named out.txt and err.txt respectively.
        Submission script in submit.py should contain the -o and -e flags.
        """
        content = ''
        cluster_soft = servers[self.server]['cluster_soft'].lower()
        if cluster_soft in ['oge', 'sge', 'slurm', 'pbs', 'htcondor']:
            local_file_path_1 = os.path.join(self.local_path, 'out.txt')
            local_file_path_2 = os.path.join(self.local_path, 'err.txt')
            local_file_path_3 = os.path.join(self.local_path, 'job.log')
            if self.server != 'local' and self.remote_path is not None and not self.testing:
                remote_file_path_1 = os.path.join(self.remote_path, 'out.txt')
                remote_file_path_2 = os.path.join(self.remote_path, 'err.txt')
                remote_file_path_3 = os.path.join(self.remote_path, 'job.log')
                with SSHClient(self.server) as ssh:
                    for local_file_path, remote_file_path in zip([local_file_path_1,
                                                                  local_file_path_2,
                                                                  local_file_path_3],
                                                                 [remote_file_path_1,
                                                                  remote_file_path_2,
                                                                  remote_file_path_3]):
                        try:
                            ssh.download_file(remote_file_path=remote_file_path,
                                              local_file_path=local_file_path)
                        except (TypeError, IOError) as e:
                            logger.warning(f'Got the following error when trying to download {remote_file_path} for '
                                           f'{self.job_name}. Please check that the submission script contains -o and -e '
                                           f'flags with stdout and stderr of out.txt and err.txt, respectively '
                                           f'(e.g., "#SBATCH -o out.txt"). Error message:')
                            logger.warning(e)
            for local_file_path in [local_file_path_1, local_file_path_2, local_file_path_3]:
                if os.path.isfile(local_file_path):
                    with open(local_file_path, 'r') as f:
                        lines = f.readlines()
                    content += ''.join([line for line in lines])
                    content += '\n'
        else:
            raise ValueError(f'Unrecognized cluster software: {cluster_soft}')
        if content:
            self.additional_job_info = content.lower()

    def _check_job_server_status(self) -> str:
        """
        Possible statuses: ``initializing``, ``running``, ``errored on node xx``, ``done``.
        """
        if self.server != 'local' and not self.testing:
            with SSHClient(self.server) as ssh:
                return ssh.check_job_status(self.job_id)
        else:
            return check_job_status(self.job_id)

    def _check_job_ess_status(self):
        """
        Check the status of the job ran by the electronic structure software (ESS).
        Possible statuses: ``initializing``, ``running``, ``errored: {error type / message}``, ``unconverged``, ``done``.

        Raises:
            IOError: If the output file and any additional server information cannot be found.
        """
        if self.server != 'local' and self.execution_type != 'incore':
            if os.path.exists(self.local_path_to_output_file):
                os.remove(self.local_path_to_output_file)
            if os.path.exists(self.local_path_to_orbitals_file):
                os.remove(self.local_path_to_orbitals_file)
            if os.path.exists(self.local_path_to_check_file):
                os.remove(self.local_path_to_check_file)
            self.download_files()  # Also sets initial and final times.
        else:
            # If running locally (local queue or incore),
            # just rename the output file to "output.out" for consistency between software.
            if self.final_time is None or self.initial_time is None:
                if self.server == 'local':
                    self.set_initial_and_final_times()
        self.rename_output_file()
        xyz_path = os.path.join(self.local_path, 'scr', 'optim.xyz')
        if os.path.isfile(xyz_path):
            self.local_path_to_xyz = xyz_path
        self.determine_run_time()
        if os.path.isfile(self.local_path_to_output_file):
            status, keywords, error, line = determine_ess_status(output_path=self.local_path_to_output_file,
                                                                 species_label=self.species_label,
                                                                 job_type=self.job_type,
                                                                 job_log=self.additional_job_info,
                                                                 software=self.job_adapter,
                                                                 )
            if status != 'done' and self.final_time is not None \
                    and datetime.datetime.now() - self.final_time < datetime.timedelta(seconds=30):
                # The head node might be busy, allow it some time to process, and check the status again.
                time.sleep(30)
                status, keywords, error, line = determine_ess_status(output_path=self.local_path_to_output_file,
                                                                     species_label=self.species_label,
                                                                     job_type=self.job_type,
                                                                     job_log=self.additional_job_info,
                                                                     software=self.job_adapter,
                                                                     )
        else:
            status, keywords, error, line = '', '', '', ''
        self.job_status[1]['status'] = status
        self.job_status[1]['keywords'] = keywords
        self.job_status[1]['error'] = error
        self.job_status[1]['line'] = line.rstrip()
        if 'Syntax' in self.job_status[1]['keywords']:
            logger.error(error)
            raise JobError(f'Got a syntax error in {self.job_adapter}.\n'
                           f'Check the level of theory ({self.level}), its compatibility, and any other keywords.')

    def rename_output_file(self):
        """
        A helper function for renaming the job output file as 'output.out'.
        The renaming should happen automatically, this method functions to troubleshoot
        cases where renaming wasn't successful the first time.
        """
        if not os.path.isfile(self.local_path_to_output_file) and not self.local_path_to_output_file.endswith('.yml'):
            rename_output(local_file_path=self.local_path_to_output_file, software=self.job_adapter)

    def add_to_args(self,
                    val: str,
                    key1: str = 'keyword',
                    key2: str = 'general',
                    separator: Optional[str] = None,
                    check_val_exists: bool = True,
                    ):
        """
        Add arguments to self.args in a nested dictionary under self.args[key1][key2].

        Args:
            val (str): The value to add.
            key1 (str, optional): Either 'keyword' or 'block'.
            key2 (str, optional): Key2.
            separator (str, optional): A separator (e.g., ``' '``  or ``'\\n'``)
                                       to apply between existing values and new values.
            check_val_exists (bool, optional): Only append ``val`` if it doesn't exist in the dictionary.
        """
        separator = separator if separator is not None else '\n\n' if key1 == 'block' else ' '
        if key1 not in list(self.args.keys()):
            self.args[key1] = dict()
        val_exists = self.args[key1] and key2 in self.args[key1] and val in self.args[key1][key2]
        if not (check_val_exists and val_exists):
            separator = separator if key2 in list(self.args[key1].keys()) else ''
            self.args[key1][key2] = val if key2 not in self.args[key1] else self.args[key1][key2] + separator + val

    def _log_job_execution(self):
        """
        Log executing this job.
        """
        info = ''
        if self.fine:
            info += ' (fine opt)'
        if self.job_type == 'scan':
            constraints = list()
            for constraint_tuple in self.constraints:
                constraints.append(f'{constraint_type_dict[len(constraint_tuple[0])]} '
                                   f'{constraint_tuple[0]} {constraint_tuple[1]}:.2f')
            constraints = constraints[0] if len(constraints) == 1 else constraints
            info += f' constraints: {constraints})' if constraints else ''
        local, server, job_server_name = '', '', ''
        if self.execution_type != 'incore':
            if self.server == 'local':
                local = 'local '
            else:
                server = f' on {self.server}'
        if 'conf_opt' in self.job_name or 'tsg' in self.job_name:
            job_server_name = f' ({self.job_server_name})'
        execution_type = {'incore': 'incore job', 'queue': 'queue job', 'pipe': 'job array (pipe)'}[self.execution_type]
        pivots = f' for pivots {[[tor[1] + 1, tor[2] + 1] for tor in self.torsions]}' if self.torsions is not None else ''
        dihedrals = f' for dihedrals {self.dihedrals}' if self.dihedrals is not None else ''
        tsconf = f' (conformer {self.species[0].chosen_ts})' if self.species is not None \
            and len(self.species) == 1 and self.species[0].is_ts and self.species[0].chosen_ts is not None \
            and 'opt' in self.job_type else ''
        logger.info(f'Running {local}{execution_type}{server} {self.job_name}{job_server_name}{pivots}{dihedrals} '
                    f'using {self.job_adapter} for {self.species_label}{info}{tsconf}')

    def get_file_property_dictionary(self,
                                     file_name: str,
                                     local: str = '',
                                     remote: str = '',
                                     source: str = 'path',
                                     make_x: bool = False,
                                     ) -> dict:
        """
        Get a dictionary that represents a file to be uploaded or downloaded to/from a server via SSH.

        Args:
            file_name (str): The file name.
            local (str, optional): The full local path.
            remote (str, optional): The full remote path.
            source (str, optional): Either ``'path'`` to treat the ``'local'`` attribute as a file path,
                                    or ``'input_files'`` to take the respective entry from inputs.py.
            make_x (bool, optional): Whether to make the file executable, default: ``False``.

        Returns:
            dict: A file representation.
        """
        if not file_name:
            raise ValueError('file_name cannot be empty')
        if source not in ['path', 'input_files']:
            raise ValueError(f'The source argument must be either "path" or "input_files", got {source}.')
        local = local or os.path.join(self.local_path, file_name)
        remote = remote or os.path.join(self.remote_path, file_name) if self.remote_path is not None else None
        return {'file_name': file_name,
                'local': local,
                'remote': remote,
                'source': source,
                'make_x': make_x,
                }

    def generate_scan_points(self,
                             species: 'ARCSpecies',
                             cont_only: bool = False,
                             ) -> List[DataPoint]:
        """
        Generate all coordinates in advance for "brute force" (non-continuous) directed scans,
        or the *next* coordinates for a continuous scan.

        Directed scan types could be one of the following: ``'brute_force_sp'``, ``'brute_force_opt'``,
        ``'brute_force_sp_diagonal'``, ``'brute_force_opt_diagonal'``, ``'cont_opt'``, or ``'cont_opt_diagonal'``.
        The differentiation between ``'sp'`` and ``'opt'`` is done in at the Job level.

        Args:
            species (ARCSpecies): The species to consider.
            cont_only (bool, optional): Whether to only return the next point in continuous scans.

        Raises:
            ValueError: If the species directed scan type has an unexpected value.

        Returns: List[DataPoint]
            Entries are DataPoint instances.
        """
        data_list = list()

        if divmod(360, self.scan_res)[1]:
            raise ValueError(f'Got an illegal scan resolution of {self.scan_res}.')

        for rotor_dict in species.rotors_dict.values():
            directed_scan_type = rotor_dict['directed_scan_type']
            if cont_only and 'cont' not in directed_scan_type:
                # Visiting this method again for a cont scan should not re-trigger all other scans.
                continue

            torsions = rotor_dict['torsion']
            if isinstance(torsions[0], int):
                torsions = [torsions]
            xyz = species.get_xyz(generate=True)

            if not ('cont' in directed_scan_type or 'brute' in directed_scan_type or 'ess' in directed_scan_type):
                raise ValueError(f'directed_scan_type must be either continuous or brute force, got: {directed_scan_type}')

            if directed_scan_type == 'ess' and not rotor_dict['scan_path'] and rotor_dict['success'] is None:
                # Allow the ESS to control the scan.
                data_list.append(DataPoint(job_types=['scan'],
                                           label=species.label,
                                           level=self.level,
                                           xyz_1=species.get_xyz(generate=True),
                                           args=self.args,
                                           charge=species.charge,
                                           constraints=self.constraints,
                                           cpu_cores=self.cpu_cores,
                                           multiplicity=species.multiplicity,
                                           torsions=torsions,
                                           ))

            elif 'brute' in directed_scan_type:
                # Spawn jobs all at once.
                dihedrals = dict()
                for torsion in torsions:
                    original_dihedral = calculate_dihedral_angle(coords=xyz['coords'], torsion=torsion, index=0)
                    dihedrals[tuple(torsion)] = [round(original_dihedral + i * self.scan_res
                                                       if original_dihedral + i * self.scan_res <= 180.0
                                                       else original_dihedral + i * self.scan_res - 360.0, 2)
                                                 for i in range(int(360 / self.scan_res) + 1)]
                modified_xyz = xyz.copy()
                if 'diagonal' not in directed_scan_type:
                    # Increment dihedrals one by one (results in an ND scan).
                    all_dihedral_combinations = list(itertools.product(*[dihedrals[tuple(torsion)] for torsion in torsions]))
                    for dihedral_tuple in all_dihedral_combinations:
                        for torsion, dihedral in zip(torsions, dihedral_tuple):
                            species.set_dihedral(scan=torsions_to_scans(torsion),
                                                 deg_abs=dihedral,
                                                 count=False,
                                                 xyz=modified_xyz)
                            modified_xyz = species.initial_xyz
                        rotor_dict['number_of_running_jobs'] += 1
                        data_list.append(DataPoint(job_types=['opt'] if 'opt' in directed_scan_type else ['sp'],
                                                   label=species.label,
                                                   level=self.level,
                                                   xyz_1=modified_xyz.copy(),
                                                   args=self.args,
                                                   charge=species.charge,
                                                   constraints=self.constraints,
                                                   cpu_cores=self.cpu_cores,
                                                   multiplicity=species.multiplicity,
                                                   ))
                else:
                    # Increment all dihedrals at once (results in a 1D scan along simultaneously-changing dimensions).
                    for i in range(len(dihedrals[tuple(torsions[0])])):
                        for torsion in torsions:
                            dihedral = dihedrals[tuple(torsion)][i]
                            species.set_dihedral(scan=torsions_to_scans(torsion),
                                                 deg_abs=dihedral,
                                                 count=False,
                                                 xyz=modified_xyz)
                            modified_xyz = species.initial_xyz
                        directed_dihedrals = [dihedrals[tuple(torsion)][i] for torsion in torsions]
                        rotor_dict['number_of_running_jobs'] += 1
                        data_list.append(DataPoint(job_types=['opt'] if 'opt' in directed_scan_type else ['sp'],
                                                   label=species.label,
                                                   level=self.level,
                                                   xyz_1=modified_xyz.copy(),
                                                   args=self.args,
                                                   charge=species.charge,
                                                   constraints=self.constraints,
                                                   cpu_cores=self.cpu_cores,
                                                   dihedrals=directed_dihedrals,
                                                   multiplicity=species.multiplicity,
                                                   ))

            elif 'cont' in directed_scan_type:
                # Set up the next DataPoint only.
                if not len(rotor_dict['cont_indices']):
                    rotor_dict['cont_indices'] = [0] * len(torsions)
                if not len(rotor_dict['original_dihedrals']):
                    rotor_dict['original_dihedrals'] = \
                        [f'{calculate_dihedral_angle(coords=xyz["coords"], torsion=scan, index=1):.2f}'
                         for scan in rotor_dict['scan']]  # Store the dihedrals as strings for the YAML restart file.
                torsions = rotor_dict['torsion']
                max_num = 360 / self.scan_res + 1  # Dihedral angles per scan
                original_dihedrals = list()
                for dihedral in rotor_dict['original_dihedrals']:
                    f_dihedral = float(dihedral)
                    original_dihedrals.append(f_dihedral if f_dihedral < 180.0 else f_dihedral - 360.0)
                if not any(rotor_dict['cont_indices']):
                    # This is the first call to the cont_opt directed rotor, spawn the first job w/o changing dihedrals.
                    data_list.append(DataPoint(job_types=['opt'],
                                               label=species.label,
                                               level=self.level,
                                               xyz_1=species.final_xyz,
                                               args=self.args,
                                               charge=species.charge,
                                               constraints=self.constraints,
                                               cpu_cores=self.cpu_cores,
                                               dihedrals=original_dihedrals,
                                               multiplicity=species.multiplicity,
                                               ))
                    rotor_dict['cont_indices'][0] += 1
                    continue
                else:
                    # This is NOT the first call for this cont_opt directed rotor.
                    # Check whether this rotor is done.
                    if rotor_dict['cont_indices'][-1] == max_num - 1:  # 0-indexed
                        # No more counters to increment, all done!
                        logger.info(f"Completed all jobs for the continuous directed rotor scan for species "
                                    f"{species.label} between pivots {rotor_dict['pivots']}")
                        continue

                modified_xyz = xyz.copy()
                dihedrals = list()
                for index, (original_dihedral, torsion_) in enumerate(zip(original_dihedrals, torsions)):
                    dihedral = original_dihedral + rotor_dict['cont_indices'][index] * self.scan_res
                    # Change the original dihedral so we won't end up with two calcs for 180.0, but none for -180.0
                    # (it only matters for plotting, the geometry is of course the same).
                    dihedral = dihedral if dihedral <= 180.0 else dihedral - 360.0
                    dihedrals.append(dihedral)
                    # Only change the dihedrals in the xyz if this torsion corresponds to the current index,
                    # or if this is a cont diagonal scan.
                    # Species.set_dihedral() uses .final_xyz or the given xyz to modify the .initial_xyz
                    # attribute to the desired dihedral.
                    species.set_dihedral(scan=torsions_to_scans(torsion_),
                                         deg_abs=dihedral,
                                         count=False,
                                         xyz=modified_xyz)
                    modified_xyz = species.initial_xyz
                data_list.append(DataPoint(job_types=['opt'],
                                           label=species.label,
                                           level=self.level,
                                           xyz_1=modified_xyz,
                                           args=self.args,
                                           charge=species.charge,
                                           constraints=self.constraints,
                                           cpu_cores=self.cpu_cores,
                                           dihedrals=dihedrals,
                                           multiplicity=species.multiplicity,
                                           ))

                if 'diagonal' in directed_scan_type:
                    # Increment ALL counters for a diagonal scan.
                    rotor_dict['cont_indices'] = [rotor_dict['cont_indices'][0] + 1] * len(torsions)
                else:
                    # Increment the counter sequentially for a non-diagonal scan.
                    for index in range(len(torsions)):
                        if rotor_dict['cont_indices'][index] < max_num - 1:
                            rotor_dict['cont_indices'][index] += 1
                            break
                        elif rotor_dict['cont_indices'][index] == max_num - 1 and index < len(torsions) - 1:
                            rotor_dict['cont_indices'][index] = 0

        return data_list

    def troubleshoot_server(self):
        """
        Troubleshoot server errors.
        """
        node, run_job = trsh_job_on_server(server=self.server,
                                           job_name=self.job_name,
                                           job_id=self.job_id,
                                           job_server_status=self.job_status[0],
                                           remote_path=self.remote_path,
                                           server_nodes=self.server_nodes,
                                           )
        if node is not None:
            self.server_nodes.append(node)
        if run_job:
            # resubmit job
            self.execute()

    def troubleshoot_queue(self) -> bool:
        """Troubleshoot queue errors.

        Returns:
            Boolean: Whether to run the job again.
        """
        queues, run_job = trsh_job_queue(job_name=self.job_name,
                                         server=self.server,
                                         max_time=24,
                                         attempted_queues = self.attempted_queues,
                                        )
        
        if queues is not None:
            # We use self.max_job_time to determine which queues to troubleshoot.
            #filtered_queues = {queue_name: walltime for queue_name, walltime in queues.items() if convert_to_hours(walltime) >= self.max_job_time}
            
            sorted_queues = sorted(queues.items(), key=lambda x: convert_to_hours(x[1]), reverse=False)

            self.queue = sorted_queues[0][0]
            if self.queue not in self.attempted_queues:
                self.attempted_queues.append(self.queue)
        return run_job

    def save_output_file(self,
                         key: Optional[str] = None,
                         val: Optional[Union[float, dict, np.ndarray]] = None,
                         content_dict: Optional[dict] = None,
                         ):
        """
        Save the output of a job to the YAML output file.

        Args:
            key (str, optional): The key for the YAML output file.
            val (Union[float, dict, np.ndarray], optional): The value to be stored.
            content_dict (dict, optional): A dictionary to store.

        """
        yml_out_path = os.path.join(self.local_path, 'output.yml')
        content = read_yaml_file(yml_out_path) if os.path.isfile(yml_out_path) else dict()
        if content_dict is not None:
            content.update(content_dict)
        if key is not None:
            content[key] = val
        save_yaml_file(path=yml_out_path, content=content)
