"""
An adapter for executing PySCF jobs locally (a local in-core ESS, no PBS/queue round-trip).

PySCF runs DFT (sp, opt, freq, optfreq) on the workstation and hands off an ``output.yml`` in
ARC's internal format, which ARC's ``YAMLParser`` re-parses (the YAML-handoff pattern, mirroring
``ase_adapter.py``). The engine script runs inside the dedicated ``pyscf_env`` conda environment
via ``run_in_conda_env`` so ARC's activation variables do not leak into the child process.

Scope: sp + opt + freq (closed- and open-shell, RKS/UKS). IRC and scans/rotors are out of scope.
"""

import datetime
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from arc.common import ARC_PATH, get_logger, read_yaml_file, save_yaml_file
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter
from arc.job.env_run import run_in_conda_env
from arc.job.factory import register_job_adapter
from arc.settings.settings import PYSCF_PYTHON

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction
    from arc.species.species import ARCSpecies

logger = get_logger()

PYSCF_SCRIPT_PATH = os.path.join(ARC_PATH, 'arc', 'job', 'adapters', 'scripts', 'pyscf_script.py')


class PySCFAdapter(JobAdapter):
    """
    A class for executing PySCF jobs.

    Args:
        project (str): The project's name. Used for setting the remote path.
        project_directory (str): The path to the local project directory.
        job_type (list, str): The job's type, validated against ``JobTypeEnum``.
        args (dict, optional): Methods (e.g., troubleshooting) to be used in input files.
        bath_gas (str, optional): A bath gas.
        checkfile (str, optional): The path to a previous Gaussian checkfile to be used in the job.
        conformer (int, optional): Conformer number if optimizing conformers.
        constraints (list, optional): A list of constraints to use during an optimization or scan.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
        dihedral_increment (float, optional): The degrees increment to use when scanning dihedrals.
        dihedrals (list, optional): The dihedral angles of a directed scan job.
        directed_scan_type (str, optional): The type of the directed scan.
        ess_settings (dict, optional): A dictionary of available ESS and a corresponding server list.
        ess_trsh_methods (list, optional): A list of troubleshooting methods already tried out.
        execution_type (str, optional): The execution type, 'incore', 'queue', or 'pipe'.
        fine (bool, optional): Whether to use fine geometry optimization parameters. Default: ``False``.
        initial_time (datetime.datetime or str, optional): The time at which this job was initiated.
        irc_direction (str, optional): The direction of the IRC job (`forward` or `reverse`).
        job_id (int, optional): The job's ID determined by the server.
        job_memory_gb (int, optional): The total job allocated memory in GB (14 by default).
        job_name (str, optional): The job's name (e.g., 'opt_a103').
        job_num (int, optional): Used as the entry number in the database, as well as in ``job_name``.
        job_server_name (str, optional): Job's name on the server (e.g., 'a103').
        job_status (list, optional): The job's server and ESS statuses.
        level (Level, optional): The level of theory to use.
        max_job_time (float, optional): The maximal allowed job time on the server in hours.
        run_multi_species (bool, optional): Whether to run a job for multiple species in the same input file.
        reactions (list, optional): Entries are ARCReaction instances, used for TS search methods.
        rotor_index (int, optional): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        server (str, optional): The server to run on.
        server_nodes (list, optional): The nodes this job was previously submitted to.
        species (list, optional): Entries are ARCSpecies instances.
                                  Either ``reactions`` or ``species`` must be given.
        testing (bool, optional): Whether the object is generated for testing purposes, ``True`` if it is.
        times_rerun (int, optional): Number of times this job was re-run with the same arguments.
        torsions (list, optional): The 0-indexed atom indices of the torsion(s).
        tsg (int, optional): TSGuess number if optimizing TS guesses.
        xyz (dict, optional): The 3D coordinates to use in the job.
    """

    def __init__(self,
                 project: str,
                 project_directory: str,
                 job_type: Union[List[str], str],
                 args: Optional[dict] = None,
                 bath_gas: Optional[str] = None,
                 checkfile: Optional[str] = None,
                 conformer: Optional[int] = None,
                 constraints: Optional[List[Tuple[List[int], float]]] = None,
                 cpu_cores: Optional[str] = None,
                 dihedral_increment: Optional[float] = None,
                 dihedrals: Optional[List[float]] = None,
                 directed_scan_type: Optional[str] = None,
                 ess_settings: Optional[dict] = None,
                 ess_trsh_methods: Optional[List[str]] = None,
                 execution_type: Optional[str] = None,
                 fine: bool = False,
                 initial_time: Optional[Union[datetime.datetime, str]] = None,
                 irc_direction: Optional[str] = None,
                 job_id: Optional[int] = None,
                 job_memory_gb: float = 14.0,
                 job_name: Optional[str] = None,
                 job_num: Optional[int] = None,
                 job_server_name: Optional[str] = None,
                 job_status: Optional[List[Union[dict, str]]] = None,
                 level: Optional['Level'] = None,
                 max_job_time: Optional[float] = None,
                 run_multi_species: bool = False,
                 reactions: Optional[List['ARCReaction']] = None,
                 rotor_index: Optional[int] = None,
                 server: Optional[str] = None,
                 server_nodes: Optional[list] = None,
                 queue: Optional[str] = None,
                 attempted_queues: Optional[List[str]] = None,
                 species: Optional[List['ARCSpecies']] = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: Optional[List[List[int]]] = None,
                 tsg: Optional[int] = None,
                 xyz: Optional[dict] = None,
                 ):

        self.job_adapter = 'pyscf'
        self.execution_type = execution_type or 'incore'
        self.incore_capacity = 100
        self.command = None
        self.url = 'https://pyscf.org/'

        self.sp = None
        self.opt_xyz = None
        self.freqs = None

        self.script_path = PYSCF_SCRIPT_PATH

        _initialize_adapter(obj=self,
                            is_ts=False,
                            project=project,
                            project_directory=project_directory,
                            job_type=job_type,
                            args=args,
                            bath_gas=bath_gas,
                            checkfile=checkfile,
                            conformer=conformer,
                            constraints=constraints,
                            cpu_cores=cpu_cores,
                            dihedral_increment=dihedral_increment,
                            dihedrals=dihedrals,
                            directed_scan_type=directed_scan_type,
                            ess_settings=ess_settings,
                            ess_trsh_methods=ess_trsh_methods,
                            fine=fine,
                            initial_time=initial_time,
                            irc_direction=irc_direction,
                            job_id=job_id,
                            job_memory_gb=job_memory_gb,
                            job_name=job_name,
                            job_num=job_num,
                            job_server_name=job_server_name,
                            job_status=job_status,
                            level=level,
                            max_job_time=max_job_time,
                            run_multi_species=run_multi_species,
                            reactions=reactions,
                            rotor_index=rotor_index,
                            server=server,
                            server_nodes=server_nodes,
                            queue=queue,
                            attempted_queues=attempted_queues,
                            species=species,
                            testing=testing,
                            times_rerun=times_rerun,
                            torsions=torsions,
                            tsg=tsg,
                            xyz=xyz,
                            )

    def determine_settings(self) -> dict:
        """
        Build the ``settings`` block passed to pyscf_script.py: the level's method (the xc
        functional) and basis set, the job memory, the ``fine`` flag and cpu core count (which
        pins PySCF's thread count), plus any user ``args['keyword']`` knobs.

        Returns:
            dict: The resolved PySCF run settings.
        """
        settings_dict = dict((self.args or dict()).get('keyword', dict()))
        if self.level is not None:
            for key, level_value in (('method', self.level.method), ('basis', self.level.basis)):
                if key in settings_dict and settings_dict[key] != level_value:
                    logger.warning(f"PySCF job {self.job_name}: ignoring args['keyword']['{key}']="
                                   f"{settings_dict[key]!r}; using the job's level of theory "
                                   f"{key}={level_value!r} so the computed result matches the level "
                                   f"ARC records.")
                settings_dict[key] = level_value
        settings_dict.setdefault('memory_mb', int(self.job_memory_gb * 1024))
        settings_dict.setdefault('fine', self.fine)
        if self.cpu_cores:
            settings_dict.setdefault('cpu_cores', self.cpu_cores)
        return settings_dict

    def write_input_file(self) -> None:
        """
        Write the input file (input.yml) for pyscf_script.py.
        """
        input_dict = {
            'job_type': self.job_type,
            'xyz': self.xyz,
            'charge': self.charge,
            'multiplicity': self.multiplicity,
            'is_ts': self.species[0].is_ts if self.species else False,
            'constraints': self.constraints,
            'settings': self.determine_settings(),
        }
        save_yaml_file(os.path.join(self.local_path, 'input.yml'), input_dict)

    def execute_incore(self):
        """
        Execute a job incore: run pyscf_script.py inside pyscf_env, then parse output.yml.
        """
        self._log_job_execution()
        self.write_input_file()
        output = run_in_conda_env(PYSCF_PYTHON, self.script_path, '--yml_path', self.local_path)
        if output.returncode:
            logger.warning(f'PySCF subprocess for {self.job_name} returned a non-zero code.\n'
                           f'Got return code: {output.returncode}\n'
                           f'stdout: {output.stdout}\nstderr: {output.stderr}')
        self.parse_results()

    def execute_queue(self):
        """
        PySCF is a local in-core ESS; fall back to incore execution when queued.

        ``execution_type`` is updated to reflect what actually happened, so that downstream
        status handling (e.g., ``determine_job_status()``) does not query a queueing system
        for a job that never reached one.
        """
        logger.debug('Queue execution is not supported for PySCF in ARC; running incore instead.')
        self.execution_type = 'incore'
        self.execute_incore()

    def parse_results(self) -> None:
        """
        Read output.yml written by pyscf_script.py into the job-object attributes.

        Note: these attributes are dead-end writes; the authoritative contract is the on-disk
        output.yml, which ARC re-parses via ``YAMLParser``.
        """
        out_path = os.path.join(self.local_path, 'output.yml')
        if os.path.isfile(out_path):
            results = read_yaml_file(out_path)
            self.sp = results.get('sp')
            self.opt_xyz = results.get('opt_xyz') or results.get('xyz')
            self.freqs = results.get('freqs')
        else:
            logger.error(f'PySCF job {self.job_name} did not generate an output.yml file.')

    def set_files(self) -> None:
        """
        Set files to be uploaded and downloaded.
        """
        if self.execution_type != 'incore':
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='submit.sh'))
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='input.yml'))
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='pyscf_script.py',
                                                                          local=self.script_path))
        self.files_to_download.append(self.get_file_property_dictionary(file_name='output.yml'))

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        """
        pass

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        pass

    def write_submit_script(self) -> None:
        """
        Write the submission script (for the queue fallback path).
        """
        command = f"{PYSCF_PYTHON} {self.script_path} --yml_path {self.remote_path}"
        content = f"#!/bin/bash\n\n{command}\n"
        with open(os.path.join(self.local_path, 'submit.sh'), 'w') as f:
            f.write(content)


register_job_adapter('pyscf', PySCFAdapter)
