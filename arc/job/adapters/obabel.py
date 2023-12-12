"""
An adapter for executing Openbabel jobs
http://openbabel.org/docs/current/
"""

import datetime
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import subprocess

from arc.common import ARC_PATH, get_logger, save_yaml_file, read_yaml_file
from arc.imports import settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter
from arc.job.factory import register_job_adapter
from arc.level import Level
from arc.settings.settings import ob_default_settings
from arc.species.converter import xyz_to_xyz_file_format, str_to_xyz

if TYPE_CHECKING:
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()

default_job_settings, global_ess_settings, input_filenames, output_filenames, servers, submit_filenames, OB_PYTHON = \
    settings['default_job_settings'], settings['global_ess_settings'], settings['input_filenames'], \
    settings['output_filenames'], settings['servers'], settings['submit_filenames'], settings['OB_PYTHON']

OB_SCRIPT_PATH = os.path.join(ARC_PATH, 'arc', 'job', 'adapters', 'scripts', 'ob_script.py')

class OpenbabelAdapter(JobAdapter):
    """
    A class for executing Openbabel jobs.

    Args:
        project (str): The project's name. Used for setting the remote path.
        project_directory (str): The path to the local project directory.
        job_type (list, str): The job's type, validated against ``JobTypeEnum``. If it's a list, pipe.py will be called.
        args (dict, optional): Methods (including troubleshooting) to be used in input files.
                               Keys are either 'keyword', 'block', or 'trsh', values are dictionaries with values
                               to be used either as keywords or as blocks in the respective software input file.
                               If 'trsh' is specified, an action might be taken instead of appending a keyword or a
                               block to the input file (e.g., change server or change scan resolution).
        bath_gas (str, optional): A bath gas. Currently only used in OneDMin to calculate L-J parameters.
        checkfile (str, optional): The path to a previous Gaussian checkfile to be used in the current job.
        conformer (int, optional): Conformer number if optimizing conformers.
        constraints (list, optional): A list of constraints to use during an optimization or scan.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
        dihedral_increment (float, optional): The degrees increment to use when scanning dihedrals of TS guesses.
        dihedrals (List[float], optional): The dihedral angels corresponding to self.torsions.
        directed_scan_type (str, optional): The type of the directed scan.
        ess_settings (dict, optional): A dictionary of available ESS and a corresponding server list.
        ess_trsh_methods (List[str], optional): A list of troubleshooting methods already tried out.
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
        max_job_time (float, optional): The maximal allowed job time on the server in hours (can be fractional).
        reactions (List[ARCReaction], optional): Entries are ARCReaction instances, used for TS search methods.
        rotor_index (int, optional): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        server (str): The server to run on.
        server_nodes (list, optional): The nodes this job was previously submitted to.
        species (List[ARCSpecies], optional): Entries are ARCSpecies instances.
                                              Either ``reactions`` or ``species`` must be given.
        testing (bool, optional): Whether the object is generated for testing purposes, ``True`` if it is.
        times_rerun (int, optional): Number of times this job was re-run with the same arguments (no trsh methods).
        torsions (List[List[int]], optional): The 0-indexed atom indices of the torsion(s).
        tsg (int, optional): TSGuess number if optimizing TS guesses.
        xyz (dict, optional): The 3D coordinates to use. If not give, species.get_xyz() will be used.
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
                 initial_time: Optional[Union['datetime.datetime', str]] = None,
                 irc_direction: Optional[str] = None,
                 job_id: Optional[int] = None,
                 job_memory_gb: float = 14.0,
                 job_name: Optional[str] = None,
                 job_num: Optional[int] = None,
                 job_server_name: Optional[str] = None,
                 job_status: Optional[List[Union[dict, str]]] = None,
                 level: Optional[Level] = None,
                 max_job_time: Optional[float] = None,
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

        self.incore_capacity = 100
        self.job_adapter = 'openbabel'
        self.execution_type = execution_type or 'incore'
        self.command = None
        self.url = 'https://github.com/openbabel/openbabel'

        self.sp = None
        self.opt_xyz = None

        if species is None:
            raise ValueError('Cannot execute Openbabel without an ARCSpecies object.')

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

    def write_input_file(self, settings) -> None:
        """
        Write the input file to execute the job on the server.
        """
        if self.level.method is not None:
            ob_default_settings["FF"] = self.level.method
        
        content = {"xyz" : xyz_to_xyz_file_format(self.opt_xyz or self.xyz),
                   "job_type" : self.job_type,
                   "constraints" : self.constraints}
        content.update(settings)
        save_yaml_file(path = os.path.join(self.local_path, "input.yml"), content=content)

    def check_settings(self, settings):
        if "FF" in settings.keys() and settings["FF"].lower() not in ["mmff94s", "mmff94", "gaff", "uff", "ghemical"]:
            return False
        if "opt_gradient_settings" in settings.keys():
            if "steps" in settings["opt_gradient_settings"].keys() and not isinstance(settings["opt_gradient_settings"]["steps"], int):
                return False
            if "econv" in settings["opt_gradient_settings"].keys() and not isinstance(settings["opt_gradient_settings"]["econv"], (float, int)):
                return False
        return True

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

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        self.local_path_to_output_file = os.path.join(self.local_path, 'output.yml')

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        pass

    def execute_incore(self):
        """
        Execute a job incore.
        """
        self._log_job_execution()

        if "freq" in self.job_type:
            raise NotImplementedError("Openbabel does not contain a frequency calculation option.\n"
                                      "Use another ESS from the available options.")
        if self.level.method is not None:
            ob_default_settings["FF"] = self.level.method

        if not self.check_settings(ob_default_settings):
            raise ValueError(f'Given wrong settings in input: {ob_default_settings}\n'
                             f'Fix input file or settings and try again.')

        self.write_input_file(ob_default_settings)
        commands = ['source ~/.bashrc',
                   f'{OB_PYTHON} {OB_SCRIPT_PATH} '
                   f'--yml_path {self.local_path}']
        command = '; '.join(commands)
        output = subprocess.run(command, shell=True, executable='/bin/bash')
        if output.returncode:
            logger.warning(f'Openbabel subprocess ran and did not '
                           f'give a successful return code for {self.job_id}.\n'
                           f'Got return code: {output.returncode}\n'
                           f'stdout: {output.stdout}\n'
                           f'stderr: {output.stderr}')
        try:
            output = read_yaml_file(path=os.path.join(self.local_path, "output.yml"))
        except FileNotFoundError:
            logger.error("Could not find output file")
            output = dict()
        self.sp, self.force, self.opt_xyz, self.freqs = output["sp"] if "sp" in output.keys() else None,\
                                                        output["force"] if "force" in output.keys() else None, \
                                                        str_to_xyz(output["opt_xyz"]) if "opt_xyz" in output.keys() else None, \
                                                        output["freqs"] if "freqs" in output.keys() else None

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        logger.warning('Queue execution is not yet supported for Openbabel in ARC, executing incore instead.')
        self.execute_incore()


register_job_adapter('openbabel', OpenbabelAdapter)
