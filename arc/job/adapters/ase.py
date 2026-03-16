"""
An adapter for executing ASE (Atomic Simulation Environment) jobs
"""

import datetime
import os
import subprocess
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from arc.common import get_logger, read_yaml_file, save_yaml_file
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter
from arc.job.factory import register_job_adapter
from arc.imports import settings
from arc.settings.settings import ARC_PYTHON, find_executable

if TYPE_CHECKING:
    from arc.level import Level
    from arc.species.species import ARCSpecies
    from arc.reaction import ARCReaction

logger = get_logger()

# Default mapping if not yet fully defined in settings.py
DEFAULT_ASE_ENV = {
    'torchani': 'TANI_PYTHON',
    'xtb': 'XTB_PYTHON',
}

class ASEAdapter(JobAdapter):
    """
    A generic adapter for ASE (Atomic Simulation Environment) jobs.
    Supports multiple calculators and environments.
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
        
        self.job_adapter = 'ase'
        self.execution_type = execution_type or 'incore'
        self.incore_capacity = 100
        
        self.sp = None
        self.opt_xyz = None
        self.freqs = None

        self.args = args or dict()
        self.python_executable = self.get_python_executable()
        self.script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'ase_script.py')

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

    def get_python_executable(self) -> str:
        """
        Identify the correct Python executable based on the calculator.
        """
        calc = self.args.get('keyword', {}).get('calculator', '').lower()
        env_mapping = settings.get('ASE_CALCULATORS_ENV', DEFAULT_ASE_ENV)
        env_var_name = env_mapping.get(calc)
        
        if env_var_name and env_var_name in settings:
            exe = settings[env_var_name]
            if exe:
                return exe
        
        # Fallback to calculator-specific env if it exists
        found_exe = find_executable(f'{calc}_env')
        if found_exe:
            return found_exe
            
        return ARC_PYTHON or 'python'

    def write_input_file(self) -> None:
        """
        Write the input file for ase_script.py.
        """
        input_dict = {
            'job_type': self.job_type,
            'xyz': self.xyz,
            'charge': self.charge,
            'multiplicity': self.multiplicity,
            'constraints': self.constraints,
            'settings': self.args.get('keyword', {}),
        }
        save_yaml_file(os.path.join(self.local_path, 'input.yml'), input_dict)

    def execute_incore(self) -> None:
        """
        Execute the job incore.
        """
        self.write_input_file()
        cmd = [self.python_executable, self.script_path, '--yml_path', self.local_path]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.returncode != 0:
            logger.error(f"ASE job failed incore:\n{process.stderr}")
        self.parse_results()

    def execute_queue(self) -> None:
        """
        Execute a job to the server's queue.
        """
        self.write_input_file()
        self.write_submit_script()
        self.set_files()
        if self.server_adapter is not None:
            for file_dict in self.files_to_upload:
                self.server_adapter.upload_file(remote_path=file_dict['remote'],
                                               local_path=file_dict['local'])
            self.server_adapter.submit_job(self.remote_path)

    def set_files(self) -> None:
        """
        Set files to be uploaded and downloaded.
        """
        # 1. Upload
        if self.execution_type != 'incore':
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='submit.sh'))
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='input.yml'))
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='ase_script.py',
                                                                         local=self.script_path))
        # 2. Download
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
        Write the submission script.
        """
        remote_script_path = os.path.join(self.remote_path, 'ase_script.py')
        command = f"{self.python_executable} {remote_script_path} --yml_path {self.remote_path}"
        content = f"#!/bin/bash\n\n{command}\n"
        with open(os.path.join(self.local_path, 'submit.sh'), 'w') as f:
            f.write(content)

    def parse_results(self) -> None:
        """
        Parse the output.yml generated by ase_script.py.
        """
        out_path = os.path.join(self.local_path, 'output.yml')
        if os.path.isfile(out_path):
            results = read_yaml_file(out_path)
            self.electronic_energy = results.get('sp')
            self.xyz_out = results.get('opt_xyz') or results.get('xyz')
            self.frequencies = results.get('freqs')
            self.hessian = results.get('hessian')
            self.normal_modes = results.get('modes')
            self.reduced_masses = results.get('reduced_masses')
            self.force_constants = results.get('force_constants')
            if 'error' in results:
                logger.error(f"ASE job error: {results['error']}")

register_job_adapter('ase', ASEAdapter)
