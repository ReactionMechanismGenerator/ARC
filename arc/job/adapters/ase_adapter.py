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
from arc.imports import ase_submit, settings
from arc.settings.settings import ARC_PYTHON, UMA_LATEST_MODEL, find_executable

servers = settings['servers']
submit_filenames = settings['submit_filenames']
t_max_format = settings['t_max_format']

if TYPE_CHECKING:
    from arc.level import Level
    from arc.species.species import ARCSpecies
    from arc.reaction import ARCReaction

logger = get_logger()

# Default mapping if not yet fully defined in settings.py
DEFAULT_ASE_ENV = {
    'torchani': 'TANI_PYTHON',
    'xtb': 'XTB_PYTHON',
    'uma': 'UMA_PYTHON',
}

# Level methods that select the UMA calculator. 'uma' resolves to UMA_LATEST_MODEL; specific checkpoints named explicitly.
UMA_METHODS = ('uma', 'uma-s-1', 'uma-s-1p1', 'uma-s-1p2', 'uma-m-1p1')

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
        self.level = level  # also set by _initialize_adapter; needed early by get_python_executable
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

    def determine_calculator_name(self) -> str:
        """
        Determine the ASE calculator name, from ``args['keyword']['calculator']`` if given,
        otherwise inferred from the level method (e.g., a 'uma' method selects the UMA calculator).

        Returns:
            str: The lowercased calculator name (empty string if undetermined).
        """
        calc = (self.args or dict()).get('keyword', dict()).get('calculator', '')
        if not calc and self.level is not None and getattr(self.level, 'method', None) \
                and self.level.method.lower() in UMA_METHODS:
            calc = 'uma'
        return calc.lower()

    def determine_settings(self) -> dict:
        """
        Build the ``settings`` block passed to ase_script.py: the user's ``args['keyword']`` plus
        a resolved ``calculator`` and, for UMA, default ``model`` (the level method, with 'uma'
        resolving to the latest model), ``task``, and ``device``.

        Returns:
            dict: The resolved ASE run settings.
        """
        settings_dict = dict((self.args or dict()).get('keyword', dict()))
        calc = self.determine_calculator_name()
        if calc:
            settings_dict.setdefault('calculator', calc)
        if calc == 'uma':
            if 'model' not in settings_dict:
                method = self.level.method.lower() if self.level is not None and self.level.method else 'uma'
                settings_dict['model'] = UMA_LATEST_MODEL if method == 'uma' else method
            settings_dict.setdefault('task', 'omol')
            settings_dict.setdefault('device', 'cpu')
        return settings_dict

    def get_python_executable(self) -> str:
        """
        Identify the correct Python executable based on the calculator.
        """
        calc = self.determine_calculator_name()
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

    def determine_constraints(self) -> List[Tuple[List[int], float]]:
        """
        Determine the internal coordinate constraints to apply.

        A directed rotor scan is spawned by the Scheduler as one constrained optimization per
        dihedral point, but ``Scheduler.run_job()`` always passes ``constraints=None`` and hands the
        adapter ``torsions`` + ``dihedrals`` instead. Without translating those into a constraint the
        "scan" is an unconstrained optimization repeated at every point, and every point relaxes back
        to the same minimum, giving a flat V(phi). Torsions are 0-indexed; ARC constraints are
        1-indexed (as in the xTB and Gaussian adapters).

        Returns:
            List[Tuple[List[int], float]]: The constraints, as (1-indexed atom indices, value) pairs.
        """
        if self.constraints or self.job_type != 'directed_scan' or not self.torsions or not self.dihedrals:
            return self.constraints
        return [([index + 1 for index in torsion], dihedral)
                for torsion, dihedral in zip(self.torsions, self.dihedrals)]

    def write_input_file(self) -> None:
        """
        Write the input file for ase_script.py.
        """
        input_dict = {
            'job_type': self.job_type,
            'xyz': self.xyz,
            'charge': self.charge,
            'multiplicity': self.multiplicity,
            'is_ts': self.species[0].is_ts if self.species else False,
            'constraints': self.determine_constraints(),
            'irc_direction': self.irc_direction,
            'settings': self.determine_settings(),
        }
        save_yaml_file(os.path.join(self.local_path, 'input.yml'), input_dict)

    def warn_if_unreliable_uma_sp(self) -> bool:
        """
        Warn if this is a UMA single point on a species whose absolute UMA energy is unreliable
        (an isolated atom or triplet O2). UMA's geometries/frequencies are fine; only the absolute
        energy of these under-represented species is off, so a DFT single point is preferable.

        Reference: This is a known issue for general machine learning interatomic potentials (MLIPs)
        such as UMA (https://arxiv.org/abs/2405.20235), where atomic energy offsets do not accurately
        model isolated non-bonded atoms or highly specific spin states like triplet O2.
        """
        if self.job_type not in ['sp', 'conf_sp'] or self.determine_calculator_name() != 'uma':
            return False
        symbols = self.xyz['symbols'] if self.xyz is not None else tuple()
        is_atom = len(symbols) == 1
        is_triplet_o2 = len(symbols) == 2 and all(s == 'O' for s in symbols) and self.multiplicity == 3
        if is_atom or is_triplet_o2:
            label = self.species[0].label if self.species else 'species'
            logger.warning(f'Computing a UMA single point for {label} (an isolated atom or triplet O2). '
                           f'UMA absolute energies are unreliable for these under-represented species; '
                           f'consider using a DFT single point instead.')
            return True
        return False

    def execute_incore(self) -> None:
        """
        Execute the job incore.
        """
        self.warn_if_unreliable_uma_sp()
        self.write_input_file()
        cmd = [self.python_executable, self.script_path, '--yml_path', self.local_path]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.returncode != 0:
            logger.error(f"ASE job failed incore:\n{process.stderr}")
        self.parse_results()

    def execute_queue(self) -> None:
        """
        Execute a job to the server's queue.

        ``set_files()`` wrote the files and ``JobAdapter.execute()`` uploaded them, so all that is
        left here is the submission itself, through the same path every other adapter uses.
        """
        self.legacy_queue_execution()

    def set_files(self) -> None:
        """
        Set files to be uploaded and downloaded. Writes the files if needed.
        """
        # 1. Upload
        if self.execution_type != 'incore':
            # ``JobAdapter.execute()`` calls ``upload_files()`` *before* ``execute_queue()``, and
            # ``_initialize_adapter()`` calls this method while the job is being constructed, so a
            # queue job's files have to be written here - as the Gaussian, Orca and xTB adapters do
            # - or the upload raises "InputError: Cannot upload a non-existing file".
            # An incore job is not uploaded and writes its input in ``execute_incore()``.
            self.write_submit_script()
            self.files_to_upload.append(self.get_file_property_dictionary(file_name=self.determine_submit_filename()))
            self.write_input_file()
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

    def determine_submit_config(self) -> dict:
        """
        Determine the cluster submission knobs for this job, taken from the level's ``args['block']``.

        Recognized keys (all optional):

        - ``env_setup``: shell lines to run on the compute node before the ASE script, e.g.
          ``conda activate uma_env``. Note that ARC lowercases level args, so a case-sensitive
          module name must be sourced from a file on the server rather than written inline.
        - ``gpu_resource``: a scheduler GPU request, appended to the PBS ``select`` statement
          (e.g. ``ngpus=1``) or used as the Slurm ``--gres`` value (e.g. ``gpu:1``).
        - ``python``: the python executable **on the server**. ``self.python_executable`` is
          resolved against the ARC host's conda envs and generally does not exist on a remote server.
        - ``queue``: the queue to submit to, if not already set on the job or in the server settings.

        Returns:
            dict: The resolved submit configuration.
        """
        block = (self.args or dict()).get('block', dict()) or dict()
        default_queue, _ = next(iter(servers.get(self.server, dict()).get('queues', dict()).items()), (None, None))
        return {'queue': self.queue or block.get('queue') or default_queue,
                'env_setup': block.get('env_setup', ''),
                'gpu_resource': block.get('gpu_resource', ''),
                'python': block.get('python', ''),
                }

    def determine_submit_filename(self) -> str:
        """
        Return the filename ARC will submit for this job.

        A queue-executed PBS/Slurm job must be written under the scheduler-specific name that
        ``submit_job()`` invokes (``submit_filenames``, e.g. ``submit.sl`` for Slurm), or the
        submission fails because the file it names is not on disk. Everything else uses the plain
        ``submit.sh`` the bare script is written to.

        Returns:
            str: The submit-script filename.
        """
        cluster_soft = servers.get(self.server, dict()).get('cluster_soft', '') if self.server is not None else ''
        if self.execution_type != 'incore' and cluster_soft.lower() in ('pbs', 'slurm'):
            return submit_filenames[cluster_soft]
        return 'submit.sh'

    def get_queue_submit_script(self, command: str, config: dict, cluster_soft: str) -> str:
        """
        Compose a cluster submit script for a queue-executed ASE job.

        Formats the server-independent ``ase_submit`` template (in ``arc/settings/submit.py``,
        keyed by cluster software) with this job's resources and submit config. The thread-pool
        exports pin the numerical libraries (torch, NumPy) to the cores the scheduler granted, so a
        shared node is not oversubscribed.

        Args:
            command (str): The command running the ASE script on the compute node.
            config (dict): The output of ``determine_submit_config()``.
            cluster_soft (str): The lowercased cluster software name ('pbs' or 'slurm').

        Returns:
            str: The submit script content.
        """
        if cluster_soft not in ase_submit:
            raise NotImplementedError(f"No ASE submit template for cluster software '{cluster_soft}'. "
                                      f"Available templates: {list(ase_submit.keys())}")
        memory = int(self.submit_script_memory) if isinstance(self.submit_script_memory, (int, float)) \
            else self.submit_script_memory
        time_format = next((v for k, v in t_max_format.items() if k.lower() == cluster_soft), 'hours')
        pwd = self.local_path if self.server is None or str(self.server).lower() == 'local' else self.remote_path
        queue, gpu_resource = config['queue'], config['gpu_resource']
        format_kwargs = {'name': self.job_server_name, 'cpus': self.cpu_cores, 'memory': memory,
                         't_max': self.format_max_job_time(time_format=time_format), 'pwd': pwd,
                         'env_setup': config['env_setup'], 'command': command}
        if cluster_soft == 'pbs':
            format_kwargs['queue_directive'] = f'#PBS -q {queue}\n' if queue else ''
            format_kwargs['gpu_select'] = f':{gpu_resource}' if gpu_resource else ''
        else:
            format_kwargs['queue_directive'] = f'#SBATCH -p {queue}\n' if queue else ''
            format_kwargs['gpu_directive'] = f'#SBATCH --gres={gpu_resource}\n' if gpu_resource else ''
        return ase_submit[cluster_soft].format(**format_kwargs)

    def write_submit_script(self) -> None:
        """
        Write the submission script.

        An incore job only has to invoke the ASE script. A queue job additionally needs cluster
        scheduler directives, an environment setup preamble (``conda activate uma_env`` for UMA), a
        server-side python executable, and - for a GPU run - a GPU resource request; a bare
        ``#!/bin/bash`` script carries none of those and lands on the queue's defaults with a python
        path that only exists on the ARC host. See ``determine_submit_config()`` for the knobs.
        """
        config = self.determine_submit_config()
        cluster_soft = servers.get(self.server, dict()).get('cluster_soft', '').lower() \
            if self.server is not None else ''
        queue_job = self.execution_type != 'incore' and cluster_soft in ('pbs', 'slurm')
        if queue_job and not config['python']:
            logger.warning(f"Job {self.job_name} is submitted to {self.server}, but no server-side python "
                           f"was given in args['block']['python']; falling back to {self.python_executable}, "
                           f"which was resolved on this machine and may not exist there.")
        python_executable = config['python'] or self.python_executable
        if queue_job:
            # trsh_job_queue() skips queues already in attempted_queues; record the one we submit to
            # here, as JobAdapter.write_submit_script() does, so a failed submission moves on.
            if config['queue'] and config['queue'] not in self.attempted_queues:
                self.attempted_queues.append(config['queue'])
            # The script cd's into the job directory, so address the ASE script relative to it.
            command = f'{python_executable} "$JOB_DIR/ase_script.py" --yml_path "$JOB_DIR"'
            content = self.get_queue_submit_script(command=command, config=config, cluster_soft=cluster_soft)
        else:
            # A job with no server (and hence no remote path) runs out of its local directory.
            path = self.remote_path or self.local_path
            remote_script_path = os.path.join(path, 'ase_script.py')
            command = f"{python_executable} {remote_script_path} --yml_path {path}"
            content = f"#!/bin/bash\n\n{command}\n"
        with open(os.path.join(self.local_path, self.determine_submit_filename()), 'w') as f:
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
