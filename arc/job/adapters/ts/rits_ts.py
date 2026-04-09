"""
An adapter for executing RitS (Right into the Saddle) TS-guess jobs.

RitS is a flow-matching ML model that generates 3D transition-state geometries
directly from atom-mapped reactant + product structures, without requiring an
initial guess. Unlike GCN (which is restricted to isomerizations), RitS can
handle bimolecular reactions and supports charged species, so it covers a
strictly larger reaction space.

Code source     : https://github.com/isayevlab/RitS
Paper           : 10.26434/chemrxiv.15001681/v1
Pretrained ckpt : https://doi.org/10.5281/zenodo.19474153

Implementation notes
--------------------
* The heavy ML stack (torch + torch-geometric + megalodon) lives in its own
  conda env (``rits_env``), so this adapter never imports it directly. It
  shells out to ``arc/job/adapters/scripts/rits_script.py`` via subprocess,
  which in turn invokes RitS's own ``scripts/sample_transition_state.py``.
* RitS requires the reactant and product XYZ files to have the *same atom
  count and the same atom ordering* (it aligns them by index). ARC's
  ``rxn.get_reactants_xyz`` / ``get_products_xyz`` already produce mapped
  outputs via ``rxn.atom_map``, so we can use them as-is.
* Multiple samples per reaction are produced in a single subprocess call
  (RitS's ``--n_samples`` flag), avoiding the per-sample model-load overhead
  that GCN incurs.
* If ``rits_env`` or the pretrained checkpoint is missing on the host, the
  adapter logs a warning and exits cleanly without raising — the rest of
  ARC's TS-search pipeline (heuristics, GCN, AutoTST, …) keeps running.
* ``incore_capacity = 1`` so the scheduler serializes RitS jobs and a single
  GPU is not asked to load multiple checkpoints in parallel.
"""

import datetime
import os
import subprocess
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from arc.common import ARC_PATH, get_logger, save_yaml_file, read_yaml_file
from arc.imports import settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter
from arc.job.factory import register_job_adapter
from arc.plotter import save_geo
from arc.species.converter import compare_confs, str_to_xyz, xyz_to_str
from arc.species.species import ARCSpecies, TSGuess, colliding_atoms

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction


RITS_PYTHON = settings.get('RITS_PYTHON')
RITS_REPO_PATH = settings.get('RITS_REPO_PATH')
RITS_CKPT_PATH = settings.get('RITS_CKPT_PATH')

RITS_SCRIPT_PATH = os.path.join(ARC_PATH, 'arc', 'job', 'adapters', 'scripts', 'rits_script.py')
DEFAULT_N_SAMPLES = 10
DEFAULT_BATCH_SIZE = 32

logger = get_logger()


class RitSAdapter(JobAdapter):
    """
    A class for executing RitS (Right into the Saddle) TS-guess jobs.

    Args:
        project (str): The project's name. Used for setting the remote path.
        project_directory (str): The path to the local project directory.
        job_type (list, str): The job's type, validated against ``JobTypeEnum``.
        args (dict, optional): Methods (including troubleshooting) to be used in
            input files. For RitS the only currently-honored entry is
            ``args['keyword']['n_samples']`` (int, default 10).
        bath_gas (str, optional): A bath gas. Currently only used in OneDMin.
        checkfile (str, optional): The path to a previous Gaussian checkfile.
        conformer (int, optional): Conformer number if optimizing conformers.
        constraints (list, optional): A list of constraints.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
        dihedral_increment (float, optional): Unused for RitS.
        dihedrals (List[float], optional): The dihedral angles corresponding to
            self.torsions.
        directed_scan_type (str, optional): The type of the directed scan.
        ess_settings (dict, optional): A dictionary of available ESS.
        ess_trsh_methods (List[str], optional): A list of troubleshooting methods.
        execution_type (str, optional): The execution type, 'incore', 'queue', or 'pipe'.
        fine (bool, optional): Whether to use fine geometry optimization parameters.
        initial_time (datetime.datetime or str, optional): The time at which this job was initiated.
        irc_direction (str, optional): The direction of the IRC job.
        job_id (int, optional): The job's ID determined by the server.
        job_memory_gb (int, optional): The total job allocated memory in GB.
        job_name (str, optional): The job's name.
        job_num (int, optional): Used as the entry number in the database.
        job_server_name (str, optional): Job's name on the server.
        job_status (list, optional): The job's server and ESS statuses.
        level (Level, optional): The level of theory to use.
        max_job_time (float, optional): The maximal allowed job time on the server in hours.
        run_multi_species (bool, optional): Whether to run a job for multiple species in the same input file.
        reactions (List[ARCReaction], optional): Entries are ARCReaction instances.
        rotor_index (int, optional): The 0-indexed rotor number.
        server (str): The server to run on.
        server_nodes (list, optional): The nodes this job was previously submitted to.
        species (List[ARCSpecies], optional): Entries are ARCSpecies instances.
        testing (bool, optional): Whether the object is generated for testing purposes.
        times_rerun (int, optional): Number of times this job was re-run.
        torsions (List[List[int]], optional): The 0-indexed atom indices of the torsion(s).
        tsg (int, optional): TSGuess number if optimizing TS guesses.
        xyz (dict, optional): The 3D coordinates to use.
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

        # Single in-flight job per scheduler tick — RitS holds an ML model in
        # GPU memory, parallelizing it across reactions would risk OOM.
        self.incore_capacity = 1
        self.job_adapter = 'rits'
        self.execution_type = execution_type or 'incore'
        self.command = 'sample_transition_state.py'
        self.url = 'https://github.com/isayevlab/RitS'

        if reactions is None:
            raise ValueError('Cannot execute RitS without ARCReaction object(s).')

        # Number of TS samples to draw per reaction. Honored from args['keyword']['n_samples']
        # so users can bump it via the standard ARC adapter-args path.
        self.n_samples = DEFAULT_N_SAMPLES
        if args and isinstance(args, dict):
            kw = args.get('keyword') or dict()
            if 'n_samples' in kw:
                try:
                    self.n_samples = int(kw['n_samples'])
                except (TypeError, ValueError):
                    logger.warning(
                        f"RitS adapter: could not parse args['keyword']['n_samples']="
                        f"{kw['n_samples']!r} as an int; falling back to "
                        f"DEFAULT_N_SAMPLES={DEFAULT_N_SAMPLES}."
                    )

        _initialize_adapter(obj=self,
                            is_ts=True,
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

    def write_input_file(self) -> None:
        """No standalone input file — see set_files() (writes input.yml)."""
        pass

    def set_files(self) -> None:
        """
        Set files to be uploaded and downloaded for queue execution.

        ``self.files_to_upload`` is a list of dictionaries, each with the keys
        ``'name'``, ``'source'``, ``'make_x'``, ``'local'``, and ``'remote'``.
        """
        # 1. Upload
        if self.execution_type != 'incore':
            self.write_submit_script()
            from arc.imports import settings as _s
            self.files_to_upload.append(self.get_file_property_dictionary(
                file_name=_s['submit_filenames'][_s['servers'][self.server]['cluster_soft']]))
        if os.path.isfile(self.yml_in_path):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='input.yml'))
        if os.path.isfile(self.reactant_xyz_path):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='reactant.xyz'))
        if os.path.isfile(self.product_xyz_path):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='product.xyz'))
        # 2. Download
        self.files_to_download.append(self.get_file_property_dictionary(file_name='output.yml'))
        self.files_to_download.append(self.get_file_property_dictionary(file_name='rits_ts.xyz'))

    def set_additional_file_paths(self) -> None:
        """Set the local file paths used by RitS at job time."""
        self.reactant_xyz_path = os.path.join(self.local_path, 'reactant.xyz')
        self.product_xyz_path = os.path.join(self.local_path, 'product.xyz')
        self.ts_out_xyz_path = os.path.join(self.local_path, 'rits_ts.xyz')
        self.yml_in_path = os.path.join(self.local_path, 'input.yml')
        self.yml_out_path = os.path.join(self.local_path, 'output.yml')

    def set_input_file_memory(self) -> None:
        """Set the input file memory attribute."""
        self.cpu_cores, self.job_memory_gb = 1, 1

    def execute_incore(self):
        """Execute the RitS job locally (in-process subprocess)."""
        self._log_job_execution()
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()
        self.execute_rits()
        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        """Execute the RitS job to the server's queue."""
        self.execute_rits(exe_type='queue')

    def execute_rits(self, exe_type: str = 'incore'):
        """
        Drive the RitS subprocess and stitch its output back into ARC.

        Args:
            exe_type (str, optional): Either ``'incore'`` (run locally now) or
                ``'queue'`` (just stage the input.yml + submit script).
        """
        if not _rits_environment_ready():
            return
        rxn = self.reactions[0]
        if rxn.ts_species is None:
            rxn.ts_species = ARCSpecies(label=self.species_label,
                                        is_ts=True,
                                        charge=rxn.charge,
                                        multiplicity=rxn.multiplicity,
                                        )

        # Build atom-aligned reactant + product XYZ files. ARC's get_reactants_xyz /
        # get_products_xyz already use rxn.atom_map to align orderings.
        try:
            r_xyz_dict = rxn.get_reactants_xyz(return_format='dict')
            p_xyz_dict = rxn.get_products_xyz(return_format='dict')
        except Exception as e:
            logger.warning(f'RitS: could not build mapped XYZs for {rxn.label}: {e}')
            return
        if r_xyz_dict is None or p_xyz_dict is None:
            logger.warning(f'RitS: empty mapped XYZs for {rxn.label}')
            return
        if len(r_xyz_dict['symbols']) != len(p_xyz_dict['symbols']):
            logger.warning(f'RitS: atom count mismatch for {rxn.label} '
                           f'(R has {len(r_xyz_dict["symbols"])}, P has {len(p_xyz_dict["symbols"])}). '
                           f'Skipping.')
            return

        write_xyz_file(r_xyz_dict, self.reactant_xyz_path, comment=f'{rxn.label} reactant')
        write_xyz_file(p_xyz_dict, self.product_xyz_path, comment=f'{rxn.label} product')

        input_dict = {
            'reactant_xyz_path': self.reactant_xyz_path,
            'product_xyz_path': self.product_xyz_path,
            'rits_repo_path': RITS_REPO_PATH,
            'ckpt_path': RITS_CKPT_PATH,
            'output_xyz_path': self.ts_out_xyz_path,
            'yml_out_path': self.yml_out_path,
            'config_path': os.path.join(RITS_REPO_PATH, 'scripts', 'conf', 'rits.yaml'),
            'n_samples': self.n_samples,
            'batch_size': DEFAULT_BATCH_SIZE,
            'charge': int(rxn.charge or 0),
            'device': 'auto',
        }
        save_yaml_file(path=self.yml_in_path, content=input_dict)

        if exe_type == 'queue':
            self.legacy_queue_execution()
            return

        # Incore: subprocess into rits_script.py inside rits_env.
        # Pass argv as a list (not shell=True) so paths containing spaces or
        # shell-special characters are handled safely without quoting.
        cmd = [RITS_PYTHON, RITS_SCRIPT_PATH, '--yml_in_path', self.yml_in_path]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.warning(f'RitS subprocess returned non-zero exit code {result.returncode} for {rxn.label}.')
            return

        if not os.path.isfile(self.yml_out_path):
            logger.warning(f'RitS produced no output YAML at {self.yml_out_path} for {rxn.label}.')
            return

        tsg_dicts = read_yaml_file(self.yml_out_path) or list()
        n_added = 0
        for tsg_dict in tsg_dicts:
            if process_rits_tsg(tsg_dict=tsg_dict,
                                local_path=self.local_path,
                                ts_species=rxn.ts_species):
                n_added += 1

        if len(self.reactions) < 5:
            if n_added:
                logger.info(f'RitS successfully found {n_added} TS guesses for {rxn.label}.')
            else:
                logger.info(f'RitS did not find any successful TS guesses for {rxn.label}.')


def write_xyz_file(xyz_dict: dict, path: str, comment: str = '') -> None:
    """
    Write an ARC xyz dict to a plain XYZ file with the standard
    ``<n_atoms>\\n<comment>\\n<symbol x y z>...`` header.

    Args:
        xyz_dict (dict): An ARC xyz dictionary.
        path (str): Output file path.
        comment (str): Optional comment line (kept on a single line).
    """
    body = xyz_to_str(xyz_dict)
    n_atoms = len(xyz_dict['symbols'])
    safe_comment = comment.replace('\n', ' ').strip()
    with open(path, 'w') as f:
        f.write(f'{n_atoms}\n{safe_comment}\n{body}\n')


def process_rits_tsg(tsg_dict: dict,
                     local_path: str,
                     ts_species: ARCSpecies) -> bool:
    """
    Convert a single TSGuess-shaped dict from ``rits_script.py`` into an ARC
    ``TSGuess`` object, dedup against existing guesses, and append it.

    Dedup uses :func:`arc.species.converter.compare_confs`, which compares
    *internal distance matrices* — so it correctly merges two RitS samples
    that are the same TS structure in different rigid orientations. This is
    a stricter test than the byte-level ``almost_equal_coords`` ARC's older
    adapters use; RitS specifically benefits from it because every flow-
    matching sample lands the molecule in its own random orientation, so
    rotated duplicates are the common case.

    Args:
        tsg_dict (dict): One entry from the YAML written by rits_script.py.
        local_path (str): The job's local working directory (used by save_geo).
        ts_species (ARCSpecies): The reaction's TS species accumulator.

    Returns:
        bool: ``True`` if a new (unique, non-colliding) TS guess was appended,
        ``False`` otherwise.
    """
    if not tsg_dict.get('success') or not tsg_dict.get('initial_xyz'):
        return False
    try:
        ts_xyz = str_to_xyz(tsg_dict['initial_xyz'])
    except Exception as e:
        logger.warning(f'RitS: could not parse TS xyz: {e}')
        return False
    if colliding_atoms(ts_xyz):
        return False

    # Dedup against every existing TSGuess (regardless of method) using a
    # rotation/translation-invariant distance-matrix comparator. If a match
    # is found, augment the existing guess's method label instead of
    # appending a duplicate.
    for other_tsg in ts_species.ts_guesses:
        if other_tsg.success and other_tsg.initial_xyz is not None \
                and other_tsg.initial_xyz.get('symbols') == ts_xyz['symbols'] \
                and compare_confs(ts_xyz, other_tsg.initial_xyz):
            if 'rits' not in other_tsg.method.lower():
                other_tsg.method += ' and RitS'
            return False

    method_index = int(tsg_dict.get('method_index', 0))
    tsg = TSGuess(method='RitS',
                  method_direction=tsg_dict.get('method_direction', 'F'),
                  method_index=method_index,
                  index=len(ts_species.ts_guesses),
                  success=True,
                  )
    tsg.process_xyz(ts_xyz)
    ts_species.ts_guesses.append(tsg)
    save_geo(xyz=ts_xyz,
             path=local_path,
             filename=f'RitS {method_index}',
             format_='xyz',
             comment=f'RitS sample {method_index}',
             )
    return True


def _rits_environment_ready() -> bool:
    """
    Check that everything RitS needs at runtime is in place. Logs a clear
    one-line warning per missing piece and returns ``False`` so the adapter
    can skip cleanly without raising.
    """
    ok = True
    if not RITS_PYTHON or not os.path.isfile(RITS_PYTHON):
        logger.warning('RitS adapter: rits_env python not found '
                       '(set RITS_PYTHON or run `make install-rits`). Skipping RitS TS guesses.')
        ok = False
    if not RITS_REPO_PATH or not os.path.isdir(RITS_REPO_PATH):
        logger.warning('RitS adapter: RitS source checkout not found '
                       '(set ARC_RITS_REPO or run `make install-rits`). Skipping RitS TS guesses.')
        ok = False
    if not RITS_CKPT_PATH or not os.path.isfile(RITS_CKPT_PATH):
        logger.warning('RitS adapter: pretrained checkpoint not found '
                       '(set ARC_RITS_CKPT or run `make install-rits`). Skipping RitS TS guesses.')
        ok = False
    return ok


register_job_adapter('rits', RitSAdapter)
