"""
An adapter for dummy jobs, meant for testing and debugging only.

Forges skeletal Gaussian-format `.log` files for ARC's pipeline using
fixture data extracted from real DFT runs. The fixture file is selected
from the level of theory: ``mockter3/def2tzvp`` loads
``arc/testing/mockter_fixtures/mockter3.yml``; ``mockter_CBS-QB3_3``
(a composite mockter level, no basis set) loads the same fixture.

A fixture entry may carry a ``raise:`` clause to encode interrupt
scenarios deterministically — mockter raises ``MockterAbort`` (or, for
``sigterm``, sends SIGTERM to its own process) when one of those entries
is looked up. Tests that exercise interrupts must run ARC in a subprocess.

When the fixture file or the looked-up entry is missing, mockter falls
back to today's hand-rolled values, logs a single WARN, and writes a
``mockter_fallback.flag`` marker in the job's local directory so tests
can detect the fallback.
"""

import datetime
import os
import re
import signal
from typing import TYPE_CHECKING

from arc.common import ARC_PATH, get_logger, read_yaml_file, save_yaml_file
from arc.imports import settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter, update_input_dict_with_args
from arc.job.adapters.mockter_fixture import Fixture, FixtureError
from arc.job.adapters.mockter_renderer import render_gaussian_log
from arc.job.factory import register_job_adapter
from arc.level import Level
from arc.species.converter import xyz_to_str, str_to_xyz

if TYPE_CHECKING:
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies


logger = get_logger()

default_job_settings, global_ess_settings, input_filenames, output_filenames, servers, submit_filenames = \
    settings['default_job_settings'], settings['global_ess_settings'], settings['input_filenames'], \
    settings['output_filenames'], settings['servers'], settings['submit_filenames']

MOCKTER_FIXTURES_DIR = os.path.join(ARC_PATH, 'arc', 'testing', 'mockter_fixtures')
MOCKTER_INDEX_RE = re.compile(r'^mockter(\d{1,2})$', re.IGNORECASE)
MOCKTER_COMPOSITE_RE = re.compile(r'^mockter_[A-Za-z0-9-]+_(\d{1,2})$', re.IGNORECASE)


class MockterAbort(Exception):
    """Raised when a fixture entry's ``raise:`` clause is encountered."""

    def __init__(self, kind: str, message: str | None = None):
        self.kind = kind
        self.message = message
        super().__init__(f'mockter aborted ({kind}): {message or "no message"}')


def _parse_mockter_index(method: str | None) -> int | None:
    """
    Extract the fixture index N from a mockter level method.

    Args:
        method (str | None): The level's ``method`` attribute (e.g. ``'mockter3'``,
                             ``'mockter_CBS-QB3_3'``). Case-insensitive.

    Returns:
        int | None: The index N (1-99), or None if ``method`` is not a mockter level.
    """
    if not method:
        return None
    m = MOCKTER_INDEX_RE.match(method) or MOCKTER_COMPOSITE_RE.match(method)
    return int(m.group(1)) if m else None


_FIXTURE_CACHE: dict[int, Fixture | None] = {}


def _load_fixture_cached(index: int) -> Fixture | None:
    """
    Load (and cache) the fixture YAML for a given index.

    Args:
        index (int): The fixture index (1-99).

    Returns:
        Fixture | None: The loaded Fixture, or None if the file is absent or malformed.
    """
    if index in _FIXTURE_CACHE:
        return _FIXTURE_CACHE[index]
    path = os.path.join(MOCKTER_FIXTURES_DIR, f'mockter{index}.yml')
    try:
        fixture = Fixture.load(path)
    except FixtureError as exc:
        logger.warning(f'mockter fixture {path} failed to load: {exc}')
        fixture = None
    _FIXTURE_CACHE[index] = fixture
    return fixture


class MockAdapter(JobAdapter):
    """
    A class for executing mock jobs.

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
        dihedrals (list[float], optional): The dihedral angels corresponding to self.torsions.
        directed_scan_type (str, optional): The type of the directed scan.
        ess_settings (dict, optional): A dictionary of available ESS and a corresponding server list.
        ess_trsh_methods (list[str], optional): A list of troubleshooting methods already tried out.
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
        run_multi_species (bool, optional): Whether to run a job for multiple species in the same input file.
        reactions (list[ARCReaction], optional): Entries are ARCReaction instances, used for TS search methods.
        rotor_index (int, optional): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        server (str): The server to run on.
        server_nodes (list, optional): The nodes this job was previously submitted to.
        species (list[ARCSpecies], optional): Entries are ARCSpecies instances.
                                              Either ``reactions`` or ``species`` must be given.
        testing (bool, optional): Whether the object is generated for testing purposes, ``True`` if it is.
        times_rerun (int, optional): Number of times this job was re-run with the same arguments (no trsh methods).
        torsions (list[list[int]], optional): The 0-indexed atom indices of the torsion(s).
        tsg (int, optional): TSGuess number if optimizing TS guesses.
        xyz (dict, optional): The 3D coordinates to use. If not give, species.get_xyz() will be used.
    """

    def __init__(self,
                 project: str,
                 project_directory: str,
                 job_type: list[str] | str,
                 args: dict | None = None,
                 bath_gas: str | None = None,
                 checkfile: str | None = None,
                 conformer: int | None = None,
                 constraints: list[tuple[list[int], float]] | None = None,
                 cpu_cores: str | None = None,
                 dihedral_increment: float | None = None,
                 dihedrals: list[float] | None = None,
                 directed_scan_type: str | None = None,
                 ess_settings: dict | None = None,
                 ess_trsh_methods: list[str] | None = None,
                 execution_type: str | None = None,
                 fine: bool = False,
                 initial_time: datetime.datetime | str | None = None,
                 irc_direction: str | None = None,
                 job_id: int | None = None,
                 job_memory_gb: float = 14.0,
                 job_name: str | None = None,
                 job_num: int | None = None,
                 job_server_name: str | None = None,
                 job_status: list[dict | str] | None = None,
                 level: Level | None = None,
                 max_job_time: float | None = None,
                 run_multi_species: bool = False,
                 reactions: list[ARCReaction] | None = None,
                 rotor_index: int | None = None,
                 server: str | None = None,
                 server_nodes: list | None = None,
                 queue: str | None = None,
                 attempted_queues: list[str] | None = None,
                 species: list[ARCSpecies] | None = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: list[list[int]] | None = None,
                 tsg: int | None = None,
                 xyz: dict | None = None,
                 ):

        self.incore_capacity = 1
        self.job_adapter = 'mockter'
        self.execution_type = 'incore'
        self.command = 'mockter'
        self.url = ''

        if species is None and reactions is None:
            raise ValueError('Cannot execute Mockter without an ARCSpecies or an ARCReaction object.')

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

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.
        """
        input_dict = dict()
        input_dict['basis'] = self.level.basis or ''
        input_dict['charge'] = self.charge
        input_dict['label'] = self.species_label
        input_dict['memory'] = self.input_file_memory
        input_dict['method'] = self.level.method
        input_dict['multiplicity'] = self.multiplicity
        input_dict['xyz'] = xyz_to_str(self.get_mock_xyz())
        input_dict['job_type'] = self.job_type
        input_dict['memory'] = self.input_file_memory

        input_dict = update_input_dict_with_args(args=self.args, input_dict=input_dict)
        save_yaml_file(path=os.path.join(self.local_path, input_filenames[self.job_adapter]), content=input_dict)

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
        # 1. ** Upload **
        # 1.1. submit file
        if self.execution_type != 'incore':
            # we need a submit file for single or array jobs (either submitted to local or via SSH)
            self.write_submit_script()
            self.files_to_upload.append(self.get_file_property_dictionary(
                file_name=submit_filenames[servers[self.server]['cluster_soft']]))
        # 1.2. input file
        if not self.iterate_by:
            # if this is not a job array, we need the ESS input file
            self.write_input_file()
            self.files_to_upload.append(self.get_file_property_dictionary(file_name=input_filenames[self.job_adapter]))
        # 1.3. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        # 1.4 job.sh
        job_sh_dict = self.set_job_shell_file_to_upload()  # Set optional job.sh files if relevant.
        if job_sh_dict is not None:
            self.files_to_upload.append(job_sh_dict)
        # 2. ** Download **
        # 2.1. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_download.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        else:
            # 2.2. output file
            self.files_to_download.append(self.get_file_property_dictionary(file_name=output_filenames[self.job_adapter]))

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        pass

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        self.input_file_memory = self.job_memory_gb

    def get_mock_xyz(self) -> dict | None:
        """
        Get the xyz coordinates for the mock job.
        """
        if self.xyz is not None:
            return self.xyz
        if self.species is not None and len(self.species):
            return self.species[0].get_xyz()
        if self.reactions is not None and len(self.reactions):
            ts_xyz_str = ''
            reactants, _ = self.reactions[0].get_reactants_and_products()
            for r in reactants:
                ts_xyz_str += xyz_to_str(r.get_xyz()) + '\n'
            return str_to_xyz(ts_xyz_str)
        return None

    def execute_incore(self):
        """
        Execute a mock job incore by forging a skeletal Gaussian log from
        fixture data (or hand-rolled fallback values when no fixture matches).

        Looks up the fixture entry by ``(species_label, job_type, conformer,
        tsg, torsions, irc_direction, fine, is_ts)``, handles ``raise:``
        clauses, then writes ``output.log`` (canonical) and a debug
        ``output.yml`` mirror in ``self.local_path``.

        Raises:
            MockterAbort: When the fixture entry encodes an interrupt
                          (``crash``, ``oom``, ``timeout``, ``scf_nonconvergence``,
                          ``sigterm``). For ``sigterm``, also sends SIGTERM to
                          the current process before raising.
        """
        method = self.level.method if self.level is not None else None
        fixture_index = _parse_mockter_index(method)
        fixture = _load_fixture_cached(fixture_index) if fixture_index is not None else None

        entry = None
        if fixture is not None:
            entry = fixture.lookup(
                label=self.species_label if isinstance(self.species_label, str) else (self.species_label or [None])[0],
                job_type=self.job_type,
                conformer=self.conformer,
                tsg=self.tsg,
                torsions=self.torsions,
                irc_direction=self.irc_direction,
                fine=getattr(self, 'fine', False),
                is_ts=bool(self.is_ts),
            )

        if entry is not None and entry.is_raise():
            kind = entry.raise_kind()
            message = entry.raise_message()
            logger.warning(f'mockter raising {kind} for {self.job_name}: {message}')
            if kind == 'sigterm':
                os.kill(os.getpid(), signal.SIGTERM)
            raise MockterAbort(kind=kind, message=message)

        if entry is not None:
            log_text, debug_dict = self._render_from_fixture(entry.payload, fixture)
        else:
            if fixture_index is not None:
                self._write_fallback_flag(reason='fixture-miss')
                logger.warning(
                    f'mockter fixture lookup miss for label={self.species_label!r} '
                    f'job_type={self.job_type!r} (fixture mockter{fixture_index}.yml); using fallback values.'
                )
            log_text, debug_dict = self._render_from_fallback()

        with open(os.path.join(self.local_path, 'output.log'), 'w', encoding='utf-8') as f:
            f.write(log_text)
        save_yaml_file(path=os.path.join(self.local_path, 'output.yml'), content=debug_dict)

    def _render_from_fixture(self, payload: dict, fixture: 'Fixture') -> tuple[str, dict]:
        """
        Render the canonical Gaussian log + debug YAML from a fixture payload.

        Method and basis fields used in the forged log come from the fixture's
        provenance (the real DFT level used to produce the data), not from the
        mockter alias. This is what makes ``CCSD(T)=`` lines emit when the
        underlying sp was a CC calculation, even though the mockter level name
        is ``mockter4``.

        Args:
            payload (dict): The leaf entry from the fixture (e.g. an opt block,
                            a freq block, an sp block, a conformer entry).
            fixture (Fixture): The loaded fixture (used for provenance lookup).

        Returns:
            tuple[str, dict]: (forged log text, debug dict mirror).
        """
        xyz_str = payload.get('xyz')
        e_elect = payload.get('e_elect')
        freqs = payload.get('freqs')
        zpe = payload.get('zpe')
        hessian_block = payload.get('hessian_block')
        t1 = payload.get('t1_diagnostic')
        real_level = self._provenance_level_for_job_type(fixture)
        real_method = (real_level or {}).get('method') or (self.level.method if self.level else 'mockter')
        real_basis = (real_level or {}).get('basis') or (self.level.basis or '' if self.level else '')
        is_t1 = self.job_type == 'sp' and 'cc' in real_method.lower()

        log_text = render_gaussian_log(
            job_type=self.job_type,
            xyz=xyz_str if xyz_str else (self.get_mock_xyz() if hasattr(self, 'get_mock_xyz') else None),
            e_elect_hartree=e_elect,
            method=real_method,
            basis=real_basis,
            multiplicity=self.multiplicity if isinstance(self.multiplicity, int) else 1,
            charge=self.charge if isinstance(self.charge, int) else 0,
            freqs_cm1=freqs,
            zpe_hartree=zpe,
            hessian_block=hessian_block,
            is_t1_capable=is_t1,
            t1_diagnostic=t1,
            title=self.species_label if isinstance(self.species_label, str) else 'mockter',
        )
        debug_dict = {
            'adapter': 'mockter',
            'job_type': self.job_type,
            'fixture_payload_keys': sorted(payload.keys()),
            'xyz': str_to_xyz(xyz_str) if xyz_str else None,
            'sp': e_elect,
            'freqs': list(freqs) if freqs is not None else None,
            'zpe': zpe,
            'T1': t1,
        }
        return log_text, debug_dict

    def _render_from_fallback(self) -> tuple[str, dict]:
        """
        Render a forged log using the legacy hand-rolled values (the
        adapter's pre-fixture behavior). Used when no fixture matches the
        current job. Marker file ``mockter_fallback.flag`` is written
        separately so tests can detect the fallback.

        Returns:
            tuple[str, dict]: (forged log text, debug dict mirror).
        """
        xyz = self.get_mock_xyz()
        e_elect = 0.0 if not self.is_ts else 0.05
        n_atoms = len(xyz['symbols']) if xyz else 1
        freqs = [500.0 + 20.0 * i for i in range(max(3 * n_atoms - 6, 1))]
        if self.is_ts:
            freqs[0] = -500.0
        zpe = 0.005 * len(freqs)

        log_text = render_gaussian_log(
            job_type=self.job_type,
            xyz=xyz,
            e_elect_hartree=e_elect,
            method=(self.level.method if self.level else 'mockter'),
            basis=(self.level.basis or '' if self.level else ''),
            multiplicity=self.multiplicity if isinstance(self.multiplicity, int) else 1,
            charge=self.charge if isinstance(self.charge, int) else 0,
            freqs_cm1=freqs if self.job_type in ('freq', 'composite') else None,
            zpe_hartree=zpe if self.job_type in ('freq', 'composite') else None,
            title=self.species_label if isinstance(self.species_label, str) else 'mockter',
        )
        debug_dict = {
            'adapter': 'mockter',
            'job_type': self.job_type,
            'fallback': True,
            'xyz': xyz,
            'sp': e_elect,
            'freqs': freqs,
            'zpe': zpe,
            'T1': 0.0002,
        }
        return log_text, debug_dict

    def _provenance_level_for_job_type(self, fixture: 'Fixture') -> dict | None:
        """
        Pick the right provenance level dict for the current job_type.

        sp / composite / scan jobs read their level from the matching
        provenance entry; other job types fall back to opt_level.

        Args:
            fixture (Fixture): The loaded fixture.

        Returns:
            dict | None: The provenance level dict (with 'method', 'basis'
                         keys), or None if not present.
        """
        provenance = fixture.provenance or {}
        if self.job_type == 'sp':
            return provenance.get('sp_level') or provenance.get('opt_level')
        if self.job_type == 'composite':
            return provenance.get('composite_method') or provenance.get('opt_level')
        if self.job_type == 'freq':
            return provenance.get('freq_level') or provenance.get('opt_level')
        return provenance.get('opt_level')

    def _write_fallback_flag(self, reason: str) -> None:
        """
        Write a ``mockter_fallback.flag`` marker file in this job's local
        directory so tests can assert that no fallback occurred when running
        a fully fixture-backed scenario.

        Args:
            reason (str): Short text identifying why fallback was used.
        """
        flag_path = os.path.join(self.local_path, 'mockter_fallback.flag')
        try:
            with open(flag_path, 'w', encoding='utf-8') as f:
                f.write(f'reason: {reason}\n')
        except OSError:
            pass

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        self.legacy_queue_execution()


register_job_adapter('mockter', MockAdapter)
