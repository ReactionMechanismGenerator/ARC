"""
An adapter for executing Orca 5/6 jobs

https://orcaforum.kofo.mpg.de/app.php/portal
"""

import datetime
import math
import os
from typing import TYPE_CHECKING

from mako.template import Template

from arc.common import get_logger, torsions_to_scans
from arc.imports import incore_commands, settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import (_initialize_adapter,
                                     is_restricted,
                                     update_input_dict_with_args,
                                     which,
                                     )
from arc.job.factory import register_job_adapter
from arc.job.local import execute_command
from arc.level import Level
from arc.settings.settings import orca_default_options_dict
from arc.species.converter import xyz_to_str
from arc.species.vectors import calculate_dihedral_angle

if TYPE_CHECKING:
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies


logger = get_logger()

ORCA_METHOD_ALIASES = {
    'wb97xd3': 'wb97x-d3',
}


def _format_orca_method(method: str) -> str:
    """
    Convert ARC method names to ORCA-friendly labels when needed.
    """
    if not method:
        return method
    if method.lower() == 'wb97xd':
        logger.warning('ORCA does not support wb97xd; use wb97x or wb97x-d3.')
    return ORCA_METHOD_ALIASES.get(method.lower(), method)


def _format_orca_basis_token(token: str) -> str:
    """
    Convert def2 basis tokens to ORCA formatting (e.g., def2tzvp -> def2-tzvp).
    """
    if not token:
        return token
    parts = token.split('/')
    base = parts[0]
    if base.lower().startswith('def2'):
        base_rest = base[4:]
        if base_rest.startswith('-'):
            base_rest = base_rest[1:]
        if base_rest:
            base = f"def2-{base_rest.lower()}"
    if len(parts) > 1:
        parts = [base] + [part.lower() for part in parts[1:]]
        return '/'.join(parts)
    return base


def _format_orca_basis(basis: str) -> str:
    """
    Convert basis strings to ORCA-friendly labels where applicable.
    """
    if not basis:
        return basis
    return ' '.join(_format_orca_basis_token(token) for token in basis.split())

default_job_settings, global_ess_settings, input_filenames, output_filenames, servers, submit_filenames = \
    settings['default_job_settings'], settings['global_ess_settings'], settings['input_filenames'], \
    settings['output_filenames'], settings['servers'], settings['submit_filenames']

# job_type_1: 'SP', 'Opt', 'OptTS', 'Freq'
# job_type_2: reserved for Opt + Freq.
# restricted: 'R' = closed-shell SCF, 'U' = spin unrestricted SCF, 'RO' = open-shell spin restricted SCF
# auxiliary_basis: required for DLPNO calculations (speed up calculation)
# cabs: Complementary Auxiliary Basis Set for F12 calculations (e.g., cc-pVTZ-F12-CABS)
# memory: MB per core (must increase as system gets larger)
# cpus: must be less than number of electron pairs, defaults to min(heavy atoms, cpus limit)
# job_options_blocks: input blocks that enable detailed control over program
# job_options_keywords: input keywords that control the job
# method_class: 'HF' for wavefunction methods (hf, mp, cc, dlpno ...). 'KS' for DFT methods.
# options: additional keywords to control job (e.g., TightSCF, NormalPNO ...)
input_template = """!${restricted}${method_class} ${method} ${basis} ${auxiliary_basis}${cabs} ${keywords}
!${job_type_1} 
${job_type_2}${orbital_guess}
%%maxcore ${memory}
%%pal nprocs ${cpus} end

* xyz ${charge} ${multiplicity}
${xyz}
*

%%scf
MaxIter 999${scf_keys}
end${scan}
${block}
"""


ORBITALS_DOWNLOAD_JOB_TYPES = ['composite', 'opt', 'optfreq', 'stability']
ORBITALS_GUESS_JOB_TYPES = ['conf_opt', 'conf_sp', 'freq', 'opt', 'optfreq', 'scan', 'sp', 'stability']


class OrcaAdapter(JobAdapter):
    """
    A class for executing Orca jobs.

    Args:
        project (str): The project's name. Used for setting the remote path.
        project_directory (str): The path to the local project directory.
        job_type (list, str): The job's type, validated against ``JobTypeEnum``. If it's a list, pipe.py will be called.
        args (dict, optional): Methods (including troubleshooting) to be used in input files.
                               Keys are either 'keyword', 'block', or 'trsh', values are dictionaries with values
                               to be used either as keywords or as blocks in the respective software input file.
                               If 'trsh' is specified, an action might be taken instead of appending a keyword or a
                               block to the input file (e.g., change server or change scan resolution).
        bath_gas (str, optional): A bath gas. Currently only used in OneDMin to calculate L-J parameters. Allowed values
                                  are: ``'He'``, ``'Ne'``, ``'Ar'``, ``'Kr'``, ``'H2'``, ``'N2'``, or ``'O2'``.
        checkfile (str, optional): The path to a previous job's orbitals file (``.gbw``) to be used in the current job.
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

    check_file_name = 'input.gbw'
    guess_file_name = 'guess.gbw'

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
        self.job_adapter = 'orca'
        self.execution_type = execution_type or 'queue'
        self.command = 'orca'
        self.url = 'https://orcaforum.kofo.mpg.de/app.php/portal'

        if species is None:
            raise ValueError('Cannot execute Orca without an ARCSpecies object.')

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

        if self.checkfile is None:
            if os.path.isfile(os.path.join(self.local_path, self.check_file_name)):
                self.checkfile = os.path.join(self.local_path, self.check_file_name)
            elif self.species[0].checkfile is not None and os.path.isfile(self.species[0].checkfile):
                self.checkfile = self.readable_checkfile(self.species[0].checkfile)

    def reads_orbital_guess(self) -> bool:
        """
        Report whether this job starts its SCF from a previous job's orbitals.

        The single predicate behind both halves of reading a guess: the ``!MORead`` and
        ``%moinp`` keywords ``write_input_file`` emits and the ``guess.gbw`` upload
        ``set_files`` adds. Emitting the keywords without uploading the file aborts the job on
        a missing guess, so the two are answered here rather than tested twice.

        ``ORBITALS_GUESS_JOB_TYPES`` holds the job types this adapter writes an SCF on one
        starting structure for: the ``opt``, ``conf_opt``, ``optfreq`` and ``scan`` jobs, whose
        first SCF is the one the guess seeds and whose later points ORCA propagates orbitals
        through itself, and the ``freq``, ``sp``, ``conf_sp`` and ``stability`` jobs, which run
        a single SCF. The job types absent from it are those ``write_input_file`` writes no
        keyword for, so ORCA is handed no calculation for a guess to seed: ``composite``, for
        which ORCA offers no composite method; ``irc`` and ``orbitals``, for which this adapter
        writes neither a path-following nor an orbital-printing input; ``directed_scan``, for
        which it writes neither the scan block nor the constraints such a job needs; and
        ``gen_confs``, ``tsg`` and ``onedmin``, which belong to other adapters entirely.

        A job array writes no input file at all and its members share one remote path, where a
        single uploaded guess would stand in for every member; this is the reason its orbitals
        are not downloaded either. A monatomic species is excluded as it is in Gaussian: ARC
        spawns it neither an optimization nor a frequency job, so there is no chain to hold.

        ORCA names its own orbitals after the input file and cannot read and write one file the
        way Gaussian reuses a single checkfile, so the guess is uploaded under a name the job
        will not overwrite. ORCA projects a guess written in another basis set onto this job's
        basis, so no level or basis is tracked here and the question is only whether a checkfile
        this adapter's ESS wrote exists.

        Returns: bool
            Whether this job reads a previous job's orbitals as its initial guess.
        """
        return self.job_type in ORBITALS_GUESS_JOB_TYPES \
            and not self.iterate_by \
            and self.species[0].number_of_atoms > 1 \
            and self.checkfile is not None \
            and os.path.isfile(self.checkfile)

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.

        Where ``reads_orbital_guess`` holds, the input carries ``!MORead`` and a ``%moinp``
        naming the uploaded ``guess.gbw``, so the SCF starts from the orbitals a previous job
        converged and a species' jobs describe one wavefunction rather than whichever solution
        each fresh SCF happens to reach.

        A ``stability`` job is a single point that adds ``STABPerform`` and
        ``STABRestartUHFifUnstable true`` to the ``%scf`` block. ORCA follows an instability it
        finds and analyses the relaxed solution again, so such a log holds two analyses; the
        verdict of the wavefunction under test is the first of them.
        ``docs/source/advanced.rst`` records why the follow is not optional.
        """
        if 'f12' in self.level.method and not self.level.cabs:
            raise ValueError(
                f"Level '{self.level}' uses an F12 method without a CABS basis. "
                f"Set `cabs:` in the level spec (e.g. cc-pVTZ-F12-CABS). "
                f"Without it ORCA runs with DimCABS = 0 and returns non-F12 energies."
            )
        input_dict = dict()
        for key in ['block',
                    'orbital_guess',
                    'scan',
                    'scf_keys',
                    'job_type_1',
                    'job_type_2',
                    'keywords',
                    ]:
            input_dict[key] = ''
        input_dict['auxiliary_basis'] = _format_orca_basis(self.level.auxiliary_basis or '')
        input_dict['basis'] = _format_orca_basis(self.level.basis or '')
        input_dict['cabs'] = f' {_format_orca_basis(self.level.cabs)}' if self.level.cabs else ''
        input_dict['charge'] = self.charge
        input_dict['cpus'] = self.cpu_cores
        input_dict['label'] = self.species_label
        input_dict['memory'] = self.input_file_memory
        input_dict['method'] = _format_orca_method(self.level.method) if 'mrci' not in self.level.method else ''
        input_dict['multiplicity'] = self.multiplicity
        input_dict['xyz'] = xyz_to_str(self.xyz)

        self.args['keyword'].setdefault(
            'scf_convergence',
            orca_default_options_dict['global']['keyword'].get('scf_convergence', '').lower())
        if not self.args['keyword']['scf_convergence']:
            raise ValueError('Orca SCF convergence is not specified. Please specify this variable either in '
                             'settings.py as default or in the input file as additional options.')

        # Orca requires different blocks for wavefunction methods and DFT methods
        if self.level.method_type == 'dft':
            input_dict['method_class'] = 'KS'
            # Use a consistent DFT grid for fine_opt jobs and for any job with a frequency calculation
            # (`freq` and `optfreq`), so `optfreq` is treated like `freq` here and defaults to `defgrid3`.
            # Users can override by setting `dft_grid` in args.keyword (e.g. dft_grid: DEFGRID1).
            self.args['keyword'].setdefault('dft_grid', 'defgrid3' if self.fine or self.job_type in ['freq', 'optfreq', 'stability'] else 'defgrid2')
        elif self.level.method_type == 'wavefunction':
            input_dict['method_class'] = 'HF'
            if 'dlpno' in self.level.method:
                self.args['keyword'].setdefault(
                    'dlpno_threshold',
                    orca_default_options_dict['global']['keyword'].get('dlpno_threshold', '').lower())
                if not self.args['keyword']['dlpno_threshold']:
                    raise ValueError('Orca DLPNO threshold is not specified. Please specify this variable either in '
                                     'settings.py as default or in the input file as additional options.')
        else:
            logger.debug(f'Running {self.level.method_type} {self.level.method} method in Orca.')

        input_dict['restricted'] = 'r' if is_restricted(self) else 'u'

        # Job type specific options
        if self.job_type in ['opt', 'conf_opt', 'optfreq']:
            opt_convergence_key = 'fine_opt_convergence' if self.fine else 'opt_convergence'
            opt_convergence = self.args['keyword'].get(opt_convergence_key, '').lower() or \
                orca_default_options_dict['opt']['keyword'].get(opt_convergence_key, '').lower()
            if not opt_convergence:
                raise ValueError('Orca optimization convergence (NormalOpt or TightOpt) is not specified. '
                                 'Please specify this variable either in the settings.py as default options '
                                 'or in the input file as additional options.')
            self.add_to_args(val=opt_convergence, key1='keyword')
            if not self.is_ts:
                input_dict['job_type_1'] = 'Opt'
            else:
                input_dict['job_type_1'] = 'OptTS'
                self.add_to_args(val="""
%geom
Calc_Hess true # calculation of the exact Hessian before the first opt step
end
""",
                                 key1='block')
            if 'dlpno' in self.level.method:
                input_dict['job_type_1'] += '\n!NUMGRAD'  # Numerical gradient for DLPNO opt

        elif self.job_type in ['freq', 'optfreq']:
            if self.job_type == 'freq':
                input_dict['job_type_1'] = 'Freq'
            elif self.job_type == 'optfreq':
                input_dict['job_type_2'] = '!Freq'
            use_num_freq = self.args['keyword'].get('use_num_freq', False) \
                or orca_default_options_dict['freq']['keyword'].get('use_num_freq', False)
            if use_num_freq:
                self.add_to_args(val='NumFreq', key1='keyword')
                logger.info('Using numerical frequencies calculation in Orca. Note: This job might therefore be '
                            'time-consuming.')

        elif self.job_type in ['sp', 'conf_sp']:
            input_dict['job_type_1'] = 'sp'
            if 'mrci' in self.level.method and self.species[0].active is not None:
                if '_' in self.level.method:
                    methods = self.level.method.split('_')
                    block = ''
                    for method in methods:
                        if method == 'mp2':
                            block += '\n\n%mp2\n    RI true\nend'
                        elif method == 'casscf':
                            block += (f'\n\n%casscf\n    nel {self.species[0].active[0]}'
                                      f'\n    norb {self.species[0].active[1]}\n    nroots 1\n    maxiter 999\nend')
                        elif method == 'mrci':
                            block += f'\n\n%mrci\n    citype MRCI\n    davidsonopt true\n    maxiter 999\nend\n'
                    input_dict['block'] += block

        elif self.job_type == 'stability':
            input_dict['job_type_1'] = 'sp'
            input_dict['scf_keys'] = '\nSTABPerform true\nSTABRestartUHFifUnstable true'

        elif self.job_type == 'scan':
            scans, torsion_strings = list(), list()
            if self.rotor_index is not None:
                if self.species[0].rotors_dict \
                        and self.species[0].rotors_dict[self.rotor_index]['directed_scan_type'] == 'ess':
                    scans = self.species[0].rotors_dict[self.rotor_index]['scan']
                    scans = [scans] if not isinstance(scans[0], list) else scans
            elif len(self.torsions):
                scans = torsions_to_scans(self.torsions)
            if self.torsions is None or not len(self.torsions):
                self.torsions = torsions_to_scans(scans, direction=-1)
            for torsion_indices in self.torsions:
                torsion_strings.append(' '.join([str(atom_index) for atom_index in torsion_indices]))
            input_dict['job_type_1'] = f"Opt{'Ts' if self.is_ts else ''}"
            input_dict['scan'] = '\n%geom Scan'
            for i, torsion in enumerate(torsion_strings):
                dihedral = calculate_dihedral_angle(coords=self.species[0].get_xyz(), torsion=self.torsions[i])
                input_dict['scan'] += f'\nD {torsion} =  {dihedral:.1f}, {dihedral - self.scan_res:.1f}, {self.scan_res:.1f}\n'
            input_dict['scan'] += '\nend\nend' if len(self.torsions) > 1 else '\nend'

        if self.level.solvation_method:
            if self.level.solvation_method.lower() == 'smd':
                self.add_to_args(val=f"""
%cpcm SMD true
      SMDsolvent "{self.level.solvent}"
end
""",
                                key1='block')
            elif self.level.solvation_method.lower() in ['pcm', 'cpcm']:
                self.add_to_args(val=f"""
!CPCM({self.level.solvent})
""",
                                key1='block')

        if self.reads_orbital_guess():
            orbital_guess = f'!MORead\n%moinp "{self.guess_file_name}"'
            input_dict['orbital_guess'] = f'\n{orbital_guess}' if input_dict['job_type_2'] else orbital_guess

        input_dict = update_input_dict_with_args(args=self.args, input_dict=input_dict)

        with open(os.path.join(self.local_path, input_filenames[self.job_adapter]), 'w') as f:
            f.write(Template(input_template).render(**input_dict))

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

        THE ORBITALS FILE. Where ``reads_orbital_guess`` holds, the checkfile the job holds is
        uploaded as ``guess.gbw``, a name ORCA will not overwrite with the ``input.gbw`` the
        job itself writes; that same predicate decides the ``!MORead`` and ``%moinp`` keywords
        of the input file, so the file is uploaded for exactly the jobs that read it.
        ``input.gbw`` is downloaded only for the job types something later reads it from: the
        ``opt``, ``optfreq`` and ``composite`` jobs ``Scheduler.end_job`` adopts a checkfile
        from, and the ``stability`` job, whose own orbitals are the relaxed solution. Every
        other job type would download tens of MB that nothing consumes; a job reading a guess
        is not thereby a job whose own orbitals anything reads. A job array takes the
        ``data.hdf5`` branch and downloads no orbitals at all, since its members share one
        remote path and would overwrite one another's.
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
        # 1.3. orbitals file, uploaded under a name the job will not overwrite with its own
        if self.reads_orbital_guess():
            self.files_to_upload.append(self.get_file_property_dictionary(file_name=self.guess_file_name,
                                                                          local=self.checkfile))
        # 1.4. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        # 1.5 job.sh
        job_sh_dict = self.set_job_shell_file_to_upload()  # Set optional job.sh files if relevant.
        if job_sh_dict is not None:
            self.files_to_upload.append(job_sh_dict)
        # 2. ** Download **
        # 2.1. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_download.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        else:
            # 2.2. log file
            self.files_to_download.append(self.get_file_property_dictionary(
                file_name=output_filenames[self.job_adapter]))
            # 2.3. orbitals file, the guess of any job that follows
            if self.job_type in ORBITALS_DOWNLOAD_JOB_TYPES:
                self.files_to_download.append(self.get_file_property_dictionary(file_name=self.check_file_name))
            # 2.4. Hessian file generated by frequency calculations
            # The Hessian file is useful when the user would like to project out the rotors
            if self.job_type in ['freq', 'optfreq']:
                self.files_to_download.append(self.get_file_property_dictionary(file_name='input.hess'))

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
        # Orca's memory is per cpu core and in MB
        self.input_file_memory = math.ceil(self.job_memory_gb * 1024 / self.cpu_cores)

    def execute_incore(self):
        """
        Execute a job incore.
        """
        which(self.command,
              return_bool=True,
              raise_error=True,
              raise_msg=f'Please install {self.job_adapter}, see {self.url} for more information.',
              )
        self._log_job_execution()
        execute_command(incore_commands[self.job_adapter])

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        self.legacy_queue_execution()


register_job_adapter('orca', OrcaAdapter)
