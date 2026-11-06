"""
An adapter for executing Gaussian jobs

https://gaussian.com/
"""

import datetime
import math
import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from mako.template import Template

from arc.common import get_logger, get_memory_headroom_fraction, torsions_to_scans
from arc.imports import incore_commands, settings
from arc.job.adapter import JobAdapter, constraint_type_dict
from arc.job.adapters.common import (_initialize_adapter,
                                     is_restricted,
                                     species_may_read_previous_orbitals,
                                     update_input_dict_with_args,
                                     which,
                                     combine_parameters
                                     )
from arc.job.factory import register_job_adapter
from arc.job.local import execute_command
from arc.level import Level
from arc.species.converter import xyz_to_str

if TYPE_CHECKING:
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies


logger = get_logger()

default_job_settings, gaussian_memory_headroom_fractions, global_ess_settings, input_filenames, \
    output_filenames, servers, submit_filenames = \
    settings['default_job_settings'], settings['gaussian_memory_headroom_fractions'], settings['global_ess_settings'], \
    settings['input_filenames'], settings['output_filenames'], settings['servers'], settings['submit_filenames']

# Gaussian should not consume the entire scheduler allocation. ARC reserves a
# fixed fraction of the submit-script memory for non-Gaussian overhead such as
# the scheduler, runtime, scratch bookkeeping, and Gaussian allocations outside
# the explicit %mem budget. On a Gaussian memory allocation failure, ARC steps down
# gaussian_memory_headroom_fractions (see settings.py) while holding the queue
# reservation constant.
GAUSSIAN_MEMORY_HEADROOM_FRACTION = gaussian_memory_headroom_fractions[0]

STABILITY_KEYWORD = 'stable=(rext,noopt)'


INTEGRAL_GRID_NAMES = frozenset({'coarse', 'coarsegrid', 'fine', 'finegrid', 'pass0grid', 'sg1', 'sg1grid',
                                 'superfine', 'superfinegrid', 'ultrafine', 'ultrafinegrid'})
INTEGRAL_RELATIVISTIC_NAMES = frozenset({'dkh', 'dkh0', 'dkh2', 'dkhso', 'douglaskrollhess', 'nodkh',
                                         'nonrelativistic', 'resc'})
OPT_CONVERGENCE_NAMES = frozenset({'loose', 'tight', 'verytight'})
SCF_ALGORITHM_NAMES = frozenset({'qc', 'xqc', 'yqc'})
SCF_CONVERGENCE_NAMES = frozenset({'tight', 'verytight'})
GUESS_SOURCE_NAMES = frozenset({'am1', 'cards', 'checkpoint', 'core', 'harris', 'huckel', 'indo', 'local', 'read'})
INTEGRAL_GROUPS = {'grid': INTEGRAL_GRID_NAMES, 'relativistic': INTEGRAL_RELATIVISTIC_NAMES}
OPT_GROUPS = {'convergence': OPT_CONVERGENCE_NAMES}
SCF_GROUPS = {'algorithm': SCF_ALGORITHM_NAMES, 'convergence': SCF_CONVERGENCE_NAMES}
GUESS_GROUPS = {'source': GUESS_SOURCE_NAMES}


def get_integral_option_name(option: str) -> str:
    """
    Get the name under which a Gaussian ``Integral`` option competes with other options.
    All integration grid specifications (``grid=ultrafine``, ``UltraFineGrid``, ``grid=300590``, ``-96032``) share
    the name ``grid``, and all relativistic Hamiltonian options (``DKH``, ``DKH0``, ``DKH2``, ``DKHSO``,
    ``DouglasKrollHess``, ``RESC``, ``NoDKH``, ``NonRelativistic``) share the name ``relativistic``.

    Args:
        option (str): A single option of the ``Integral`` keyword.

    Returns:
        str: The lowercase option name.
    """
    option = option.strip().lower()
    if re.fullmatch(r'-?\d+', option):
        return 'grid'
    return get_route_option_name(option, INTEGRAL_GROUPS)


def get_route_option_name(option: str, groups: dict[str, frozenset[str]] | None = None) -> str:
    """
    Get the name under which an option of a route keyword competes with other options of that keyword.

    Args:
        option (str): A single option, e.g., ``'maxcycle=100'`` or ``'tight'``.
        groups (dict, optional): Names of mutually exclusive bare options, keyed by the shared option name.

    Returns:
        str: The lowercase option name.
    """
    option = option.strip().lower()
    if '=' in option:
        return option.split('=', 1)[0].strip()
    return next((name for name, members in (groups or dict()).items() if option in members), option)


def pop_route_options(text: str, keywords: tuple[str, ...]) -> tuple[str, list[str]]:
    """
    Remove all occurrences of the given keywords from a string of route keywords and return their options.

    Args:
        text (str): Route keywords, e.g., ``'int=dkh2 scf=(tight)'``.
        keywords (tuple[str, ...]): The keyword names to remove, e.g., ``('int', 'integral')``.

    Returns:
        tuple[str, list[str]]: The text without the keywords, and their options in order of appearance.
    """
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(keywords) + r')(?:=(\([^)]*\)|[^\s()]+)|(\([^)]*\)))',
                         re.IGNORECASE)
    options = list()
    for match in pattern.finditer(text):
        options.extend(option for option in re.split(r'[,\s]+', (match.group(1) or match.group(2)).strip('()'))
                       if option)
    return pattern.sub('', text), options


def fold_route_options(option_lists: list[list[str]],
                       name_fn: Callable[[str], str] = get_route_option_name,
                       ) -> list[str]:
    """
    Fold lists of options of one route keyword into a single list in which each option name appears once.
    A later list overrides an earlier one on the same option name, keeping the position of the first occurrence.

    Args:
        option_lists (list[list[str]]): Lists of options, from lowest to highest precedence.
        name_fn (Callable, optional): Maps an option to the name under which it competes with others.

    Returns:
        list[str]: The folded options.
    """
    folded = dict()
    for options in option_lists:
        for option in options:
            folded[name_fn(option)] = option
    return list(folded.values())


def get_opt_option_name(option: str) -> str:
    """The name under which an ``Opt`` option competes with others (see ``get_route_option_name``)."""
    return get_route_option_name(option, OPT_GROUPS)


def get_scf_option_name(option: str) -> str:
    """The name under which an ``SCF`` option competes with others (see ``get_route_option_name``)."""
    return get_route_option_name(option, SCF_GROUPS)


def get_guess_option_name(option: str) -> str:
    """The name under which a ``Guess`` option competes with others (see ``get_route_option_name``)."""
    return get_route_option_name(option, GUESS_GROUPS)


def pop_integral_options(text: str) -> tuple[str, list[str]]:
    """
    Remove all ``Integral`` (``int``) keywords from a string of route keywords and return their options.

    Args:
        text (str): Route keywords, e.g., ``'int=dkh2 scf=(tight)'``.

    Returns:
        tuple[str, list[str]]: The text without the ``Integral`` keywords, and their options in order of appearance.
    """
    return pop_route_options(text, ('int', 'integral'))


def fold_integral_options(*option_lists: list[str]) -> str:
    """
    Fold lists of ``Integral`` options into a single ``integral=(...)`` keyword.
    A later list overrides an earlier one on the same option name, keeping the position of the first occurrence.

    Args:
        *option_lists (list[str]): Lists of options, from lowest to highest precedence.

    Returns:
        str: The single ``Integral`` keyword, or an empty string if there are no options.
    """
    folded = fold_route_options(list(option_lists), get_integral_option_name)
    return f"integral=({', '.join(folded)})" if folded else ''


# job_type_1: '' for sp, irc, or composite methods, 'opt=calcfc', 'opt=(calcfc,ts,noeigen)',
# job_type_2: '' or 'freq iop(7/33=1)' (cannot be combined with CBS-QB3)
#             'scf=(tight,direct) int=finegrid irc=(rcfc,forward,maxpoints=100,stepsize=10) geom=check' for irc f
#             'scf=(tight,direct) int=finegrid irc=(rcfc,reverse,maxpoints=100,stepsize=10) geom=check' for irc r
# scan: '\nD 3 1 5 8 S 36 10.000000' (with the line break)
# restricted: '' or 'u' for restricted / unrestricted
# `iop(2/9=2000)` makes Gaussian print the geometry in the input orientation even for molecules with more
#   than 50 atoms (important so it matches the hessian, and so that Arkane can parse the geometry)
input_template = """${checkfile}
%%mem=${memory}mb
%%NProcShared=${cpus}

#P ${job_type_1} ${restricted}${method}${slash_1}${basis}${slash_2}${auxiliary_basis} ${job_type_2} ${fine} IOp(2/9=2000) ${keywords} ${dispersion} ${trsh}

${label}

${charge} ${multiplicity}
${xyz}${scan}${scan_trsh}${block}


"""


class GaussianAdapter(JobAdapter):
    """
    A class for executing Gaussian jobs.

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
        xyz (dict | list[dict], optional): The 3D coordinates to use. If not give, species.get_xyz() will be used.
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
                 xyz: dict | list[dict] | None = None,
                 ):

        self.incore_capacity = 1
        self.job_adapter = 'gaussian'
        self.execution_type = execution_type or 'queue'
        self.command = ['g16', 'g09', 'g03']
        self.url = 'https://gaussian.com/'

        if species is None:
            raise ValueError('Cannot execute Gaussian without an ARCSpecies object.')

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
        if self.checkfile is None and species_may_read_previous_orbitals(self.species[0]):
            if os.path.isfile(os.path.join(self.local_path, self.check_file_name)):
                self.checkfile = self.readable_checkfile(os.path.join(self.local_path, self.check_file_name))
            elif self.species[0].checkfile is not None:
                self.checkfile = self.readable_checkfile(self.species[0].checkfile)

    def _user_requested_verytight(self) -> bool:
        """
        Return True if the user passed ``verytight`` through ``self.args``.

        When True, the fine-mode opt-keyword builder will skip auto-adding
        ``tight`` so the user's ``verytight`` wins unambiguously in the final
        ``opt=(...)`` clause assembled by ``combine_parameters``.

        Scope: only the ``keyword`` channel is meaningful here — ``block`` is for
        Gaussian input blocks and ``trsh`` is for ARC's troubleshooting layer,
        neither of which is the right place for a user opt-cutoff override.

        Notes for the implementer:
            - ``self.args['keyword']`` is a ``dict[str, str]`` (see
              ``Level._check_args`` at arc/level.py:281). Values may be in any
              case and may contain other keywords too (e.g. ``"opt=(verytight)
              freq=hpmodes"``).
            - Match must be word-bounded: a stray ``verytightscf`` should not
              count, and ``tight`` alone must NOT match.
            - Be defensive: ``self.args`` or ``self.args['keyword']`` may be
              missing or empty depending on how the Level was constructed.
        """
        keyword_args = (self.args or {}).get('keyword') or {}
        joined = ' '.join(str(v) for v in keyword_args.values())
        return re.search(r'\bverytight\b', joined, re.IGNORECASE) is not None

    @staticmethod
    def _fold_route_keyword(input_dict: dict,
                            keys: list[str],
                            keywords: tuple[str, ...],
                            name_fn: Callable[[str], str],
                            ) -> tuple[dict, list[str]]:
        """
        Remove a route keyword from the given entries of ``input_dict`` and fold its options into one list.
        The entries are read in the order given, so a later entry overrides an earlier one on the same option name.

        Args:
            input_dict (dict): The input dictionary.
            keys (list[str]): The entries to read, from lowest to highest precedence.
            keywords (tuple[str, ...]): The names of the route keyword, e.g., ``('scf',)``.
            name_fn (Callable): Maps an option to the name under which it competes with others.

        Returns:
            tuple[dict, list[str]]: A copy of ``input_dict`` without the keyword, and the folded options.
        """
        input_dict = input_dict.copy()
        option_lists = list()
        for key in keys:
            input_dict[key], options = pop_route_options(input_dict[key], keywords)
            option_lists.append(options)
        return input_dict, fold_route_options(option_lists, name_fn)

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.

        Each of the ``integral``, ``opt``, ``scf`` and ``guess`` keywords is written once per route. Where an option
        is given more than once, the precedence is ARC's job default, then the level's option, then a troubleshooting
        remedy, and the highest wins. ARC's job defaults are collected apart from the troubleshooting remedies.
        """
        input_dict = dict()
        for key in ['block',
                    'dispersion',
                    'arc_defaults',
                    'fine',
                    'job_type_1',
                    'job_type_2',
                    'keywords',
                    'restricted',
                    'scan',
                    'slash_1',
                    'slash_2',
                    'trsh'
                    ]:
            input_dict[key] = ''
        update_input_dict_with_args(args=self.args, input_dict=input_dict)
        input_dict['auxiliary_basis'] = self.level.auxiliary_basis or ''
        input_dict['basis'] = re.sub('def2-', 'def2', self.level.basis.lower()) if self.level.basis else ''
        input_dict['charge'] = self.charge
        input_dict['checkfile'] = '%chk=check.chk'
        input_dict['cpus'] = self.cpu_cores
        input_dict['label'] = self.species_label
        input_dict['memory'] = self.input_file_memory
        input_dict['method'] = 'cbs-qb3' if self.level.method == 'cbs-qb3-paraskevas' else self.level.method
        input_dict['multiplicity'] = self.multiplicity
        input_dict['scan_trsh'] = self.args['keyword']['scan_trsh'] if 'scan_trsh' in self.args['keyword'] else ''
        acc2e_options = ['Acc2E=14'] if 'Acc2E=14' in input_dict['trsh'] else list()
        input_dict['trsh'], trsh_options = pop_integral_options(input_dict['trsh'])
        grid_remedies = [method.split('int=grid=', 1)[1] for method in (self.ess_trsh_methods or list())
                         if method.startswith('int=grid=')]
        trsh_integral_options = acc2e_options + trsh_options + [f'grid={grid}' for grid in grid_remedies[:1]]
        input_dict['keywords'], level_integral_options = pop_integral_options(input_dict.get('keywords', ''))
        integral = fold_integral_options(['grid=ultrafine', 'Acc2E=12'], level_integral_options, trsh_integral_options)
        input_dict['xyz'] = [xyz_to_str(xyz) for xyz in self.xyz] if self.run_multi_species else xyz_to_str(self.xyz)

        if self.level.basis is not None:
            input_dict['slash_1'] = '/'
            if self.level.auxiliary_basis is not None:
                input_dict['slash_2'] = '/'

        if self.level.method_type in ['semiempirical', 'force_field']:
            self.checkfile = None

        if self.level.dispersion is not None:
            input_dict['dispersion'] = self.level.dispersion

        if self.level.method[:2] == 'ro':
            self.add_to_args(val='use=L506')
        elif 'no_xqc' not in self.ess_trsh_methods \
                and not('no_xqc' in list(self.args['trsh'].values())) and 'qc' in input_dict['trsh']:
            # xqc will do qc (quadratic convergence) if the job fails w/o it, so use it by default.
            # replace qc with xqc if it's not already there.
            # 'no_xqc' in ess_trsh_methods records that the xqc algorithm itself already failed
            # (Gaussian l508), so don't upgrade qc to xqc in that case (see arc.job.trsh.trsh_keyword_no_qc).
            input_dict['trsh'] = input_dict['trsh'].replace('qc', 'xqc')

        # Job type specific options
        max_c = 100
        if 'trsh' in self.args and 'trsh' in self.args['trsh']:
            for item in self.args['trsh']['trsh']:
                match = re.search(r'maxcycle=(\d+)', item)
                if match:
                    max_c = int(match.group(1))
                    break

        if self.job_type in ['opt', 'conf_opt', 'optfreq', 'composite']:
            keywords = ['ts', 'calcfc', 'noeigentest', f'maxcycle={max_c}'] if self.is_ts else ['calcfc']
            if self.level.method in ['rocbs-qb3']:
                # There are no analytical 2nd derivatives (FC) for this method.
                keywords = ['ts', 'noeigentest', f'maxcycle={max_c}'] if self.is_ts else []
            if self.fine:
                if self.level.method_type in ['dft', 'composite']:
                    # Note that the Acc2E argument is not available in Gaussian03
                    input_dict['fine'] = integral
                    input_dict['arc_defaults'] += ' scf=(tight,direct)'
                # 'no_tight' is set by trsh_keyword_loose_disp when a previous attempt hit
                # MaxOptCycles with forces converged but displacement criteria unreachable.
                # 'tight' is also dropped when the user requested verytight, which replaces it.
                drop_tight = 'no_tight' in self.ess_trsh_methods or self._user_requested_verytight()
                tight_kw = [] if drop_tight else ['tight']
                if self.is_ts:
                    keywords.extend([*tight_kw, 'maxstep=5'])
                else:
                    keywords.extend([*tight_kw, 'maxstep=5', f'maxcycle={max_c}'])
            input_dict['job_type_1'] = "opt" if self.level.method_type not in ['dft', 'composite', 'wavefunction']\
                else f"opt=({', '.join(key for key in keywords)})"

        elif self.job_type == 'freq':
            input_dict['job_type_2'] = f'freq IOp(7/33=1) scf=(tight, direct) {integral}'

        elif self.job_type == 'optfreq':
            input_dict['job_type_2'] = 'freq IOp(7/33=1)'

        elif self.job_type in ['sp', 'conf_sp']:
            input_dict['job_type_1'] = integral
            input_dict['arc_defaults'] += ' scf=(tight, direct)'

        elif self.job_type == 'stability':
            input_dict['job_type_1'] = f'{STABILITY_KEYWORD} {integral}'
            input_dict['arc_defaults'] += ' scf=(tight, direct)'

        elif self.job_type == 'scan':
            scans, scans_strings = list(), list()
            if self.rotor_index is not None and self.species[0].rotors_dict:
                scans = self.species[0].rotors_dict[self.rotor_index]['scan']
            elif len(self.torsions):
                scans = torsions_to_scans(self.torsions)
            else:
                raise ValueError(f'Could not determine scan parameters for scan job {self.job_name}')
            scans = [scans] if not isinstance(scans[0], list) else scans
            for scan_indices in scans:
                scans_strings.append(' '.join([str(atom_index) for atom_index in scan_indices]))
            if self.torsions is None or not len(self.torsions):
                self.torsions = torsions_to_scans(scans, direction=-1)

            ts = 'ts, ' if self.is_ts else ''
            input_dict['job_type_1'] = f'opt=({ts}modredundant, calcfc, noeigentest, maxStep=5){integral}'
            input_dict['arc_defaults'] += ' scf=(tight, direct)'
            input_dict['scan'] = '\n\n' if not input_dict['scan'] else input_dict['scan']
            for scan in scans_strings:
                input_dict['scan'] += f'D {scan} S {int(360 / self.scan_res)} {self.scan_res:.1f}\n'

        elif self.job_type == 'irc':
            if self.fine:
                # Note that the Acc2E argument is not available in Gaussian03
                input_dict['fine'] = integral
                input_dict['arc_defaults'] += ' scf=(direct)'
            input_dict['job_type_1'] = f'irc=(CalcAll, {self.irc_direction}, maxpoints=50, stepsize=7)'

        for constraint_tuple in self.constraints:
            constraint_type = constraint_type_dict[len(constraint_tuple[0])]
            constraint_atom_indices = ' '.join([str(atom_index) for atom_index in constraint_tuple[0]])
            input_dict['scan'] = '\n\n' if not input_dict['scan'] else input_dict['scan']
            input_dict['scan'] += f"{constraint_type} {constraint_atom_indices} ={constraint_tuple[1]:.2f} B\n" \
                                  f"{constraint_type} {constraint_atom_indices} F\n"

        if self.level.solvation_method is not None:
            input_dict['job_type_1'] += f' SCRF=({self.level.solvation_method}, Solvent={self.level.solvent})'

        input_dict['keywords'], level_guess_options = pop_route_options(input_dict['keywords'], ('guess',))
        input_dict['trsh'], trsh_guess_options = pop_route_options(input_dict['trsh'], ('guess',))
        if self.species[0].number_of_atoms > 1:
            guess_options = fold_route_options([level_guess_options, trsh_guess_options], get_guess_option_name)
            if self.checkfile is not None and os.path.isfile(self.checkfile):
                default_guess = 'read'
            elif any(spc.is_ts or (spc.multiplicity == 1 and spc.number_of_radicals is not None
                                   and spc.number_of_radicals > 1)
                     for spc in self.species):
                # guess=mix seeds a broken-symmetry unrestricted wavefunction. It is wanted for TSs and
                # open-shell singlets (biradicals) only; restricted closed-shell species and high-spin
                # radicals converge to the correct state from the default guess.
                default_guess = 'mix'
            else:
                default_guess = None
            always = any(option.lower() == 'always' for option in guess_options)
            if default_guess is not None \
                    and not any(get_guess_option_name(option) == 'source' for option in guess_options) \
                    and not (default_guess == 'read' and always):
                guess_options = [default_guess] + [option for option in guess_options
                                                   if option.lower() != default_guess]
            if guess_options:
                if input_dict['job_type_1']:
                    input_dict['job_type_1'] += ' '
                guess = f"guess={guess_options[0]}" if len(guess_options) == 1 \
                    else f"guess=({','.join(guess_options)})"
                input_dict['job_type_1'] += guess if trsh_guess_options else f' {guess}'
        elif level_guess_options or trsh_guess_options:
            guess_options = fold_route_options([level_guess_options, trsh_guess_options], get_guess_option_name)
            input_dict['keywords'] = ' '.join(part for part in [input_dict['keywords'],
                                                                f"guess=({','.join(guess_options)})"] if part)

        if not any(integral in input_dict[key] for key in ['job_type_1', 'job_type_2', 'fine']):
            input_dict['keywords'] = ' '.join(
                part for part in [input_dict['keywords'],
                                  fold_integral_options(level_integral_options, trsh_integral_options)] if part)

        route_keys = ['arc_defaults', 'job_type_1', 'job_type_2', 'fine', 'keywords', 'trsh']
        input_dict, opt_options = self._fold_route_keyword(input_dict, route_keys, ('opt',), get_opt_option_name)
        if opt_options:
            if 'calcall' in opt_options:
                opt_options = [p for p in opt_options if p != 'calcfc' and not p.startswith('recalcfc')]
            elif any(p.startswith('recalcfc') for p in opt_options):
                opt_options = [p for p in opt_options if p != 'calcfc']
            input_dict['job_type_1'] = f"opt=({','.join(sorted(opt_options))}) {input_dict['job_type_1']}"

        no_xqc = 'no_xqc' in (self.ess_trsh_methods or list()) or 'no_xqc' in list(self.args['trsh'].values())
        input_dict, scf_options = self._fold_route_keyword(input_dict, route_keys, ('scf',), get_scf_option_name)
        if no_xqc:
            scf_options = [option for option in scf_options if option.lower() not in SCF_ALGORITHM_NAMES]
        if scf_options:
            input_dict['trsh'] += f" scf=({','.join(sorted(scf_options))})"

        # Fix IRC
        terms_irc = [r'irc=\((.*?)\)', r'irc=(\w+)']
        input_dict, parameters_irc = combine_parameters(input_dict, terms_irc)
        if parameters_irc:
            input_dict['job_type_1'] = f"irc=({','.join(parameters_irc)}) {input_dict['job_type_1']}"
        
        # Remove double spaces
        input_dict['job_type_1'] = input_dict['job_type_1'].replace('  ', ' ')
        
        input_file = ''
        input_dict_origin = input_dict.copy()

        restricted_list_bool = is_restricted(self)
        restricted_list = ["" if flag else 'u' for flag in ([restricted_list_bool]
                           if isinstance(restricted_list_bool, bool) else restricted_list_bool)]

        with open(os.path.join(self.local_path, input_filenames[self.job_adapter]), 'w') as f:
            if not self.run_multi_species:
                input_dict['restricted'] = restricted_list[0]
                f.write(Template(input_template).render(**input_dict))
            else:
                for index, spc in enumerate(self.species):
                    input_dict['charge'] = input_dict_origin['charge'][index]
                    input_dict['label'] = input_dict_origin['label'][index]
                    input_dict['multiplicity'] = input_dict_origin['multiplicity'][index]
                    input_dict['xyz'] = input_dict_origin['xyz'][index]
                    input_dict['restricted'] = restricted_list[index]
                    input_file += Template(input_template).render(**input_dict)
                    if index < len(self.species) - 1:
                        input_file += '\n--link1--\n'
                f.write(input_file)

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
        # 1.3. checkfile
        if self.checkfile is not None and os.path.isfile(self.checkfile):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='check.chk',
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
            # 2.3. checkfile
            self.files_to_download.append(self.get_file_property_dictionary(file_name='check.chk'))

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
        # Gaussian's %mem is the total memory budget for the process. ARC keeps
        # scheduler memory in MiB and intentionally gives Gaussian only part of
        # that total so the queue allocation retains headroom. This matters most
        # on capped nodes: e.g., a human "10 GB" node is only ~9.31 GiB, and if
        # ARC already requests ~95% of a node, passing the entire allocation to
        # %mem leaves too little room for runtime overhead and can trigger galloc.
        submit_script_memory_mib = self.submit_script_memory_mib or math.ceil(self.job_memory_gb * 1024)
        headroom_fraction = get_memory_headroom_fraction(self.ess_trsh_methods)
        self.input_file_memory = max(1, math.floor(submit_script_memory_mib * headroom_fraction))

    def execute_incore(self):
        """
        Execute a job incore.
        """
        binary = which(self.command,
                       return_bool=False,
                       raise_error=True,
                       raise_msg=f'Please install {self.job_adapter}, see {self.url} for more information.',
                       )
        binary_name = os.path.basename(binary)
        self._log_job_execution()
        commands = [cmd.replace('g16', binary_name) for cmd in incore_commands[self.job_adapter]]
        execute_command(commands)

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        self.legacy_queue_execution()


register_job_adapter('gaussian', GaussianAdapter)
