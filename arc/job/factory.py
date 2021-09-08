"""
A module for generating job adapters.
"""

from typing import TYPE_CHECKING, List, Optional, Type, Tuple, Union

from arc.exceptions import JobError
from arc.job.adapter import JobAdapter, JobEnum, JobTypeEnum
from arc.reaction import ARCReaction
from arc.species import ARCSpecies

if TYPE_CHECKING:
    import datetime
    from arc.level import Level

_registered_job_adapters = {}  # keys are JobEnum, values are JobAdapter subclasses


def register_job_adapter(job_adapter_label: str,
                         job_adapter_class: Type[JobAdapter],
                         ) -> None:
    """
    A register for job adapters.

    Args:
        job_adapter_label (str): A string representation for a job adapter.
        job_adapter_class (JobAdapter): The job adapter class (a child of JobAdapter).

    Raises:
        TypeError: If job_adapter_class is not a subclass of JobAdapter.
    """
    if not issubclass(job_adapter_class, JobAdapter):
        raise TypeError(f'Job adapter class {job_adapter_class} is not a subclass JobAdapter.')
    _registered_job_adapters[JobEnum(job_adapter_label.lower())] = job_adapter_class


def job_factory(job_adapter: str,
                project: str,
                project_directory: str,
                job_type: Optional[Union[List[str], str]] = None,
                args: Optional[Union[dict, str]] = None,
                bath_gas: Optional[str] = None,
                checkfile: Optional[str] = None,
                conformer: Optional[int] = None,
                constraints: Optional[List[Tuple[List[int], float]]] = None,
                cpu_cores: Optional[str] = None,
                dihedrals: Optional[List[float]] = None,
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
                reactions: Optional[List['ARCReaction']] = None,
                rotor_index: Optional[int] = None,
                server: Optional[str] = None,
                server_nodes: Optional[list] = None,
                species: Optional[List['ARCSpecies']] = None,
                testing: bool = False,
                times_rerun: int = 0,
                torsions: Optional[List[List[int]]] = None,
                tsg: Optional[int] = None,
                xyz: Optional[dict] = None,
                dihedral_increment: Optional[float] = None,
                ) -> JobAdapter:
    """
    A factory generating a job adapter corresponding to ``job_adapter``.

    Args:
        job_adapter (str): The string representation of the job adapter, validated against ``JobEnum``.
        project (str): The project's name. Used for setting the remote path.
        project_directory (str): The path to the local project directory.
        job_type (list, str): The job's type, validated against ``JobTypeEnum``. If it's a list, pipe.py will be called.
        args (dict, str, optional): Methods (including troubleshooting) to be used in input files.
                                    Keys are either 'keyword' or 'block', values are dictionaries with values to be used
                                    either as keywords or as blocks in the respective software input file. If given as
                                    a string, it will be converted to a dictionary format with ``'keyword'`` and
                                    ``'general'`` keys.
        bath_gas (str, optional): A bath gas. Currently only used in OneDMin to calculate L-J parameters. Allowed values
                                  are: ``'He'``, ``'Ne'``, ``'Ar'``, ``'Kr'``, ``'H2'``, ``'N2'``, or ``'O2'``.
        checkfile (str, optional): The path to a previous Gaussian checkfile to be used in the current job.
        conformer (int, optional): Conformer number if optimizing conformers.
        constraints (list, optional): A list of constraints to use during an optimization or scan.
                                      Entries are constraint tuples. The first entry in the tuple is a list of
                                      1-indexed atom indices (the list length determined the constraint type),
                                      and the second entry is the constraint value in Angstroms or degrees.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
                                   ARC adopts the following naming system to describe computing hardware hierarchy:
                                   node > cpu > cpu_cores > cpu_threads.
        dihedrals (List[float], optional): The dihedral angels corresponding to self.torsions.
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
                                     The job server status is in job_status[0] and can be either ``'initializing'``,
                                     ``'running'``, ``'errored'``, or ``'done'``. The job ESS status is in job_status[1]
                                     and it is a dict of: {'status': str, 'keywords': list, 'error': str, 'line': str}.
                                     The values of 'status' can be either ``'initializing'``, ``'running'``,
                                     ``'errored'``, ``'unconverged'``, or ``'done'``. If the status is ``'errored'``,
                                     then standardized error keywords, the error description and the identified error
                                     line from the ESS log file are given as well.
        level (Level): The level of theory to use.
        max_job_time (float, optional): The maximal allowed job time on the server in hours (can be fractional).
        reactions (List[ARCReaction], optional): Entries are ARCReaction instances, used for TS search methods.
        rotor_index (int, optional): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        server (str, optional): The server's name.
        server_nodes (list, optional): The nodes this job was previously submitted to.
        species (List[ARCSpecies], optional): Entries are ARCSpecies instances.
                                              Either ``reactions`` or ``species`` must be given.
        testing (bool, optional): Whether the object is generated for testing purposes, ``True`` if it is.
        times_rerun (int, optional): Number of times this job was re-run with the same arguments (no trsh methods).
        torsions (List[List[int]], optional): The 0-indexed atom indices of the torsion(s).
        tsg (int, optional): TSGuess number if optimizing TS guesses.
        xyz (dict, optional): The 3D coordinates to use. If not give, species.get_xyz() will be used.
        dihedral_increment (float, optional): The degrees increment to use when scanning dihedrals of TS guesses.

        # shift (str, optional): A string representation alpha- and beta-spin orbitals shifts (molpro only).  # use args
        # is_ts (bool): Whether this species represents a transition structure. Default: ``False``.  # use species
        # occ (int, optional): The number of occupied orbitals (core + val) from a molpro CCSD sp calc.
        # number_of_radicals (int, optional): The number of radicals (inputted by the user, ARC won't attempt to
        #                                     determine it). Defaults to None. Important, e.g., if a Species is a bi-rad
        #                                     singlet, in which case the job should be unrestricted with
        #                                     multiplicity = 1.  # use the species
        # radius (float, optional): The species radius in Angstrom.  # use the species
        # pivots (list, optional): The rotor scan pivots, if the job type is scan. Not used directly in these methods,
        #                          but used to identify the rotor.  # use scan[1:3]
        # scan_type (str, optional): The scan type. Either of: ``'ess'``, ``'brute_force_sp'``, ``'brute_force_opt'``,
        #                            ``'cont_opt'``, ``'brute_force_sp_diagonal'``, ``'brute_force_opt_diagonal'``,
        #                            ``'cont_opt_diagonal'``.

    Returns:
        JobAdapter: The requested JobAdapter subclass, initialized with the respective arguments.
    """

    if job_adapter not in _registered_job_adapters.keys():
        raise ValueError(f'The "job_adapter" argument of {job_adapter} was not present in the keys for the '
                         f'_registered_job_adapters dictionary: {list(_registered_job_adapters.keys())}'
                         f'\nPlease check that the job adapter was registered properly.')

    if reactions is None and species is None:
        raise JobError('Either reactions or species must be given, got neither.')
    if reactions is not None and any(not isinstance(reaction, ARCReaction) for reaction in reactions):
        raise JobError(f'The reactions argument must contain only ARCReaction instance entries, '
                       f'got types {[type(reaction) for reaction in reactions]}.')
    if species is not None and any(not isinstance(spc, ARCSpecies) for spc in species):
        raise JobError(f'The species argument must contain only ARCSpecies instance entries, '
                       f'got types {[type(spc) for spc in species]}.')
    if isinstance(args, str):
        args = {'keyword': {'general': args}}

    job_adapter = JobEnum(job_adapter.lower())

    job_adapter_class = _registered_job_adapters[job_adapter](project=project,
                                                              project_directory=project_directory,
                                                              job_type=JobTypeEnum(job_type).value,
                                                              args=args,
                                                              bath_gas=bath_gas,
                                                              checkfile=checkfile,
                                                              conformer=conformer,
                                                              constraints=constraints,
                                                              cpu_cores=cpu_cores,
                                                              dihedrals=dihedrals,
                                                              ess_settings=ess_settings,
                                                              ess_trsh_methods=ess_trsh_methods,
                                                              execution_type=execution_type,
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
                                                              species=species,
                                                              testing=testing,
                                                              times_rerun=times_rerun,
                                                              torsions=torsions,
                                                              tsg=tsg,
                                                              xyz=xyz,
                                                              )
    return job_adapter_class
