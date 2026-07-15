"""
An adapter for executing Gaussian QST2 transition-state search jobs.

QST2 is Gaussian's synchronous-transit-guided quasi-Newton method (``opt=qst2``)
that locates a TS from a reactant and a product geometry (two molecule
specifications in a single input file). It has been found to converge to TSs
that other ARC TS-search adapters miss (e.g., 1,2-halogen migrations).

https://gaussian.com/opt/
"""

import datetime
import os
from typing import TYPE_CHECKING

from mako.template import Template

from arc.common import get_logger
from arc.imports import incore_commands, settings
from arc.job.adapters.common import which
from arc.job.adapters.gaussian import GaussianAdapter
from arc.job.factory import register_job_adapter
from arc.job.local import execute_command
from arc.level import Level
from arc.parser.parser import parse_geometry
from arc.species import ARCSpecies, TSGuess
from arc.species.converter import xyz_to_str

if TYPE_CHECKING:
    from arc.reaction import ARCReaction


logger = get_logger()

input_filenames, output_filenames, servers, submit_filenames, qst2_settings = \
    settings['input_filenames'], settings['output_filenames'], settings['servers'], settings['submit_filenames'], \
    settings.get('qst2_settings', {})

input_template = """%%chk=check.chk
%%mem=${memory}mb
%%NProcShared=${cpus}

#P opt=(qst2,calcfc,noeigentest,maxcycle=${maxcycle}) freq ${method}/${basis} int=ultrafine nosymm

QST2 ${label} reactant

${charge} ${multiplicity}
${reactant_xyz}

QST2 ${label} product

${charge} ${multiplicity}
${product_xyz}

"""


class QST2Adapter(GaussianAdapter):
    """
    A class for executing Gaussian QST2 TS-search jobs.

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

        if reactions is None:
            raise ValueError('Cannot execute a QST2 TS search without an ARCReaction object.')

        if reactions and reactions[0].ts_species is None:
            # Create a dummy TS species from the reactants (or products).
            # This is a temporary placeholder to satisfy GaussianAdapter's species requirement.
            # The actual TS geometry will be obtained from the QST2 calculation.
            if reactions[0].r_species:
                reactions[0].ts_species = ARCSpecies(label=f'{reactions[0].label}_TS',
                                                     is_ts=True,
                                                     charge=reactions[0].charge,
                                                     multiplicity=reactions[0].multiplicity,
                                                     xyz=reactions[0].r_species[0].get_xyz())
            else:
                raise ValueError('ARCReaction object must contain reactant species to initialize a QST2 job.')

        level = level or qst2_settings.get('level', '')
        if not level:
            raise ValueError('A level of theory must be specified for QST2 jobs, either in the job arguments '
                             'or in the settings file.')
        species_for_super = [reactions[0].ts_species]
        super().__init__(project=project,
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
                         run_multi_species=run_multi_species,
                         reactions=reactions,
                         rotor_index=rotor_index,
                         server=server,
                         server_nodes=server_nodes,
                         queue=queue,
                         attempted_queues=attempted_queues,
                         species=species_for_super,
                         testing=testing,
                         times_rerun=times_rerun,
                         torsions=torsions,
                         tsg=tsg,
                         xyz=xyz,
                         )

        self.job_adapter = 'qst2'
        self.url = 'https://gaussian.com/opt/'
        # ``self.command`` is inherited from GaussianAdapter (['g16', 'g09', 'g03']),
        # since QST2 is executed by the Gaussian binary.
        self.local_path_to_output_file = os.path.join(self.local_path, output_filenames[self.job_adapter])
        self.execution_type = execution_type or 'queue'

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.
        A single Gaussian input file holding two molecule specifications
        (reactant block, then product block) is generated.
        """
        atom_map = self.reactions[0].atom_map
        if atom_map is None:
            raise ValueError('Cannot write a QST2 input file without an atom map in the reaction.')

        reactant_xyz = self.reactions[0].get_reactants_xyz(return_format=dict)
        product_xyz = self.reactions[0].get_products_xyz(return_format=dict)  # This implicitly uses the atom map.

        input_dict = {'memory': self.input_file_memory,
                      'cpus': self.cpu_cores,
                      'maxcycle': qst2_settings.get('maxcycle', 150),
                      'method': self.level.method,
                      'basis': self.level.basis,
                      'label': self.species_label,
                      'charge': self.charge,
                      'multiplicity': self.multiplicity,
                      'reactant_xyz': xyz_to_str(reactant_xyz),
                      'product_xyz': xyz_to_str(product_xyz),
                      }

        with open(os.path.join(self.local_path, input_filenames[self.job_adapter]), 'w') as f:
            f.write(Template(input_template).render(**input_dict))

    @property
    def ess_software(self) -> str:
        """QST2 is a TS-search adapter, but its output is a Gaussian log."""
        return 'gaussian'

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        super().set_input_file_memory()

    def write_submit_script(self) -> None:
        """
        Write a submit script to execute the job.
        """
        original_job_adapter = self.job_adapter
        self.job_adapter = 'gaussian'  # Temporarily change to 'gaussian' for submit script lookup.
        try:
            super().write_submit_script()
        finally:
            self.job_adapter = original_job_adapter  # Revert job_adapter.

    def process_run(self):
        """
        Process a completed QST2 run, parsing the optimized TS geometry into a TSGuess.
        """
        tsg = TSGuess(method='qst2',
                      index=len(self.reactions[0].ts_species.ts_guesses),
                      success=False,
                      t0=self.initial_time,
                      )
        if os.path.isfile(self.local_path_to_output_file):
            tsg.initial_xyz = parse_geometry(self.local_path_to_output_file)
            tsg.execution_time = self.final_time - self.initial_time
            tsg.log_path = self.local_path_to_output_file
            tsg.success = True
        self.reactions[0].ts_species.ts_guesses.append(tsg)

    def cleanup_files(self):
        """Remove unneeded files after run."""
        file_path = os.path.join(self.local_path, input_filenames[self.job_adapter])
        if os.path.exists(file_path):
            os.remove(file_path)

    def execute_incore(self):
        """
        Execute a job incore.
        """
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()
        binary = which(self.command,
                       return_bool=False,
                       raise_error=True,
                       raise_msg=f'Please install Gaussian, see {self.url} for more information.',
                       )
        binary_name = os.path.basename(binary)
        self._log_job_execution()
        commands = [cmd.replace('g16', binary_name) for cmd in incore_commands['gaussian']]
        execute_command([f'cd {self.local_path}'] + commands, executable='/bin/bash')
        self.final_time = datetime.datetime.now()
        self.process_run()

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        self.legacy_queue_execution()


register_job_adapter('qst2', QST2Adapter)
