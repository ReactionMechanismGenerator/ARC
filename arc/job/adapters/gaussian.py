"""
An adapter for executing Gaussian jobs

https://gaussian.com/
"""

import datetime
import math
import os
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from mako.template import Template

from arc.common import get_logger, torsions_to_scans
from arc.imports import incore_commands, settings
from arc.job.adapter import JobAdapter, constraint_type_dict
from arc.job.adapters.common import (_initialize_adapter,
                                     is_restricted,
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

default_job_settings, global_ess_settings, input_filenames, output_filenames, servers, submit_filenames = \
    settings['default_job_settings'], settings['global_ess_settings'], settings['input_filenames'], \
    settings['output_filenames'], settings['servers'], settings['submit_filenames']


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
        run_multi_species (bool, optional): Whether to run a job for multiple species in the same input file.
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
        xyz (Union[dict,List[dict]], optional): The 3D coordinates to use. If not give, species.get_xyz() will be used.
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
                 xyz: Optional[Union[dict,List[dict]]] = None,
                 ):

        self.incore_capacity = 1
        self.job_adapter = 'gaussian'
        self.execution_type = execution_type or 'queue'
        self.command = ['g03', 'g09', 'g16']
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
        if isinstance(self.level, Level) and self.level.basis is not None:
            self.level.basis = re.sub('def2-', 'def2', self.level.basis.lower())

        if self.checkfile is None:
            if os.path.isfile(os.path.join(self.local_path, 'check.chk')):
                self.checkfile = os.path.join(self.local_path, 'check.chk')
            elif self.species[0].checkfile is not None and os.path.isfile(self.species[0].checkfile):
                self.checkfile = self.species[0].checkfile

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.
        """
        input_dict = dict()
        for key in ['block',
                    'dispersion',
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
        input_dict['basis'] = self.level.basis or ''
        input_dict['charge'] = self.charge
        input_dict['checkfile'] = '%chk=check.chk'
        input_dict['cpus'] = self.cpu_cores
        input_dict['label'] = self.species_label
        input_dict['memory'] = self.input_file_memory
        input_dict['method'] = self.level.method
        input_dict['multiplicity'] = self.multiplicity
        input_dict['scan_trsh'] = self.args['keyword']['scan_trsh'] if 'scan_trsh' in self.args['keyword'] else ''
        integral_algorithm = 'Acc2E=14' if 'Acc2E=14' in input_dict['trsh'] else 'Acc2E=12'
        input_dict['trsh'] = input_dict['trsh'].replace('int=(Acc2E=14)', '') if 'Acc2E=14' in input_dict['trsh'] else input_dict['trsh']
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
        elif not('no_xqc' in list(self.args['trsh'].values())) and 'qc' in input_dict['trsh']:
            # xqc will do qc (quadratic convergence) if the job fails w/o it, so use it by default.
            # replace qc with xqc if it's not already there
            input_dict['trsh'] = input_dict['trsh'].replace('qc', 'xqc')

        if self.level.method == 'cbs-qb3-paraskevas':
            # convert cbs-qb3-paraskevas to cbs-qb3
            self.level.method = 'cbs-qb3'

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
                    input_dict['fine'] = f'integral=(grid=ultrafine, {integral_algorithm})'
                    # input_dict['trsh'] may have scf=(...) in it, so we need to add the tight and direct keywords to it
                    scf_start = input_dict['trsh'].find('scf=(')
                    scf_end = input_dict['trsh'].find(')', scf_start)
                    scf_fine_content = 'tight,direct'
                    if scf_start != -1 and scf_end != -1:
                        scf_content = input_dict['trsh'][scf_start + 5:scf_end]
                        scf_fine_content += ',' + scf_content
                        # Remove ')' if it's the last character in the scf content
                        if scf_fine_content == ')':
                            scf_fine_content = scf_fine_content[:-1]
                        input_dict['trsh'] = input_dict['trsh'][:scf_start + 4] + '(' + scf_fine_content + ')' + input_dict['trsh'][scf_end + 1:]
                    else:
                        if input_dict['trsh']:
                            input_dict['trsh'] += ' '
                        input_dict['trsh'] += 'scf=(tight,direct)'
                if self.is_ts:
                    keywords.extend(['tight', 'maxstep=5'])
                else:
                    keywords.extend(['tight', 'maxstep=5', f'maxcycle={max_c}'])
            input_dict['job_type_1'] = "opt" if self.level.method_type not in ['dft', 'composite', 'wavefunction']\
                else f"opt=({', '.join(key for key in keywords)})"

        elif self.job_type == 'freq':
            input_dict['job_type_2'] = f'freq IOp(7/33=1) scf=(tight, direct) integral=(grid=ultrafine, {integral_algorithm})'

        elif self.job_type == 'optfreq':
            input_dict['job_type_2'] = 'freq IOp(7/33=1)'

        elif self.job_type == 'sp':
            input_dict['job_type_1'] = f'integral=(grid=ultrafine, {integral_algorithm})'
            if input_dict['trsh']:
                input_dict['trsh'] += ' '
            input_dict['trsh'] += 'scf=(tight, direct)'

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
            input_dict['job_type_1'] = f'opt=({ts}modredundant, calcfc, noeigentest, maxStep=5)' \
                                       f'integral=(grid=ultrafine, {integral_algorithm})'
            if input_dict['trsh']:
                input_dict['trsh'] += ' '
            input_dict['trsh'] += 'scf=(tight, direct)'
            input_dict['scan'] = '\n\n' if not input_dict['scan'] else input_dict['scan']
            for scan in scans_strings:
                input_dict['scan'] += f'D {scan} S {int(360 / self.scan_res)} {self.scan_res:.1f}\n'

        elif self.job_type == 'irc':
            if self.fine:
                # Note that the Acc2E argument is not available in Gaussian03
                input_dict['fine'] = f'integral=(grid=ultrafine, {integral_algorithm})'
                # We need to add scf=(direct) to the trsh argument
                # But we to check if it's already there, and
                if 'direct' not in input_dict['trsh']:
                    scf_start = input_dict['trsh'].find('scf=(')
                    scf_end = input_dict['trsh'].find(')', scf_start)
                    scf_fine_content = 'direct'
                    if scf_start != -1 and scf_end != -1:
                        scf_content = input_dict['trsh'][scf_start + 5:scf_end]
                        scf_fine_content += ',' + scf_content
                        # Remove ')' if it's the last character in the scf content
                        if scf_fine_content == ')':
                            scf_fine_content = scf_fine_content[:-1]
                        input_dict['trsh'] = input_dict['trsh'][:scf_start + 4] + '(' + scf_fine_content + ')' + input_dict['trsh'][scf_end + 1:]
                    else:
                        if input_dict['trsh']:
                            input_dict['trsh'] += ' '
                        input_dict['trsh'] += 'scf=(direct)'
                
            input_dict['job_type_1'] = f'irc=(CalcAll, {self.irc_direction}, maxpoints=50, stepsize=7)'

        for constraint_tuple in self.constraints:
            constraint_type = constraint_type_dict[len(constraint_tuple[0])]
            constraint_atom_indices = ' '.join([str(atom_index) for atom_index in constraint_tuple[0]])
            input_dict['scan'] = '\n\n' if not input_dict['scan'] else input_dict['scan']
            input_dict['scan'] += f"{constraint_type} {constraint_atom_indices} ={constraint_tuple[1]:.2f} B\n" \
                                  f"{constraint_type} {constraint_atom_indices} F"

        if self.level.solvation_method is not None:
            input_dict['job_type_1'] += f' SCRF=({self.level.solvation_method}, Solvent={self.level.solvent})'

        if self.species[0].number_of_atoms > 1:
            if input_dict['job_type_1']:
                input_dict['job_type_1'] += ' '
            if 'guess=INDO' in input_dict['trsh']:
                input_dict['job_type_1'] += 'guess=INDO'
                input_dict['trsh'] = input_dict['trsh'].replace('guess=INDO', '')
            else:
                input_dict['job_type_1'] += ' guess=read' if self.checkfile is not None and os.path.isfile(self.checkfile) \
                    else ' guess=mix'

        # Fix OPT
        terms_opt = [r'opt=\((.*?)\)', r'opt=(\w+)']
        input_dict, parameters_opt = combine_parameters(input_dict, terms_opt)
        # If 'opt' parameters are found, concatenate and reinsert them
        if parameters_opt:
            # Remove duplicate parameters
            combined_opt_params = ','.join(parameters_opt)
            input_dict['job_type_1'] = f"opt=({combined_opt_params}) {input_dict['job_type_1']}"

        #Fix SCF
        # This may be redundant due to additional fixes in the above code
        terms = [r'scf=\((.*?)\)', r'scf=(\w+)']
        input_dict, parameters = combine_parameters(input_dict, terms)
        if parameters:
            input_dict['trsh'] += f" scf=({','.join(parameters)})"

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
        # Gaussian's memory is in MB, total for all cpu cores
        self.input_file_memory = math.ceil(self.job_memory_gb * 1024)

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


register_job_adapter('gaussian', GaussianAdapter)
