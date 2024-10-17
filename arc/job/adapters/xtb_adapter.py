"""
An adapter for executing xTB (Semiempirical Extended Tight-Binding Program Package) jobs

https://github.com/grimme-lab/xtb
https://xtb-docs.readthedocs.io/en/latest/contents.html
run types: https://xtb-docs.readthedocs.io/en/latest/commandline.html?highlight=--grad#runtypes
"""

import datetime
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from mako.template import Template

from arc.common import get_logger, read_yaml_file, save_yaml_file, torsions_to_scans
from arc.imports import incore_commands, settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter
from arc.job.factory import register_job_adapter
from arc.job.local import execute_command
from arc.level import Level
from arc.species.converter import species_to_sdf_file, xyz_from_data, xyz_to_coords_and_element_numbers
from arc.species.vectors import calculate_dihedral_angle

if TYPE_CHECKING:
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()

input_filenames, output_filenames, servers, submit_filenames, XTB = \
    settings['input_filenames'], settings['output_filenames'], \
    settings['servers'], settings['submit_filenames'], settings['XTB']


input_template = """#!/usr/bin/env python3
# encoding: utf-8

import pandas as pd
import yaml
from ase import Atoms
from sella import Sella
from xtb.ase.calculator import XTB

coords = ${coords}
numbers = ${numbers}
atoms = Atoms(positions=coords, numbers=numbers)
atoms.calc = XTB(method='${method}')

opt = Sella(
    atoms,
    logfile='${logfile}',
    trajectory='${trajectory}',
)
opt.run(${fmax}, ${steps})

coords = opt.atoms.positions
energy = float(pd.read_csv('${logfile}').iloc[-1].values[0].split()[3])

yaml_str = yaml.dump(data={'coords': coords.tolist(),
                           'sp': energy,
                           })
with open('output.yml', 'w') as f:
    f.write(yaml_str)

"""


class xTBAdapter(JobAdapter):
    """
    A class for executing xTB jobs.

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

        self.incore_capacity = 100
        self.job_adapter = 'xtb'
        self.execution_type = execution_type or 'incore'
        self.command = XTB
        self.long_command = None
        self.url = 'https://github.com/grimme-lab/xtb'

        if species is None:
            raise ValueError('Cannot execute xTB without an ARCSpecies object.')

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
        directives, block = '', ''
        uhf = self.species[0].number_of_radicals or self.multiplicity - 1

        if self.job_type in ['opt', 'conf_opt', 'scan']:
            directives += ' --opt'
            directives += self.add_accuracy()
            if self.constraints and self.job_type != 'scan':
                block += f'$constrain\n'
                block += self.add_constraints_to_block()
                block += f'$opt\n   maxcycle=5\n$end'

        elif self.job_type in ['freq']:
            directives = ' --hess'

        elif self.job_type in ['optfreq']:
            directives += ' --ohess'
            directives += self.add_accuracy()

        elif self.job_type in ['fukui']:
            directives += ' --vfukui'

        elif self.job_type == 'sp':
            pass

        directives += f' --{self.level.method}' if self.level is not None and self.level.method != 'xtb' else ' --gfn2'

        if self.level is not None and self.level.solvent is not None:
            directives += f' --alpb {self.level.solvent.lower()}'

        if self.job_type in ['scan']:
            scans = list()
            if self.rotor_index is not None:
                if self.species[0].rotors_dict \
                        and self.species[0].rotors_dict[self.rotor_index]['directed_scan_type'] == 'ess':
                    scans = self.species[0].rotors_dict[self.rotor_index]['scan']
                    scans = [scans] if not isinstance(scans[0], list) else scans
            elif self.torsions is not None and len(self.torsions):
                scans = torsions_to_scans(self.torsions)

            force_constant = 0.05
            block += f'$constrain\n   force constant={force_constant}\n'
            block += self.add_constraints_to_block()
            block += '$scan\n'
            for scan in scans:
                scan_string = '   dihedral: '
                dihedral_angle = int(calculate_dihedral_angle(coords=self.xyz, torsion=scan, index=1))
                scan_string += ', '.join([str(atom_index) for atom_index in scan])
                scan_string += f', {dihedral_angle}; {dihedral_angle}, {dihedral_angle + 360.0}, {int(360 / self.scan_res)}\n'
                block += scan_string
            block += '$end'

        if block:
            directives += ' --input input.in'
            with open(os.path.join(self.local_path, 'input.in'), 'w') as f:
                f.write(block)

        if 'parallel' not in self.args['keyword'].keys() or self.args['keyword']['parallel'] != 'no':
            # Use parallelization by default unless self.args['keyword']['parallel'] is False.
            directives += ' --parallel'

        uhf_charge = f'--uhf {uhf} --chrg {self.charge}'
        self.long_command = f'{self.command} mol.sdf{directives} {uhf_charge} > {output_filenames[self.job_adapter]}\n'
        
        if self.species[0].mol is None:
            self.species[0].mol_from_xyz()
        if not self.is_opt_ts_job() and self.species[0].mol is not None:
            species_to_sdf_file(species=self.species[0], path=self.sdf_path)
            with open(os.path.join(self.local_path, input_filenames[self.job_adapter]), 'w') as f:
                f.write(self.long_command)

    def is_opt_ts_job(self) -> bool:
        """
        Determine whether this is a transition state geometry optimization job.

        Returns:
            bool: Whether this is a transition state geometry optimization job.
        """
        if self.species is not None and len(self.species) and self.species[0].is_ts \
                and self.job_type in ['opt', 'conf_opt']:
            return True
        return False

    def opt_ts(self):
        """
        Optimize a transition state to a saddle point.
        """
        trajectory_path = os.path.join(self.local_path, 'ts.traj')
        logfile = os.path.join(self.local_path, 'ts.log')
        coords, z_list = xyz_to_coords_and_element_numbers(self.xyz)

        if self.level.method not in ['xtb2', 'gfn2']:
            raise ValueError(f'xTB TS optimization is only supported for the GFN2 level. Got {self.level.method}')

        input_dict = {'method': 'GFN2-xTB',
                      'coords': coords,
                      'numbers': z_list,
                      'logfile': logfile,
                      'trajectory': trajectory_path,
                      'fmax': 1e-3,
                      'steps': 1000,
                      }

        with open(self.sella_runner_path, 'w') as f:
            f.write(Template(input_template).render(**input_dict))

        execute_command([f'cd {self.local_path}'] + incore_commands['sella'], executable='/bin/bash')

        results = read_yaml_file(os.path.join(self.local_path_to_output_file))
        self.species[0].e_elect = results['sp']
        results['xyz'] = self.species[0].final_xyz = xyz_from_data(coords=results['coords'], numbers=z_list)
        save_yaml_file(self.local_path_to_output_file, results)
        os.remove(self.sella_runner_path)

    def add_constraints_to_block(self) -> str:
        """
        Add the constraints section to an xTB input file.

        Returns:
            str: The updated ``block``.
        """
        block = ''
        if self.constraints is not None:
            for constraint in self.constraints:
                if len(constraint[0]) == 2:
                    block += f'   distance: {constraint[0][0]}, {constraint[0][1]}, {constraint[1]}\n'
                if len(constraint[0]) == 3:
                    block += f'   angle: {constraint[0][0]}, {constraint[0][1]}, {constraint[0][2]}, {constraint[1]}\n'
                if len(constraint[0]) == 4:
                    block += f'   dihedral: {constraint[0][0]}, {constraint[0][1]}, {constraint[0][2]}, {constraint[0][3]}, {constraint[1]}\n'
        else:
            block += '\n'
        return block

    def add_accuracy(self) -> str:
        """
        Add an accuracy level to the job specifications.

        Returns:
            str: The updated ``job_type``.
        """
        directive = ''
        if self.fine:
            directive += ' vtight'
        elif self.args and 'keyword' in self.args.keys() and 'accuracy' in self.args['keyword'].keys():
            # Accuracy threshold level ('crude', 'sloppy', 'loose', 'lax', 'normal', 'tight', 'vtight', 'extreme').
            directive += f' {self.args["keyword"]["accuracy"]}'
        else:
            directive += ' tight'
        return directive

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
        # 1.2. input files
        self.write_input_file()
        self.files_to_upload.append(self.get_file_property_dictionary(file_name=input_filenames[self.job_adapter]))
        self.files_to_upload.append(self.get_file_property_dictionary(file_name='input.in'))
        self.files_to_upload.append(self.get_file_property_dictionary(file_name='mol.sdf'))
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
            # 2.2. log file
            self.files_to_download.append(self.get_file_property_dictionary(
                file_name=output_filenames[self.job_adapter]))
            # 2.3. scan log file
            self.files_to_download.append(self.get_file_property_dictionary(file_name='xtbscan.log'))
            # 2.4. normal mode displacement log file
            self.files_to_download.append(self.get_file_property_dictionary(file_name='g98.out'))
            # 2.5. hessian log file
            self.files_to_download.append(self.get_file_property_dictionary(file_name='hessian'))

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        self.sdf_path = os.path.join(self.local_path, 'mol.sdf')
        self.sella_runner_path = os.path.join(self.local_path, 'sella_runner.py')
        if self.is_opt_ts_job():
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
        if self.is_opt_ts_job():
            # Use Sella with xTB via ASE
            self.opt_ts()
        else:
            execute_command([f'cd {self.local_path}'] + incore_commands[self.job_adapter], executable='/bin/bash')

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        self.legacy_queue_execution()


register_job_adapter('xtb', xTBAdapter)
