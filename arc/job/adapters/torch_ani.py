"""
An adapter for executing TorhANI jobs

https://aiqm.github.io/torchani/
https://github.com/aiqm/torchani
ASE: https://wiki.fysik.dtu.dk/ase/index.html, https://core.ac.uk/download/84004505.pdf
"""

import datetime
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from ase.constraints import FixInternals
from ase.optimize import BFGS
from ase.optimize.sciopt import Converged, OptimizerConvergenceError, SciPyFminBFGS, SciPyFminCG
import torch
import torchani

from arc.common import convert_list_index_0_to_1, get_logger
from arc.imports import settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter
from arc.job.factory import register_job_adapter
from arc.level import Level
from arc.species.converter import hartree_to_si, xyz_from_data, xyz_to_ase, xyz_to_coords_and_element_numbers

if TYPE_CHECKING:
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()

default_job_settings, global_ess_settings, input_filenames, output_filenames, servers, submit_filenames = \
    settings['default_job_settings'], settings['global_ess_settings'], settings['input_filenames'], \
    settings['output_filenames'], settings['servers'], settings['submit_filenames']


class TorchANIAdapter(JobAdapter):
    """
    A class for executing TorchANI jobs.

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
                 ):

        self.incore_capacity = 100
        self.job_adapter = 'torchani'
        self.execution_type = execution_type or 'incore'
        self.command = None
        self.url = 'https://github.com/aiqm/torchani'

        self.sp = None
        self.opt_xyz = None
        self.freqs = None
        self.force = None

        if species is None:
            raise ValueError('Cannot execute TorchANI without an ARCSpecies object.')

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

    def run_sp(self):
        """
        Run a single-point energy calculation.
        """
        coords, z_list = xyz_to_coords_and_element_numbers(self.xyz)
        coordinates = torch.tensor([coords], requires_grad=True, device=self.device)
        species = torch.tensor([z_list], device=self.device)
        energy = self.model((species, coordinates)).energies
        self.sp = hartree_to_si(energy.item(), kilo=True)
        self.save_output_file(key='sp', val=self.sp)

    def run_force(self):
        """
        Compute the force matrix.
        """
        coords, z_list = xyz_to_coords_and_element_numbers(self.xyz)
        coordinates = torch.tensor([coords], requires_grad=True, device=self.device)
        species = torch.tensor([z_list], device=self.device)
        energy = self.model((species, coordinates)).energies
        derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
        force = -derivative
        self.force = force.squeeze().numpy()
        self.save_output_file(key='force', val=self.force.tolist())

    def run_opt(self,
                fmax: float = 0.001,
                steps: Optional[int] = None,
                engine: str = 'SciPyFminBFGS',
                ):
        """
        Run a geometry optimization calculation with optional constraints.
        The convergence criteria satisfied when the forces on all individual
        atoms are less than ``fmax`` or when the number of ``steps`` exceeds.

        Args:
            fmax (float, optional): The maximal force for convergence.
            steps (int, optional): The maximal number of steps for the optimization.
            engine (str, optional): The optimizer to use.
                                    'BFGS': Broyden–Fletcher–Goldfarb–Shanno. This algorithm chooses each step from
                                            the current atomic forces and an approximation of the Hessian matrix.
                                             The Hessian is established from an initial guess which is gradually
                                             improved as more forces are evaluated. Implemented in ASE.
                                    'SciPyFminBFGS': A Quasi-Newton method (Broydon-Fletcher-Goldfarb-Shanno).
                                                     An ASE interface to SciPy.
                                    'SciPyFminCG': A non-linear (Polak-Ribiere) conjugate gradient algorithm.
                                                   An ASE interface to SciPy.
        """
        self.opt_xyz = None
        calculator = torchani.models.ANI1ccx().ase()
        atoms = xyz_to_ase(self.xyz)
        atoms.set_calculator(calculator)

        if self.constraints is not None:
            bonds, angles, dihedrals = list(), list(), list()
            for constraint in self.constraints:
                atom_indices = convert_list_index_0_to_1(constraint[0], direction=-1)
                if len(atom_indices) == 2:
                    bonds.append([constraint[1], atom_indices])
                if len(atom_indices) == 3:
                    angles.append([constraint[1], atom_indices])
                if len(atom_indices) == 4:
                    dihedrals.append([constraint[1], atom_indices])
            constraints = FixInternals(bonds=bonds, angles_deg=angles, dihedrals_deg=dihedrals)
            atoms.set_constraint(constraints)

        engine_list = ['BFGS', 'SciPyFminBFGS', 'SciPyFminCG']
        engine_set = set([engine] + engine_list)
        engine_dict = {'BFGS': BFGS, 'SciPyFminBFGS': SciPyFminBFGS, 'SciPyFminCG': SciPyFminCG}
        for opt_engine_name in engine_set:
            opt_engine = engine_dict[opt_engine_name]
            opt = opt_engine(atoms, logfile=None)
            try:
                opt.run(fmax=fmax, steps=steps)
            except (Converged, NotImplementedError, OptimizerConvergenceError):
                pass
            else:
                break
        else:
            self.save_output_file(key='xyz', val=self.opt_xyz)  # Saved "xyz: None".
            return
        coords = atoms.get_positions()
        self.opt_xyz = xyz_from_data(coords=coords, symbols=self.xyz['symbols'])
        self.save_output_file(key='xyz', val=self.opt_xyz)

    def run_vibrational_analysis(self):
        """
        Compute the Hessian matrix along with vibrational frequencies (cm^-1),
        normal mode displacements, force constants (mDyne/A), and reduced masses (AMU).
        """
        atoms = xyz_to_ase(self.opt_xyz or self.xyz)
        species = torch.tensor(atoms.get_atomic_numbers(), device=self.device, dtype=torch.long).unsqueeze(0)
        coordinates = torch.from_numpy(atoms.get_positions()).unsqueeze(0).requires_grad_(True)
        masses = torchani.utils.get_atomic_masses(species)
        energies = self.model.double()((species, coordinates)).energies
        hessian = torchani.utils.hessian(coordinates, energies=energies)
        freqs, modes, force_constants, reduced_masses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MDU')
        freqs = freqs.numpy()
        self.freqs = freqs[~np.isnan(freqs)]
        results = {'hessian': hessian.tolist(),
                   'freqs': self.freqs.tolist(),
                   'modes': modes.tolist(),
                   'force_constants': force_constants.tolist(),
                   'reduced_masses': reduced_masses.tolist(),
                   }
        self.save_output_file(content_dict=results)

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.
        """
        pass

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
        pass

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
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

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.model = torchani.models.ANI2x(periodic_table_index=True).to(self.device)

        if self.job_type == 'sp':
            self.run_sp()

        if self.job_type == 'force':
            self.run_force()

        if self.job_type in ['opt', 'conformers', 'optfreq']:
            self.run_opt()
            self.run_sp()

        if self.job_type in ['freq', 'optfreq']:
            self.run_vibrational_analysis()

        if self.job_type == 'scan':
            raise NotImplementedError("Scan job type is not implemented for TorchANI. Use ARC's directed scan instead")

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        logger.warning('Queue execution is not yet supported for TorchANI in ARC, executing incore instead.')
        self.execute_incore()


register_job_adapter('torchani', TorchANIAdapter)
