#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.adapters.ase module.
These tests verify IO and logic without executing the external ASE script in CI.
"""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from unittest.mock import MagicMock, patch
import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.optimize import BFGS, FIRE, LBFGS
from ase.vibrations import Vibrations

from arc.common import read_yaml_file, save_yaml_file
from arc.job.adapters.ase_adapter import ASEAdapter, active_worker_tickets, servers, submit_filenames
from arc.parser.parser import parse_1d_scan_coords, parse_1d_scan_energies
from arc.species.species import ARCSpecies
from arc.job.adapters.scripts import ase_script
from arc.job.adapters.scripts.ase_script import (BatchedEvaluator,
                                                 ImageCalculator,
                                                 _scan_walk,
                                                 apply_constraints,
                                                 batch_evaluation_function,
                                                 claim_ticket,
                                                 get_ticket_status,
                                                 is_linear,
                                                 is_transient_gpu_error,
                                                 live_tickets,
                                                 live_workers,
                                                 merge_scan_branches,
                                                 numpy_vibrational_analysis,
                                                 relaxed_torsion_scan,
                                                 relax_endpoint,
                                                 rotor_top,
                                                 run_band,
                                                 run_job,
                                                 run_scan_group,
                                                 run_torsion_scan,
                                                 run_vibrational_analysis,
                                                 scan_convergence_warning,
                                                 scan_group_key,
                                                 sella_hessian_function,
                                                 serve,
                                                 spool_dirs,
                                                 submit_ticket,
                                                 to_kJmol,
                                                 uma_inference_settings,
                                                 well_defined_torsion,
                                                 withdraw_ticket,
                                                 write_json_atomic)

FREQ_FACTOR = 1.0 / (2.0 * np.pi * 299792458.0 * 100.0) * np.sqrt(1.602176565e-19 * 1.0e20 / 1.660538921e-27)


def spring_hessian(positions: np.ndarray, springs: list) -> np.ndarray:
    """
    The Hessian (eV/A^2) of harmonic springs at their rest lengths, which is exactly free of translations
    and rotations. ``springs`` holds (atom i, atom j, force constant in eV/A^2).
    """
    hessian = np.zeros((3 * len(positions), 3 * len(positions)))
    for i, j, k in springs:
        bond = positions[j] - positions[i]
        block = k * np.outer(bond, bond) / np.dot(bond, bond)
        for a, b, sign in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            hessian[3 * a:3 * a + 3, 3 * b:3 * b + 3] += sign * block
    return hessian


class BondBarrier(Calculator):
    """
    A base calculator plus a Gaussian barrier, height ``a`` (eV) and width ``s`` (A), centered at ``r0`` along one atom
    pair's distance: a toy TS whose reactive bond stretch is the unstable mode.
    """
    implemented_properties = ['energy', 'free_energy', 'forces']

    def __init__(self, base: Calculator, pair: tuple, r0: float, a: float = 20.0, s: float = 0.3):
        super().__init__()
        self.base, self.pair, self.r0, self.a, self.s = base, pair, r0, a, s

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        bare = Atoms(numbers=self.atoms.numbers, positions=self.atoms.positions)
        bare.calc = self.base
        energy, forces = bare.get_potential_energy(), bare.get_forces().copy()
        i, j = self.pair
        bond = bare.positions[i] - bare.positions[j]
        r = np.linalg.norm(bond)
        barrier = self.a * np.exp(-(r - self.r0) ** 2 / (2 * self.s ** 2))
        energy += barrier
        forces[i] += barrier * (r - self.r0) / self.s ** 2 * bond / r
        forces[j] -= barrier * (r - self.r0) / self.s ** 2 * bond / r
        self.results = {'energy': energy, 'free_energy': energy, 'forces': forces}


class SpringNetwork(Calculator):
    """Harmonic springs between all atom pairs closer than ``cutoff`` (A): rigid, but free to rotate about bonds."""
    implemented_properties = ['energy', 'free_energy', 'forces']

    def __init__(self, atoms: Atoms, cutoff: float = 2.4, k: float = 10.0):
        super().__init__()
        distances = atoms.get_all_distances()
        self.springs = [(i, j, distances[i, j]) for i in range(len(atoms)) for j in range(i + 1, len(atoms))
                        if distances[i, j] < cutoff]
        self.k = k

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions, energy, forces = self.atoms.positions, 0.0, np.zeros((len(self.atoms), 3))
        for i, j, r0 in self.springs:
            bond = positions[i] - positions[j]
            r = np.linalg.norm(bond)
            energy += 0.5 * self.k * (r - r0) ** 2
            forces[i] -= self.k * (r - r0) * bond / r
            forces[j] += self.k * (r - r0) * bond / r
        self.results = {'energy': energy, 'free_energy': energy, 'forces': forces}


def finite_difference_hessian(calc: Calculator, delta: float = 1e-3):
    """A Hessian function (eV/A^2) for Sella that differentiates the forces of ``calc`` by central differences."""
    def hessian(structure: Atoms) -> np.ndarray:
        work = Atoms(numbers=structure.numbers, positions=structure.positions)
        work.calc = calc
        positions, n_coords = work.positions.copy(), 3 * len(work)
        result = np.zeros((n_coords, n_coords))
        for coordinate in range(n_coords):
            for sign in (1, -1):
                displaced = positions.copy()
                displaced.flat[coordinate] += sign * delta
                work.positions = displaced
                result[coordinate] -= sign * work.get_forces().ravel() / (2 * delta)
        return 0.5 * (result + result.T)
    return hessian


ETHANE_XYZ = {'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'),
              'isotopes': (12, 12, 1, 1, 1, 1, 1, 1),
              'coords': ((0.0, 0.0, 0.761395),
                         (0.0, 0.0, -0.761395),
                         (0.0, 1.017111, 1.157241),
                         (0.880844, -0.508556, 1.157241),
                         (-0.880844, -0.508556, 1.157241),
                         (0.880844, 0.508556, -1.157241),
                         (-0.880844, 0.508556, -1.157241),
                         (0.0, -1.017111, -1.157241))}


class TestASEAdapter(unittest.TestCase):
    """
    Contains unit tests for the ASEAdapter class and ase_script utility functions.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.project_directory = tempfile.mkdtemp(prefix='arc_test_ase_')
        cls.addClassCleanup(shutil.rmtree, cls.project_directory, ignore_errors=True)

        xyz = {'symbols': ('O', 'H', 'H'),
               'isotopes': (16, 1, 1),
               'coords': ((0.0, 0.0, 0.0),
                          (0.0, 0.75, 0.58),
                          (0.0, -0.75, 0.58))}
        
        cls.job_1 = ASEAdapter(execution_type='incore',
                               job_type='sp',
                               project='test_1',
                               project_directory=os.path.join(cls.project_directory, 'test_1'),
                               species=[ARCSpecies(label='H2O', xyz=xyz)],
                               args={'keyword': {'calculator': 'torchani', 'model': 'ANI2x'}},
                               testing=True)

        cls.job_2 = ASEAdapter(execution_type='queue',
                               job_type='opt',
                               project='test_2',
                               project_directory=os.path.join(cls.project_directory, 'test_2'),
                               species=[ARCSpecies(label='H2O', xyz=xyz)],
                               args={'keyword': {'calculator': 'xtb', 'method': 'GFN2-xTB'}},
                               testing=True)
                               
        # ``set_files()`` already ran inside the constructor, against the local_path the adapter
        # resolved for itself. The reassignment below repoints the other tests at a flat scratch
        # directory, so keep the constructed paths for assertions about what ``set_files()`` wrote.
        cls.job_1_constructed_local_path = cls.job_1.local_path
        cls.job_2_constructed_local_path = cls.job_2.local_path

        cls.job_1.local_path = os.path.join(cls.project_directory, 'test_1')
        cls.job_2.local_path = os.path.join(cls.project_directory, 'test_2')
        cls.job_2.remote_path = '/path/to/remote'
        os.makedirs(cls.job_1.local_path, exist_ok=True)
        os.makedirs(cls.job_2.local_path, exist_ok=True)

    def test_get_python_executable(self):
        """Test resolving the python executable environment"""
        with patch('arc.job.adapters.ase_adapter.settings', {'TANI_PYTHON': '/path/to/tani_python'}):
            exe = self.job_1.get_python_executable()
            self.assertEqual(exe, '/path/to/tani_python')

        with patch('arc.job.adapters.ase_adapter.settings', {'XTB_PYTHON': '/path/to/xtb_python'}):
            exe = self.job_2.get_python_executable()
            self.assertEqual(exe, '/path/to/xtb_python')

    def test_write_input_file(self):
        """Test writing the YAML input file for the ASE script"""
        self.job_1.write_input_file()
        input_path = os.path.join(self.job_1.local_path, 'input.yml')
        self.assertTrue(os.path.isfile(input_path))
        data = read_yaml_file(input_path)
        self.assertEqual(data['job_type'], 'sp')
        self.assertEqual(data['settings']['calculator'], 'torchani')
        self.assertEqual(data['settings']['model'], 'ANI2x')
        self.assertEqual(data['xyz']['symbols'], ('O', 'H', 'H'))

    def test_write_submit_script(self):
        """Test writing the submission script for queue execution"""
        self.job_2.python_executable = '/fake/python'
        self.job_2.write_submit_script()
        submit_path = os.path.join(self.job_2.local_path, 'submit.sh')
        self.assertTrue(os.path.isfile(submit_path))
        with open(submit_path, 'r') as f:
            content = f.read()
        self.assertIn('/fake/python', content)
        self.assertIn('--yml_path /path/to/remote', content)
        self.assertIn('ase_script.py', content)

    def test_set_files(self):
        """Test properly assigning upload and download files"""
        self.job_2.set_files()
        self.assertTrue(any('submit.sh' in f['local'] for f in self.job_2.files_to_upload))
        self.assertTrue(any('input.yml' in f['local'] for f in self.job_2.files_to_upload))
        self.assertTrue(any('ase_script.py' in f['local'] for f in self.job_2.files_to_upload))
        self.assertTrue(any('output.yml' in f['local'] for f in self.job_2.files_to_download))

    def test_set_files_writes_the_files_of_a_queue_job(self):
        """Test that constructing a queue job already writes its submit script and input file"""
        # JobAdapter.execute() calls upload_files() before execute_queue(), so the files must be on
        # disk once the job object exists, not when it is executed.
        xyz = {'symbols': ('O', 'H', 'H'),
               'isotopes': (16, 1, 1),
               'coords': ((0.0, 0.0, 0.0), (0.0, 0.75, 0.58), (0.0, -0.75, 0.58))}
        fake_server = {'test_server': {'cluster_soft': 'PBS', 'un': 'test_user'}}
        # arc.job.adapter and arc.job.adapters.ase_adapter share this one dict object.
        with patch.dict(servers, fake_server):
            job_3 = ASEAdapter(execution_type='queue',
                               job_type='directed_scan',
                               project='test_3',
                               project_directory=os.path.join(self.project_directory, 'test_3'),
                               species=[ARCSpecies(label='H2O', xyz=xyz)],
                               args={'keyword': {'calculator': 'xtb'},
                                     'block': {'queue': 'test_q',
                                               'env_setup': 'conda activate uma_env',
                                               'python': '/remote/python'}},
                               server='test_server',
                               testing=True)
        submit_path = os.path.join(job_3.local_path, 'submit.sh')
        self.assertTrue(os.path.isfile(submit_path))
        self.assertTrue(os.path.isfile(os.path.join(job_3.local_path, 'input.yml')))
        # Every file the job says it will upload must exist, or ssh.upload_file() raises an InputError.
        for file_dict in job_3.files_to_upload:
            self.assertTrue(os.path.isfile(file_dict['local']), msg=f"missing {file_dict['file_name']}")
        with open(submit_path, 'r') as f:
            content = f.read()
        self.assertIn('#PBS -q test_q', content)
        self.assertIn('conda activate uma_env', content)
        self.assertIn('/remote/python', content)

    def build_queue_job(self, cluster_soft, project, server='test_server', block=None, **kwargs):
        """
        Build a queue-executed ASE job against a fake server, and read back the submit script it
        wrote. Everything happens inside the ``servers`` patch, since the adapter re-reads
        ``servers[self.server]['cluster_soft']`` whenever it resolves the submit filename.

        Args:
            cluster_soft (str): The ``cluster_soft`` entry to give the fake server.
            project (str): The project name, used to keep the jobs' directories apart.
            server (str, optional): The server name to register the fake server under.
            block (dict, optional): The level's ``args['block']`` execution knobs.
            kwargs: Further keyword arguments for the ``ASEAdapter`` constructor.

        Returns:
            Tuple[ASEAdapter, str, str]: The job, its submit filename, and the script's content.
        """
        xyz = {'symbols': ('O', 'H', 'H'),
               'isotopes': (16, 1, 1),
               'coords': ((0.0, 0.0, 0.0), (0.0, 0.75, 0.58), (0.0, -0.75, 0.58))}
        fake_server = {server: {'cluster_soft': cluster_soft, 'un': 'test_user',
                                'queues': {'server_default_q': '24:00:00', 'other_q': '48:00:00'}}}
        with patch.dict(servers, fake_server):
            job = ASEAdapter(execution_type='queue',
                             job_type='opt',
                             project=project,
                             project_directory=os.path.join(self.project_directory, project),
                             species=[ARCSpecies(label='H2O', xyz=xyz)],
                             args={'keyword': {'calculator': 'xtb'}, 'block': block or dict()},
                             server=server,
                             testing=True,
                             **kwargs)
            submit_filename = job.determine_submit_filename()
        with open(os.path.join(job.local_path, submit_filename), 'r') as f:
            content = f.read()
        return job, submit_filename, content

    def test_submit_filename_is_the_one_the_submission_path_invokes(self):
        """Test that a queue job's script is written under the filename submit_job() invokes"""
        # local.submit_job()/ssh.submit_job() name the file to submit as submit_filenames[cluster_soft],
        # looked up verbatim, so writing 'submit.sh' on a Slurm server submits a file that is not there.
        for cluster_soft in ('PBS', 'Slurm'):
            with self.subTest(cluster_soft=cluster_soft):
                job, submit_filename, _ = self.build_queue_job(cluster_soft, f'test_fname_{cluster_soft}')
                self.assertEqual(submit_filename, submit_filenames[cluster_soft])
                self.assertTrue(os.path.isfile(os.path.join(job.local_path, submit_filenames[cluster_soft])))
                self.assertIn(submit_filenames[cluster_soft], [f['file_name'] for f in job.files_to_upload])

    def test_submit_filename_guard_admits_only_what_the_lookup_can_serve(self):
        """Test that a cluster_soft spelling settings does not carry falls back instead of raising"""
        # submit_filenames is keyed by the exact settings spelling ('PBS', 'Slurm'), so a guard that
        # accepts any casing hands the lookup a key it does not have and raises KeyError mid-construction.
        for cluster_soft in ('slurm', 'pbs', 'SLURM', 'HTCondor', 'OGE'):
            with self.subTest(cluster_soft=cluster_soft):
                job, submit_filename, _ = self.build_queue_job(cluster_soft, f'test_case_{cluster_soft}')
                self.assertEqual(submit_filename, 'submit.sh')

    def test_write_submit_script_warns_on_a_scheduler_with_no_template(self):
        """Test that a queue job on a scheduler ase_submit has no template for says so"""
        # ase_submit covers PBS and Slurm only; HTCondor's submit file is a condor description
        # file rather than a shell script, so the adapter writes the bare script and warns.
        with self.assertLogs('arc', level='WARNING') as captured:
            _, submit_filename, content = self.build_queue_job('HTCondor', 'test_no_template')
        self.assertEqual(submit_filename, 'submit.sh')
        self.assertNotIn('#PBS', content)
        self.assertNotIn('#SBATCH', content)
        self.assertTrue(any('ASE submit templates exist only for' in line for line in captured.output))

    def test_write_submit_script_enters_the_local_path_of_a_local_server(self):
        """Test that a queue job on the 'local' server cd's into its local path, not its remote path"""
        # The shared queue path submits a local job from local_path, while _initialize_adapter() still
        # constructs a remote_path; a script that cd's into the latter fails before ASE ever runs.
        job, _, content = self.build_queue_job('PBS', 'test_local_pwd', server='local')
        self.assertIn(f'cd "{job.local_path}"', content)
        self.assertNotIn(job.remote_path, content)

    def test_write_submit_script_records_the_resolved_queue(self):
        """Test that the queue a job is submitted to is recorded in attempted_queues"""
        # trsh_job_queue() moves a failed job to the next queue by filtering on attempted_queues,
        # so a queue that never lands there is retried forever.
        job, _, content = self.build_queue_job('PBS', 'test_queue_default')
        self.assertEqual(job.attempted_queues, ['server_default_q'])
        self.assertIn('#PBS -q server_default_q', content)

        job, _, content = self.build_queue_job('PBS', 'test_queue_block', block={'queue': 'block_q'})
        self.assertEqual(job.attempted_queues, ['block_q'])
        self.assertIn('#PBS -q block_q', content)

    def test_write_submit_script_requests_the_job_walltime(self):
        """Test that max_job_time reaches the submit script rather than the queue's own default"""
        _, _, pbs_content = self.build_queue_job('PBS', 'test_walltime_pbs', max_job_time=12.0)
        self.assertIn('#PBS -l walltime=12:00:00', pbs_content)

        _, _, slurm_content = self.build_queue_job('Slurm', 'test_walltime_slurm', max_job_time=48.0)
        self.assertIn('#SBATCH -t 2-0:00:00', slurm_content)  # Slurm's t_max_format is days

    def test_execute_queue_submits_through_the_shared_queue_path(self):
        """Test that a queue job is submitted through JobAdapter.legacy_queue_execution()"""
        # set_files() wrote the files and JobAdapter.execute() uploaded them, so an execute_queue()
        # that submits by itself bypasses the shared ssh pool and the 'local' server branch with it.
        job, _, _ = self.build_queue_job('PBS', 'test_execute_queue')
        with patch('arc.job.adapter.JobAdapter.legacy_queue_execution') as submission:
            job.execute_queue()
        submission.assert_called_once_with()

    def test_get_queue_submit_script_rejects_a_scheduler_with_no_template(self):
        """Test that composing a script for a cluster software ase_submit has no template for raises"""
        # Reachable only by adding a scheduler to QUEUE_CLUSTER_SOFT without giving ase_submit its
        # template, which would otherwise write a submit script with no scheduler directives at all.
        job, _, _ = self.build_queue_job('PBS', 'test_no_ase_template')
        with self.assertRaises(NotImplementedError):
            job.get_queue_submit_script(command='python ase_script.py',
                                        config=job.determine_submit_config(),
                                        cluster_soft='htcondor')

    def test_set_files_puts_the_script_in_a_local_queue_job_directory(self):
        """A queue job on this machine is not uploaded, so its directory needs the ASE script its submit script runs"""
        with tempfile.TemporaryDirectory() as tmp:
            job = ASEAdapter(execution_type='queue',
                             job_type='sp',
                             project='test_local_queue',
                             project_directory=os.path.join(tmp, 'project'),
                             species=[ARCSpecies(label='ethane', xyz=ETHANE_XYZ)],
                             args={'keyword': {'calculator': 'emt'}},
                             server='local',
                             testing=True)
            self.assertTrue(os.path.isfile(os.path.join(job.local_path, 'ase_script.py')))
            self.assertTrue(os.path.isfile(os.path.join(job.local_path, 'input.yml')))

    def test_set_files_does_not_write_for_an_incore_job(self):
        """Test that an incore job writes no submit script (it writes its input when it executes)"""
        # Assert against the directory set_files() actually wrote to, not the scratch directory
        # setUpClass repoints local_path at afterwards - there, nothing is ever written and the
        # assertion cannot fail.
        self.assertTrue(os.path.isdir(self.job_1_constructed_local_path))
        self.assertEqual([entry for entry in os.listdir(self.job_1_constructed_local_path)
                          if entry.startswith('submit.')], list())
        self.assertTrue(all('submit.sh' not in f['local'] for f in self.job_1.files_to_upload))

    def test_determine_constraints(self):
        """Test that a directed scan derives a 1-indexed dihedral constraint from torsions/dihedrals"""
        xyz = {'symbols': ('O', 'H', 'H'),
               'isotopes': (16, 1, 1),
               'coords': ((0.0, 0.0, 0.0), (0.0, 0.75, 0.58), (0.0, -0.75, 0.58))}
        job = ASEAdapter(execution_type='incore',
                         job_type='directed_scan',
                         project='test_c',
                         project_directory=os.path.join(self.project_directory, 'test_c'),
                         species=[ARCSpecies(label='H2O', xyz=xyz)],
                         torsions=[[0, 1, 2, 3]],
                         dihedrals=[60.0],
                         args={'keyword': {'calculator': 'xtb'}},
                         testing=True)
        # torsions are 0-indexed; the returned constraint must be 1-indexed, as xTB/Gaussian expect.
        self.assertEqual(job.determine_constraints(), [([1, 2, 3, 4], 60.0)])

    def test_apply_constraints_converts_to_zero_indexed(self):
        """Test that apply_constraints translates ARC's 1-indexed dihedral to ASE's 0-indexed FixInternals"""
        atoms = Atoms('C4', positions=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0),
                                       (2.0, 1.4, 0.0), (3.5, 1.4, 0.0)])
        apply_constraints(atoms, [([1, 2, 3, 4], 90.0)])
        self.assertEqual(len(atoms.constraints), 1)
        dihedrals = atoms.constraints[0].todict()['kwargs']['dihedrals_deg']
        self.assertEqual(dihedrals, [[90.0, [0, 1, 2, 3]]])

    def test_parse_results(self):
        """Test parsing dummy output YAML back into object attributes"""
        output_data = {
            'sp': -76.0,
            'opt_xyz': {'symbols': ('O', 'H', 'H'), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.76, 0.59), (0.0, -0.76, 0.59))},
            'freqs': [1500.0, 3600.0, 3700.0],
            'modes': [[[0.0, 0.0, 0.1]]],
            'reduced_masses': [1.0, 1.0, 1.0],
            'force_constants': [1.0, 2.0, 3.0]
        }
        save_yaml_file(os.path.join(self.job_1.local_path, 'output.yml'), output_data)
        self.job_1.parse_results()
        self.assertEqual(self.job_1.electronic_energy, -76.0)
        self.assertEqual(self.job_1.frequencies, [1500.0, 3600.0, 3700.0])
        self.assertEqual(self.job_1.force_constants, [1.0, 2.0, 3.0])
        self.assertIsNotNone(self.job_1.xyz_out)
        self.assertAlmostEqual(self.job_1.xyz_out['coords'][1][1], 0.76)

    def test_to_kJmol(self):
        """Test utility conversion function to_kJmol"""
        self.assertAlmostEqual(to_kJmol(1.0), 96.48534, places=5)
        self.assertAlmostEqual(to_kJmol(27.21138), 2625.49937, places=5)

    def test_is_linear(self):
        """Test the is_linear helper function in ase_script"""
        from ase import Atoms
        # 1. Monatomic (H)
        h = Atoms('H', positions=[(0.0, 0.0, 0.0)])
        self.assertFalse(is_linear(h))

        # 2. Diatomic (H2)
        h2 = Atoms('H2', positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])
        self.assertTrue(is_linear(h2))

        # 3. Linear triatomic (CO2)
        co2 = Atoms('CO2', positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.16), (0.0, 0.0, -1.16)])
        self.assertTrue(is_linear(co2))

        # 4. Non-linear triatomic (H2O)
        h2o = Atoms('H2O', positions=[(0.0, 0.0, 0.0), (0.0, 0.75, 0.58), (0.0, -0.75, 0.58)])
        self.assertFalse(is_linear(h2o))

    def test_numpy_vibrational_analysis(self):
        """Test fallback numpy vibrational analysis directly"""
        masses = np.array([16.0, 1.0, 1.0])
        n_atoms = len(masses)
        # Create a hessian with some very small eigenvalues (for translations/rotations)
        # and some large ones.
        hessian = np.zeros((3 * n_atoms, 3 * n_atoms))
        for i in range(6, 9):
            hessian[i, i] = 10.0
        
        results = numpy_vibrational_analysis(masses, hessian)
        self.assertIn('freqs', results)
        self.assertIn('modes', results)
        self.assertIn('force_constants', results)
        self.assertIn('reduced_masses', results)
        print(results['freqs'])
        # nonlinear (len > 2), filters out first 6 modes
        self.assertEqual(len(results['freqs']), 3)
        self.assertEqual(len(results['modes']), 3)
        self.assertEqual(len(results['force_constants']), 3)
        self.assertEqual(len(results['reduced_masses']), 3)

        # 3 atom linear species, actual hessian from computation in Orca 6.0.0 r2scan-3c for O=C=O
        hessian = np.array(\
            [[ 1.47625806e-01,  2.29436182e-05,  1.34550341e-05, -7.38129035e-02,
            -7.97506445e-06, -4.70698398e-06, -7.38129030e-02, -7.97743334e-06,
            -4.70838645e-06],
            [ 2.29436182e-05,  1.49024440e+00,  7.76950445e-01, -1.49650473e-05,
            -7.45122209e-01, -3.88386820e-01, -1.49696914e-05, -7.45122187e-01,
            -3.88386839e-01],
            [ 1.34550341e-05,  7.76950445e-01,  5.96226997e-01, -8.74599081e-06,
            -3.88563609e-01, -2.98113486e-01, -8.74870706e-06, -3.88563621e-01,
            -2.98113511e-01],
            [-7.38129035e-02, -1.49650473e-05, -8.74599081e-06,  3.68163144e-02,
            9.08896394e-06,  5.35084314e-06,  3.69965890e-02,  2.38052672e-06,
            1.37531791e-06],
            [-7.97506445e-06, -7.45122209e-01, -3.88563609e-01,  9.08896394e-06,
            7.93484601e-01,  4.37753696e-01,  2.38165713e-06, -4.83623921e-02,
            -4.92784793e-02],
            [-4.70698398e-06, -3.88386820e-01, -2.98113486e-01,  5.35084314e-06,
            4.37753696e-01,  2.89551995e-01,  1.37597060e-06, -4.92784828e-02,
            8.56149142e-03],
            [-7.38129030e-02, -1.49696914e-05, -8.74870706e-06,  3.69965890e-02,
            2.38165713e-06,  1.37597060e-06,  3.68163139e-02,  9.09247043e-06,
            5.35290250e-06],
            [-7.97743334e-06, -7.45122187e-01, -3.88563621e-01,  2.38052672e-06,
            -4.83623921e-02, -4.92784828e-02,  9.09247043e-06,  7.93484579e-01,
            4.37753711e-01],
            [-4.70838645e-06, -3.88386839e-01, -2.98113511e-01,  1.37531791e-06,
            -4.92784793e-02,  8.56149142e-03,  5.35290250e-06,  4.37753711e-01,
            2.89552020e-01],])
        masses = np.array([12.0, 16.0, 16.0])
        freqs = np.array([0., 0., 0., 0., 0., 666.85873322, 668.56887375, 1362.1172728, 2423.3776014])

        conv_factor = 27.211386245988 / (0.529177210903 ** 2)
        results = numpy_vibrational_analysis(masses, hessian * conv_factor, is_linear=True)
        self.assertEqual(len(results['freqs']), 4)
        for i, val in enumerate(freqs[5:]):
            self.assertAlmostEqual(results['freqs'][i], val, delta=1e-3)

    def test_numpy_vibrational_analysis_projects_out_a_spurious_rotation(self):
        """Test that, given the geometry, a spurious rotation is projected out instead of reported as a vibration"""
        water = Atoms('OH2', positions=[(0.0, 0.0, 0.0), (0.757, 0.586, 0.0), (-0.757, 0.586, 0.0)])
        masses, positions = water.get_masses(), water.get_positions()
        hessian = spring_hessian(positions, [(0, 1, 50.0), (0, 2, 50.0), (1, 2, 5.0)])
        clean = numpy_vibrational_analysis(masses, hessian)['freqs']
        self.assertEqual(len(clean), 3)

        sqrt_mass = np.sqrt(masses)[:, np.newaxis]
        rotation = (np.cross([0.0, 0.0, 1.0], positions - water.get_center_of_mass()) * sqrt_mass).ravel()
        rotation /= np.linalg.norm(rotation)
        weighted = rotation * np.repeat(np.sqrt(masses), 3)
        contaminated = hessian - (60.0 / FREQ_FACTOR) ** 2 * np.outer(weighted, weighted)

        unprojected = numpy_vibrational_analysis(masses, contaminated)['freqs']
        self.assertEqual(len(unprojected), 4)
        self.assertAlmostEqual(unprojected[0], -60.0, places=6)

        projected = numpy_vibrational_analysis(masses, contaminated, positions=positions)
        np.testing.assert_allclose(projected['freqs'], clean, atol=1e-6)
        self.assertEqual(np.array(projected['modes']).shape, (3, 3, 3))

    def test_numpy_vibrational_analysis_projects_a_linear_molecule(self):
        """For a linear molecule, the two rotations are projected out and both bends are kept"""
        r = 1.16
        co2 = Atoms('OCO', positions=[(0, 0, -r), (0, 0, 0), (0, 0, r)])
        masses, positions = co2.get_masses(), co2.get_positions()
        hessian = spring_hessian(positions, [(0, 1, 100.0), (1, 2, 100.0)])
        for axis in (0, 1):
            bend = np.zeros(9)
            bend[[axis, 3 + axis, 6 + axis]] = 1.0 / r, -2.0 / r, 1.0 / r
            hessian += 5.0 * np.outer(bend, bend)
        inv_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
        eigenvalues = np.linalg.eigvalsh(hessian * np.outer(inv_sqrt_mass, inv_sqrt_mass))
        expected = FREQ_FACTOR * np.sqrt(np.sort(eigenvalues)[5:])

        freqs = numpy_vibrational_analysis(masses, hessian, is_linear=True, positions=positions)['freqs']
        np.testing.assert_allclose(freqs, expected, rtol=1e-8)
        self.assertGreater(min(freqs), 100.0)
        self.assertAlmostEqual(freqs[0], freqs[1], places=6)

    def test_numpy_vibrational_analysis_returns_3n_minus_5_or_6_modes_given_positions(self):
        """Test the number of modes for an atom, a diatomic, a linear and a nonlinear molecule"""
        cases = [(Atoms('He', positions=[(0, 0, 0)]), [], 0),
                 (Atoms('OH', positions=[(0, 0, 0), (0, 0, 0.97)]), [(0, 1, 50.0)], 1),
                 (Atoms('OCO', positions=[(0, 0, -1.16), (0, 0, 0), (0, 0, 1.16)]), [(0, 1, 100.0), (1, 2, 100.0)], 4),
                 (Atoms(ETHANE_XYZ['symbols'], positions=ETHANE_XYZ['coords']),
                  [(i, j, 30.0) for i in range(8) for j in range(i + 1, 8)], 18)]
        for atoms, springs, n_modes in cases:
            positions = atoms.get_positions()
            results = numpy_vibrational_analysis(atoms.get_masses(), spring_hessian(positions, springs),
                                                 is_linear=is_linear(atoms), positions=positions)
            self.assertEqual(len(results['freqs']), n_modes, atoms.get_chemical_formula())
            self.assertEqual(len(results['modes']), n_modes)

    def test_uma_inference_settings(self):
        """Test that a Hessian predictor runs at full precision whatever the job's preset, and is cached apart"""
        self.assertEqual(uma_inference_settings({}), (None, None))
        self.assertEqual(uma_inference_settings({'inference_settings': 'traineval'}), ('traineval', 'traineval'))
        custom = {'tf32': False, 'activation_checkpointing': False}
        settings, key = uma_inference_settings({'inference_settings': custom})
        self.assertEqual(settings, custom)
        self.assertEqual(key, uma_inference_settings({'inference_settings': dict(reversed(custom.items()))})[1])
        hash(key)

        settings, key = uma_inference_settings({'inference_settings': 'default'}, hessian=True)
        self.assertEqual(settings, {'tf32': False, 'activation_checkpointing': False, 'merge_mole': False,
                                    'compile': False, 'base_precision_dtype': 'float32', 'hessian_vmap': False,
                                    'predict_untrained_hessian': ['omol']})
        self.assertNotEqual(key, uma_inference_settings({'inference_settings': 'default'})[1])
        hash(key)

        settings, _ = uma_inference_settings({'inference_settings': {'base_precision_dtype': 'float64', 'tf32': True},
                                              'hessian_vmap': True, 'task': 'omat'}, hessian=True)
        self.assertEqual(settings['base_precision_dtype'], 'float64')
        self.assertFalse(settings['tf32'])
        self.assertTrue(settings['hessian_vmap'])
        self.assertEqual(settings['predict_untrained_hessian'], ['omat'])

    def test_sella_hessian_function(self):
        """A UMA TS search gets the exact autograd Hessian; other calculators, and a fairchem without one, do not"""
        self.assertIsNone(sella_hessian_function({'calculator': 'emt'}))
        self.assertIsNone(sella_hessian_function({'calculator': 'uma', 'hessian': 'finite_difference'}))
        with patch('arc.job.adapters.scripts.ase_script.load_uma_predictor', side_effect=NotImplementedError):
            self.assertIsNone(sella_hessian_function({'calculator': 'uma'}))
        with patch('arc.job.adapters.scripts.ase_script.load_uma_predictor'), \
                patch('arc.job.adapters.scripts.ase_script.uma_hessian', return_value=np.eye(3)) as uma_hessian:
            hessian_function = sella_hessian_function({'calculator': 'uma'}, cache=dict())
            np.testing.assert_array_equal(hessian_function(Atoms('H')), np.eye(3))
            uma_hessian.assert_called_once()

    def test_run_vibrational_analysis_chooses_the_hessian(self):
        """Test that UMA uses its autograd Hessian unless told otherwise or unable, and other calculators use finite differences"""
        water = Atoms('OH2', positions=[(0.0, 0.0, 0.0), (0.757, 0.586, 0.0), (-0.757, 0.586, 0.0)])
        water.calc = EMT()
        springs = spring_hessian(water.get_positions(), [(0, 1, 50.0), (0, 2, 50.0), (1, 2, 5.0)])
        with tempfile.TemporaryDirectory() as work_dir:
            with patch('arc.job.adapters.scripts.ase_script.uma_hessian', return_value=springs) as uma_hessian:
                results = run_vibrational_analysis(water, {'calculator': 'uma'}, work_dir=work_dir)
                self.assertEqual(results['hessian_method'], 'analytic')
                np.testing.assert_allclose(results['hessian'], springs)
                self.assertEqual(uma_hessian.call_count, 1)

                results = run_vibrational_analysis(water, {'calculator': 'uma', 'hessian': 'finite_difference'},
                                                   work_dir=work_dir)
                self.assertEqual(results['hessian_method'], 'finite_difference_gradient')
                self.assertEqual(uma_hessian.call_count, 1)

                results = run_vibrational_analysis(water, {'calculator': 'emt'}, work_dir=work_dir)
                self.assertEqual(results['hessian_method'], 'finite_difference_gradient')
                self.assertEqual(len(results['freqs']), 3)
                self.assertEqual(uma_hessian.call_count, 1)

            with patch('arc.job.adapters.scripts.ase_script.uma_hessian', return_value=None):
                results = run_vibrational_analysis(water, {'calculator': 'uma'}, work_dir=work_dir)
                self.assertEqual(results['hessian_method'], 'finite_difference_gradient')

            with self.assertRaises(ValueError):
                run_vibrational_analysis(water, {'calculator': 'uma', 'hessian': 'numerical'}, work_dir=work_dir)

    def test_write_input_file_scan(self):
        """Test that a scan job writes torsions and scan resolution into the input file"""
        scan_dir = os.path.join(self.project_directory, 'scan_input')
        os.makedirs(scan_dir, exist_ok=True)
        job = ASEAdapter(execution_type='incore',
                         job_type='scan',
                         project='test_scan',
                         project_directory=scan_dir,
                         species=[ARCSpecies(label='ethane', xyz=ETHANE_XYZ)],
                         torsions=[[2, 0, 1, 5]],
                         args={'keyword': {'calculator': 'uma', 'model': 'uma-s-1p2'},
                               'trsh': {'scan_res': 8.0}},
                         testing=True)
        job.local_path = scan_dir
        job.write_input_file()
        data = read_yaml_file(os.path.join(scan_dir, 'input.yml'))
        self.assertEqual(data['job_type'], 'scan')
        self.assertEqual(data['torsions'], [[2, 0, 1, 5]])
        self.assertEqual(data['scan_res'], 8.0)

    def test_rotor_top(self):
        """Test resolving the rotating group across a pivot bond"""
        atoms = Atoms(symbols=ETHANE_XYZ['symbols'], positions=ETHANE_XYZ['coords'])
        # The dihedral is [2, 0, 1, 5]; pivots are atoms 0 and 1. The group on the atom-1 side is
        # that carbon and its three hydrogens (5, 6, 7).
        self.assertEqual(rotor_top(atoms, 0, 1), [1, 5, 6, 7])
        self.assertEqual(rotor_top(atoms, 1, 0), [0, 2, 3, 4])
        # A pivot bond inside a ring is ill-defined for a 1D rotor.
        cyclopropane = Atoms(symbols=('C', 'C', 'C'),
                             positions=((0.0, 0.87, 0.0), (0.75, -0.43, 0.0), (-0.75, -0.43, 0.0)))
        with self.assertRaises(ValueError):
            rotor_top(cyclopropane, 0, 1)
        # Pivots that are not bonded cannot define a rotating group.
        with self.assertRaises(ValueError):
            rotor_top(atoms, 2, 5)
        # A stretched pivot bond (the TS case) falls outside the default cutoff, but only the pivot
        # pair gets the looser allowance, so the top is still resolved and no spurious ring appears.
        stretched = ETHANE_XYZ['coords'][:1] + ((0.0, 0.0, -1.9),) + ETHANE_XYZ['coords'][2:]
        ts_like = Atoms(symbols=ETHANE_XYZ['symbols'], positions=stretched)
        self.assertEqual(rotor_top(ts_like, 0, 1), [1, 5, 6, 7])
        # Pulled far enough apart, they are no longer a bond at all.
        with self.assertRaises(ValueError):
            rotor_top(ts_like, 0, 1, pivot_mult=1.0)

    def test_rotor_top_does_not_bond_geminal_heavy_atoms(self):
        """In dimethyl ether the carbons are 2.33 A apart, which ASE's default neighbor-list skin would bond"""
        dme = Atoms('COCHHHHHH', positions=[(1.1650, -0.0001, -0.2066), (0.0, 0.0, 0.5714), (-1.1650, 0.0001, -0.2066),
                                            (2.0267, 0.0006, 0.4591), (1.2136, -0.8955, -0.8458),
                                            (1.2129, 0.8944, -0.8471), (-2.0267, -0.0006, 0.4591),
                                            (-1.2136, 0.8955, -0.8458), (-1.2129, -0.8944, -0.8471)])
        self.assertAlmostEqual(dme.get_distance(0, 2), 2.33, places=2)
        self.assertEqual(rotor_top(dme, 1, 0), [0, 3, 4, 5])

    def test_well_defined_torsion(self):
        """A dihedral end atom collinear with the pivot bond is replaced by the next atom off the axis"""
        hoccoh = Atoms('OCCOHH', positions=[(0.0, 0.0, 0.0), (1.3, 0.0, 0.0), (2.5, 0.0, 0.0), (3.8, 0.0, 0.0),
                                            (-0.3, 0.92, 0.0), (4.1, 0.0, 0.92)])
        self.assertEqual(well_defined_torsion(hoccoh, [4, 0, 1, 2]), [4, 0, 1, 5])
        self.assertEqual(well_defined_torsion(hoccoh, [5, 3, 2, 1]), [5, 3, 2, 4])
        ethane = Atoms('CCHHHHHH', positions=[(0.0, 0.0, 0.765), (0.0, 0.0, -0.765), (1.018, 0.0, 1.164),
                                              (-0.509, 0.882, 1.164), (-0.509, -0.882, 1.164),
                                              (-1.018, 0.0, -1.164), (0.509, -0.882, -1.164),
                                              (0.509, 0.882, -1.164)])
        self.assertEqual(well_defined_torsion(ethane, [2, 0, 1, 5]), [2, 0, 1, 5])
        acyl = Atoms('COCOH', positions=[(0.0, 0.0, 0.0), (-0.7, 1.0, 0.0), (1.5, 0.0, 0.0), (2.65, 0.0, 0.0),
                                         (-0.6, -0.9, 0.0)])
        with self.assertRaises(ValueError):
            well_defined_torsion(acyl, [1, 0, 2, 3])

    def test_relaxed_torsion_scan_scans_a_well_defined_dihedral(self):
        """A torsion through a near-linear chain is scanned about a well-defined dihedral of the same pivot bond"""
        hoccoh = Atoms('OCCOHH', positions=[(0.0, 0.0, 0.0), (1.3, 0.0, 0.0), (2.5, 0.0, 0.0), (3.8, 0.0, 0.0),
                                            (-0.3, 0.92, 0.0), (4.1, 0.0, 0.92)])
        hoccoh.calc = EMT()
        with patch('arc.job.adapters.scripts.ase_script._scan_walk', side_effect=StopIteration) as scan_walk:
            with self.assertRaises(StopIteration):
                relaxed_torsion_scan(hoccoh, [4, 0, 1, 2], step_deg=8.0, nsteps=45)
        self.assertEqual(scan_walk.call_args.args[1], [4, 0, 1, 5])

    @staticmethod
    def _run_fake_irc_job(run) -> dict:
        """Run ase_script.main() on an IRC job whose Sella IRC.run is ``run(irc)``, and return its output.yml"""
        from arc.job.adapters.scripts import ase_script

        class FakeIRC:
            def __init__(self, atoms, logfile=None, trajectory=None, **kwargs):
                self.atoms, self.trajectory = atoms, trajectory

            def run(self, fmax, steps, direction):
                run(self)

        fake_sella = type(sys)('sella')
        fake_sella.IRC = FakeIRC
        with tempfile.TemporaryDirectory() as tmp:
            input_path = os.path.join(tmp, 'input.yml')
            save_yaml_file(input_path, {'job_type': 'irc', 'xyz': ETHANE_XYZ, 'charge': 0, 'multiplicity': 1,
                                        'settings': {'calculator': 'emt'}})
            with patch.dict(sys.modules, {'sella': fake_sella}), \
                    patch.object(sys, 'argv', ['ase_script.py', '--yml_path', input_path]):
                ase_script.main()
            return read_yaml_file(os.path.join(tmp, 'output.yml'))

    @staticmethod
    def _write_irc_points(irc, n_points: int):
        """Write ``n_points`` geometries, each shifted further along x, to the fake IRC's trajectory"""
        from ase.io.trajectory import Trajectory
        with Trajectory(irc.trajectory, 'w') as traj:
            for n in range(n_points):
                image = Atoms(irc.atoms.get_chemical_symbols(), positions=irc.atoms.get_positions())
                image.positions[0, 0] += 0.1 * n
                traj.write(image)

    def test_an_irc_job_reports_an_untrustworthy_path(self):
        """An IRC whose inner loop failed to converge reports its path as incomplete, and still returns it"""
        def run(irc):
            self._write_irc_points(irc, 4)
            for _ in range(2):
                warnings.warn('IRC inner loop failed to converge! The trajectory is no longer a trustworthy IRC.')

        output = self._run_fake_irc_job(run)
        self.assertIs(output['irc_complete'], False)
        self.assertEqual(len(output['warnings']), 1)
        self.assertIn('2 step(s)', output['warnings'][0])
        self.assertEqual(len(output['irc_traj']), 4)
        self.assertNotIn('error', output)

    def test_an_irc_job_that_stops_partway_reports_it(self):
        """An IRC that raised after walking part of its path keeps that part and reports the path as incomplete"""
        def run(irc):
            self._write_irc_points(irc, 3)
            raise RuntimeError('inner loop diverged')

        output = self._run_fake_irc_job(run)
        self.assertIs(output['irc_complete'], False)
        self.assertIn('stopped after 3 points: RuntimeError: inner loop diverged', output['warnings'][0])
        self.assertEqual(len(output['irc_traj']), 3)
        self.assertNotIn('error', output)

    def test_an_irc_job_with_an_unreadable_trajectory_errors(self):
        """An IRC trajectory that cannot be read is an error, not an empty path"""
        def run(irc):
            with open(irc.trajectory, 'wb') as f:
                f.write(b'not a trajectory')

        output = self._run_fake_irc_job(run)
        self.assertIn('could not be read', output['error'])
        self.assertNotIn('irc_traj', output)

    def test_relaxed_torsion_scan(self):
        """Test a full 1D relaxed torsional scan on the machinery (EMT keeps it hermetic)"""
        atoms = Atoms(symbols=ETHANE_XYZ['symbols'], positions=ETHANE_XYZ['coords'])
        atoms.calc = EMT()
        result = relaxed_torsion_scan(atoms, [2, 0, 1, 5], step_deg=8.0, nsteps=45,
                                      fmax=0.05, steps=100)
        # 8 deg over 360 deg is 46 points including the duplicated endpoint.
        self.assertEqual(len(result['energies']), 46)
        self.assertEqual(len(result['angles']), 46)
        self.assertEqual(result['angles'][0], 0.0)
        self.assertEqual(result['angles'][-1], 360.0)
        # The endpoint duplicates the start (a periodic grid).
        self.assertEqual(result['energies'][0], result['energies'][-1])
        self.assertTrue(all(np.isfinite(e) for e in result['energies']))
        self.assertEqual(result['top'], [1, 5, 6, 7])
        self.assertIn('fmax_worst', result)
        self.assertIn('branch_gap_max', result)
        # A non-positive resolution or a zero-length grid is rejected before any division.
        with self.assertRaises(ValueError):
            relaxed_torsion_scan(atoms, [2, 0, 1, 5], step_deg=8.0, nsteps=0)
        with self.assertRaises(ValueError):
            relaxed_torsion_scan(atoms, [2, 0, 1, 5], step_deg=0.0, nsteps=45)

    def test_a_scan_batches_its_two_walks(self):
        """The two walks of a scan reach the calculator together, with the result of walking them one by one"""
        atoms = Atoms(symbols=ETHANE_XYZ['symbols'], positions=ETHANE_XYZ['coords'])
        atoms.calc = EMT()
        batch_sizes, evaluate = list(), batch_evaluation_function(EMT())

        def recording_evaluate(structures: list) -> list:
            batch_sizes.append(len(structures))
            return evaluate(structures)

        result = relaxed_torsion_scan(atoms, [2, 0, 1, 5], step_deg=8.0, nsteps=45, fmax=0.05, steps=100,
                                      evaluator=BatchedEvaluator(recording_evaluate))
        top = rotor_top(atoms, 0, 1)
        e_f = _scan_walk(atoms, [2, 0, 1, 5], 8.0, 45, top, 0.05, 100, +1, LBFGS)[0]
        e_b = _scan_walk(atoms, [2, 0, 1, 5], 8.0, 45, top, 0.05, 100, -1, LBFGS)[0]
        np.testing.assert_allclose(result['energies'][:-1], merge_scan_branches(e_f, e_b, 45)[0], rtol=0, atol=1e-10)
        self.assertEqual(max(batch_sizes), 2)
        self.assertGreater(batch_sizes.count(2), len(batch_sizes) / 2)
        self.assertFalse(result['saddle'])

    def test_a_scan_rotates_the_given_group(self):
        """A scan turns ARC's rotating group, which in a TS reaches across a partial bond that distances miss"""
        from ase.build import molecule
        atoms = molecule('C3H8')
        distances = atoms.get_all_distances()
        carbons = [atom.index for atom in atoms if atom.symbol == 'C']
        center = next(c for c in carbons if sum(distances[c, o] < 1.7 for o in carbons if o != c) == 2)
        end_a, end_c = [c for c in carbons if c != center]

        def hydrogens(carbon: int) -> list:
            return [h for h in range(len(atoms)) if atoms[h].symbol == 'H' and distances[carbon, h] < 1.2]

        group_c = [end_c] + hydrogens(end_c)
        atoms.positions[group_c] += 0.8 * (atoms.positions[end_c] - atoms.positions[center]) / distances[center, end_c]
        torsion, top = [hydrogens(end_a)[0], end_a, center, end_c], sorted([center] + hydrogens(center) + group_c)
        self.assertNotIn(end_c, rotor_top(atoms, end_a, center))
        turned = dict()
        for given in (None, top):
            atoms.calc = SpringNetwork(atoms)
            result = run_torsion_scan(atoms, {'torsions': [torsion], 'scan_res': 30.0, **({'top': given} if given else {})},
                                      settings={'fmax': 0.01, 'steps': 200})
            dihedrals = [Atoms(atoms.get_chemical_symbols(), positions=xyz['coords']).get_dihedral(*torsion)
                         for xyz in result['scan_coords']]
            turned[given is not None] = np.round((np.array(dihedrals) - dihedrals[0]) % 360) % 360
        np.testing.assert_array_equal(turned[False], np.zeros(13))
        np.testing.assert_array_equal(turned[True], [30.0 * n % 360 for n in range(13)])
        atoms.calc = SpringNetwork(atoms)
        result = run_torsion_scan(atoms, {'torsions': [torsion], 'scan_res': 30.0, 'partial_bonds': [[center, end_c]],
                                          'top': [end_a]}, settings={'fmax': 0.01, 'steps': 200})
        self.assertEqual(result['top'], top)
        atoms.calc = SpringNetwork(atoms)
        result = run_torsion_scan(atoms, {'torsions': [torsion], 'scan_res': 30.0, 'top': [end_a] + hydrogens(end_a)},
                                  settings={'fmax': 0.01, 'steps': 200})
        self.assertEqual(result['top'], top)

    def test_a_batch_that_runs_out_of_gpu_memory_is_halved(self):
        """Walks waiting together are evaluated in smaller batches if their batch runs out of GPU memory"""
        sizes = list()

        def small_gpu(structures: list) -> list:
            sizes.append(len(structures))
            if len(structures) > 3:
                raise RuntimeError('CUDA out of memory. Tried to allocate 286.00 MiB.')
            return [(float(len(structure)), np.zeros((len(structure), 3))) for structure in structures]

        evaluator = BatchedEvaluator(small_gpu)
        evaluator.join(6)

        def walk(n_atoms: int) -> float:
            try:
                structure = Atoms('H' * n_atoms, positions=[(float(i), 0.0, 0.0) for i in range(n_atoms)])
                return evaluator.request('forces', structure)[0]
            finally:
                evaluator.leave()

        with ThreadPoolExecutor(max_workers=6) as pool:
            energies = list(pool.map(walk, range(1, 7)))
        self.assertEqual(sorted(energies), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(energies, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(sizes, [6, 3, 3])
        self.assertEqual(evaluator.max_batch, 3)

    def test_a_batch_that_hits_a_transient_cublas_error_is_retried(self):
        """A batch that fails with a transient cuBLAS error is evaluated again, whole, and its walks get their results"""
        sizes = list()

        def flaky_gpu(structures: list) -> list:
            sizes.append(len(structures))
            if len(sizes) == 1:
                raise RuntimeError('CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgetrfBatched('
                                   'handle, n, dA_array, ldda, ipiv_array, info_array, batchsize)`')
            return [(float(len(structure)), np.zeros((len(structure), 3))) for structure in structures]

        evaluator = BatchedEvaluator(flaky_gpu)
        evaluator.join(4)

        def walk(n_atoms: int) -> float:
            try:
                structure = Atoms('H' * n_atoms, positions=[(float(i), 0.0, 0.0) for i in range(n_atoms)])
                return evaluator.request('forces', structure)[0]
            finally:
                evaluator.leave()

        with ThreadPoolExecutor(max_workers=4) as pool:
            energies = list(pool.map(walk, range(1, 5)))
        self.assertEqual(energies, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(sizes, [4, 4])
        self.assertEqual(evaluator.n_retries, 1)
        self.assertEqual(evaluator.max_batch, 256)

    def test_a_persistent_cublas_error_reaches_the_walks(self):
        """A cuBLAS error that repeats on every retry is passed to the waiting walks, as other errors are"""
        calls = list()

        def broken_gpu(structures: list) -> list:
            calls.append(len(structures))
            raise RuntimeError('CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle )`')

        evaluator = BatchedEvaluator(broken_gpu, max_retries=2)
        evaluator.join(2)

        def walk(n_atoms: int) -> str:
            try:
                evaluator.request('forces', Atoms('H' * n_atoms, positions=[(float(i), 0.0, 0.0)
                                                                             for i in range(n_atoms)]))
            except RuntimeError as exc:
                return str(exc)
            finally:
                evaluator.leave()

        with ThreadPoolExecutor(max_workers=2) as pool:
            errors = list(pool.map(walk, range(1, 3)))
        self.assertTrue(all('CUBLAS_STATUS_EXECUTION_FAILED' in e for e in errors))
        self.assertEqual(calls, [2, 2, 2])
        self.assertEqual(evaluator.n_retries, 2)

    def test_a_hessian_that_hits_a_transient_cublas_error_is_retried(self):
        """A Hessian request is retried after a transient cuBLAS error; other errors are passed on at once"""
        attempts = list()

        def flaky_hessian(structure: Atoms) -> np.ndarray:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError('CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle )`')
            return np.eye(3 * len(structure))

        evaluator = BatchedEvaluator(lambda structures: [], hessian=flaky_hessian)
        evaluator.join(1)
        try:
            hessian = evaluator.request('hessian', Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.74)]))
        finally:
            evaluator.leave()
        np.testing.assert_array_equal(hessian, np.eye(6))
        self.assertEqual(len(attempts), 2)
        self.assertTrue(is_transient_gpu_error(RuntimeError('CUBLAS_STATUS_INTERNAL_ERROR')))
        self.assertFalse(is_transient_gpu_error(RuntimeError('CUBLAS_STATUS_ALLOC_FAILED')))
        self.assertFalse(is_transient_gpu_error(ValueError('Unknown element Xx')))

    @staticmethod
    def toy_ts() -> tuple:
        """Ethane on a spring network, with a barrier along one C-H bond, a little off that saddle, and its Hessian."""
        atoms = Atoms(symbols=ETHANE_XYZ['symbols'], positions=ETHANE_XYZ['coords'])
        r0 = atoms.get_distance(0, 2)
        springs = SpringNetwork(atoms)
        atoms.calc = BondBarrier(springs, pair=(0, 2), r0=r0)
        atoms.positions[2] += 0.01 * (atoms.positions[2] - atoms.positions[0]) / r0
        return atoms, r0, finite_difference_hessian(BondBarrier(springs, pair=(0, 2), r0=r0))

    def assert_scan_reaches_its_dihedrals(self, result: dict, torsion: list, start: float, step_deg: float):
        """Each scan point's dihedral is the target of its grid point."""
        for n, xyz in enumerate(result['scan_coords'][:-1]):
            image = Atoms(symbols=ETHANE_XYZ['symbols'], positions=xyz)
            self.assertAlmostEqual(((image.get_dihedral(*torsion) - start - n * step_deg + 180.0) % 360.0) - 180.0,
                                   0.0, delta=0.5)

    @unittest.skipUnless(importlib.util.find_spec('sella'), 'Sella is not installed')
    def test_a_ts_scan_stays_on_the_saddle(self):
        """A TS's walks search each point for a saddle point; minimizing each point instead slides off the saddle"""
        atoms, r0, hessian_function = self.toy_ts()
        torsion, start = [3, 0, 1, 5], atoms.get_dihedral(3, 0, 1, 5)
        drift = dict()
        for saddle in (False, True):
            result = relaxed_torsion_scan(atoms, torsion, step_deg=20.0, nsteps=18, fmax=0.002, steps=300,
                                          optimizer=FIRE, saddle=saddle, hessian_function=hessian_function)
            self.assertEqual(result['top'], [1, 5, 6, 7])
            self.assert_scan_reaches_its_dihedrals(result, torsion, start, 20.0)
            drift[saddle] = max(abs(np.linalg.norm(np.subtract(xyz[0], xyz[2])) - r0) for xyz in result['scan_coords'])
        self.assertLess(drift[True], 0.02)
        self.assertGreater(drift[False], 0.5)

    def test_a_ts_walk_runs_sella_with_the_batched_hessian(self):
        """A TS walk runs an order-1 Sella search at every point, whose Hessian requests go through the evaluator
        together with the other walk's force calls"""
        calls = list()

        class FakeSella:
            def __init__(self, atoms, order, logfile=None, hessian_function=None):
                self.atoms, self.order, self.hessian_function = atoms, order, hessian_function

            def run(self, fmax, steps):
                calls.append((self.order, self.hessian_function(self.atoms).shape))
                self.atoms.get_forces()

        atoms, r0, hessian_function = self.toy_ts()
        torsion, start = [3, 0, 1, 5], atoms.get_dihedral(3, 0, 1, 5)
        evaluator = BatchedEvaluator(batch_evaluation_function(atoms.calc), hessian=hessian_function)
        with patch.dict(sys.modules, {'sella': type(sys)('sella')}):
            sys.modules['sella'].Sella = FakeSella
            result = relaxed_torsion_scan(atoms, torsion, step_deg=60.0, nsteps=6, saddle=True, evaluator=evaluator)
        self.assertEqual(len(calls), 2 * 7)
        self.assertTrue(all(call == (1, (24, 24)) for call in calls))
        self.assertTrue(result['saddle'])
        self.assert_scan_reaches_its_dihedrals(result, torsion, start, 60.0)
        self.assertEqual(evaluator._clients, 0)

    def test_a_failed_round_reaches_every_walk(self):
        """An error raised while a round is evaluated, even by the GPU memory release, ends every walk of the round"""
        for evaluate, release in ((MagicMock(side_effect=ValueError('bad batch')), None),
                                  (MagicMock(side_effect=RuntimeError('CUBLAS_STATUS_EXECUTION_FAILED')),
                                   RuntimeError('empty_cache failed'))):
            with self.subTest(evaluate=evaluate.side_effect), \
                    patch('arc.job.adapters.scripts.ase_script.release_gpu_memory', side_effect=release):
                evaluator, errors = BatchedEvaluator(evaluate), list()
                evaluator.join(2)

                def walk():
                    try:
                        evaluator.request('forces', Atoms('H'))
                    except Exception as exc:
                        errors.append(exc)
                    finally:
                        evaluator.leave()

                threads = [threading.Thread(target=walk) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)
                self.assertFalse(any(thread.is_alive() for thread in threads))
                self.assertEqual(len(errors), 2)

    def test_a_scan_whose_second_walk_cannot_start_ends(self):
        """A scan whose second walk cannot be submitted lets the first walk finish alone, and raises"""
        from concurrent.futures import ThreadPoolExecutor
        atoms = Atoms(symbols=ETHANE_XYZ['symbols'], positions=ETHANE_XYZ['coords'])
        atoms.calc = SpringNetwork(atoms)
        evaluator = BatchedEvaluator(batch_evaluation_function(atoms.calc))
        submit, outcome = ThreadPoolExecutor.submit, dict()

        def submit_once(pool, *args, **kwargs):
            if getattr(pool, 'submitted', False):
                raise RuntimeError("can't start new thread")
            pool.submitted = True
            return submit(pool, *args, **kwargs)

        def scan():
            try:
                relaxed_torsion_scan(atoms, [2, 0, 1, 5], step_deg=60.0, nsteps=6, evaluator=evaluator)
            except RuntimeError as exc:
                outcome['error'] = exc

        with patch.object(ThreadPoolExecutor, 'submit', submit_once):
            thread = threading.Thread(target=scan)
            thread.start()
            thread.join(timeout=60)
        self.assertFalse(thread.is_alive())
        self.assertIn("can't start new thread", str(outcome.get('error')))
        self.assertEqual(evaluator._clients, 0)

    def test_run_torsion_scan_output_is_parseable(self):
        """Test that run_torsion_scan output is readable by ARC's YAML scan parser, in kJ/mol"""
        atoms = Atoms(symbols=ETHANE_XYZ['symbols'], positions=ETHANE_XYZ['coords'])
        atoms.calc = EMT()
        input_dict = {'torsions': [[2, 0, 1, 5]], 'scan_res': 8.0}
        result = run_torsion_scan(atoms, input_dict, settings={'fmax': 0.05, 'steps': 100})
        self.assertEqual(len(result['energies']), 46)
        scan_dir = os.path.join(self.project_directory, 'scan_output')
        os.makedirs(scan_dir, exist_ok=True)
        out_path = os.path.join(scan_dir, 'output.yml')
        save_yaml_file(out_path, {'energies': result['energies'], 'angles': result['angles']})
        energies, angles = parse_1d_scan_energies(log_file_path=out_path)
        self.assertIsNotNone(energies)
        self.assertIsNotNone(angles)
        self.assertEqual(len(energies), 46)
        self.assertEqual(len(angles), 46)
        self.assertAlmostEqual(min(energies), 0.0)  # parser zeroes the minimum, in kJ/mol
        self.assertGreaterEqual(min(energies), 0.0)

    def test_run_torsion_scan_rejects_multiple_torsions(self):
        """Test that a 1D scan refuses more than one torsion"""
        atoms = Atoms(symbols=ETHANE_XYZ['symbols'], positions=ETHANE_XYZ['coords'])
        atoms.calc = EMT()
        with self.assertRaises(ValueError):
            run_torsion_scan(atoms, {'torsions': [[2, 0, 1, 5], [3, 0, 1, 6]], 'scan_res': 8.0}, settings={})
        with self.assertRaises(ValueError):
            run_torsion_scan(atoms, {'torsions': None, 'scan_res': 8.0}, settings={})
        # A non-positive scan resolution is rejected before the 360/scan_res division.
        with self.assertRaises(ValueError):
            run_torsion_scan(atoms, {'torsions': [[2, 0, 1, 5]], 'scan_res': 0.0}, settings={})
        # A resolution that does not close the revolution would corrupt the periodic grid.
        with self.assertRaises(ValueError):
            run_torsion_scan(atoms, {'torsions': [[2, 0, 1, 5]], 'scan_res': 7.0}, settings={})

    def test_merge_scan_branches(self):
        """Test folding the forward and backward walks onto one grid and keeping the lower branch"""
        nsteps = 4
        # The forward walk is trapped high from its second point on; the backward walk (which
        # visits the grid in reverse) is the low branch there. The merge must take each pointwise.
        e_f = [0.0, 1.0, 9.0, 9.0, 0.0]
        e_b = [0.0, 5.0, 2.0, 3.0, 0.0]
        merged, grid_f, grid_b, coords = merge_scan_branches(e_f, e_b, nsteps)
        # Backward index n lands on grid point (-n) % 4, so e_b maps onto [0.0, 3.0, 2.0, 5.0].
        self.assertEqual(list(grid_f), [0.0, 1.0, 9.0, 9.0])
        self.assertEqual(list(grid_b), [0.0, 3.0, 2.0, 5.0])
        self.assertEqual(list(merged), [0.0, 1.0, 2.0, 5.0])
        self.assertIsNone(coords)  # no geometries supplied
        # The endpoint of each walk folds back onto grid point 0 and must not lose a lower value.
        merged, _, _, _ = merge_scan_branches([5.0, 1.0, 2.0, 3.0, 0.0], [5.0, 6.0, 7.0, 8.0, 9.0], nsteps)
        self.assertEqual(merged[0], 0.0)
        # Geometries follow whichever branch won at each grid point.
        coords_f = [('f', n) for n in range(nsteps + 1)]
        coords_b = [('b', n) for n in range(nsteps + 1)]
        merged, _, _, coords = merge_scan_branches(e_f, e_b, nsteps, coords_f, coords_b)
        self.assertEqual(coords[1], ('f', 1))
        merged, _, _, coords = merge_scan_branches(e_f, e_b, nsteps, coords_f, coords_b,
                                                   converged_f=[True, False, True, True, True],
                                                   converged_b=[True] * 5)
        self.assertEqual(merged[1], 3.0)
        self.assertEqual(coords[1], ('b', 3))
        merged, _, _, _ = merge_scan_branches(e_f, e_b, nsteps, converged_f=[True, False, True, True, True],
                                              converged_b=[True, True, True, False, True])
        self.assertEqual(merged[1], 1.0)  # forward won here (1.0 < 3.0)
        self.assertEqual(coords[2], ('b', 2))  # backward won here (2.0 < 9.0)
        self.assertEqual(coords[3], ('b', 1))  # backward index 1 folds onto grid point 3

    def test_scan_coords_are_parseable(self):
        """Test that the relaxed scan geometries round-trip through ARC's YAML coords parser"""
        atoms = Atoms(symbols=ETHANE_XYZ['symbols'], positions=ETHANE_XYZ['coords'])
        atoms.calc = EMT()
        input_dict = {'torsions': [[2, 0, 1, 5]], 'scan_res': 40.0, 'xyz': ETHANE_XYZ}
        result = run_torsion_scan(atoms, input_dict, settings={'fmax': 0.05, 'steps': 50})
        self.assertEqual(len(result['scan_coords']), 10)
        self.assertEqual(result['scan_coords'][0]['symbols'], ETHANE_XYZ['symbols'])
        self.assertEqual(result['scan_coords'][0]['isotopes'], ETHANE_XYZ['isotopes'])
        scan_dir = os.path.join(self.project_directory, 'scan_coords')
        os.makedirs(scan_dir, exist_ok=True)
        out_path = os.path.join(scan_dir, 'output.yml')
        save_yaml_file(out_path, {'energies': result['energies'], 'angles': result['angles'],
                                  'scan_coords': result['scan_coords']})
        traj = parse_1d_scan_coords(log_file_path=out_path)
        self.assertIsNotNone(traj)
        self.assertEqual(len(traj), 10)
        self.assertEqual(len(traj[0]['coords']), len(ETHANE_XYZ['symbols']))

    def test_scan_convergence_warning(self):
        """Test that only an untrustworthy scan is called out"""
        clean = {'converged': True, 'fmax_worst': 0.004, 'branch_gap_max': 0.001}
        self.assertIsNone(scan_convergence_warning(clean, fmax=0.005))
        loose = {'converged': False, 'fmax_worst': 0.9, 'branch_gap_max': 0.001}
        self.assertIn('0.9', scan_convergence_warning(loose, fmax=0.005))
        split = {'converged': True, 'fmax_worst': 0.004, 'branch_gap_max': 0.5}
        self.assertIn('different conformers', scan_convergence_warning(split, fmax=0.005))

    def test_set_scan_torsions_from_rotors_dict(self):
        """Test that a scheduler-dispatched rotor (rotors_dict + rotor_index, torsions None) resolves"""
        scan_dir = os.path.join(self.project_directory, 'scan_rotor_index')
        os.makedirs(scan_dir, exist_ok=True)
        spc = ARCSpecies(label='ethane', xyz=ETHANE_XYZ)
        spc.determine_rotors()
        job = ASEAdapter(execution_type='incore',
                         job_type='scan',
                         project='test_scan',
                         project_directory=scan_dir,
                         species=[spc],
                         rotor_index=0,
                         args={'keyword': {'calculator': 'uma', 'model': 'uma-s-1p2'}},
                         testing=True)
        self.assertIsNone(job.torsions)  # the scheduler leaves this unset
        job.local_path = scan_dir
        job.write_input_file()
        # The scan is resolved from rotors_dict and back-filled, since the scheduler later reads
        # job.torsions[0] directly when the job returns.
        expected = [[atom - 1 for atom in spc.rotors_dict[0]['scan']]]
        self.assertEqual(job.torsions, expected)
        data = read_yaml_file(os.path.join(scan_dir, 'input.yml'))
        self.assertEqual(data['torsions'], expected)
        self.assertIsNotNone(data['torsions'][0])
        self.assertEqual(data['top'], [atom - 1 for atom in spc.rotors_dict[0]['top']])
        self.assertNotIn('partial_bonds', data)
        spc.scan_partial_bonds = [[1, 3]]
        job.write_input_file()
        self.assertEqual(read_yaml_file(os.path.join(scan_dir, 'input.yml'))['partial_bonds'], [[0, 2]])


WATER_POSITIONS = ((0.0, 0.0, 0.0), (0.0, 0.76, 0.59), (0.0, -0.76, 0.59))


class TestASEWorker(unittest.TestCase):
    """
    Contains unit tests for the persistent ASE worker: its spool, and running jobs through it.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spool = os.path.join(self.tmp.name, 'spool')
        spool_dirs(self.spool)

    def tearDown(self):
        self.tmp.cleanup()

    def write_job(self, name: str, job_type: str = 'sp', settings: dict | None = None, **fields) -> str:
        """Write an ethane job's input.yml (EMT unless ``settings`` say otherwise) and return its path."""
        job_dir = os.path.join(self.tmp.name, name)
        os.makedirs(job_dir)
        input_path = os.path.join(job_dir, 'input.yml')
        save_yaml_file(input_path, {'job_type': job_type, 'xyz': ETHANE_XYZ, 'charge': 0, 'multiplicity': 1,
                                    'settings': settings or {'calculator': 'emt'}, **fields})
        return input_path

    def write_scan(self, name: str) -> str:
        """Write a quick EMT scan job of ethane's C-C torsion and return its input.yml path."""
        return self.write_job(name, 'scan', settings={'calculator': 'emt', 'fmax': 0.05, 'steps': 100},
                              torsions=[[2, 0, 1, 5]], scan_res=8.0)

    def test_scan_group_key(self):
        """Only UMA scans with the same settings share a key"""
        uma = {'calculator': 'uma', 'model': 'uma-s-1p2', 'device': 'cuda', 'inference_settings': 'traineval'}
        key = scan_group_key(self.write_job('a', 'scan', settings=uma))
        self.assertIsNotNone(key)
        self.assertEqual(scan_group_key(self.write_job('b', 'scan', settings=dict(uma))), key)
        self.assertNotEqual(scan_group_key(self.write_job('c', 'scan', settings=dict(uma, model='uma-m-1p1'))), key)
        self.assertIsNone(scan_group_key(self.write_job('d', 'sp', settings=uma)))
        self.assertIsNone(scan_group_key(self.write_scan('e')))
        self.assertIsNone(scan_group_key(os.path.join(self.tmp.name, 'missing', 'input.yml')))

    def test_a_ticket_claim_skips_what_it_does_not_accept(self):
        """A claim with a filter takes the oldest acceptable ticket and leaves the others queued"""
        first = submit_ticket(self.spool, '/a/input.yml')
        second = submit_ticket(self.spool, '/b/input.yml')
        ticket, _ = claim_ticket(self.spool, 'w1', accept=lambda content: content['input_path'] == '/b/input.yml')
        self.assertEqual(ticket, second)
        self.assertEqual(get_ticket_status(self.spool, first)[0], 'queued')
        self.assertIsNone(claim_ticket(self.spool, 'w1', accept=lambda content: False))

    def test_scans_run_together_share_each_force_call(self):
        """The walks of scans run as a group all reach the calculator together, and each scan finishes its ticket"""
        group = list()
        for name in ('scan_1', 'scan_2'):
            submit_ticket(self.spool, self.write_scan(name), label=name)
            group.append(claim_ticket(self.spool, 'w1'))
        batch_sizes, make_function = list(), ase_script.batch_evaluation_function

        def recording_function(calc):
            evaluate = make_function(calc)

            def recording_evaluate(structures: list) -> list:
                batch_sizes.append(len(structures))
                return evaluate(structures)

            return recording_evaluate

        with patch.object(ase_script, 'batch_evaluation_function', recording_function):
            results = run_scan_group(self.spool, group, 'w1', dict())
        self.assertEqual([result['status'] for result in results], ['done', 'done'])
        self.assertEqual(max(batch_sizes), 4)
        for (ticket, content), result in zip(group, results):
            self.assertEqual(get_ticket_status(self.spool, ticket)[0], 'done')
            output = read_yaml_file(os.path.join(os.path.dirname(content['input_path']), 'output.yml'))
            self.assertEqual(len(output['energies']), 46)

    def test_serve_groups_scans_and_runs_other_jobs_alone(self):
        """A worker that claims a groupable scan also claims the other queued scans with its key"""
        for name in ('scan_1', 'scan_2'):
            submit_ticket(self.spool, self.write_scan(name), label=name)
        submit_ticket(self.spool, self.write_job('sp_1'), label='sp_1')
        groups, run_group = list(), ase_script.run_scan_group

        def recording_run_group(spool_dir, group, worker_id, cache):
            groups.append(len(group))
            return run_group(spool_dir, group, worker_id, cache)

        with patch.object(ase_script, 'scan_group_key', lambda path: 'emt' if 'scan' in path else None), \
                patch.object(ase_script, 'run_scan_group', recording_run_group):
            self.assertEqual(serve(self.spool, max_jobs=3, scan_group_wait=0.1), 3)
        self.assertEqual(groups, [2])
        self.assertEqual(len(os.listdir(os.path.join(self.spool, 'done'))), 3)

    def beat(self, worker_id: str, age: float = 0.0) -> None:
        """Write a worker heartbeat that is ``age`` seconds old."""
        path = os.path.join(self.spool, 'workers', f'{worker_id}.json')
        with open(path, 'w') as f:
            f.write('{}')
        then = time.time() - age
        os.utime(path, (then, then))

    def make_job(self, execution_type: str = 'queue', calculator: str = 'emt', **kwargs) -> ASEAdapter:
        """Make an ethane single-point job with a worker spool configured for its calculator (EMT by default)."""
        if 'level' not in kwargs:
            kwargs['args'] = {'keyword': {'calculator': calculator}}
        with patch.dict('arc.job.adapters.ase_adapter.settings', {'ASE_WORKERS': {calculator: self.spool}}):
            job = ASEAdapter(execution_type=execution_type,
                             job_type='sp',
                             project='test_worker',
                             project_directory=os.path.join(self.tmp.name, 'project'),
                             species=[ARCSpecies(label='ethane', xyz=ETHANE_XYZ)],
                             testing=True,
                             **kwargs)
        job.local_path = os.path.join(self.tmp.name, 'job')
        os.makedirs(job.local_path, exist_ok=True)
        return job

    def test_run_vibrational_analysis_ignores_a_stale_cache(self):
        """Displacement forces left in the job directory, e.g., by an interrupted job, are not reused"""
        clean_dir, stale_dir = os.path.join(self.tmp.name, 'clean'), os.path.join(self.tmp.name, 'stale')
        os.makedirs(clean_dir)
        os.makedirs(stale_dir)
        other = Atoms('OH2', positions=((0.0, 0.0, 0.0), (0.0, 0.9, 0.4), (0.0, -0.9, 0.4)))
        other.calc = EMT()
        Vibrations(other, name=os.path.join(stale_dir, 'vib_tmp'), nfree=4).run()
        results = list()
        for work_dir in (clean_dir, stale_dir):
            water = Atoms('OH2', positions=WATER_POSITIONS)
            water.calc = EMT()
            results.append(run_vibrational_analysis(water, {'calculator': 'emt'}, work_dir=work_dir))
        np.testing.assert_allclose(results[1]['freqs'], results[0]['freqs'])
        self.assertEqual(os.listdir(os.path.join(stale_dir, 'vib_tmp')), list())
        self.assertFalse(os.path.exists(os.path.join(os.getcwd(), 'vib_tmp')))

    def test_a_claimed_ticket_cannot_be_claimed_or_withdrawn(self):
        """Only one worker gets a ticket, and a running ticket stays with it"""
        ticket = submit_ticket(self.spool, '/path/to/input.yml', label='sp_a1')
        self.assertTrue(ticket.startswith('sp_a1_'))
        self.assertEqual(get_ticket_status(self.spool, ticket)[0], 'queued')
        claimed_ticket, content = claim_ticket(self.spool, 'w1')
        self.assertEqual(claimed_ticket, ticket)
        self.assertEqual(content['worker'], 'w1')
        self.assertEqual(content['input_path'], '/path/to/input.yml')
        self.assertIsNone(claim_ticket(self.spool, 'w2'))
        self.assertFalse(withdraw_ticket(self.spool, ticket))
        self.assertEqual(get_ticket_status(self.spool, ticket)[0], 'running')
        other = submit_ticket(self.spool, '/path/to/other/input.yml')
        self.assertTrue(withdraw_ticket(self.spool, other))
        self.assertEqual(get_ticket_status(self.spool, other), ('missing', None))

    def test_an_irc_job_ends_at_the_end_of_its_path(self):
        """An IRC job's final geometry is the last point of its path, from which ARC builds the IRC species"""
        from ase.io.trajectory import Trajectory
        calls = dict()

        class FakeIRC:
            def __init__(self, atoms, logfile=None, trajectory=None, **kwargs):
                self.atoms, self.trajectory = atoms, trajectory
                calls['kwargs'] = kwargs

            def run(self, fmax, steps, direction):
                calls['fmax'] = fmax
                with Trajectory(self.trajectory, 'w') as traj:
                    for shift in (0.0, 0.1, 0.2):
                        image = Atoms(self.atoms.get_chemical_symbols(), positions=self.atoms.get_positions())
                        image.positions[0, 0] += shift
                        traj.write(image)

        fake_sella = type(sys)('sella')
        fake_sella.IRC = FakeIRC
        with patch.dict(sys.modules, {'sella': fake_sella}):
            output = run_job(self.write_job('irc', job_type='irc'))
        self.assertEqual(len(output['irc_traj']), 3)
        self.assertAlmostEqual(output['opt_xyz']['coords'][0][0], ETHANE_XYZ['coords'][0][0] + 0.2)
        self.assertEqual(output['opt_xyz'], output['irc_traj'][-1])
        self.assertEqual(calls['fmax'], 0.01)
        self.assertTrue(calls['kwargs']['keep_going'])
        self.assertNotIn('irc_complete', output)
        self.assertNotIn('warnings', output)

    def test_live_workers_ignores_a_stale_heartbeat(self):
        """A worker whose heartbeat stopped is presumed dead"""
        self.beat('w1')
        self.beat('w2', age=600)
        self.assertEqual(live_workers(self.spool), ['w1'])

    def test_serve_runs_tickets_and_survives_a_failed_job(self):
        """The worker writes each job's output.yml and records a failure without stopping"""
        good = submit_ticket(self.spool, self.write_job('good'), label='sp_a1')
        bad = submit_ticket(self.spool, os.path.join(self.tmp.name, 'missing', 'input.yml'), label='sp_a2')
        self.assertEqual(serve(self.spool, max_jobs=2), 2)
        state, content = get_ticket_status(self.spool, good)
        self.assertEqual(state, 'done')
        output = read_yaml_file(os.path.join(self.tmp.name, 'good', 'output.yml'))
        self.assertIn('sp', output)
        state, content = get_ticket_status(self.spool, bad)
        self.assertEqual(state, 'errored')
        self.assertIn('FileNotFoundError', content['error'])
        self.assertEqual(os.listdir(os.path.join(self.spool, 'running')), list())
        self.assertEqual(live_workers(self.spool), list())

    def test_no_live_worker_leaves_the_execution_type_alone(self):
        """Without a live worker, a job runs as the scheduler asked"""
        job = self.make_job(execution_type='queue')
        self.assertIsNone(job.worker_spool)
        self.assertEqual(job.execution_type, 'queue')

    def test_a_live_worker_takes_jobs_without_blocking(self):
        """With a live worker, a job is submitted as a local queue job and stays in flight until its ticket ends"""
        self.beat('w1')
        jobs = [self.make_job() for _ in range(3)]
        for job in jobs:
            self.assertEqual(job.worker_spool, self.spool)
            self.assertEqual(job.execution_type, 'queue')
            self.assertEqual(job.server, 'local')
            job.execute_queue()
            self.assertTrue(job.is_worker_ticket())
            self.assertEqual(job.job_status[0], 'running')
        with patch.dict('arc.job.adapters.ase_adapter.settings', {'ASE_WORKERS': {'emt': self.spool}}):
            self.assertEqual(sorted(active_worker_tickets()), sorted(job.job_id for job in jobs))
        self.assertEqual([job._check_job_server_status() for job in jobs], ['running'] * 3)
        self.assertFalse(os.path.isfile(os.path.join(jobs[0].local_path, 'submit.sh')))

    def test_a_worker_job_ends_when_its_ticket_does(self):
        """A real worker runs the ticket; the job then reports done, with the worker's start and end times"""
        script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'ase_script.py')
        worker = subprocess.Popen([sys.executable, script_path, '--serve', self.spool,
                                   '--max_jobs', '1', '--idle_timeout', '120'],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.time() + 60
            while not live_workers(self.spool) and time.time() < deadline:
                time.sleep(0.1)
            job = self.make_job(execution_type='queue')
            job.execute_queue()
            worker.wait(timeout=60)
            self.assertEqual(worker.returncode, 0, worker.stdout.read())
            with patch.dict('arc.job.adapters.ase_adapter.settings', {'ASE_WORKERS': {'emt': self.spool}}):
                self.assertEqual(active_worker_tickets(), list())
            self.assertEqual(job._check_job_server_status(), 'done')
            self.assertLessEqual(job.initial_time, job.final_time)
            self.assertIn('sp', read_yaml_file(os.path.join(job.local_path, 'output.yml')))
        finally:
            if worker.poll() is None:
                worker.kill()
            worker.stdout.close()

    def test_a_ticket_no_worker_is_left_to_take_runs_directly(self):
        """A queued ticket whose workers all died is withdrawn, and the job runs directly"""
        self.beat('w1')
        job = self.make_job()
        job.execute_queue()
        self.beat('w1', age=600)
        with patch.object(ASEAdapter, 'run_directly') as run_directly, self.assertLogs('arc', level='WARNING'):
            self.assertEqual(job._check_job_server_status(), 'done')
        run_directly.assert_called_once()
        self.assertEqual(os.listdir(os.path.join(self.spool, 'queue')), list())

    def test_a_job_whose_worker_died_runs_directly(self):
        """A running ticket whose worker died is marked errored in the spool, and the job runs directly"""
        self.beat('w1')
        job = self.make_job()
        job.execute_queue()
        claim_ticket(self.spool, 'w2')
        with patch.object(ASEAdapter, 'run_directly') as run_directly, self.assertLogs('arc', level='WARNING'):
            self.assertEqual(job._check_job_server_status(), 'done')
        run_directly.assert_called_once()
        state, content = get_ticket_status(self.spool, job.job_id)
        self.assertEqual(state, 'done')
        self.assertTrue(content['ran_directly'])

    def test_a_queue_job_id_is_not_a_ticket(self):
        """A job ID from a cluster queue is checked with the queue, not the spool"""
        self.beat('w1')
        job = self.make_job()
        job.job_id = '123456'
        self.assertFalse(job.is_worker_ticket())
        job.job_id = f'{job.job_name}_0123456789ab'
        self.assertTrue(job.is_worker_ticket())

    def write_ticket(self, directory: str, ticket: str, **content) -> None:
        """Write a ticket file into a spool directory ('queue' or 'running')."""
        write_json_atomic(os.path.join(self.spool, directory, f'{ticket}.json'),
                          {'input_path': '/path/to/input.yml', 'submitted': time.time(), **content},
                          os.path.join(self.spool, 'tmp'))

    def test_live_tickets(self):
        """A ticket is live while a live worker can claim or runs it, and only until its deadline"""
        self.beat('w1')
        self.write_ticket('queue', 'queued')
        self.write_ticket('queue', 'queued_past_deadline', deadline=time.time() - 1)
        self.write_ticket('running', 'running_on_w1', worker='w1')
        self.write_ticket('running', 'running_on_dead_w2', worker='w2')
        self.write_ticket('running', 'being_claimed')
        self.write_ticket('running', 'running_past_deadline', worker='w1', deadline=time.time() - 1)
        self.assertEqual(sorted(live_tickets(self.spool)), ['being_claimed', 'queued', 'running_on_w1'])
        self.beat('w1', age=600)
        self.assertEqual(live_tickets(self.spool), list())

    def test_an_incore_job_runs_without_the_worker(self):
        """A job requested to run incore runs incore, even with a live worker"""
        self.beat('w1')
        job = self.make_job(execution_type='incore')
        self.assertIsNone(job.worker_spool)
        self.assertEqual(job.execution_type, 'incore')

    def test_a_restored_worker_job_keeps_its_spool(self):
        """A restored job whose ID is its worker ticket keeps the spool with no live worker, also for a level dict"""
        job = self.make_job(calculator='uma', level={'method': 'uma'}, job_name='sp_a7', job_id='sp_a7_0123456789ab',
                            server='local')
        self.assertEqual(job.worker_spool, self.spool)
        self.assertTrue(job.is_worker_ticket())
        self.assertEqual(job.execution_type, 'queue')

    def test_a_restored_queue_job_keeps_its_server(self):
        """A restored job with a cluster queue ID stays on its server, even with a live worker"""
        self.beat('w1')
        job = self.make_job(job_name='sp_a7', job_id=123456, server='server1')
        self.assertIsNone(job.worker_spool)
        self.assertEqual(job.server, 'server1')

    def test_delete_withdraws_a_queued_ticket(self):
        """Deleting a worker job withdraws its queued ticket, and leaves a running one to its worker"""
        self.beat('w1')
        job = self.make_job()
        job.execute_queue()
        with patch('arc.job.adapter.delete_job') as delete_job:
            job.delete()
        delete_job.assert_not_called()
        self.assertEqual(get_ticket_status(self.spool, job.job_id)[0], 'missing')
        job.execute_queue()
        claim_ticket(self.spool, 'w1')
        with patch('arc.job.adapter.delete_job') as delete_job, self.assertLogs('arc', level='WARNING'):
            job.delete()
        delete_job.assert_not_called()
        self.assertEqual(get_ticket_status(self.spool, job.job_id)[0], 'running')

    def test_a_job_run_directly_is_not_run_again(self):
        """A job run directly after its ticket vanished is marked done, so a second status check does not rerun it"""
        self.beat('w1')
        job = self.make_job()
        job.execute_queue()
        withdraw_ticket(self.spool, job.job_id)
        with patch.object(ASEAdapter, 'run_directly') as run_directly, self.assertLogs('arc', level='WARNING'):
            self.assertEqual(job._check_job_server_status(), 'done')
        self.assertEqual(job._check_job_server_status(), 'done')
        run_directly.assert_called_once()

    def test_a_ticket_past_its_deadline_errors(self):
        """A ticket that did not finish within the job's max_job_time is withdrawn and the job errors"""
        self.beat('w1')
        job = self.make_job()
        job.max_job_time = -1.0
        job.execute_queue()
        with patch.dict('arc.job.adapters.ase_adapter.settings', {'ASE_WORKERS': {'emt': self.spool}}):
            self.assertEqual(active_worker_tickets(), list())
        with self.assertLogs('arc', level='ERROR'):
            self.assertEqual(job._check_job_server_status(), 'errored')
        self.assertEqual(os.listdir(os.path.join(self.spool, 'queue')), list())

    def test_submitting_to_a_worker_removes_an_earlier_output(self):
        """An output.yml left in the job directory by an earlier run is removed when the job is submitted"""
        self.beat('w1')
        job = self.make_job()
        save_yaml_file(job.local_path_to_output_file, {'sp': 1.0})
        job.execute_queue()
        self.assertFalse(os.path.isfile(job.local_path_to_output_file))

    def test_run_neb(self):
        """A 'neb' job returns the highest interior image of the band as an ARC TS guess (EMT; NEB, DyNEB, AutoNEB)."""
        xyz = {'symbols': ('H', 'C', 'N'), 'isotopes': (1, 12, 14),
               'coords': ((-1.07, 0.0, 0.0), (0.0, 0.0, 0.0), (1.16, 0.0, 0.0))}
        product_xyz = {'symbols': ('H', 'C', 'N'), 'isotopes': (1, 12, 14),
                       'coords': ((2.15, 0.3, 0.0), (0.0, 0.0, 0.0), (1.17, 0.0, 0.0))}
        for method in ('neb', 'dyneb'):
            job_dir = tempfile.mkdtemp(prefix=f'test_run_neb_{method}_')
            self.addCleanup(shutil.rmtree, job_dir, ignore_errors=True)
            input_dict = {'job_type': 'neb', 'xyz': xyz, 'product_xyz': product_xyz, 'charge': 0, 'multiplicity': 1,
                          'settings': {'calculator': 'emt', 'neb_method': method, 'neb_images': 3, 'neb_steps': 20}}
            input_path = os.path.join(job_dir, 'input.yml')
            save_yaml_file(input_path, input_dict)
            output = run_job(input_path, input_dict=input_dict)
            self.assertNotIn('error', output)
            self.assertEqual(len(output['neb_energies']), 5)
            self.assertEqual(output['neb_energies'][0], 0.0)
            self.assertIn(output['neb_highest_image'], (1, 2, 3))
            guess = output['ts_guesses'][0]
            self.assertEqual(guess['method'], 'ase_neb')
            self.assertEqual(tuple(guess['initial_xyz']['symbols']), xyz['symbols'])
            self.assertEqual(output['opt_xyz']['coords'], guess['initial_xyz']['coords'])
            self.assertGreater(output['neb_force_calls'], 0)
        job_dir = tempfile.mkdtemp(prefix='test_run_neb_bands_')
        self.addCleanup(shutil.rmtree, job_dir, ignore_errors=True)
        flipped = {'symbols': xyz['symbols'], 'isotopes': xyz['isotopes'],
                   'coords': ((-2.15, 0.3, 0.0), (0.0, 0.0, 0.0), (1.17, 0.0, 0.0))}
        two = dict(input_dict, settings=dict(input_dict['settings'], neb_method='neb'),
                   bands=[{'xyz': xyz, 'product_xyz': product_xyz}, {'xyz': xyz, 'product_xyz': flipped}])
        output = run_job(os.path.join(job_dir, 'input.yml'), input_dict=two)
        self.assertEqual(sorted(g['method_index'] for g in output['ts_guesses']), [0, 1])
        self.assertEqual(len(output['neb_bands']), 2)
        self.assertIn(output['neb_best_band'], (0, 1))
        self.assertTrue(os.path.isfile(os.path.join(job_dir, 'band_1', 'neb.log')))
        bad = dict(input_dict, product_xyz={'symbols': ('H', 'N', 'C'), 'coords': product_xyz['coords']})
        job_dir = tempfile.mkdtemp(prefix='test_run_neb_bad_')
        self.addCleanup(shutil.rmtree, job_dir, ignore_errors=True)
        output = run_job(os.path.join(job_dir, 'input.yml'), input_dict=bad)
        self.assertIn('error', output)

    @staticmethod
    def band_job(well: str = 'double', start: float = 1.0, end: float = 3.0) -> tuple:
        """A diatomic on a bond potential with a 1 eV barrier at 2.2 A between wells at 1.0 and 3.4 A ('double'), or
        with a single well at 2.2 A ('single'), and the band's endpoints at bond lengths ``start`` and ``end``."""
        atoms = Atoms('H2', positions=[(0.0, 0.0, 0.0), (start, 0.0, 0.0)])
        atoms.calc = PairCalculator({(0, 1): ('double_well', 2.2, 1.2, 1.0 if well == 'double' else -1.0)})
        endpoint = lambda r: {'symbols': ('H', 'H'), 'isotopes': (1, 1), 'coords': ((0.0, 0.0, 0.0), (r, 0.0, 0.0))}
        return atoms, endpoint(start), endpoint(end)

    def test_run_band_with_autoneb(self):
        """AutoNEB climbs to the saddle of a barrier, and its convergence is checked at its end"""
        atoms, xyz, product_xyz = self.band_job()
        settings = {'neb_method': 'autoneb', 'neb_images': 5, 'neb_steps': 300, 'neb_relax_endpoints': False}
        with tempfile.TemporaryDirectory() as job_dir:
            output = run_band(atoms, xyz, product_xyz, settings, job_dir)
        self.assertFalse(any('AutoNEB stopped' in w for w in output.get('warnings', list())), output.get('warnings'))
        self.assertTrue(output['neb_converged'])
        self.assertTrue(output['ts_guesses'][0]['success'])
        highest = np.array(output['neb_path'][output['neb_highest_image']]['coords'])
        self.assertAlmostEqual(float(np.linalg.norm(highest[1] - highest[0])), 2.2, delta=0.05)
        self.assertAlmostEqual(output['neb_energies'][output['neb_highest_image']], to_kJmol(1.0), delta=2.0)

        with tempfile.TemporaryDirectory() as job_dir:
            output = run_band(atoms, xyz, product_xyz, dict(settings, neb_steps=2), job_dir)
        self.assertFalse(output['neb_converged'])
        self.assertTrue(any('above fmax' in w for w in output['warnings']))

    def test_a_band_without_a_barrier_gives_no_successful_guess(self):
        """A band whose interior is below both endpoints reports its highest interior image as unsuccessful"""
        atoms, xyz, product_xyz = self.band_job(well='single')
        with tempfile.TemporaryDirectory() as job_dir:
            output = run_band(atoms, xyz, product_xyz, {'neb_images': 3, 'neb_steps': 100,
                                                        'neb_relax_endpoints': False}, job_dir)
        self.assertFalse(output['ts_guesses'][0]['success'])
        self.assertTrue(any('no interior maximum' in w for w in output['warnings']))

    def test_run_band_relaxes_its_endpoints(self):
        """A band starts from its endpoints relaxed onto the wells of its potential"""
        atoms, xyz, product_xyz = self.band_job(start=1.2, end=2.8)
        structure = {'bonds': [(0, 1)]}
        with tempfile.TemporaryDirectory() as job_dir:
            output = run_band(atoms, xyz, product_xyz, {'neb_images': 3, 'neb_steps': 200}, job_dir,
                              reactant_structure=structure, product_structure=structure)
        for index, length in ((0, 1.0), (-1, 3.4)):
            coords = np.array(output['neb_path'][index]['coords'])
            self.assertAlmostEqual(float(np.linalg.norm(coords[1] - coords[0])), length, delta=0.02)

    def test_relax_endpoint(self):
        """A one-molecule endpoint keeps its bonds or stays unrelaxed; a two-molecule one is relaxed fragment by
        fragment and keeps its contact"""
        image = Atoms('H4', positions=[(0.0, 0.0, 0.0), (1.3, 0.0, 0.0), (4.3, 0.0, 0.0), (5.6, 0.0, 0.0)],
                      tags=[1, 2, 3, 4])
        calc = PairCalculator({(1, 2): ('harmonic', 1.0, 10.0), (3, 4): ('harmonic', 1.0, 10.0),
                               (2, 3): ('harmonic', 1.0, 10.0)})

        def attach(images):
            for each in images:
                each.calc = ImageCalculator(calc, dict())

        with tempfile.TemporaryDirectory() as job_dir:
            log = os.path.join(job_dir, 'relax.log')
            fragments = [{'atoms': [0, 1], 'charge': 0, 'multiplicity': 1},
                         {'atoms': [2, 3], 'charge': 0, 'multiplicity': 1}]
            relaxed, warning = relax_endpoint(image, attach, log, fragments=fragments)
            self.assertIsNone(warning)
            self.assertAlmostEqual(relaxed.get_distance(0, 1), 1.0, delta=0.02)
            self.assertAlmostEqual(relaxed.get_distance(2, 3), 1.0, delta=0.02)
            self.assertAlmostEqual(relaxed.get_distance(1, 2), 3.3, delta=0.05)

            relaxed, warning = relax_endpoint(image, attach, log, bonds=[(0, 1), (2, 3)])
            self.assertIsNone(warning)
            self.assertAlmostEqual(relaxed.get_distance(1, 2), 1.0, delta=0.05)
            stretched = PairCalculator({(1, 2): ('harmonic', 4.0, 10.0)})
            attach_stretched = lambda images: [setattr(each, 'calc', ImageCalculator(stretched, dict())) for each in images]
            kept, warning = relax_endpoint(image, attach_stretched, log, bonds=[(0, 1)])
            self.assertIn('broke', warning)
            np.testing.assert_allclose(kept.get_positions(), image.get_positions())


class PairCalculator(Calculator):
    """
    A test calculator summing pair potentials over pairs of atoms, keyed by their tags (by their indices if every tag
    is 0): ``('harmonic', r0, k)``, or ``('double_well', c, w, h)``, h((r - c)^2 / w^2 - 1)^2.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, pairs: dict, **kwargs):
        super().__init__(**kwargs)
        self.pairs = pairs

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions, tags = self.atoms.get_positions(), self.atoms.get_tags()
        keys = list(tags) if any(tags) else list(range(len(self.atoms)))
        energy, forces = 0.0, np.zeros_like(positions)
        for (key_i, key_j), (kind, *params) in self.pairs.items():
            if key_i not in keys or key_j not in keys:
                continue
            i, j = keys.index(key_i), keys.index(key_j)
            vector = positions[i] - positions[j]
            r = float(np.linalg.norm(vector))
            if kind == 'harmonic':
                r0, k = params
                energy, slope = energy + 0.5 * k * (r - r0) ** 2, k * (r - r0)
            else:
                c, w, h = params
                u = (r - c) / w
                energy, slope = energy + h * (u ** 2 - 1) ** 2, 4.0 * h * u * (u ** 2 - 1) / w
            forces[i] -= slope * vector / r
            forces[j] += slope * vector / r
        self.results = {'energy': energy, 'forces': forces}


class TestASECalculators(unittest.TestCase):
    """
    Contains unit tests for the MLIP calculators of the ASE script, run on stand-ins for their packages.
    """

    @staticmethod
    def modules(**modules) -> dict:
        """Stand-in modules by dotted name, with their parent packages, for patching sys.modules."""
        result = dict()
        for name, attributes in modules.items():
            parts = name.split('.')
            for i in range(1, len(parts)):
                result.setdefault('.'.join(parts[:i]), type(sys)('.'.join(parts[:i])))
            module = type(sys)(name)
            for key, value in attributes.items():
                setattr(module, key, value)
            result[name] = module
        return result

    def test_mace_model(self):
        """MACE models are named by family and size, and a checkpoint needs its family"""
        from arc.job.adapters.scripts.ase_script import mace_model
        self.assertEqual(mace_model({}), ('omol', None))
        self.assertEqual(mace_model({'model': 'omol-extra_large'}), ('omol', 'extra_large'))
        self.assertEqual(mace_model({'model': 'off-medium'}), ('off', 'medium'))
        self.assertEqual(mace_model({'model': 'small'}), ('off', 'small'))
        with tempfile.NamedTemporaryFile(suffix='.model') as checkpoint:
            self.assertEqual(mace_model({'model': checkpoint.name, 'family': 'omol'}), ('omol', checkpoint.name))
            with self.assertRaises(ValueError):
                mace_model({'model': checkpoint.name})
        with self.assertRaises(ValueError):
            mace_model({'model': 'off-huge'})

    def test_mace_calculators(self):
        """MACE-OFF refuses charged and open-shell species, and a cached calculator is reset and kept per device"""
        from arc.job.adapters.scripts.ase_script import get_calculator
        mace_off, mace_omol = MagicMock(name='mace_off'), MagicMock(name='mace_omol')
        mace_omol.side_effect = lambda **kwargs: MagicMock(name='MACE-OMol')
        with patch.dict(sys.modules, self.modules(**{'mace.calculators': {'mace_off': mace_off,
                                                                          'mace_omol': mace_omol}})):
            with self.assertRaises(ValueError):
                get_calculator({'calculator': 'mace', 'model': 'off-small'}, charge=0, multiplicity=2)
            mace_off.assert_not_called()
            get_calculator({'calculator': 'mace', 'model': 'off-small'})
            mace_off.assert_called_once_with(model='small', device='cpu', default_dtype='float64')

            cache = dict()
            first = get_calculator({'calculator': 'mace'}, charge=1, multiplicity=2, cache=cache)
            again = get_calculator({'calculator': 'mace'}, charge=0, multiplicity=1, cache=cache)
            self.assertIs(first, again)
            self.assertEqual(again.reset.call_count, 2)
            on_gpu = get_calculator({'calculator': 'mace', 'device': 'cuda'}, cache=cache)
            self.assertIsNot(on_gpu, first)
            self.assertEqual(mace_omol.call_args.kwargs, {'model': None, 'device': 'cuda', 'default_dtype': 'float64'})
        with patch.dict(sys.modules, self.modules(**{'mace.calculators': {'mace_off': mace_off}})):
            with self.assertRaisesRegex(ImportError, '0.3.14'):
                get_calculator({'calculator': 'mace'})

    def test_orb_calculators(self):
        """Only the Orb omol models take charge and spin, and orb-models before 0.6.0 is named in the error"""
        from arc.job.adapters.scripts.ase_script import get_calculator
        model, adapter = MagicMock(name='orbff'), MagicMock(name='adapter')
        pretrained = {'orb_v3_conservative_omol': MagicMock(return_value=(model, adapter)),
                      'orb_v3_conservative_inf_mpa': MagicMock(return_value=(model, adapter))}
        orb_calculator = MagicMock(name='ORBCalculator')
        with patch.dict(sys.modules, self.modules(**{'orb_models.forcefield.pretrained': pretrained,
                                                     'orb_models.forcefield.inference.calculator':
                                                         {'ORBCalculator': orb_calculator}})):
            sys.modules['orb_models.forcefield'].pretrained = sys.modules['orb_models.forcefield.pretrained']
            get_calculator({'calculator': 'orb'}, charge=-1, multiplicity=2)
            orb_calculator.assert_called_once_with(model, atoms_adapter=adapter, device='cpu')
            pretrained['orb_v3_conservative_omol'].assert_called_once_with(device='cpu', precision='float64')
            with self.assertRaises(ValueError):
                get_calculator({'calculator': 'orb', 'model': 'orb-v3-conservative-inf-mpa'}, charge=1)
        with patch.dict(sys.modules, self.modules(**{'orb_models.forcefield.pretrained': pretrained})):
            with self.assertRaisesRegex(ImportError, '0.6.0'):
                get_calculator({'calculator': 'orb'})

    def test_aimnet2_calculators(self):
        """AIMNet2 runs on the job's device, and only an NSE model takes a multiplicity"""
        from arc.job.adapters.scripts.ase_script import get_calculator
        base, ase_calculator = MagicMock(name='AIMNet2Calculator'), MagicMock(name='AIMNet2ASE')
        with patch.dict(sys.modules, self.modules(**{'aimnet.calculators': {'AIMNet2Calculator': base},
                                                     'aimnet.calculators.aimnet2ase': {'AIMNet2ASE': ase_calculator}})):
            with self.assertRaises(ValueError):
                get_calculator({'calculator': 'aimnet2', 'model': 'aimnet2'}, charge=0, multiplicity=2)
            get_calculator({'calculator': 'aimnet2', 'model': 'aimnet2'}, charge=1, multiplicity=1)
            base.assert_called_once_with('aimnet2', device='cpu')
            cache = dict()
            get_calculator({'calculator': 'aimnet2'}, charge=0, multiplicity=2, cache=cache)
            get_calculator({'calculator': 'aimnet2', 'device': 'cuda'}, cache=cache)
            get_calculator({'calculator': 'aimnet2'}, cache=cache)
            self.assertEqual([c.kwargs['device'] for c in base.call_args_list[1:]], ['cpu', 'cuda'])
            self.assertEqual(ase_calculator.call_args_list[1].kwargs, {'charge': 0, 'mult': 2})

    def test_fairchem_hessian_support(self):
        """Autograd Hessians need fairchem-core 2.17, and 2.18 to 2.21 give them only with vmap"""
        from importlib.metadata import PackageNotFoundError
        from arc.job.adapters.scripts.ase_script import fairchem_hessian_support
        for release, settings, supported in (('2.14.0', {}, False), ('2.16.0', {}, False), ('2.17.0', {}, True),
                                             ('2.19.1', {}, False), ('2.21.0', {'hessian_vmap': True}, True),
                                             ('2.22.0', {}, True), ('2.25.0', {}, True)):
            with self.subTest(release=release, settings=settings), \
                    patch('arc.job.adapters.scripts.ase_script.package_version', return_value=release):
                self.assertEqual(fairchem_hessian_support(settings) is None, supported)
        with patch('arc.job.adapters.scripts.ase_script.package_version', side_effect=PackageNotFoundError):
            self.assertIsNotNone(fairchem_hessian_support({}))

    def fake_fairchem(self, hessian: np.ndarray) -> tuple:
        """Stand-in fairchem modules whose calculator returns ``hessian``, and their get_predict_unit mock."""
        from dataclasses import dataclass, field

        @dataclass
        class InferenceSettings:
            tf32: bool = True
            activation_checkpointing: bool = True
            merge_mole: bool = True
            compile: bool = True
            base_precision_dtype: str = 'float32'
            hessian_vmap: bool = True
            predict_untrained_hessian: set = field(default_factory=set)

        class FAIRChemCalculator:
            def __init__(self, predictor, task_name):
                self.predictor, self.task_name = predictor, task_name

            def get_property(self, name, atoms):
                return hessian.ravel()

        get_predict_unit = MagicMock(side_effect=lambda model, **kwargs: MagicMock(name=model))
        modules = self.modules(**{'fairchem.core': {'pretrained_mlip': MagicMock(get_predict_unit=get_predict_unit),
                                                    'FAIRChemCalculator': FAIRChemCalculator},
                                  'fairchem.core.units.mlip_unit.api.inference':
                                      {'InferenceSettings': InferenceSettings}})
        return modules, get_predict_unit

    def test_uma_hessian(self):
        """The UMA Hessian predictor is loaded with the Hessian task, cached, and its Hessian reshaped to (3N, 3N)"""
        from arc.job.adapters.scripts.ase_script import get_calculator, uma_hessian
        water = Atoms('OH2', positions=[(0.0, 0.0, 0.0), (0.757, 0.586, 0.0), (-0.757, 0.586, 0.0)])
        modules, get_predict_unit = self.fake_fairchem(np.arange(81.0))
        settings, cache = {'calculator': 'uma', 'model': 'uma-s-1p2'}, dict()
        with patch.dict(sys.modules, modules), \
                patch('arc.job.adapters.scripts.ase_script.package_version', return_value='2.25.0'):
            hessian = uma_hessian(water, settings, cache)
            np.testing.assert_array_equal(hessian, np.arange(81.0).reshape(9, 9))
            uma_hessian(water, settings, cache)
            get_predict_unit.assert_called_once()
            inference_settings = get_predict_unit.call_args.kwargs['inference_settings']
            self.assertEqual(inference_settings.predict_untrained_hessian, {'omol'})
            self.assertFalse(inference_settings.hessian_vmap)
            self.assertFalse(inference_settings.tf32)
        with patch.dict(sys.modules, modules), \
                patch('arc.job.adapters.scripts.ase_script.package_version', return_value='2.14.0'):
            self.assertIsNone(uma_hessian(water, settings, dict()))
            get_predict_unit.reset_mock()
            calc = get_calculator(settings, cache=dict(), hessian=True)
            get_predict_unit.assert_called_once_with('uma-s-1p2', device='cpu')
            self.assertEqual(calc.task_name, 'omol')

    def test_a_uma_ts_search_gives_sella_the_hessian(self):
        """A TS optimization passes the UMA Hessian function to Sella, and a minimization does not"""
        from arc.job.adapters.scripts import ase_script
        calls = list()

        class FakeSella:
            def __init__(self, atoms, **kwargs):
                calls.append(kwargs)

            def run(self, fmax, steps):
                pass

        def hessian_function(atoms):
            return np.eye(3 * len(atoms))

        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(sys.modules, {'sella': self.modules(sella={'Sella': FakeSella})['sella']}), \
                patch('arc.job.adapters.scripts.ase_script.sella_hessian_function',
                      return_value=hessian_function) as sella_hessian:
            for is_ts in (True, False):
                input_path = os.path.join(tmp, 'input.yml')
                save_yaml_file(input_path, {'job_type': 'opt', 'xyz': ETHANE_XYZ, 'charge': 0, 'multiplicity': 1,
                                            'is_ts': is_ts, 'settings': {'calculator': 'emt', 'optimizer': 'sella'}})
                ase_script.run_job(input_path)
        self.assertIs(calls[0]['hessian_function'], hessian_function)
        self.assertEqual(calls[0]['order'], 1)
        self.assertNotIn('hessian_function', calls[1])
        sella_hessian.assert_called_once()

    def test_isotope_masses(self):
        """A job on a labelled species passes its isotope masses, which the script sets on the atoms"""
        from arc.job.adapters.scripts.ase_script import run_job
        cd4_xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'), 'isotopes': (12, 2, 2, 2, 2),
                   'coords': ((0.0, 0.0, 0.0), (0.629, 0.629, 0.629), (-0.629, -0.629, 0.629),
                              (-0.629, 0.629, -0.629), (0.629, -0.629, -0.629))}
        ch4_xyz = dict(cd4_xyz, isotopes=(12, 1, 1, 1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [ASEAdapter(execution_type='incore', job_type='freq', project='test_isotopes',
                               project_directory=os.path.join(tmp, label), species=[ARCSpecies(label=label, xyz=xyz)],
                               args={'keyword': {'calculator': 'emt'}}, testing=True)
                    for label, xyz in (('CD4', cd4_xyz), ('CH4', ch4_xyz))]
            self.assertIsNone(jobs[1].isotope_masses())
            masses = jobs[0].isotope_masses()
            self.assertAlmostEqual(masses[1], 2.014, places=3)
            with patch('arc.job.adapters.scripts.ase_script.run_vibrational_analysis',
                       return_value={'freqs': []}) as vibrations:
                for job in jobs:
                    os.makedirs(job.local_path, exist_ok=True)
                    job.write_input_file()
                    run_job(os.path.join(job.local_path, 'input.yml'))
            deuterated, plain = (call.args[0] for call in vibrations.call_args_list)
            np.testing.assert_allclose(deuterated.get_masses(), masses)
            self.assertAlmostEqual(plain.get_masses()[1], 1.008, places=3)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
