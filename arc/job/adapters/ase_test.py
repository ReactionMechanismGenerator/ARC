#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.adapters.ase module.
These tests verify IO and logic without executing the external ASE script in CI.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations

from arc.common import ARC_TESTING_PATH, read_yaml_file, save_yaml_file
from arc.job.adapters.ase_adapter import ASEAdapter, active_worker_tickets, servers, submit_filenames
from arc.parser.parser import parse_1d_scan_coords, parse_1d_scan_energies
from arc.species.species import ARCSpecies
from arc.job.adapters.scripts.ase_script import (apply_constraints,
                                                 claim_ticket,
                                                 get_ticket_status,
                                                 is_linear,
                                                 live_workers,
                                                 merge_scan_branches,
                                                 numpy_vibrational_analysis,
                                                 relaxed_torsion_scan,
                                                 rotor_top,
                                                 run_torsion_scan,
                                                 run_vibrational_analysis,
                                                 scan_convergence_warning,
                                                 sella_hessian_function,
                                                 serve,
                                                 spool_dirs,
                                                 submit_ticket,
                                                 to_kJmol,
                                                 uma_inference_settings,
                                                 withdraw_ticket)

# Converts sqrt(eV / (A^2 * amu)) to cm^-1, as numpy_vibrational_analysis() does.
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
        cls.project_directory = os.path.join(ARC_TESTING_PATH, 'test_ASEAdapter')
        os.makedirs(cls.project_directory, exist_ok=True)  # parallel workers race here

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

        # Curvature along the in-plane rotation, as a 2-point finite-difference Hessian gives: about 60i cm^-1.
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
        self.assertEqual(coords[1], ('f', 1))  # forward won here (1.0 < 3.0)
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

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        shutil.rmtree(cls.project_directory, ignore_errors=True)


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

    def write_job(self, name: str, job_type: str = 'sp') -> str:
        """Write an EMT job's input.yml and return its path."""
        job_dir = os.path.join(self.tmp.name, name)
        os.makedirs(job_dir)
        input_path = os.path.join(job_dir, 'input.yml')
        save_yaml_file(input_path, {'job_type': job_type, 'xyz': ETHANE_XYZ, 'charge': 0, 'multiplicity': 1,
                                    'settings': {'calculator': 'emt'}})
        return input_path

    def beat(self, worker_id: str, age: float = 0.0) -> None:
        """Write a worker heartbeat that is ``age`` seconds old."""
        path = os.path.join(self.spool, 'workers', f'{worker_id}.json')
        with open(path, 'w') as f:
            f.write('{}')
        then = time.time() - age
        os.utime(path, (then, then))

    def make_job(self, execution_type: str = 'incore') -> ASEAdapter:
        """Make an EMT single-point job with a worker spool configured for EMT."""
        with patch.dict('arc.job.adapters.ase_adapter.settings', {'ASE_WORKERS': {'emt': self.spool}}):
            job = ASEAdapter(execution_type=execution_type,
                             job_type='sp',
                             project='test_worker',
                             project_directory=os.path.join(self.tmp.name, 'project'),
                             species=[ARCSpecies(label='ethane', xyz=ETHANE_XYZ)],
                             args={'keyword': {'calculator': 'emt'}},
                             testing=True)
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
        self.assertEqual(os.listdir(os.path.join(stale_dir, 'vib_tmp')), list())  # cleared after the run
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
        self.assertEqual(live_workers(self.spool), list())  # the heartbeat is removed on exit

    def test_no_live_worker_leaves_the_execution_type_alone(self):
        """Without a live worker, a job runs as the scheduler asked"""
        job = self.make_job(execution_type='queue')
        self.assertIsNone(job.worker_spool)
        self.assertEqual(job.execution_type, 'queue')

    def test_a_live_worker_takes_jobs_without_blocking(self):
        """With a live worker, a job is submitted as a local queue job and stays in flight until its ticket ends"""
        self.beat('w1')
        jobs = [self.make_job(execution_type='incore') for _ in range(3)]
        for job in jobs:
            self.assertEqual(job.worker_spool, self.spool)
            self.assertEqual(job.execution_type, 'queue')
            self.assertEqual(job.server, 'local')
            job.execute_queue()  # returns at once: nothing takes the tickets in this test
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
        claim_ticket(self.spool, 'w2')  # w2 has no heartbeat
        with patch.object(ASEAdapter, 'run_directly') as run_directly, self.assertLogs('arc', level='WARNING'):
            self.assertEqual(job._check_job_server_status(), 'done')
        run_directly.assert_called_once()
        self.assertEqual(get_ticket_status(self.spool, job.job_id)[0], 'errored')

    def test_a_queue_job_id_is_not_a_ticket(self):
        """A job ID from a cluster queue is checked with the queue, not the spool"""
        self.beat('w1')
        job = self.make_job()
        job.job_id = '123456'
        self.assertFalse(job.is_worker_ticket())
        job.job_id = f'{job.job_name}_0123456789ab'
        self.assertTrue(job.is_worker_ticket())

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
