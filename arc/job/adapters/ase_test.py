#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.adapters.ase module.
These tests verify IO and logic without executing the external ASE script in CI.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT

from arc.common import read_yaml_file, save_yaml_file
from arc.job.adapters.ase_adapter import ASEAdapter
from arc.parser.parser import parse_1d_scan_coords, parse_1d_scan_energies
from arc.species.species import ARCSpecies
from arc.job.adapters.scripts.ase_script import (is_linear,
                                                 merge_scan_branches,
                                                 numpy_vibrational_analysis,
                                                 relaxed_torsion_scan,
                                                 rotor_top,
                                                 run_torsion_scan,
                                                 scan_convergence_warning,
                                                 to_kJmol)

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


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
