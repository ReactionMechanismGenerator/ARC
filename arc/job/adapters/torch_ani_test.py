#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.torchani module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH, almost_equal_lists, read_yaml_file
from arc.job.adapters.torch_ani import TorchANIAdapter
from arc.settings.settings import tani_default_options_dict
from arc.species import ARCSpecies
from arc.species.vectors import calculate_distance, calculate_angle, calculate_dihedral_angle
from arc.utils.wip import work_in_progress


class TestTorchANIAdapter(unittest.TestCase):
    """
    Contains unit tests for the TorchANIAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = TorchANIAdapter(execution_type='incore',
                                    job_type='sp',
                                    project='test_1',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_1'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO')],
                                    )
        cls.job_2 = TorchANIAdapter(execution_type='incore',
                                    job_type='opt',
                                    project='test_2',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_2'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO')],
                                    )
        etoh_xyz = """ C       -0.93790674    0.28066443    0.10572942
                       C        0.35659906   -0.44954997    0.05020174
                       O        0.36626530   -1.59397979   -0.38012632
                       H       -1.68923915   -0.33332195    0.61329151
                       H       -0.85532021    1.23909997    0.62578027
                       H       -1.30704889    0.46001151   -0.90948878
                       H        0.76281007   -0.50036590    1.06483009
                       H        1.04287051    0.12137561   -0.58236096
                       H        1.27820018   -1.93031032   -0.35203473"""
        cls.job_3 = TorchANIAdapter(execution_type='incore',
                                    job_type='opt',
                                    project='test_3',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_3'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                    constraints=[([1, 2, 3], 109), ([2, 3], 1.4), [(3, 2, 1, 5), 179.8]],
                                    )
        cls.job_4 = TorchANIAdapter(execution_type='incore',
                                    job_type='opt',
                                    project='test_4',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_4'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                    )
        cls.job_5 = TorchANIAdapter(execution_type='incore',
                                    job_type='sp',
                                    project='test_5',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_5'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                    )
        cls.job_6 = TorchANIAdapter(execution_type='incore',
                                    job_type='freq',
                                    project='test_6',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_6'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                    )
        cls.job_7 = TorchANIAdapter(execution_type='incore',
                                    job_type='optfreq',
                                    project='test_7',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_7'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                    )
        cls.job_8 = TorchANIAdapter(execution_type='incore',
                                    job_type='sp',
                                    project='test_8',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_8'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO')],
                                    )
        cls.job_9 = TorchANIAdapter(execution_type='incore',
                                    job_type='force',
                                    project='test_9',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_9'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO')],
                                    )
        cls.job_10 = TorchANIAdapter(execution_type='incore',
                                     job_type='freq',
                                     project='test_10',
                                     project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_10'),
                                     species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                     )
        ts_xyz = """O       0.42776400   -1.13025800    0.74130700
                    C      -0.88445900   -1.00301900    0.18936400
                    C      -1.16676700    0.43713000   -0.12950100
                    O      -0.22398600    0.99395700   -0.98854800
                    C       1.10086500    0.85201400   -0.46107200
                    C       1.39139600   -0.60462800   -0.16166000
                    H      -0.95925400   -1.60819800   -0.72246500
                    H      -1.57919200   -1.38708700    0.93411500
                    H      -1.38327700    1.07075400    0.73574200
                    H      -2.32127000    0.44750900   -0.84259100
                    H       1.19559900    1.44803100    0.45354000
                    H       1.77762800    1.24491200   -1.21872300
                    H       1.38296600   -1.18269200   -1.09426000
                    H       2.36446300   -0.71448700    0.31642200
                    H      -3.07277200    0.44675200   -1.38374200"""
        cls.job_ts = TorchANIAdapter(execution_type='incore',
                                     job_type='freq',
                                     project='test_5',
                                     project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_ts'),
                                     species=[ARCSpecies(label='ts', xyz=ts_xyz, is_ts=True)],)

    def test_run_sp(self):
        """Test the run_sp() method"""
        self.assertIsNone(self.job_8.sp)
        self.job_8.execute()
        self.assertAlmostEqual(self.job_8.sp, -406925.6663602, places=3)

    def test_run_force(self):
        """Test the run_force() method."""
        self.job_9.execute()
        self.assertTrue(almost_equal_lists(self.job_9.force,
                                           [[0.0016908727120608091, 0.009869818575680256, 0.0010390167590230703],
                                            [0.008561883121728897, 0.013575542718172073, 0.0018425204325467348],
                                            [0.010151226073503494, 0.0019111409783363342, 0.0008542370051145554],
                                            [0.0007229172624647617, -0.0034123272635042667, 0.0017430195584893227],
                                            [0.0009874517563730478, -0.0030386836733669043, -0.002235014922916889],
                                            [-0.0011509160976856947, -0.00131673039868474, -0.00019995146431028843],
                                            [-0.003339727409183979, -0.012097489088773727, 0.0027393437922000885],
                                            [-0.0028020991012454033, -0.011338669806718826, -0.005346616730093956],
                                            [-0.014821597374975681, 0.005847393535077572, -0.0004365567583590746]],
                                           rtol=1e-3, atol=1e-5))

    def test_run_opt(self):
        """Test the run_opt() method."""
        self.assertIsNone(self.job_2.opt_xyz)
        self.job_2.execute()
        self.assertEqual(self.job_2.opt_xyz['symbols'], ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'))

        self.assertIsNone(self.job_3.opt_xyz)
        self.job_3.execute()
        self.assertAlmostEqual(calculate_distance(coords=self.job_3.opt_xyz['coords'], atoms=[2, 3], index=1), 1.4, places=2)
        self.assertAlmostEqual(calculate_angle(coords=self.job_3.opt_xyz['coords'], atoms=[1, 2, 3], index=1), 109, places=2)
        self.assertAlmostEqual(calculate_dihedral_angle(coords=self.job_3.opt_xyz['coords'], torsion=[3, 2, 1, 5], index=1),
                               179.8, places=2)

        self.assertIsNone(self.job_4.opt_xyz)
        self.job_4.execute()
        self.assertEqual(self.job_4.opt_xyz['symbols'], ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'))

    def test_run_vibrational_analysis(self):
        """Test the run_vibrational_analysis() method."""
        self.job_10.execute()
        results = read_yaml_file(path=self.job_10.local_path_to_output_file)
        self.assertEqual(list(results.keys()), ['force_constants', 'freqs', 'hessian', 'modes', 'reduced_masses'])
        self.assertEqual(len(results['force_constants']), 3*self.job_10.species[0].mol.get_num_atoms())
        self.assertEqual(len(results['freqs']), 3*self.job_10.species[0].mol.get_num_atoms())
        self.assertEqual(len(results['hessian'][0][0]), 3*self.job_10.species[0].mol.get_num_atoms())
        self.assertEqual(len(results['hessian'][0]), 3*self.job_10.species[0].mol.get_num_atoms())
        self.assertEqual(len(results['modes'][0][0]), 3)
        self.assertEqual(len(results['modes'][0]), 3**2)
        self.assertEqual(len(results['reduced_masses']), 3*self.job_10.species[0].mol.get_num_atoms())

    def test_run_freq(self):
        """Test the run_freq() method."""
        self.assertIsNone(self.job_6.freqs)
        self.assertIsNone(self.job_7.freqs)
        self.job_6.execute()
        self.job_7.execute()
        self.assertAlmostEqual(self.job_6.freqs[-1], 3756.57643, places=3)
        self.assertGreater(self.job_7.freqs[-1], 3800)
        self.assertEqual(len(self.job_7.freqs), self.job_7.species[0].mol.get_num_atoms()*3-5)
        self.assertEqual(len(self.job_6.freqs), self.job_6.species[0].mol.get_num_atoms()*3-5)

    @work_in_progress
    def test_ts_freqs(self):
        self.assertIsNone(self.job_ts.freqs)
        self.job_ts.execute()
        self.assertLess(min(self.job_ts.freqs), 0)
        self.assertEqual(sum([freq > 0 for freq in self.job_ts.freqs]), 3*self.job_ts.species[0].mol.get_num_atoms()-6-1) #3N-6 freqs, minus one imaginary freq.

    def test_check_settings(self):
        """Test the test_check_settings method"""
        dummy = self.job_1
        self.assertFalse(dummy.check_settings(settings={"model":"Kinetic Filter and Imputation Regression"}))
        self.assertFalse(dummy.check_settings(settings={"device":"gpu"}))
        self.assertFalse(dummy.check_settings(settings={"engine":"Knowledge-based Approach for Linear And Nonlinear optimization"}))
        self.assertFalse(dummy.check_settings(settings={"fmax":"0.001"}))
        self.assertFalse(dummy.check_settings(settings={"steps":0.001}))
        self.assertTrue(dummy.check_settings(settings={"model":"ANI1x"}))
        self.assertTrue(dummy.check_settings(settings={"device":"CUDA"}))
        self.assertTrue(dummy.check_settings(settings={"engine":"SciPyFminBFGS"}))
        self.assertTrue(dummy.check_settings(settings={"fmax":0.001}))
        self.assertTrue(dummy.check_settings(settings={"steps":2000}))

    def test_write_input_file(self):
        """Test the write_input_file function"""
        self.job_8.write_input_file(settings = tani_default_options_dict)
        self.assertTrue(os.path.isfile(os.path.join(self.job_8.local_path, "input.yml")))
        content = read_yaml_file(os.path.join(self.job_8.local_path, "input.yml"))
        expected = {'xyz': {'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                    'isotopes': (12, 12, 16, 1, 1, 1, 1, 1, 1),
                    'coords': ((-1.0219247317890747, 0.04463710864128214, -0.09392250756371581),
                               (0.45215334957943165, -0.29310459026320146, -0.02762318136049812),
                               (1.1946118815163072, 0.9061450016152853, 0.13427251272933854),
                               (-1.3413296099276413, 0.5590849830805962, 0.8184500099080437),
                               (-1.2251569876563784, 0.7230760434250652, -0.9290968752227021),
                               (-1.624455565486164, -0.8593711938803317, -0.21881071555745987), 
                               (0.6572919629992373, -0.9519409562010626, 0.8214428556138298),
                               (0.7750285429958658, -0.7857391782795276, -0.9496446043469213),
                               (2.133781157768429, 0.657212781861891, 0.1733450946053844))},
                    'job_type': 'sp',
                    'constraints': [],
                    'model': 'ani2x',
                    'device': 'cpu',
                    'fmax': 0.001,
                    'engine': 'bfgs',
                    'steps': None}
        self.assertEqual(content, expected)
        self.job_10.write_input_file(settings = tani_default_options_dict)
        self.assertTrue(os.path.isfile(os.path.join(self.job_10.local_path, "input.yml")))
        content = read_yaml_file(os.path.join(self.job_10.local_path, "input.yml"))
        expected = {'xyz': {'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                            'isotopes': (12, 12, 16, 1, 1, 1, 1, 1, 1),
                            'coords': ((-0.93790674, 0.28066443, 0.10572942),
                                       (0.35659906, -0.44954997, 0.05020174),
                                       (0.3662653, -1.59397979, -0.38012632),
                                       (-1.68923915, -0.33332195, 0.61329151),
                                       (-0.85532021, 1.23909997, 0.62578027), 
                                       (-1.30704889, 0.46001151, -0.90948878),
                                       (0.76281007, -0.5003659, 1.06483009),
                                       (1.04287051, 0.12137561, -0.58236096),
                                       (1.27820018, -1.93031032, -0.35203473))},
                    'job_type': 'freq',
                    'constraints': [],
                    'model': 'ani2x',
                    'device': 'cpu',
                    'fmax': 0.001,
                    'engine': 'bfgs',
                    'steps': None}
        self.assertEqual(content, expected)


    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        for i in list(range(1, 11))+["ts"]:
            path = os.path.join(ARC_PATH, 'arc', 'testing', f'test_TorchANIAdapter_{i}')
            shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
