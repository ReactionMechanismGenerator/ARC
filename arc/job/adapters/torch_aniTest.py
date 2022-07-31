#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.torchani module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH, almost_equal_coords, almost_equal_lists, read_yaml_file
from arc.job.adapters.torch_ani import TorchANIAdapter
from arc.species import ARCSpecies
from arc.species.vectors import calculate_distance, calculate_angle, calculate_dihedral_angle


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

    def test_run_sp(self):
        """Test the run_sp() method"""
        self.assertIsNone(self.job_8.sp)
        self.job_8.execute()
        self.assertAlmostEqual(self.job_8.sp, -406925.6663602, places=5)

    def test_run_force(self):
        """Test the run_force() method."""
        self.job_1.execute()
        self.job_1.run_force()
        self.assertTrue(almost_equal_lists(self.job_1.force,
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
        self.job_5.execute()
        self.job_5.run_vibrational_analysis()
        results = read_yaml_file(path=self.job_5.local_path_to_output_file)
        self.assertTrue(almost_equal_lists(results['hessian'][0][0],
                                           [2.05839689e+00, 4.23168552e-02, 8.99141272e-03, -8.52832328e-01,
                                            2.94829475e-01, 1.96506945e-02, 2.89284564e-02, -2.89667006e-04,
                                            3.82898604e-03, -6.23718748e-01, -3.78768865e-01, 3.05305614e-01,
                                            -1.90607616e-01, -7.05216951e-02, -3.90087890e-02, -2.80615896e-01,
                                            4.89203736e-02, -2.94846146e-01, -5.09288349e-02, 2.91931411e-02,
                                            -5.12687819e-03, -9.03079841e-02, 3.84447916e-02, 3.17928302e-03,
                                            1.68605651e-03, -4.12440913e-03, -1.97417713e-03],
                                           rtol=1e-3, atol=1e-5))
        self.assertTrue(almost_equal_lists(results['freqs'],
                                           [1.5990907380252005e-05, 2.3879323261905037e-05, 216.62654411778206,
                                            350.3016452990423, 570.0221108885962, 811.4857084668979, 988.0011567410421,
                                            1123.1719679205025, 1234.037604138696, 1434.8963178724466,
                                            1457.1182547591543, 1515.7704291092418, 1517.2463447563814,
                                            1526.2259007106625, 1540.9820899730173, 1623.8830480125573,
                                            1993.6470673836843, 3088.059595900005, 3103.895165061055,
                                            3119.2004301707084, 3168.8963228731536, 3212.0401945048247,
                                            3756.5764018240966],
                                           rtol=1e-3, atol=1e-5))
        self.assertTrue(almost_equal_lists(results['modes'][0],
                                           [[-0.00111122, -0.00423765, 0.00860585],
                                            [-0.02460106, -0.05224492, 0.13896354],
                                            [0.02388287, 0.05283278, -0.13888713],
                                            [-0.07373273, 0.02218154, -0.06496827],
                                            [-0.01312883, -0.01403369, 0.02904308],
                                            [0.1046359, 0.01367739, -0.02627537],
                                            [-0.21524512, -0.27805918, 0.20518337],
                                            [0.11017874, 0.05819121, 0.38598261],
                                            [0.01460158, 0.03250765, -0.08293438]],
                                           rtol=1e-3, atol=1e-5))
        self.assertTrue(almost_equal_lists(results['force_constants'],
                                           [-4.72317069e-01, -3.18322408e-01, -3.41694897e-02, -2.58567636e-15,
                                            1.01478134e-16, 2.40677209e-15, 2.83592995e-02, 1.61204129e-01,
                                            2.03562141e-01, 4.27723914e-01, 1.03743110e+00, 1.28358247e+00,
                                            1.45646684e+00, 1.35728055e+00, 1.49816033e+00, 1.47682490e+00,
                                            1.50485364e+00, 1.42740505e+00, 1.55674731e+00, 1.98736338e+00,
                                            2.60567507e+01, 5.85144037e+00, 5.97735126e+00, 6.28612634e+00,
                                            6.52742297e+00, 6.69095626e+00, 8.86359249e+00],
                                           rtol=1e-3, atol=1e-5))
        self.assertTrue(almost_equal_lists(results['reduced_masses'],
                                           [2.45710903, 3.78910432, 2.41801511, 5.11877778, 5.11877778, 5.11877778,
                                            1.02570369, 2.22967504, 1.06331849, 1.10243132, 1.80382383, 1.72695503,
                                            1.62328052, 1.1188667, 1.1976184, 1.09096774, 1.10951155, 1.0400623,
                                            1.11268621, 1.27913844, 11.12688151, 1.04145666, 1.05303899, 1.0965951,
                                            1.10325386, 1.10071785, 1.0660439],
                                           rtol=1e-3, atol=1e-5))

    def test_run_freq(self):
        """Test the run_freq() method."""
        self.assertIsNone(self.job_6.freqs)
        self.assertIsNone(self.job_7.freqs)
        self.job_6.execute()
        self.job_7.execute()
        self.assertAlmostEqual(self.job_6.freqs[-1], 3756.57643, places=3)
        self.assertGreater(self.job_7.freqs[-1], 3900)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        for i in range(10):
            path = os.path.join(ARC_PATH, 'arc', 'testing', f'test_TorchANIAdapter_{i}')
            shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
