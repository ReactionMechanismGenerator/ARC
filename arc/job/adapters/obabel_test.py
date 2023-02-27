#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.torchani module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH, read_yaml_file
from arc.job.adapters.obabel import OpenbabelAdapter
from arc.level import Level
from arc.settings.settings import ob_default_settings
from arc.species import ARCSpecies
from arc.species.vectors import calculate_distance, calculate_angle, calculate_dihedral_angle


class TestOpenbabelAdapter(unittest.TestCase):
    """
    Contains unit tests for the OpenbabelAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = OpenbabelAdapter(execution_type='incore',
                                    job_type='sp',
                                    project='test_1',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_OpenbabelAdapter_1'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO')],
                                    level=Level(method="MMFF94")
                                    )
        cls.job_2 = OpenbabelAdapter(execution_type='incore',
                                    job_type='opt',
                                    project='test_2',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_OpenbabelAdapter_2'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO')],
                                    level=Level(method="MMFF94s")
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
        cls.job_3 = OpenbabelAdapter(execution_type='incore',
                                     job_type='opt',
                                     project='test_3',
                                     project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_OpenbabelAdapter_3'),
                                     species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                     constraints=[([1, 2, 3], 109), ([2, 3], 1.4), [(3, 2, 1, 5), 179.8]],
                                     level=Level(method="ghemical")
                                     )
        cls.job_4 = OpenbabelAdapter(execution_type='incore',
                                     job_type='opt',
                                     project='test_4',
                                     project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_OpenbabelAdapter_4'),
                                     species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                     level=Level(method="gaff")
                                     )
        cls.job_5 = OpenbabelAdapter(execution_type='incore',
                                     job_type='sp',
                                     project='test_9',
                                     project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_OpenbabelAdapter_5'),
                                     species=[ARCSpecies(label='EtOH', smiles='CCO')],
                                     level=Level(method="MMFF94s")
                                     )

    def test_run_sp(self):
        """Test the run_sp() method"""
        self.assertIsNone(self.job_5.sp)
        self.job_5.execute()
        self.assertAlmostEqual(self.job_5.sp, -6.3475, places=3)

    def test_run_opt(self):
        """Test the run_opt() method."""
        self.assertIsNone(self.job_2.opt_xyz)
        self.job_2.execute()
        self.assertEqual(self.job_2.opt_xyz['symbols'], ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'))

        self.assertIsNone(self.job_3.opt_xyz)
        self.job_3.execute()

        self.assertAlmostEqual(calculate_distance(coords=self.job_3.opt_xyz['coords'], atoms=[2, 3], index=1), 1.43, places=2)
        self.assertAlmostEqual(calculate_angle(coords=self.job_3.opt_xyz['coords'], atoms=[1, 2, 3], index=1), 109.37, places=2)
        self.assertAlmostEqual(calculate_dihedral_angle(coords=self.job_3.opt_xyz['coords'], torsion=[3, 2, 1, 5], index=1),
                               179.9, places=2)

        self.assertIsNone(self.job_4.opt_xyz)
        self.job_4.execute()
        self.assertEqual(self.job_4.opt_xyz['symbols'], ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'))

    def test_check_settings(self):
        """Test the test_check_settings method"""
        dummy = self.job_1
        self.assertFalse(dummy.check_settings(settings={"FF":"Kinetic Filter and Imputation Regression"}))
        self.assertFalse(dummy.check_settings(settings={"opt_gradient_settings" : {"steps" : "2000"}}))
        self.assertFalse(dummy.check_settings(settings={"opt_gradient_settings" : {"econv" : "1e-6"}}))
        self.assertTrue(dummy.check_settings(settings={"FF":"MMFF94s"}))
        self.assertTrue(dummy.check_settings(settings={"opt_gradient_settings" : {"steps" : 2000}}))
        self.assertTrue(dummy.check_settings(settings={"opt_gradient_settings" : {"econv" : 1e-6}}))

    def test_write_input_file(self):
        """Test the write_input_file function"""
        self.job_3.write_input_file(settings = ob_default_settings)
        self.assertTrue(os.path.isfile(os.path.join(self.job_3.local_path, "input.yml")))
        content = read_yaml_file(os.path.join(self.job_3.local_path, "input.yml"))
        expected = {'FF': 'ghemical',
 'constraints': [([1, 2, 3], 109), ([2, 3], 1.4), [(3, 2, 1, 5), 179.8]],
 'job_type': 'opt',
 'opt_gradient_settings': {'econv': 1e-06, 'steps': 2000},
 'xyz': """9

C      -0.90396000    0.24943000    0.09740000
C       0.37548000   -0.33646000    0.11128000
O       0.30460000   -1.64639000   -0.46537000
H      -1.60352000   -0.37089000    0.67821000
H      -0.86207000    1.25648000    0.54019000
H      -1.26689000    0.32613000   -0.93891000
H       0.73807000   -0.41081000    1.14881000
H       1.07412000    0.28450000   -0.47164000
H       1.16141000   -2.05836000   -0.46415000
"""}
        for key in expected:
            self.assertEqual(expected[key], content[key])

        self.job_4.write_input_file(settings = ob_default_settings)
        self.assertTrue(os.path.isfile(os.path.join(self.job_4.local_path, "input.yml")))
        content = read_yaml_file(os.path.join(self.job_4.local_path, "input.yml"))
        expected = {'FF': 'gaff',
 'job_type': 'opt',
 'opt_gradient_settings': {'econv': 1e-06, 'steps': 2000},
 'xyz': """9

C      -0.96457000    0.28365000    0.09973000
C       0.42621000   -0.37164000    0.10627000
O       0.34081000   -1.67524000   -0.47009000
H      -1.67310000   -0.31337000    0.67867000
H      -0.91415000    1.28173000    0.54022000
H      -1.34066000    0.37616000   -0.92183000
H       0.79106000   -0.44925000    1.13421000
H       1.12336000    0.23983000   -0.47329000
H       1.22826000   -2.07823000   -0.45808000
"""}
        for key in expected:
            self.assertEqual(expected[key], content[key])

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        for i in list(range(1, 6)):
            path = os.path.join(ARC_PATH, 'arc', 'testing', f'test_OpenbabelAdapter_{i}')
            shutil.rmtree(path, ignore_errors=True)

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
