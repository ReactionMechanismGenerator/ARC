#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.gcn module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH
import arc.job.adapters.ts.gcn_ts as ts_gcn
from arc.reaction import ARCReaction
from arc.rmgdb import load_families_only, make_rmg_database_object
from arc.species.converter import str_to_xyz
from arc.species.species import ARCSpecies, TSGuess


class TestGCNAdapter(unittest.TestCase):
    """
    Contains unit tests for the GCNAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = make_rmg_database_object()
        load_families_only(cls.rmgdb)
        cls.output_dir = os.path.join(ARC_PATH, 'arc', 'testing', 'GCN')
        if not os.path.isdir(cls.output_dir):
            os.makedirs(cls.output_dir)
        cls.rxn_1 = ARCReaction(r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
                                p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')])
        cls.reactant_path = os.path.join(cls.output_dir, 'react.sdf')
        cls.product_path = os.path.join(cls.output_dir, 'prod.sdf')
        cls.ts_path = os.path.join(cls.output_dir, 'ts.xyz')

    def test_write_sdf_files(self):
        """Test the write_sdf_files() function."""
        self.assertFalse(os.path.isfile(self.reactant_path))
        self.assertFalse(os.path.isfile(self.product_path))
        ts_gcn.write_sdf_files(rxn=self.rxn_1,
                               reactant_path=self.reactant_path,
                               product_path=self.product_path,
                               )
        with open(self.reactant_path, 'r') as f:
            content_r_sdf = f.read()
        with open(self.product_path, 'r') as f:
            content_p_sdf = f.read()
        expected_r_sdf = """
     RDKit          3D

 10  9  0  0  0  0  0  0  0  0999 V2000
    1.3390    0.2885    0.5167 C   0  0  0  0  0  3  0  0  0  0  0  0
    0.1866   -0.4096   -0.1211 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1463    0.0874    0.4131 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.3016   -0.2071    0.5757 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2891    1.3478    0.7427 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.2306   -0.2567   -1.2045 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.2728   -1.4857    0.0639 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2230   -0.0695    1.4944 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9703   -0.4510   -0.0661 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2801    1.1560    0.2137 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  4  1  0
  1  5  1  0
  2  3  1  0
  2  6  1  0
  2  7  1  0
  3  8  1  0
  3  9  1  0
  3 10  1  0
M  RAD  1   1   2
M  END
$$$$
"""
        expected_p_sdf = """
     RDKit          3D

 10  9  0  0  0  0  0  0  0  0999 V2000
   -1.2887    0.0629    0.1089 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0110   -0.4576   -0.3934 C   0  0  0  0  0  3  0  0  0  0  0  0
    1.2841    0.1132    0.1221 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4984    1.0458   -0.3224 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2825    0.1465    1.1995 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0984   -0.6166   -0.1732 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0274   -1.0601   -1.2952 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4596    1.1037   -0.3073 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1226   -0.5341   -0.1516 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2634    0.1963    1.2126 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  4  1  0
  1  5  1  0
  1  6  1  0
  2  3  1  0
  2  7  1  0
  3  8  1  0
  3  9  1  0
  3 10  1  0
M  RAD  1   2   2
M  END
$$$$
"""
        self.assertEqual(content_r_sdf, expected_r_sdf)
        self.assertEqual(content_p_sdf, expected_p_sdf)

    def test_run_subprocess_locally(self):
        """Test the run_subprocess_locally() function"""
        self.assertFalse(os.path.isfile(self.ts_path))
        self.rxn_1.ts_species = ARCSpecies(label='TS', is_ts=True)
        ts_gcn.run_subprocess_locally(direction='F',
                                      reactant_path=self.reactant_path,
                                      product_path=self.product_path,
                                      ts_path=self.ts_path,
                                      local_path=self.output_dir,
                                      ts_species=self.rxn_1.ts_species,
                                      )
        if os.path.isfile(self.ts_path):
            # The gcn_env automated creation isn't robust, only test this if an output file was generated.
            xyz = str_to_xyz(self.ts_path)
            self.assertEqual(xyz['symbols'], ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'))

    def test_process_tsg(self):
        """Test the process_tsg() function."""
        expected_ts_xyz_path = os.path.join(self.output_dir, 'GCN R 0.xyz')
        if os.path.isfile(expected_ts_xyz_path):
            os.remove(expected_ts_xyz_path)
        self.rxn_1.ts_species = ARCSpecies(label='TS', is_ts=True)
        ts_gcn.process_tsg(direction='R',
                           ts_xyz=str_to_xyz(os.path.join(ARC_PATH, 'arc', 'testing', 'opt', 'TS_nC3H7-iC3H7.out')),
                           local_path=self.output_dir,
                           ts_species=self.rxn_1.ts_species,
                           tsg=TSGuess(method=f'GCN',
                                       method_direction='R',
                                       index=0,
                                       ),
                           )
        self.assertTrue(os.path.isfile(expected_ts_xyz_path))
        xyz = str_to_xyz(expected_ts_xyz_path)
        self.assertEqual(xyz['symbols'], ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'))

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'GCN'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
