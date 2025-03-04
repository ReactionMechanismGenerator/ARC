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
        expected_r_atoms = sorted(list(self.rxn_1.r_species[0].get_xyz()["symbols"]), key=ord)
        expected_p_atoms = sorted(list(self.rxn_1.p_species[0].get_xyz()["symbols"]), key=ord)
        r_xyz_atoms = sorted(list(filter(("").__ne__, content_r_sdf.split(" "))), key =len, reverse= False)
        p_xyz_atoms = sorted(list(filter(("").__ne__, content_p_sdf.split(" "))), key =len, reverse= False)
        i = 0
        r_atoms = []
        while len(r_xyz_atoms[i]) == 1:
            if r_xyz_atoms[i] in ["C", "H", "O", "N"]:
                r_atoms.append(r_xyz_atoms[i])
            i += 1
        i = 0
        p_atoms = []
        while len(p_xyz_atoms[i]) == 1:
            if p_xyz_atoms[i] in ["C", "H", "O", "N"]:
                p_atoms.append(p_xyz_atoms[i])
            i += 1
        self.assertEqual(r_atoms, expected_r_atoms)
        self.assertEqual(p_atoms, expected_p_atoms)

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
