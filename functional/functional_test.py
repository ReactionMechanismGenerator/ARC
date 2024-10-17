#!/usr/bin/env python3
# encoding: utf-8

"""
This module that contains functional tests for ARC
"""

import os
import shutil
import unittest

from arc import ARC
from arc.common import ARC_PATH
from arc.exceptions import InputError
from arc.imports import settings
from arc.reaction import ARCReaction
from arc.species import ARCSpecies

settings['ts_adapters'] = ['xtb-gsm']


class TestFunctional(unittest.TestCase):
    """
    Contains functional tests for ARC.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.has_settings = False
        
        cls.job_types = {'conf_opt': True,
                         'opt': True,
                         'fine_grid': False,
                         'freq': True,
                         'sp': True,
                         'conf_sp': False,
                         'rotors': False,
                         'irc': False,
                         }

        cls.species_list_1 = [ARCSpecies(label="2-propanol", smiles="CC(O)C"),
                              ARCSpecies(label="1-propanol", smiles="CCCO"),
                              ARCSpecies(label="NN", smiles="NN")]
        cls.arc_object_1 = ARC(project='FunctionalThermoTest',
                               project_directory=os.path.join(ARC_PATH, "functional", "test", "thermo"),
                               species=cls.species_list_1,
                               job_types=cls.job_types,
                               conformer_level='gfn2',
                               level_of_theory='gfn2',
                               freq_scale_factor=1.0,
                               n_confs=2,
                               bac_type=None,
                               verbose=1,
                               )

        with open(os.path.join(ARC_PATH, "functional", "ts_guess.xyz"), 'r') as f:
            cls.ts_guess = f.read()
        cls.species_list_2 = [ARCSpecies(label="iC3H7", smiles="C[CH]C"), ARCSpecies(label="nC3H7", smiles="CC[CH2]"),
                              ARCSpecies(label="TS0", xyz=cls.ts_guess, is_ts=True)]

        cls.arc_object_2 = ARC(project='FunctionalKineticTest',
                               project_directory=os.path.join(ARC_PATH, "functional", "test", "kinetic"),
                               reactions=[ARCReaction(label="iC3H7 <=> nC3H7", ts_label="TS0")],
                               species=cls.species_list_2,
                               job_types=cls.job_types,
                               conformer_level='gfn2',
                               level_of_theory='gfn2',
                               ts_guess_level='gfn2',
                               freq_scale_factor=1.0,
                               n_confs=2,
                               dont_gen_confs=["TS0"],
                               bac_type=None,
                               verbose=1,
                               compare_to_rmg=False,
                               )
    
    def testThermo(self):
        """Test thermo"""
        self.arc_object_1.execute()
        summary = self.arc_object_1.summary()
        for _, ter in summary.items():
            self.assertTrue(ter)
        self.assertTrue(os.path.exists(os.path.join(ARC_PATH, "functional", "test", "thermo", "output", "RMG libraries", "thermo")))
        with open(file=os.path.join(ARC_PATH, "functional", "test", "thermo", "output", "RMG libraries", "thermo", "FunctionalThermoTest.py"), mode='r') as f:
            entry = 0
            for line in f.readlines():
                if "entry" in line:
                    entry += 1
        self.assertEqual(entry, len(self.species_list_1))

    def testKinetic(self):
        """Test kinetics"""
        try:
            self.arc_object_2.execute()
        except InputError as e:
            self.skipTest(f"execution of ARC failed: {e}, skipping this test for now...")
        summary = self.arc_object_2.summary()
        for _, ter in summary.items():
            self.assertTrue(ter)
        self.assertTrue(os.path.exists(os.path.join(ARC_PATH, "functional", "test", "kinetic", "output", "RMG libraries", "kinetics")))
        has_content = False
        with open(file=os.path.join(ARC_PATH, "functional", "test", "kinetic", "output", "RMG libraries", "kinetics", "reactions.py"), mode='r') as f:
            for line in f.readlines():
                if "Arrhenius" in line:
                    has_content = True
                    break
        self.assertTrue(has_content)
    
    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        for folder in ['thermo', 'kinetic']:
            shutil.rmtree(os.path.join(ARC_PATH, 'functional', 'test', folder), ignore_errors=True)
        file_paths = [os.path.join(ARC_PATH, 'functional', 'nul'), os.path.join(ARC_PATH, 'functional', 'run.out')]
        for file_path in file_paths:
            if os.path.isfile(file_path):
                os.remove(file_path)

 
if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
