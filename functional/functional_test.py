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
                               arkane_level_of_theory='cbs-qb3',
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
                               arkane_level_of_theory='cbs-qb3',
                               freq_scale_factor=1.0,
                               n_confs=2,
                               dont_gen_confs=["TS0"],
                               bac_type=None,
                               verbose=1,
                               compare_to_rmg=False,
                               )
    
    def test_thermo(self):
        """Test thermo"""
        self.arc_object_1.execute()
        summary = self.arc_object_1.summary()
        for _, ter in summary.items():
            self.assertTrue(ter)
        self.assertTrue(os.path.isfile(os.path.join(ARC_PATH, "functional", "test", "thermo", "calcs", "statmech", "thermo", "arkane.log")))
        species, thermo = 0, 0
        with open(file=os.path.join(ARC_PATH, "functional", "test", "thermo", "calcs", "statmech", "thermo", "input.py"), mode='r') as f:
            for line in f.readlines():
                if "species('" in line:
                    species += 1
                if "thermo('" in line:
                    thermo += 1
        self.assertEqual(species, len(self.species_list_1))
        self.assertEqual(thermo, len(self.species_list_1))

    def test_kinetic(self):
        """Test kinetics"""
        self.arc_object_2.execute()
        summary = self.arc_object_2.summary()
        for _, ter in summary.items():
            self.assertTrue(ter)
        base_path = os.path.join(ARC_PATH, "functional", "test", "kinetic", "calcs", "statmech")
        self.assertTrue(os.path.isfile(os.path.join(base_path, "thermo", "input.py")))
        self.assertTrue(os.path.isfile(os.path.join(base_path, "kinetics", "input.py")))
        self.assertTrue(os.path.isfile(os.path.join(base_path, "kinetics", "species", "iC3H7.py")))
        self.assertTrue(os.path.isfile(os.path.join(base_path, "kinetics", "species", "nC3H7.py")))
        self.assertTrue(os.path.isfile(os.path.join(base_path, "kinetics", "TSs", "TS0.py")))
        species, thermo, kinetics, ts = 0, 0, 0, 0
        with open(file=os.path.join(base_path, "kinetics", "input.py"), mode='r') as f:
            for line in f.readlines():
                if "species('" in line:
                    species += 1
                elif "thermo('" in line:
                    thermo += 1
                elif "kinetics(label='" in line:
                    kinetics += 1
                elif "transitionState" in line:
                    ts += 1
        self.assertEqual(species, 2)
        self.assertEqual(thermo, 0)
        self.assertEqual(kinetics, 1)
        self.assertEqual(ts, 2)

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
