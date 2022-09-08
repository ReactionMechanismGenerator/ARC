#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains functional tests for ARC
"""

import os
import shutil
import subprocess
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
        
        cls.job_types = {'conformers': True,
                    'opt': True,
                    'fine_grid': True,
                    'freq': True,
                    'sp': True,
                    'rotors': False,
                    'irc': False,
                    }

        cls.species_list_1 = [ARCSpecies(label= "2-propanol", smiles="CC(O)C"), ARCSpecies(label= "1-propanol", smiles="CCCO"), ARCSpecies(label= "NN", smiles="NN")]

        cls.arc_object_1 = ARC(project='FunctionalThermoTest',
                               project_directory=f"{ARC_PATH}/functional/test/thermo",
                               species=cls.species_list_1,
                               job_types=cls.job_types,
                               conformer_level='gfn2',
                               level_of_theory='gfn2',
                               freq_scale_factor=1.0,
                               )
        cls.arc_object_1.execute()

        cls.species_list_2 = [ARCSpecies(label= "iC3H7", smiles="C[CH]C"), ARCSpecies(label= "nC3H7", smiles="CC[CH2]")]

        cls.arc_object_2 = ARC(project='FunctionalKineticTest',
                               project_directory=f"{ARC_PATH}/functional/test/kinetic",
                               reactions=[ARCReaction(label= "iC3H7 <=> nC3H7",r_species=[cls.species_list_2[0]], p_species=[cls.species_list_2[1]])],
                               species=cls.species_list_2,
                               job_types=cls.job_types,
                               conformer_level='gfn2',
                               level_of_theory='gfn2',
                               ts_guess_level = 'gfn2',
                               freq_scale_factor=1.0,
                               )
        cls.arc_object_2.execute()
    
    def testThermo(self):
        """Test thermo"""
        summary = self.arc_object_1.summary()
        for _, ter in summary.items():
            self.assertTrue(ter)

    def testKinetic(self):
        """Test kinetics"""
        summary = self.arc_object_2.summary()
        for _, ter in summary.items():
            self.assertTrue(ter)
    
    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'functional', 'test', 'kinetic'), ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_PATH, 'functional', 'test', 'thermo'), ignore_errors=True)

 
if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
