#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.processor module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os

from arc.processor import Processor
from arc.species.species import ARCSpecies
from arc.common import arc_path


################################################################################


class TestProcessor(unittest.TestCase):
    """
    Contains unit tests for the Processor class
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        project = 'processor_test'
        project_directory = os.path.join(arc_path, 'Projects', project)
        ch4 = ARCSpecies(label='CH4', smiles='C')
        nh3 = ARCSpecies(label='NH3', smiles='N')
        h = ARCSpecies(label='H', smiles='[H]')
        ch4_bde_1_2_A = ARCSpecies(label='CH4_BDE_1_2_A', smiles='[CH3]')
        ch4.e0, h.e0, ch4_bde_1_2_A.e0 = 10, 25, 35
        ch4.bdes = [(1, 2)]
        species_dict = {'CH4': ch4, 'NH3': nh3, 'H': h, 'CH4_BDE_1_2_A': ch4_bde_1_2_A}
        output_dict = {'CH4':
                           {'conformers': '',
                            'convergence': True,
                            'errors': '',
                            'info': '',
                            'isomorphism': '',
                            'job_types': {'rotors': True,
                                          'composite': False,
                                          'conformers': True,
                                          'fine_grid': False,
                                          'freq': True,
                                          'lennard_jones': False,
                                          'onedmin': False,
                                          'opt': True,
                                          'orbitals': False,
                                          'bde': True,
                                          'sp': True},
                            'paths': {'composite': '',
                                      'freq': '',
                                      'geo': '',
                                      'sp': ''},
                            'restart': '',
                            'warnings': ''},
                       'NH3':
                           {'conformers': '',
                            'convergence': True,
                            'errors': '',
                            'info': '',
                            'isomorphism': '',
                            'job_types': {'rotors': True,
                                          'composite': False,
                                          'conformers': True,
                                          'fine_grid': False,
                                          'freq': True,
                                          'lennard_jones': False,
                                          'onedmin': False,
                                          'opt': True,
                                          'orbitals': False,
                                          'bde': True,
                                          'sp': True},
                            'paths': {'composite': '',
                                      'freq': '',
                                      'geo': '',
                                      'sp': ''},
                            'restart': '',
                            'warnings': ''},
                       'H':
                           {'conformers': '',
                            'convergence': True,
                            'errors': '',
                            'info': '',
                            'isomorphism': '',
                            'job_types': {'rotors': True,
                                          'composite': False,
                                          'conformers': True,
                                          'fine_grid': False,
                                          'freq': False,
                                          'lennard_jones': False,
                                          'onedmin': False,
                                          'opt': False,
                                          'orbitals': False,
                                          'bde': True,
                                          'sp': True},
                            'paths': {'composite': '',
                                      'freq': '',
                                      'geo': '',
                                      'sp': ''},
                            'restart': '',
                            'warnings': ''},
                       'CH4_BDE_1_2_A':
                           {'conformers': '',
                            'convergence': True,
                            'errors': '',
                            'info': '',
                            'isomorphism': '',
                            'job_types': {'rotors': True,
                                          'composite': False,
                                          'conformers': True,
                                          'fine_grid': False,
                                          'freq': True,
                                          'lennard_jones': False,
                                          'onedmin': False,
                                          'opt': True,
                                          'orbitals': False,
                                          'bde': True,
                                          'sp': True},
                            'paths': {'composite': '',
                                      'freq': '',
                                      'geo': '',
                                      'sp': ''},
                            'restart': '',
                            'warnings': ''},
                       }
        cls.processor0 = Processor(project=project, project_directory=project_directory, species_dict=species_dict,
                                   rxn_list=list(), output=output_dict, use_bac=True,
                                   model_chemistry='b3lyp/6-311+g(3df,2p)', lib_long_desc='', rmgdatabase=None)

    def test_process_bdes(self):
        """Test the process_bdes method"""
        bde_report = self.processor0.process_bdes(label='CH4')
        self.assertEqual(bde_report, {(1, 2): 50})


################################################################################


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
