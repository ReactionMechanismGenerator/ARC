#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the plotter functions
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os
import shutil

from arc.settings import arc_path
from arc import plotter
from arc.species.species import ARCSpecies

################################################################################


class TestPlotter(unittest.TestCase):
    """
    Contains unit tests for the parser functions
    """

    def test_save_geo(self):
        """Test saving the geometry files for a species"""
        spc = ARCSpecies(label='methylamine', smiles=str('CN'), multiplicity=1, charge=0)
        spc.final_xyz = """N      -0.74566988   -0.11773792    0.00000000
C       0.70395487    0.03951260    0.00000000
H       1.12173564   -0.45689176   -0.87930074
H       1.06080468    1.07995075    0.00000000
H       1.12173564   -0.45689176    0.87930074
H      -1.16115119    0.31478894    0.81506145
H      -1.16115119    0.31478894   -0.81506145"""
        spc.opt_level = 'opt/level'
        project = 'arc_project_for_testing_delete_after_usage'
        project_directory = os.path.join(arc_path, 'Projects', project)
        xyz_path = os.path.join(project_directory, 'output', 'Species', spc.label, 'geometry', 'methylamine.xyz')
        gjf_path = os.path.join(project_directory, 'output', 'Species', spc.label, 'geometry', 'methylamine.gjf')
        plotter.save_geo(species=spc, project_directory=project_directory)
        xyz_data = """7
methylamine optimized at opt/level
N      -0.74566988   -0.11773792    0.00000000
C       0.70395487    0.03951260    0.00000000
H       1.12173564   -0.45689176   -0.87930074
H       1.06080468    1.07995075    0.00000000
H       1.12173564   -0.45689176    0.87930074
H      -1.16115119    0.31478894    0.81506145
H      -1.16115119    0.31478894   -0.81506145
"""
        gjf_data = """# hf/3-21g

methylamine optimized at opt/level

0 1
N      -0.74566988   -0.11773792    0.00000000
C       0.70395487    0.03951260    0.00000000
H       1.12173564   -0.45689176   -0.87930074
H       1.06080468    1.07995075    0.00000000
H       1.12173564   -0.45689176    0.87930074
H      -1.16115119    0.31478894    0.81506145
H      -1.16115119    0.31478894   -0.81506145
"""
        with open(xyz_path, 'r') as f:
            data = f.read()
        self.assertEqual(data, xyz_data)
        with open(gjf_path, 'r') as f:
            data = f.read()
        self.assertEqual(data, gjf_data)

    @classmethod
    def tearDownClass(cls):
        """A function that is run ONCE after all unit tests in this class."""
        project = 'arc_project_for_testing_delete_after_usage'
        project_directory = os.path.join(arc_path, 'Projects', project)
        shutil.rmtree(project_directory)


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
