#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the plotter functions
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import shutil
import unittest

from arc import plotter
from arc.settings import arc_path
from arc.species.species import ARCSpecies
from arc.species.converter import str_to_xyz


class TestPlotter(unittest.TestCase):
    """
    Contains unit tests for the parser functions
    """

    def test_save_geo(self):
        """Test saving the geometry files for a species"""
        spc = ARCSpecies(label='methylamine', smiles=str('CN'), multiplicity=1, charge=0)
        spc.final_xyz = str_to_xyz("""N      -0.74566988   -0.11773792    0.00000000
C       0.70395487    0.03951260    0.00000000
H       1.12173564   -0.45689176   -0.87930074
H       1.06080468    1.07995075    0.00000000
H       1.12173564   -0.45689176    0.87930074
H      -1.16115119    0.31478894    0.81506145
H      -1.16115119    0.31478894   -0.81506145""")
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

    def test_save_conformers_file(self):
        """test the save_conformers_file function"""
        project = 'arc_project_for_testing_delete_after_usage'
        project_directory = os.path.join(arc_path, 'Projects', project)
        label = 'butanol'
        spc1 = ARCSpecies(label=label, smiles='CCCCO')
        spc1.generate_conformers(confs_to_dft=3)
        self.assertIn(len(spc1.conformers), [2, 3])
        plotter.save_conformers_file(project_directory=project_directory, label=spc1.label,
                                     xyzs=spc1.conformers, level_of_theory='APFD/def2tzvp',
                                     multiplicity=spc1.multiplicity, charge=spc1.charge, is_ts=False,
                                     energies=spc1.conformer_energies)
        conf_file_path = os.path.join(project_directory, 'output', 'Species', label, 'geometry', 'conformers',
                                      'conformers_before_optimization.txt')
        self.assertTrue(os.path.isfile(conf_file_path))

    def test_save_rotor_text_file(self):
        """Test the save_rotor_text_file function"""
        project = 'arc_project_for_testing_delete_after_usage'
        angles = [0, 90, 180, 270, 360]
        energies = [0, 10, 0, 10, 0]
        pivots = [1, 2]
        path = os.path.join(arc_path, 'Projects', project, 'rotors', '{0}_directed_scan.txt'.format(pivots))
        plotter.save_rotor_text_file(angles, energies, path)
        self.assertTrue(os.path.isfile(path))
        with open(path, 'r') as f:
            lines = f.readlines()
        self.assertIn('Angle (degrees)        Energy (kJ/mol)\n', lines)

    @classmethod
    def tearDownClass(cls):
        """A function that is run ONCE after all unit tests in this class."""
        project = 'arc_project_for_testing_delete_after_usage'
        project_directory = os.path.join(arc_path, 'Projects', project)
        shutil.rmtree(project_directory)

################################################################################


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
