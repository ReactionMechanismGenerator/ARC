#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the plotter functions
"""

import os
import shutil
import unittest

import arc.plotter as plotter
from arc.common import ARC_PATH, read_yaml_file, safe_copy_file
from arc.species.converter import str_to_xyz
from arc.species.species import ARCSpecies


class TestPlotter(unittest.TestCase):
    """
    Contains unit tests for the parser functions
    """

    def test_save_geo(self):
        """Test saving the geometry files for a species"""
        spc = ARCSpecies(label='methylamine', smiles='CN', multiplicity=1, charge=0)
        spc.final_xyz = str_to_xyz("""N      -0.74566988   -0.11773792    0.00000000
C       0.70395487    0.03951260    0.00000000
H       1.12173564   -0.45689176   -0.87930074
H       1.06080468    1.07995075    0.00000000
H       1.12173564   -0.45689176    0.87930074
H      -1.16115119    0.31478894    0.81506145
H      -1.16115119    0.31478894   -0.81506145""")
        spc.opt_level = 'opt/level'
        project = 'arc_project_for_testing_delete_after_usage'
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
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

    def test_augment_arkane_yml_file_with_mol_repr(self):
        """Test the augment_arkane_yml_file_with_mol_repr() function"""
        project = 'arc_project_for_testing_delete_after_usage'
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
        n4h6_yml_path = os.path.join(ARC_PATH, 'arc', 'testing', 'yml_testing', 'N4H6.yml')
        n4h6_yml_path_copy = os.path.join(project_directory, 'Species', 'N4H6', 'N4H6.yml')
        os.makedirs(os.path.join(project_directory, 'Species', 'N4H6'), exist_ok=True)
        safe_copy_file(source=n4h6_yml_path, destination=n4h6_yml_path_copy)
        content_0 = read_yaml_file(path=n4h6_yml_path_copy)
        self.assertNotIn('mol', content_0.keys())
        n4h6 = ARCSpecies(label='N4H6', smiles='NNNN')
        plotter.augment_arkane_yml_file_with_mol_repr(species=n4h6, output_directory=project_directory)
        content_1 = read_yaml_file(path=n4h6_yml_path_copy)
        self.assertIn('mol', content_1.keys())

    def test_save_conformers_file(self):
        """test the save_conformers_file function"""
        project = 'arc_project_for_testing_delete_after_usage'
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
        label = 'butanol'
        spc1 = ARCSpecies(label=label, smiles='CCCCO')
        spc1.generate_conformers(n_confs=3)
        self.assertIn(len(spc1.conformers), [2, 3])
        plotter.save_conformers_file(project_directory=project_directory, label=spc1.label,
                                     xyzs=spc1.conformers, level_of_theory='APFD/def2tzvp',
                                     multiplicity=spc1.multiplicity, charge=spc1.charge, is_ts=False,
                                     energies=spc1.conformer_energies,
                                     before_optimization=True,)
        conf_file_path = os.path.join(project_directory, 'output', 'Species', label, 'geometry', 'conformers',
                                      'conformers_before_optimization.txt')
        self.assertTrue(os.path.isfile(conf_file_path))

    def test_save_rotor_text_file(self):
        """Test the save_rotor_text_file function"""
        project = 'arc_project_for_testing_delete_after_usage'
        angles = [0, 90, 180, 270, 360]
        energies = [0, 10, 0, 10, 0]
        pivots = [1, 2]
        path = os.path.join(ARC_PATH, 'Projects', project, 'rotors', '{0}_directed_scan.txt'.format(pivots))
        plotter.save_rotor_text_file(angles, energies, path)
        self.assertTrue(os.path.isfile(path))
        with open(path, 'r') as f:
            lines = f.readlines()
        self.assertIn('Angle (degrees)        Energy (kJ/mol)\n', lines)

    def test_log_bde_report(self):
        """Test the log_bde_report() function"""
        path = os.path.join(ARC_PATH, 'arc', 'testing', 'bde_report_test.txt')
        bde_report = {'aniline': {(1, 2): 431.43, (5, 8): 465.36, (6, 9): 458.70, (3, 10): 463.16, (4, 11): 463.16,
                                  (7, 12): 458.70, (1, 13): 372.31, (1, 14): 372.31, (5, 6): 'N/A'}}
        xyz = """N       2.28116100   -0.20275000   -0.29653100
        C       0.90749600   -0.08067400   -0.11852200
        C       0.09862900   -1.21367300   -0.02143500
        C       0.30223500    1.17638000   -0.08930600
        C      -1.87236600    0.16329100    0.13332800
        C      -1.27400900   -1.08769400    0.10342700
        C      -1.07133200    1.29144700    0.03586700
        H      -2.94554700    0.25749800    0.23136900
        H      -1.88237600   -1.98069300    0.17844600
        H       0.55264300   -2.19782900   -0.04842100
        H       0.91592000    2.06653500   -0.16951700
        H      -1.51965000    2.27721000    0.05753400
        H       2.68270800   -1.06667200    0.02551200
        H       2.82448700    0.59762700   -0.02174900"""
        aniline = ARCSpecies(label='aniline', xyz=xyz, smiles='c1ccc(cc1)N', bdes=['all_h', (1, 2), (5, 6)])
        spc_dict = {'aniline': aniline}
        plotter.log_bde_report(path, bde_report, spc_dict)

        with open(path, 'r') as f:
            content = f.read()
        expected_content = """ BDE report for aniline:
  Pivots           Atoms        BDE (kJ/mol)
 --------          -----        ------------
 (1, 13)           N - H           372.31
 (1, 14)           N - H           372.31
 (1, 2)            N - C           431.43
 (6, 9)            C - H           458.70
 (7, 12)           C - H           458.70
 (3, 10)           C - H           463.16
 (4, 11)           C - H           463.16
 (5, 8)            C - H           465.36
 (5, 6)            C - C           N/A


"""
        self.assertEqual(content, expected_content)

    def test_clean_scan_results(self):
        """Test the clean_scan_results function"""
        correct_results = {(1, 1): {'energy': 0},
                           (1, 2): {'energy': 7},
                           (1, 3): {'energy': 4.5},
                           (1, 4): {'energy': 5}}

        results_1 = {(1, 1): {'energy': -2},
                     (1, 2): {'energy': '5'},
                     (1, 3): {'energy': 2.5},
                     (1, 4): {'energy': 3}}
        filtered_results_1 = plotter.clean_scan_results(results_1)
        self.assertEqual(filtered_results_1, correct_results)

        results_2 = {(1, 1): {'energy': '-2'},
                     (1, 2): {'energy': 5},
                     (1, 3): {'energy': 2.5},
                     (1, 4): {'energy': 3},
                     (1, 5): {'energy': 1100}}
        filtered_results_2 = plotter.clean_scan_results(results_2)
        self.assertEqual(filtered_results_2, correct_results)

    def test_make_multi_species_output_file(self):
        """Test the make_multi_species_output_file function"""
        # The xyzs used in the ARCSpecies are dummy xyzs, they are not the actual xyzs used in the output file
        plotter.make_multi_species_output_file(species_list=[ARCSpecies(label='water', smiles='O', multi_species='mltspc1'),
                                                             ARCSpecies(label='acetylene', smiles='C#C', multi_species='mltspc1'),
                                                             ARCSpecies(label='N-Valeric_Acid', smiles='CCCCC(O)=O', multi_species='mltspc1')],
                                               label='mltspc1',
                                               path=os.path.join(ARC_PATH, 'arc', 'testing', 'mltspc_output.out'),
                                               )
        self.assertTrue(os.path.isfile(os.path.join(ARC_PATH, 'arc', 'testing', 'water.log')))
        self.assertTrue(os.path.isfile(os.path.join(ARC_PATH, 'arc', 'testing', 'acetylene.log')))
        self.assertTrue(os.path.isfile(os.path.join(ARC_PATH, 'arc', 'testing', 'N-Valeric_Acid.log')))

    def test_delete_multi_species_output_file(self):
        """Test the delete_multi_species_output_file function"""
        # The xyzs used in the ARCSpecies are dummy xyzs, they are not the actual xyzs used in the output file
        species_list = [ARCSpecies(label='water', smiles='O', multi_species='mltspc1'),
                        ARCSpecies(label='acetylene', smiles='C#C', multi_species='mltspc1'),
                        ARCSpecies(label='N-Valeric_Acid', smiles='CCCCC(O)=O', multi_species='mltspc1')]
        multi_species_path_dict = plotter.make_multi_species_output_file(species_list=species_list,
                                                                         label='mltspc1',
                                                                         path=os.path.join(ARC_PATH, 'arc', 'testing', 'mltspc_output.out'),
                                                                         )
        self.assertTrue(os.path.isfile(os.path.join(ARC_PATH, 'arc', 'testing', 'water.log')))
        self.assertTrue(os.path.isfile(os.path.join(ARC_PATH, 'arc', 'testing', 'acetylene.log')))
        self.assertTrue(os.path.isfile(os.path.join(ARC_PATH, 'arc', 'testing', 'N-Valeric_Acid.log')))
        plotter.delete_multi_species_output_file(species_list=species_list,
                                                 label='mltspc1',
                                                 multi_species_path_dict=multi_species_path_dict,
                                                 )
        self.assertFalse(os.path.isfile(os.path.join(ARC_PATH, 'arc', 'testing', 'water.log')))
        self.assertFalse(os.path.isfile(os.path.join(ARC_PATH, 'arc', 'testing', 'acetylene.log')))
        self.assertFalse(os.path.isfile(os.path.join(ARC_PATH, 'arc', 'testing', 'N-Valeric_Acid.log')))

    def test_save_irc_traj_animation(self):
        """Test the save_irc_traj_animation function"""
        irc_f_path = os.path.join(ARC_PATH, 'arc', 'testing', 'irc', 'rxn_1_irc_1.out')
        irc_r_path = os.path.join(ARC_PATH, 'arc', 'testing', 'irc', 'rxn_1_irc_2.out')
        out_path = os.path.join(ARC_PATH, 'arc', 'testing', 'irc', 'rxn_1_irc_animation.out')
        self.assertFalse(os.path.isfile(out_path))
        plotter.save_irc_traj_animation(irc_f_path, irc_r_path, out_path)
        self.assertTrue(os.path.isfile(out_path))


    @classmethod
    def tearDownClass(cls):
        """A function that is run ONCE after all unit tests in this class."""
        project = 'arc_project_for_testing_delete_after_usage'
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
        shutil.rmtree(project_directory, ignore_errors=True)
        files_to_remove = [os.path.join(ARC_PATH, 'arc', 'testing', 'bde_report_test.txt'),
                           os.path.join(ARC_PATH, 'arc', 'testing', 'water.log'),
                           os.path.join(ARC_PATH, 'arc', 'testing', 'acetylene.log'),
                           os.path.join(ARC_PATH, 'arc', 'testing', 'N-Valeric_Acid.log'),
                           os.path.join(ARC_PATH, 'arc', 'testing', 'irc', 'rxn_1_irc_animation.out'),
                           ]
        for file_path in files_to_remove:
            if os.path.isfile(file_path):
                os.remove(file_path)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
