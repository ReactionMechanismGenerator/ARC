#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the plotter functions
"""

import os
import shutil
import subprocess
import tempfile
import unittest

import arc.plotter as plotter
from arc.common import ARC_TESTING_PATH, read_yaml_file, safe_copy_file
from arc.species.converter import str_to_xyz
from arc.species.species import ARCSpecies


OH_ADJLIST = """1 O u1 p2 c0 {2,S}
2 H u0 p0 c0 {1,S}"""

OH_NASA = ("NASA(polynomials=[NASAPolynomial(coeffs=[3.49683, 0.000188285, -1.03135e-06, 1.63951e-09, "
           "-6.45157e-13, 2675.74, 1.48391], Tmin=(10,'K'), Tmax=(974.045,'K')), "
           "NASAPolynomial(coeffs=[3.44056, -0.000267412, 7.28022e-07, -2.88523e-10, 3.54839e-14, 2719.28, "
           "1.92114], Tmin=(974.045,'K'), Tmax=(3000,'K'))], Tmin=(10,'K'), Tmax=(3000,'K'))")


def _rmg_env_available() -> bool:
    """Check whether the rmg_env conda environment is available."""
    try:
        result = subprocess.run(['conda', 'run', '-n', 'rmg_env', 'python', '-c', 'import rmgpy'],
                                capture_output=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False


RMG_ENV = _rmg_env_available()


def _make_oh_species(label: str) -> ARCSpecies:
    """Return an OH ARCSpecies carrying enough computed thermo to be written to a library."""
    spc = ARCSpecies(label=label, smiles='[OH]', multiplicity=2)
    spc.final_xyz = str_to_xyz("""O 0.0 0.0 0.0\nH 0.0 0.0 0.97""")
    spc.external_symmetry, spc.optical_isomers = 1, 1
    spc.thermo.data = OH_NASA
    spc.thermo.H298, spc.thermo.S298 = 30.93, 178.17
    return spc


class TestSaveThermoLibDuplicates(unittest.TestCase):
    """
    A reaction whose two reactants are the same species reaches save_thermo_lib as two separately
    labeled entries with identical adjacency lists. RMG refuses to load a library containing both.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='test_thermo_lib_')
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _lib_path(self, name='aa_project'):
        return os.path.join(self.tmp_dir, 'thermo', f'{name}.py')

    def test_identical_species_written_once(self):
        """The A+A duplicate is omitted from the library; the distinct species are both kept."""
        r1, r2 = _make_oh_species('R1'), _make_oh_species('R2')
        p1 = ARCSpecies(label='P1', smiles='O', multiplicity=1)
        p1.final_xyz = str_to_xyz("""O 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24""")
        p1.external_symmetry, p1.optical_isomers = 2, 1
        p1.thermo.data = OH_NASA
        p1.thermo.H298, p1.thermo.S298 = -240.63, 188.62
        plotter.save_thermo_lib([r1, r2, p1], path=self.tmp_dir, name='aa_project', lib_long_desc='test')
        with open(self._lib_path(), 'r') as f:
            content = f.read()
        self.assertIn('label = "R1"', content)
        self.assertNotIn('label = "R2"', content)
        self.assertIn('label = "P1"', content)
        self.assertEqual(content.count('entry('), 2)

    def test_distinct_species_all_written(self):
        """Regression: species that are not duplicates are all written."""
        a = _make_oh_species('A')
        b = ARCSpecies(label='B', smiles='O', multiplicity=1)
        b.final_xyz = str_to_xyz("""O 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24""")
        b.external_symmetry, b.optical_isomers = 2, 1
        b.thermo.data = OH_NASA
        b.thermo.H298, b.thermo.S298 = -240.63, 188.62
        plotter.save_thermo_lib([a, b], path=self.tmp_dir, name='distinct', lib_long_desc='test')
        with open(self._lib_path('distinct'), 'r') as f:
            content = f.read()
        self.assertEqual(content.count('entry('), 2)
        self.assertIn('label = "A"', content)
        self.assertIn('label = "B"', content)

    @unittest.skipUnless(RMG_ENV, 'rmg_env not available')
    def test_generated_library_loads_in_rmg(self):
        """The generated library must actually load with RMG, not merely look right."""
        r1, r2 = _make_oh_species('R1'), _make_oh_species('R2')
        plotter.save_thermo_lib([r1, r2], path=self.tmp_dir, name='aa_project', lib_long_desc='test')
        loader = os.path.join(self.tmp_dir, 'load_lib.py')
        with open(loader, 'w') as f:
            f.write('import sys\n'
                    'from rmgpy.data.thermo import ThermoLibrary\n'
                    'from rmgpy.thermo import NASA, NASAPolynomial, ThermoData, Wilhoit\n'
                    'local_context = {"ThermoData": ThermoData, "Wilhoit": Wilhoit,\n'
                    '                 "NASAPolynomial": NASAPolynomial, "NASA": NASA}\n'
                    'lib = ThermoLibrary()\n'
                    f'lib.load({self._lib_path()!r}, local_context, {{}})\n'
                    'sys.stdout.write("LOADED %d\\n" % len(lib.entries))\n')
        result = subprocess.run(['conda', 'run', '-n', 'rmg_env', 'python', loader],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0,
                         f'RMG could not load the generated thermo library:\n{result.stdout}\n{result.stderr}')
        self.assertIn('LOADED 1', result.stdout)


class TestPlotter(unittest.TestCase):
    """
    Contains unit tests for the parser functions
    """

    def setUp(self):
        """A method that is run before each unit test in this class."""
        self.scratch_dir = tempfile.mkdtemp(prefix='arc_test_plotter_')
        self.addCleanup(shutil.rmtree, self.scratch_dir, ignore_errors=True)
        self.project_directory = os.path.join(self.scratch_dir, 'arc_project_for_testing_delete_after_usage')

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
        project_directory = self.project_directory
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
        project_directory = self.project_directory
        n4h6_yml_path = os.path.join(ARC_TESTING_PATH, 'yml_testing', 'N4H6.yml')
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
        project_directory = self.project_directory
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
        angles = [0, 90, 180, 270, 360]
        energies = [0, 10, 0, 10, 0]
        pivots = [1, 2]
        path = os.path.join(self.project_directory, 'rotors', '{0}_directed_scan.txt'.format(pivots))
        plotter.save_rotor_text_file(angles, energies, path)
        self.assertTrue(os.path.isfile(path))
        with open(path, 'r') as f:
            lines = f.readlines()
        self.assertIn('Angle (degrees)        Energy (kJ/mol)\n', lines)

    def test_log_bde_report(self):
        """Test the log_bde_report() function"""
        path = os.path.join(self.scratch_dir, 'bde_report_test.txt')
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
        path = os.path.join(self.scratch_dir, 'mltspc_output.out')
        safe_copy_file(source=os.path.join(ARC_TESTING_PATH, 'mltspc_output.out'), destination=path)
        plotter.make_multi_species_output_file(species_list=[ARCSpecies(label='water', smiles='O', multi_species='mltspc1'),
                                                             ARCSpecies(label='acetylene', smiles='C#C', multi_species='mltspc1'),
                                                             ARCSpecies(label='N-Valeric_Acid', smiles='CCCCC(O)=O', multi_species='mltspc1')],
                                               label='mltspc1',
                                               path=path,
                                               )
        self.assertTrue(os.path.isfile(os.path.join(self.scratch_dir, 'water.log')))
        self.assertTrue(os.path.isfile(os.path.join(self.scratch_dir, 'acetylene.log')))
        self.assertTrue(os.path.isfile(os.path.join(self.scratch_dir, 'N-Valeric_Acid.log')))

    def test_delete_multi_species_output_file(self):
        """Test the delete_multi_species_output_file function"""
        # The xyzs used in the ARCSpecies are dummy xyzs, they are not the actual xyzs used in the output file
        path = os.path.join(self.scratch_dir, 'mltspc_output.out')
        safe_copy_file(source=os.path.join(ARC_TESTING_PATH, 'mltspc_output.out'), destination=path)
        species_list = [ARCSpecies(label='water', smiles='O', multi_species='mltspc1'),
                        ARCSpecies(label='acetylene', smiles='C#C', multi_species='mltspc1'),
                        ARCSpecies(label='N-Valeric_Acid', smiles='CCCCC(O)=O', multi_species='mltspc1')]
        multi_species_path_dict = plotter.make_multi_species_output_file(species_list=species_list,
                                                                         label='mltspc1',
                                                                         path=path,
                                                                         )
        self.assertTrue(os.path.isfile(os.path.join(self.scratch_dir, 'water.log')))
        self.assertTrue(os.path.isfile(os.path.join(self.scratch_dir, 'acetylene.log')))
        self.assertTrue(os.path.isfile(os.path.join(self.scratch_dir, 'N-Valeric_Acid.log')))
        plotter.delete_multi_species_output_file(species_list=species_list,
                                                 label='mltspc1',
                                                 multi_species_path_dict=multi_species_path_dict,
                                                 )
        self.assertFalse(os.path.isfile(os.path.join(self.scratch_dir, 'water.log')))
        self.assertFalse(os.path.isfile(os.path.join(self.scratch_dir, 'acetylene.log')))
        self.assertFalse(os.path.isfile(os.path.join(self.scratch_dir, 'N-Valeric_Acid.log')))

    def test_save_irc_traj_animation(self):
        """Test the save_irc_traj_animation function"""
        irc_f_path = os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out')
        irc_r_path = os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out')
        out_path = os.path.join(self.scratch_dir, 'irc', 'rxn_1_irc_animation.out')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.assertFalse(os.path.isfile(out_path))
        plotter.save_irc_traj_animation(irc_f_path, irc_r_path, out_path)
        self.assertTrue(os.path.isfile(out_path))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
