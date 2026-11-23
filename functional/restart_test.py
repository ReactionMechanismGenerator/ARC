#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.main module related to the restart feature of ARC
"""

import os
import shutil
import tempfile
import unittest
import warnings

from arc.molecule.molecule import Molecule

from arc.common import ARC_PATH, read_yaml_file
from arc.main import ARC


class TestRestart(unittest.TestCase):
    """
    Contains unit tests for restarting ARC.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        warnings.filterwarnings(action='ignore', module='.*matplotlib.*')

    def setUp(self):
        """Run each test from a scratch working directory so that ESS helper programs do not write into the repo."""
        working_directory = tempfile.mkdtemp(prefix='arc_test_restart_cwd_')
        original_directory = os.getcwd()
        os.chdir(working_directory)
        self.addCleanup(os.chdir, original_directory)
        self.addCleanup(shutil.rmtree, working_directory, ignore_errors=True)

    def make_project_directory(self, project: str) -> str:
        """Get a path to a temporary project directory that is deleted when the test ends."""
        project_directory = os.path.join(tempfile.mkdtemp(prefix='arc_test_restart_'), project)
        self.addCleanup(shutil.rmtree, os.path.dirname(project_directory), ignore_errors=True)
        return project_directory

    def copy_restart_file(self, restart_dir: str, project_directory: str) -> str:
        """Copy a fixture restart.yml into the project directory and return the copy's path."""
        os.makedirs(project_directory, exist_ok=True)
        return shutil.copy(os.path.join(restart_dir, 'restart.yml'), project_directory)

    def test_restart_thermo(self):
        """
        Test restarting ARC through the ARC class in main.py via the input_dict argument of the API
        Rather than through ARC.py. Check that all files are in place and the log file content.
        """
        restart_dir = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '1_restart_thermo')
        project = 'arc_project_for_testing_delete_after_usage_restart_thermo'
        project_directory = self.make_project_directory(project)
        restart_path = self.copy_restart_file(restart_dir, project_directory)
        os.makedirs(os.path.dirname(project_directory), exist_ok=True)
        shutil.copytree(os.path.join(restart_dir, 'calcs'), os.path.join(project_directory, 'calcs', 'Species'), dirs_exist_ok=True)
        input_dict = read_yaml_file(path=restart_path, project_directory=project_directory)
        input_dict['project'], input_dict['project_directory'] = project, project_directory
        arc1 = ARC(**input_dict)
        arc1.execute()
        self.assertEqual(arc1.freq_scale_factor, 0.988)

        self.assertTrue(os.path.isfile(os.path.join(project_directory, 'output', 'thermo.info')))
        with open(os.path.join(project_directory, 'output', 'thermo.info'), 'r') as f:
            thermo_dft_ccsdtf12_bac = False
            for line in f.readlines():
                if 'thermo_DFT_CCSDTF12_BAC' in line:
                    thermo_dft_ccsdtf12_bac = True
                    break
        self.assertTrue(thermo_dft_ccsdtf12_bac)

        with open(os.path.join(project_directory, f'{project}.info'), 'r') as f:
            sts, n2h3, oet, lot, ap = False, False, False, False, False
            for line in f.readlines():
                if 'Considered the following species and TSs:' in line:
                    sts = True
                elif 'Species N2H3' in line:
                    n2h3 = True
                elif 'Overall time since project initiation:' in line:
                    oet = True
                elif 'Levels of theory used:' in line:
                    lot = True
                elif f'ARC project {project}' in line:
                    ap = True
        self.assertTrue(sts)
        self.assertTrue(n2h3)
        self.assertTrue(oet)
        self.assertTrue(lot)
        self.assertTrue(ap)

        with open(os.path.join(project_directory, 'arc.log'), 'r') as f:
            aei, ver, git, spc, rtm, ldb, therm, src, ter =\
                False, False, False, False, False, False, False, False, False
            for line in f.readlines():
                if 'ARC execution initiated on' in line:
                    aei = True
                elif '#   Version:' in line:
                    ver = True
                elif 'The current git HEAD for ARC is:' in line:
                    git = True
                elif 'Considering species: CH3CO2_rad' in line:
                    spc = True
                elif 'All jobs for species N2H3 successfully converged. Run time' in line:
                    rtm = True
                elif '   H2O2              -135.82          239.05' in line:
                    therm = True
                elif 'Sources of thermodynamic properties determined by RMG for the parity plots:' in line:
                    src = True
                elif 'ARC execution terminated on' in line:
                    ter = True
        self.assertTrue(aei)
        self.assertTrue(ver)
        self.assertTrue(git)
        self.assertTrue(spc)
        self.assertTrue(rtm)
        self.assertTrue(therm)
        self.assertTrue(src)
        self.assertTrue(ter)

        self.assertTrue(os.path.isfile(os.path.join(project_directory, 'output', 'thermo_parity_plots.pdf')))

        status = read_yaml_file(os.path.join(project_directory, 'output', 'status.yml'))
        self.assertEqual(status['CH3CO2_rad']['isomorphism'],
                         'opt passed isomorphism check; '
                         'Conformers optimized and compared at b3lyp/6-31g(d,p) empiricaldispersion=gd3bj; ')
        self.assertTrue(status['CH3CO2_rad']['job_types']['sp'])

        with open(os.path.join(project_directory, 'output', 'RMG libraries', 'thermo', 'species_dictionary.txt'), 'r') as f:
            lines = f.readlines()
        species_name = 'H2O2'
        adj_lines = list()
        found = False
        for line in lines:
            text = line.strip()
            if not found:
                if text == species_name:
                    found = True
                continue
            if text == '':
                break
            adj_lines.append(line)
        adj_list = ''.join(adj_lines)
        mol1 = Molecule().from_adjacency_list(adj_list)
        self.assertEqual(mol1.to_smiles(), 'OO')

    def test_restart_rate_1(self):
        """Test restarting ARC and attaining a reaction rate coefficient"""
        restart_dir = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '2_restart_rate')
        project = 'arc_project_for_testing_delete_after_usage_restart_rate_1'
        project_directory = self.make_project_directory(project)
        restart_path = self.copy_restart_file(restart_dir, project_directory)
        os.makedirs(os.path.dirname(project_directory), exist_ok=True)
        shutil.copytree(os.path.join(restart_dir, 'calcs'), os.path.join(project_directory, 'calcs'), dirs_exist_ok=True)
        input_dict = read_yaml_file(path=restart_path, project_directory=project_directory)
        input_dict['project'], input_dict['project_directory'] = project, project_directory
        arc1 = ARC(**input_dict)
        arc1.execute()

        kinetics_library_path = os.path.join(project_directory, 'output', 'RMG libraries', 'kinetics', 'reactions.py')

        with open(kinetics_library_path, 'r') as f:
            got_rate = False
            for line in f.readlines():
                if "kinetics = Arrhenius(A=(6.37e-02, 'cm^3/(mol*s)'), n=4.08, Ea=(57.55, 'kJ/mol')," in line:
                    got_rate = True
                    break
        self.assertTrue(got_rate)

    def test_restart_rate_2(self):
        """Test restarting ARC and attaining a reaction rate coefficient"""
        project = 'arc_project_for_testing_delete_after_usage_restart_rate_2'
        project_directory = self.make_project_directory(project)
        base_path = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '5_TS1')
        restart_path = self.copy_restart_file(base_path, project_directory)
        input_dict = read_yaml_file(path=restart_path, project_directory=project_directory)
        input_dict['output']['TS0']['paths']['freq'] = os.path.join(ARC_PATH, input_dict['output']['TS0']['paths']['freq'])
        input_dict['output']['TS0']['paths']['geo'] = os.path.join(ARC_PATH, input_dict['output']['TS0']['paths']['geo'])
        input_dict['output']['TS0']['paths']['sp'] = os.path.join(ARC_PATH, input_dict['output']['TS0']['paths']['sp'])
        for spc in input_dict['species']:
            if 'TS' not in spc['label']:
                spc['yml_path'] = os.path.join(ARC_PATH, spc['yml_path'])
        input_dict['project'], input_dict['project_directory'] = project, project_directory
        arc1 = ARC(**input_dict)
        arc1.execute()

        kinetics_library_path = os.path.join(project_directory, 'output', 'RMG libraries', 'kinetics', 'reactions.py')

        with open(kinetics_library_path, 'r') as f:
            got_rate = False
            for line in f.readlines():
                if "Arrhenius(A=" in line:
                    got_rate = True
                    break
        self.assertTrue(got_rate)

    def test_restart_bde (self):
        """Test restarting ARC and attaining a BDE for anilino_radical."""
        restart_dir   = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '3_restart_bde')
        project = 'test_restart_bde'
        project_directory = self.make_project_directory(project)
        restart_path = self.copy_restart_file(restart_dir, project_directory)
        os.makedirs(os.path.dirname(project_directory), exist_ok=True)
        shutil.copytree(os.path.join(restart_dir, 'calcs'), os.path.join(project_directory, 'calcs'), dirs_exist_ok=True)
        input_dict = read_yaml_file(path=restart_path, project_directory=project_directory)
        input_dict['project'], input_dict['project_directory'] = project, project_directory
        arc1 = ARC(**input_dict)
        arc1.execute()

        report_path = os.path.join(project_directory, 'output', 'BDE_report.txt')
        with open(report_path, 'r') as f:
            lines = f.readlines()
        self.assertIn(' BDE report for anilino_radical:\n', lines)
        self.assertIn(' (1, 9)            N - H           353.92\n', lines)

    def test_globalize_paths(self):
        """Test modifying a YAML file's contents to correct absolute file paths"""
        fixture_dir = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '4_globalized_paths')
        project_directory = os.path.join(tempfile.mkdtemp(prefix='arc_test_restart_'), '4_globalized_paths')
        self.addCleanup(shutil.rmtree, os.path.dirname(project_directory), ignore_errors=True)
        shutil.copytree(fixture_dir, project_directory)
        restart_path = os.path.join(project_directory, 'restart_paths.yml')
        input_dict = read_yaml_file(path=restart_path, project_directory=project_directory)
        input_dict['project_directory'] = project_directory
        ARC(**input_dict)
        globalized_restart_path = os.path.join(project_directory, 'restart_paths_globalized.yml')
        content = read_yaml_file(globalized_restart_path)  # not giving a project directory, this is tested in common
        self.assertEqual(content['output']['restart'], 'Restarted ARC at 2020-02-28 12:51:14.446086; ')
        self.assertIn(os.path.join(project_directory, 'calcs', 'Species', 'HCN', 'freq_a38229', 'output.out'),
                      content['output']['spc']['paths']['freq'])
        self.assertNotIn('gpfs/workspace/users/user', content['output']['spc']['paths']['freq'])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))

