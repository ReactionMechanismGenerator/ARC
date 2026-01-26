#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.main module related to the restart feature of ARC
"""

import os
import shutil
import unittest
import warnings
from unittest import mock

from arc.molecule.molecule import Molecule

from arc.common import ARC_PATH, read_yaml_file, save_yaml_file
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

    def test_restart_thermo(self):
        """
        Test restarting ARC through the ARC class in main.py via the input_dict argument of the API
        Rather than through ARC.py. Check that all files are in place and the log file content.
        """
        restart_dir = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '1_restart_thermo')
        restart_path = os.path.join(restart_dir, 'restart.yml')
        project = 'arc_project_for_testing_delete_after_usage_restart_thermo'
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
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

        with open(os.path.join(project_directory, 'arc_project_for_testing_delete_after_usage_restart_thermo.info'), 'r') as f:
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
                elif 'ARC project arc_project_for_testing_delete_after_usage_restart_thermo' in line:
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
        restart_path = os.path.join(restart_dir, 'restart.yml')
        project = 'arc_project_for_testing_delete_after_usage_restart_rate_1'
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
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
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
        base_path = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '5_TS1')
        restart_path = os.path.join(base_path, 'restart.yml')
        input_dict = read_yaml_file(path=restart_path, project_directory=project_directory)
        input_dict['output']['TS0']['paths']['freq'] = os.path.join(ARC_PATH, input_dict['output']['TS0']['paths']['freq'])
        input_dict['output']['TS0']['paths']['goe'] = os.path.join(ARC_PATH, input_dict['output']['TS0']['paths']['geo'])
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
        restart_path  = os.path.join(restart_dir, 'restart.yml')
        project = 'test_restart_bde'
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
        os.makedirs(os.path.dirname(project_directory), exist_ok=True)
        shutil.copytree(os.path.join(restart_dir, 'calcs'), os.path.join(project_directory, 'calcs'), dirs_exist_ok=True)
        input_dict = read_yaml_file(path=restart_path, project_directory=project_directory)
        input_dict['project'], input_dict['project_directory'] = project, project_directory
        arc1 = ARC(**input_dict)
        arc1.execute()

        report_path = os.path.join(ARC_PATH, 'Projects', 'test_restart_bde', 'output', 'BDE_report.txt')
        with open(report_path, 'r') as f:
            lines = f.readlines()
        self.assertIn(' BDE report for anilino_radical:\n', lines)
        self.assertIn(' (1, 9)            N - H           353.92\n', lines)

    def test_globalize_paths(self):
        """Test modifying a YAML file's contents to correct absolute file paths"""
        project_directory = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '4_globalized_paths')
        restart_path = os.path.join(project_directory, 'restart_paths.yml')
        input_dict = read_yaml_file(path=restart_path, project_directory=project_directory)
        input_dict['project_directory'] = project_directory
        ARC(**input_dict)
        globalized_restart_path = os.path.join(project_directory, 'restart_paths_globalized.yml')
        content = read_yaml_file(globalized_restart_path)  # not giving a project directory, this is tested in common
        self.assertEqual(content['output']['restart'], 'Restarted ARC at 2020-02-28 12:51:14.446086; ')
        self.assertIn('ARC/arc/testing/restart/4_globalized_paths/calcs/Species/HCN/freq_a38229/output.out',
                      content['output']['spc']['paths']['freq'])
        self.assertNotIn('gpfs/workspace/users/user', content['output']['spc']['paths']['freq'])

    def test_restart_sanitizes_ts_output(self):
        """Test sanitizing inconsistent TS output on restart."""
        project = 'arc_project_for_testing_delete_after_usage_restart_sanitize_ts'
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
        os.makedirs(project_directory, exist_ok=True)
        restart_path = os.path.join(project_directory, 'restart.yml')
        restart_dict = {
            'project': project,
            'project_directory': project_directory,
            'job_types': {'conf_opt': False, 'conf_sp': False, 'opt': True, 'freq': True, 'sp': True,
                          'rotors': False, 'irc': False, 'fine': False},
            'species': [{'label': 'TS0', 'is_ts': True, 'multiplicity': 1, 'charge': 0}],
            'output': {
                'TS0': {
                    'paths': {'geo': '', 'freq': '', 'sp': '', 'composite': ''},
                    'restart': '',
                    'convergence': True,
                    'job_types': {'conf_opt': False, 'conf_sp': False, 'opt': True, 'freq': True, 'sp': True,
                                  'rotors': False, 'irc': False, 'fine': False, 'composite': False},
                }
            },
            'running_jobs': {'TS0': []},
        }
        save_yaml_file(path=restart_path, content=restart_dict)
        input_dict = read_yaml_file(path=restart_path, project_directory=project_directory)
        input_dict['project'], input_dict['project_directory'] = project, project_directory
        with mock.patch('arc.scheduler.Scheduler.schedule_jobs', return_value=None), \
                mock.patch('arc.main.process_arc_project', return_value=None):
            arc1 = ARC(**input_dict)
            arc1.execute()
        self.assertFalse(arc1.scheduler.output['TS0']['convergence'])

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage_restart_thermo',
                    'arc_project_for_testing_delete_after_usage_restart_rate_1',
                    'arc_project_for_testing_delete_after_usage_restart_rate_2',
                    'test_restart_bde',
                    'arc_project_for_testing_delete_after_usage_restart_sanitize_ts',
                    ]
        for project in projects:
            project_directory = os.path.join(ARC_PATH, 'Projects', project)
            shutil.rmtree(project_directory, ignore_errors=True)

        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '4_globalized_paths',
                                   'log_and_restart_archive'), ignore_errors=True)
        for file_name in ['arc.log', 'restart_paths_globalized.yml']:
            file_path = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '4_globalized_paths', file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        file_paths = [os.path.join(ARC_PATH, 'functional', 'nul'), os.path.join(ARC_PATH, 'functional', 'run.out')]
        project_names = ['1_restart_thermo', '2_restart_rate', '3_restart_bde', '5_TS1']
        for project_name in project_names:
            file_paths.append(os.path.join(ARC_PATH, 'arc', 'testing', 'restart', project_name, 'restart_globalized.yml'))
        for file_path in file_paths:
            if os.path.isfile(file_path):
                os.remove(file_path)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
