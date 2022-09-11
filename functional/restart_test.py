#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.main module related to the restart feature of ARC
"""

import os
import shutil
import unittest
import warnings

from rmgpy import settings as rmg_settings
from rmgpy.data.rmg import RMGDatabase
from rmgpy.molecule.molecule import Molecule
from rmgpy.species import Species

from arc.common import ARC_PATH, read_yaml_file
from arc.main import ARC


class TestARC(unittest.TestCase):
    """
    Contains unit tests for the ARC class
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
        restart_path = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '1_restart_thermo', 'restart.yml')
        input_dict = read_yaml_file(path=restart_path)
        project = 'arc_project_for_testing_delete_after_usage_restart_thermo'
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
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
                elif 'Loading the RMG database...' in line:
                    ldb = True
                elif 'Thermodynamics for H2O2' in line:
                    therm = True
                elif 'Sources of thermoproperties determined by RMG for the parity plots:' in line:
                    src = True
                elif 'ARC execution terminated on' in line:
                    ter = True
        self.assertTrue(aei)
        self.assertTrue(ver)
        self.assertTrue(git)
        self.assertTrue(spc)
        self.assertTrue(rtm)
        self.assertTrue(ldb)
        self.assertTrue(therm)
        self.assertTrue(src)
        self.assertTrue(ter)

        self.assertTrue(os.path.isfile(os.path.join(project_directory, 'output', 'thermo_parity_plots.pdf')))

        status = read_yaml_file(os.path.join(project_directory, 'output', 'status.yml'))
        self.assertEqual(status['CH3CO2_rad']['isomorphism'],
                         'opt passed isomorphism check; '
                         'Conformers optimized and compared at b3lyp/6-31g(d,p) empiricaldispersion=gd3bj; ')
        self.assertTrue(status['CH3CO2_rad']['job_types']['sp'])

        with open(os.path.join(project_directory, 'output', 'Species', 'H2O2', 'arkane', 'species_dictionary.txt'),
                  'r') as f:
            lines = f.readlines()
        adj_list = ''
        for line in lines:
            if 'H2O2' not in line:
                adj_list += line
            if line == '\n':
                break
        mol1 = Molecule().from_adjacency_list(adj_list)
        self.assertEqual(mol1.to_smiles(), 'OO')

        thermo_library_path = os.path.join(project_directory, 'output', 'RMG libraries', 'thermo',
                                           'arc_project_for_testing_delete_after_usage_restart_thermo.py')
        new_thermo_library_path = os.path.join(rmg_settings['database.directory'], 'thermo', 'libraries',
                                               'arc_project_for_testing_delete_after_usage_restart_thermo.py')
        # copy the generated library to RMG-database
        shutil.copyfile(thermo_library_path, new_thermo_library_path)
        db = RMGDatabase()
        db.load(
            path=rmg_settings['database.directory'],
            thermo_libraries=['arc_project_for_testing_delete_after_usage_restart_thermo'],
            transport_libraries=[],
            reaction_libraries=[],
            seed_mechanisms=[],
            kinetics_families='none',
            kinetics_depositories=[],
            statmech_libraries=None,
            depository=False,
            solvation=False,
            testing=True,
        )

        spc2 = Species(smiles='CC([O])=O')
        spc2.generate_resonance_structures()
        spc2.thermo = db.thermo.get_thermo_data(spc2)
        self.assertAlmostEqual(spc2.get_enthalpy(298), -212439.26998495663, 1)
        self.assertAlmostEqual(spc2.get_entropy(298), 283.3972662956835, 1)
        self.assertAlmostEqual(spc2.get_heat_capacity(1000), 118.751379824224, 1)
        self.assertTrue('arc_project_for_testing_delete_after_usage_restart_thermo' in spc2.thermo.comment)

        # delete the generated library from RMG-database
        os.remove(new_thermo_library_path)

    def test_restart_rate(self):
        """Test restarting ARC and attaining reaction a rate coefficient"""
        restart_path = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '2_restart_rate', 'restart.yml')
        input_dict = read_yaml_file(path=restart_path)
        project = 'arc_project_for_testing_delete_after_usage_restart_rate'
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
        input_dict['project'], input_dict['project_directory'] = project, project_directory
        arc1 = ARC(**input_dict)
        arc1.execute()

        kinetics_library_path = os.path.join(project_directory, 'output', 'RMG libraries', 'kinetics', 'reactions.py')

        with open(kinetics_library_path, 'r') as f:
            got_rate = False
            for line in f.readlines():
                if "Arrhenius(A=(0.0636958,'cm^3/(mol*s)'), n=4.07981, Ea=(57.5474,'kJ/mol')" in line:
                    got_rate = True
                    break
        self.assertTrue(got_rate)

    def test_restart_bde(self):
        """Test restarting ARC and attaining reaction a rate coefficient"""
        restart_path = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '3_restart_bde', 'restart.yml')
        input_dict = read_yaml_file(path=restart_path)
        arc1 = ARC(**input_dict)
        arc1.execute()

        report_path = os.path.join(ARC_PATH, 'Projects', 'test_restart_bde', 'output', 'BDE_report.txt')
        with open(report_path, 'r') as f:
            lines = f.readlines()
        self.assertIn(' BDE report for anilino_radical:\n', lines)
        self.assertIn(' (1, 9)            N - H           353.92\n', lines)
        self.assertIn(' (3, 4)            C - H           454.12\n', lines)
        self.assertIn(' (5, 10)           C - H           461.75\n', lines)

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

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage_restart_thermo',
                    'arc_project_for_testing_delete_after_usage_restart_rate',
                    'test_restart_bde',
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


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
