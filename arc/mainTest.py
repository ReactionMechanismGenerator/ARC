#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the arc.main module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os
import shutil

from rmgpy import settings
from rmgpy.data.rmg import RMGDatabase
from rmgpy.species import Species
from rmgpy.molecule.molecule import Molecule

from arc.main import ARC
from arc.species.species import ARCSpecies
from arc.settings import arc_path
from arc.arc_exceptions import InputError

################################################################################


class TestARC(unittest.TestCase):
    """
    Contains unit tests for the ARC class
    """

    def test_as_dict(self):
        """Test the as_dict() method of ARC"""
        self.maxDiff = None
        ess_settings = {}
        spc1 = ARCSpecies(label='spc1', smiles=str('CC'), generate_thermo=False)
        arc0 = ARC(project='arc_test', ess_settings=ess_settings, scan_rotors=False, initial_trsh='scf=(NDump=30)',
                   arc_species_list=[spc1])
        restart_dict = arc0.as_dict()
        expected_dict = {'composite_method': '',
                         'conformer_level': 'b97-d3/6-311+g(d,p)',
                         'ts_guess_level': 'b3lyp/6-31+g(d,p)',
                         'ess_settings': {'ssh': True},
                         'fine': True,
                         'opt_level': 'wb97xd/6-311++g(d,p)',
                         'freq_level': 'wb97xd/6-311++g(d,p)',
                         'generate_conformers': True,
                         'initial_trsh': 'scf=(NDump=30)',
                         'max_job_time': 120,
                         'model_chemistry': 'ccsd(t)-f12/cc-pvtz-f12',
                         'output': {},
                         'project': 'arc_test',
                         'running_jobs': {},
                         'reactions': [],
                         'scan_level': '',
                         'scan_rotors': False,
                         'sp_level': 'ccsd(t)-f12/cc-pvtz-f12',
                         't_min': None,
                         't_max': None,
                         't_count': None,
                         'use_bac': True,
                         'allow_nonisomorphic_2d': False,
                         'species': [{'bond_corrections': {'C-C': 1, 'C-H': 6},
                                      'arkane_file': None,
                                      'E0': None,
                                      'charge': 0,
                                      'external_symmetry': None,
                                      'optical_isomers': None,
                                      'final_xyz': '',
                                      'generate_thermo': False,
                                      'is_ts': False,
                                      'label': u'spc1',
                                      'long_thermo_description': "Bond corrections: {'C-C': 1, 'C-H': 6}\n",
                                      'mol': '1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}\n2 C u0 p0 c0 {1,S} {6,S} {7,S} {8,S}\n3 H u0 p0 c0 {1,S}\n4 H u0 p0 c0 {1,S}\n5 H u0 p0 c0 {1,S}\n6 H u0 p0 c0 {2,S}\n7 H u0 p0 c0 {2,S}\n8 H u0 p0 c0 {2,S}\n',
                                      'multiplicity': 1,
                                      'neg_freqs_trshed': [],
                                      'number_of_rotors': 0,
                                      'opt_level': '',
                                      'rotors_dict': {},
                                      't1': None}],
                         }
        self.assertEqual(restart_dict, expected_dict)

    def test_from_dict(self):
        """Test the from_dict() method of ARC"""
        restart_dict = {'composite_method': '',
                         'conformer_level': 'b97-d3/6-311+g(d,p)',
                         'ess_settings': {'gaussian': 'server1',
                                          'molpro': 'server1',
                                          'qchem': u'server1',
                                          'ssh': True},
                         'fine': True,
                         'freq_level': 'wb97x-d3/6-311+g(d,p)',
                         'generate_conformers': True,
                         'initial_trsh': 'scf=(NDump=30)',
                         'model_chemistry': 'ccsd(t)-f12/cc-pvtz-f12',
                         'opt_level': 'wb97x-d3/6-311+g(d,p)',
                         'output': {},
                         'project': 'arc_test',
                         'rxn_list': [],
                         'scan_level': '',
                         'scan_rotors': False,
                         'sp_level': 'ccsdt-f12/cc-pvqz-f12',
                         'species': [{'bond_corrections': {'C-C': 1, 'C-H': 6},
                                      'charge': 1,
                                      'conformer_energies': [],
                                      'conformers': [],
                                      'external_symmetry': 1,
                                      'final_xyz': '',
                                      'generate_thermo': False,
                                      'is_ts': False,
                                      'label': 'testing_spc1',
                                      'mol': '1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}\n2 C u0 p0 c0 {1,S} {6,S} {7,S} {8,S}\n3 H u0 p0 c0 {1,S}\n4 H u0 p0 c0 {1,S}\n5 H u0 p0 c0 {1,S}\n6 H u0 p0 c0 {2,S}\n7 H u0 p0 c0 {2,S}\n8 H u0 p0 c0 {2,S}\n',
                                      'multiplicity': 1,
                                      'neg_freqs_trshed': [],
                                      'number_of_rotors': 0,
                                      'opt_level': '',
                                      'optical_isomers': 1,
                                      'rotors_dict': {},
                                      'xyzs': []}],
                         'use_bac': True}
        arc1 = ARC(project='wrong', ess_settings=dict())
        project = 'arc_project_for_testing_delete_after_usage1'
        project_directory = os.path.join(arc_path, 'Projects', project)
        arc1.from_dict(input_dict=restart_dict, project='testing_from_dict', project_directory=project_directory)
        self.assertEqual(arc1.project, 'testing_from_dict')
        self.assertTrue('arc_project_for_testing_delete_after_usage' in arc1.project_directory)
        self.assertTrue(arc1.fine)
        self.assertFalse(arc1.scan_rotors)
        self.assertEqual(arc1.sp_level, 'ccsdt-f12/cc-pvqz-f12')
        self.assertEqual(arc1.arc_species_list[0].label, 'testing_spc1')
        self.assertFalse(arc1.arc_species_list[0].is_ts)
        self.assertEqual(arc1.arc_species_list[0].charge, 1)

    def test_check_project_name(self):
        """Test project name invalidity"""
        ess_settings = {}
        with self.assertRaises(InputError):
            ARC(project='ar c', ess_settings=ess_settings)
        with self.assertRaises(InputError):
            ARC(project='ar:c', ess_settings=ess_settings)
        with self.assertRaises(InputError):
            ARC(project='ar<c', ess_settings=ess_settings)
        with self.assertRaises(InputError):
            ARC(project='ar%c', ess_settings=ess_settings)

    def test_restart(self):
        """
        Test restarting ARC through the ARC class in main.py via the input_dict argument of the API
        Rather than through ARC.py. Check that all files are in place and tst file content.
        """
        restart_path = os.path.join(arc_path, 'arc', 'testing', 'restart(H,H2O2,N2H3,CH3CO2).yml')
        project = 'arc_project_for_testing_delete_after_usage2'
        project_directory = os.path.join(arc_path, 'Projects', project)
        arc1 = ARC(project=project, ess_settings=dict(),
                   input_dict=restart_path, project_directory=project_directory)
        arc1.execute()

        with open(os.path.join(project_directory, 'output', 'thermo.info'), 'r') as f:
            thermo_sft_ccsdtf12_bac = False
            for line in f.readlines():
                if 'thermo_DFT_CCSDTF12_BAC' in line:
                    thermo_sft_ccsdtf12_bac = True
                    break
        self.assertTrue(thermo_sft_ccsdtf12_bac)

        with open(os.path.join(project_directory, 'arc_project_for_testing_delete_after_usage2.info'), 'r') as f:
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
                elif 'ARC project arc_project_for_testing_delete_after_usage2' in line:
                    ap = True
        self.assertTrue(sts)
        self.assertTrue(n2h3)
        self.assertTrue(oet)
        self.assertTrue(lot)
        self.assertTrue(ap)

        with open(os.path.join(project_directory, 'arc.log'), 'r') as f:
            aei, ver, git, spc, rtm, ldb, therm, src, ter = False, False, False, False, False, False, False, False, False
            for line in f.readlines():
                if 'ARC execution initiated on' in line:
                    aei = True
                elif '#   Version:' in line:
                    ver = True
                elif 'The current git HEAD for ARC is:' in line:
                    git = True
                elif 'Considering species: CH3CO2_rad' in line:
                    spc = True
                elif 'All jobs for species N2H3 successfully converged. Run time: 1:16:03' in line:
                    rtm = True
                elif 'Loading the RMG database...' in line:
                    ldb = True
                elif 'Thermodynamics for H2O2:' in line:
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

        with open(os.path.join(project_directory, 'output', 'Species', 'H2O2', 'species_dictionary.txt'), 'r') as f:
            lines = f.readlines()
        adj_list = str(''.join([line for line in lines if (line and 'H2O2' not in line)]))
        mol1 = Molecule().fromAdjacencyList(adj_list)
        self.assertEqual(mol1.toSMILES(), str('OO'))

        thermo_library_path = os.path.join(project_directory, 'output', 'RMG libraries', 'thermo',
                                           'arc_project_for_testing_delete_after_usage2.py')
        new_thermo_library_path = os.path.join(settings['database.directory'], 'thermo', 'libraries',
                                               'arc_project_for_testing_delete_after_usage2.py')
        # copy the generated library to RMG-database
        shutil.copyfile(thermo_library_path, new_thermo_library_path)
        db = RMGDatabase()
        db.load(
            path=settings['database.directory'],
            thermoLibraries=[str('arc_project_for_testing_delete_after_usage2')],
            transportLibraries=[],
            reactionLibraries=[],
            seedMechanisms=[],
            kineticsFamilies='none',
            kineticsDepositories=[],
            statmechLibraries=None,
            depository=False,
            solvation=False,
            testing=True,
        )

        spc2 = Species().fromSMILES(str('CC([O])=O'))
        spc2.generate_resonance_structures()
        spc2.thermo = db.thermo.getThermoData(spc2)
        self.assertAlmostEqual(spc2.getEnthalpy(298), -178003.44650359568, 1)
        self.assertAlmostEqual(spc2.getEntropy(298), 283.5983103176096, 1)
        self.assertAlmostEqual(spc2.getHeatCapacity(1000), 118.99753808225603, 1)
        self.assertTrue('arc_project_for_testing_delete_after_usage2' in spc2.thermo.comment)

        # delete the generated library from RMG-database
        os.remove(new_thermo_library_path)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage1', 'arc_project_for_testing_delete_after_usage2',
                    'ar c', 'ar:c', 'ar<c', 'ar%c']
        for project in projects:
            project_directory = os.path.join(arc_path, 'Projects', project)
            shutil.rmtree(project_directory)

################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
