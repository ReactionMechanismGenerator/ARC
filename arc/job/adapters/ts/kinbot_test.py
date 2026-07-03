#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.kinbot_ts module
"""

import os
import shutil
import subprocess
import unittest
from unittest import mock

from arc.common import ARC_TESTING_PATH, read_yaml_file, save_yaml_file
import arc.job.adapters.ts.kinbot_ts as kinbot_ts
from arc.job.adapters.ts.kinbot_ts import KinBotAdapter
from arc.reaction import ARCReaction
from arc.species import ARCSpecies


def kinbot_list_to_coords(structure: list) -> list:
    """A helper function to convert a flat KinBot structure list into an N x 3 coordinates list."""
    coords = list()
    for i in range(0, len(structure), 4):
        coords.append([float(structure[i + 1]), float(structure[i + 2]), float(structure[i + 3])])
    return coords


class TestKinBotAdapter(unittest.TestCase):
    """
    Contains unit tests for the KinBotAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.project_dir = os.path.join(ARC_TESTING_PATH, 'test_KinBot')

    def setUp(self):
        """
        A method that is run before each unit test in this class.
        """
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.rxn_1 = ARCReaction(reactants=['CC[O]'], products=['[CH2]CO'],
                                 r_species=[ARCSpecies(label='CC[O]', smiles='CC[O]')],
                                 p_species=[ARCSpecies(label='[CH2]CO', smiles='[CH2]CO')])

    def get_adapter(self, dir_name: str = 'tst1') -> KinBotAdapter:
        """A helper function to instantiate a KinBotAdapter instance."""
        return KinBotAdapter(job_type='tsg',
                             reactions=[self.rxn_1],
                             testing=True,
                             project='test',
                             project_directory=os.path.join(self.project_dir, dir_name),
                             )

    def test_family_map(self):
        """Test that the KinBot family map supports the expected RMG families."""
        self.assertEqual(self.rxn_1.family, 'intra_H_migration')
        adapter = self.get_adapter()
        self.assertIn('intra_H_migration', adapter.supported_families)
        self.assertEqual(adapter.family_map['intra_H_migration'],
                         ['intra_H_migration', 'intra_H_migration_suprafacial'])
        self.assertEqual(adapter.family_map['R_Addition_COm'], ['R_Addition_COm3_R'])

    def test_missing_kinbot_python(self):
        """Test that execute_incore() raises if the kinbot_env python executable is missing."""
        adapter = self.get_adapter()
        with mock.patch.object(kinbot_ts, 'KINBOT_PYTHON', None):
            with self.assertRaises(FileNotFoundError):
                adapter.execute_incore()

    def test_intra_h_migration(self):
        """Test KinBot for intra H migration reactions with a mocked subprocess boundary."""
        self.assertEqual(self.rxn_1.family, 'intra_H_migration')
        adapter = self.get_adapter()
        captured = dict()

        def fake_run_in_conda_env(python_executable, script_path, *script_args, check=False):
            """Mimic the KinBot worker script: read the input file, write an output file."""
            captured['python_executable'] = python_executable
            captured['script_path'] = script_path
            captured['script_args'] = script_args
            self.assertEqual(script_args[0], '--yml_in_path')
            input_dict = read_yaml_file(path=script_args[1])
            captured['input_dict'] = input_dict
            results = list()
            for well in input_dict['wells']:
                # Return the (valid, non-colliding) well geometry itself as the "TS guess".
                results.append({'direction': well['direction'],
                                'success': True,
                                'coords': kinbot_list_to_coords(well['structure']),
                                'execution_time': '0:00:00.104819',
                                })
            save_yaml_file(path=input_dict['yml_out_path'], content=results)
            return subprocess.CompletedProcess(args=[], returncode=0, stdout='', stderr='')

        with mock.patch.object(kinbot_ts, 'KINBOT_PYTHON', kinbot_ts.__file__), \
                mock.patch.object(kinbot_ts, 'run_in_conda_env', side_effect=fake_run_in_conda_env) as run_mock:
            adapter.execute_incore()

        # Assert the boundary was crossed correctly.
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(captured['python_executable'], kinbot_ts.__file__)
        self.assertEqual(captured['script_path'], kinbot_ts.KINBOT_SCRIPT_PATH)
        self.assertTrue(captured['script_path'].endswith(os.path.join('scripts', 'kinbot_script.py')))

        # Assert the input file contents.
        input_dict = captured['input_dict']
        self.assertEqual(input_dict['charge'], 0)
        self.assertEqual(input_dict['multiplicity'], 2)
        self.assertEqual(input_dict['families'], ['intra_H_migration', 'intra_H_migration_suprafacial'])
        self.assertEqual(input_dict['step'], 20)
        self.assertEqual(input_dict['yml_out_path'], adapter.yml_out_path)
        directions = [well['direction'] for well in input_dict['wells']]
        self.assertIn('F', directions)
        self.assertIn('R', directions)
        for well in input_dict['wells']:
            self.assertIsInstance(well['smiles'], str)
            self.assertEqual(len(well['structure']), 4 * 8)  # 8 atoms, 4 entries per atom.

        # Assert the output parsing.
        self.assertTrue(self.rxn_1.ts_species.is_ts)
        self.assertEqual(self.rxn_1.ts_species.charge, 0)
        self.assertEqual(self.rxn_1.ts_species.multiplicity, 2)
        self.assertEqual(len(self.rxn_1.ts_species.ts_guesses), 2)
        self.assertEqual(self.rxn_1.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(self.rxn_1.ts_species.ts_guesses[1].initial_xyz['symbols'],
                         ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(len(self.rxn_1.ts_species.ts_guesses[1].initial_xyz['coords']), 8)
        self.assertEqual(self.rxn_1.ts_species.ts_guesses[0].method, 'kinbot')
        self.assertEqual(self.rxn_1.ts_species.ts_guesses[1].method, 'kinbot')
        self.assertEqual(self.rxn_1.ts_species.ts_guesses[0].method_index, 0)
        self.assertEqual(self.rxn_1.ts_species.ts_guesses[1].method_index, 1)
        self.assertEqual(self.rxn_1.ts_species.ts_guesses[0].method_direction, 'F')
        self.assertEqual(self.rxn_1.ts_species.ts_guesses[1].method_direction, 'R')
        self.assertTrue(self.rxn_1.ts_species.ts_guesses[0].success)
        self.assertTrue(self.rxn_1.ts_species.ts_guesses[1].success)

    def test_subprocess_failure(self):
        """Test that a failed KinBot subprocess results in no TS guesses without raising."""
        adapter = self.get_adapter(dir_name='tst2')

        def fake_failing_run(python_executable, script_path, *script_args, check=False):
            # Simulate a crashed worker: non-zero return code, no output file written.
            return subprocess.CompletedProcess(args=[], returncode=1, stdout='', stderr='kinbot crashed')

        with mock.patch.object(kinbot_ts, 'KINBOT_PYTHON', kinbot_ts.__file__), \
                mock.patch.object(kinbot_ts, 'run_in_conda_env', side_effect=fake_failing_run):
            adapter.execute_incore()
        self.assertEqual(len(self.rxn_1.ts_species.ts_guesses), 0)

    def test_parse_kinbot_results(self):
        """Test parsing worker results: duplicates are merged, failures are kept as unsuccessful guesses."""
        adapter = self.get_adapter(dir_name='tst3')
        self.rxn_1.ts_species = ARCSpecies(label='TS', is_ts=True, charge=0, multiplicity=2)
        species_to_explore = {'F': self.rxn_1.r_species[0], 'R': self.rxn_1.p_species[0]}
        coords_f = [list(coord) for coord in self.rxn_1.r_species[0].get_xyz()['coords']]
        results = [{'direction': 'F', 'success': True, 'coords': coords_f, 'execution_time': '0:00:00.05'},
                   {'direction': 'F', 'success': True, 'coords': coords_f, 'execution_time': '0:00:00.05'},  # duplicate
                   {'direction': 'R', 'success': False, 'coords': None, 'execution_time': '0:00:00.01',
                    'error': 'ValueError: kinbot could not modify the geometry'},
                   {'direction': 'bogus', 'success': True, 'coords': coords_f, 'execution_time': '0:00:00.01'},
                   ]
        adapter.parse_kinbot_results(rxn=self.rxn_1, results=results, species_to_explore=species_to_explore)
        ts_guesses = self.rxn_1.ts_species.ts_guesses
        self.assertEqual(len(ts_guesses), 2)  # The duplicate and the bogus direction were not appended.
        self.assertTrue(ts_guesses[0].success)
        self.assertEqual(ts_guesses[0].method, 'kinbot')
        self.assertEqual(ts_guesses[0].method_direction, 'F')
        self.assertEqual(ts_guesses[0].method_index, 0)
        self.assertFalse(ts_guesses[1].success)
        self.assertEqual(ts_guesses[1].method_direction, 'R')
        self.assertEqual(ts_guesses[1].method_index, 1)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
