#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.kinbot_ts module
"""

import math
import os
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock as mock

import numpy as np

from arc.common import ARC_TESTING_PATH, get_single_bond_length, read_yaml_file, save_yaml_file
from arc.exceptions import DependencyError
import arc.job.adapters.ts.kinbot_ts as kinbot_ts
from arc.reaction import ARCReaction
from arc.species import ARCSpecies
from arc.species.converter import order_xyz_by_atom_map, xyz_from_data, xyz_to_str


def kinbot_list_to_coords(structure: list) -> list:
    """A helper function to convert a flat KinBot structure list into an N x 3 coordinates list."""
    coords = list()
    for i in range(0, len(structure), 4):
        coords.append([float(structure[i + 1]), float(structure[i + 2]), float(structure[i + 3])])
    return coords


def species_from_fixture(entry: dict) -> ARCSpecies:
    """
    Build an ARCSpecies from a fixture entry.

    Args:
        entry (dict): A species entry with ``label``, ``smiles``, ``multiplicity``, ``symbols`` and ``coords``.

    Returns:
        ARCSpecies: The species.
    """
    xyz = {'symbols': tuple(entry['symbols']), 'isotopes': None, 'coords': tuple(tuple(c) for c in entry['coords'])}
    return ARCSpecies(label=entry['label'], smiles=entry['smiles'], multiplicity=entry['multiplicity'], xyz=xyz)


def stretched_reactant_bonds(rxn: ARCReaction, xyz: dict) -> list:
    """
    Get the reactant bonds that are neither broken in the reaction nor kept at a bonded length in ``xyz``.

    A bond is counted as stretched above 1.3 times its single-bond length, or 1.5 times if it involves an H atom.

    Args:
        rxn (ARCReaction): The reaction.
        xyz (dict): A TS guess, read in reactant atom order.

    Returns:
        list: The stretched reactant bonds.
    """
    r_symbols = rxn.get_reactants_xyz(return_format='dict')['symbols']
    r_bonds, _ = rxn.get_bonds(r_bonds_only=True)
    _, broken = rxn.get_formed_and_broken_bonds()
    broken = {tuple(sorted(bond)) for bond in broken}
    stretched = list()
    for i, j in r_bonds:
        if tuple(sorted((i, j))) in broken:
            continue
        limit = 1.5 if 'H' in (r_symbols[i], r_symbols[j]) else 1.3
        distance = float(np.linalg.norm(np.array(xyz['coords'][i]) - np.array(xyz['coords'][j])))
        if distance / get_single_bond_length(r_symbols[i], r_symbols[j]) > limit:
            stretched.append((i, j))
    return stretched


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
        cls.project_dir = tempfile.mkdtemp(prefix='arc_test_kinbot_')
        cls.addClassCleanup(shutil.rmtree, cls.project_dir, ignore_errors=True)

    def setUp(self):
        """
        A method that is run before each unit test in this class.
        """
        self.rxn_1 = ARCReaction(reactants=['CC[O]'], products=['[CH2]CO'],
                                 r_species=[ARCSpecies(label='CC[O]', smiles='CC[O]')],
                                 p_species=[ARCSpecies(label='[CH2]CO', smiles='[CH2]CO')])

    def get_adapter(self, dir_name: str) -> kinbot_ts.KinBotAdapter:
        """A helper function to instantiate a KinBotAdapter instance."""
        project_directory = os.path.join(self.project_dir, dir_name)
        return kinbot_ts.KinBotAdapter(job_type='tsg',
                             reactions=[self.rxn_1],
                             testing=True,
                             project='test',
                             project_directory=project_directory,
                             )

    def test_family_map(self):
        """Test that the KinBot family map supports the expected RMG families."""
        self.assertEqual(self.rxn_1.family, 'intra_H_migration')
        adapter = self.get_adapter(dir_name='tst_family_map')
        self.assertIn('intra_H_migration', adapter.supported_families)
        self.assertEqual(adapter.family_map['intra_H_migration'],
                         ['intra_H_migration', 'intra_H_migration_suprafacial'])
        self.assertEqual(adapter.family_map['R_Addition_COm'], ['R_Addition_COm3_R'])

    def test_missing_kinbot_python(self):
        """Test that execute_incore() raises if the kinbot_env python executable is missing."""
        adapter = self.get_adapter(dir_name='tst_missing_python')
        with mock.patch.object(kinbot_ts, 'KINBOT_PYTHON', None):
            with self.assertRaises(DependencyError):
                adapter.execute_incore()

    def test_intra_h_migration(self):
        """Test KinBot for intra H migration reactions with a mocked subprocess boundary."""
        self.assertEqual(self.rxn_1.family, 'intra_H_migration')
        adapter = self.get_adapter(dir_name='tst1')
        captured = dict()

        def fake_run_in_conda_env(python_executable, script_path, *script_args, check=False,
                                  strip_pythonpath=False):
            """Mimic the KinBot worker script: read the input file, write an output file."""
            captured['python_executable'] = python_executable
            captured['strip_pythonpath'] = strip_pythonpath
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
                                'uma_refined': False,
                                })
            save_yaml_file(path=input_dict['yml_out_path'], content=results)
            return subprocess.CompletedProcess(args=[], returncode=0, stdout='', stderr='')

        with mock.patch.object(kinbot_ts, 'KINBOT_PYTHON', kinbot_ts.__file__), \
                mock.patch.object(kinbot_ts, 'run_in_conda_env', side_effect=fake_run_in_conda_env) as run_mock:
            adapter.execute_incore()

        # Assert the boundary was crossed correctly.
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(captured['python_executable'], kinbot_ts.__file__)
        # PYTHONPATH must be stripped so a stale KinBot checkout can't shadow kinbot_env.
        self.assertTrue(captured['strip_pythonpath'])
        self.assertEqual(captured['script_path'], kinbot_ts.KINBOT_SCRIPT_PATH)
        self.assertTrue(captured['script_path'].endswith(os.path.join('scripts', 'kinbot_script.py')))

        # Assert the input file contents.
        input_dict = captured['input_dict']
        self.assertEqual(input_dict['charge'], 0)
        self.assertEqual(input_dict['multiplicity'], 2)
        self.assertEqual(input_dict['families'], ['intra_H_migration', 'intra_H_migration_suprafacial'])
        self.assertIn('uma', input_dict)
        self.assertFalse(input_dict['uma']['refine'])  # UMA refinement is opt-in, off by default.
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

        def fake_failing_run(python_executable, script_path, *script_args, check=False,
                             strip_pythonpath=False):
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

    def ts13_reaction(self) -> ARCReaction:
        """
        Build the pentyl1 <=> pentyl2 reaction of the fixture, with the atom map of the original run.

        Returns:
            ARCReaction: The reaction, with an empty TS species.
        """
        entry = read_yaml_file(os.path.join(ARC_TESTING_PATH, 'reactions', 'ts_guess_reactant_order.yml'))['ts13']
        rxn = ARCReaction(r_species=[species_from_fixture(spc) for spc in entry['reactants']],
                          p_species=[species_from_fixture(spc) for spc in entry['products']])
        rxn.family = entry['family']
        rxn.atom_map = list(entry['atom_map'])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, charge=0, multiplicity=2)
        self.ts13_guess = entry['kinbot_r_guess']
        return rxn

    def test_reverse_guess_remapped_to_reactant_order(self):
        """
        Test that a KinBot guess built from the product ('R') is stored in reactant atom order.

        Here the product and the reactant list their elements in the same sequence, so the element sequence alone
        cannot tell the two orders apart; the guess must be the atom-map permutation of KinBot's output and must
        keep every non-reacting reactant bond.
        """
        rxn = self.ts13_reaction()
        adapter = self.get_adapter(dir_name='tst_reverse_order')
        product_order = xyz_from_data(coords=self.ts13_guess['coords'], symbols=self.ts13_guess['symbols'])
        self.assertNotEqual(stretched_reactant_bonds(rxn, product_order), [])
        results = [{'direction': 'R', 'success': True, 'coords': self.ts13_guess['coords'], 'execution_time': '0:00:01'}]
        adapter.parse_kinbot_results(rxn=rxn, results=results, species_to_explore={'R': rxn.p_species[0]})
        tsg = rxn.ts_species.ts_guesses[0]
        self.assertTrue(tsg.success)
        self.assertEqual(xyz_to_str(tsg.initial_xyz),
                         xyz_to_str(order_xyz_by_atom_map(xyz=product_order, atom_map=rxn.atom_map)))
        self.assertEqual(stretched_reactant_bonds(rxn, tsg.initial_xyz), [])

    def test_reverse_guess_without_atom_map_is_unsuccessful(self):
        """Test that a KinBot 'R' guess is kept as unsuccessful, with a warning, when the reaction has no atom map."""
        rxn = self.ts13_reaction()
        adapter = self.get_adapter(dir_name='tst_reverse_no_map')
        results = [{'direction': 'R', 'success': True, 'coords': self.ts13_guess['coords'], 'execution_time': '0:00:01'}]
        with mock.patch.object(ARCReaction, 'atom_map', new_callable=mock.PropertyMock, return_value=None), \
                self.assertLogs(logger='arc', level='WARNING') as cm:
            adapter.parse_kinbot_results(rxn=rxn, results=results, species_to_explore={'R': rxn.p_species[0]})
        self.assertEqual(len(rxn.ts_species.ts_guesses), 1)
        self.assertFalse(rxn.ts_species.ts_guesses[0].success)
        self.assertIsNone(rxn.ts_species.ts_guesses[0].initial_xyz)
        self.assertTrue(any('atom map' in line for line in cm.output))

    def test_reverse_guess_with_failing_atom_map_is_unsuccessful(self):
        """
        Test that a KinBot 'R' guess is kept as unsuccessful, with a warning naming the reaction, when computing
        the atom map raises, and that the next guess of the same result set is still processed.
        """
        rxn = self.ts13_reaction()
        rxn.label = 'map_crash_rxn'
        adapter = self.get_adapter(dir_name='tst_reverse_map_crash')
        reactant_xyz = rxn.r_species[0].get_xyz()
        results = [{'direction': 'R', 'success': True, 'coords': self.ts13_guess['coords'], 'execution_time': '0:00:01'},
                   {'direction': 'F', 'success': True, 'coords': [list(c) for c in reactant_xyz['coords']],
                    'execution_time': '0:00:01'}]
        with mock.patch.object(ARCReaction, 'atom_map', new_callable=mock.PropertyMock,
                               side_effect=ValueError('map crash')), \
                self.assertLogs(logger='arc', level='WARNING') as cm:
            adapter.parse_kinbot_results(rxn=rxn, results=results,
                                         species_to_explore={'R': rxn.p_species[0], 'F': rxn.r_species[0]})
        ts_guesses = rxn.ts_species.ts_guesses
        self.assertEqual(len(ts_guesses), 2)
        self.assertEqual(ts_guesses[0].method_direction, 'R')
        self.assertFalse(ts_guesses[0].success)
        self.assertIsNone(ts_guesses[0].initial_xyz)
        self.assertEqual(ts_guesses[1].method_direction, 'F')
        self.assertTrue(ts_guesses[1].success)
        self.assertEqual(xyz_to_str(ts_guesses[1].initial_xyz), xyz_to_str(reactant_xyz))
        self.assertTrue(any('map_crash_rxn' in line and 'map crash' in line for line in cm.output))

    def test_reactant_order_guess_unchanged(self):
        """
        Test that a KinBot guess built from the reactant ('F') is stored exactly as KinBot returned it.

        The guess is already in reactant order. This guards a no-op and passes before the fix too; it fails if the
        reordering is applied regardless of the direction.
        """
        rxn = self.ts13_reaction()
        adapter = self.get_adapter(dir_name='tst_forward_order')
        reactant_xyz = rxn.r_species[0].get_xyz()
        results = [{'direction': 'F', 'success': True, 'coords': [list(c) for c in reactant_xyz['coords']],
                    'execution_time': '0:00:01'}]
        adapter.parse_kinbot_results(rxn=rxn, results=results, species_to_explore={'F': rxn.r_species[0]})
        tsg = rxn.ts_species.ts_guesses[0]
        self.assertTrue(tsg.success)
        self.assertEqual(xyz_to_str(tsg.initial_xyz), xyz_to_str(reactant_xyz))

    def test_uma_refinement_flag(self):
        """Test that the UMA settings cross the boundary and that refined guesses are marked in the method."""
        adapter = self.get_adapter(dir_name='tst4')
        captured = dict()

        def fake_run_in_conda_env(python_executable, script_path, *script_args, check=False,
                                  strip_pythonpath=False):
            input_dict = read_yaml_file(path=script_args[1])
            captured['input_dict'] = input_dict
            f_well = next(well for well in input_dict['wells'] if well['direction'] == 'F')
            r_well = next(well for well in input_dict['wells'] if well['direction'] == 'R')
            results = [{'direction': 'F', 'success': True,
                        'coords': kinbot_list_to_coords(f_well['structure']),
                        'execution_time': '0:00:01.5', 'uma_refined': True},
                       {'direction': 'R', 'success': True,
                        'coords': kinbot_list_to_coords(r_well['structure']),
                        'execution_time': '0:00:01.5', 'uma_refined': False,
                        'uma_error': 'The Sella saddle-point search on the UMA potential did not converge.'},
                       ]
            save_yaml_file(path=input_dict['yml_out_path'], content=results)
            return subprocess.CompletedProcess(args=[], returncode=0, stdout='', stderr='')

        uma_on = dict(kinbot_ts.KINBOT_UMA_SETTINGS)
        uma_on['refine'] = True
        with mock.patch.object(kinbot_ts, 'KINBOT_PYTHON', kinbot_ts.__file__), \
                mock.patch.object(kinbot_ts, 'KINBOT_UMA_SETTINGS', uma_on), \
                mock.patch.object(kinbot_ts, 'run_in_conda_env', side_effect=fake_run_in_conda_env):
            adapter.execute_incore()

        self.assertTrue(captured['input_dict']['uma']['refine'])
        ts_guesses = self.rxn_1.ts_species.ts_guesses
        self.assertEqual(len(ts_guesses), 2)
        # A UMA-refined guess is distinguishable, and both variants keep containing 'kinbot'.
        self.assertEqual(ts_guesses[0].method, 'kinbot-uma')
        self.assertEqual(ts_guesses[1].method, 'kinbot')
        self.assertTrue(all('kinbot' in tsg.method for tsg in ts_guesses))

    @unittest.skipIf(kinbot_ts.KINBOT_PYTHON is None or not os.path.isfile(kinbot_ts.KINBOT_PYTHON),
                     'A kinbot_env installation is required for this test')
    def test_template_modification_end_to_end(self):
        """
        Test, against a real kinbot_env, that the constraint templates are actually applied:
        the TS guess geometries must differ from the well geometries they started from.
        This exercises non-empty get_constraints() 'change' lists for the intra_H_migration
        family through the full per-family step sequence (a regression test for calling
        get_constraints() with a step beyond the family's max_step, which yields empty
        constraints and returns the unmodified well geometry).
        """
        adapter = self.get_adapter(dir_name='tst_e2e')
        adapter.execute_incore()
        ts_guesses = [tsg for tsg in self.rxn_1.ts_species.ts_guesses if tsg.success]
        self.assertGreaterEqual(len(ts_guesses), 2)
        well_xyzs = {'F': self.rxn_1.r_species[0].get_xyz(), 'R': self.rxn_1.p_species[0].get_xyz()}
        directions = set()
        for tsg in ts_guesses:
            directions.add(tsg.method_direction)
            well_coords = well_xyzs[tsg.method_direction]['coords']
            max_displacement = max(math.dist(guess_coord, well_coord)
                                   for guess_coord, well_coord
                                   in zip(tsg.initial_xyz['coords'], well_coords))
            self.assertGreater(max_displacement, 0.1)
        self.assertEqual(directions, {'F', 'R'})


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
