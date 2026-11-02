#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.gcn module
"""

import importlib.util
import os
import shutil
import subprocess
import unittest
from unittest.mock import patch

from arc.common import ARC_PATH, ARC_TESTING_PATH, read_yaml_file
import arc.job.adapters.ts.gcn_ts as ts_gcn
from arc.job.adapters.ts.gcn_ts import GCNAdapter
from arc.reaction import ARCReaction
from arc.species.converter import str_to_xyz
from arc.species.species import ARCSpecies, TSGuess

GCN_SCRIPT_PATH = os.path.join(ARC_PATH, 'arc', 'job', 'adapters', 'scripts', 'gcn_script.py')

TS_XYZ_F = """3

O    0.00000000    0.00000000    0.00000000
H    0.00000000    0.00000000    0.97000000
H    0.94000000    0.00000000   -0.24000000
"""

TS_XYZ_R = """3

O    0.00000000    0.00000000    0.10000000
H    0.00000000    0.10000000    1.10000000
H    1.05000000    0.00000000   -0.35000000
"""


def load_gcn_script():
    """Load the standalone gcn_script.py (not a package module) for testing its plumbing."""
    spec = importlib.util.spec_from_file_location('gcn_script', GCN_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_run_in_conda_env_factory(returncode: int = 0, write_ts_file: bool = True):
    """
    Return a fake run_in_conda_env() side effect that mimics gcn_script.py:
    it writes the TS xyz file passed via --ts_xyz_path (direction-dependent content)
    and returns a CompletedProcess with the requested return code.
    """
    def fake_run_in_conda_env(python_executable, script_path, *script_args):
        args = list(script_args)
        ts_xyz_path = args[args.index('--ts_xyz_path') + 1]
        if write_ts_file:
            with open(ts_xyz_path, 'w') as f:
                f.write(TS_XYZ_F if 'fwd' in os.path.basename(ts_xyz_path).lower() else TS_XYZ_R)
        return subprocess.CompletedProcess(args=[python_executable, script_path] + args,
                                           returncode=returncode, stdout='', stderr='')
    return fake_run_in_conda_env


class TestGCNAdapter(unittest.TestCase):
    """
    Contains unit tests for the GCNAdapter class.
    """

    def setUp(self):
        """
        A method that is run before each unit test in this class.
        Tests run in parallel (pytest-xdist), so each test gets its own directory.
        """
        self.maxDiff = None
        self.output_dir = os.path.join(ARC_TESTING_PATH, f'GCN_{self._testMethodName}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.output_dir, ignore_errors=True)
        self.reactant_path = os.path.join(self.output_dir, 'react.sdf')
        self.product_path = os.path.join(self.output_dir, 'prod.sdf')
        self.ts_fwd_path = os.path.join(self.output_dir, 'TS_fwd.xyz')

    @staticmethod
    def get_reaction() -> ARCReaction:
        """Get a fresh isomerization reaction instance."""
        return ARCReaction(r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
                           p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')])

    def get_adapter(self, rxn: ARCReaction) -> GCNAdapter:
        """Get a GCNAdapter instance for testing."""
        project_dir = os.path.join(self.output_dir, 'project')
        return GCNAdapter(job_type='tsg',
                          reactions=[rxn],
                          testing=True,
                          project='test_GCNAdapter',
                          project_directory=project_dir,
                          dihedral_increment=1,
                          )

    def test_gcn_available(self):
        """Test the gcn_available() function."""
        self.assertTrue(os.path.isfile(ts_gcn.GCN_SCRIPT_PATH))
        with patch.object(ts_gcn, 'TS_GCN_PYTHON', __file__):
            # Any existing file stands in for the interpreter.
            self.assertTrue(ts_gcn.gcn_available())
        with patch.object(ts_gcn, 'TS_GCN_PYTHON', None):
            self.assertFalse(ts_gcn.gcn_available())
        with patch.object(ts_gcn, 'TS_GCN_PYTHON', '/path/does/not/exist/python'):
            self.assertFalse(ts_gcn.gcn_available())

    def test_write_sdf_files(self):
        """Test the write_sdf_files() function."""
        rxn = self.get_reaction()
        ts_gcn.write_sdf_files(rxn=rxn,
                               reactant_path=self.reactant_path,
                               product_path=self.product_path,
                               )
        self.assertTrue(os.path.isfile(self.reactant_path))
        self.assertTrue(os.path.isfile(self.product_path))
        with open(self.reactant_path, 'r') as f:
            content_r_sdf = f.read()
        # 10 atoms (C3H7), the atom count appears in the SDF counts line.
        self.assertIn(' 10  9', content_r_sdf)

    def test_run_subprocess_locally_success(self):
        """Test run_subprocess_locally() passing the correct arguments and parsing the output."""
        ts_gcn.write_sdf_files(rxn=self.get_reaction(),
                               reactant_path=self.reactant_path,
                               product_path=self.product_path,
                               )
        ts_species = ARCSpecies(label='TS', is_ts=True)
        with patch.object(ts_gcn, 'TS_GCN_PYTHON', '/fake/envs/ts_gcn/bin/python'), \
                patch.object(ts_gcn, 'run_in_conda_env',
                             side_effect=fake_run_in_conda_env_factory()) as mock_run:
            ts_gcn.run_subprocess_locally(direction='F',
                                          reactant_path=self.reactant_path,
                                          product_path=self.product_path,
                                          ts_path=self.ts_fwd_path,
                                          local_path=self.output_dir,
                                          ts_species=ts_species,
                                          )
        self.assertEqual(mock_run.call_count, 1)
        args = mock_run.call_args.args
        self.assertEqual(args[0], '/fake/envs/ts_gcn/bin/python')
        self.assertEqual(args[1], ts_gcn.GCN_SCRIPT_PATH)
        # The reactant SDF must be passed to --r_sdf_path and the product SDF to --p_sdf_path.
        flags = dict(zip(args[2::2], args[3::2]))
        self.assertEqual(flags['--r_sdf_path'], self.reactant_path)
        self.assertEqual(flags['--p_sdf_path'], self.product_path)
        self.assertEqual(flags['--ts_xyz_path'], self.ts_fwd_path)
        self.assertEqual(len(ts_species.ts_guesses), 1)
        tsg = ts_species.ts_guesses[0]
        self.assertTrue(tsg.success)
        self.assertEqual(tsg.method, 'gcn')
        self.assertEqual(tsg.method_direction, 'F')
        self.assertEqual(tsg.initial_xyz['symbols'], ('O', 'H', 'H'))

    def test_run_subprocess_locally_failure(self):
        """Test that a failing GCN subprocess is handled gracefully."""
        ts_species = ARCSpecies(label='TS', is_ts=True)
        with patch.object(ts_gcn, 'TS_GCN_PYTHON', '/fake/envs/ts_gcn/bin/python'), \
                patch.object(ts_gcn, 'run_in_conda_env',
                             side_effect=fake_run_in_conda_env_factory(returncode=1, write_ts_file=False)) as mock_run:
            ts_gcn.run_subprocess_locally(direction='R',
                                          reactant_path=self.product_path,
                                          product_path=self.reactant_path,
                                          ts_path=os.path.join(self.output_dir, 'TS_rev.xyz'),
                                          local_path=self.output_dir,
                                          ts_species=ts_species,
                                          )
        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(len(ts_species.ts_guesses), 0)

    def test_execute_incore(self):
        """Test the full incore execution path through the mocked subprocess boundary."""
        rxn = self.get_reaction()
        adapter = self.get_adapter(rxn)
        with patch.object(ts_gcn, 'TS_GCN_PYTHON', '/fake/envs/ts_gcn/bin/python'), \
                patch.object(ts_gcn, 'gcn_available', return_value=True), \
                patch.object(ts_gcn, 'run_in_conda_env',
                             side_effect=fake_run_in_conda_env_factory()) as mock_run:
            adapter.execute_incore()
        # dihedral_increment=1 -> one repetition -> one forward + one reverse call.
        self.assertEqual(mock_run.call_count, 2)
        # The input SDF files must have been written before calling the subprocess.
        self.assertTrue(os.path.isfile(adapter.reactant_path))
        self.assertTrue(os.path.isfile(adapter.product_path))
        fwd_flags = dict(zip(mock_run.call_args_list[0].args[2::2], mock_run.call_args_list[0].args[3::2]))
        rev_flags = dict(zip(mock_run.call_args_list[1].args[2::2], mock_run.call_args_list[1].args[3::2]))
        self.assertEqual(fwd_flags['--r_sdf_path'], adapter.reactant_path)
        self.assertEqual(fwd_flags['--p_sdf_path'], adapter.product_path)
        self.assertEqual(fwd_flags['--ts_xyz_path'], adapter.ts_fwd_path)
        # The reverse direction swaps reactant and product.
        self.assertEqual(rev_flags['--r_sdf_path'], adapter.product_path)
        self.assertEqual(rev_flags['--p_sdf_path'], adapter.reactant_path)
        self.assertEqual(rev_flags['--ts_xyz_path'], adapter.ts_rev_path)
        self.assertEqual(len(rxn.ts_species.ts_guesses), 2)
        self.assertEqual(rxn.ts_species.ts_guesses[0].method_direction, 'F')
        self.assertEqual(rxn.ts_species.ts_guesses[1].method_direction, 'R')
        self.assertTrue(all(tsg.success for tsg in rxn.ts_species.ts_guesses))

    def test_execute_incore_gcn_unavailable(self):
        """Test that execution degrades gracefully (no crash, no subprocess) when the ts_gcn env is missing."""
        rxn = self.get_reaction()
        adapter = self.get_adapter(rxn)
        with patch.object(ts_gcn, 'TS_GCN_PYTHON', None), \
                patch.object(ts_gcn, 'run_in_conda_env') as mock_run:
            adapter.execute_incore()
        mock_run.assert_not_called()
        self.assertTrue(rxn.ts_species is None or len(rxn.ts_species.ts_guesses) == 0)

    def test_process_tsg(self):
        """Test the process_tsg() function."""
        expected_ts_xyz_path = os.path.join(self.output_dir, 'GCN R 0.xyz')
        ts_species = ARCSpecies(label='TS', is_ts=True)
        ts_gcn.process_tsg(direction='R',
                           ts_xyz=str_to_xyz(os.path.join(ARC_TESTING_PATH, 'opt', 'TS_nC3H7-iC3H7.out')),
                           local_path=self.output_dir,
                           ts_species=ts_species,
                           tsg=TSGuess(method='GCN',
                                       method_direction='R',
                                       index=0,
                                       ),
                           )
        self.assertTrue(os.path.isfile(expected_ts_xyz_path))
        xyz = str_to_xyz(expected_ts_xyz_path)
        self.assertEqual(xyz['symbols'], ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(len(ts_species.ts_guesses), 1)


class TestGCNScript(unittest.TestCase):
    """
    Contains unit tests for the standalone gcn_script.py
    (its plumbing only; the actual inference import requires the ts_gcn env).
    """

    def setUp(self):
        """
        A method that is run before each unit test in this class.
        """
        self.maxDiff = None
        self.output_dir = os.path.join(ARC_TESTING_PATH, f'GCN_script_{self._testMethodName}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.output_dir, ignore_errors=True)
        self.gcn_script = load_gcn_script()

    def test_no_arc_imports(self):
        """gcn_script.py runs in the ts_gcn env and must not import any arc modules."""
        with open(GCN_SCRIPT_PATH, 'r') as f:
            content = f.read()
        self.assertNotIn('import arc', content)
        self.assertNotIn('from arc', content)

    def test_parse_command_line_arguments(self):
        """Test both CLI modes of gcn_script.py."""
        args = self.gcn_script.parse_command_line_arguments(['--yml_in_path', 'input.yml'])
        self.assertEqual(args.yml_in_path, 'input.yml')
        self.assertIsNone(args.r_sdf_path)
        args = self.gcn_script.parse_command_line_arguments(
            ['--r_sdf_path', 'r.sdf', '--p_sdf_path', 'p.sdf', '--ts_xyz_path', 'ts.xyz'])
        self.assertIsNone(args.yml_in_path)
        self.assertEqual(args.r_sdf_path, 'r.sdf')
        self.assertEqual(args.p_sdf_path, 'p.sdf')
        self.assertEqual(args.ts_xyz_path, 'ts.xyz')

    def test_initialize_gcn_run(self):
        """Test the batch (queue) mode: input dict in, TS guess list YAML out."""
        yml_out_path = os.path.join(self.output_dir, 'output.yml')
        input_dict = {'reactant_path': os.path.join(self.output_dir, 'reactant.sdf'),
                      'product_path': os.path.join(self.output_dir, 'product.sdf'),
                      'local_path': self.output_dir,
                      'yml_out_path': yml_out_path,
                      'repetitions': 2,
                      }

        def fake_run_gcn(r_sdf_path, p_sdf_path, ts_xyz_path):
            with open(ts_xyz_path, 'w') as f:
                f.write(TS_XYZ_F)
            return True

        original_run_gcn = self.gcn_script.run_gcn
        self.gcn_script.run_gcn = fake_run_gcn
        self.addCleanup(setattr, self.gcn_script, 'run_gcn', original_run_gcn)
        self.gcn_script.initialize_gcn_run(input_dict=input_dict)
        self.assertTrue(os.path.isfile(yml_out_path))
        tsgs = read_yaml_file(yml_out_path)
        self.assertEqual(len(tsgs), 4)  # 2 repetitions x 2 directions
        self.assertEqual([tsg['method_direction'] for tsg in tsgs], ['F', 'R', 'F', 'R'])
        for tsg in tsgs:
            self.assertEqual(tsg['method'], 'GCN')
            self.assertTrue(tsg['success'])
            # The two xyz header lines must be stripped.
            self.assertEqual(tsg['initial_xyz'].split()[0], 'O')

    def test_import_inference_raises_informatively(self):
        """Without the ts_gcn env, import_inference() must raise an informative ImportError."""
        with self.assertRaises(ImportError) as cm:
            self.gcn_script.import_inference()
        self.assertIn('TS-GCN', str(cm.exception))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
