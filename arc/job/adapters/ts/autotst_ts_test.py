#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.autotst_ts module
"""

import os
import shutil
import subprocess
import unittest
from unittest import mock

from arc.common import ARC_TESTING_PATH, get_logger
import arc.job.adapters.common as common
import arc.job.adapters.ts.autotst_ts as autotst_ts
from arc.job.adapters.ts.autotst_ts import AutoTSTAdapter
from arc.reaction import ARCReaction
from arc.species import ARCSpecies


logger = get_logger()

TRACEBACK = """Traceback (most recent call last):
  File "autotst_script.py", line 1, in <module>
    raise ValueError('a very long AutoTST traceback')
ValueError: a very long AutoTST traceback"""


class TestAutoTSTAdapter(unittest.TestCase):
    """
    Contains unit tests for the AutoTSTAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.project_dir = os.path.join(ARC_TESTING_PATH, 'test_AutoTST')

    def setUp(self):
        """
        A method that is run before each unit test in this class.
        """
        self.rxn_1 = ARCReaction(reactants=['CC[O]'], products=['[CH2]CO'],
                                 r_species=[ARCSpecies(label='CC[O]', smiles='CC[O]')],
                                 p_species=[ARCSpecies(label='[CH2]CO', smiles='[CH2]CO')])

    def _remove_test_dir(self, path: str):
        """A helper function to remove a single test's project directory (and the shared
        parent directory if it is empty). Tests may run in parallel (pytest-xdist), so
        each test must only ever remove its own subdirectory."""
        shutil.rmtree(path, ignore_errors=True)
        try:
            os.rmdir(self.project_dir)
        except OSError:
            pass

    def get_adapter(self, dir_name: str) -> AutoTSTAdapter:
        """A helper function to instantiate an AutoTSTAdapter instance."""
        project_directory = os.path.join(self.project_dir, dir_name)
        self.addCleanup(self._remove_test_dir, project_directory)
        return AutoTSTAdapter(job_type='tsg',
                              reactions=[self.rxn_1],
                              testing=True,
                              project='test',
                              project_directory=project_directory,
                              )

    def test_supported_families(self):
        """Test that the AutoTST adapter advertises the expected RMG families (gate 1)."""
        adapter = self.get_adapter(dir_name='tst_supported_families')
        for family in ['intra_H_migration', 'H_Abstraction', 'R_Addition_MultipleBond', 'Disproportionation']:
            self.assertIn(family, adapter.supported_families)

    def test_disproportionation_passes_both_gates(self):
        """Test that a Disproportionation reaction is admitted through BOTH TS-adapter gates for autotst.

        Gate 1: the adapter's own ``supported_families`` includes 'Disproportionation'.
        Gate 2: ``ts_adapters_by_rmg_family`` maps 'Disproportionation' to a list that includes 'autotst'.
        """
        # Gate 2: the RMG-family -> adapters registry (checked directly, no RMG classification needed).
        self.assertIn('Disproportionation', common.ts_adapters_by_rmg_family)
        self.assertIn('autotst', common.ts_adapters_by_rmg_family['Disproportionation'])

        # Gate 1: the adapter advertises Disproportionation as supported.
        adapter = self.get_adapter(dir_name='tst_disprop_gates')
        self.assertIn('Disproportionation', adapter.supported_families)

    def test_intra_h_migration_still_supported(self):
        """Test that enabling Disproportionation did not disturb the pre-existing intra_H_migration gate."""
        adapter = self.get_adapter(dir_name='tst_intra_h')
        self.assertIn('intra_H_migration', adapter.supported_families)



    def test_save_subprocess_error_log(self):
        """Test saving the subprocess stdout and stderr to a dedicated file"""
        adapter = self.get_adapter(dir_name='tst_save_error_log')
        output = subprocess.CompletedProcess(args=[], returncode=1, stdout='some stdout', stderr=TRACEBACK)
        err_path = adapter.save_subprocess_error_log(output=output, rxn=self.rxn_1, direction_str='forward')
        self.assertEqual(err_path, os.path.join(os.path.dirname(adapter.output_path), 'autotst_err.log'))
        self.assertTrue(os.path.isfile(err_path))
        with open(err_path, 'r') as f:
            content = f.read()
        self.assertIn('some stdout', content)
        self.assertIn('ValueError: a very long AutoTST traceback', content)
        self.assertIn('returned code 1', content)
        adapter.save_subprocess_error_log(output=output, rxn=self.rxn_1, direction_str='reverse')
        with open(err_path, 'r') as f:
            content = f.read()
        self.assertIn('forward', content)
        self.assertIn('reverse', content)

    def test_save_subprocess_error_log_does_not_raise(self):
        """Test that a failure to write the error log does not break the run"""
        adapter = self.get_adapter(dir_name='tst_save_error_log_fail')
        output = subprocess.CompletedProcess(args=[], returncode=1, stdout='', stderr='')
        with mock.patch('builtins.open', side_effect=OSError('read-only file system')):
            err_path = adapter.save_subprocess_error_log(output=output, rxn=self.rxn_1, direction_str='forward')
        self.assertIsNone(err_path)

    def test_failed_subprocess_logs_a_single_line(self):
        """Test that a failed AutoTST subprocess does not dump its traceback into the log"""
        self.assertEqual(self.rxn_1.family, 'intra_H_migration')
        adapter = self.get_adapter(dir_name='tst_single_line_warning')

        def fake_run_in_conda_env(python_executable, script_path, *script_args, check=False,
                                  strip_pythonpath=False):
            """Mimic a crashing AutoTST worker script."""
            return subprocess.CompletedProcess(args=[], returncode=1, stdout='', stderr=TRACEBACK)

        with mock.patch.object(autotst_ts, 'AUTOTST_PYTHON', autotst_ts.__file__), \
                mock.patch.object(autotst_ts, 'run_in_conda_env', side_effect=fake_run_in_conda_env):
            with self.assertLogs('arc', level='WARNING') as cm:
                adapter.execute_incore()

        warnings = [record.getMessage() for record in cm.records if record.levelname == 'WARNING'
                    and 'AutoTST subprocess' in record.getMessage()]
        self.assertEqual(len(warnings), 2)
        err_path = os.path.join(os.path.dirname(adapter.output_path), 'autotst_err.log')
        for message, direction_str in zip(warnings, ['forward', 'reverse']):
            self.assertEqual(len(message.splitlines()), 1)
            self.assertNotIn('Traceback', message)
            self.assertIn(direction_str, message)
            self.assertIn('returned code 1', message)
            self.assertIn(err_path, message)
        self.assertTrue(os.path.isfile(err_path))
        with open(err_path, 'r') as f:
            content = f.read()
        self.assertIn('ValueError: a very long AutoTST traceback', content)

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
