#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for the AutoTST TS-search adapter, with the tst_env subprocess boundary mocked.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock

import arc.job.adapters.ts.autotst_ts as autotst_ts
from arc.reaction import ARCReaction
from arc.species import ARCSpecies


class TestAutoTSTAdapter(unittest.TestCase):
    """Contains unit tests for the AutoTSTAdapter class."""

    def setUp(self):
        self.project_directory = tempfile.mkdtemp(prefix='test_AutoTST_')
        self.addCleanup(shutil.rmtree, self.project_directory, ignore_errors=True)
        self.rxn = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='OH', smiles='[OH]')],
                               p_species=[ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2O', smiles='O')])

    def test_autotst_runs_without_the_callers_pythonpath(self):
        """tst_env's own rmgpy must not be shadowed by a source checkout on ARC's PYTHONPATH."""
        self.assertEqual(self.rxn.family, 'H_Abstraction')
        adapter = autotst_ts.AutoTSTAdapter(job_type='tsg',
                                            reactions=[self.rxn],
                                            testing=True,
                                            project='test',
                                            project_directory=self.project_directory,
                                            )
        calls = list()

        def fake_run_in_conda_env(python_executable, script_path, *script_args, check=False,
                                  strip_pythonpath=False):
            """Record the call and report an AutoTST run that found nothing."""
            calls.append({'script_path': script_path, 'script_args': script_args,
                          'strip_pythonpath': strip_pythonpath})
            return subprocess.CompletedProcess(args=[], returncode=0, stdout='', stderr='')

        with unittest.mock.patch.object(autotst_ts, 'AUTOTST_PYTHON', autotst_ts.__file__), \
                unittest.mock.patch.object(autotst_ts, 'run_in_conda_env', side_effect=fake_run_in_conda_env):
            adapter.execute_incore()

        self.assertEqual(len(calls), 2)  # forward and reverse
        for call in calls:
            self.assertTrue(call['script_path'].endswith(os.path.join('scripts', 'autotst_script.py')))
            self.assertTrue(call['strip_pythonpath'])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
