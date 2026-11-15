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

from arc.common import NUMBER_BY_SYMBOL, save_yaml_file
import arc.job.adapters.ts.autotst_ts as autotst_ts
from arc.reaction import ARCReaction
from arc.species import ARCSpecies
from arc.species.converter import str_to_xyz
from arc.species.species import TSGuess


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


    def test_autotst_duplicate_credits_autotst_in_method_sources(self):
        """An AutoTST result that matches an existing guess adds 'autotst' to that guess's method_sources."""
        xyz = str_to_xyz("""O       1.29503100   -0.10783400    0.00000100
H       1.38087500    0.85868300   -0.00000100
C      -1.20395400    0.00755100    0.00000100
H       0.00766600   -0.17858500   -0.00002000
H      -1.38410100    1.08884500    0.00008300
H      -1.57048400   -0.47571900   -0.91306300
H      -1.57047900   -0.47585800    0.91299300""")
        self.rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        self.rxn.ts_species.ts_guesses.append(TSGuess(method='GoFlow', index=0, success=True, xyz=xyz))
        adapter = autotst_ts.AutoTSTAdapter(job_type='tsg',
                                            reactions=[self.rxn],
                                            testing=True,
                                            project='test',
                                            project_directory=self.project_directory,
                                            )

        def fake_run_in_conda_env(python_executable, script_path, *script_args, check=False,
                                  strip_pythonpath=False):
            """Write one AutoTST result equal to the existing guess."""
            save_yaml_file(path=script_args[1],
                           content=[{'coords': [list(coord) for coord in xyz['coords']],
                                     'numbers': [NUMBER_BY_SYMBOL[symbol] for symbol in xyz['symbols']]}])
            return subprocess.CompletedProcess(args=[], returncode=0, stdout='', stderr='')

        with unittest.mock.patch.object(autotst_ts, 'AUTOTST_PYTHON', autotst_ts.__file__), \
                unittest.mock.patch.object(autotst_ts, 'run_in_conda_env', side_effect=fake_run_in_conda_env):
            adapter.execute_incore()

        self.assertEqual(len(self.rxn.ts_species.ts_guesses), 1)
        self.assertEqual(self.rxn.ts_species.ts_guesses[0].method_sources, ['goflow', 'autotst'])

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
