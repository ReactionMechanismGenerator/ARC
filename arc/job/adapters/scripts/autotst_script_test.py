#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for ``arc.job.adapters.scripts.autotst_script``.

The script runs inside ``tst_env``, where AutoTST is importable. These tests run it in ARC's own
interpreter against a stub ``autotst`` package, so they exercise the script's exit contract
without AutoTST installed.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from arc.common import ARC_PATH

SCRIPT_PATH = os.path.join(ARC_PATH, 'arc', 'job', 'adapters', 'scripts', 'autotst_script.py')

STUB_REACTION_MODULE = '''
class Reaction(object):
    """A stand-in for autotst.reaction.Reaction that fails the way a broken AutoTST setup does."""
    def __init__(self, label):
        self.label = label

    def get_labeled_reaction(self):
        raise RuntimeError('stub AutoTST could not label the reaction')
'''


class TestAutoTSTScriptFailure(unittest.TestCase):
    """A failure inside AutoTST must reach ARC as a non-zero exit with a traceback."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='test_autotst_script_')
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        stub_dir = os.path.join(self.tmp_dir, 'stub', 'autotst')
        os.makedirs(stub_dir)
        with open(os.path.join(stub_dir, '__init__.py'), 'w') as f:
            f.write('')
        with open(os.path.join(stub_dir, 'reaction.py'), 'w') as f:
            f.write(STUB_REACTION_MODULE)
        self.output_path = os.path.join(self.tmp_dir, 'out', 'ts_results.yml')

    def run_script(self) -> subprocess.CompletedProcess:
        """Run the script against the stub AutoTST package."""
        env = dict(os.environ, PYTHONPATH=os.path.join(self.tmp_dir, 'stub'))
        return subprocess.run([sys.executable, SCRIPT_PATH, 'C+[OH]_[CH3]+O', self.output_path],
                              capture_output=True, text=True, env=env, timeout=120)

    def test_autotst_failure_exits_non_zero_with_a_traceback(self):
        """A bare `return None` exits 0 and reads as "AutoTST found no TS" instead of an error."""
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Traceback', result.stderr)
        self.assertIn('stub AutoTST could not label the reaction', result.stderr)
        self.assertFalse(os.path.isfile(self.output_path))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
