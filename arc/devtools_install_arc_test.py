#!/usr/bin/env python3
# encoding: utf-8

"""
Tests for the conda (de)activation hooks written by devtools/install_arc.sh.

The two hook heredocs are extracted verbatim from the installer, rendered into a temporary directory
the same way the installer writes them, and sourced in clean bash subshells.
"""

import os
import re
import shutil
import subprocess
import tempfile
import unittest

from arc.common import ARC_PATH


INSTALL_ARC = os.path.join(ARC_PATH, 'devtools', 'install_arc.sh')
BASE_PATH = '/usr/bin:/bin'
SEP = '@@@'
HOOK_BLOCK = re.compile(r'^cat >"\$(?:DE)?ACTIVATE_DIR/arc_env\.sh" <<\'?EOF\'?\n.*?^EOF\n', re.M | re.S)

REPORT = r"""
_report () {
    printf 'PATH=%s\n' "$PATH"
    if [ -n "${PYTHONPATH+x}" ]; then printf 'PYTHONPATH=%s\n' "$PYTHONPATH"; else printf 'PYTHONPATH<unset>\n'; fi
}
_names () {
    local PATH=/usr/bin:/bin
    compgen -v | grep -vx '_'; declare -F | sed 's/^declare -f //' | grep -vx '_report\|_names'
    true
}
"""


def render_hooks(target_dir: str, arc_repo: str) -> tuple:
    """
    Render the activation and deactivation hooks exactly as install_arc.sh writes them.

    Args:
        target_dir (str): A directory to write the hooks into.
        arc_repo (str): The ARC repository path to hard-code into the activation hook.

    Returns:
        tuple: The paths to the rendered activation and deactivation hook files.
    """
    with open(INSTALL_ARC) as f:
        blocks = HOOK_BLOCK.findall(f.read())
    if len(blocks) != 2:
        raise ValueError(f'Expected 2 hook heredocs in {INSTALL_ARC}, found {len(blocks)}')
    act_dir, deact_dir = os.path.join(target_dir, 'activate.d'), os.path.join(target_dir, 'deactivate.d')
    os.makedirs(act_dir)
    os.makedirs(deact_dir)
    env = {'PATH': BASE_PATH, 'ACTIVATE_DIR': act_dir, 'DEACTIVATE_DIR': deact_dir, 'ARC_REPO': arc_repo}
    subprocess.run(['bash', '-c', 'set -e\n' + ''.join(blocks)], env=env, check=True, timeout=60)
    return os.path.join(act_dir, 'arc_env.sh'), os.path.join(deact_dir, 'arc_env.sh')


def round_trip(act: str, deact: str, setup: str = '', activations: int = 1) -> list:
    """
    Source the hooks in a clean bash subshell and report the environment at each stage.

    Args:
        act (str): The path to the activation hook.
        deact (str): The path to the deactivation hook.
        setup (str): Shell code that sets the pre-activation environment.
        activations (int): How many times to source the activation hook before deactivating once.

    Returns:
        list: The report before activation, after activation, and after deactivation,
              followed by the shell names defined before activation and after deactivation.
    """
    script = '\n'.join([REPORT, setup, '_names', f'echo {SEP}', '_report', f'echo {SEP}']
                       + [f'. "{act}" || exit 3'] * activations
                       + ['_report', f'echo {SEP}', f'. "{deact}" || exit 3', '_report', f'echo {SEP}', '_names'])
    proc = subprocess.run(['env', '-i', f'PATH={BASE_PATH}', 'bash', '-c', script],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode or proc.stderr:
        raise AssertionError(f'Hook subshell exited {proc.returncode}:\n{proc.stderr}\n{proc.stdout}')
    names_before, before, active, after, names_after = proc.stdout.split(SEP + '\n')
    return [before, active, after, set(names_before.split()), set(names_after.split())]


class TestInstallArcHooks(unittest.TestCase):
    """
    Contains unit tests for the conda hooks written by install_arc.sh.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.tmp = tempfile.mkdtemp()
        cls.arc = '/opt/fake/ARC'
        cls.act, cls.deact = render_hooks(os.path.join(cls.tmp, 'plain'), cls.arc)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        """
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_hooks_render(self):
        """Test that both hooks render and the deactivation hook carries no literal backslash escapes"""
        self.assertTrue(os.path.isfile(self.act))
        with open(self.deact) as f:
            self.assertNotIn('\\$', f.read())

    def test_round_trip_pythonpath_unset(self):
        """Test that PYTHONPATH unset before activation is unset again after deactivation"""
        before, active, after, _, _ = round_trip(self.act, self.deact)
        self.assertEqual(before, f'PATH={BASE_PATH}\nPYTHONPATH<unset>\n')
        self.assertEqual(active, f'PATH={self.arc}:{BASE_PATH}\nPYTHONPATH={self.arc}\n')
        self.assertEqual(after, before)

    def test_round_trip_pythonpath_set(self):
        """Test that a set PYTHONPATH is restored exactly after deactivation"""
        before, active, after, _, _ = round_trip(self.act, self.deact, setup='export PYTHONPATH=/x/lib:/y/lib')
        self.assertEqual(active, f'PATH={self.arc}:{BASE_PATH}\nPYTHONPATH={self.arc}:/x/lib:/y/lib\n')
        self.assertEqual(after, before)

    def test_activate_twice_deactivate_once(self):
        """Test that activating twice and deactivating once restores the original environment"""
        for setup in ['', 'export PYTHONPATH=/x/lib']:
            before, active, after, _, _ = round_trip(self.act, self.deact, setup=setup, activations=2)
            self.assertEqual(active.count(self.arc), 2, msg=active)
            self.assertEqual(after, before, msg=setup)

    def test_round_trip_pythonpath_empty(self):
        """Test that a set-but-empty PYTHONPATH stays set and empty after deactivation"""
        before, active, after, _, _ = round_trip(self.act, self.deact, setup='export PYTHONPATH=')
        self.assertEqual(before, f'PATH={BASE_PATH}\nPYTHONPATH=\n')
        self.assertEqual(active, f'PATH={self.arc}:{BASE_PATH}\nPYTHONPATH={self.arc}\n')
        self.assertEqual(after, before)

    def test_preexisting_entries_survive(self):
        """Test that ARC entries the user had before activation are kept after deactivation"""
        setup = f'export PATH={BASE_PATH}:{self.arc}; export PYTHONPATH=/x/lib:{self.arc}'
        before, active, after, _, _ = round_trip(self.act, self.deact, setup=setup)
        self.assertEqual(active, before)
        self.assertEqual(after, before)

    def test_round_trip_path_with_spaces_and_globs(self):
        """Test a repository path with spaces and glob characters, next to an entry the glob would match"""
        arc = os.path.join(self.tmp, 'a b', '[ARC]*')
        act, deact = render_hooks(os.path.join(self.tmp, 'globs'), arc)
        sibling = os.path.join(self.tmp, 'a b', 'A')
        setup = f'export PATH="{sibling}:{BASE_PATH}"; export PYTHONPATH="{sibling}"'
        before, active, after, _, _ = round_trip(act, deact, setup=setup)
        self.assertEqual(active, f'PATH={arc}:{sibling}:{BASE_PATH}\nPYTHONPATH={arc}:{sibling}\n')
        self.assertEqual(after, before)

    def test_no_leftovers(self):
        """Test that deactivation leaves no variables or functions behind"""
        for setup in ['', 'export PYTHONPATH=/x/lib']:
            _, _, _, names_before, names_after = round_trip(self.act, self.deact, setup=setup)
            self.assertEqual(names_after - names_before, set(), msg=setup)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
