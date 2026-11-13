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
HOOK_BLOCK = re.compile(r'^# ── activation hook.*?^\} >"\$DEACTIVATE_DIR/arc_env\.sh"\n', re.M | re.S)

REPORT = r"""
_report () {
    printf 'PATH=%s\n' "$PATH"
    if [ -n "${ARC_ROOT+x}" ]; then printf 'ARC_ROOT=%s\n' "$ARC_ROOT"; else printf 'ARC_ROOT<unset>\n'; fi
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
    if len(blocks) != 1:
        raise ValueError(f'Expected 1 hook-writing block in {INSTALL_ARC}, found {len(blocks)}')
    act_dir, deact_dir = os.path.join(target_dir, 'activate.d'), os.path.join(target_dir, 'deactivate.d')
    os.makedirs(act_dir, exist_ok=True)
    os.makedirs(deact_dir, exist_ok=True)
    env = {'PATH': BASE_PATH, 'ACTIVATE_DIR': act_dir, 'DEACTIVATE_DIR': deact_dir, 'ARC_REPO': arc_repo}
    subprocess.run(['bash', '-c', 'set -e\n' + ''.join(blocks)], env=env, check=True, timeout=60)
    return os.path.join(act_dir, 'arc_env.sh'), os.path.join(deact_dir, 'arc_env.sh')


def link_of(act: str) -> str:
    """
    Return the PATH entry the activation hook adds: the symlink install_arc.sh makes next to activate.d.

    Args:
        act (str): The path to the activation hook.

    Returns:
        str: The symlink path.
    """
    return os.path.join(os.path.dirname(os.path.dirname(act)), 'arc_repo')


def round_trip(act: str, deact: str, setup: str = '', activate: str = '') -> list:
    """
    Source the hooks in a clean bash subshell and report the environment at each stage.

    Args:
        act (str): The path to the activation hook.
        deact (str): The path to the deactivation hook.
        setup (str): Shell code that sets the pre-activation environment.
        activate (str): Shell code for the activation stage. Defaults to sourcing the activation hook once.

    Returns:
        list: The report before activation, after activation, and after deactivation,
              followed by the shell names defined before activation and after deactivation.
    """
    activate = activate or f'. "{act}" || exit 3'
    script = '\n'.join([REPORT, setup, '_names', f'echo {SEP}', '_report', f'echo {SEP}', activate,
                        '_report', f'echo {SEP}', f'. "{deact}" || exit 3', '_report', f'echo {SEP}', '_names'])
    proc = subprocess.run(['env', '-i', f'PATH={BASE_PATH}', 'bash', '-c', script],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode or proc.stderr:
        raise AssertionError(f'Hook subshell exited {proc.returncode}:\n{proc.stderr}\n{proc.stdout}')
    names_before, before, active, after, names_after = proc.stdout.split(SEP + '\n')
    return [before, active, after, set(names_before.split()), set(names_after.split())]


def conda_reactivate(act: str, deact: str) -> str:
    """
    Shell code that activates, then reactivates in the order conda uses.

    On reactivation conda runs the deactivation hook, re-exports the PATH it read before that hook
    ran, then runs the activation hook.

    Args:
        act (str): The path to the activation hook.
        deact (str): The path to the deactivation hook.

    Returns:
        str: The shell code.
    """
    return '\n'.join([f'. "{act}" || exit 3', '_snapshot=$PATH', f'. "{deact}" || exit 3',
                      'export PATH=$_snapshot; unset _snapshot', f'. "{act}" || exit 3'])


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
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        cls.arc = '/opt/fake/ARC'
        cls.act, cls.deact = render_hooks(os.path.join(cls.tmp, 'plain'), cls.arc)
        cls.link = link_of(cls.act)

    def test_hooks_render(self):
        """Test that both hooks render and the deactivation hook carries no literal backslash escapes"""
        self.assertTrue(os.path.isfile(self.act))
        with open(self.deact) as f:
            self.assertNotIn('\\$', f.read())

    def test_round_trip_pythonpath_unset(self):
        """Test that PYTHONPATH unset before activation is unset again after deactivation"""
        before, active, after, _, _ = round_trip(self.act, self.deact)
        self.assertEqual(before, f'PATH={BASE_PATH}\nARC_ROOT<unset>\nPYTHONPATH<unset>\n')
        self.assertEqual(active, f'PATH={self.link}:{BASE_PATH}\nARC_ROOT={self.arc}\nPYTHONPATH={self.arc}\n')
        self.assertEqual(after, before)

    def test_round_trip_pythonpath_set(self):
        """Test that a set PYTHONPATH is restored exactly after deactivation"""
        before, active, after, _, _ = round_trip(self.act, self.deact, setup='export PYTHONPATH=/x/lib:/y/lib')
        self.assertEqual(active, f'PATH={self.link}:{BASE_PATH}\nARC_ROOT={self.arc}\nPYTHONPATH={self.arc}:/x/lib:/y/lib\n')
        self.assertEqual(after, before)

    def test_conda_reactivation_then_deactivation(self):
        """Test that activate, reactivate (in conda's order), deactivate restores the original environment"""
        for setup in ['', 'export PYTHONPATH=/x/lib', 'export ARC_ROOT=/mine']:
            before, active, after, _, _ = round_trip(self.act, self.deact, setup=setup,
                                                     activate=conda_reactivate(self.act, self.deact))
            self.assertEqual(active.count(self.link), 1, msg=active)
            self.assertEqual(after, before, msg=setup)

    def test_activation_hook_sourced_twice(self):
        """Test that sourcing the activation hook twice adds nothing twice and still restores on deactivation"""
        for setup in ['', 'export ARC_ROOT=/mine']:
            before, active, after, _, _ = round_trip(self.act, self.deact, setup=setup,
                                                     activate=f'. "{self.act}"; . "{self.act}"')
            self.assertEqual(active.count(self.link), 1, msg=active)
            self.assertEqual(after, before, msg=setup)

    @unittest.skipIf(shutil.which('conda') is None, 'conda is not on PATH')
    def test_real_conda_reactivation(self):
        """Test the hooks in a real conda env: activate, activate again, deactivate leaves PATH and ARC_ROOT as they were"""
        prefix = os.path.join(self.tmp, 'conda_env')
        os.makedirs(os.path.join(prefix, 'conda-meta'))
        act, deact = render_hooks(os.path.join(prefix, 'etc', 'conda'), self.arc)
        base = subprocess.run(['conda', 'info', '--base'], check=True, capture_output=True, text=True).stdout.strip()
        script = '\n'.join([REPORT, f'. "{base}/etc/profile.d/conda.sh"', '_report', f'echo {SEP}',
                            f'conda activate "{prefix}"', f'conda activate "{prefix}"', '_report', f'echo {SEP}',
                            'conda deactivate', '_report'])
        proc = subprocess.run(['env', '-i', f'PATH={BASE_PATH}', f'HOME={self.tmp}', 'bash', '--norc', '-c', script],
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        before, active, after = proc.stdout.split(SEP + '\n')
        link = link_of(act)
        self.assertEqual(active.count(link), 1, msg=active)
        self.assertEqual(after, before)
        self.assertIn('ARC_ROOT<unset>', after)

    def test_user_pythonpath_entry_added_while_active_survives(self):
        """Test that an ARC entry the user adds to PYTHONPATH while active is kept after deactivation"""
        activate = f'. "{self.act}" || exit 3; export PYTHONPATH="$PYTHONPATH:{self.arc}"'
        before, active, after, _, _ = round_trip(self.act, self.deact, setup='export PYTHONPATH=/x/lib', activate=activate)
        self.assertEqual(after, f'PATH={BASE_PATH}\nARC_ROOT<unset>\nPYTHONPATH=/x/lib:{self.arc}\n')

    def test_round_trip_pythonpath_empty(self):
        """Test that a set-but-empty PYTHONPATH stays set and empty after deactivation"""
        before, active, after, _, _ = round_trip(self.act, self.deact, setup='export PYTHONPATH=')
        self.assertEqual(before, f'PATH={BASE_PATH}\nARC_ROOT<unset>\nPYTHONPATH=\n')
        self.assertEqual(active, f'PATH={self.link}:{BASE_PATH}\nARC_ROOT={self.arc}\nPYTHONPATH={self.arc}\n')
        self.assertEqual(after, before)

    def test_preexisting_entries_survive(self):
        """Test that ARC entries and an ARC_ROOT the user had before activation are kept after deactivation"""
        setup = f'export PATH={BASE_PATH}:{self.arc}; export PYTHONPATH=/x/lib:{self.arc}; export ARC_ROOT=/mine'
        before, active, after, _, _ = round_trip(self.act, self.deact, setup=setup)
        self.assertEqual(active, f'PATH={self.link}:{BASE_PATH}:{self.arc}\nARC_ROOT={self.arc}\nPYTHONPATH=/x/lib:{self.arc}\n')
        self.assertEqual(after, before)

    def test_round_trip_path_with_spaces_and_globs(self):
        """Test a repository path with spaces and glob characters, next to an entry the glob would match"""
        arc = os.path.join(self.tmp, 'a b', '[ARC]*')
        act, deact = render_hooks(os.path.join(self.tmp, 'globs'), arc)
        sibling = os.path.join(self.tmp, 'a b', 'A')
        setup = f'export PATH="{sibling}:{BASE_PATH}"; export PYTHONPATH="{sibling}"'
        before, active, after, _, _ = round_trip(act, deact, setup=setup)
        self.assertEqual(active, f'PATH={link_of(act)}:{sibling}:{BASE_PATH}\nARC_ROOT={arc}\nPYTHONPATH={arc}:{sibling}\n')
        self.assertEqual(after, before)

    def test_round_trip_path_with_shell_metacharacters(self):
        """Test a repository path holding a double quote, a single quote, a dollar sign and a backtick"""
        arc = os.path.join(self.tmp, 'q"u\'o $HOME `id` x')
        act, deact = render_hooks(os.path.join(self.tmp, 'meta'), arc)
        before, active, after, _, _ = round_trip(act, deact)
        self.assertEqual(os.path.realpath(link_of(act)), arc)
        self.assertEqual(active, f'PATH={link_of(act)}:{BASE_PATH}\nARC_ROOT={arc}\nPYTHONPATH={arc}\n')
        self.assertEqual(after, before)

    def test_no_leftovers(self):
        """Test that deactivation leaves no variables or functions behind"""
        for setup in ['', 'export PYTHONPATH=/x/lib', 'export ARC_ROOT=/mine']:
            _, _, _, names_before, names_after = round_trip(self.act, self.deact, setup=setup)
            self.assertEqual(names_after - names_before, set(), msg=setup)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
