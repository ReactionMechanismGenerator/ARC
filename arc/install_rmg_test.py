#!/usr/bin/env python3
# encoding: utf-8

"""
Tests for the package-mode conda spec built by devtools/install_rmg.sh.

The script is run with ``--print-spec`` (no conda needed), or against a stub ``conda`` on PATH.
No real conda environment is created, read, or modified.
"""

import os
import re
import stat
import subprocess
import tempfile
import textwrap
import unittest

from arc.common import ARC_PATH


SCRIPT = os.path.join(ARC_PATH, 'devtools', 'install_rmg.sh')

# Directories that may hold a getopt supporting the long options the installer passes. macOS ships
# a BSD getopt in /usr/bin which rejects --long, so an isolated PATH of just /usr/bin:/bin makes the
# installer abort while parsing flags -- a failure of this harness rather than of the script. The
# Homebrew locations are searched too, so a Mac with gnu-getopt installed runs these tests for real
# instead of skipping them.
GETOPT_CANDIDATES = ('/usr/bin', '/bin',
                     '/opt/homebrew/opt/gnu-getopt/bin',   # Homebrew on Apple silicon
                     '/usr/local/opt/gnu-getopt/bin')      # Homebrew on Intel macOS


def gnu_getopt_dir() -> str | None:
    """
    Find a directory whose ``getopt`` accepts the long options the installer uses.

    Returns:
        str | None: The first candidate directory whose ``getopt`` parses ``--long``, or ``None``
                    if none of them does.
    """
    for directory in GETOPT_CANDIDATES:
        binary = os.path.join(directory, 'getopt')
        if not os.access(binary, os.X_OK):
            continue
        if subprocess.run([binary, '-o', 'a', '--long', 'alpha', '--', '--alpha'],
                          capture_output=True, text=True).returncode == 0:
            return directory
    return None


GETOPT_DIR = gnu_getopt_dir()
# Only the getopt directory is added, so a real conda frontend still cannot reach the isolated PATH.
BASE_PATH = '/usr/bin:/bin' if GETOPT_DIR in (None, '/usr/bin', '/bin') else f'{GETOPT_DIR}:/usr/bin:/bin'
requires_gnu_getopt = unittest.skipIf(
    GETOPT_DIR is None,
    'no getopt supporting --long is available; the installer needs GNU getopt (macOS: brew install gnu-getopt)')


def run_script(args: list, extra_env: dict = None, path: str = BASE_PATH) -> subprocess.CompletedProcess:
    """
    Run install_rmg.sh in an allowlisted environment: no conda, no shell startup hooks, an empty HOME.

    Args:
        args (list): Command line arguments for the script.
        extra_env (dict, optional): Variables to add to the environment.
        path (str): The PATH the script sees.

    Returns:
        subprocess.CompletedProcess: The finished process.
    """
    with tempfile.TemporaryDirectory() as home:
        env = {'PATH': path, 'HOME': home, 'LC_ALL': 'C.UTF-8'}
        env.update(extra_env or {})
        return subprocess.run(['/bin/bash', '--noprofile', '--norc', SCRIPT] + args,
                              env=env, capture_output=True, text=True, timeout=60)


def print_spec(extra_env: dict = None) -> list:
    """
    Return the package-mode spec tokens the script prints.

    Args:
        extra_env (dict, optional): Variables to add to the environment.

    Returns:
        list: One token per printed line.
    """
    result = run_script(['--print-spec'], extra_env=extra_env)
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def version_tuple(version: str) -> tuple:
    """
    Convert a dotted version string into a tuple of ints.

    Args:
        version (str): A version such as '0.9.41'.

    Returns:
        tuple: The integer components.
    """
    return tuple(int(part) for part in version.split('.'))


def write_stub_conda(stub_dir: str, python_version: str, rmg_version: str, list_fails: bool = False,
                     env_list_fails: bool = False) -> str:
    """
    Write a fake package manager that reports an existing rmg_env and logs install/create calls.

    The stub is written as ``micromamba``, ``mamba`` and ``conda``, so the script picks it whichever
    frontend it looks for first, and never reaches a real one later on PATH.

    Args:
        stub_dir (str): Directory to write the stub into.
        python_version (str): The python version the fake rmg_env reports.
        rmg_version (str): The rmg version the fake rmg_env reports.
        list_fails (bool): Whether ``conda list`` should fail, as for an unreadable env.
        env_list_fails (bool): Whether ``conda env list`` should fail, as for a broken conda config.

    Returns:
        str: The path of the call log the stub writes for install/create.
    """
    log = os.path.join(stub_dir, 'calls.log')
    stub = os.path.join(stub_dir, 'conda')
    with open(stub, 'w') as f:
        f.write(textwrap.dedent(f"""\
            #!/usr/bin/env bash
            if [[ "$1" == env && "$2" == list && "{str(env_list_fails).lower()}" == true ]]; then
                echo "CondaError: stub env list failure" >&2
                exit 1
            elif [[ "$1" == env && "$2" == list ]]; then
                echo "base      /fake/base"
                echo "rmg_env   /fake/base/envs/rmg_env"
            elif [[ "$1" == list && "{str(list_fails).lower()}" == true ]]; then
                echo "EnvironmentLocationNotFound: stub failure" >&2
                exit 1
            elif [[ "$1" == list ]]; then
                echo "python    {python_version}   h0_cpython   conda-forge"
                echo "rmg       {rmg_version}      py0          rmg"
            else
                echo "$@" >> "{log}"
            fi
            """))
    os.chmod(stub, os.stat(stub).st_mode | stat.S_IEXEC)
    for alias in ('micromamba', 'mamba'):
        os.symlink(stub, os.path.join(stub_dir, alias))
    return log


@requires_gnu_getopt
class TestInstallRmgSpec(unittest.TestCase):
    """
    Contains unit tests for the package-mode spec of install_rmg.sh.
    """

    def test_print_spec_exits_zero_without_conda(self):
        """Test that --print-spec runs with no conda frontend on PATH."""
        result = run_script(['--print-spec'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip())

    def test_package_spec_pins_rmg_4(self):
        """Test that the default spec pins rmg 4.0.0."""
        self.assertIn('rmg=4.0.0', print_spec())

    def test_package_spec_pins_python_311(self):
        """Test that the default spec pins Python 3.11."""
        tokens = print_spec()
        self.assertIn('python=3.11', tokens)
        self.assertEqual(len([t for t in tokens if t.startswith('python=')]), 1)

    def test_package_spec_numdifftools_from_conda_forge(self):
        """Test that numdifftools comes from conda-forge at version 0.9.41 or later."""
        matches = [re.fullmatch(r'conda-forge::numdifftools>=([\d.]+)', t) for t in print_spec()]
        versions = [m.group(1) for m in matches if m]
        self.assertEqual(len(versions), 1)
        self.assertGreaterEqual(version_tuple(versions[0]), (0, 9, 41))
        self.assertEqual(len([t for t in print_spec() if 'numdifftools' in t]), 1)

    def test_rmg_version_override(self):
        """Test that RMG_VERSION and RMG_PYTHON_VERSION still override the defaults."""
        tokens = print_spec({'RMG_VERSION': '9.9.9', 'RMG_PYTHON_VERSION': '3.12'})
        self.assertIn('rmg=9.9.9', tokens)
        self.assertIn('python=3.12', tokens)
        self.assertNotIn('rmg=4.0.0', tokens)

    def test_spec_written_once(self):
        """Test that the package spec is defined in one place, so create and update cannot drift."""
        with open(SCRIPT) as f:
            text = f.read()
        self.assertEqual(len(re.findall(r'"rmg=\$\{RMG_VERSION\}"', text)), 1)
        self.assertEqual(len(re.findall(r'numdifftools', text)), 1)

    def test_plain_dotted_overrides_are_accepted(self):
        """Test that the override check passes every version spelling the installer can re-recognize."""
        for version in ['4', '4.0', '4.0.0', '2026.1.0']:
            tokens = print_spec({'RMG_VERSION': version, 'RMG_PYTHON_VERSION': '3.11'})
            self.assertIn(f'rmg={version}', tokens)

    def test_rejects_override_syntax_it_cannot_recognize_later(self):
        """Test that a range or wildcard override is refused up front, naming the syntax.

        The existing-environment check matches an installed version against the pin as an exact
        value or a dotted prefix. A conda-style range would install and then be refused by that
        check, which would advise removing the environment it had just matched, so the installer
        must reject the syntax before it creates anything.
        """
        for version in ['4.0.*', '>=4.0', '4.0|4.1', '4.0,<5', '~=4.0']:
            result = run_script(['--print-spec'], extra_env={'RMG_VERSION': version})
            self.assertNotEqual(result.returncode, 0, f'{version!r} was accepted: {result.stdout}')
            self.assertIn('RMG_VERSION must be a plain dotted version', result.stderr)
        result = run_script(['--print-spec'], extra_env={'RMG_PYTHON_VERSION': '3.11.*'})
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('RMG_PYTHON_VERSION must be a plain dotted version', result.stderr)

    def test_help_does_not_depend_on_a_valid_override(self):
        """Test that --help still prints with a bad override, so the user can read the rules."""
        result = run_script(['--help'], extra_env={'RMG_VERSION': '4.0.*'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--print-spec', result.stdout)


@requires_gnu_getopt
class TestInstallRmgExistingEnv(unittest.TestCase):
    """
    Contains unit tests for updating an existing rmg_env, using a stub conda.
    """

    def setUp(self):
        """Provide a per-test temporary directory for the stub conda."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_refuses_env_on_other_python(self):
        """Test that an existing rmg_env on another python/rmg is refused, not changed in place."""
        log = write_stub_conda(self.tmp, python_version='3.9.23', rmg_version='3.3.0')
        result = run_script([], path=f'{self.tmp}:{BASE_PATH}')
        self.assertNotEqual(result.returncode, 0)
        output = result.stdout + result.stderr
        self.assertIn('env remove -n rmg_env', output)
        self.assertIn('3.9.23', output)
        self.assertFalse(os.path.isfile(log), 'conda install/create must not run')

    def test_retry_command_reproduces_the_requested_versions(self):
        """Test that the printed retry command carries the overrides that produced the mismatch.

        A bare ``bash install_rmg.sh`` would rebuild the default pins, so a user who followed the
        advice after asking for other versions would silently get 4.0.0/3.11 instead.
        """
        write_stub_conda(self.tmp, python_version='3.11.18', rmg_version='4.0.0')
        result = run_script([], extra_env={'RMG_VERSION': '3.3.0', 'RMG_PYTHON_VERSION': '3.9'},
                            path=f'{self.tmp}:{BASE_PATH}')
        self.assertNotEqual(result.returncode, 0)
        retry = next((line for line in result.stderr.splitlines() if line.strip().endswith(f'bash {SCRIPT}')), None)
        self.assertIsNotNone(retry, result.stderr)
        self.assertIn('RMG_VERSION=3.3.0', retry)
        self.assertIn('RMG_PYTHON_VERSION=3.9', retry)

    def test_refuses_when_env_cannot_be_listed(self):
        """Test that a failed package listing aborts with its own error, not with removal advice."""
        log = write_stub_conda(self.tmp, python_version='3.11.18', rmg_version='4.0.0', list_fails=True)
        result = run_script([], path=f'{self.tmp}:{BASE_PATH}')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Could not list the packages in rmg_env', result.stderr)
        self.assertIn('stub failure', result.stderr)
        self.assertNotIn('env remove', result.stdout + result.stderr)
        self.assertFalse(os.path.isfile(log), 'conda install/create must not run')

    def test_refuses_when_envs_cannot_be_listed(self):
        """Test that a failed environment listing aborts, rather than creating over an env it could not see."""
        log = write_stub_conda(self.tmp, python_version='3.11.18', rmg_version='4.0.0', env_list_fails=True)
        result = run_script([], path=f'{self.tmp}:{BASE_PATH}')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('stub env list failure', result.stderr)
        self.assertFalse(os.path.isfile(log), 'conda install/create must not run')

    def test_updates_env_matching_short_override(self):
        """Test that a short override such as RMG_VERSION=4.0 accepts the 4.0.0 it installed."""
        log = write_stub_conda(self.tmp, python_version='3.11.18', rmg_version='4.0.0')
        result = run_script([], extra_env={'RMG_VERSION': '4.0'}, path=f'{self.tmp}:{BASE_PATH}')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with open(log) as f:
            self.assertIn('rmg=4.0 ', f.read())

    def test_updates_env_matching_pin(self):
        """Test that an existing rmg_env already on the pin is updated with the pinned spec."""
        log = write_stub_conda(self.tmp, python_version='3.11.18', rmg_version='4.0.0')
        result = run_script([], path=f'{self.tmp}:{BASE_PATH}')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with open(log) as f:
            calls = f.read()
        self.assertIn('install -n rmg_env', calls)
        self.assertIn('rmg=4.0.0', calls)
        self.assertIn('python=3.11', calls)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
