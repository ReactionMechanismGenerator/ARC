#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.env_run module.
"""

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arc.job.env_run import (
    _detect_launcher,
    _run_flags_for,
    env_prefix_from_python,
    rmg_env_command,
    run_in_conda_env,
)


def _completed(returncode=0, stdout='', stderr=''):
    """Build a CompletedProcess for mocking subprocess.run."""
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


class TestEnvPrefixFromPython(unittest.TestCase):
    """env_prefix_from_python should derive the env prefix from any
    interpreter path, not assume a literal envs/ segment."""

    def test_standard_layout(self):
        self.assertEqual(
            env_prefix_from_python('/opt/conda/envs/tst_env/bin/python'),
            '/opt/conda/envs/tst_env',
        )

    def test_conda_envs_path_layout(self):
        # CONDA_ENVS_PATH lets users place envs anywhere — no `envs/` segment.
        self.assertEqual(
            env_prefix_from_python('/scratch/conda_envs/ts_gcn/bin/python'),
            '/scratch/conda_envs/ts_gcn',
        )

    def test_user_home_micromamba_layout(self):
        self.assertEqual(
            env_prefix_from_python('/home/alice/micromamba/envs/tani_env/bin/python'),
            '/home/alice/micromamba/envs/tani_env',
        )

    def test_rejects_non_python_binary(self):
        with self.assertRaises(ValueError):
            env_prefix_from_python('/usr/bin/awk')

    def test_rejects_non_bin_parent(self):
        with self.assertRaises(ValueError):
            env_prefix_from_python('/some/where/python')

    def test_python_is_symlink_to_versioned_binary(self):
        """Real conda/mamba/micromamba envs ship ``python`` as a symlink
        to ``python3.X``. The function must validate lexically — if it
        followed the symlink it would see ``python3.12`` and reject."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = Path(tmpdir) / "envs" / "tani_env"
            (env / "bin").mkdir(parents=True)
            versioned = env / "bin" / "python3.12"
            versioned.write_text("#!/bin/sh\nexec true\n")
            os.symlink("python3.12", env / "bin" / "python")
            self.assertEqual(
                env_prefix_from_python(str(env / "bin" / "python")),
                str(env),
            )


class TestRunFlagsFor(unittest.TestCase):
    """_run_flags_for chooses stdio flags by launcher basename so symlinks
    and odd MAMBA_EXE-points-at-micromamba setups still get the right one."""

    def test_conda_needs_no_capture_output(self):
        self.assertEqual(_run_flags_for('/opt/conda/bin/conda'), ['--no-capture-output'])

    def test_mamba_needs_no_capture_output(self):
        self.assertEqual(_run_flags_for('/opt/conda/bin/mamba'), ['--no-capture-output'])

    def test_micromamba_omits_no_capture_output(self):
        self.assertEqual(_run_flags_for('/usr/local/bin/micromamba'), [])

    def test_decides_by_basename_not_path(self):
        # MAMBA_EXE pointing at micromamba is a real configuration.
        self.assertEqual(_run_flags_for('/whatever/path/micromamba'), [])


class TestDetectLauncher(unittest.TestCase):
    """_detect_launcher prefers the active launcher (CONDA_EXE/MAMBA_EXE)
    over PATH lookup, and falls back to conda → mamba → micromamba."""

    def test_prefers_conda_exe_when_set(self):
        with patch.dict('os.environ', {'CONDA_EXE': '/opt/conda/bin/conda'}, clear=True), \
                patch('arc.job.env_run.os.path.isfile', return_value=True), \
                patch('arc.job.env_run.shutil.which') as mock_which:
            launcher, flags = _detect_launcher()
        self.assertEqual(launcher, '/opt/conda/bin/conda')
        self.assertEqual(flags, ['--no-capture-output'])
        mock_which.assert_not_called()

    def test_falls_back_to_mamba_exe(self):
        with patch.dict('os.environ', {'MAMBA_EXE': '/opt/mamba/bin/micromamba'}, clear=True), \
                patch('arc.job.env_run.os.path.isfile', return_value=True), \
                patch('arc.job.env_run.shutil.which'):
            launcher, flags = _detect_launcher()
        # Basename is micromamba, so no --no-capture-output even though
        # MAMBA_EXE was the env var that pointed us at it.
        self.assertEqual(launcher, '/opt/mamba/bin/micromamba')
        self.assertEqual(flags, [])

    def test_falls_back_to_path_lookup(self):
        which_returns = {'conda': None, 'mamba': '/usr/bin/mamba', 'micromamba': None}
        with patch.dict('os.environ', {}, clear=True), \
                patch('arc.job.env_run.shutil.which', side_effect=lambda n: which_returns[n]):
            launcher, flags = _detect_launcher()
        self.assertEqual(launcher, '/usr/bin/mamba')
        self.assertEqual(flags, ['--no-capture-output'])

    def test_path_lookup_prefers_conda_over_mamba(self):
        which_returns = {
            'conda': '/usr/bin/conda',
            'mamba': '/usr/bin/mamba',
            'micromamba': '/usr/bin/micromamba',
        }
        with patch.dict('os.environ', {}, clear=True), \
                patch('arc.job.env_run.shutil.which', side_effect=lambda n: which_returns[n]):
            launcher, _ = _detect_launcher()
        self.assertEqual(launcher, '/usr/bin/conda')

    def test_raises_when_no_launcher_found(self):
        with patch.dict('os.environ', {}, clear=True), \
                patch('arc.job.env_run.shutil.which', return_value=None):
            with self.assertRaises(FileNotFoundError):
                _detect_launcher()


class TestRunInCondaEnv(unittest.TestCase):
    """run_in_conda_env should build the right argv and shell out without a shell."""

    def test_argv_uses_prefix_and_extra_flags(self):
        with patch('arc.job.env_run._detect_launcher',
                   return_value=('/opt/conda/bin/conda', ['--no-capture-output'])), \
                patch('arc.job.env_run.subprocess.run',
                      return_value=_completed()) as mock_run:
            run_in_conda_env(
                '/opt/conda/envs/tst_env/bin/python',
                '/path/to/script.py',
                '--flag', 'value',
            )
        mock_run.assert_called_once()
        argv = mock_run.call_args.args[0]
        self.assertEqual(
            argv,
            [
                '/opt/conda/bin/conda', 'run', '--no-capture-output',
                '-p', '/opt/conda/envs/tst_env',
                'python', '/path/to/script.py',
                '--flag', 'value',
            ],
        )
        # Streams must be captured so the helper can log them centrally.
        kwargs = mock_run.call_args.kwargs
        self.assertTrue(kwargs.get('capture_output'))
        self.assertTrue(kwargs.get('text'))
        # No shell=True — args go through as a list.
        self.assertNotIn('shell', kwargs)

    def test_micromamba_omits_no_capture_flag(self):
        with patch('arc.job.env_run._detect_launcher',
                   return_value=('/usr/bin/micromamba', [])), \
                patch('arc.job.env_run.subprocess.run',
                      return_value=_completed()) as mock_run:
            run_in_conda_env(
                '/scratch/envs/ts_gcn/bin/python',
                '/path/to/gcn.py',
            )
        argv = mock_run.call_args.args[0]
        self.assertEqual(
            argv,
            [
                '/usr/bin/micromamba', 'run',
                '-p', '/scratch/envs/ts_gcn',
                'python', '/path/to/gcn.py',
            ],
        )

    def test_check_kwarg_passes_through(self):
        with patch('arc.job.env_run._detect_launcher',
                   return_value=('/opt/conda/bin/conda', ['--no-capture-output'])), \
                patch('arc.job.env_run.subprocess.run',
                      return_value=_completed()) as mock_run:
            run_in_conda_env(
                '/opt/conda/envs/tst_env/bin/python',
                '/path/to/script.py',
                check=True,
            )
        self.assertTrue(mock_run.call_args.kwargs.get('check'))

    def test_default_inherits_parent_environment(self):
        """Without strip_pythonpath the child inherits the parent env untouched (env=None)."""
        with patch('arc.job.env_run._detect_launcher',
                   return_value=('/opt/conda/bin/conda', ['--no-capture-output'])), \
                patch('arc.job.env_run.subprocess.run',
                      return_value=_completed()) as mock_run:
            run_in_conda_env(
                '/opt/conda/envs/ts_gcn/bin/python',
                '/path/to/gcn.py',
            )
        self.assertIsNone(mock_run.call_args.kwargs.get('env'))

    def test_strip_pythonpath_removes_only_pythonpath(self):
        """strip_pythonpath drops PYTHONPATH from the child env and keeps everything else,
        so a stale source checkout on the caller's PYTHONPATH cannot shadow the target
        env's installed package."""
        parent_env = {'PATH': '/usr/bin', 'HOME': '/home/u',
                      'PYTHONPATH': '/home/u/Code/KinBot-2.0.6'}
        with patch('arc.job.env_run._detect_launcher',
                   return_value=('/opt/conda/bin/conda', ['--no-capture-output'])), \
                patch('arc.job.env_run.subprocess.run',
                      return_value=_completed()) as mock_run, \
                patch.dict('arc.job.env_run.os.environ', parent_env, clear=True):
            run_in_conda_env(
                '/opt/conda/envs/kinbot_env/bin/python',
                '/path/to/kinbot_script.py',
                strip_pythonpath=True,
            )
        child_env = mock_run.call_args.kwargs.get('env')
        self.assertIsNotNone(child_env)
        self.assertNotIn('PYTHONPATH', child_env)
        self.assertEqual(child_env['PATH'], '/usr/bin')
        self.assertEqual(child_env['HOME'], '/home/u')

    def test_failure_logs_warning_with_captured_streams(self):
        completed = _completed(returncode=2, stdout='partial output\n',
                               stderr='Traceback...\nValueError: boom\n')
        with patch('arc.job.env_run._detect_launcher',
                   return_value=('/opt/conda/bin/conda', ['--no-capture-output'])), \
                patch('arc.job.env_run.subprocess.run', return_value=completed), \
                patch('arc.job.env_run.logger') as mock_logger:
            result = run_in_conda_env(
                '/opt/conda/envs/tst_env/bin/python',
                '/path/to/script.py',
            )
        self.assertEqual(result.returncode, 2)
        mock_logger.warning.assert_called_once()
        # Render the warning to verify it carries the actual stderr contents.
        fmt, *args = mock_logger.warning.call_args.args
        rendered = fmt % tuple(args)
        self.assertIn('ValueError: boom', rendered)
        self.assertIn('partial output', rendered)
        self.assertIn('/path/to/script.py', rendered)

    def test_success_logs_debug_not_warning(self):
        with patch('arc.job.env_run._detect_launcher',
                   return_value=('/opt/conda/bin/conda', ['--no-capture-output'])), \
                patch('arc.job.env_run.subprocess.run',
                      return_value=_completed(stdout='ok\n')), \
                patch('arc.job.env_run.logger') as mock_logger:
            run_in_conda_env(
                '/opt/conda/envs/tst_env/bin/python',
                '/path/to/script.py',
            )
        mock_logger.warning.assert_not_called()
        mock_logger.debug.assert_called()


class TestRmgEnvCommand(unittest.TestCase):
    """rmg_env_command must shell-quote caller-supplied cwd/env_vars values;
    wrapping them in bare double quotes does not stop command substitution."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        # A stand-in launcher: strips ``run -n <env> python`` and execs the
        # rest with the real interpreter, so the generated script can be
        # executed end-to-end without a real conda/mamba env present.
        fake_mamba = Path(self.tmpdir.name) / 'fake_mamba'
        fake_mamba.write_text(f'#!/bin/bash\nshift 4\nexec "{sys.executable}" "$@"\n')
        fake_mamba.chmod(0o755)
        self.fake_mamba = str(fake_mamba)
        self.env_patch = patch.dict(os.environ, {'MAMBA_EXE': self.fake_mamba})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    @staticmethod
    def _run_script(script: str) -> subprocess.CompletedProcess:
        return subprocess.run(['bash', '-c', script], capture_output=True, text=True)

    def test_cwd_command_substitution_is_neutralized(self):
        marker = Path(self.tmpdir.name) / 'pwned_dollar'
        cwd = f'{self.tmpdir.name}/$(touch {marker})'
        script = rmg_env_command("-c 'pass'", cwd=cwd)
        self._run_script(script)
        self.assertFalse(marker.exists())

    def test_cwd_with_backticks_is_neutralized(self):
        marker = Path(self.tmpdir.name) / 'pwned_backtick'
        cwd = f'{self.tmpdir.name}/`touch {marker}`'
        script = rmg_env_command("-c 'pass'", cwd=cwd)
        self._run_script(script)
        self.assertFalse(marker.exists())

    def test_env_vars_command_substitution_is_neutralized(self):
        marker = Path(self.tmpdir.name) / 'pwned_env'
        script = rmg_env_command("-c 'pass'", env_vars={'FOO': f'$(touch {marker})'})
        self._run_script(script)
        self.assertFalse(marker.exists())

    def test_cwd_with_space_is_quoted_in_script(self):
        cwd = f'{self.tmpdir.name}/a dir with spaces'
        script = rmg_env_command("-c 'pass'", cwd=cwd)
        self.assertIn(shlex.quote(cwd), script)

    def test_cwd_with_single_quote_is_quoted_in_script(self):
        cwd = f"{self.tmpdir.name}/o'brien"
        script = rmg_env_command("-c 'pass'", cwd=cwd)
        self.assertIn(shlex.quote(cwd), script)

    def test_env_vars_with_space_is_quoted_in_script(self):
        script = rmg_env_command("-c 'pass'", env_vars={'FOO': 'a value with spaces'})
        self.assertIn(shlex.quote('a value with spaces'), script)

    def test_cwd_happy_path_still_cds(self):
        target = Path(self.tmpdir.name) / 'target dir'
        target.mkdir()
        marker = target / 'marker'
        script = rmg_env_command("-c \"open('marker', 'w').close()\"", cwd=str(target))
        result = self._run_script(script)
        self.assertTrue(marker.exists(), f'stdout={result.stdout!r} stderr={result.stderr!r}')

    def test_py_args_list_with_space_survives_as_single_token(self):
        marker = Path(self.tmpdir.name) / 'received_arg.txt'
        path_with_space = '/tmp/My Runs/proj/RMG_thermo.yml'
        script = rmg_env_command(['-c',
                                  f"import sys; open({str(marker)!r}, 'w').write(sys.argv[1])",
                                  path_with_space])
        result = self._run_script(script)
        self.assertTrue(marker.exists(), f'stdout={result.stdout!r} stderr={result.stderr!r}')
        self.assertEqual(marker.read_text(), path_with_space)

    def test_py_args_list_with_parens_runs_successfully(self):
        marker = Path(self.tmpdir.name) / 'received_arg_parens.txt'
        path_with_parens = '/tmp/2026-08 (rerun)/x.yml'
        script = rmg_env_command(['-c',
                                  f"import sys; open({str(marker)!r}, 'w').write(sys.argv[1])",
                                  path_with_parens])
        result = self._run_script(script)
        self.assertTrue(marker.exists(), f'stdout={result.stdout!r} stderr={result.stderr!r}')
        self.assertEqual(marker.read_text(), path_with_parens)

    def test_py_args_list_dollar_paren_substitution_is_neutralized(self):
        marker = Path(self.tmpdir.name) / 'pwned_py_args_dollar'
        malicious = f'$(touch {marker})'
        script = rmg_env_command(['-c', 'pass', malicious])
        self._run_script(script)
        self.assertFalse(marker.exists())

    def test_py_args_list_backtick_substitution_is_neutralized(self):
        marker = Path(self.tmpdir.name) / 'pwned_py_args_backtick'
        malicious = f'`touch {marker}`'
        script = rmg_env_command(['-c', 'pass', malicious])
        self._run_script(script)
        self.assertFalse(marker.exists())

    def test_py_args_str_backward_compatible_unquoted(self):
        script = rmg_env_command("-c 'pass'")
        self.assertIn("python -c 'pass'", script)

    def test_py_args_list_ordinary_path_unmangled(self):
        script = rmg_env_command(['script.py', '/tmp/plain/path.yml'])
        self.assertIn('python script.py /tmp/plain/path.yml', script)

    def test_py_args_list_arkane_module_invocation_three_tokens(self):
        script = rmg_env_command(['-m', 'arkane', 'input.py'])
        self.assertIn('python -m arkane input.py', script)


class TestRmgEnvCommandPythonPath(unittest.TestCase):
    """PYTHONPATH is normalized identically on all three branches: a launcher's
    ``run`` never manages it, and the direct-interpreter branch has no launcher,
    so ARC's own PYTHONPATH would otherwise leak into the child. When RMG_PATH is
    set the child gets exactly that (so a source-tree RMG-Py/Arkane checkout
    reachable only via PYTHONPATH is importable); otherwise PYTHONPATH is
    scrubbed so rmg_env's own site-packages win."""

    def setUp(self):
        # Force the RMG_PYTHON branch: no MAMBA_EXE, and RMG_PYTHON resolves
        # to a real file (RMG_PATH/RMG_PYTHON come from the patched settings
        # dict below; os.path.isfile still needs a real path on disk, so
        # point it at this test file itself).
        self.env_patch = patch.dict(os.environ, {}, clear=False)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        os.environ.pop('MAMBA_EXE', None)
        self.fake_rmg_python = __file__

    def test_pythonpath_reexported_from_rmg_path(self):
        settings_overrides = {'RMG_ENV_NAME': 'rmg_env', 'RMG_PYTHON': self.fake_rmg_python,
                               'RMG_PATH': '/opt/RMG-Py'}
        with patch.dict('arc.job.env_run.settings', settings_overrides):
            script = rmg_env_command("-c 'pass'")
        self.assertIn(f'export PYTHONPATH={shlex.quote("/opt/RMG-Py")}', script)
        # The re-export must come after the unset, so it is not clobbered.
        unset_idx = script.index('unset ')
        export_idx = script.index('export PYTHONPATH=')
        self.assertLess(unset_idx, export_idx)

    def test_no_pythonpath_export_when_rmg_path_falsy(self):
        settings_overrides = {'RMG_ENV_NAME': 'rmg_env', 'RMG_PYTHON': self.fake_rmg_python,
                               'RMG_PATH': None}
        with patch.dict('arc.job.env_run.settings', settings_overrides):
            script = rmg_env_command("-c 'pass'")
        self.assertNotIn('export PYTHONPATH=', script)

    def test_mamba_branch_exports_pythonpath_from_rmg_path(self):
        os.environ['MAMBA_EXE'] = __file__  # a real file forces the MAMBA_EXE branch
        settings_overrides = {'RMG_ENV_NAME': 'rmg_env', 'RMG_PATH': '/opt/RMG-Py'}
        with patch.dict('arc.job.env_run.settings', settings_overrides):
            script = rmg_env_command("-c 'pass'")
        self.assertIn(f'export PYTHONPATH={shlex.quote("/opt/RMG-Py")}', script)
        # Set before the launcher's ``run`` so the child inherits it.
        self.assertLess(script.index('export PYTHONPATH='), script.index('run -n rmg_env'))

    def test_mamba_branch_scrubs_pythonpath_when_rmg_path_falsy(self):
        os.environ['MAMBA_EXE'] = __file__
        settings_overrides = {'RMG_ENV_NAME': 'rmg_env', 'RMG_PATH': None}
        with patch.dict('arc.job.env_run.settings', settings_overrides):
            script = rmg_env_command("-c 'pass'")
        self.assertIn('unset PYTHONPATH', script)
        self.assertNotIn('export PYTHONPATH=', script)

    def test_hunt_branch_exports_pythonpath_from_rmg_path(self):
        os.environ.pop('MAMBA_EXE', None)  # no MAMBA_EXE + no RMG_PYTHON forces the hunt branch
        settings_overrides = {'RMG_ENV_NAME': 'rmg_env', 'RMG_PYTHON': None, 'RMG_PATH': '/opt/RMG-Py'}
        with patch.dict('arc.job.env_run.settings', settings_overrides):
            script = rmg_env_command("-c 'pass'")
        self.assertIn(f'export PYTHONPATH={shlex.quote("/opt/RMG-Py")}', script)
        # Set inside the login-shell heredoc, before the launcher-hunt loop.
        self.assertLess(script.index('export PYTHONPATH='), script.index('for _launcher in'))

    def test_hunt_branch_scrubs_pythonpath_when_rmg_path_falsy(self):
        os.environ.pop('MAMBA_EXE', None)
        settings_overrides = {'RMG_ENV_NAME': 'rmg_env', 'RMG_PYTHON': None, 'RMG_PATH': None}
        with patch.dict('arc.job.env_run.settings', settings_overrides):
            script = rmg_env_command("-c 'pass'")
        self.assertIn('unset PYTHONPATH', script)
        self.assertNotIn('export PYTHONPATH=', script)


if __name__ == '__main__':
    unittest.main()
