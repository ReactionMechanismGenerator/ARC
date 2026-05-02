#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.env_run module.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arc.job.env_run import (
    _detect_launcher,
    _run_flags_for,
    env_prefix_from_python,
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


if __name__ == '__main__':
    unittest.main()
