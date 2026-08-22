#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for arc.settings.crest
"""

import os
import stat
import tempfile
import unittest
from unittest.mock import patch

from arc.settings.crest import (
    DEFAULT_STANDALONE_DIR,
    find_crest_executable,
    find_env_activation_command,
    find_highest_version_in_directory,
    get_crest_paths,
    parse_version,
)


class TestCrestSettingsUtils(unittest.TestCase):

    def _make_executable(self, path: str):
        """Create an executable stub file at ``path``."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("#!/bin/bash\n")
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IXUSR)

    def _make_activation_script(self, root: str, name: str):
        """Create an ``etc/profile.d/<name>`` activation script under ``root``."""
        script_path = os.path.join(root, "etc", "profile.d", name)
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, "w") as f:
            f.write("# activation stub\n")
        return script_path

    def _isolated_env(self, standalone_dir: str = ""):
        """Return a patcher pinning ARC_CREST_STANDALONE_DIR so the host filesystem cannot leak in."""
        return patch.dict(os.environ, {"ARC_CREST_STANDALONE_DIR": standalone_dir}, clear=False)

    def test_parse_version(self):
        """Test parsing versions out of folder names"""
        self.assertEqual(parse_version("crest-3.0.2"), (3, 0, 2))
        self.assertEqual(parse_version("v212"), (2, 1, 2))
        self.assertEqual(parse_version("version-2.1"), (2, 1, 0))
        self.assertEqual(parse_version("foo"), (0, 0, 0))

    def test_find_highest_version_in_directory(self):
        """Test that the highest-version crest build in a directory is selected"""
        with tempfile.TemporaryDirectory() as td:
            low = os.path.join(td, "crest-2.1")
            high = os.path.join(td, "crest-3.0.2")
            os.makedirs(low)
            os.makedirs(high)
            self._make_executable(os.path.join(low, "crest"))
            self._make_executable(os.path.join(high, "crest"))

            found = find_highest_version_in_directory(td, "crest")
            self.assertEqual(found, os.path.join(high, "crest"))

    def test_find_highest_version_in_directory_missing_dir(self):
        """Test that a missing or unnamed directory yields None rather than raising"""
        self.assertIsNone(find_highest_version_in_directory("", "crest"))
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(find_highest_version_in_directory(os.path.join(td, "nope"), "crest"))

    def test_find_highest_version_in_directory_unreadable_dir(self):
        """Test that an OSError from os.listdir yields None, keeping arc.settings.settings importable"""
        with tempfile.TemporaryDirectory() as td:
            with patch("arc.settings.crest.os.listdir", side_effect=OSError("stale file handle")):
                self.assertIsNone(find_highest_version_in_directory(td, "crest"))

    def test_find_crest_executable_prefers_standalone(self):
        """Test that a standalone build wins over any environment installation"""
        with tempfile.TemporaryDirectory() as td:
            standalone = os.path.join(td, "crest-3.0.2")
            os.makedirs(standalone)
            standalone_crest = os.path.join(standalone, "crest")
            self._make_executable(standalone_crest)

            with self._isolated_env(standalone_dir=td):
                path, env_cmd = find_crest_executable()
            self.assertEqual(path, standalone_crest)
            self.assertEqual(env_cmd, "")

    def test_find_crest_executable_standalone_dir_default_and_override(self):
        """Test that the standalone directory defaults to /Local/ce_dana and is disabled by an empty override"""
        with tempfile.TemporaryDirectory() as td:
            fake_home = os.path.join(td, "home")
            os.makedirs(fake_home)
            for standalone_dir, expected_probe in [(None, DEFAULT_STANDALONE_DIR), ("", "")]:
                with patch.dict(os.environ, {}, clear=False):
                    if standalone_dir is None:
                        os.environ.pop("ARC_CREST_STANDALONE_DIR", None)
                    else:
                        os.environ["ARC_CREST_STANDALONE_DIR"] = standalone_dir
                    with patch("arc.settings.crest.os.path.expanduser", return_value=fake_home):
                        with patch("arc.settings.crest.sys.executable", os.path.join(td, "python")):
                            with patch("arc.settings.crest.shutil.which", return_value=None):
                                with patch("arc.settings.crest.find_highest_version_in_directory") as mock_find:
                                    path, env_cmd = find_crest_executable()
                mock_find.assert_called_once_with(expected_probe, "crest")
                self.assertIsNone(path)
                self.assertIsNone(env_cmd)

    def test_find_crest_executable_env_detection(self):
        """Test detecting a conda env installation and its activation command"""
        with tempfile.TemporaryDirectory() as td:
            fake_home = os.path.join(td, "home")
            conda_root = os.path.join(fake_home, "miniforge3")
            crest_path = os.path.join(conda_root, "envs", "crest_env", "bin", "crest")
            self._make_executable(crest_path)
            conda_sh = self._make_activation_script(conda_root, "conda.sh")

            with self._isolated_env():
                with patch("arc.settings.crest.os.path.expanduser", return_value=fake_home):
                    with patch("arc.settings.crest.sys.executable", os.path.join(td, "python")):
                        with patch("arc.settings.crest.shutil.which", return_value=None):
                            path, env_cmd = find_crest_executable()
            self.assertEqual(path, crest_path)
            self.assertEqual(env_cmd, f"source {conda_sh} && conda activate crest_env")

    def test_find_crest_executable_micromamba_env_detection(self):
        """Test detecting a micromamba env installation and its activation command"""
        with tempfile.TemporaryDirectory() as td:
            fake_home = os.path.join(td, "home")
            mamba_root = os.path.join(fake_home, "micromamba")
            crest_path = os.path.join(mamba_root, "envs", "crest_env", "bin", "crest")
            self._make_executable(crest_path)
            micromamba_sh = self._make_activation_script(mamba_root, "micromamba.sh")

            with self._isolated_env():
                with patch("arc.settings.crest.os.path.expanduser", return_value=fake_home):
                    with patch("arc.settings.crest.sys.executable", os.path.join(td, "python")):
                        with patch("arc.settings.crest.shutil.which", return_value=None):
                            path, env_cmd = find_crest_executable()
            self.assertEqual(path, crest_path)
            self.assertEqual(env_cmd, f"source {micromamba_sh} && micromamba activate crest_env")

    def test_find_crest_executable_dot_conda_is_not_a_conda_root(self):
        """Test that ~/.conda, which holds envs but no activation script, does not yield an activation command"""
        with tempfile.TemporaryDirectory() as td:
            fake_home = os.path.join(td, "home")
            crest_path = os.path.join(fake_home, ".conda", "envs", "crest_env", "bin", "crest")
            self._make_executable(crest_path)
            self._make_activation_script(os.path.join(fake_home, "miniconda3"), "conda.sh")

            with self._isolated_env():
                with patch("arc.settings.crest.os.path.expanduser", return_value=fake_home):
                    with patch("arc.settings.crest.sys.executable", os.path.join(td, "python")):
                        with patch("arc.settings.crest.shutil.which", return_value=None):
                            path, env_cmd = find_crest_executable()
            self.assertEqual(path, crest_path)
            self.assertEqual(env_cmd, "")
            self.assertNotIn(".conda", env_cmd)

    def test_find_crest_executable_in_active_env_bin(self):
        """Test that a crest in the active env bin is used as-is, without sourcing its path as a conda root"""
        with tempfile.TemporaryDirectory() as td:
            fake_home = os.path.join(td, "home")
            os.makedirs(fake_home)
            active_env = os.path.join(td, "miniconda3", "envs", "arc_env")
            crest_path = os.path.join(active_env, "bin", "crest")
            self._make_executable(crest_path)
            self._make_activation_script(os.path.join(td, "miniconda3"), "conda.sh")

            with self._isolated_env():
                with patch("arc.settings.crest.os.path.expanduser", return_value=fake_home):
                    with patch("arc.settings.crest.sys.executable", os.path.join(active_env, "bin", "python")):
                        with patch("arc.settings.crest.shutil.which", return_value=None):
                            path, env_cmd = find_crest_executable()
            self.assertEqual(path, crest_path)
            self.assertEqual(env_cmd, "")
            self.assertNotIn(active_env, env_cmd)
            self.assertNotIn("conda activate", env_cmd)

    def test_find_crest_executable_path_fallback(self):
        """Test falling back to a crest found on PATH"""
        with tempfile.TemporaryDirectory() as td:
            fake_home = os.path.join(td, "home")
            os.makedirs(fake_home)
            path_crest = os.path.join(td, "usr", "bin", "crest")
            self._make_executable(path_crest)

            with self._isolated_env():
                with patch("arc.settings.crest.os.path.expanduser", return_value=fake_home):
                    with patch("arc.settings.crest.sys.executable", os.path.join(td, "python")):
                        with patch("arc.settings.crest.shutil.which", return_value=path_crest):
                            path, env_cmd = find_crest_executable()
            self.assertEqual(path, path_crest)
            self.assertEqual(env_cmd, "")

    def test_find_crest_executable_not_found(self):
        """Test that (None, None) is returned when no crest executable exists anywhere"""
        with tempfile.TemporaryDirectory() as td:
            fake_home = os.path.join(td, "home")
            os.makedirs(fake_home)

            with self._isolated_env(standalone_dir=os.path.join(td, "no_such_dir")):
                with patch("arc.settings.crest.os.path.expanduser", return_value=fake_home):
                    with patch("arc.settings.crest.sys.executable", os.path.join(td, "python")):
                        with patch("arc.settings.crest.shutil.which", return_value=None):
                            path, env_cmd = find_crest_executable()
            self.assertIsNone(path)
            self.assertIsNone(env_cmd)

    def test_find_env_activation_command_without_env_marker(self):
        """Test that an executable outside an envs/crest_env directory yields no activation command"""
        self.assertEqual(find_env_activation_command("/opt/crest-3.0.2/crest"), "")
        self.assertEqual(find_env_activation_command("envs/crest_env/bin/crest"), "")

    def test_get_crest_paths_is_cached(self):
        """Test that the crest lookup touches the filesystem only once"""
        get_crest_paths.cache_clear()
        self.addCleanup(get_crest_paths.cache_clear)
        with patch("arc.settings.crest.find_crest_executable",
                   return_value=("/opt/crest", "")) as mock_find:
            first = get_crest_paths()
            second = get_crest_paths()
        self.assertEqual(first, ("/opt/crest", ""))
        self.assertEqual(second, first)
        self.assertEqual(mock_find.call_count, 1)


if __name__ == "__main__":
    unittest.main()
