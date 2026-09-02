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
    find_crest_executable,
    find_highest_version_in_directory,
    parse_version,
)


class TestCrestSettingsUtils(unittest.TestCase):

    def _make_executable(self, path: str):
        with open(path, "w") as f:
            f.write("#!/bin/bash\n")
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IXUSR)

    def test_parse_version(self):
        self.assertEqual(parse_version("crest-3.0.2"), (3, 0, 2))
        self.assertEqual(parse_version("v212"), (2, 1, 2))
        self.assertEqual(parse_version("version-2.1"), (2, 1, 0))
        self.assertEqual(parse_version("foo"), (0, 0, 0))

    def test_find_highest_version_in_directory(self):
        with tempfile.TemporaryDirectory() as td:
            low = os.path.join(td, "crest-2.1")
            high = os.path.join(td, "crest-3.0.2")
            os.makedirs(low)
            os.makedirs(high)
            self._make_executable(os.path.join(low, "crest"))
            self._make_executable(os.path.join(high, "crest"))

            found = find_highest_version_in_directory(td, "crest")
            self.assertEqual(found, os.path.join(high, "crest"))

    def test_find_crest_executable_prefers_standalone(self):
        with tempfile.TemporaryDirectory() as td:
            standalone = os.path.join(td, "crest-3.0.2")
            os.makedirs(standalone)
            standalone_crest = os.path.join(standalone, "crest")
            self._make_executable(standalone_crest)

            with patch.dict(os.environ, {"ARC_CREST_STANDALONE_DIR": td}, clear=False):
                path, env_cmd = find_crest_executable()
            self.assertEqual(path, standalone_crest)
            self.assertEqual(env_cmd, "")

    def test_find_crest_executable_env_detection(self):
        with tempfile.TemporaryDirectory() as td:
            fake_home = os.path.join(td, "home")
            os.makedirs(fake_home)
            crest_path = os.path.join(fake_home, "miniforge3", "envs", "crest_env", "bin", "crest")
            os.makedirs(os.path.dirname(crest_path), exist_ok=True)
            self._make_executable(crest_path)

            with patch("arc.settings.crest.os.path.expanduser", return_value=fake_home):
                with patch("arc.settings.crest.sys.executable", os.path.join(td, "python")):
                    with patch("arc.settings.crest.shutil.which", return_value=None):
                        path, env_cmd = find_crest_executable()
            self.assertEqual(path, crest_path)
            self.assertIn("conda activate crest_env", env_cmd)


if __name__ == "__main__":
    unittest.main()

