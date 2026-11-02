#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for the pure-Python helpers in
``arc.job.adapters.scripts.rits_script``.

The script as a whole is intended to run inside ``rits_env`` (where torch
+ megalodon are importable). The helpers tested here are stdlib-only and
run in ARC's main env — that's by design: I/O parsing, multi-frame XYZ
splitting, and YAML serialization must work even before the heavy ML
stack is installed.

Heavy paths (``run_rits`` orchestration with the real subprocess) are
exercised by the env-gated Tier-2 tests in
``arc/job/adapters/ts/rits_test.py``.
"""

import os
import shutil
import tempfile
import unittest

from arc.job.adapters.scripts.rits_script import parse_multi_frame_xyz, read_yaml_file, save_yaml_file


class TestRitSScriptParser(unittest.TestCase):
    """Direct unit tests for arc/job/adapters/scripts/rits_script.py:parse_multi_frame_xyz."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix='rits_script_parser_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _write(self, name: str, body: str) -> str:
        path = os.path.join(self.tmp_dir, name)
        with open(path, 'w') as f:
            f.write(body)
        return path

    def test_single_frame_xyz(self):
        body = "3\n\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH -1.0 0.0 0.0\n"
        frames = parse_multi_frame_xyz(self._write('one.xyz', body))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].splitlines()[0].split()[0], 'C')

    def test_multi_frame_xyz(self):
        body = ("3\n\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH -1.0 0.0 0.0\n"
                "3\n\nC 0.1 0.0 0.0\nH 1.1 0.0 0.0\nH -0.9 0.0 0.0\n")
        frames = parse_multi_frame_xyz(self._write('two.xyz', body))
        self.assertEqual(len(frames), 2)
        # Frame 0 starts at the origin; frame 1 is shifted by +0.1 in x
        self.assertAlmostEqual(float(frames[0].splitlines()[0].split()[1]), 0.0)
        self.assertAlmostEqual(float(frames[1].splitlines()[0].split()[1]), 0.1)

    def test_missing_file_returns_empty_list(self):
        frames = parse_multi_frame_xyz(os.path.join(self.tmp_dir, 'nope.xyz'))
        self.assertEqual(frames, list())

    def test_garbage_does_not_loop_forever(self):
        body = "this is not an xyz\nat all\n"
        frames = parse_multi_frame_xyz(self._write('garbage.xyz', body))
        self.assertEqual(frames, list())

    def test_yaml_round_trip_uses_safe_dumper(self):
        """save_yaml_file + read_yaml_file must round-trip a TSGuess-shaped list
        (incl. multi-line xyz blocks) using only safe YAML constructs."""
        content = [{'method': 'RitS', 'method_direction': 'F', 'method_index': 0,
                    'initial_xyz': 'C 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH -1.0 0.0 0.0',
                    'success': True, 'execution_time': '0:00:01.5'}]
        path = os.path.join(self.tmp_dir, 'round_trip.yml')
        save_yaml_file(path=path, content=content)
        self.assertEqual(read_yaml_file(path), content)
        with open(path) as f:
            raw = f.read()
        self.assertNotIn('!!python', raw)
        self.assertIn('|', raw)  # multi-line strings use a block literal


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
