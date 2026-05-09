#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for the pure-Python helpers in
``arc.job.adapters.scripts.goflow_script``.

The script as a whole is intended to run inside ``goflow_env`` (where torch
+ goflow are importable). The helpers tested here are stdlib-only and run in
ARC's main env — that's by design: I/O parsing, YAML serialization, and
sentinel construction must work even before the heavy ML stack is installed.

Heavy paths (Hydra config compose, ckpt load, ODE sampling) are exercised by
the env-gated Tier-2 tests in ``arc/job/adapters/ts/goflow_test.py``.
"""

import datetime
import os
import tempfile
import unittest

import yaml

from arc.job.adapters.scripts.goflow_script import (
    _failed_guess,
    format_xyz_block,
    read_xyz_positions,
    save_yaml_file_local,
    string_representer,
    write_multi_frame_xyz,
)


class TestReadXyzPositions(unittest.TestCase):
    """`read_xyz_positions(path)` parses a single-frame plain XYZ → (N, 3) array."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile('w', suffix='.xyz', delete=False)

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_parses_three_atom_xyz(self):
        self.tmp.write('3\n# comment\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0\n')
        self.tmp.close()
        pos = read_xyz_positions(self.tmp.name)
        self.assertEqual(pos.shape, (3, 3))
        self.assertAlmostEqual(pos[1, 0], 1.0)
        self.assertAlmostEqual(pos[2, 1], 1.0)

    def test_handles_trailing_blank_lines(self):
        self.tmp.write('2\n\nN 0.5 0.5 0.5\nO 1.5 0.5 0.5\n\n\n')
        self.tmp.close()
        pos = read_xyz_positions(self.tmp.name)
        self.assertEqual(pos.shape, (2, 3))

    def test_raises_on_truncated_file(self):
        """A header declaring more atoms than rows present must raise, not silently return fewer coordinates."""
        self.tmp.write('5\n# comment\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n')
        self.tmp.close()
        with self.assertRaisesRegex(ValueError, 'truncated'):
            read_xyz_positions(self.tmp.name)

    def test_raises_on_malformed_row(self):
        self.tmp.write('2\n# comment\nC 0.0 0.0 0.0\nH 1.0 0.0\n')
        self.tmp.close()
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            read_xyz_positions(self.tmp.name)

    def test_raises_on_empty_file(self):
        self.tmp.write('\n\n')
        self.tmp.close()
        with self.assertRaisesRegex(ValueError, 'empty'):
            read_xyz_positions(self.tmp.name)


class TestFormatXyzBlock(unittest.TestCase):
    """`format_xyz_block(symbols, pos)` returns body-only XYZ (no header)."""

    def test_emits_no_header_no_comment(self):
        block = format_xyz_block(['H', 'O', 'H'],
                                 [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        lines = block.strip().split('\n')
        self.assertEqual(len(lines), 3)
        # First token of every line is the symbol.
        self.assertEqual(lines[0].split()[0], 'H')
        self.assertEqual(lines[1].split()[0], 'O')
        # Should NOT start with a numeric atom-count header.
        self.assertFalse(lines[0].strip().isdigit())


class TestWriteMultiFrameXyz(unittest.TestCase):
    """`write_multi_frame_xyz(path, symbols, pos_S_N_3)` round-trips."""

    def test_round_trips_two_frames(self):
        symbols = ['C', 'H']
        pos_S_N_3 = [[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
                     [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]]
        with tempfile.NamedTemporaryFile('w', suffix='.xyz', delete=False) as f:
            path = f.name
        try:
            write_multi_frame_xyz(path, symbols, pos_S_N_3)
            with open(path) as fh:
                lines = [ln.rstrip('\n') for ln in fh]
            # Each frame: 1 atom-count line + 1 comment line + 2 atom lines = 4 lines.
            # 2 frames = 8 lines (allowing trailing newlines).
            self.assertGreaterEqual(len(lines), 8)
            atom_count_lines = [ln for ln in lines if ln.strip().isdigit()]
            self.assertEqual(len(atom_count_lines), 2)
            self.assertEqual(int(atom_count_lines[0]), 2)
        finally:
            os.unlink(path)


class TestFailedGuessSentinel(unittest.TestCase):
    """`_failed_guess(elapsed)` returns the standard failure dict shape."""

    def test_sentinel_has_required_keys_and_method_name(self):
        sentinel = _failed_guess(datetime.timedelta(seconds=1), index=0)
        for key in ('method', 'method_direction', 'method_index',
                    'initial_xyz', 'success', 'execution_time'):
            self.assertIn(key, sentinel)
        self.assertEqual(sentinel['method'], 'GoFlow')
        self.assertFalse(sentinel['success'])
        self.assertIsNone(sentinel['initial_xyz'])


class TestSaveYamlFileBlockLiteral(unittest.TestCase):
    """`save_yaml_file_local(path, content)` writes multi-line strings as block literals."""

    def test_multi_line_xyz_string_uses_block_literal_style(self):
        content = [{'method': 'GoFlow', 'method_index': 0,
                    'initial_xyz': 'C 0 0 0\nH 1 0 0\nH 0 1 0',
                    'success': True}]
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            path = f.name
        try:
            save_yaml_file_local(path, content)
            with open(path) as fh:
                text = fh.read()
            # Block literal style uses a `|` indicator on the value line.
            self.assertIn('initial_xyz: |', text)
            with open(path) as fh:
                self.assertEqual(yaml.safe_load(fh), content)
        finally:
            os.unlink(path)

    def test_dump_does_not_pollute_global_default_dumper(self):
        """The block-literal str representer must be registered on SafeDumper only,
        leaving the global yaml.Dumper untouched for other modules."""
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            path = f.name
        try:
            save_yaml_file_local(path, [{'initial_xyz': 'C 0 0 0\nH 1 0 0'}])
            self.assertIsNot(yaml.Dumper.yaml_representers.get(str), string_representer)
            self.assertIs(yaml.SafeDumper.yaml_representers.get(str), string_representer)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
