#!/usr/bin/env python3
# encoding: utf-8

"""Tests for the standalone TCKDB upload-sweep CLI."""

import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import yaml

from arc.tckdb.cli import main, parse_args
from arc.tckdb.adapter_test import _reaction_output_doc


# ---------------------------------------------------------------------------
# Stubs and fixtures
# ---------------------------------------------------------------------------


class _StubAdapter:
    """Captures which submit_* methods the sweep called and with what."""

    def __init__(self):
        self.reaction_calls: list[str] = []
        self.species_calls: list[str] = []
        self.conformer_calls: list[str] = []

    def submit_computed_reaction_from_output(
        self, *, output_doc, reaction_record, is_partial=False,
    ):
        self.reaction_calls.append(reaction_record.get('label', ''))
        return None

    def submit_computed_species_from_output(self, *, output_doc, species_record):
        self.species_calls.append(species_record.get('label', ''))
        return None

    def submit_from_output(self, *, output_doc, species_record):
        self.conformer_calls.append(species_record.get('label', ''))
        return None


def _make_project(tmp_dir, *, with_output=True, project_dir_in_input=None):
    """Lay out a fake ARC project with input.yml + (optionally) output.yml."""
    proj = os.path.join(tmp_dir, 'proj')
    os.makedirs(proj)
    if with_output:
        out_dir = os.path.join(proj, 'output')
        os.makedirs(out_dir)
        doc = _reaction_output_doc()
        for s in doc['species']:
            s['converged'] = True
        for ts in doc['transition_states']:
            ts['converged'] = True
        with open(os.path.join(out_dir, 'output.yml'), 'w') as fh:
            yaml.safe_dump(doc, fh)
    input_path = os.path.join(proj, 'input.yml')
    body = {
        'project': 'cli-test',
        'tckdb': {
            'enabled': True,
            'base_url': 'http://localhost:8000/api/v1',
            'upload_mode': 'computed_reaction',
            'upload': False,
        },
    }
    if project_dir_in_input:
        body['project_directory'] = project_dir_in_input
    with open(input_path, 'w') as fh:
        yaml.safe_dump(body, fh)
    return proj, input_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestArgParsing(unittest.TestCase):
    def test_positional_input_file_required(self):
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_default_flags(self):
        args = parse_args(['/tmp/foo.yml'])
        self.assertEqual(args.input_file, '/tmp/foo.yml')
        self.assertIsNone(args.project_directory)
        self.assertIsNone(args.upload_mode)
        self.assertFalse(args.offline)

    def test_all_overrides(self):
        args = parse_args([
            'input.yml', '-p', '/some/where', '--offline',
            '--upload-mode', 'computed_reaction',
        ])
        self.assertEqual(args.project_directory, '/some/where')
        self.assertTrue(args.offline)
        self.assertEqual(args.upload_mode, 'computed_reaction')

    def test_invalid_upload_mode_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(['input.yml', '--upload-mode', 'nonsense'])


class TestCLIDispatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='arc-cli-')
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _run(self, argv, *, adapter=None):
        adapter = adapter or _StubAdapter()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = main(argv, adapter_factory=lambda cfg, pdir: adapter)
        return rc, stdout.getvalue(), stderr.getvalue(), adapter

    # ---------------- 1: happy path
    def test_dispatches_to_reaction_sweep_in_reaction_mode(self):
        proj, input_path = _make_project(self.tmp)
        rc, stdout, _, adapter = self._run([input_path])
        self.assertEqual(rc, 0)
        self.assertEqual(adapter.reaction_calls, ['CHO + CH4 <=> CH2O + CH3'])
        self.assertEqual(adapter.species_calls, [])
        self.assertEqual(adapter.conformer_calls, [])
        self.assertIn('computed-reaction bundle', stdout)

    # ---------------- 2: --upload-mode override
    def test_upload_mode_override_redirects_dispatch(self):
        proj, input_path = _make_project(self.tmp)
        rc, _, _, adapter = self._run(
            [input_path, '--upload-mode', 'computed_species']
        )
        self.assertEqual(rc, 0)
        # Now species path is hit, not reaction
        self.assertEqual(adapter.reaction_calls, [])
        self.assertEqual(len(adapter.species_calls), 4)
        self.assertEqual(adapter.conformer_calls, [])

    # ---------------- 3: --offline forces config.upload=False
    def test_offline_flag_overrides_config_upload(self):
        proj, input_path = _make_project(self.tmp)
        # Bake upload: true into the input.yml so we can prove --offline overrides
        with open(input_path) as fh:
            body = yaml.safe_load(fh)
        body['tckdb']['upload'] = True
        with open(input_path, 'w') as fh:
            yaml.safe_dump(body, fh)

        captured_cfgs: list = []

        def _factory(cfg, pdir):
            captured_cfgs.append(cfg)
            return _StubAdapter()

        rc = main([input_path, '--offline'], adapter_factory=_factory)
        self.assertEqual(rc, 0)
        self.assertEqual(len(captured_cfgs), 1)
        self.assertFalse(
            captured_cfgs[0].upload,
            'config.upload should be False after --offline override',
        )

    # ---------------- 4: project-directory resolution order
    def test_project_directory_cli_flag_wins(self):
        proj, input_path = _make_project(self.tmp)
        # Make a different project dir with its own output.yml
        other = os.path.join(self.tmp, 'other')
        os.makedirs(os.path.join(other, 'output'))
        doc = _reaction_output_doc()
        # Only one species converged in this output, distinguishable from default
        doc['reactions'][0]['label'] = 'OTHER_RXN'
        with open(os.path.join(other, 'output', 'output.yml'), 'w') as fh:
            yaml.safe_dump(doc, fh)
        rc, _, _, adapter = self._run([input_path, '-p', other])
        self.assertEqual(rc, 0)
        self.assertEqual(adapter.reaction_calls, ['OTHER_RXN'])

    def test_project_directory_from_input_yml(self):
        proj, input_path = _make_project(self.tmp)
        # Move output to a sibling dir and point input.yml at it
        sibling = os.path.join(self.tmp, 'sibling')
        shutil.move(os.path.join(proj, 'output'), os.path.join(sibling, 'output'))
        with open(input_path) as fh:
            body = yaml.safe_load(fh)
        body['project_directory'] = sibling
        with open(input_path, 'w') as fh:
            yaml.safe_dump(body, fh)
        rc, stdout, _, adapter = self._run([input_path])
        self.assertEqual(rc, 0)
        self.assertEqual(adapter.reaction_calls, ['CHO + CH4 <=> CH2O + CH3'])

    def test_project_directory_falls_back_to_input_dir(self):
        proj, input_path = _make_project(self.tmp)
        rc, stdout, _, adapter = self._run([input_path])
        self.assertEqual(rc, 0)
        self.assertIn(proj, stdout)

    # ---------------- 5: missing files / bad config exit codes
    def test_missing_input_file_exits_2(self):
        rc, _, stderr, _ = self._run(['/nonexistent/input.yml'])
        self.assertEqual(rc, 2)
        self.assertIn('input file not found', stderr)

    def test_missing_project_directory_exits_2(self):
        proj, input_path = _make_project(self.tmp)
        rc, _, stderr, _ = self._run([input_path, '-p', '/nonexistent/proj'])
        self.assertEqual(rc, 2)
        self.assertIn('project directory does not exist', stderr)

    def test_no_tckdb_block_exits_2(self):
        proj, input_path = _make_project(self.tmp)
        with open(input_path, 'w') as fh:
            yaml.safe_dump({'project': 'no-tckdb-here'}, fh)
        rc, _, stderr, _ = self._run([input_path])
        self.assertEqual(rc, 2)
        self.assertIn('no tckdb block', stderr)

    def test_disabled_tckdb_exits_2(self):
        proj, input_path = _make_project(self.tmp)
        with open(input_path) as fh:
            body = yaml.safe_load(fh)
        body['tckdb']['enabled'] = False
        with open(input_path, 'w') as fh:
            yaml.safe_dump(body, fh)
        rc, _, stderr, _ = self._run([input_path])
        # enabled: false → from_dict returns None, treated same as missing block
        self.assertEqual(rc, 2)

    def test_invalid_tckdb_config_exits_2(self):
        proj, input_path = _make_project(self.tmp)
        with open(input_path) as fh:
            body = yaml.safe_load(fh)
        # Drop required base_url; from_dict should raise
        body['tckdb'].pop('base_url')
        with open(input_path, 'w') as fh:
            yaml.safe_dump(body, fh)
        rc, _, stderr, _ = self._run([input_path])
        self.assertEqual(rc, 2)
        self.assertIn('invalid tckdb config', stderr)

    # ---------------- 6: missing output.yml is a soft skip, not an error
    def test_missing_output_yml_is_soft_skip(self):
        proj, input_path = _make_project(self.tmp, with_output=False)
        rc, stdout, _, adapter = self._run([input_path])
        # The sweep itself prints "TCKDB upload skipped" but returns 0
        # — the run-not-completed case shouldn't be a hard failure.
        self.assertEqual(rc, 0)
        self.assertIn('not found', stdout)
        self.assertEqual(adapter.reaction_calls, [])


if __name__ == '__main__':
    unittest.main()
