#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for the filesystem-discovery helpers in
``arc/settings/external_paths.py``.

Each test fully isolates filesystem + env-var state so it doesn't accidentally
match the developer's real ~/Code/goflow_lean or ~/Code/RitS checkout if one
exists.
"""

import os
import pickle
import tempfile
import unittest
from unittest import mock

from arc.settings import external_paths


class TestFindGoFlowRepo(unittest.TestCase):
    """find_goflow_repo() — locates a goflow_lean source checkout."""

    def test_returns_none_when_no_candidates_exist(self):
        """No env var, no shipped path on disk → None."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {'HOME': tmp}, clear=False):
                os.environ.pop('ARC_GOFLOW_REPO', None)
                with mock.patch.object(external_paths, '_goflow_sibling_of_arc',
                                       return_value=os.path.join(tmp, 'definitely_no_goflow_here')):
                    self.assertIsNone(external_paths.find_goflow_repo())

    def test_uses_env_var_override_when_repo_is_real(self):
        """ARC_GOFLOW_REPO points at a dir with src/goflow/__init__.py → returns it."""
        with tempfile.TemporaryDirectory() as tmp:
            init_dir = os.path.join(tmp, 'src', 'goflow')
            os.makedirs(init_dir)
            with open(os.path.join(init_dir, '__init__.py'), 'w') as f:
                f.write('')
            with mock.patch.dict(os.environ, {'ARC_GOFLOW_REPO': tmp}):
                self.assertEqual(os.path.abspath(tmp), external_paths.find_goflow_repo())

    def test_env_var_pointing_at_dir_without_src_goflow_returns_none(self):
        """ARC_GOFLOW_REPO points at the wrong directory → not "found" → None."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {'ARC_GOFLOW_REPO': tmp, 'HOME': tmp}):
                with mock.patch.object(external_paths, '_goflow_sibling_of_arc',
                                       return_value=os.path.join(tmp, 'no_goflow')):
                    self.assertIsNone(external_paths.find_goflow_repo())


class TestFindGoFlowCkpt(unittest.TestCase):
    """find_goflow_ckpt() — locates the pretrained checkpoint file."""

    def test_returns_none_when_no_repo_and_no_env_var(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ARC_GOFLOW_CKPT', None)
            self.assertIsNone(external_paths.find_goflow_ckpt(repo_path=None))

    def test_uses_env_var_when_set_and_file_is_large_enough(self):
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            f.write(b'\0' * (1_000_001))  # >= 1 MB
            ckpt_path = f.name
        try:
            with mock.patch.dict(os.environ, {'ARC_GOFLOW_CKPT': ckpt_path}):
                self.assertEqual(os.path.abspath(ckpt_path),
                                 external_paths.find_goflow_ckpt(repo_path=None))
        finally:
            os.unlink(ckpt_path)

    def test_rejects_undersized_ckpt_file_45_bytes_placeholder(self):
        """The 45-byte LFS-pointer file shipped in goflow_lean must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, 'data', 'RDB7', 'epoch_337.ckpt')
            os.makedirs(os.path.dirname(ckpt_path))
            with open(ckpt_path, 'wb') as f:
                f.write(b'\0' * 45)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop('ARC_GOFLOW_CKPT', None)
                self.assertIsNone(external_paths.find_goflow_ckpt(repo_path=tmp))

    def test_accepts_ckpt_in_repo_when_size_is_realistic(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, 'data', 'RDB7', 'epoch_337.ckpt')
            os.makedirs(os.path.dirname(ckpt_path))
            with open(ckpt_path, 'wb') as f:
                f.write(b'\0' * (1_000_001))
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop('ARC_GOFLOW_CKPT', None)
                self.assertEqual(os.path.abspath(ckpt_path),
                                 external_paths.find_goflow_ckpt(repo_path=tmp))


class TestFindGoFlowFeatDict(unittest.TestCase):
    """find_goflow_feat_dict() — locates the atom-feature codebook pickle."""

    def test_returns_none_when_no_repo_and_no_env_var(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ARC_GOFLOW_FEAT_DICT', None)
            self.assertIsNone(external_paths.find_goflow_feat_dict(repo_path=None))

    def test_rejects_feat_dict_file_below_size_threshold(self):
        """Trivially-small feat_dict files (<100 B) must be rejected by the size guard.

        Note: the 387-byte ``feat_dict_organic.pkl`` shipped in goflow_lean@main
        is a real (small) pickle and is *accepted*; the size guard only catches
        clearly-empty stubs."""
        with tempfile.TemporaryDirectory() as tmp:
            fd_path = os.path.join(tmp, 'data', 'RDB7', 'feat_dict_organic.pkl')
            os.makedirs(os.path.dirname(fd_path))
            with open(fd_path, 'wb') as f:
                f.write(b'\0' * 50)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop('ARC_GOFLOW_FEAT_DICT', None)
                self.assertIsNone(external_paths.find_goflow_feat_dict(repo_path=tmp))

    def test_accepts_real_pickle_when_above_size_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            fd_path = os.path.join(tmp, 'data', 'RDB7', 'feat_dict_organic.pkl')
            os.makedirs(os.path.dirname(fd_path))
            real_dict = {f'feat_{i}': {j: j for j in range(20)} for i in range(20)}
            with open(fd_path, 'wb') as f:
                pickle.dump(real_dict, f)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop('ARC_GOFLOW_FEAT_DICT', None)
                self.assertEqual(os.path.abspath(fd_path),
                                 external_paths.find_goflow_feat_dict(repo_path=tmp))


class TestFindRitsRepo(unittest.TestCase):
    """find_rits_repo() — locates a RitS source checkout."""

    def test_returns_none_when_no_candidates_exist(self):
        """No env var, no ~/Code/RitS, no sibling-of-ARC → None."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {'HOME': tmp}, clear=False):
                os.environ.pop('ARC_RITS_REPO', None)
                with mock.patch.object(external_paths, '_rits_sibling_of_arc',
                                       return_value=os.path.join(tmp, 'definitely_no_rits_here')):
                    self.assertIsNone(external_paths.find_rits_repo())

    def test_uses_env_var_override_when_repo_is_real(self):
        """ARC_RITS_REPO points at a dir with scripts/sample_transition_state.py → returns it."""
        with tempfile.TemporaryDirectory() as tmp:
            scripts_dir = os.path.join(tmp, 'scripts')
            os.makedirs(scripts_dir)
            with open(os.path.join(scripts_dir, 'sample_transition_state.py'), 'w') as f:
                f.write('')
            with mock.patch.dict(os.environ, {'ARC_RITS_REPO': tmp}):
                self.assertEqual(os.path.abspath(tmp), external_paths.find_rits_repo())

    def test_env_var_pointing_at_dir_without_sampler_returns_none(self):
        """ARC_RITS_REPO points at the wrong directory → not "found" → None."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {'ARC_RITS_REPO': tmp, 'HOME': tmp}):
                with mock.patch.object(external_paths, '_rits_sibling_of_arc',
                                       return_value=os.path.join(tmp, 'no_rits')):
                    self.assertIsNone(external_paths.find_rits_repo())

    def test_finds_repo_via_sibling_of_arc_fallback(self):
        """No env var; sibling-of-ARC contains a valid checkout → returns it."""
        with tempfile.TemporaryDirectory() as tmp:
            home = os.path.join(tmp, 'home')
            os.makedirs(home)
            sibling = os.path.join(tmp, 'RitS')
            scripts_dir = os.path.join(sibling, 'scripts')
            os.makedirs(scripts_dir)
            with open(os.path.join(scripts_dir, 'sample_transition_state.py'), 'w') as f:
                f.write('')
            with mock.patch.dict(os.environ, {'HOME': home}, clear=False):
                os.environ.pop('ARC_RITS_REPO', None)
                with mock.patch.object(external_paths, '_rits_sibling_of_arc',
                                       return_value=sibling):
                    self.assertEqual(os.path.abspath(sibling), external_paths.find_rits_repo())


class TestFindRitsCkpt(unittest.TestCase):
    """find_rits_ckpt() — locates the pretrained RitS checkpoint."""

    def test_returns_none_when_no_repo_and_no_env_var(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ARC_RITS_CKPT', None)
            self.assertIsNone(external_paths.find_rits_ckpt(repo_path=None))

    def test_uses_env_var_override_when_file_exists(self):
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            f.write(b'\0' * 1024)
            ckpt_path = f.name
        try:
            with mock.patch.dict(os.environ, {'ARC_RITS_CKPT': ckpt_path}):
                self.assertEqual(os.path.abspath(ckpt_path),
                                 external_paths.find_rits_ckpt(repo_path=None))
        finally:
            os.unlink(ckpt_path)

    def test_env_var_pointing_at_missing_file_returns_none(self):
        with mock.patch.dict(os.environ, {'ARC_RITS_CKPT': '/nonexistent/path/to/rits.ckpt'}):
            self.assertIsNone(external_paths.find_rits_ckpt(repo_path=None))

    def test_finds_ckpt_at_repo_data_rits_ckpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, 'data', 'rits.ckpt')
            os.makedirs(os.path.dirname(ckpt_path))
            with open(ckpt_path, 'wb') as f:
                f.write(b'\0' * 1024)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop('ARC_RITS_CKPT', None)
                self.assertEqual(os.path.abspath(ckpt_path),
                                 external_paths.find_rits_ckpt(repo_path=tmp))

    def test_returns_none_when_repo_lacks_data_rits_ckpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop('ARC_RITS_CKPT', None)
                self.assertIsNone(external_paths.find_rits_ckpt(repo_path=tmp))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
