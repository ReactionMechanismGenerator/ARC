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

    def test_invalid_env_var_override_does_not_fall_through(self):
        """An invalid ARC_GOFLOW_REPO must not silently fall back to another
        candidate, even when that candidate is valid — the override is
        authoritative and its failure must be reported, not swallowed."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_override = os.path.join(tmp, 'not_a_goflow_repo')
            os.makedirs(bad_override)
            valid_fallback = os.path.join(tmp, 'goflow_lean')
            init_dir = os.path.join(valid_fallback, 'src', 'goflow')
            os.makedirs(init_dir)
            with open(os.path.join(init_dir, '__init__.py'), 'w') as f:
                f.write('')
            with mock.patch.dict(os.environ, {'ARC_GOFLOW_REPO': bad_override, 'HOME': tmp}):
                with mock.patch.object(external_paths, '_goflow_sibling_of_arc',
                                       return_value=valid_fallback):
                    with self.assertLogs(logger='arc', level='WARNING') as cm:
                        result = external_paths.find_goflow_repo()
            self.assertIsNone(result)
            self.assertTrue(any('ARC_GOFLOW_REPO' in line and bad_override in line
                                for line in cm.output))

    def test_valid_env_var_override_wins_over_valid_fallback(self):
        """Both the override and a fallback candidate are valid → the
        override must win (authoritative), not merely "a" valid candidate."""
        with tempfile.TemporaryDirectory() as tmp:
            override = os.path.join(tmp, 'override_goflow')
            override_init = os.path.join(override, 'src', 'goflow')
            os.makedirs(override_init)
            with open(os.path.join(override_init, '__init__.py'), 'w') as f:
                f.write('')
            fallback = os.path.join(tmp, 'goflow_lean')
            fallback_init = os.path.join(fallback, 'src', 'goflow')
            os.makedirs(fallback_init)
            with open(os.path.join(fallback_init, '__init__.py'), 'w') as f:
                f.write('')
            with mock.patch.dict(os.environ, {'ARC_GOFLOW_REPO': override, 'HOME': tmp}):
                with mock.patch.object(external_paths, '_goflow_sibling_of_arc',
                                       return_value=fallback):
                    result = external_paths.find_goflow_repo()
            self.assertEqual(os.path.abspath(override), result)


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

    def test_invalid_env_var_override_does_not_fall_through(self):
        """An undersized ARC_GOFLOW_CKPT must not silently fall back to a
        valid in-repo checkpoint — the override is authoritative."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_override = os.path.join(tmp, 'tiny.ckpt')
            with open(bad_override, 'wb') as f:
                f.write(b'\0' * 45)
            ckpt_path = os.path.join(tmp, 'data', 'RDB7', 'epoch_337.ckpt')
            os.makedirs(os.path.dirname(ckpt_path))
            with open(ckpt_path, 'wb') as f:
                f.write(b'\0' * (1_000_001))
            with mock.patch.dict(os.environ, {'ARC_GOFLOW_CKPT': bad_override}):
                with self.assertLogs(logger='arc', level='WARNING') as cm:
                    result = external_paths.find_goflow_ckpt(repo_path=tmp)
            self.assertIsNone(result)
            self.assertTrue(any('ARC_GOFLOW_CKPT' in line and bad_override in line
                                for line in cm.output))

    def test_valid_env_var_override_wins_over_valid_repo_ckpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = os.path.join(tmp, 'override.ckpt')
            with open(override, 'wb') as f:
                f.write(b'\0' * (1_000_001))
            ckpt_path = os.path.join(tmp, 'data', 'RDB7', 'epoch_337.ckpt')
            os.makedirs(os.path.dirname(ckpt_path))
            with open(ckpt_path, 'wb') as f:
                f.write(b'\0' * (1_000_001))
            with mock.patch.dict(os.environ, {'ARC_GOFLOW_CKPT': override}):
                result = external_paths.find_goflow_ckpt(repo_path=tmp)
            self.assertEqual(os.path.abspath(override), result)


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

    def test_invalid_env_var_override_does_not_fall_through(self):
        """A trivially-small ARC_GOFLOW_FEAT_DICT must not silently fall back
        to a valid in-repo feat-dict — the override is authoritative."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_override = os.path.join(tmp, 'tiny.pkl')
            with open(bad_override, 'wb') as f:
                f.write(b'\0' * 10)
            fd_path = os.path.join(tmp, 'data', 'RDB7', 'feat_dict_organic.pkl')
            os.makedirs(os.path.dirname(fd_path))
            real_dict = {f'feat_{i}': {j: j for j in range(20)} for i in range(20)}
            with open(fd_path, 'wb') as f:
                pickle.dump(real_dict, f)
            with mock.patch.dict(os.environ, {'ARC_GOFLOW_FEAT_DICT': bad_override}):
                with self.assertLogs(logger='arc', level='WARNING') as cm:
                    result = external_paths.find_goflow_feat_dict(repo_path=tmp)
            self.assertIsNone(result)
            self.assertTrue(any('ARC_GOFLOW_FEAT_DICT' in line and bad_override in line
                                for line in cm.output))

    def test_valid_env_var_override_wins_over_valid_repo_feat_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = os.path.join(tmp, 'override_feat_dict.pkl')
            real_dict = {f'feat_{i}': {j: j for j in range(20)} for i in range(20)}
            with open(override, 'wb') as f:
                pickle.dump(real_dict, f)
            fd_path = os.path.join(tmp, 'data', 'RDB7', 'feat_dict_organic.pkl')
            os.makedirs(os.path.dirname(fd_path))
            with open(fd_path, 'wb') as f:
                pickle.dump(real_dict, f)
            with mock.patch.dict(os.environ, {'ARC_GOFLOW_FEAT_DICT': override}):
                result = external_paths.find_goflow_feat_dict(repo_path=tmp)
            self.assertEqual(os.path.abspath(override), result)


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

    def test_invalid_env_var_override_does_not_fall_through(self):
        """An invalid ARC_RITS_REPO must not silently fall back to another
        candidate, even when that candidate is valid."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_override = os.path.join(tmp, 'not_a_rits_repo')
            os.makedirs(bad_override)
            valid_fallback = os.path.join(tmp, 'RitS')
            scripts_dir = os.path.join(valid_fallback, 'scripts')
            os.makedirs(scripts_dir)
            with open(os.path.join(scripts_dir, 'sample_transition_state.py'), 'w') as f:
                f.write('')
            with mock.patch.dict(os.environ, {'ARC_RITS_REPO': bad_override, 'HOME': tmp}):
                with mock.patch.object(external_paths, '_rits_sibling_of_arc',
                                       return_value=valid_fallback):
                    with self.assertLogs(logger='arc', level='WARNING') as cm:
                        result = external_paths.find_rits_repo()
            self.assertIsNone(result)
            self.assertTrue(any('ARC_RITS_REPO' in line and bad_override in line
                                for line in cm.output))

    def test_valid_env_var_override_wins_over_valid_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = os.path.join(tmp, 'override_rits')
            override_scripts = os.path.join(override, 'scripts')
            os.makedirs(override_scripts)
            with open(os.path.join(override_scripts, 'sample_transition_state.py'), 'w') as f:
                f.write('')
            fallback = os.path.join(tmp, 'RitS')
            fallback_scripts = os.path.join(fallback, 'scripts')
            os.makedirs(fallback_scripts)
            with open(os.path.join(fallback_scripts, 'sample_transition_state.py'), 'w') as f:
                f.write('')
            with mock.patch.dict(os.environ, {'ARC_RITS_REPO': override, 'HOME': tmp}):
                with mock.patch.object(external_paths, '_rits_sibling_of_arc',
                                       return_value=fallback):
                    result = external_paths.find_rits_repo()
            self.assertEqual(os.path.abspath(override), result)


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

    def test_invalid_env_var_override_does_not_fall_through(self):
        """A missing-file ARC_RITS_CKPT must not silently fall back to a
        valid in-repo checkpoint — the override is authoritative."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_override = os.path.join(tmp, 'does_not_exist.ckpt')
            ckpt_path = os.path.join(tmp, 'data', 'rits.ckpt')
            os.makedirs(os.path.dirname(ckpt_path))
            with open(ckpt_path, 'wb') as f:
                f.write(b'\0' * 1024)
            with mock.patch.dict(os.environ, {'ARC_RITS_CKPT': bad_override}):
                with self.assertLogs(logger='arc', level='WARNING') as cm:
                    result = external_paths.find_rits_ckpt(repo_path=tmp)
            self.assertIsNone(result)
            self.assertTrue(any('ARC_RITS_CKPT' in line and bad_override in line
                                for line in cm.output))

    def test_valid_env_var_override_wins_over_valid_repo_ckpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = os.path.join(tmp, 'override.ckpt')
            with open(override, 'wb') as f:
                f.write(b'\0' * 1024)
            ckpt_path = os.path.join(tmp, 'data', 'rits.ckpt')
            os.makedirs(os.path.dirname(ckpt_path))
            with open(ckpt_path, 'wb') as f:
                f.write(b'\0' * 1024)
            with mock.patch.dict(os.environ, {'ARC_RITS_CKPT': override}):
                result = external_paths.find_rits_ckpt(repo_path=tmp)
            self.assertEqual(os.path.abspath(override), result)


class TestDeferredWarnings(unittest.TestCase):
    """queue_deferred_warning() / drain_deferred_warnings() — the buffer
    that lets import-time warnings (logged before arc.common.initialize_log()
    attaches handlers to the 'arc' logger) reach arc.log once it flushes."""

    def setUp(self):
        external_paths.drain_deferred_warnings()

    def test_drain_returns_empty_list_when_nothing_queued(self):
        self.assertEqual(external_paths.drain_deferred_warnings(), [])

    def test_queued_message_is_returned_by_drain(self):
        external_paths.queue_deferred_warning('first warning')
        external_paths.queue_deferred_warning('second warning')
        self.assertEqual(external_paths.drain_deferred_warnings(), ['first warning', 'second warning'])

    def test_drain_is_idempotent_and_clears_the_buffer(self):
        external_paths.queue_deferred_warning('only once')
        self.assertEqual(external_paths.drain_deferred_warnings(), ['only once'])
        self.assertEqual(external_paths.drain_deferred_warnings(), [])

    def test_invalid_env_var_override_queues_a_deferred_warning(self):
        """A validation-failure logger.warning() call site must also queue
        the same message, so it isn't lost when logged before handlers exist."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_override = os.path.join(tmp, 'not_a_goflow_repo')
            os.makedirs(bad_override)
            with mock.patch.dict(os.environ, {'ARC_GOFLOW_REPO': bad_override, 'HOME': tmp}):
                with mock.patch.object(external_paths, '_goflow_sibling_of_arc',
                                       return_value=os.path.join(tmp, 'no_fallback')):
                    external_paths.find_goflow_repo()
            queued = external_paths.drain_deferred_warnings()
            self.assertTrue(any('ARC_GOFLOW_REPO' in msg and bad_override in msg for msg in queued))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
