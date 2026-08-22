"""
This module contains unit tests for the arc.imports module.
"""

import logging
import os
import tempfile
import unittest
from unittest.mock import patch

from arc.imports import _report_unusable_overlay, resolve_overridden_dependents
from arc.settings import external_paths

# The finders consult these before ``repo_path``, and CI exports two of them
# (see .github/workflows/ci.yml), so tests that exercise repo-path derivation
# must clear them or they resolve CI's installed checkouts instead.
_EXTERNAL_PATH_ENV_VARS = ('ARC_GOFLOW_REPO', 'ARC_GOFLOW_CKPT', 'ARC_GOFLOW_FEAT_DICT',
                           'ARC_RITS_REPO', 'ARC_RITS_CKPT')


class TestResolveOverriddenDependents(unittest.TestCase):
    """resolve_overridden_dependents() must re-derive a dependent setting
    when its parent was overridden by a local ~/.arc/settings.py but the
    dependent itself was not, so the two never end up describing mismatched
    checkouts/artifacts.

    Mirrors the real call site in arc/imports.py: settings.update(local_settings_dict)
    runs before resolve_overridden_dependents(), so settings[parent_key] already
    reflects the override by the time the function reads it.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        env_patch = patch.dict(os.environ, {var: '' for var in _EXTERNAL_PATH_ENV_VARS})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        for var in _EXTERNAL_PATH_ENV_VARS:
            del os.environ[var]

    def _make_rits_repo_with_ckpt(self, size: int) -> str:
        repo = os.path.join(self.tmpdir.name, 'repo')
        os.makedirs(os.path.join(repo, 'data'), exist_ok=True)
        with open(os.path.join(repo, 'data', 'rits.ckpt'), 'wb') as f:
            f.write(b'0' * size)
        return repo

    def test_parent_overridden_dependent_not_rederives_dependent(self):
        repo = self._make_rits_repo_with_ckpt(10)
        settings = {'RITS_REPO_PATH': '/old/stale/repo', 'RITS_CKPT_PATH': '/old/stale/repo/data/rits.ckpt'}
        local_settings_dict = {'RITS_REPO_PATH': repo}
        settings.update(local_settings_dict)
        resolve_overridden_dependents(settings, local_settings_dict)
        self.assertEqual(settings['RITS_CKPT_PATH'], os.path.abspath(os.path.join(repo, 'data', 'rits.ckpt')))

    def test_both_overridden_explicit_dependent_wins(self):
        repo = self._make_rits_repo_with_ckpt(10)
        explicit_ckpt = os.path.join(self.tmpdir.name, 'explicit.ckpt')
        with open(explicit_ckpt, 'wb') as f:
            f.write(b'0' * 5)
        settings = {'RITS_REPO_PATH': '/old/stale/repo', 'RITS_CKPT_PATH': '/old/stale/repo/data/rits.ckpt'}
        local_settings_dict = {'RITS_REPO_PATH': repo, 'RITS_CKPT_PATH': explicit_ckpt}
        settings.update(local_settings_dict)
        resolve_overridden_dependents(settings, local_settings_dict)
        self.assertEqual(settings['RITS_CKPT_PATH'], explicit_ckpt)

    def test_neither_overridden_both_untouched(self):
        settings = {'RITS_REPO_PATH': '/some/repo', 'RITS_CKPT_PATH': '/some/repo/data/rits.ckpt'}
        local_settings_dict = {}
        settings.update(local_settings_dict)
        resolve_overridden_dependents(settings, local_settings_dict)
        self.assertEqual(settings['RITS_REPO_PATH'], '/some/repo')
        self.assertEqual(settings['RITS_CKPT_PATH'], '/some/repo/data/rits.ckpt')

    def test_parent_overridden_to_path_with_no_valid_ckpt_becomes_none(self):
        empty_repo = os.path.join(self.tmpdir.name, 'empty_repo')
        os.makedirs(empty_repo, exist_ok=True)
        settings = {'RITS_REPO_PATH': '/old/stale/repo', 'RITS_CKPT_PATH': '/old/stale/repo/data/rits.ckpt'}
        local_settings_dict = {'RITS_REPO_PATH': empty_repo}
        settings.update(local_settings_dict)
        resolve_overridden_dependents(settings, local_settings_dict)
        self.assertIsNone(settings['RITS_CKPT_PATH'])

    def test_parent_overridden_to_path_with_no_valid_ckpt_queues_a_warning(self):
        empty_repo = os.path.join(self.tmpdir.name, 'empty_repo')
        os.makedirs(empty_repo, exist_ok=True)
        settings = {'RITS_REPO_PATH': '/old/stale/repo', 'RITS_CKPT_PATH': '/old/stale/repo/data/rits.ckpt'}
        local_settings_dict = {'RITS_REPO_PATH': empty_repo}
        settings.update(local_settings_dict)
        external_paths.drain_deferred_warnings()
        resolve_overridden_dependents(settings, local_settings_dict)
        messages = external_paths.drain_deferred_warnings()
        self.assertEqual(len(messages), 1)
        self.assertIn('RITS_REPO_PATH', messages[0])
        self.assertIn(empty_repo, messages[0])
        self.assertIn('RITS_CKPT_PATH', messages[0])

    def test_goflow_ckpt_and_feat_dict_both_rederived_from_new_repo(self):
        repo = os.path.join(self.tmpdir.name, 'goflow_repo')
        data_dir = os.path.join(repo, 'data', 'RDB7')
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, 'epoch_10.ckpt'), 'wb') as f:
            f.write(b'0' * 2_000_000)
        with open(os.path.join(data_dir, 'feat_dict_organic.pkl'), 'wb') as f:
            f.write(b'0' * 200)
        settings = {
            'GOFLOW_REPO_PATH': '/old/stale/goflow',
            'GOFLOW_CKPT_PATH': '/old/stale/goflow/data/RDB7/epoch_1.ckpt',
            'GOFLOW_FEAT_DICT_PATH': '/old/stale/goflow/data/RDB7/feat_dict_organic.pkl',
        }
        local_settings_dict = {'GOFLOW_REPO_PATH': repo}
        settings.update(local_settings_dict)
        resolve_overridden_dependents(settings, local_settings_dict)
        self.assertEqual(
            settings['GOFLOW_CKPT_PATH'],
            os.path.abspath(os.path.join(data_dir, 'epoch_10.ckpt')),
        )
        self.assertEqual(
            settings['GOFLOW_FEAT_DICT_PATH'],
            os.path.abspath(os.path.join(data_dir, 'feat_dict_organic.pkl')),
        )

    def test_env_var_override_outranks_the_re_derived_repo_path(self):
        """An ARC_* override is authoritative, so re-derivation must not overrule it."""
        repo = self._make_rits_repo_with_ckpt(10)
        env_ckpt = os.path.join(self.tmpdir.name, 'env_rits.ckpt')
        with open(env_ckpt, 'wb') as f:
            f.write(b'0' * 10)
        settings = {'RITS_REPO_PATH': '/old/stale/repo', 'RITS_CKPT_PATH': '/old/stale/repo/data/rits.ckpt'}
        local_settings_dict = {'RITS_REPO_PATH': repo}
        settings.update(local_settings_dict)
        with patch.dict(os.environ, {'ARC_RITS_CKPT': env_ckpt}):
            resolve_overridden_dependents(settings, local_settings_dict)
        self.assertEqual(settings['RITS_CKPT_PATH'], os.path.abspath(env_ckpt))


class TestReportUnusableOverlay(unittest.TestCase):
    """A ~/.arc overlay that cannot be used must say so, not leave ARC quietly on its defaults."""

    OVERLAY = '/home/user/.arc/submit.py'

    def setUp(self):
        """Report every overlay afresh, since the warning is emitted once per file per run."""
        reported = patch('arc.imports._UNUSABLE_OVERLAYS_REPORTED', set())
        reported.start()
        self.addCleanup(reported.stop)
        queued = patch('arc.imports.queue_deferred_warning')
        self.queued = queued.start()
        self.addCleanup(queued.stop)

    def _report(self, module, what='submit_scripts'):
        """Report a failed overlay import, and return the records it logged.

        The levels are read as ``levelno`` rather than ``levelname`` because
        ``arc.common.initialize_log`` renames the level names process-wide, so
        ``levelname`` is 'WARNING' or 'Warning: ' depending on whether anything has
        initialized ARC's log yet.
        """
        error = ImportError(f'No module named {module!r}')
        with self.assertLogs('arc', level='DEBUG') as captured:
            _report_unusable_overlay(self.OVERLAY, module, error, what)
        return captured.records

    def test_a_file_that_did_not_load_is_a_warning(self):
        """Every setting in the file is lost, and nothing else in the run says so."""
        records = self._report('a_module_that_is_not_loaded')
        self.assertEqual([record.levelno for record in records], [logging.WARNING])
        self.assertIn(self.OVERLAY, records[0].getMessage())
        self.assertIn('ImportError', records[0].getMessage())

    def test_a_file_that_did_not_load_is_queued_for_the_log_file(self):
        """The overlay is read before the log exists, so the warning has to survive until it does."""
        self._report('a_module_that_is_not_loaded')
        self.queued.assert_called_once()
        self.assertIn(self.OVERLAY, self.queued.call_args[0][0])

    def test_a_file_that_did_not_load_is_reported_once(self):
        """submit.py is imported from three times, and one broken file is one problem."""
        self._report('a_module_that_is_not_loaded')
        with patch('arc.imports.logger.warning') as warning:
            _report_unusable_overlay(self.OVERLAY, 'a_module_that_is_not_loaded',
                                     ImportError('boom'), 'pipe_submit')
        warning.assert_not_called()

    def test_a_loaded_file_missing_one_name_is_only_a_debug_line(self):
        """Overriding one setting and not the rest is the ordinary way to use an overlay."""
        records = self._report('unittest')
        self.assertEqual([record.levelno for record in records], [logging.DEBUG])
        self.assertIn('submit_scripts', records[0].getMessage())
        self.queued.assert_not_called()


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
