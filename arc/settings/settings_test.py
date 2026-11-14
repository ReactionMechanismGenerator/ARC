#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for arc/settings/settings.py.

Path finder-helper tests live in ``arc/settings/external_paths_test.py``
(GoFlow + RitS sister-repo discovery) — they cover the discovery logic
itself; this module only checks that the resulting paths are exposed as
settings globals and that the default ``ts_adapters`` list is correct.
"""

import unittest

from arc.settings import settings as settings_mod


class TestExternalAdapterSettingsExposed(unittest.TestCase):
    """All sister-repo discovery globals must exist on the settings module after import."""

    def test_goflow_globals_are_defined(self):
        for name in ('GOFLOW_PYTHON', 'GOFLOW_REPO_PATH', 'GOFLOW_CKPT_PATH', 'GOFLOW_FEAT_DICT_PATH'):
            self.assertTrue(hasattr(settings_mod, name), f"settings module missing attribute {name}")

    def test_rits_globals_are_defined(self):
        for name in ('RITS_PYTHON', 'RITS_REPO_PATH', 'RITS_CKPT_PATH'):
            self.assertTrue(hasattr(settings_mod, name), f"settings module missing attribute {name}")

    def test_default_ts_adapters_list(self):
        """The default ``ts_adapters`` list is every registered adapter the family map dispatches."""
        self.assertEqual(settings_mod.ts_adapters, ['heuristics', 'linear', 'autotst', 'gcn', 'xtb_gsm', 'orca_neb',
                                                    'kinbot', 'goflow', 'rits'])

    def test_ts_adapters_include_goflow_and_rits_by_default(self):
        """GoFlow and RitS are in the family map, so a default run tries them; an absent engine is skipped."""
        self.assertIn('goflow', settings_mod.ts_adapters)
        self.assertIn('rits', settings_mod.ts_adapters)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
