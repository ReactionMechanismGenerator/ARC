#!/usr/bin/env python3
# encoding: utf-8

"""Tests for the migration-inference helpers extracted from
:mod:`local_geometry`.

These helpers were moved during the cleanup out of
:mod:`arc.job.adapters.ts.linear_utils.local_geometry` into
:mod:`arc.job.adapters.ts.linear_utils.migration_inference`.  The
existing local_geometry test file already exercises the helpers via
the backwards-compatible re-export.  This file proves the *new* import
path also works and that both import paths reference the same
function objects.
"""

import unittest

from arc.job.adapters.ts.linear_utils import local_geometry as legacy_lg
from arc.job.adapters.ts.linear_utils import migration_inference as mi


class TestMigrationInferenceImports(unittest.TestCase):
    """The helpers are importable from the new module path and the
    backwards-compatible re-export from ``local_geometry`` resolves to
    the *same* function objects.
    """

    def test_identify_h_migration_pairs_re_export_identity(self):
        self.assertIs(
            legacy_lg.identify_h_migration_pairs,
            mi.identify_h_migration_pairs,
            msg='legacy local_geometry.identify_h_migration_pairs and the '
                'new migration_inference.identify_h_migration_pairs must '
                'be the SAME object after the cleanup extraction')

    def test_infer_frag_fallback_h_migration_re_export_identity(self):
        self.assertIs(
            legacy_lg.infer_frag_fallback_h_migration,
            mi.infer_frag_fallback_h_migration,
            msg='legacy local_geometry.infer_frag_fallback_h_migration and '
                'the new migration_inference.infer_frag_fallback_h_migration '
                'must be the SAME object after the cleanup extraction')

    def test_private_helpers_removed_from_legacy(self):
        """Verify private helpers are no longer re-exported from local_geometry."""
        self.assertFalse(hasattr(legacy_lg, '_split_into_fragments'),
                         msg='_split_into_fragments should no longer be re-exported')
        self.assertFalse(hasattr(legacy_lg, '_heavy_formula_of_fragment'),
                         msg='_heavy_formula_of_fragment should no longer be re-exported')
        self.assertFalse(hasattr(legacy_lg, '_h_count_of_fragment'),
                         msg='_h_count_of_fragment should no longer be re-exported')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
