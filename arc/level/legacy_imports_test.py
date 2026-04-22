#!/usr/bin/env python3
# encoding: utf-8

"""
Backward-compatibility tests for the ``arc.level`` package.

These tests assert that every public symbol that used to live in the legacy
``arc/level.py`` module is still importable from ``arc.level`` after the package
relocation. They guard the public surface so an accidental re-organisation of
the new package internals cannot break the existing 50+ external call sites.
"""

import importlib
import unittest


class TestLegacyArcLevelImports(unittest.TestCase):
    """Verify the public surface of ``arc.level`` is preserved."""

    def test_from_arc_level_import_Level(self):
        """``from arc.level import Level`` resolves to the legacy class."""
        from arc.level import Level

        instance = Level(method="b3lyp", basis="def2tzvp")
        self.assertEqual(instance.method, "b3lyp")
        self.assertEqual(instance.basis, "def2tzvp")

    def test_from_arc_level_import_assign_frequency_scale_factor(self):
        """``assign_frequency_scale_factor`` is still re-exported."""
        from arc.level import assign_frequency_scale_factor

        self.assertTrue(callable(assign_frequency_scale_factor))

    def test_from_arc_level_import_module_singletons(self):
        """``levels_ess`` and ``supported_ess`` are still accessible."""
        from arc.level import levels_ess, supported_ess

        self.assertIsNotNone(levels_ess)
        self.assertIsNotNone(supported_ess)

    def test_import_arc_level_as_module(self):
        """``import arc.level`` succeeds (the side-effect import in arc/__init__.py).

        Loaded via importlib so this test file's source contains only
        ``from arc.level import …`` statements (CodeQL flags mixing both
        styles in the same module).
        """
        module = importlib.import_module("arc.level")
        self.assertTrue(hasattr(module, "Level"))
        self.assertTrue(hasattr(module, "assign_frequency_scale_factor"))

    def test_alias_import(self):
        """``from arc.level import Level as Lvl`` keeps working (used in tests)."""
        from arc.level import Level as Lvl

        self.assertIs(Lvl.__name__, "Level")

    def test_level_class_is_a_real_class(self):
        """Sanity check: re-export is the actual class, not a re-binding."""
        from arc.level import Level
        from arc.level.level import Level as LevelDirect

        self.assertIs(Level, LevelDirect)


if __name__ == "__main__":
    unittest.main()
