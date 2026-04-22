#!/usr/bin/env python3
# encoding: utf-8

"""
Tests that every ``examples/Composite/*/input.yml`` example is valid YAML and
that its ``sp_composite`` block (or per-species ``sp_composite`` entries)
builds a valid :class:`CompositeProtocol` via
:meth:`CompositeProtocol.from_user_input`. Keeps the docs + examples honest.
"""

import glob
import os
import unittest

import yaml

from arc.common import ARC_PATH
from arc.level.protocol import CompositeProtocol


EXAMPLES_DIR = os.path.join(ARC_PATH, "examples", "Composite")


class TestCompositeExamples(unittest.TestCase):
    """Parse every shipped example and validate its sp_composite payload."""

    def _example_files(self):
        pattern = os.path.join(EXAMPLES_DIR, "*", "input.yml")
        return sorted(glob.glob(pattern))

    def test_examples_directory_ships_at_least_four_inputs(self):
        self.assertGreaterEqual(len(self._example_files()), 4)

    def test_examples_readme_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(EXAMPLES_DIR, "README.md")))

    def test_every_example_is_valid_yaml(self):
        for path in self._example_files():
            with self.subTest(path=path):
                with open(path, "r") as fh:
                    data = yaml.safe_load(fh)
                self.assertIsInstance(data, dict)
                self.assertIn("project", data)
                self.assertIn("species", data)

    def test_every_project_level_sp_composite_builds(self):
        """Project-level ``sp_composite`` (if present) is parseable."""
        for path in self._example_files():
            with open(path, "r") as fh:
                data = yaml.safe_load(fh)
            sp = data.get("sp_composite")
            if sp is None:
                continue
            with self.subTest(path=path):
                protocol = CompositeProtocol.from_user_input(sp)
                self.assertIsInstance(protocol, CompositeProtocol)

    def test_every_species_sp_composite_builds_if_explicit(self):
        """Per-species ``sp_composite`` (string/dict, not null) is parseable."""
        for path in self._example_files():
            with open(path, "r") as fh:
                data = yaml.safe_load(fh)
            for spc in data.get("species", []):
                sp = spc.get("sp_composite", "__missing__")
                if sp == "__missing__":
                    continue
                if sp is None:
                    continue
                with self.subTest(path=path, label=spc.get("label")):
                    protocol = CompositeProtocol.from_user_input(sp)
                    self.assertIsInstance(protocol, CompositeProtocol)

    def test_all_four_forms_covered(self):
        """Each of the four documented YAML forms must appear at least once."""
        form1 = form2 = form3 = form4 = False
        for path in self._example_files():
            with open(path, "r") as fh:
                data = yaml.safe_load(fh)
            sp = data.get("sp_composite")
            if isinstance(sp, str):
                form1 = True
            elif isinstance(sp, dict) and "preset" in sp:
                form2 = True
            elif isinstance(sp, dict) and "base" in sp:
                form3 = True
            for spc in data.get("species", []):
                if "sp_composite" in spc:
                    form4 = True
        self.assertTrue(form1, "Form 1 (preset by name) not demonstrated.")
        self.assertTrue(form2, "Form 2 (preset + override) not demonstrated.")
        self.assertTrue(form3, "Form 3 (fully explicit recipe) not demonstrated.")
        self.assertTrue(form4, "Form 4 (per-species override) not demonstrated.")

    def test_explicit_recipe_example_includes_cbs_extrapolation(self):
        path = os.path.join(EXAMPLES_DIR, "explicit_fpa", "input.yml")
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        corrections = data["sp_composite"]["corrections"]
        term_types = {c["type"] for c in corrections}
        self.assertIn("cbs_extrapolation", term_types)


if __name__ == "__main__":
    unittest.main()
