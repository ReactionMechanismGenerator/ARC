#!/usr/bin/env python3
# encoding: utf-8

"""Unit tests for arc.tckdb.idempotency."""

import re
import unittest

from arc.tckdb.idempotency import IdempotencyInputs, build_idempotency_key


_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._:\-]{16,200}$")


def _inputs(**overrides):
    base = dict(
        project_label="projA",
        species_label="ethanol",
        conformer_label="conf0",
        payload_kind="conformer_calculation",
        payload={"hello": "world"},
    )
    base.update(overrides)
    return IdempotencyInputs.from_payload(**base)


class TestIdempotency(unittest.TestCase):

    def test_key_matches_server_pattern(self):
        key = build_idempotency_key(_inputs())
        self.assertRegex(key, _KEY_PATTERN)

    def test_key_stable_across_calls(self):
        a = build_idempotency_key(_inputs())
        b = build_idempotency_key(_inputs())
        self.assertEqual(a, b)

    def test_key_changes_on_payload_change(self):
        a = build_idempotency_key(_inputs(payload={"v": 1}))
        b = build_idempotency_key(_inputs(payload={"v": 2}))
        self.assertNotEqual(a, b)

    def test_key_changes_on_species_change(self):
        a = build_idempotency_key(_inputs(species_label="ethanol"))
        b = build_idempotency_key(_inputs(species_label="methanol"))
        self.assertNotEqual(a, b)

    def test_key_changes_on_project_change(self):
        a = build_idempotency_key(_inputs(project_label="projA"))
        b = build_idempotency_key(_inputs(project_label="projB"))
        self.assertNotEqual(a, b)

    def test_no_project_label_still_works(self):
        key = build_idempotency_key(_inputs(project_label=None))
        self.assertRegex(key, _KEY_PATTERN)
        self.assertTrue(key.startswith("arc:"))

    def test_payload_dict_ordering_does_not_change_key(self):
        a = build_idempotency_key(_inputs(payload={"a": 1, "b": 2}))
        b = build_idempotency_key(_inputs(payload={"b": 2, "a": 1}))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
