#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for ``arc.level.species_state`` — the three-state model that
distinguishes, at the per-species level:

* ``"inherit"``   — user did not set ``sp_composite`` for this species; fall back
  to the project-wide protocol (if any).
* ``"opt_out"``   — user wrote ``sp_composite: null`` for this species; explicitly
  bypass the project-wide protocol and use plain ``sp_level``.
* ``"explicit"``  — user provided a preset name or a recipe dict for this species.

The ``INHERIT`` sentinel is the default argument value used to distinguish
"user passed nothing" from "user passed ``None``" at constructor time.
"""

import unittest

from arc.exceptions import InputError
from arc.level.protocol import CompositeProtocol
from arc.level.species_state import (
    INHERIT,
    SP_COMPOSITE_STATES,
    active_composite_for,
)


def _dummy_protocol(label: str = "base") -> CompositeProtocol:
    return CompositeProtocol.from_user_input({
        "base": {"method": "hf", "basis": "cc-pVTZ"},
        "corrections": [],
    })


class TestInheritSentinel(unittest.TestCase):
    def test_sentinel_is_a_singleton(self):
        from arc.level.species_state import INHERIT as a, INHERIT as b
        self.assertIs(a, b)

    def test_sentinel_is_not_none(self):
        self.assertIsNot(INHERIT, None)

    def test_sentinel_has_a_readable_repr(self):
        self.assertIn("INHERIT", repr(INHERIT))

    def test_sentinel_is_exported_from_package(self):
        from arc.level import INHERIT as via_package
        self.assertIs(via_package, INHERIT)


class TestStatesConstant(unittest.TestCase):
    def test_exact_membership(self):
        self.assertEqual(set(SP_COMPOSITE_STATES), {"inherit", "opt_out", "explicit"})


class TestActiveCompositeFor(unittest.TestCase):
    def test_explicit_returns_species_protocol(self):
        species_proto = _dummy_protocol()
        global_proto = _dummy_protocol()
        result = active_composite_for("explicit", species_proto, global_proto)
        self.assertIs(result, species_proto)

    def test_opt_out_returns_none_even_with_global(self):
        global_proto = _dummy_protocol()
        result = active_composite_for("opt_out", None, global_proto)
        self.assertIsNone(result)

    def test_inherit_returns_global(self):
        global_proto = _dummy_protocol()
        result = active_composite_for("inherit", None, global_proto)
        self.assertIs(result, global_proto)

    def test_inherit_with_no_global_returns_none(self):
        result = active_composite_for("inherit", None, None)
        self.assertIsNone(result)

    def test_explicit_without_species_protocol_errors(self):
        # An "explicit" state with no protocol is a construction bug, not a user error.
        with self.assertRaises(InputError):
            active_composite_for("explicit", None, None)

    def test_unknown_state_rejected(self):
        with self.assertRaises(InputError):
            active_composite_for("bogus", None, None)


if __name__ == "__main__":
    unittest.main()
