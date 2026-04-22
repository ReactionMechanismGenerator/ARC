#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for ``arc.level.presets`` — preset loading and override merging.

Presets are data: every entry in ``presets.yml`` should round-trip through
:meth:`CompositeProtocol.from_user_input` and through :meth:`CompositeProtocol.from_dict`
without loss. Preset overrides may replace named keys on individual terms but may not
introduce new term labels or unknown fields.
"""

import unittest

from arc.exceptions import InputError
from arc.level.presets import PRESETS, REGISTERED_PRESET_NAMES, expand_preset
from arc.level.protocol import CompositeProtocol


class TestPresetRegistry(unittest.TestCase):
    """The ``presets.yml`` data file ships at least three named protocols."""

    def test_registry_non_empty(self):
        self.assertGreaterEqual(len(REGISTERED_PRESET_NAMES), 3)

    def test_known_presets_present(self):
        for name in ("HEAT-345", "HEAT-345Q", "FPA-min"):
            self.assertIn(name, REGISTERED_PRESET_NAMES)

    def test_each_preset_carries_a_reference_field(self):
        """Every preset entry must include a `reference:` string with citation + DOI."""
        for name in REGISTERED_PRESET_NAMES:
            entry = PRESETS[name]
            self.assertIn("reference", entry, f"Preset '{name}' missing 'reference' field.")
            ref = entry["reference"]
            self.assertIsInstance(ref, str)
            self.assertGreater(len(ref), 20, f"Preset '{name}' reference too short.")
            self.assertIn("DOI", ref.upper(), f"Preset '{name}' reference must mention a DOI.")

    def test_each_preset_round_trips_to_protocol(self):
        for name in REGISTERED_PRESET_NAMES:
            with self.subTest(name=name):
                protocol = CompositeProtocol.from_user_input(name)
                rebuilt = CompositeProtocol.from_dict(protocol.as_dict())
                self.assertEqual(rebuilt.base.label, protocol.base.label)
                self.assertEqual(
                    [t.label for t in rebuilt.corrections],
                    [t.label for t in protocol.corrections],
                )


class TestExpandPreset(unittest.TestCase):
    def test_unknown_preset_raises(self):
        with self.assertRaises(InputError) as ctx:
            expand_preset("not_a_real_preset")
        # The error message should help the user discover the available presets.
        self.assertIn("HEAT-345", str(ctx.exception))

    def test_returns_dict_with_base_and_corrections(self):
        recipe = expand_preset("HEAT-345Q")
        self.assertIn("base", recipe)
        self.assertIn("corrections", recipe)
        self.assertIsInstance(recipe["corrections"], list)

    def test_no_overrides_returns_canonical_recipe(self):
        a = expand_preset("HEAT-345Q")
        b = expand_preset("HEAT-345Q")
        self.assertEqual(a, b)

    def test_returns_a_deep_copy(self):
        """Mutating the returned recipe must not affect later calls."""
        recipe = expand_preset("HEAT-345Q")
        recipe["base"] = "tampered"
        recipe["corrections"].clear()
        again = expand_preset("HEAT-345Q")
        self.assertNotEqual(again["base"], "tampered")
        self.assertGreater(len(again["corrections"]), 0)


class TestExpandPresetOverrides(unittest.TestCase):
    """Overrides target named term labels and replace specific fields on them."""

    def test_override_replaces_basis_on_named_delta_term(self):
        recipe = expand_preset(
            "HEAT-345Q",
            overrides={"delta_T": {"high": {"method": "ccsdt", "basis": "cc-pVTZ"}}},
        )
        delta_t = next(c for c in recipe["corrections"] if c["label"] == "delta_T")
        self.assertEqual(delta_t["high"]["basis"], "cc-pVTZ")

    def test_override_only_touches_named_term(self):
        recipe = expand_preset(
            "HEAT-345Q",
            overrides={"delta_T": {"high": {"method": "ccsdt", "basis": "cc-pVTZ"}}},
        )
        delta_q = next(c for c in recipe["corrections"] if c["label"] == "delta_Q")
        # delta_Q should be untouched.
        original = expand_preset("HEAT-345Q")
        original_delta_q = next(c for c in original["corrections"] if c["label"] == "delta_Q")
        self.assertEqual(delta_q, original_delta_q)

    def test_override_unknown_label_raises(self):
        with self.assertRaises(InputError):
            expand_preset("HEAT-345Q", overrides={"not_a_term": {"high": "hf/cc-pVDZ"}})

    def test_override_base_replaces_base_level(self):
        recipe = expand_preset(
            "HEAT-345Q",
            overrides={"base": {"method": "ccsd(t)-f12", "basis": "cc-pVQZ-f12"}},
        )
        self.assertEqual(recipe["base"]["basis"], "cc-pVQZ-f12")

    def test_overridden_preset_still_parses_into_a_protocol(self):
        recipe = expand_preset(
            "HEAT-345Q",
            overrides={"delta_T": {"high": {"method": "ccsdt", "basis": "cc-pVTZ"}}},
        )
        protocol = CompositeProtocol.from_user_input(recipe)
        delta_t = next(c for c in protocol.corrections if c.label == "delta_T")
        self.assertEqual(delta_t.high.basis, "cc-pvtz")

    # --- Phase 5.5 hardening --------------------------------------------- #

    def test_override_unknown_field_on_delta_rejected(self):
        """Typo guard: ``hihg`` is not a valid field of a delta term."""
        with self.assertRaises(InputError) as ctx:
            expand_preset("HEAT-345Q", overrides={
                "delta_T": {"hihg": {"method": "ccsdt", "basis": "cc-pVTZ"}},
            })
        self.assertIn("hihg", str(ctx.exception))

    def test_override_unknown_field_on_base_rejected(self):
        """``methhod`` is not a valid Level field."""
        with self.assertRaises(InputError) as ctx:
            expand_preset("HEAT-345Q", overrides={
                "base": {"methhod": "hf"},
            })
        self.assertIn("methhod", str(ctx.exception))

    def test_override_unknown_field_on_cbs_rejected(self):
        """Typo on a cbs_extrapolation term is caught (FPA-min has a CBS term)."""
        with self.assertRaises(InputError) as ctx:
            expand_preset("FPA-min", overrides={
                "cbs_corr": {"formla": "helgaker_corr_2pt"},
            })
        self.assertIn("formla", str(ctx.exception))

    def test_override_deep_merges_high_level_dict(self):
        """Overriding ``delta_T.high.basis`` preserves the existing ``method``."""
        recipe = expand_preset(
            "HEAT-345Q",
            overrides={"delta_T": {"high": {"basis": "cc-pVTZ"}}},
        )
        delta_t = next(c for c in recipe["corrections"] if c["label"] == "delta_T")
        self.assertEqual(delta_t["high"]["basis"], "cc-pVTZ")
        # Original method ("ccsdt") is preserved by the deep-merge.
        self.assertEqual(delta_t["high"]["method"], "ccsdt")

    def test_override_deep_merges_base_dict(self):
        recipe = expand_preset(
            "HEAT-345Q",
            overrides={"base": {"basis": "cc-pVQZ-f12"}},
        )
        self.assertEqual(recipe["base"]["basis"], "cc-pVQZ-f12")
        # Existing method ("ccsd(t)-f12") preserved.
        self.assertEqual(recipe["base"]["method"], "ccsd(t)-f12")


class TestPresetIntegrationWithFromUserInput(unittest.TestCase):
    def test_string_form_dispatches_to_preset(self):
        protocol = CompositeProtocol.from_user_input("HEAT-345Q")
        self.assertIsInstance(protocol, CompositeProtocol)

    def test_preset_with_overrides_form(self):
        protocol = CompositeProtocol.from_user_input({
            "preset": "HEAT-345Q",
            "overrides": {"delta_T": {"high": {"method": "ccsdt", "basis": "cc-pVTZ"}}},
        })
        delta_t = next(c for c in protocol.corrections if c.label == "delta_T")
        self.assertEqual(delta_t.high.basis, "cc-pvtz")


if __name__ == "__main__":
    unittest.main()
