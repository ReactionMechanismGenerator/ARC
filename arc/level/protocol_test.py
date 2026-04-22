#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for ``arc.level.protocol`` — the data model behind ``sp_composite``.

These tests cover only the pure data-model layer (no scheduler, no IO). They verify:
* ``Term`` subclasses know which sub-jobs they need and how to combine results.
* ``CompositeProtocol`` sums correctly and survives YAML round-trips.
* Construction-time validation rejects malformed inputs with :class:`InputError`.
"""

import copy
import unittest

from arc.exceptions import InputError
from arc.level import Level
from arc.level.protocol import (
    CBSExtrapolationTerm,
    CompositeProtocol,
    DeltaTerm,
    SinglePointTerm,
    Term,
    build_protocol,
)


# --------------------------------------------------------------------------- #
#  Term subclasses                                                            #
# --------------------------------------------------------------------------- #


class TestSinglePointTerm(unittest.TestCase):
    def test_required_levels_one_entry(self):
        lvl = Level(method="ccsd(t)-f12", basis="cc-pVTZ-f12")
        t = SinglePointTerm(label="base", level=lvl)
        self.assertEqual(t.required_levels(), [("base", lvl)])

    def test_evaluate_returns_input(self):
        lvl = Level(method="hf", basis="cc-pVTZ")
        t = SinglePointTerm(label="base", level=lvl)
        self.assertEqual(t.evaluate({"base": -76.0}), -76.0)

    def test_evaluate_missing_key_raises(self):
        t = SinglePointTerm(label="base", level=Level(method="hf", basis="cc-pVTZ"))
        with self.assertRaises(KeyError):
            t.evaluate({"other": -1.0})

    def test_round_trip_dict(self):
        original = SinglePointTerm(label="base", level=Level(method="hf", basis="cc-pVTZ"))
        as_dict = original.as_dict()
        self.assertEqual(as_dict["type"], "single_point")
        self.assertEqual(as_dict["label"], "base")
        rebuilt = Term.from_dict(as_dict)
        self.assertIsInstance(rebuilt, SinglePointTerm)
        self.assertEqual(rebuilt.level.method, "hf")


class TestDeltaTerm(unittest.TestCase):
    def setUp(self):
        self.high = Level(method="ccsdt", basis="cc-pVDZ")
        self.low = Level(method="ccsd(t)", basis="cc-pVDZ")

    def test_required_levels_returns_high_and_low(self):
        t = DeltaTerm(label="delta_T", high=self.high, low=self.low)
        pairs = dict(t.required_levels())
        self.assertEqual(set(pairs.keys()), {"delta_T__high", "delta_T__low"})
        self.assertIs(pairs["delta_T__high"], self.high)
        self.assertIs(pairs["delta_T__low"], self.low)

    def test_evaluate_high_minus_low(self):
        t = DeltaTerm(label="delta_T", high=self.high, low=self.low)
        result = t.evaluate({"delta_T__high": -100.0, "delta_T__low": -98.0})
        self.assertEqual(result, -2.0)

    def test_evaluate_independent_of_other_keys(self):
        t = DeltaTerm(label="delta_T", high=self.high, low=self.low)
        result = t.evaluate(
            {"delta_T__high": -100.0, "delta_T__low": -98.0, "noise": 999.0}
        )
        self.assertEqual(result, -2.0)

    def test_round_trip_dict(self):
        original = DeltaTerm(label="delta_T", high=self.high, low=self.low)
        rebuilt = Term.from_dict(original.as_dict())
        self.assertIsInstance(rebuilt, DeltaTerm)
        self.assertEqual(rebuilt.label, "delta_T")
        self.assertEqual(rebuilt.high.method, "ccsdt")
        self.assertEqual(rebuilt.low.method, "ccsd(t)")

    def test_construction_requires_both_high_and_low(self):
        with self.assertRaises(InputError):
            DeltaTerm(label="bad", high=self.high, low=None)
        with self.assertRaises(InputError):
            DeltaTerm(label="bad", high=None, low=self.low)


class TestCBSExtrapolationTerm(unittest.TestCase):
    def setUp(self):
        self.tz = Level(method="ccsd(t)", basis="cc-pVTZ")
        self.qz = Level(method="ccsd(t)", basis="cc-pVQZ")
        self.fz = Level(method="ccsd(t)", basis="cc-pV5Z")

    def test_required_levels_uses_cardinal_in_sub_label(self):
        term = CBSExtrapolationTerm(
            label="cbs_corr", formula="helgaker_corr_2pt", levels=[self.tz, self.qz]
        )
        keys = set(k for k, _ in term.required_levels())
        self.assertEqual(keys, {"cbs_corr__card_3", "cbs_corr__card_4"})

    def test_evaluate_calls_builtin_formula(self):
        term = CBSExtrapolationTerm(
            label="cbs_corr", formula="helgaker_corr_2pt", levels=[self.tz, self.qz]
        )
        # Same formula as cbs_test::test_known_values:
        # E_CBS = (27*(-0.30) - 64*(-0.31)) / (27 - 64)
        result = term.evaluate({"cbs_corr__card_3": -0.30, "cbs_corr__card_4": -0.31})
        expected = (27 * -0.30 - 64 * -0.31) / (27 - 64)
        self.assertAlmostEqual(result, expected, places=12)

    def test_evaluate_user_formula(self):
        term = CBSExtrapolationTerm(
            label="cbs_user",
            formula="(X**3 * E_X - Y**3 * E_Y) / (X**3 - Y**3)",
            levels=[self.tz, self.qz],
        )
        result = term.evaluate({"cbs_user__card_3": -0.30, "cbs_user__card_4": -0.31})
        expected = (27 * -0.30 - 64 * -0.31) / (27 - 64)
        self.assertAlmostEqual(result, expected, places=12)

    def test_three_point_martin(self):
        term = CBSExtrapolationTerm(
            label="cbs_m", formula="martin_3pt", levels=[self.tz, self.qz, self.fz]
        )
        # Synthetic E(L) = -1 + 0.05/(L+0.5)^4 + 0.01/(L+0.5)^6
        e3 = -1.0 + 0.05 / 3.5**4 + 0.01 / 3.5**6
        e4 = -1.0 + 0.05 / 4.5**4 + 0.01 / 4.5**6
        e5 = -1.0 + 0.05 / 5.5**4 + 0.01 / 5.5**6
        result = term.evaluate(
            {"cbs_m__card_3": e3, "cbs_m__card_4": e4, "cbs_m__card_5": e5}
        )
        self.assertAlmostEqual(result, -1.0, places=10)

    def test_round_trip_dict(self):
        original = CBSExtrapolationTerm(
            label="cbs_corr", formula="helgaker_corr_2pt", levels=[self.tz, self.qz]
        )
        rebuilt = Term.from_dict(original.as_dict())
        self.assertIsInstance(rebuilt, CBSExtrapolationTerm)
        self.assertEqual(rebuilt.formula, "helgaker_corr_2pt")
        self.assertEqual(len(rebuilt.levels), 2)

    def test_validate_too_few_levels(self):
        with self.assertRaises(InputError):
            CBSExtrapolationTerm(label="x", formula="helgaker_corr_2pt", levels=[self.tz])

    def test_validate_method_mismatch_across_levels(self):
        mixed = Level(method="ccsdt", basis="cc-pVQZ")
        with self.assertRaises(InputError):
            CBSExtrapolationTerm(
                label="x", formula="helgaker_corr_2pt", levels=[self.tz, mixed]
            )

    def test_validate_indistinct_cardinals(self):
        another_tz = Level(method="ccsd(t)", basis="cc-pVTZ")
        with self.assertRaises(InputError):
            CBSExtrapolationTerm(
                label="x", formula="helgaker_corr_2pt", levels=[self.tz, another_tz]
            )

    def test_validate_unknown_builtin_formula(self):
        with self.assertRaises(InputError):
            CBSExtrapolationTerm(
                label="x", formula="not_a_real_formula_2pt", levels=[self.tz, self.qz]
            )

    def test_user_formula_accepted_at_construction(self):
        term = CBSExtrapolationTerm(
            label="x",
            formula="E_X + E_Y",
            levels=[self.tz, self.qz],
        )
        self.assertEqual(term.formula, "E_X + E_Y")

    def test_user_formula_with_disallowed_node_rejected_at_construction(self):
        with self.assertRaises(InputError):
            CBSExtrapolationTerm(
                label="x",
                formula="(0).__class__",
                levels=[self.tz, self.qz],
            )

    def test_components_default_total(self):
        term = CBSExtrapolationTerm(
            label="x", formula="helgaker_corr_2pt", levels=[self.tz, self.qz]
        )
        self.assertEqual(term.components, "total")

    def test_components_total_is_default_and_valid(self):
        term = CBSExtrapolationTerm(
            label="x", formula="helgaker_corr_2pt", levels=[self.tz, self.qz],
        )
        self.assertEqual(term.components, "total")

    def test_components_total_explicit_accepted(self):
        CBSExtrapolationTerm(
            label="x", formula="helgaker_corr_2pt", levels=[self.tz, self.qz],
            components="total",
        )

    def test_components_corr_rejected_until_component_parsing_exists(self):
        """Phase 5.5: reject components != 'total'. parse_e_elect returns total
        energies, so extrapolating them while claiming 'corr' or 'hf' would
        silently produce a wrong answer."""
        with self.assertRaises(InputError):
            CBSExtrapolationTerm(
                label="x", formula="helgaker_corr_2pt", levels=[self.tz, self.qz],
                components="corr",
            )

    def test_components_hf_rejected_until_component_parsing_exists(self):
        with self.assertRaises(InputError):
            CBSExtrapolationTerm(
                label="x", formula="helgaker_hf_2pt", levels=[self.tz, self.qz],
                components="hf",
            )

    def test_components_bogus_rejected(self):
        with self.assertRaises(InputError):
            CBSExtrapolationTerm(
                label="x", formula="helgaker_corr_2pt", levels=[self.tz, self.qz],
                components="bogus",
            )

    # --- Phase 5.5: formula arity at construction ---------------------------- #

    def test_martin_3pt_with_2_levels_rejected_at_construction(self):
        """martin_3pt needs exactly 3 levels — rejected eagerly."""
        with self.assertRaises(InputError) as ctx:
            CBSExtrapolationTerm(
                label="m", formula="martin_3pt",
                levels=[self.tz, self.qz],
            )
        self.assertIn("martin_3pt", str(ctx.exception))
        self.assertIn("3", str(ctx.exception))

    def test_helgaker_corr_2pt_with_3_levels_rejected_at_construction(self):
        with self.assertRaises(InputError):
            CBSExtrapolationTerm(
                label="h", formula="helgaker_corr_2pt",
                levels=[self.tz, self.qz, self.fz],
            )

    def test_user_formula_with_4_levels_rejected(self):
        """User formulas expose only X/Y/Z; 4+ levels are rejected."""
        fourth = Level(method="ccsd(t)", basis="cc-pV6Z")
        with self.assertRaises(InputError) as ctx:
            CBSExtrapolationTerm(
                label="u", formula="E_X + E_Y + E_Z",
                levels=[self.tz, self.qz, self.fz, fourth],
            )
        self.assertIn("3", str(ctx.exception))


# --------------------------------------------------------------------------- #
#  CompositeProtocol                                                          #
# --------------------------------------------------------------------------- #


def _hf_base():
    return SinglePointTerm(label="base", level=Level(method="ccsd(t)-f12", basis="cc-pVTZ-f12"))


def _delta_t():
    return DeltaTerm(
        label="delta_T",
        high=Level(method="ccsdt", basis="cc-pVDZ"),
        low=Level(method="ccsd(t)", basis="cc-pVDZ"),
    )


def _delta_q():
    return DeltaTerm(
        label="delta_Q",
        high=Level(method="ccsdt(q)", basis="cc-pVDZ"),
        low=Level(method="ccsdt", basis="cc-pVDZ"),
    )


class TestCompositeProtocolBasics(unittest.TestCase):
    def test_evaluate_two_term(self):
        protocol = CompositeProtocol(base=_hf_base(), corrections=[_delta_t()])
        energies = {"base": -100.0, "delta_T__high": -100.5, "delta_T__low": -100.4}
        # base + (high - low) = -100.0 + (-100.5 - -100.4) = -100.1
        self.assertAlmostEqual(protocol.evaluate(energies), -100.1, places=12)

    def test_evaluate_three_term(self):
        protocol = CompositeProtocol(base=_hf_base(), corrections=[_delta_t(), _delta_q()])
        energies = {
            "base": -100.0,
            "delta_T__high": -100.5, "delta_T__low": -100.4,    # δT = -0.1
            "delta_Q__high": -100.55, "delta_Q__low": -100.5,   # δQ = -0.05
        }
        self.assertAlmostEqual(protocol.evaluate(energies), -100.15, places=12)

    def test_evaluate_with_cbs_term(self):
        cbs_term = CBSExtrapolationTerm(
            label="cbs_corr",
            formula="helgaker_corr_2pt",
            levels=[Level(method="ccsd(t)", basis="cc-pVTZ"),
                    Level(method="ccsd(t)", basis="cc-pVQZ")],
        )
        protocol = CompositeProtocol(base=_hf_base(), corrections=[cbs_term])
        energies = {
            "base": -100.0,
            "cbs_corr__card_3": -0.30,
            "cbs_corr__card_4": -0.31,
        }
        cbs_value = (27 * -0.30 - 64 * -0.31) / (27 - 64)
        self.assertAlmostEqual(protocol.evaluate(energies), -100.0 + cbs_value, places=12)

    def test_iter_required_jobs_yields_every_sub_job(self):
        protocol = CompositeProtocol(base=_hf_base(), corrections=[_delta_t(), _delta_q()])
        triples = list(protocol.iter_required_jobs())
        sub_labels = sorted(t[1] for t in triples)
        self.assertEqual(
            sub_labels,
            sorted(["base", "delta_T__high", "delta_T__low",
                    "delta_Q__high", "delta_Q__low"]),
        )
        # Each triple is (term_label, sub_label, Level).
        for term_label, sub_label, level in triples:
            self.assertIsInstance(level, Level)
            self.assertTrue(sub_label.startswith(term_label))

    def test_base_is_a_single_point_term(self):
        with self.assertRaises(InputError):
            CompositeProtocol(base=_delta_t(), corrections=[])

    def test_duplicate_term_labels_rejected(self):
        with self.assertRaises(InputError):
            CompositeProtocol(base=_hf_base(), corrections=[_delta_t(), _delta_t()])

    def test_label_collision_with_base_rejected(self):
        clash = SinglePointTerm(label="base", level=Level(method="hf", basis="cc-pVTZ"))
        with self.assertRaises(InputError):
            CompositeProtocol(base=_hf_base(), corrections=[clash])

    def test_sub_label_collision_across_terms_rejected(self):
        """Phase 5.5: a SinglePointTerm whose label matches a DeltaTerm's
        generated sub_label must be rejected at construction time. Without this
        check, the scheduler's pending/completed maps would get silent overwrites."""
        with self.assertRaises(InputError) as ctx:
            CompositeProtocol(
                base=_hf_base(),
                corrections=[
                    SinglePointTerm(
                        label="delta_T__high",
                        level=Level(method="hf", basis="cc-pVDZ"),
                    ),
                    _delta_t(),  # produces sub_labels delta_T__high, delta_T__low
                ],
            )
        self.assertIn("delta_T__high", str(ctx.exception))


# --------------------------------------------------------------------------- #
#  YAML / dict round-trip                                                     #
# --------------------------------------------------------------------------- #


class TestCompositeProtocolFromDict(unittest.TestCase):
    def test_explicit_form_round_trip(self):
        protocol = CompositeProtocol(
            base=_hf_base(), corrections=[_delta_t(), _delta_q()]
        )
        rebuilt = CompositeProtocol.from_dict(protocol.as_dict())
        # Round-trip preserves base + every correction.
        self.assertEqual(rebuilt.base.label, "base")
        self.assertEqual([t.label for t in rebuilt.corrections], ["delta_T", "delta_Q"])

    def test_explicit_form_evaluate_after_round_trip(self):
        protocol = CompositeProtocol(base=_hf_base(), corrections=[_delta_t()])
        rebuilt = CompositeProtocol.from_dict(protocol.as_dict())
        energies = {"base": -100.0, "delta_T__high": -100.5, "delta_T__low": -100.4}
        self.assertAlmostEqual(rebuilt.evaluate(energies), -100.1, places=12)

    def test_explicit_user_input_minimal(self):
        raw = {
            "base": {"method": "ccsd(t)-f12", "basis": "cc-pVTZ-f12"},
            "corrections": [
                {
                    "label": "delta_T", "type": "delta",
                    "high": {"method": "ccsdt", "basis": "cc-pVDZ"},
                    "low": {"method": "ccsd(t)", "basis": "cc-pVDZ"},
                },
            ],
        }
        protocol = CompositeProtocol.from_user_input(raw)
        self.assertEqual(protocol.base.level.method, "ccsd(t)-f12")
        self.assertEqual(protocol.corrections[0].label, "delta_T")

    def test_explicit_user_input_with_string_levels(self):
        raw = {
            "base": "ccsd(t)-f12/cc-pVTZ-f12",
            "corrections": [
                {"label": "delta_T", "type": "delta",
                 "high": "ccsdt/cc-pVDZ", "low": "ccsd(t)/cc-pVDZ"},
            ],
        }
        protocol = CompositeProtocol.from_user_input(raw)
        self.assertEqual(protocol.base.level.basis, "cc-pvtz-f12")
        self.assertEqual(protocol.corrections[0].high.method, "ccsdt")

    def test_user_input_missing_base_rejected(self):
        with self.assertRaises(InputError):
            CompositeProtocol.from_user_input({"corrections": []})

    def test_user_input_unknown_term_type_rejected(self):
        with self.assertRaises(InputError):
            CompositeProtocol.from_user_input({
                "base": "hf/cc-pVTZ",
                "corrections": [{"label": "x", "type": "bogus_term"}],
            })


class TestBuildProtocolHelper(unittest.TestCase):
    """``build_protocol`` is the public adapter from any user input form."""

    def test_dict_form_routed_to_from_user_input(self):
        protocol = build_protocol({
            "base": "hf/cc-pVTZ",
            "corrections": [],
        })
        self.assertIsInstance(protocol, CompositeProtocol)
        self.assertEqual(protocol.base.level.method, "hf")

    def test_already_a_protocol_returned_as_is(self):
        original = CompositeProtocol(base=_hf_base(), corrections=[])
        self.assertIs(build_protocol(original), original)

    def test_invalid_type_rejected(self):
        with self.assertRaises(InputError):
            build_protocol(12345)


class TestFromUserInputNoMutation(unittest.TestCase):
    """Phase 5.5: ``from_user_input`` must not mutate caller-owned dicts.

    Pre-5.5 the base-dict branch popped the ``label`` key off the caller's
    input, breaking idempotent re-parse and polluting restart state.
    """

    def test_explicit_dict_not_mutated(self):
        original = {
            "base": {"method": "hf", "basis": "cc-pVTZ", "label": "base"},
            "corrections": [
                {"label": "delta_T", "type": "delta",
                 "high": {"method": "ccsdt", "basis": "cc-pVDZ"},
                 "low": {"method": "ccsd(t)", "basis": "cc-pVDZ"}},
            ],
            "reference": "test DOI",
        }
        snapshot = copy.deepcopy(original)
        CompositeProtocol.from_user_input(original)
        self.assertEqual(original, snapshot, "from_user_input mutated its input dict")

    def test_preset_with_overrides_dict_not_mutated(self):
        raw = {"preset": "HEAT-345Q", "overrides": {"delta_T": {"high": {"method": "ccsdt", "basis": "cc-pVTZ"}}}}
        snapshot = copy.deepcopy(raw)
        CompositeProtocol.from_user_input(raw)
        self.assertEqual(raw, snapshot)

    def test_base_dict_with_label_key_not_mutated(self):
        base_dict = {"method": "hf", "basis": "cc-pVTZ", "label": "my_base"}
        recipe = {"base": base_dict, "corrections": []}
        snapshot = copy.deepcopy(base_dict)
        protocol = CompositeProtocol.from_user_input(recipe)
        self.assertEqual(base_dict, snapshot)
        self.assertEqual(protocol.base.label, "my_base")
        self.assertEqual(protocol.base.level.method, "hf")

    def test_build_protocol_not_mutating(self):
        raw = {"base": {"method": "hf", "basis": "cc-pVTZ"}, "corrections": []}
        snapshot = copy.deepcopy(raw)
        build_protocol(raw)
        self.assertEqual(raw, snapshot)


class TestFromUserInputPresetMetadataPreservation(unittest.TestCase):
    """Phase 5.5: serialised ``as_dict()`` output must round-trip preset_name
    through ``from_user_input`` (in addition to ``from_dict``)."""

    def test_from_user_input_reads_preset_name_from_as_dict_output(self):
        proto1 = CompositeProtocol.from_user_input("HEAT-345Q")
        serialised = proto1.as_dict()
        self.assertEqual(serialised["preset_name"], "HEAT-345Q")
        proto2 = CompositeProtocol.from_user_input(serialised)
        self.assertEqual(proto2.preset_name, "HEAT-345Q")
        self.assertEqual(proto2.reference, proto1.reference)

    def test_from_user_input_reads_reference_from_as_dict_output(self):
        recipe = {
            "base": {"method": "hf", "basis": "cc-pVTZ"},
            "corrections": [],
            "reference": "DOI: 10.1/test",
        }
        proto1 = CompositeProtocol.from_user_input(recipe)
        proto2 = CompositeProtocol.from_user_input(proto1.as_dict())
        self.assertEqual(proto2.reference, "DOI: 10.1/test")

    def test_build_protocol_path_also_preserves_preset_name(self):
        proto1 = CompositeProtocol.from_user_input("FPA-min")
        proto2 = build_protocol(proto1.as_dict())
        self.assertEqual(proto2.preset_name, "FPA-min")


if __name__ == "__main__":
    unittest.main()
