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
    excitation_rank,
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


class TestExcitationRank(unittest.TestCase):
    """``excitation_rank`` parses the CC excitation rank off a method string."""

    def test_plain_cc_methods(self):
        self.assertEqual(excitation_rank('ccsd'), 2)
        self.assertEqual(excitation_rank('ccsdt'), 3)
        self.assertEqual(excitation_rank('ccsdtq'), 4)

    def test_perturbative_top_counts_like_iterative(self):
        self.assertEqual(excitation_rank('ccsd(t)'), 3)
        self.assertEqual(excitation_rank('ccsdt(q)'), 4)
        self.assertEqual(excitation_rank('ccsdtq(p)'), 5)

    def test_case_insensitive(self):
        self.assertEqual(excitation_rank('CCSD'), 2)
        self.assertEqual(excitation_rank('CCSDT(Q)'), 4)

    def test_f12_suffix_stripped(self):
        self.assertEqual(excitation_rank('ccsd(t)-f12'), 3)
        self.assertEqual(excitation_rank('ccsd-f12a'), 2)
        self.assertEqual(excitation_rank('CCSD(T)-F12b'), 3)

    def test_spin_restriction_prefixes(self):
        self.assertEqual(excitation_rank('uccsd(t)'), 3)
        self.assertEqual(excitation_rank('rccsd(t)'), 3)

    def test_non_cc_methods_return_none(self):
        for method in ('hf', 'mp2', 'b3lyp', 'wb97xd', 'cisd', 'cbs-qb3', ''):
            self.assertIsNone(excitation_rank(method), method)


class TestDeltaTermIsTriviallyZero(unittest.TestCase):
    """``DeltaTerm.is_trivially_zero`` — the proactive δ≡0 decision."""

    def setUp(self):
        self.delta_t = DeltaTerm(label='delta_T',
                                 high=Level(method='ccsdt', basis='cc-pVDZ'),
                                 low=Level(method='ccsd(t)', basis='cc-pVDZ'))
        self.delta_q = DeltaTerm(label='delta_Q',
                                 high=Level(method='ccsdt(q)', basis='cc-pVDZ'),
                                 low=Level(method='ccsdt', basis='cc-pVDZ'))

    def test_h_atom_skips_delta_t(self):
        self.assertTrue(self.delta_t.is_trivially_zero(1))

    def test_he_skips_delta_t_and_delta_q(self):
        self.assertTrue(self.delta_t.is_trivially_zero(2))
        self.assertTrue(self.delta_q.is_trivially_zero(2))

    def test_three_correlated_electrons_keep_delta_t(self):
        # CCSD(T)'s perturbative triples are inexact for 3 correlated electrons.
        self.assertFalse(self.delta_t.is_trivially_zero(3))

    def test_strict_boundary_rank_equals_n_corr_not_skipped(self):
        # Be all-electron: n_corr=4 vs δ[(Q)] rank 4 — 4 < 4 is False.
        self.assertFalse(self.delta_q.is_trivially_zero(4))

    def test_three_correlated_electrons_skip_delta_q(self):
        # CCSDT is FCI for 3 electrons and quadruples cannot exist, so (Q) = 0.
        self.assertTrue(self.delta_q.is_trivially_zero(3))

    def test_perturbative_low_leg_at_boundary_not_skipped(self):
        # CCSDTQ is FCI for 3 electrons but CCSD(T) is not — δ ≠ 0.
        term = DeltaTerm(label='d',
                         high=Level(method='ccsdtq', basis='cc-pVDZ'),
                         low=Level(method='ccsd(t)', basis='cc-pVDZ'))
        self.assertFalse(term.is_trivially_zero(3))

    def test_non_cc_low_leg_never_skipped(self):
        term = DeltaTerm(label='d',
                         high=Level(method='ccsd', basis='cc-pVDZ'),
                         low=Level(method='mp2', basis='cc-pVDZ'))
        self.assertFalse(term.is_trivially_zero(1))

    def test_polyatomic_counts_keep_all_deltas(self):
        for n_corr in (8, 10, 50):
            self.assertFalse(self.delta_t.is_trivially_zero(n_corr))
            self.assertFalse(self.delta_q.is_trivially_zero(n_corr))


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

    def test_evaluate_user_formula_zero_division_reports_formula_and_inputs(self):
        term = CBSExtrapolationTerm(
            label="cbs_user",
            formula="E_X / (E_Y - E_X)",
            levels=[self.tz, self.qz],
        )
        with self.assertRaises(InputError) as cm:
            term.evaluate({"cbs_user__card_3": -0.30, "cbs_user__card_4": -0.30})
        message = str(cm.exception)
        self.assertIn("E_X / (E_Y - E_X)", message)
        self.assertIn("{3: -0.3, 4: -0.3}", message)

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
        """Reject components != 'total'. parse_e_elect returns total
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

    # --- formula arity at construction --------------------------------------- #

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

    def test_cbs_term_in_corrections_rejected(self):
        """A cbs_extrapolation correction with components='total' extrapolates
        absolute energies and would double-count the base — rejected at
        construction with a pointer to CBS-as-base usage."""
        cbs_term = CBSExtrapolationTerm(
            label="cbs_corr",
            formula="helgaker_corr_2pt",
            levels=[Level(method="ccsd(t)", basis="cc-pVTZ"),
                    Level(method="ccsd(t)", basis="cc-pVQZ")],
        )
        with self.assertRaises(InputError) as ctx:
            CompositeProtocol(base=_hf_base(), corrections=[cbs_term])
        message = str(ctx.exception)
        self.assertIn("base", message)
        self.assertIn("HEAT", message)

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

    def test_delta_base_rejected(self):
        """A DeltaTerm provides no absolute energy and cannot anchor a protocol."""
        with self.assertRaises(InputError):
            CompositeProtocol(base=_delta_t(), corrections=[])

    def test_primary_base_accessors_for_single_point_base(self):
        protocol = CompositeProtocol(base=_hf_base(), corrections=[_delta_t()])
        self.assertEqual(protocol.base_sub_labels, ["base"])
        self.assertEqual(protocol.primary_base_sub_label, "base")
        self.assertEqual(protocol.primary_base_level, protocol.base.level)

    def test_duplicate_term_labels_rejected(self):
        with self.assertRaises(InputError):
            CompositeProtocol(base=_hf_base(), corrections=[_delta_t(), _delta_t()])

    def test_label_collision_with_base_rejected(self):
        clash = SinglePointTerm(label="base", level=Level(method="hf", basis="cc-pVTZ"))
        with self.assertRaises(InputError):
            CompositeProtocol(base=_hf_base(), corrections=[clash])

    def test_sub_label_collision_across_terms_rejected(self):
        """A SinglePointTerm whose label matches a DeltaTerm's generated
        sub_label must be rejected at construction time. Without this check,
        the scheduler's pending/completed maps would get silent overwrites."""
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


def _cbs_base():
    return CBSExtrapolationTerm(
        label="base",
        formula="helgaker_corr_2pt",
        levels=[Level(method="ccsd(t)", basis="cc-pVTZ"),
                Level(method="ccsd(t)", basis="cc-pVQZ")],
    )


class TestCompositeProtocolCBSBase(unittest.TestCase):
    """CBS-as-base: the canonical FPA shape — base is the absolute CBS energy."""

    def test_cbs_base_accepted(self):
        protocol = CompositeProtocol(base=_cbs_base(), corrections=[])
        self.assertIsInstance(protocol.base, CBSExtrapolationTerm)

    def test_evaluate_reproduces_hand_computed_fpa(self):
        """E_final = E_CBS(T,Q) + δT, checked against a hand-computed value."""
        protocol = CompositeProtocol(base=_cbs_base(), corrections=[_delta_t()])
        e_3, e_4 = -100.30, -100.31
        energies = {
            "base__card_3": e_3,
            "base__card_4": e_4,
            "delta_T__high": -100.5,
            "delta_T__low": -100.4,
        }
        e_cbs = (27 * e_3 - 64 * e_4) / (27 - 64)
        expected = e_cbs + (-100.5 - -100.4)
        self.assertAlmostEqual(protocol.evaluate(energies), expected, places=10)

    def test_base_sub_labels_deterministic_from_cardinals(self):
        protocol = CompositeProtocol(base=_cbs_base(), corrections=[])
        self.assertEqual(protocol.base_sub_labels, ["base__card_3", "base__card_4"])

    def test_primary_base_is_largest_cardinal_leg(self):
        protocol = CompositeProtocol(base=_cbs_base(), corrections=[])
        self.assertEqual(protocol.primary_base_sub_label, "base__card_4")
        self.assertEqual(protocol.primary_base_level.basis, "cc-pvqz")

    def test_iter_required_jobs_includes_one_job_per_cardinal(self):
        protocol = CompositeProtocol(base=_cbs_base(), corrections=[_delta_t()])
        sub_labels = sorted(sl for _t, sl, _l in protocol.iter_required_jobs())
        self.assertEqual(sub_labels, ["base__card_3", "base__card_4",
                                      "delta_T__high", "delta_T__low"])

    def test_round_trip_dict_preserves_sub_labels(self):
        """Restart rehydration keys off sub_labels — they must survive
        ``as_dict`` → ``from_dict`` byte-identically."""
        protocol = CompositeProtocol(base=_cbs_base(), corrections=[_delta_t()])
        rebuilt = CompositeProtocol.from_dict(protocol.as_dict())
        self.assertIsInstance(rebuilt.base, CBSExtrapolationTerm)
        self.assertEqual(
            [(t, sl) for t, sl, _l in rebuilt.iter_required_jobs()],
            [(t, sl) for t, sl, _l in protocol.iter_required_jobs()],
        )

    def test_round_trip_evaluate_identical(self):
        protocol = CompositeProtocol(base=_cbs_base(), corrections=[_delta_t()])
        rebuilt = CompositeProtocol.from_dict(protocol.as_dict())
        energies = {
            "base__card_3": -100.30, "base__card_4": -100.31,
            "delta_T__high": -100.5, "delta_T__low": -100.4,
        }
        self.assertAlmostEqual(rebuilt.evaluate(energies),
                               protocol.evaluate(energies), places=12)

    def test_from_user_input_cbs_base_dict(self):
        """A typed cbs_extrapolation base dict (no label) defaults to label 'base'."""
        raw = {
            "base": {
                "type": "cbs_extrapolation",
                "formula": "helgaker_corr_2pt",
                "levels": [{"method": "ccsd(t)", "basis": "cc-pVTZ"},
                           {"method": "ccsd(t)", "basis": "cc-pVQZ"}],
            },
            "corrections": [],
        }
        protocol = CompositeProtocol.from_user_input(raw)
        self.assertIsInstance(protocol.base, CBSExtrapolationTerm)
        self.assertEqual(protocol.base_sub_labels, ["base__card_3", "base__card_4"])

    def test_from_user_input_delta_base_rejected(self):
        raw = {
            "base": {"type": "delta", "label": "base",
                     "high": "ccsdt/cc-pVDZ", "low": "ccsd(t)/cc-pVDZ"},
            "corrections": [],
        }
        with self.assertRaises(InputError):
            CompositeProtocol.from_user_input(raw)

    def test_from_user_input_cbs_correction_rejected(self):
        raw = {
            "base": "ccsd(t)-f12/cc-pVTZ-f12",
            "corrections": [
                {"label": "cbs_corr", "type": "cbs_extrapolation",
                 "formula": "helgaker_corr_2pt", "components": "total",
                 "levels": [{"method": "ccsd(t)", "basis": "cc-pVTZ"},
                            {"method": "ccsd(t)", "basis": "cc-pVQZ"}]},
            ],
        }
        with self.assertRaises(InputError):
            CompositeProtocol.from_user_input(raw)


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
    """``from_user_input`` must not mutate caller-owned dicts.

    Previously the base-dict branch popped the ``label`` key off the caller's
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
    """Serialised ``as_dict()`` output must round-trip preset_name
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
