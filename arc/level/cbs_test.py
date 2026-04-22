#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for ``arc.level.cbs`` — basis-set cardinal inference, built-in CBS
extrapolation formulas, and the safe AST evaluator for user-supplied formulas.

References whose values are checked here:

* Helgaker, Klopper, Koch, Noga, *J. Chem. Phys.* **106**, 9639 (1997).
  DOI: 10.1063/1.473863 — two-point correlation extrapolation.
* Halkier, Helgaker, Jørgensen, Klopper, Koch, Olsen, Wilson,
  *Chem. Phys. Lett.* **286**, 243-252 (1998). DOI: 10.1016/S0009-2614(98)00111-0
  — two-point HF extrapolation, Table 2 reports α = 1.63.
* Martin, *Chem. Phys. Lett.* **259**, 669-678 (1996).
  DOI: 10.1016/0009-2614(96)00898-6 — three-point Schwartz expansion.
"""

import math
import unittest

from arc.exceptions import InputError
from arc.level.cbs import (
    BUILTIN_FORMULAS,
    cardinal_from_basis,
    helgaker_corr_2pt,
    helgaker_hf_2pt,
    martin_3pt,
    safe_eval_formula,
)


class TestCardinalFromBasis(unittest.TestCase):
    """``cardinal_from_basis`` covers the common Dunning families and def2."""

    def test_cc_pvxz(self):
        self.assertEqual(cardinal_from_basis("cc-pVDZ"), 2)
        self.assertEqual(cardinal_from_basis("cc-pVTZ"), 3)
        self.assertEqual(cardinal_from_basis("cc-pVQZ"), 4)
        self.assertEqual(cardinal_from_basis("cc-pV5Z"), 5)
        self.assertEqual(cardinal_from_basis("cc-pV6Z"), 6)

    def test_aug_cc_pvxz(self):
        self.assertEqual(cardinal_from_basis("aug-cc-pVDZ"), 2)
        self.assertEqual(cardinal_from_basis("aug-cc-pVTZ"), 3)
        self.assertEqual(cardinal_from_basis("aug-cc-pVQZ"), 4)
        self.assertEqual(cardinal_from_basis("aug-cc-pV5Z"), 5)

    def test_cc_pcvxz_core_valence(self):
        self.assertEqual(cardinal_from_basis("cc-pCVDZ"), 2)
        self.assertEqual(cardinal_from_basis("cc-pCVTZ"), 3)
        self.assertEqual(cardinal_from_basis("cc-pCVQZ"), 4)
        self.assertEqual(cardinal_from_basis("aug-cc-pCVTZ"), 3)

    def test_def2_family(self):
        self.assertEqual(cardinal_from_basis("def2-SVP"), 2)
        self.assertEqual(cardinal_from_basis("def2-TZVP"), 3)
        self.assertEqual(cardinal_from_basis("def2-QZVP"), 4)
        self.assertEqual(cardinal_from_basis("def2-TZVPP"), 3)
        self.assertEqual(cardinal_from_basis("def2-QZVPP"), 4)

    def test_case_insensitive(self):
        self.assertEqual(cardinal_from_basis("cc-pvtz"), 3)
        self.assertEqual(cardinal_from_basis("CC-PVTZ"), 3)
        self.assertEqual(cardinal_from_basis("Aug-CC-pVQZ"), 4)
        self.assertEqual(cardinal_from_basis("DEF2-tzvp"), 3)

    def test_unknown_basis_raises(self):
        with self.assertRaises(InputError):
            cardinal_from_basis("6-31G*")
        with self.assertRaises(InputError):
            cardinal_from_basis("STO-3G")
        with self.assertRaises(InputError):
            cardinal_from_basis("not-a-basis-set")
        with self.assertRaises(InputError):
            cardinal_from_basis("")


class TestHelgakerCorr2Pt(unittest.TestCase):
    """``helgaker_corr_2pt`` implements (X^3·E_X − Y^3·E_Y) / (X^3 − Y^3)."""

    def test_known_values(self):
        # E_T = 1.0, E_Q = 1.05  ->  (27*1.0 - 64*1.05) / (27 - 64) = -40.2 / -37
        result = helgaker_corr_2pt({3: 1.0, 4: 1.05})
        self.assertAlmostEqual(result, 40.2 / 37, places=12)

    def test_invariance_to_dict_insertion_order(self):
        a = helgaker_corr_2pt({3: -1.0, 4: -1.05})
        b = helgaker_corr_2pt({4: -1.05, 3: -1.0})
        self.assertAlmostEqual(a, b, places=12)

    def test_higher_basis_dominates(self):
        # E_CBS should be closer to E_Q than to E_T (since cc-pVQZ is more accurate).
        e_t, e_q = -100.0, -100.05
        cbs = helgaker_corr_2pt({3: e_t, 4: e_q})
        self.assertLess(abs(cbs - e_q), abs(cbs - e_t))

    def test_real_h2o_correlation_extrapolation(self):
        # Synthetic but representative: CCSD(T) corr energy at TZ vs QZ.
        # E_corr_TZ = -0.30, E_corr_QZ = -0.31 (Hartree)  -> CBS ≈ -0.31730
        result = helgaker_corr_2pt({3: -0.30, 4: -0.31})
        expected = (27 * (-0.30) - 64 * (-0.31)) / (27 - 64)
        self.assertAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, -0.31729729729729728, places=10)

    def test_requires_exactly_two_points(self):
        with self.assertRaises(InputError):
            helgaker_corr_2pt({3: -1.0})
        with self.assertRaises(InputError):
            helgaker_corr_2pt({3: -1.0, 4: -1.05, 5: -1.06})

    def test_rejects_equal_cardinals(self):
        with self.assertRaises(InputError):
            helgaker_corr_2pt({3: -1.0, 3: -1.05})  # noqa: F601 — Python collapses; size=1 path

    def test_q5_pair_reproduces_formula(self):
        # X=4, Y=5; E_Q = -0.310, E_5 = -0.315
        result = helgaker_corr_2pt({4: -0.310, 5: -0.315})
        expected = (4**3 * -0.310 - 5**3 * -0.315) / (4**3 - 5**3)
        self.assertAlmostEqual(result, expected, places=12)


class TestHelgakerHF2Pt(unittest.TestCase):
    """``helgaker_hf_2pt`` extrapolates HF energies via E(X) = E_CBS + A·exp(-α·X)."""

    def test_default_alpha_is_halkier_value(self):
        # Halkier et al. 1998 fitted α = 1.63 (Table 2).
        # Pick numbers and verify the formula uses α=1.63 by default.
        e_t, e_q = -76.0500, -76.0510
        from_default = helgaker_hf_2pt({3: e_t, 4: e_q})
        from_explicit = helgaker_hf_2pt({3: e_t, 4: e_q}, alpha=1.63)
        self.assertAlmostEqual(from_default, from_explicit, places=12)

    def test_known_value(self):
        # E_CBS = (E_X · exp(-α·Y) - E_Y · exp(-α·X)) / (exp(-α·Y) - exp(-α·X))
        e_t, e_q = -76.0500, -76.0510
        alpha = 1.63
        expected = (
            e_t * math.exp(-alpha * 4) - e_q * math.exp(-alpha * 3)
        ) / (math.exp(-alpha * 4) - math.exp(-alpha * 3))
        result = helgaker_hf_2pt({3: e_t, 4: e_q})
        self.assertAlmostEqual(result, expected, places=12)

    def test_alpha_override(self):
        e_t, e_q = -76.0500, -76.0510
        alpha = 1.50
        expected = (
            e_t * math.exp(-alpha * 4) - e_q * math.exp(-alpha * 3)
        ) / (math.exp(-alpha * 4) - math.exp(-alpha * 3))
        self.assertAlmostEqual(helgaker_hf_2pt({3: e_t, 4: e_q}, alpha=alpha), expected, places=12)

    def test_invariance_to_dict_insertion_order(self):
        a = helgaker_hf_2pt({3: -76.05, 4: -76.051})
        b = helgaker_hf_2pt({4: -76.051, 3: -76.05})
        self.assertAlmostEqual(a, b, places=12)

    def test_requires_exactly_two_points(self):
        with self.assertRaises(InputError):
            helgaker_hf_2pt({3: -76.05})
        with self.assertRaises(InputError):
            helgaker_hf_2pt({3: -76.05, 4: -76.051, 5: -76.0512})


class TestMartin3Pt(unittest.TestCase):
    """``martin_3pt`` solves E(L) = E_CBS + b·(L+½)⁻⁴ + c·(L+½)⁻⁶ exactly."""

    def test_recovers_constant_term(self):
        # If we feed E(L) = -1.0 + 0.05/(L+0.5)**4 + 0.01/(L+0.5)**6 for L=2,3,4
        # then E_CBS must come back as -1.0 to high precision.
        def model(L):
            return -1.0 + 0.05 / (L + 0.5) ** 4 + 0.01 / (L + 0.5) ** 6

        result = martin_3pt({2: model(2), 3: model(3), 4: model(4)})
        self.assertAlmostEqual(result, -1.0, places=10)

    def test_higher_cardinals(self):
        def model(L):
            return -100.0 + 0.123 / (L + 0.5) ** 4 - 0.045 / (L + 0.5) ** 6

        result = martin_3pt({3: model(3), 4: model(4), 5: model(5)})
        self.assertAlmostEqual(result, -100.0, places=10)

    def test_invariance_to_dict_insertion_order(self):
        e = {3: -1.0, 4: -1.05, 5: -1.06}
        a = martin_3pt(e)
        b = martin_3pt({5: e[5], 3: e[3], 4: e[4]})
        self.assertAlmostEqual(a, b, places=12)

    def test_requires_exactly_three_points(self):
        with self.assertRaises(InputError):
            martin_3pt({3: -1.0, 4: -1.05})
        with self.assertRaises(InputError):
            martin_3pt({3: -1.0, 4: -1.05, 5: -1.06, 6: -1.065})


class TestBuiltinFormulasRegistry(unittest.TestCase):
    """The string→callable registry advertised to user input."""

    def test_helgaker_corr_2pt_registered(self):
        self.assertIs(BUILTIN_FORMULAS["helgaker_corr_2pt"], helgaker_corr_2pt)

    def test_helgaker_hf_2pt_registered(self):
        self.assertIs(BUILTIN_FORMULAS["helgaker_hf_2pt"], helgaker_hf_2pt)

    def test_martin_3pt_registered(self):
        self.assertIs(BUILTIN_FORMULAS["martin_3pt"], martin_3pt)

    def test_no_other_entries(self):
        self.assertEqual(
            set(BUILTIN_FORMULAS.keys()),
            {"helgaker_corr_2pt", "helgaker_hf_2pt", "martin_3pt"},
        )


class TestSafeEvalFormula(unittest.TestCase):
    """``safe_eval_formula`` accepts arithmetic + math whitelist; rejects everything else."""

    def test_basic_arithmetic(self):
        self.assertEqual(safe_eval_formula("1 + 2", {}), 3)
        self.assertEqual(safe_eval_formula("3 * 4 - 5", {}), 7)
        self.assertEqual(safe_eval_formula("10 / 4", {}), 2.5)
        self.assertEqual(safe_eval_formula("2 ** 8", {}), 256)
        self.assertEqual(safe_eval_formula("-5 + 3", {}), -2)
        self.assertEqual(safe_eval_formula("+(7)", {}), 7)

    def test_helgaker_corr_2pt_via_safe_eval(self):
        # Reproduce the helgaker_corr_2pt formula by string.
        formula = "(X**3 * E_X - Y**3 * E_Y) / (X**3 - Y**3)"
        env = {"X": 3, "Y": 4, "E_X": -0.30, "E_Y": -0.31}
        result = safe_eval_formula(formula, env)
        self.assertAlmostEqual(result, helgaker_corr_2pt({3: -0.30, 4: -0.31}), places=12)

    def test_allowed_math_calls(self):
        self.assertAlmostEqual(safe_eval_formula("exp(1)", {}), math.e, places=12)
        self.assertAlmostEqual(safe_eval_formula("log(exp(2.5))", {}), 2.5, places=12)
        self.assertAlmostEqual(safe_eval_formula("sqrt(16)", {}), 4.0, places=12)
        self.assertAlmostEqual(safe_eval_formula("pow(2, 10)", {}), 1024.0, places=12)

    def test_user_variables_resolved(self):
        self.assertEqual(safe_eval_formula("E_X * 2", {"E_X": 5}), 10)

    def test_unknown_name_raises(self):
        with self.assertRaises(InputError):
            safe_eval_formula("os.system('rm')", {})
        with self.assertRaises(InputError):
            safe_eval_formula("E_Z", {"E_X": 1})

    def test_dunder_attribute_rejected(self):
        with self.assertRaises(InputError):
            safe_eval_formula("(0).__class__", {})

    def test_attribute_access_rejected(self):
        with self.assertRaises(InputError):
            safe_eval_formula("(0.0).real", {})

    def test_subscript_rejected(self):
        with self.assertRaises(InputError):
            safe_eval_formula("[1,2,3][0]", {})

    def test_lambda_rejected(self):
        with self.assertRaises(InputError):
            safe_eval_formula("(lambda x: x)(1)", {})

    def test_comprehension_rejected(self):
        with self.assertRaises(InputError):
            safe_eval_formula("[i for i in range(3)]", {})

    def test_call_to_unwhitelisted_function_rejected(self):
        with self.assertRaises(InputError):
            safe_eval_formula("eval('1')", {})
        with self.assertRaises(InputError):
            safe_eval_formula("__import__('os')", {})

    def test_walrus_rejected(self):
        with self.assertRaises(InputError):
            safe_eval_formula("(x := 5)", {})

    def test_string_literal_rejected(self):
        # Numeric constants only.
        with self.assertRaises(InputError):
            safe_eval_formula("'hello'", {})

    def test_syntax_error_propagates_as_input_error(self):
        with self.assertRaises(InputError):
            safe_eval_formula("1 +", {})


if __name__ == "__main__":
    unittest.main()
