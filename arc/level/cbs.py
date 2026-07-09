"""
``arc.level.cbs`` — Complete-Basis-Set extrapolation primitives.

This module implements the building blocks needed by
:class:`~arc.level.protocol.CBSExtrapolationTerm`: the cardinal-number deduction from
basis-set names, the three built-in extrapolation formulas shipped with ARC, and a
sandboxed evaluator for user-supplied formula strings.

The CBS step in a focal-point analysis takes ≥2 single-point energies computed at the
*same* method but at *different* basis-set cardinalities X (cc-pVDZ → 2, cc-pVTZ → 3,
cc-pVQZ → 4, ...) and combines them according to a closed-form expression that
extrapolates to the (formally infinite) basis-set limit.

Built-in formulas:

``helgaker_corr_2pt``
    Two-point correlation-energy extrapolation
    ``E_CBS = (X^3·E_X − Y^3·E_Y) / (X^3 − Y^3)``.
    Helgaker, Klopper, Koch, Noga, *J. Chem. Phys.* **106**, 9639 (1997),
    Eq. 4. DOI: 10.1063/1.473863.

``helgaker_hf_2pt``
    Two-point HF-energy extrapolation
    ``E(X) = E_CBS + A·exp(-α·X)``, default ``α = 1.63``.
    Halkier, Helgaker, Jørgensen, Klopper, Olsen,
    *Chem. Phys. Lett.* **302**, 437-446 (1999), "Basis-set convergence of the
    energy in molecular Hartree–Fock calculations".
    DOI: 10.1016/S0009-2614(99)00179-7.

``martin_3pt``
    Three-point Schwartz-style extrapolation
    ``E(L) = E_CBS + b·(L+½)^(-4) + c·(L+½)^(-6)`` solved exactly for the three
    unknowns. Martin, *Chem. Phys. Lett.* **259**, 669-678 (1996), Eq. 5.
    DOI: 10.1016/0009-2614(96)00898-6.

Cardinal numbers follow the Dunning correlation-consistent convention introduced in
Dunning, *J. Chem. Phys.* **90**, 1007 (1989). DOI: 10.1063/1.456153.
"""

import ast
import math
import re
from collections.abc import Callable, Mapping

import numpy as np

from arc.exceptions import InputError


# ----------------------------------------------------------------------------- #
#  Cardinal-number deduction                                                    #
# ----------------------------------------------------------------------------- #

# Map letter labels in correlation-consistent basis sets to cardinal numbers.
# D=2, T=3, Q=4 (Dunning, J. Chem. Phys. 90, 1007 (1989)).
_LETTER_CARDINAL = {"D": 2, "T": 3, "Q": 4}

# Pattern: optional aug- prefix, cc-p, optional C, V, then cardinal letter or digit, Z.
# Accepts cc-pVDZ, cc-pVTZ, cc-pVQZ, cc-pV5Z, cc-pV6Z, cc-pV7Z, cc-pCV*, aug-cc-pV*.
_DUNNING_RE = re.compile(
    r"^(?:aug-)?cc-p(?:c)?v(?P<card>[dtq2-7])z(?:-[a-z0-9]+)?$",
    re.IGNORECASE,
)

# Pattern for the def2 family (Weigend & Ahlrichs): SVP=2, TZVP=3, QZVP=4, plus PP variants.
_DEF2_RE = re.compile(
    r"^def2-(?P<card>s|tz|qz)vp+(?:d?)?$",
    re.IGNORECASE,
)

_DEF2_CARDINAL = {"S": 2, "TZ": 3, "QZ": 4}


def cardinal_from_basis(basis: str) -> int:
    """
    Return the cardinal number X for a correlation-consistent or def2 basis set.

    Args:
        basis (str): Basis-set name (case-insensitive). Supported families:

            * ``cc-pV{D,T,Q,5,6,7}Z`` — Dunning correlation-consistent.
            * ``aug-cc-pV{D,T,Q,5,6,7}Z`` — diffuse-augmented variants.
            * ``cc-pCV{D,T,Q,5,6}Z`` and ``aug-cc-pCV*`` — core-valence variants.
            * ``def2-{SVP,TZVP,QZVP}`` and the ``...PP`` variants (Weigend & Ahlrichs).

    Returns:
        int: Cardinal X (2 for double-zeta, 3 for triple-zeta, etc.).

    Raises:
        InputError: If ``basis`` does not match a known correlation-consistent or
            def2 pattern. CBS extrapolation requires a known cardinal;
            non-systematic basis sets such as ``6-31G*`` or ``STO-3G`` are
            rejected explicitly.
    """
    if not basis:
        raise InputError("Cannot deduce cardinal number from an empty basis-set name.")
    text = basis.strip()
    m = _DUNNING_RE.match(text)
    if m:
        card = m.group("card").upper()
        if card.isdigit():
            return int(card)
        return _LETTER_CARDINAL[card]
    m = _DEF2_RE.match(text)
    if m:
        return _DEF2_CARDINAL[m.group("card").upper()]
    raise InputError(
        f"Cannot deduce a CBS cardinal number from basis '{basis}'. "
        "Only correlation-consistent (cc-pV*Z, aug-cc-pV*Z, cc-pCV*Z) and def2 "
        "(def2-SVP, def2-TZVP, def2-QZVP) families are supported. Use one of "
        "these families for the levels of a cbs_extrapolation term, or add a "
        "new pattern to this function if you need a different basis family."
    )


# ----------------------------------------------------------------------------- #
#  Built-in CBS formulas                                                        #
# ----------------------------------------------------------------------------- #


def _sorted_pairs(energies: Mapping[int, float], expected: int) -> list:
    """
    Return ``[(X, E_X), ...]`` sorted by ascending cardinal, validating the count.

    Mapping keys are unique by construction, so distinct cardinals need no
    separate check: repeated cardinals collapse and fail the count check.

    Args:
        energies (Mapping[int, float]): Mapping ``{cardinal: energy}``.
        expected (int): The exact number of pairs the formula requires.

    Returns:
        list: ``(cardinal, energy)`` tuples sorted by ascending cardinal.

    Raises:
        InputError: If the number of pairs differs from ``expected``.
    """
    pairs = sorted(energies.items())
    if len(pairs) != expected:
        raise InputError(
            f"Expected exactly {expected} (cardinal, energy) pairs, got {len(pairs)}."
        )
    return pairs


def helgaker_corr_2pt(energies: Mapping[int, float]) -> float:
    """
    Two-point correlation-energy CBS extrapolation.

    Implements ``E_CBS = (X³·E_X − Y³·E_Y) / (X³ − Y³)`` per
    Helgaker, Klopper, Koch, Noga, *J. Chem. Phys.* **106**, 9639 (1997), Eq. 4.
    DOI: 10.1063/1.473863.

    Args:
        energies (Mapping[int, float]): Mapping ``{cardinal: energy}`` with exactly
            two entries. Insertion order is irrelevant: pairs are sorted by
            ascending cardinal internally.

    Returns:
        float: Extrapolated energy in the same units as the inputs.
    """
    (X, E_X), (Y, E_Y) = _sorted_pairs(energies, expected=2)
    return (X ** 3 * E_X - Y ** 3 * E_Y) / (X ** 3 - Y ** 3)


def helgaker_hf_2pt(energies: Mapping[int, float], alpha: float = 1.63) -> float:
    """
    Two-point HF (or other exponentially-converging) CBS extrapolation.

    Solves ``E(X) = E_CBS + A·exp(-α·X)`` for two cardinals analytically:
    ``E_CBS = (E_X·exp(-α·Y) − E_Y·exp(-α·X)) / (exp(-α·Y) − exp(-α·X))``.

    Halkier, Helgaker, Jørgensen, Klopper, Olsen, *Chem. Phys. Lett.* **302**,
    437-446 (1999), "Basis-set convergence of the energy in molecular
    Hartree–Fock calculations" reports the fitted value ``α = 1.63`` averaged
    across small molecules. DOI: 10.1016/S0009-2614(99)00179-7.

    Args:
        energies (Mapping[int, float]): Mapping ``{cardinal: energy}`` with exactly
            two entries.
        alpha (float): Exponential decay parameter. Defaults to 1.63
            (Halkier et al. 1999).

    Returns:
        float: Extrapolated energy.
    """
    (X, E_X), (Y, E_Y) = _sorted_pairs(energies, expected=2)
    e_x = math.exp(-alpha * X)
    e_y = math.exp(-alpha * Y)
    return (E_X * e_y - E_Y * e_x) / (e_y - e_x)


def martin_3pt(energies: Mapping[int, float]) -> float:
    """
    Three-point Schwartz-style CBS extrapolation.

    Solves the linear system

        E(L) = E_CBS + b·(L+½)⁻⁴ + c·(L+½)⁻⁶

    exactly for ``E_CBS`` given three (L, E(L)) pairs.

    Martin, *Chem. Phys. Lett.* **259**, 669-678 (1996), Eq. 5.
    DOI: 10.1016/0009-2614(96)00898-6.

    Args:
        energies (Mapping[int, float]): Mapping ``{cardinal: energy}`` with exactly
            three entries.

    Returns:
        float: Extrapolated energy.
    """
    pairs = _sorted_pairs(energies, expected=3)
    A = np.array(
        [[1.0, (L + 0.5) ** -4, (L + 0.5) ** -6] for L, _ in pairs],
        dtype=float,
    )
    b = np.array([E for _, E in pairs], dtype=float)
    e_cbs, _b, _c = np.linalg.solve(A, b)
    return float(e_cbs)


# Single source of truth for the built-in formulas advertised to user input:
# name → (callable, required number of (cardinal, energy) pairs). New built-in
# formulas are added by inserting an entry here (and a corresponding test).
_BUILTIN_FORMULA_SPECS: dict[str, tuple[Callable[..., float], int]] = {
    "helgaker_corr_2pt": (helgaker_corr_2pt, 2),
    "helgaker_hf_2pt": (helgaker_hf_2pt, 2),
    "martin_3pt": (martin_3pt, 3),
}

# Public views of the registry, derived so they can never fall out of sync.
BUILTIN_FORMULAS: dict[str, Callable[..., float]] = {
    name: func for name, (func, _arity) in _BUILTIN_FORMULA_SPECS.items()
}
BUILTIN_FORMULA_ARITY: dict[str, int] = {
    name: arity for name, (_func, arity) in _BUILTIN_FORMULA_SPECS.items()
}


# ----------------------------------------------------------------------------- #
#  Safe AST evaluator for user-supplied formula strings                         #
# ----------------------------------------------------------------------------- #

# Guardrails against resource-exhaustion through user formulas: an exponent
# like ``9**9**9`` passes syntactic validation but creates an astronomically
# large integer, and an oversized or deeply nested expression can exhaust the
# parser or the recursive evaluator. Both are rejected with InputError.
_MAX_POW_EXPONENT = 100
_MAX_AST_NODES = 200

# Functions a user formula may call. Restricted to a tiny math whitelist; no
# I/O, no introspection, no attribute access whatsoever.
_ALLOWED_CALLS = {
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "pow": math.pow,
}

# AST node classes the walker accepts. Anything else is rejected with InputError.
# Notably absent: Attribute, Subscript, Lambda, Comprehensions, NamedExpr (walrus),
# Starred, JoinedStr, FormattedValue, IfExp, Compare, BoolOp.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.UAdd,
    ast.USub,
)


def _validate_ast(node: ast.AST, env_names: set) -> None:
    """
    Raise :class:`InputError` if any descendant of ``node`` is non-whitelisted.

    Args:
        node (ast.AST): The root node to validate.
        env_names (set): Variable names a formula may legally reference.

    Raises:
        InputError: If the tree is oversized or contains any non-whitelisted
            node, constant type, name, or function call.
    """
    n_nodes = sum(1 for _ in ast.walk(node))
    if n_nodes > _MAX_AST_NODES:
        raise InputError(
            f"User formula is too large: {n_nodes} AST nodes exceed the allowed "
            f"maximum of {_MAX_AST_NODES}."
        )
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise InputError(
                f"Disallowed expression element {type(child).__name__!r} in user "
                "formula. Only basic arithmetic (+ - * / ** %), unary +/-, "
                "numeric literals, named variables, and calls to "
                f"{sorted(_ALLOWED_CALLS)} are permitted."
            )
        if isinstance(child, ast.Constant) and not isinstance(child.value, (int, float)):
            raise InputError(
                f"Only numeric constants are allowed in user formulas; got "
                f"{type(child.value).__name__} ({child.value!r})."
            )
        if isinstance(child, ast.Name) and child.id not in env_names \
                and child.id not in _ALLOWED_CALLS:
            raise InputError(
                f"Unknown name '{child.id}' in user formula. Allowed names: "
                f"variables {sorted(env_names)} and functions {sorted(_ALLOWED_CALLS)}."
            )
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in _ALLOWED_CALLS:
                raise InputError(
                    f"Disallowed function call in user formula. Only "
                    f"{sorted(_ALLOWED_CALLS)} may be called."
                )


def _parse_formula(expression: str) -> ast.Expression:
    """
    Parse ``expression`` in eval mode, mapping parser failures to InputError.

    Args:
        expression (str): The user formula string.

    Returns:
        ast.Expression: The parsed AST.

    Raises:
        InputError: If parsing fails — including pathological inputs that
            overflow the CPython parser with RecursionError or MemoryError
            rather than the usual SyntaxError.
    """
    try:
        return ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError, MemoryError) as exc:
        snippet = expression if len(expression) <= 100 else f"{expression[:100]}..."
        raise InputError(
            f"User formula failed to parse: {snippet!r} ({type(exc).__name__}: {exc})"
        )


def validate_formula(expression: str, allowed_names: set) -> None:
    """
    Parse and whitelist-validate ``expression`` without evaluating it.

    Useful at construction time to surface malformed user formulas eagerly,
    independent of any specific numeric inputs (which might cause spurious
    runtime errors like division by zero on a probe environment).

    Args:
        expression (str): The user formula string.
        allowed_names (set): Variable names the formula may reference.

    Raises:
        InputError: On any parse failure or non-whitelisted construct.
    """
    tree = _parse_formula(expression)
    _validate_ast(tree, set(allowed_names))


def safe_eval_formula(expression: str, env: Mapping[str, float]) -> float:
    """
    Evaluate an arithmetic expression against ``env`` without using :func:`eval`.

    Parses ``expression`` to an AST, validates every node against a strict whitelist
    (basic arithmetic, unary ±, numeric literals, named variables drawn from
    ``env``, and calls to :func:`math.exp`, :func:`math.log`, :func:`math.sqrt`,
    :func:`math.pow`), then walks the tree to compute the result.

    Args:
        expression (str): Arithmetic expression. Examples:
            ``"(X**3 * E_X - Y**3 * E_Y) / (X**3 - Y**3)"``,
            ``"E_X - sqrt(E_Y)"``.
        env (Mapping[str, float]): Variable bindings. Names referenced by
            ``expression`` must appear here (or be one of the allowed function names).

    Returns:
        float: Numerical value of the expression.

    Raises:
        InputError: If the expression is syntactically invalid, oversized,
            references unknown names, uses any AST construct outside the
            whitelist (attribute access, subscript, lambdas, comprehensions,
            walrus, string literals, etc.), or exponentiates with a magnitude
            above the allowed maximum.
    """
    tree = _parse_formula(expression)
    env_names = set(env.keys())
    _validate_ast(tree, env_names)
    return _eval_node(tree.body, env)


def _eval_node(node: ast.AST, env: Mapping[str, float]) -> float:
    """Recursively evaluate a whitelisted AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        # _validate_ast already rejected unknown names, so this is unreachable.
        raise InputError(f"Unknown name '{node.id}'.")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise InputError(f"Unsupported unary operator {type(node.op).__name__}.")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            # Cap the exponent magnitude: huge integer exponents (e.g. the
            # outer step of 9**9**9) would hang or exhaust memory.
            if abs(right) > _MAX_POW_EXPONENT:
                raise InputError(
                    f"Exponent {right} in user formula exceeds the allowed "
                    f"magnitude of {_MAX_POW_EXPONENT}."
                )
            return left ** right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        raise InputError(f"Unsupported binary operator {type(node.op).__name__}.")
    if isinstance(node, ast.Call):
        func = _ALLOWED_CALLS[node.func.id]
        args = [_eval_node(a, env) for a in node.args]
        return func(*args)
    raise InputError(f"Unsupported AST node {type(node).__name__}.")
