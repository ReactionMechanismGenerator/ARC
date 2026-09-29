#!/usr/bin/env python3
# encoding: utf-8

"""
Reconstruct Arkane level-of-theory objects from the repr strings ARC matches
against RMG's quantum-corrections data files.

Run from the RMG conda environment so that Arkane is importable.

The keys in ``atom_energies``, ``pbac`` and ``mbac`` are ``LevelOfTheory`` or
``CompositeLevelOfTheory`` objects, and ARC locates them by fuzzy-matching their
repr strings in the data files. Turning a matched string back into the object is
what makes the lookup exact, so it is done by parsing the expression rather than
by scraping keywords out of it.
"""

import ast

from arkane.modelchem import CompositeLevelOfTheory, LevelOfTheory


LEVEL_CONSTRUCTORS = {
    'LevelOfTheory': LevelOfTheory,
    'CompositeLevelOfTheory': CompositeLevelOfTheory,
}


def _build(node):
    """Build a level object, or a literal, from one AST node.

    Returns ``None`` for anything that is not a whitelisted level constructor
    call or a plain literal, so an unexpected expression is declined rather
    than partially evaluated.
    """
    if isinstance(node, ast.Call):
        constructor = LEVEL_CONSTRUCTORS.get(getattr(node.func, 'id', None))
        if constructor is None or node.args:
            return None
        kwargs = dict()
        for keyword in node.keywords:
            if keyword.arg is None:
                return None
            value = _build(keyword.value)
            if value is None:
                return None
            kwargs[keyword.arg] = value
        return constructor(**kwargs)
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def lot_from_string(lot_str):
    """
    Reconstruct a level of theory from its repr string.

    Handles both ``LevelOfTheory(...)`` and nested
    ``CompositeLevelOfTheory(freq=..., energy=...)``, and preserves non-scalar
    keywords such as ``args``.

    Args:
        lot_str (str): The repr string, e.g. taken from a matched data-file key.

    Returns:
        LevelOfTheory | CompositeLevelOfTheory | None: The reconstructed level,
        or ``None`` when the string is not a level expression. ``None`` is
        returned rather than a best-effort object: a level that does not
        round-trip would silently look up a *different* level's parameters.
    """
    if not lot_str:
        return None
    try:
        expression = ast.parse(str(lot_str).strip(), mode='eval')
    except SyntaxError:
        return None
    return _build(expression.body)
