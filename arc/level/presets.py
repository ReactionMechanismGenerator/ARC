"""
``arc.level.presets`` — named composite-protocol presets shipped with ARC.

Presets are loaded from the data file ``presets.yml`` located alongside this module.
Each entry maps a preset name (e.g. ``"HEAT-345Q"``) to a recipe dict in the same
shape that :meth:`arc.level.protocol.CompositeProtocol.from_user_input` accepts
(``base:`` + ``corrections:`` + ``reference:``).

The :func:`expand_preset` helper resolves a preset name (with optional per-term
overrides) to a fresh, independent recipe dict suitable for handing to
``CompositeProtocol.from_user_input``. Returned dicts are deep copies so that
caller-side mutation cannot pollute the cached registry.

References
----------

* Tajti, Szalay, Császár, Kállay, Gauss, Valeev, Flowers, Vázquez, Stanton,
  *J. Chem. Phys.* **121**, 11599 (2004). DOI: 10.1063/1.1804498 — HEAT.
* East, Allen, *J. Chem. Phys.* **99**, 4638 (1993). DOI: 10.1063/1.466062 — focal-
  point analysis methodology.
"""

import copy
import os
from typing import Any, Dict, List, Mapping, Optional

import yaml

from arc.exceptions import InputError


_HERE = os.path.dirname(os.path.abspath(__file__))
_PRESETS_PATH = os.path.join(_HERE, "presets.yml")


def _load_presets(path: str) -> Dict[str, Dict[str, Any]]:
    """Load ``presets.yml`` once; return the parsed mapping."""
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise InputError(f"Preset file {path} must parse to a mapping, got {type(data).__name__}.")
    return data


# Module-level cache. Loaded once at import time; a single source of truth.
PRESETS: Dict[str, Dict[str, Any]] = _load_presets(_PRESETS_PATH)
REGISTERED_PRESET_NAMES: List[str] = sorted(PRESETS.keys())


# Fields that may appear on a preset term by its ``type`` discriminator.
# Used to reject typos in preset overrides (e.g. ``delta_T.hihg``). The key
# ``"base"`` is not a term type — it's the protocol's base level dict, for
# which we accept any Level-level keyword plus ``label``.
_ALLOWED_OVERRIDE_FIELDS_BY_TYPE: Dict[str, set] = {
    "single_point": {"label", "type", "level"},
    "delta": {"label", "type", "high", "low"},
    "cbs_extrapolation": {"label", "type", "formula", "components", "levels"},
}

# Level dict keys — accepted on the ``base`` target and on any ``high``/``low``/
# ``level`` sub-dict the user is replacing wholesale. Kept in sync with
# ``Level.__init__`` parameters (see ``arc/level/level.py``).
_ALLOWED_LEVEL_FIELDS = {
    "repr", "method", "basis", "auxiliary_basis", "dispersion", "cabs",
    "method_type", "software", "software_version", "compatible_ess",
    "solvation_method", "solvent", "solvation_scheme_level", "args", "year",
    # Also valid in the base-of-a-preset context (YAML shorthand):
    "label", "type", "level",
}


def _deep_merge_level_dict(target: Dict[str, Any], patch: Dict[str, Any]) -> None:
    """Shallow-merge ``patch`` into ``target`` with one level of nesting for
    ``high``/``low``/``level`` — replacing fields of the inner dict rather than
    the whole dict. Mutates ``target`` in place.

    Rationale: overriding ``delta_T: {high: {basis: cc-pVTZ}}`` on a preset
    where ``high`` was ``{method: ccsdt, basis: cc-pVDZ}`` should produce
    ``{method: ccsdt, basis: cc-pVTZ}`` — not discard the method. Only the
    nested Level dicts (high/low/level) get this treatment; scalar or
    list-valued fields (formula, levels) still replace wholesale.
    """
    for key, new_val in patch.items():
        existing = target.get(key)
        if (
            key in {"high", "low", "level", "base"}
            and isinstance(existing, dict)
            and isinstance(new_val, dict)
        ):
            merged = dict(existing)
            merged.update(new_val)
            target[key] = merged
        else:
            target[key] = new_val


def _validate_override_fields(term_or_base: Dict[str, Any],
                              patch: Dict[str, Any],
                              target_name: str) -> None:
    """Reject typos in override patch keys.

    For a correction term, patch keys must match the term's ``type``-specific
    allowed fields. For ``base``, patch keys must be valid Level-dict keys
    (plus the usual level-dict extensions).
    """
    if target_name == "base":
        allowed = _ALLOWED_LEVEL_FIELDS
    else:
        term_type = term_or_base.get("type")
        allowed = _ALLOWED_OVERRIDE_FIELDS_BY_TYPE.get(term_type)
        if allowed is None:
            raise InputError(
                f"Cannot validate override for term '{target_name}': its type "
                f"'{term_type}' is not one of "
                f"{sorted(_ALLOWED_OVERRIDE_FIELDS_BY_TYPE)}."
            )
    unknown = set(patch.keys()) - allowed
    if unknown:
        raise InputError(
            f"Override for '{target_name}' has unknown field(s) "
            f"{sorted(unknown)}. Allowed for this target: {sorted(allowed)}."
        )


def _apply_overrides(
    recipe: Dict[str, Any],
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge per-term ``overrides`` into a recipe and return the result.

    ``overrides`` is a mapping ``{term_label: {field_name: new_value}}``. The
    special key ``"base"`` targets the protocol's base level rather than a
    correction.

    * **Unknown term labels** raise :class:`InputError` so a typo can't silently no-op.
    * **Unknown fields within a known term** also raise :class:`InputError` —
      see ``_validate_override_fields``.
    * Nested Level dicts (``high`` / ``low`` / ``level`` / ``base``) are
      **deep-merged** when both old and new values are dicts: overriding
      ``{high: {basis: cc-pVTZ}}`` preserves the existing ``method``. Other
      fields (``formula``, ``levels``, scalar values) replace wholesale.
    """
    if not overrides:
        return recipe

    correction_labels = {c["label"] for c in recipe.get("corrections", [])}
    valid_targets = correction_labels | {"base"}

    for target, patch in overrides.items():
        if target not in valid_targets:
            raise InputError(
                f"Override target '{target}' is not a known term in this preset. "
                f"Valid targets: {sorted(valid_targets)}."
            )
        if not isinstance(patch, dict):
            raise InputError(
                f"Override for '{target}' must be a dict; got {type(patch).__name__}."
            )
        if target == "base":
            _validate_override_fields(recipe.get("base") or {}, patch, target)
            base = recipe["base"]
            if isinstance(base, dict):
                _deep_merge_level_dict(base, patch)
            else:
                # Base was a string shorthand; replace wholesale with the patch dict.
                recipe["base"] = dict(patch)
        else:
            term = next(c for c in recipe["corrections"] if c["label"] == target)
            _validate_override_fields(term, patch, target)
            _deep_merge_level_dict(term, patch)
    return recipe


def expand_preset(
    name: str,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve a preset name (with optional overrides) to an independent recipe dict.

    Parameters
    ----------
    name : str
        One of the keys in :data:`PRESETS`. Lookup is case-sensitive.
    overrides : Mapping[str, Any], optional
        Mapping of term label → field patch. See :func:`_apply_overrides`.

    Returns
    -------
    dict
        A deep-copied recipe dict in the explicit form
        (``{base: ..., corrections: [...]}``) ready to be handed to
        :meth:`arc.level.protocol.CompositeProtocol.from_user_input`.

    Raises
    ------
    arc.exceptions.InputError
        If ``name`` is unknown or the overrides target a non-existent term.
    """
    if name not in PRESETS:
        raise InputError(
            f"Unknown sp_composite preset '{name}'. "
            f"Available presets: {REGISTERED_PRESET_NAMES}."
        )
    recipe = copy.deepcopy(PRESETS[name])
    return _apply_overrides(recipe, overrides or {})
