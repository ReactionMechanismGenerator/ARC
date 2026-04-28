"""
``arc.level.protocol`` — composite-energy protocol data model.

A ``CompositeProtocol`` describes how to compute the final electronic energy of a
stationary point as a sum of contributions, each evaluated at a different level of
theory. The motivation is HEAT-style focal-point analysis (Tajti, Szalay, Császár,
Kállay, Gauss, Valeev, Flowers, Vázquez, Stanton, *J. Chem. Phys.* **121**, 11599
(2004); DOI: 10.1063/1.1811608) and CBS extrapolation (Helgaker et al. 1997, Halkier
et al. 1998, Martin 1996), where small post-CCSD(T) corrections accumulate to
several kJ/mol — exactly the range that affects TS barriers in kinetics.

Data model
----------

A ``CompositeProtocol`` consists of:

* ``base`` — a single :class:`SinglePointTerm` providing the absolute electronic
  energy. By convention this is the "main" SP that the scheduler runs first; it is
  also the level used for AEC (atom-energy-correction) lookups when the protocol
  is wired into Arkane in a later phase.
* ``corrections`` — an ordered list of additional :class:`Term` objects of any
  subtype: :class:`SinglePointTerm`, :class:`DeltaTerm`, or
  :class:`CBSExtrapolationTerm`.

The final energy is ``base.evaluate(...) + Σ correction.evaluate(...)``.

Sub-job naming
--------------

Each ``Term`` describes the QM single-point jobs it needs via
:meth:`Term.required_levels`, returning ``[(sub_label, Level), ...]`` pairs. The
sub_labels are *globally* unique within the protocol and follow the convention:

* ``SinglePointTerm`` → ``"<term_label>"`` (one sub-job).
* ``DeltaTerm`` → ``"<term_label>__high"``, ``"<term_label>__low"``.
* ``CBSExtrapolationTerm`` → ``"<term_label>__card_<X>"`` for each cardinal ``X``.

The Phase 2 scheduler integration uses these sub_labels to track per-sub-job state
across restarts.
"""

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from arc.exceptions import InputError
from arc.level.cbs import (
    BUILTIN_FORMULAS,
    cardinal_from_basis,
    safe_eval_formula,
    validate_formula,
)
from arc.level.level import Level
from arc.level.presets import expand_preset


# --------------------------------------------------------------------------- #
#  Term hierarchy                                                             #
# --------------------------------------------------------------------------- #


class Term(ABC):
    """Abstract base class for any contribution to a composite electronic energy.

    A ``Term`` knows three things:

    1. Its ``label`` — a unique name used by the scheduler and reporter to
       identify the term in logs and the provenance notebook.
    2. The QM sub-jobs it needs, via :meth:`required_levels`.
    3. How to combine those sub-jobs' parsed energies into a single number, via
       :meth:`evaluate`.
    """

    label: str

    @abstractmethod
    def required_levels(self) -> List[Tuple[str, Level]]:
        """Return ``[(sub_label, Level), ...]`` pairs for every SP this term needs."""

    @abstractmethod
    def evaluate(self, energies: Dict[str, float]) -> float:
        """Combine sub-job energies into this term's contribution.

        The keys of ``energies`` are the ``sub_label`` strings yielded by
        :meth:`required_levels`. Units are passed through unchanged (kJ/mol in the
        ARC scheduler, but the data model is unit-agnostic).
        """

    @abstractmethod
    def as_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON/YAML-friendly dict including a discriminator ``type``."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Term":
        """Reconstruct a ``Term`` subclass from its serialised dict.

        Dispatches on the ``type`` discriminator written by :meth:`as_dict`.
        """
        if not isinstance(data, dict) or "type" not in data:
            raise InputError(
                "Term dict must include a 'type' discriminator "
                "('single_point', 'delta', or 'cbs_extrapolation')."
            )
        kind = data["type"]
        if kind == "single_point":
            return SinglePointTerm._from_dict(data)
        if kind == "delta":
            return DeltaTerm._from_dict(data)
        if kind == "cbs_extrapolation":
            return CBSExtrapolationTerm._from_dict(data)
        raise InputError(
            f"Unknown term type '{kind}'. Allowed: "
            "'single_point', 'delta', 'cbs_extrapolation'."
        )


def _coerce_level(value: Union[str, Dict[str, Any], Level]) -> Level:
    """Accept either a string, dict, or Level; return a Level instance."""
    if isinstance(value, Level):
        return value
    if isinstance(value, (str, dict)):
        return Level(repr=value)
    raise InputError(
        f"Cannot interpret {value!r} (type {type(value).__name__}) as a Level."
    )


class SinglePointTerm(Term):
    """One absolute single-point energy at one level of theory."""

    def __init__(self, label: str, level: Union[str, Dict[str, Any], Level]):
        if not label:
            raise InputError("SinglePointTerm requires a non-empty label.")
        self.label = label
        self.level = _coerce_level(level)

    def required_levels(self) -> List[Tuple[str, Level]]:
        return [(self.label, self.level)]

    def evaluate(self, energies: Dict[str, float]) -> float:
        return energies[self.label]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": "single_point",
            "label": self.label,
            "level": self.level.as_dict(),
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "SinglePointTerm":
        return cls(label=data["label"], level=data["level"])


class DeltaTerm(Term):
    """A correction ``E[high] − E[low]`` between two levels of theory.

    Used to capture, e.g., the post-(T) correction
    ``δ[CCSDT] = E[CCSDT/cc-pVDZ] − E[CCSD(T)/cc-pVDZ]``.
    """

    def __init__(
        self,
        label: str,
        high: Optional[Union[str, Dict[str, Any], Level]],
        low: Optional[Union[str, Dict[str, Any], Level]],
    ):
        if not label:
            raise InputError("DeltaTerm requires a non-empty label.")
        if high is None or low is None:
            raise InputError(
                f"DeltaTerm '{label}' requires both 'high' and 'low' levels; "
                f"got high={high!r}, low={low!r}."
            )
        self.label = label
        self.high = _coerce_level(high)
        self.low = _coerce_level(low)

    def _sub(self, suffix: str) -> str:
        return f"{self.label}__{suffix}"

    def required_levels(self) -> List[Tuple[str, Level]]:
        return [(self._sub("high"), self.high), (self._sub("low"), self.low)]

    def evaluate(self, energies: Dict[str, float]) -> float:
        return energies[self._sub("high")] - energies[self._sub("low")]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": "delta",
            "label": self.label,
            "high": self.high.as_dict(),
            "low": self.low.as_dict(),
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "DeltaTerm":
        return cls(label=data["label"], high=data["high"], low=data["low"])


# Currently only "total" is supported: the energies fed to CBS formulas come
# from ``arc.parser.parse_e_elect``, which returns the total electronic energy
# of a single-point job. There is no parser pathway that surfaces correlation-
# only or HF-only components yet, so accepting ``components='corr'`` or
# ``'hf'`` would silently extrapolate *total* energies while pretending to be
# component-specific — a correctness hazard. When adapter-level component
# parsing is added, widen this tuple and add tests that the right component is
# actually routed per sub-job.
_ALLOWED_COMPONENTS = ("total",)


class CBSExtrapolationTerm(Term):
    """Complete-Basis-Set extrapolated contribution.

    Computes one term in the composite from ≥2 single-point energies at the same
    method but different basis-set cardinalities, combined via a closed-form
    formula. ``formula`` may be the name of a built-in
    (:data:`arc.level.cbs.BUILTIN_FORMULAS`) or a user-supplied arithmetic
    expression evaluated by :func:`arc.level.cbs.safe_eval_formula`.

    Parameters
    ----------
    label : str
        Term identifier.
    formula : str
        Built-in name or arithmetic expression. User expressions may reference
        ``X``, ``Y``, ``Z`` (cardinal numbers) and ``E_X``, ``E_Y``, ``E_Z``
        (corresponding energies), bound by ascending cardinal order.
        User formulas with more than 3 levels are rejected: expose only the
        first three cardinal variables we bind.
    levels : list of Level
        ≥2 levels, all with the same method, all with deducible distinct cardinals.
    components : {'total'}
        Which energy component the extrapolation applies to. **Only ``'total'``
        is currently accepted.** Other values are rejected at construction time
        until component-specific parsing exists — see ``_ALLOWED_COMPONENTS``
        above for rationale.
    """

    def __init__(
        self,
        label: str,
        formula: str,
        levels: List[Union[str, Dict[str, Any], Level]],
        components: str = "total",
    ):
        if not label:
            raise InputError("CBSExtrapolationTerm requires a non-empty label.")
        if components not in _ALLOWED_COMPONENTS:
            raise InputError(
                f"CBSExtrapolationTerm '{label}': components={components!r} not in "
                f"{_ALLOWED_COMPONENTS}."
            )
        coerced = [_coerce_level(lvl) for lvl in levels]
        if len(coerced) < 2:
            raise InputError(
                f"CBSExtrapolationTerm '{label}' needs at least 2 levels, got {len(coerced)}."
            )
        methods = {lvl.method for lvl in coerced}
        if len(methods) > 1:
            raise InputError(
                f"CBSExtrapolationTerm '{label}': all levels must share one method, "
                f"got {sorted(methods)}."
            )
        cardinals = [cardinal_from_basis(lvl.basis) for lvl in coerced]
        if len(set(cardinals)) != len(cardinals):
            raise InputError(
                f"CBSExtrapolationTerm '{label}': cardinals must be distinct, got "
                f"{cardinals}."
            )
        # Sort levels and cardinals together by ascending cardinal so callers can rely
        # on a canonical ordering downstream.
        ordered = sorted(zip(cardinals, coerced))
        self._cardinals = [c for c, _ in ordered]
        self.levels = [lvl for _, lvl in ordered]
        self.label = label
        self.components = components
        self.formula = formula
        self._formula_callable = self._resolve_formula(formula, len(self.levels))

    # Arity required by each shipped built-in formula. Surfacing this at
    # construction time catches "martin_3pt with 2 levels" before a sub-job
    # ever runs. When new built-ins are added, update this table alongside
    # the entry in arc.level.cbs.BUILTIN_FORMULAS.
    _BUILTIN_FORMULA_ARITY: Dict[str, int] = {
        "helgaker_corr_2pt": 2,
        "helgaker_hf_2pt": 2,
        "martin_3pt": 3,
    }

    # Upper bound for user-supplied formula arity: the safe-eval variable
    # binder exposes only X/Y/Z (and E_X/E_Y/E_Z). Supporting more would
    # require extending both the binder and the safe-eval allow-list tests.
    _USER_FORMULA_MAX_LEVELS = 3

    @staticmethod
    def _resolve_formula(formula: str, n_levels: int):
        """Validate ``formula`` against the built-in registry and (if user-supplied)
        the safe-eval whitelist; return a callable taking ``{cardinal: energy}``.

        Built-in formulas additionally have their required arity enforced here
        (Phase 5.5) so a recipe with the wrong number of levels fails at
        construction, not at sub-job-completion time.
        """
        if formula in BUILTIN_FORMULAS:
            required = CBSExtrapolationTerm._BUILTIN_FORMULA_ARITY.get(formula)
            if required is not None and n_levels != required:
                raise InputError(
                    f"Built-in CBS formula '{formula}' requires exactly "
                    f"{required} levels; got {n_levels}."
                )
            return BUILTIN_FORMULAS[formula]
        # User expression: validate the AST eagerly so malformed formulas raise
        # at construction, not when sub-job energies are first plugged in. We
        # advertise X/Y/Z and E_X/E_Y/E_Z up to the number of levels.
        if n_levels > CBSExtrapolationTerm._USER_FORMULA_MAX_LEVELS:
            raise InputError(
                f"User CBS formulas currently support at most "
                f"{CBSExtrapolationTerm._USER_FORMULA_MAX_LEVELS} levels "
                f"(X/Y/Z and E_X/E_Y/E_Z variables); got {n_levels}."
            )
        allowed = {f"E_{var}" for var in ("X", "Y", "Z")[:n_levels]}
        allowed.update({var for var in ("X", "Y", "Z")[:n_levels]})
        validate_formula(formula, allowed)

        def _user_fn(energies):
            env = {}
            for idx, (X, E) in enumerate(sorted(energies.items())):
                var = ("X", "Y", "Z")[idx]
                env[var] = X
                env[f"E_{var}"] = E
            return safe_eval_formula(formula, env)

        return _user_fn

    def _sub(self, cardinal: int) -> str:
        return f"{self.label}__card_{cardinal}"

    def required_levels(self) -> List[Tuple[str, Level]]:
        return [(self._sub(c), lvl) for c, lvl in zip(self._cardinals, self.levels)]

    def evaluate(self, energies: Dict[str, float]) -> float:
        cardinal_to_energy = {c: energies[self._sub(c)] for c in self._cardinals}
        return self._formula_callable(cardinal_to_energy)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": "cbs_extrapolation",
            "label": self.label,
            "formula": self.formula,
            "components": self.components,
            "levels": [lvl.as_dict() for lvl in self.levels],
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "CBSExtrapolationTerm":
        return cls(
            label=data["label"],
            formula=data["formula"],
            levels=data["levels"],
            components=data.get("components", "total"),
        )


# --------------------------------------------------------------------------- #
#  CompositeProtocol                                                          #
# --------------------------------------------------------------------------- #


class CompositeProtocol:
    """An ordered sum of :class:`Term` objects defining the final electronic energy.

    The protocol's electronic energy is ``base.evaluate(...) + Σ correction.evaluate(...)``.

    Optional metadata:

    * ``preset_name`` — the name of the preset this protocol was expanded from
      (``"HEAT-345Q"`` etc.), or ``None`` for explicit recipes. Populated
      automatically by :meth:`from_user_input` when the input is a preset name
      or a ``{preset: ..., overrides: ...}`` dict; carried through ``as_dict``
      and restored by ``from_dict``.
    * ``reference`` — a citation string (typically a DOI) describing the source
      of the protocol. For presets, this comes from ``presets.yml``'s
      ``reference:`` field; for explicit recipes, users may supply a
      ``reference:`` key at the top level of their recipe dict.
    """

    def __init__(
        self,
        base: SinglePointTerm,
        corrections: Optional[List[Term]] = None,
        preset_name: Optional[str] = None,
        reference: Optional[str] = None,
    ):
        if not isinstance(base, SinglePointTerm):
            raise InputError(
                "CompositeProtocol.base must be a SinglePointTerm; "
                f"got {type(base).__name__}."
            )
        corrections = list(corrections) if corrections else []
        labels = [base.label] + [t.label for t in corrections]
        if len(set(labels)) != len(labels):
            raise InputError(
                f"All term labels must be unique within a CompositeProtocol; "
                f"got duplicates in {labels}."
            )
        # sub_labels are a *global* namespace within a protocol — they key the
        # scheduler's pending dict and the output-dict's 'paths/sp_composite'.
        # A collision (e.g. SinglePointTerm(label='delta_T__high') plus a
        # DeltaTerm(label='delta_T', ...) whose 'high' sub-leg also ends up as
        # 'delta_T__high') would overwrite state silently. Reject at construction.
        sub_labels: List[str] = []
        for term in [base, *corrections]:
            for sub_label, _level in term.required_levels():
                sub_labels.append(sub_label)
        if len(set(sub_labels)) != len(sub_labels):
            duplicates = sorted({s for s in sub_labels if sub_labels.count(s) > 1})
            raise InputError(
                f"CompositeProtocol has colliding sub_labels across terms: "
                f"{duplicates}. Rename the offending term(s) so their "
                f"sub_labels ('<label>', '<label>__high', '<label>__low', "
                f"'<label>__card_<X>') don't clash."
            )
        self.base = base
        self.corrections = corrections
        self.preset_name = preset_name
        self.reference = reference

    @property
    def terms(self) -> List[Term]:
        """Convenience: ``[base, *corrections]`` in protocol order."""
        return [self.base, *self.corrections]

    def evaluate(self, energies: Dict[str, float]) -> float:
        """Combine all sub-job energies into the protocol's electronic energy."""
        return sum(term.evaluate(energies) for term in self.terms)

    def iter_required_jobs(self) -> Iterable[Tuple[str, str, Level]]:
        """Yield ``(term_label, sub_label, Level)`` triples for every required SP."""
        for term in self.terms:
            for sub_label, level in term.required_levels():
                yield (term.label, sub_label, level)

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "base": self.base.as_dict(),
            "corrections": [t.as_dict() for t in self.corrections],
        }
        if self.preset_name is not None:
            out["preset_name"] = self.preset_name
        if self.reference is not None:
            out["reference"] = self.reference
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompositeProtocol":
        """Inverse of :meth:`as_dict`. Each entry must already include its discriminator."""
        if not isinstance(data, dict) or "base" not in data:
            raise InputError(
                "CompositeProtocol dict must include a 'base' entry."
            )
        base_data = data["base"]
        if isinstance(base_data, dict) and "type" not in base_data:
            base = SinglePointTerm(
                label=base_data.get("label", "base"),
                level=base_data.get("level", base_data),
            )
        else:
            term = Term.from_dict(base_data)
            if not isinstance(term, SinglePointTerm):
                raise InputError("CompositeProtocol.base must be a SinglePointTerm.")
            base = term
        corrections = [Term.from_dict(d) for d in data.get("corrections", [])]
        return cls(
            base=base,
            corrections=corrections,
            preset_name=data.get("preset_name"),
            reference=data.get("reference"),
        )

    @classmethod
    def from_user_input(cls, raw: Union[str, Dict[str, Any]]) -> "CompositeProtocol":
        """Accept the YAML-shaped user input and produce a validated protocol.

        Three forms are accepted:

        * A *string* — interpreted as a preset name. Delegates to
          :func:`arc.level.presets.expand_preset`.
        * A *dict with* ``preset:`` *key* — preset with overrides. Delegates to
          :func:`arc.level.presets.expand_preset`.
        * A *dict with* ``base:`` *and* ``corrections:`` keys — fully explicit
          recipe. ``base`` may be a Level shorthand (``"method/basis"``), a Level
          dict, or a serialised SinglePointTerm. Each entry in ``corrections``
          must include a ``type`` discriminator.

        Raises
        ------
        arc.exceptions.InputError
            On any malformed input.
        """
        preset_name: Optional[str] = None
        if isinstance(raw, str):
            preset_name = raw
            raw = expand_preset(raw)
        elif isinstance(raw, dict) and "preset" in raw:
            preset_name = raw["preset"]
            raw = expand_preset(raw["preset"], overrides=raw.get("overrides"))
        elif isinstance(raw, dict):
            # Explicit recipe (or serialised ``as_dict()`` output). Deep-copy so
            # we never mutate caller-owned state — downstream code pops fields
            # off dicts when reading legacy forms.
            raw = copy.deepcopy(raw)
        else:
            raise InputError(
                f"sp_composite must be a preset name (str) or a dict; "
                f"got {type(raw).__name__}."
            )

        if "base" not in raw:
            raise InputError("sp_composite recipe must include a 'base' level.")

        # Preserve preset_name when the caller handed us a serialised ``as_dict()``
        # payload (which carries the 'preset_name' key from a prior
        # ``from_user_input`` run). Preset name from the current call wins.
        if preset_name is None:
            preset_name = raw.get("preset_name")
        reference = raw.get("reference")

        # base may be: string ("method/basis"), Level dict, or explicit
        # SinglePointTerm dict (with type='single_point').
        base_raw = raw["base"]
        if isinstance(base_raw, str):
            base = SinglePointTerm(label="base", level=base_raw)
        elif isinstance(base_raw, dict) and base_raw.get("type") == "single_point":
            base = SinglePointTerm(
                label=base_raw.get("label", "base"),
                level=base_raw["level"],
            )
        elif isinstance(base_raw, dict):
            # Legacy/shorthand form: the dict is a Level spec that optionally
            # carries a 'label' key. Separate them without mutating the caller's
            # state (base_raw is already a deep copy from above, but be explicit).
            base_dict = {k: v for k, v in base_raw.items() if k != "label"}
            label = base_raw.get("label", "base")
            base = SinglePointTerm(label=label, level=base_dict)
        else:
            raise InputError(
                f"sp_composite 'base' must be a level string, level dict, or "
                f"SinglePointTerm dict; got {type(base_raw).__name__}."
            )

        corrections: List[Term] = []
        for entry in raw.get("corrections", []):
            if not isinstance(entry, dict):
                raise InputError(
                    f"Each correction must be a dict; got {type(entry).__name__}."
                )
            corrections.append(Term.from_dict(entry))
        return cls(
            base=base,
            corrections=corrections,
            preset_name=preset_name,
            reference=reference,
        )


# --------------------------------------------------------------------------- #
#  Public adapter                                                             #
# --------------------------------------------------------------------------- #


def build_protocol(value: Any) -> CompositeProtocol:
    """Coerce any supported user input into a :class:`CompositeProtocol`.

    * If ``value`` is already a :class:`CompositeProtocol`, returns it unchanged.
    * If it is a string or dict, delegates to
      :meth:`CompositeProtocol.from_user_input`.
    * Otherwise raises :class:`InputError`.
    """
    if isinstance(value, CompositeProtocol):
        return value
    if isinstance(value, (str, dict)):
        return CompositeProtocol.from_user_input(value)
    raise InputError(
        f"Cannot build a CompositeProtocol from {type(value).__name__}; "
        "expected str, dict, or CompositeProtocol."
    )
