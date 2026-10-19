"""
``arc.level.protocol`` — composite-energy protocol data model.

A ``CompositeProtocol`` describes how to compute the final electronic energy of a
stationary point as a sum of contributions, each evaluated at a different level of
theory. The motivation is HEAT-style focal-point analysis (Tajti, Szalay, Császár,
Kállay, Gauss, Valeev, Flowers, Vázquez, Stanton, *J. Chem. Phys.* **121**, 11599
(2004); DOI: 10.1063/1.1811608) and CBS extrapolation (Helgaker et al. 1997, Halkier
et al. 1998, Martin 1996), where small post-CCSD(T) corrections accumulate to
several kJ/mol — exactly the range that affects TS barriers in kinetics.

Data model:

A ``CompositeProtocol`` consists of:

* ``base`` — a :class:`Term` providing the absolute electronic energy: either a
  :class:`SinglePointTerm` (one anchor SP) or a :class:`CBSExtrapolationTerm`
  (the canonical FPA shape — the absolute energy is the CBS-extrapolated value
  of ≥2 SPs at increasing basis cardinality). The protocol's
  ``primary_base_level`` (the single level, or the largest-cardinal leg for a
  CBS base) is the level used for AEC (atom-energy-correction) lookups and for
  ``sp_level`` defaulting in :mod:`arc.main`.
* ``corrections`` — an ordered list of additional :class:`Term` objects:
  :class:`SinglePointTerm` or :class:`DeltaTerm`. A
  :class:`CBSExtrapolationTerm` is *not* accepted as a correction: with
  ``components='total'`` (the only supported value) it evaluates to an absolute
  energy, which would double-count the base. Use it as the ``base`` instead.

The final energy is ``base.evaluate(...) + Σ correction.evaluate(...)``.

Sub-job naming:

Each ``Term`` describes the QM single-point jobs it needs via
:meth:`Term.required_levels`, returning ``[(sub_label, Level), ...]`` pairs. The
sub_labels are *globally* unique within the protocol and follow the convention:

* ``SinglePointTerm`` → ``"<term_label>"`` (one sub-job).
* ``DeltaTerm`` → ``"<term_label>__high"``, ``"<term_label>__low"``.
* ``CBSExtrapolationTerm`` → ``"<term_label>__card_<X>"`` for each cardinal ``X``.

The scheduler integration uses these sub_labels to track per-sub-job state
across restarts.
"""

import copy
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

from arc.exceptions import InputError
from arc.level.cbs import (
    BUILTIN_FORMULA_ARITY,
    BUILTIN_FORMULAS,
    cardinal_from_basis,
    safe_eval_formula,
    validate_formula,
)
from arc.level.level import Level
from arc.level.presets import expand_preset


# --------------------------------------------------------------------------- #
#  Coupled-cluster excitation ranks                                           #
# --------------------------------------------------------------------------- #


_CC_RANK_BY_LETTER = {'s': 1, 'd': 2, 't': 3, 'q': 4, 'p': 5}
_CC_METHOD_REGEX = re.compile(r'^[ur]?cc(?P<iterative>sd[tq]*)(?:\((?P<perturbative>[tqp])\))?$')
_F12_SUFFIX_REGEX = re.compile(r'-f12[abx]?$')


def _parse_cc_method(method: str) -> tuple[int, int] | None:
    """
    Parse a coupled-cluster method string into its excitation ranks.

    Case-insensitive; explicitly-correlated ``-f12`` (``-f12a``/``-f12b``)
    suffixes and ``u``/``r`` spin-restriction prefixes are stripped.

    Args:
        method (str): The method name, e.g. ``'CCSD(T)-F12'`` or ``'ccsdt(q)'``.

    Returns:
        tuple[int, int] | None: ``(total_rank, iterative_rank)`` where the total
        rank counts a perturbative top level (the ``(T)`` in CCSD(T)) the same
        as an iterative one, or ``None`` if ``method`` is not a recognised
        coupled-cluster method.
    """
    stripped = _F12_SUFFIX_REGEX.sub('', method.strip().lower())
    match = _CC_METHOD_REGEX.match(stripped)
    if match is None:
        return None
    iterative_rank = max(_CC_RANK_BY_LETTER[char] for char in match.group('iterative'))
    perturbative = match.group('perturbative')
    total_rank = max(iterative_rank, _CC_RANK_BY_LETTER[perturbative]) if perturbative \
        else iterative_rank
    return total_rank, iterative_rank


def excitation_rank(method: str) -> int | None:
    """
    Return the coupled-cluster excitation rank of a method string.

    ``ccsd`` → 2, ``ccsd(t)`` / ``ccsdt`` → 3, ``ccsdt(q)`` / ``ccsdtq`` → 4,
    ``ccsdtq(p)`` → 5. Perturbative top levels count the same as iterative ones.

    Args:
        method (str): The method name (case-insensitive; ``-f12`` suffixes stripped).

    Returns:
        int | None: The excitation rank, or ``None`` for non-CC methods.
    """
    parsed = _parse_cc_method(method)
    return parsed[0] if parsed is not None else None


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

    Terms whose :meth:`evaluate` yields an *absolute* electronic energy (rather
    than a difference) set ``provides_absolute_energy = True`` and may serve as
    a :class:`CompositeProtocol` base. Every term also exposes a *primary* leg —
    its most complete sub-job — via :attr:`primary_sub_label` /
    :attr:`primary_level`, so callers never need to dispatch on term type.
    """

    label: str

    # True for terms whose evaluate() returns an absolute electronic energy
    # (SinglePointTerm, CBSExtrapolationTerm); False for difference terms.
    provides_absolute_energy: bool = False

    @abstractmethod
    def required_levels(self) -> list[tuple[str, Level]]:
        """Return ``[(sub_label, Level), ...]`` pairs for every SP this term needs."""

    @property
    @abstractmethod
    def primary_level(self) -> Level:
        """The Level of this term's primary (most complete) sub-job."""

    @property
    @abstractmethod
    def primary_sub_label(self) -> str:
        """The sub_label of this term's primary (most complete) sub-job."""

    @abstractmethod
    def evaluate(self, energies: dict[str, float]) -> float:
        """Combine sub-job energies into this term's contribution.

        The keys of ``energies`` are the ``sub_label`` strings yielded by
        :meth:`required_levels`. Units are passed through unchanged (kJ/mol in the
        ARC scheduler, but the data model is unit-agnostic).
        """

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Serialise to a JSON/YAML-friendly dict including a discriminator ``type``."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Term":
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


def _coerce_level(value: str | dict[str, Any] | Level) -> Level:
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

    provides_absolute_energy = True

    def __init__(self, label: str, level: str | dict[str, Any] | Level):
        if not label:
            raise InputError("SinglePointTerm requires a non-empty label.")
        self.label = label
        self.level = _coerce_level(level)

    @property
    def primary_level(self) -> Level:
        """The term's only Level."""
        return self.level

    @property
    def primary_sub_label(self) -> str:
        """The term's only sub_label (its label)."""
        return self.label

    def required_levels(self) -> list[tuple[str, Level]]:
        return [(self.label, self.level)]

    def evaluate(self, energies: dict[str, float]) -> float:
        return energies[self.label]

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "single_point",
            "label": self.label,
            "level": self.level.as_dict(),
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "SinglePointTerm":
        return cls(label=data["label"], level=data["level"])


class DeltaTerm(Term):
    """A correction ``E[high] − E[low]`` between two levels of theory.

    Used to capture, e.g., the post-(T) correction
    ``δ[CCSDT] = E[CCSDT/cc-pVDZ] − E[CCSD(T)/cc-pVDZ]``.
    """

    def __init__(
        self,
        label: str,
        high: str | dict[str, Any] | Level | None,
        low: str | dict[str, Any] | Level | None,
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

    @property
    def primary_level(self) -> Level:
        """The high leg's Level."""
        return self.high

    @property
    def primary_sub_label(self) -> str:
        """The high leg's sub_label."""
        return self._sub("high")

    def required_levels(self) -> list[tuple[str, Level]]:
        return [(self._sub("high"), self.high), (self._sub("low"), self.low)]

    def evaluate(self, energies: dict[str, float]) -> float:
        return energies[self._sub("high")] - energies[self._sub("low")]

    def is_trivially_zero(self, n_correlated: int) -> bool:
        """
        Whether ``E[high] − E[low]`` is identically zero for a species with
        ``n_correlated`` correlated electrons.

        A coupled-cluster method reproduces the exact (FCI) energy when its
        *iterative* excitation rank covers every correlated electron; a
        perturbative top level — the ``(T)`` in CCSD(T) — then vanishes too,
        because excitations beyond the electron count cannot exist. The delta
        is provably zero only when BOTH legs are exact by this criterion; a
        non-CC leg (HF, MP2, DFT) is never considered exact. For a high leg
        with a perturbative top this reduces to the rank test
        ``n_correlated < excitation_rank(high.method)``.

        Args:
            n_correlated (int): The species' correlated electron count
                (total electrons minus two per frozen-core orbital).

        Returns:
            bool: ``True`` if the delta correction is identically zero.
        """
        high = _parse_cc_method(self.high.method)
        low = _parse_cc_method(self.low.method)
        if high is None or low is None:
            return False
        return n_correlated <= high[1] and n_correlated <= low[1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "delta",
            "label": self.label,
            "high": self.high.as_dict(),
            "low": self.low.as_dict(),
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "DeltaTerm":
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

    With ``components='total'`` (the only supported value) the formula is applied
    to TOTAL electronic energies, so the term evaluates to an *absolute* CBS
    energy — it serves as a :class:`CompositeProtocol` base, not as a correction.

    A note on the formula choice: a two-point ``X^-3`` extrapolation (e.g.
    ``helgaker_corr_2pt``) applied to total energies technically mis-treats the
    HF component, which converges exponentially with cardinal number rather than
    as ``X^-3`` — the formula was derived for the correlation energy alone. At
    {T,Q} and higher cardinals the HF residual is small and extrapolating totals
    this way is common practice; the residual error is well below the other
    approximations in a typical focal-point stack. For a strictly component-wise
    treatment, wait for (or contribute) adapter-level HF/correlation component
    parsing, which will unlock ``components='hf'`` / ``'corr'``.

    Args:
        label (str): Term identifier.
        formula (str): Built-in name or arithmetic expression. User expressions
            may reference ``X``, ``Y``, ``Z`` (cardinal numbers) and ``E_X``,
            ``E_Y``, ``E_Z`` (corresponding energies), bound by ascending
            cardinal order. User formulas with more than 3 levels are rejected:
            expose only the first three cardinal variables we bind.
        levels (list): ≥2 levels, all with the same method, all with deducible
            distinct cardinals.
        components (str): Which energy component the extrapolation applies to.
            **Only ``'total'`` is currently accepted.** Other values are
            rejected at construction time until component-specific parsing
            exists — see ``_ALLOWED_COMPONENTS`` above for rationale.
    """

    def __init__(
        self,
        label: str,
        formula: str,
        levels: list[str | dict[str, Any] | Level],
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

    # Upper bound for user-supplied formula arity: the safe-eval variable
    # binder exposes only X/Y/Z (and E_X/E_Y/E_Z). Supporting more would
    # require extending both the binder and the safe-eval allow-list tests.
    _USER_FORMULA_MAX_LEVELS = 3

    @staticmethod
    def _resolve_formula(formula: str, n_levels: int) -> Callable[[dict[int, float]], float]:
        """
        Validate ``formula`` against the built-in registry and (if user-supplied)
        the safe-eval whitelist; return a callable taking ``{cardinal: energy}``.

        Built-in formulas additionally have their required arity (from
        :data:`arc.level.cbs.BUILTIN_FORMULA_ARITY`) enforced here so a recipe
        with the wrong number of levels fails at construction, not at
        sub-job-completion time.

        Args:
            formula (str): Built-in formula name or user arithmetic expression.
            n_levels (int): Number of levels supplied to the term.

        Returns:
            Callable[[dict[int, float]], float]: The formula callable.

        Raises:
            InputError: If a built-in formula's arity doesn't match ``n_levels``,
                or a user formula is malformed or uses too many levels.
        """
        if formula in BUILTIN_FORMULAS:
            required = BUILTIN_FORMULA_ARITY[formula]
            if n_levels != required:
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

        def _user_fn(energies: dict[int, float]) -> float:
            env: dict[str, float] = {}
            for idx, (X, E) in enumerate(sorted(energies.items())):
                var = ("X", "Y", "Z")[idx]
                env[var] = X
                env[f"E_{var}"] = E
            try:
                return safe_eval_formula(formula, env)
            except ZeroDivisionError as exc:
                raise InputError(
                    f"User CBS formula {formula!r} raised division by zero "
                    f"for inputs {energies}."
                ) from exc

        return _user_fn

    provides_absolute_energy = True

    def _sub(self, cardinal: int) -> str:
        return f"{self.label}__card_{cardinal}"

    @property
    def primary_level(self) -> Level:
        """The largest-cardinal leg's Level (levels are sorted ascending)."""
        return self.levels[-1]

    @property
    def primary_sub_label(self) -> str:
        """The largest-cardinal leg's sub_label."""
        return self._sub(self._cardinals[-1])

    def required_levels(self) -> list[tuple[str, Level]]:
        return [(self._sub(c), lvl) for c, lvl in zip(self._cardinals, self.levels)]

    def evaluate(self, energies: dict[str, float]) -> float:
        cardinal_to_energy = {c: energies[self._sub(c)] for c in self._cardinals}
        return self._formula_callable(cardinal_to_energy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "cbs_extrapolation",
            "label": self.label,
            "formula": self.formula,
            "components": self.components,
            "levels": [lvl.as_dict() for lvl in self.levels],
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "CBSExtrapolationTerm":
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
        base: Term,
        corrections: list[Term] | None = None,
        preset_name: str | None = None,
        reference: str | None = None,
    ):
        if not isinstance(base, Term) or not base.provides_absolute_energy:
            raise InputError(
                "CompositeProtocol.base must be a Term providing an absolute "
                "electronic energy (SinglePointTerm or CBSExtrapolationTerm); "
                f"got {type(base).__name__}."
            )
        corrections = list(corrections) if corrections else []
        for term in corrections:
            if isinstance(term, CBSExtrapolationTerm):
                raise InputError(
                    f"CBS extrapolation term '{term.label}' cannot be a correction: "
                    f"with components='total' it extrapolates absolute energies and "
                    f"would double-count the base. Use the CBS term as the protocol's "
                    f"'base' (CBS-as-base, the canonical FPA shape), or pick one of "
                    f"the HEAT / W-n presets."
                )
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
        sub_labels: list[str] = []
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
    def terms(self) -> list[Term]:
        """Convenience: ``[base, *corrections]`` in protocol order."""
        return [self.base, *self.corrections]

    @property
    def base_sub_labels(self) -> list[str]:
        """All sub_labels of the base term, in ascending-cardinal order."""
        return [sub_label for sub_label, _level in self.base.required_levels()]

    @property
    def primary_base_sub_label(self) -> str:
        """The base's primary sub_label (largest-cardinal leg for a CBS base)."""
        return self.base.primary_sub_label

    @property
    def primary_base_level(self) -> Level:
        """The base's primary Level (largest-cardinal leg for a CBS base)."""
        return self.base.primary_level

    def evaluate(self, energies: dict[str, float]) -> float:
        """Combine all sub-job energies into the protocol's electronic energy."""
        return sum(term.evaluate(energies) for term in self.terms)

    def iter_required_jobs(self) -> Iterable[tuple[str, str, Level]]:
        """Yield ``(term_label, sub_label, Level)`` triples for every required SP."""
        for term in self.terms:
            for sub_label, level in term.required_levels():
                yield (term.label, sub_label, level)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "base": self.base.as_dict(),
            "corrections": [t.as_dict() for t in self.corrections],
        }
        if self.preset_name is not None:
            out["preset_name"] = self.preset_name
        if self.reference is not None:
            out["reference"] = self.reference
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositeProtocol":
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
            base = Term.from_dict(base_data)
            if not base.provides_absolute_energy:
                raise InputError(
                    "CompositeProtocol.base must provide an absolute electronic "
                    "energy (single_point or cbs_extrapolation); got "
                    f"'{base_data.get('type')}'."
                )
        corrections = [Term.from_dict(d) for d in data.get("corrections", [])]
        return cls(
            base=base,
            corrections=corrections,
            preset_name=data.get("preset_name"),
            reference=data.get("reference"),
        )

    @classmethod
    def from_user_input(cls, raw: str | dict[str, Any]) -> "CompositeProtocol":
        """Accept the YAML-shaped user input and produce a validated protocol.

        Three forms are accepted:

        * A *string* — interpreted as a preset name. Delegates to
          :func:`arc.level.presets.expand_preset`.
        * A *dict with* ``preset:`` *key* — preset with overrides. Delegates to
          :func:`arc.level.presets.expand_preset`.
        * A *dict with* ``base:`` *and* ``corrections:`` keys — fully explicit
          recipe. ``base`` may be a Level shorthand (``"method/basis"``), a Level
          dict, or a serialised term dict (``type: single_point`` or
          ``type: cbs_extrapolation``). Each entry in ``corrections`` must
          include a ``type`` discriminator.

        Args:
            raw (str | dict): The user input, in one of the three forms above.

        Returns:
            CompositeProtocol: The validated protocol.

        Raises:
            InputError: On any malformed input.
        """
        preset_name: str | None = None
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

        # base may be: string ("method/basis"), Level dict, or an explicit
        # term dict with a 'type' discriminator ('single_point' or
        # 'cbs_extrapolation'). A typed base dict defaults to label 'base' so
        # CBS-base sub_labels ('base__card_<X>') stay deterministic for restart.
        base_raw = raw["base"]
        if isinstance(base_raw, str):
            base = SinglePointTerm(label="base", level=base_raw)
        elif isinstance(base_raw, dict) and "type" in base_raw:
            base_data = dict(base_raw)
            base_data.setdefault("label", "base")
            base = Term.from_dict(base_data)
            if not base.provides_absolute_energy:
                raise InputError(
                    "sp_composite 'base' must provide an absolute electronic "
                    "energy (single_point or cbs_extrapolation); got "
                    f"'{base_raw['type']}'."
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

        corrections: list[Term] = []
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
