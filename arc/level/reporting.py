"""
``arc.level.reporting`` — provenance artifacts for ``sp_composite`` runs.

Two reporting layers are exposed:

* :func:`format_log_event` — formats a single structured ``[sp_composite]`` log
  line. The Phase 2 scheduler integration calls this at every state transition
  (queue, sub-job complete, term evaluated, protocol finalized) so the ARC log
  alone tells the full story end-to-end.

* :func:`write_composite_notebook` — emits a single, project-level Jupyter
  notebook at ``<project>/output/sp_composite.ipynb`` with one H2 section per
  species or transition state whose ``sp_composite`` has finalized. The
  notebook is **unexecuted on write**: ARC lays down cell sources but does NOT
  populate outputs. The user opens it and "Run All"; every energy shown is
  then produced by the user's own machine invoking ARC's real parsers on the
  real QM output files — genuine independent verification rather than a
  re-display of numbers the scheduler already computed.

The notebook is self-contained per section: each section carries its own
literal recipe dict and reconstructs its :class:`CompositeProtocol` via
:meth:`CompositeProtocol.from_user_input`, so a user can move or share a run
directory without losing the provenance context.

Dependencies are limited to ``nbformat`` (already pulled in by ARC's
``environment.yml`` via ``conda-forge::jupyter``). No pandas, no executed
notebooks at write time.

References
----------

* Allen, East, Császár — focal-point analysis review (cited in per-preset
  markdown content produced by this module when a preset supplies it).
* Tajti, Szalay, Császár, Kállay, Gauss, Valeev, Flowers, Vázquez, Stanton,
  *J. Chem. Phys.* **121**, 11599 (2004). DOI: 10.1063/1.1811608 — HEAT.
"""

import hashlib
import os
import pprint
from dataclasses import dataclass, field
from typing import Any

import nbformat
import yaml
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from arc.constants import E_h_kJmol
from arc.exceptions import InputError
from arc.level.protocol import CompositeProtocol
from arc.parser.parser import parse_e_elect


# =========================================================================== #
#  format_log_event                                                           #
# =========================================================================== #


def format_log_event(species_label: str, event: str, payload: Any) -> str:
    """Format a single ``[sp_composite]`` log line.

    Examples
    --------
    >>> format_log_event("H2O", "queued", "delta_T")
    '[sp_composite] H2O — queued: delta_T'
    >>> format_log_event("H2O", "complete", None)
    '[sp_composite] H2O — complete'
    """
    prefix = f"[sp_composite] {species_label} — {event}"
    if payload is None:
        return prefix
    if isinstance(payload, dict):
        body = ", ".join(f"{k}={v}" for k, v in payload.items())
    else:
        body = str(payload)
    return f"{prefix}: {body}"


# =========================================================================== #
#  SpeciesSection — scheduler → reporter handoff                              #
# =========================================================================== #


_VALID_KINDS = ("species", "ts")


@dataclass
class SpeciesSection:
    """One stationary point's contribution to the provenance notebook.

    Attributes
    ----------
    label : str
        Species or TS label (matches the scheduler's ``label`` key).
    kind : {'species', 'ts'}
        Whether this stationary point is a well (``'species'``) or a transition
        state (``'ts'``). Controls section ordering in the notebook (species
        first, TS second).
    preset_name : str or None
        Name of the preset used (e.g. ``'HEAT-345Q'``), or ``None`` if the
        user supplied an explicit recipe.
    reference : str
        Citation string, ideally including a DOI. Deduplicated across sections
        in the notebook's References block.
    recipe : dict
        The literal explicit recipe dict (``{"base": ..., "corrections": [...]}``)
        used to construct ``protocol``. Written into the notebook verbatim so
        each section is reproducible in isolation.
    protocol : CompositeProtocol
        The composite protocol this section reports on. Used only to
        enumerate sub-job sub-labels and term types at write time; the
        notebook reconstructs its own protocol from ``recipe`` via
        :meth:`CompositeProtocol.from_user_input` when executed.
    sub_job_paths : dict[str, str]
        Mapping ``sub_label`` → absolute path to the QM output file. Rendered
        into the notebook as paths relative to the notebook directory when
        possible, absolute when the path escapes the notebook's tree.
    flags : list[str]
        Human-readable warnings surfaced by the scheduler (e.g. "δT exceeds
        10 kJ/mol, potential single-reference breakdown"). Rendered verbatim
        in the section's interpretation markdown cell.
    """

    label: str
    kind: str
    preset_name: str | None
    reference: str
    recipe: dict[str, Any]
    protocol: CompositeProtocol
    sub_job_paths: dict[str, str]
    flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise InputError(
                f"SpeciesSection.kind must be one of {_VALID_KINDS}; got {self.kind!r}."
            )
        if not self.label:
            raise InputError("SpeciesSection.label must be non-empty.")


# =========================================================================== #
#  write_composite_notebook                                                   #
# =========================================================================== #


# Pinned nbformat versioning so re-writes produce byte-identical files.
_NBFORMAT = 4
_NBFORMAT_MINOR = 5

# Pinned kernelspec — matches the standard python3 kernel that ships with
# ipykernel. Notebooks open with this kernel by default in Jupyter/VS Code/etc.
_KERNELSPEC = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}

_LANGUAGE_INFO = {
    "name": "python",
    "mimetype": "text/x-python",
    "file_extension": ".py",
    "pygments_lexer": "ipython3",
}


def write_composite_notebook(
    path: str,
    project_name: str,
    arc_version: str,
    timestamp: str,
    sections: list[SpeciesSection],
    notebook_dir: str,
) -> None:
    """Write (or overwrite) the project-level composite-provenance notebook.

    Parameters
    ----------
    path : str
        Destination file path, typically ``<project>/output/sp_composite.ipynb``.
        The parent directory must exist.
    project_name : str
        Project name, surfaced in the title banner.
    arc_version : str
        ARC version string, surfaced in the title banner.
    timestamp : str
        ISO-8601 generation timestamp. Accepted as a parameter (rather than
        read from the clock) so reruns produce byte-identical output — useful
        for snapshot testing and for idempotent regeneration across a run.
    sections : list[SpeciesSection]
        One section per species/TS that has finalized its composite. The
        writer sorts species-first / TS-second with alphabetical ordering
        within each group, independent of caller order.
    notebook_dir : str
        The directory that will host the notebook. Used to render absolute
        ``sub_job_paths`` as relative paths when they fall under this
        directory, so the notebook + outputs directory can be copied together.

    Raises
    ------
    arc.exceptions.InputError
        If any ``SpeciesSection.kind`` is not in ``{'species', 'ts'}``.
        (The dataclass also validates, but we re-check defensively.)
    """
    for s in sections:
        if s.kind not in _VALID_KINDS:
            raise InputError(
                f"SpeciesSection.kind must be one of {_VALID_KINDS}; got {s.kind!r} "
                f"on label {s.label!r}."
            )

    ordered = _sort_sections(sections)

    cells: list[Any] = []

    # --- Top-level shared cells ------------------------------------------- #
    cells.append(_title_banner_cell(
        project_name=project_name,
        arc_version=arc_version,
        timestamp=timestamp,
        sections=ordered,
    ))
    cells.append(_toc_cell(ordered))
    cells.append(_setup_cell())

    # --- Per-section cells ------------------------------------------------ #
    for section in ordered:
        cells.extend(_section_cells(section, notebook_dir))

    # --- Project summary + references ------------------------------------- #
    cells.append(_project_summary_header_cell())
    cells.append(_project_summary_code_cell(ordered))
    cells.append(_references_cell(ordered))

    nb = new_notebook(cells=cells, metadata={
        "kernelspec": dict(_KERNELSPEC),
        "language_info": dict(_LANGUAGE_INFO),
    })
    nb["nbformat"] = _NBFORMAT
    nb["nbformat_minor"] = _NBFORMAT_MINOR

    with open(path, "w", encoding="utf-8") as fh:
        nbformat.write(nb, fh, version=_NBFORMAT)


# --------------------------------------------------------------------------- #
#  Section ordering                                                           #
# --------------------------------------------------------------------------- #


def _sort_sections(sections: list[SpeciesSection]) -> list[SpeciesSection]:
    """Species first (alphabetical), then TS (alphabetical)."""
    species = sorted((s for s in sections if s.kind == "species"), key=lambda s: s.label)
    ts = sorted((s for s in sections if s.kind == "ts"), key=lambda s: s.label)
    return species + ts


# --------------------------------------------------------------------------- #
#  Cell IDs (stable across re-writes given the same inputs)                   #
# --------------------------------------------------------------------------- #


def _cell_id(section_key: str, role: str) -> str:
    """Return a stable 16-char hex cell ID for the given section + role."""
    digest = hashlib.sha1(f"{section_key}|{role}".encode("utf-8")).hexdigest()
    return digest[:16]


def _md(source: str, cell_id: str):
    cell = new_markdown_cell(source=source)
    cell["id"] = cell_id
    return cell


def _code(source: str, cell_id: str):
    cell = new_code_cell(source=source)
    cell["id"] = cell_id
    # Guarantee unexecuted: nbformat.v4 already sets outputs=[] and
    # execution_count=None for new_code_cell, but enforce explicitly.
    cell["outputs"] = []
    cell["execution_count"] = None
    return cell


# --------------------------------------------------------------------------- #
#  Top-level shared cells                                                     #
# --------------------------------------------------------------------------- #


def _title_banner_cell(project_name: str, arc_version: str, timestamp: str,
                       sections: list[SpeciesSection]):
    n_species = sum(1 for s in sections if s.kind == "species")
    n_ts = sum(1 for s in sections if s.kind == "ts")
    source = (
        f"# `sp_composite` provenance — {project_name}\n\n"
        f"**ARC version:** `{arc_version}`   \n"
        f"**Generated:** `{timestamp}`   \n"
        f"**Sections:** {n_species} species, {n_ts} transition state"
        f"{'s' if n_ts != 1 else ''}   \n\n"
        "This notebook independently verifies the composite electronic energies that "
        "ARC computed for each stationary point. Each section defines its own recipe, "
        "re-parses the QM output files via `arc.parser.parse_e_elect`, and re-evaluates "
        "the `CompositeProtocol` on the spot — so the numbers you see are produced by "
        "your machine running ARC's real parsers, not transcribed from the scheduler's "
        "memory.\n\n"
        "**To verify:** open this notebook in Jupyter or VS Code and run **Run All**. "
        "The `FINAL e_elect(...)` printed in each section is the value to compare "
        "against ARC's `output.yml`.\n"
    )
    return _md(source, _cell_id("shared", "title"))


def _toc_cell(sections: list[SpeciesSection]):
    lines = ["### Table of contents\n"]
    for s in sections:
        kind_label = "Species" if s.kind == "species" else "TS"
        lines.append(f"- [{kind_label}: {s.label}](#{kind_label}:-{s.label})")
    lines.append("- [Project summary](#Project-summary)")
    lines.append("- [References](#References)")
    return _md("\n".join(lines), _cell_id("shared", "toc"))


_SETUP_SOURCE = '''\
# Shared setup — imports + helpers used by every species/TS section below.
import arc.parser.parser as arc_parser
from arc.constants import E_h_kJmol
from arc.level.protocol import CompositeProtocol

# Accumulates per-section results so the Project summary can aggregate them.
_RESULTS = {}


def _format_breakdown(protocol, energies_kJmol):
    """Render a fixed-width per-term breakdown table as a string."""
    hdr = f"{'term':<20} {'type':<22} {'contribution (kJ/mol)':>24}"
    rule = "-" * len(hdr)
    lines = [hdr, rule]
    for term in protocol.terms:
        contribution = term.evaluate(energies_kJmol)
        lines.append(
            f"{term.label:<20} {type(term).__name__:<22} {contribution:>24.6f}"
        )
    lines.append(rule)
    return "\\n".join(lines)
'''


def _setup_cell():
    return _code(_SETUP_SOURCE, _cell_id("shared", "setup"))


# --------------------------------------------------------------------------- #
#  Per-section cells                                                          #
# --------------------------------------------------------------------------- #


def _section_cells(section: SpeciesSection, notebook_dir: str) -> list[Any]:
    key = f"{section.kind}:{section.label}"
    kind_label = "Species" if section.kind == "species" else "TS"
    return [
        _md(f"## {kind_label}: {section.label}\n", _cell_id(key, "header")),
        _md(_protocol_summary_markdown(section), _cell_id(key, "summary")),
        _code(_recipe_code(section), _cell_id(key, "recipe")),
        _code(_paths_code(section, notebook_dir), _cell_id(key, "paths")),
        _code(_parse_code(section), _cell_id(key, "parse")),
        _code(_breakdown_code(section), _cell_id(key, "breakdown")),
        _code(_final_code(section), _cell_id(key, "final")),
        _md(_interpretation_markdown(section), _cell_id(key, "interpretation")),
    ]


def _protocol_summary_markdown(section: SpeciesSection) -> str:
    protocol_name = section.preset_name or "explicit recipe"
    n_sub_jobs = sum(1 for _ in section.protocol.iter_required_jobs())
    n_corrections = len(section.protocol.corrections)
    formula_latex = _composite_formula_latex(section.protocol)
    return (
        f"**Protocol:** `{protocol_name}`   \n"
        f"**Reference:** {section.reference}   \n"
        f"**Sub-jobs:** {n_sub_jobs} across 1 base + {n_corrections} correction(s)   \n\n"
        f"**Composite formula:**\n\n"
        f"$$ {formula_latex} $$\n"
    )


def _composite_formula_latex(protocol: CompositeProtocol) -> str:
    parts = [r"E_{\mathrm{" + _latex_escape(protocol.base.label) + "}}"]
    for term in protocol.corrections:
        parts.append(r"\delta_{\mathrm{" + _latex_escape(term.label) + "}}")
    return r"E_{\mathrm{final}} = " + " + ".join(parts)


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _recipe_code(section: SpeciesSection) -> str:
    # pprint with sort_dicts=False preserves author intent; width 78 stays < 80 cols.
    recipe_repr = pprint.pformat(section.recipe, sort_dicts=False, width=78)
    return (
        f"recipe = {recipe_repr}\n"
        "protocol = CompositeProtocol.from_user_input(recipe)\n"
        f"protocol  # {section.label}"
    )


def _paths_code(section: SpeciesSection, notebook_dir: str) -> str:
    entries = []
    for sub_label, abs_path in sorted(section.sub_job_paths.items()):
        rendered = _render_path(abs_path, notebook_dir)
        entries.append(f"    {sub_label!r}: {rendered!r},")
    return "paths = {\n" + "\n".join(entries) + "\n}"


def _render_path(abs_path: str, notebook_dir: str) -> str:
    """Render ``abs_path`` relative to ``notebook_dir`` if under it, else absolute."""
    try:
        common = os.path.commonpath([os.path.abspath(abs_path),
                                     os.path.abspath(notebook_dir)])
    except ValueError:
        return abs_path
    if common == os.path.abspath(notebook_dir):
        rel = os.path.relpath(abs_path, notebook_dir)
        return rel if rel.startswith((".", os.sep)) else f"./{rel}"
    return abs_path


def _parse_code(section: SpeciesSection) -> str:
    return (
        "# Parse the electronic energy from each sub-job's QM output file.\n"
        "# arc_parser.parse_e_elect dispatches on ESS and returns kJ/mol.\n"
        "# Also verifies that `paths` covers every sub_label the protocol\n"
        "# requires — a missing entry here would fail protocol.evaluate later\n"
        "# with a less-helpful KeyError.\n"
        "_required_sub_labels = {sl for _t, sl, _l in protocol.iter_required_jobs()}\n"
        "_missing_paths = sorted(_required_sub_labels - set(paths.keys()))\n"
        "assert not _missing_paths, "
        "f'paths is missing required sub_labels: {_missing_paths}'\n"
        "energies_kJmol = {\n"
        "    sub_label: arc_parser.parse_e_elect(p)\n"
        "    for sub_label, p in paths.items()\n"
        "}\n"
        "missing = [sl for sl, v in energies_kJmol.items() if v is None]\n"
        "assert not missing, "
        "f'parse_e_elect failed for: {missing}'\n"
        "energies_kJmol"
    )


def _breakdown_code(section: SpeciesSection) -> str:
    return (
        "# Per-term breakdown: what each term contributes to the composite total.\n"
        "print(_format_breakdown(protocol, energies_kJmol))"
    )


def _final_code(section: SpeciesSection) -> str:
    # ``section.label`` flows into the generated Python source three times
    # (the _RESULTS key, the kind string, and the FINAL print). Every use
    # is rendered via ``!r`` so labels containing quotes, parens, braces,
    # or backslashes cannot break the cell's syntax.
    label_repr = repr(section.label)
    return (
        "e_total_kJmol = protocol.evaluate(energies_kJmol)\n"
        f"_RESULTS[{label_repr}] = {{\n"
        f"    'kind': {section.kind!r},\n"
        f"    'protocol_name': {section.preset_name!r},\n"
        "    'e_total_kJmol': e_total_kJmol,\n"
        "    'n_sub_jobs': len(paths),\n"
        "}\n"
        f"_label_display = {label_repr}\n"
        "print(f'FINAL e_elect({_label_display}) = "
        "{e_total_kJmol:,.3f} kJ/mol "
        "({e_total_kJmol / E_h_kJmol:.9f} Hartree)')"
    )


def _interpretation_markdown(section: SpeciesSection) -> str:
    if not section.flags:
        return "_No warnings flagged for this section._\n"
    lines = ["**Warnings from the scheduler:**\n"]
    for flag in section.flags:
        lines.append(f"- {flag}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
#  Tail cells (project summary + references)                                  #
# --------------------------------------------------------------------------- #


def _project_summary_header_cell():
    return _md(
        "## Project summary\n\n"
        "Aggregate of the composite results across every species / TS in the project. "
        "This cell reads from the `_RESULTS` dict populated by each section above, so "
        "it must be run after the per-section cells.\n",
        _cell_id("shared", "summary_header"),
    )


_SUMMARY_CODE = '''\
# Tabular summary of every section's final e_elect.
if not _RESULTS:
    print("(no composite results to summarise — did you run the per-section cells?)")
else:
    hdr = f"{'label':<20} {'kind':<10} {'protocol':<18} {'e_elect (kJ/mol)':>22}"
    rule = "-" * len(hdr)
    print(hdr)
    print(rule)
    for label, r in _RESULTS.items():
        proto = r['protocol_name'] or 'explicit'
        print(
            f"{label:<20} {r['kind']:<10} {proto:<18} "
            f"{r['e_total_kJmol']:>22,.3f}"
        )
'''


def _project_summary_code_cell(sections: list[SpeciesSection]):
    return _code(_SUMMARY_CODE, _cell_id("shared", "summary_code"))


def _references_cell(sections: list[SpeciesSection]):
    # Deduplicate references while preserving order of first appearance.
    seen = set()
    ordered_refs: list[str] = []
    for s in sections:
        if s.reference and s.reference not in seen:
            seen.add(s.reference)
            ordered_refs.append(s.reference)
    lines = ["## References\n"]
    if not ordered_refs:
        lines.append("_(No references supplied by any section.)_\n")
    else:
        for ref in ordered_refs:
            lines.append(f"- {ref}")
    return _md("\n".join(lines) + "\n", _cell_id("shared", "references"))


# =========================================================================== #
#  build_species_report_dict + write_species_report_yaml                      #
# =========================================================================== #
#
# Per-species YAML report. Companion to the project-level provenance notebook
# (which serves *independent verification* via Run-All); this writer produces
# a consumable per-species summary that's readable in plain text and easy to
# parse from downstream tooling. One file per stationary point, written by
# the scheduler at composite finalization.


def _composite_formula_text(protocol: CompositeProtocol) -> str:
    """Plain-text version of the composite formula (no LaTeX).

    The notebook uses LaTeX rendering; the YAML report is plain text — readers
    of ``cat sp_composite_report.yml`` shouldn't see ``\\delta`` macros.
    """
    parts = [f"E_{protocol.base.label}"]
    for term in protocol.corrections:
        parts.append(term.label)
    return "E_final = " + " + ".join(parts)


def build_species_report_dict(
    section: SpeciesSection,
    e_elect_kj_per_mol: float,
    timestamp: str,
    arc_version: str,
    arc_commit: str,
) -> dict[str, Any]:
    """Assemble the per-species sp_composite report as a plain dict.

    All energy values are computed by re-parsing the QM output files referenced
    in ``section.sub_job_paths`` via :func:`arc.parser.parser.parse_e_elect`
    (returns kJ/mol), then handed to each :class:`Term` to compute its
    contribution. The same evaluation path the notebook's "Run All" follows —
    the values land identically because the protocol is deterministic.

    The caller-supplied ``e_elect_kj_per_mol`` is the value the scheduler
    recorded on the ``ARCSpecies`` (set during ``_finalize_composite``). We
    don't recompute the total here; we surface what ARC actually used so any
    downstream tooling reading this report is consistent with the run's
    output.yml / restart.yml.

    Parameters
    ----------
    section : SpeciesSection
        The reporting handoff struct populated by the scheduler at
        finalization. Carries protocol, recipe, sub-job paths, flags.
    e_elect_kj_per_mol : float
        The final electronic energy ARC recorded for this species (kJ/mol).
    timestamp : str
        ISO-8601 string. Caller supplies for determinism (tests pin it).
    arc_version, arc_commit : str
        Provenance identifiers.

    Returns
    -------
    dict
        A plain dict ready for ``yaml.safe_dump`` or
        :func:`write_species_report_yaml`.
    """
    energies_kj = {sl: parse_e_elect(p) for sl, p in section.sub_job_paths.items()}

    base_term = section.protocol.base
    base_sub_label, base_level = base_term.required_levels()[0]
    base_block = {
        "sub_label": base_sub_label,
        "level": base_level.simple(),
        "energy_kj_per_mol": energies_kj[base_sub_label],
        "energy_hartree": energies_kj[base_sub_label] / E_h_kJmol,
        "path": section.sub_job_paths[base_sub_label],
    }

    terms_block: list[dict[str, Any]] = []
    for term in section.protocol.corrections:
        contribution_kj = term.evaluate(energies_kj)
        sub_jobs: list[dict[str, Any]] = []
        for sub_label, level in term.required_levels():
            sub_jobs.append({
                "sub_label": sub_label,
                "level": level.simple(),
                "energy_kj_per_mol": energies_kj[sub_label],
                "energy_hartree": energies_kj[sub_label] / E_h_kJmol,
                "path": section.sub_job_paths[sub_label],
            })
        terms_block.append({
            "label": term.label,
            "type": type(term).__name__,
            "contribution_kj_per_mol": contribution_kj,
            "contribution_hartree": contribution_kj / E_h_kJmol,
            "sub_jobs": sub_jobs,
        })

    return {
        "species": section.label,
        "kind": section.kind,
        "generated_at": timestamp,
        "arc_version": arc_version,
        "arc_commit": arc_commit,
        "protocol": {
            "preset": section.preset_name,
            "reference": section.reference,
            "formula": _composite_formula_text(section.protocol),
        },
        "units": {
            "energy": "kJ/mol",
            "energy_alt": "Hartree",
        },
        "base": base_block,
        "terms": terms_block,
        "final": {
            "e_elect_kj_per_mol": e_elect_kj_per_mol,
            "e_elect_hartree": e_elect_kj_per_mol / E_h_kJmol,
            "e_elect_source": "sp_composite",
        },
        "flags": list(section.flags),
    }


def write_species_report_yaml(
    path: str,
    section: SpeciesSection,
    e_elect_kj_per_mol: float,
    timestamp: str,
    arc_version: str,
    arc_commit: str,
) -> None:
    """Build and write the per-species sp_composite YAML report.

    Creates the parent directory if missing. Output is deterministic: keys are
    written in insertion order (Python 3.7+ dicts), flow style is block (the
    YAML default), and ``sort_keys=False`` preserves the schema's natural
    reading order (species → protocol → units → base → terms → final → flags).
    Two writes with the same inputs produce byte-identical files.
    """
    report = build_species_report_dict(
        section=section,
        e_elect_kj_per_mol=e_elect_kj_per_mol,
        timestamp=timestamp,
        arc_version=arc_version,
        arc_commit=arc_commit,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(report, fh, sort_keys=False, default_flow_style=False)
