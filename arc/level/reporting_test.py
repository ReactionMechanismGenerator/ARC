#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for ``arc.level.reporting`` — project-level Jupyter notebook artifact
+ the structured ``[sp_composite]`` log-event helper.

The notebook-emitter must deliver:

* A single ``.ipynb`` per project with one clearly-formatted section per species/TS.
* Deterministic output (byte-identical re-writes with the same timestamp).
* Self-contained per-section code: every section writes its own recipe dict and
  reconstructs its own :class:`CompositeProtocol`, then re-parses the sub-job
  output files via :func:`arc.parser.parse_e_elect` — giving the user
  *independent* verification of ARC's computed energy.
* End-to-end executability: the test suite writes a notebook against real
  fixture output files (in Gaussian format that ARC's real parser consumes),
  runs it via :mod:`nbclient`, and asserts the final ``e_elect`` printed in the
  final code cell matches the value computed outside the notebook.
"""

import os
import tempfile
import unittest

import nbclient
import nbformat

from arc.constants import E_h_kJmol
from arc.exceptions import InputError
from arc.level.protocol import CompositeProtocol
from arc.level.reporting import (
    SpeciesSection,
    build_species_report_dict,
    format_log_event,
    write_composite_notebook,
    write_species_report_yaml,
)


def _write_gaussian_fixture(path: str, e_hartree: float) -> None:
    """Minimal Gaussian-format output file parseable by ``arc.parser.parse_e_elect``.

    ``arc.parser.determine_ess`` detects 'gaussian' from a line containing that
    substring, and ``extract_scf_done`` regexes out the energy from a
    ``SCF Done: E(RHF) = <value> A.U.`` line.
    """
    with open(path, "w") as fh:
        fh.write(" Gaussian 16: test fixture\n")
        fh.write(f" SCF Done:  E(RHF) =  {e_hartree:.9f}     A.U. after    1 cycles\n")


# --------------------------------------------------------------------------- #
#  format_log_event                                                           #
# --------------------------------------------------------------------------- #


class TestFormatLogEvent(unittest.TestCase):
    def test_format_basic(self):
        line = format_log_event("H2O", "queued", "delta_T")
        self.assertEqual(line, "[sp_composite] H2O — queued: delta_T")

    def test_format_with_dict_payload(self):
        line = format_log_event("H2O", "energy", {"sub_label": "base", "value_kJmol": -200000.0})
        self.assertIn("[sp_composite] H2O — energy:", line)
        self.assertIn("base", line)
        self.assertIn("-200000", line)

    def test_format_with_None_payload(self):
        line = format_log_event("H2O", "complete", None)
        self.assertEqual(line, "[sp_composite] H2O — complete")


# --------------------------------------------------------------------------- #
#  SpeciesSection dataclass                                                   #
# --------------------------------------------------------------------------- #


def _make_two_term_section(label: str = "H2O", kind: str = "species",
                           paths: dict = None, flags: list = None) -> SpeciesSection:
    recipe = {
        "base": {"method": "hf", "basis": "cc-pVTZ"},
        "corrections": [
            {"label": "delta_T", "type": "delta",
             "high": {"method": "ccsdt", "basis": "cc-pVDZ"},
             "low": {"method": "ccsd(t)", "basis": "cc-pVDZ"}},
        ],
    }
    return SpeciesSection(
        label=label,
        kind=kind,
        preset_name=None,
        reference="DOI: 10.0/test",
        recipe=recipe,
        protocol=CompositeProtocol.from_user_input(recipe),
        sub_job_paths=paths or {"base": "/tmp/base.out",
                                "delta_T__high": "/tmp/hi.out",
                                "delta_T__low": "/tmp/lo.out"},
        flags=flags or [],
    )


class TestSpeciesSection(unittest.TestCase):
    def test_fields_accessible(self):
        s = _make_two_term_section()
        self.assertEqual(s.label, "H2O")
        self.assertEqual(s.kind, "species")
        self.assertIsNone(s.preset_name)
        self.assertIn("DOI", s.reference)
        self.assertIn("base", s.recipe)
        self.assertIsInstance(s.protocol, CompositeProtocol)
        self.assertIn("base", s.sub_job_paths)
        self.assertEqual(s.flags, [])

    def test_kind_species_valid(self):
        _make_two_term_section(kind="species")  # must not raise

    def test_kind_ts_valid(self):
        _make_two_term_section(kind="ts")  # must not raise

    def test_kind_unknown_rejected(self):
        with self.assertRaises(InputError):
            _make_two_term_section(kind="bogus")


# --------------------------------------------------------------------------- #
#  write_composite_notebook                                                   #
# --------------------------------------------------------------------------- #


class TestWriteCompositeNotebook(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.nb_path = os.path.join(self.tmp, "sp_composite.ipynb")
        self.base_path = os.path.join(self.tmp, "base.out")
        self.hi_path = os.path.join(self.tmp, "delta_T__high.out")
        self.lo_path = os.path.join(self.tmp, "delta_T__low.out")
        _write_gaussian_fixture(self.base_path, -76.345678)
        _write_gaussian_fixture(self.hi_path, -76.345600)
        _write_gaussian_fixture(self.lo_path, -76.346500)
        self.section = _make_two_term_section(
            paths={"base": self.base_path,
                   "delta_T__high": self.hi_path,
                   "delta_T__low": self.lo_path},
        )
        self.kwargs = dict(
            path=self.nb_path,
            project_name="unittest_project",
            arc_version="test",
            timestamp="2026-04-22T14:01:33Z",
            sections=[self.section],
            notebook_dir=self.tmp,
        )

    def tearDown(self):
        for p in (self.nb_path, self.base_path, self.hi_path, self.lo_path):
            if os.path.exists(p):
                os.unlink(p)
        os.rmdir(self.tmp)

    def _read(self):
        return nbformat.read(self.nb_path, as_version=4)

    def _cells(self):
        return self._read().cells

    # --- existence + validity ---------------------------------------------- #

    def test_file_written(self):
        write_composite_notebook(**self.kwargs)
        self.assertTrue(os.path.exists(self.nb_path))

    def test_valid_nbformat_validates(self):
        write_composite_notebook(**self.kwargs)
        nb = self._read()
        nbformat.validate(nb)  # raises if invalid

    def test_code_cells_unexecuted(self):
        write_composite_notebook(**self.kwargs)
        for cell in self._cells():
            if cell.cell_type == "code":
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)

    def test_notebook_has_stable_metadata(self):
        write_composite_notebook(**self.kwargs)
        nb = self._read()
        self.assertIn("kernelspec", nb.metadata)
        self.assertEqual(nb.metadata.kernelspec.get("name"), "python3")

    # --- section structure -------------------------------------------------- #

    def test_one_h2_per_section_single(self):
        write_composite_notebook(**self.kwargs)
        h2s = [c for c in self._cells()
               if c.cell_type == "markdown" and c.source.lstrip().startswith("## ")]
        # One H2 for the species + one for the project summary tail.
        species_h2s = [c for c in h2s if "H2O" in c.source]
        self.assertEqual(len(species_h2s), 1)

    def test_one_h2_per_section_multiple(self):
        s1 = _make_two_term_section(label="H2O", paths=self.section.sub_job_paths)
        s2 = _make_two_term_section(label="OH", paths=self.section.sub_job_paths)
        s3 = _make_two_term_section(label="TS1", kind="ts", paths=self.section.sub_job_paths)
        kwargs = {**self.kwargs, "sections": [s2, s1, s3]}
        write_composite_notebook(**kwargs)
        cells = self._cells()
        h2_titles = [c.source.splitlines()[0]
                     for c in cells
                     if c.cell_type == "markdown" and c.source.lstrip().startswith("## ")
                     and "Project summary" not in c.source
                     and "References" not in c.source]
        # Species first (alphabetical), then TS (alphabetical): H2O, OH, TS1.
        self.assertEqual(len(h2_titles), 3)
        self.assertIn("H2O", h2_titles[0])
        self.assertIn("OH", h2_titles[1])
        self.assertIn("TS1", h2_titles[2])

    def test_ts_sections_come_after_species(self):
        ts = _make_two_term_section(label="AATS", kind="ts", paths=self.section.sub_job_paths)
        sp = _make_two_term_section(label="ZZspecies", paths=self.section.sub_job_paths)
        kwargs = {**self.kwargs, "sections": [ts, sp]}
        write_composite_notebook(**kwargs)
        cells = self._cells()
        # Find the section-header (H2) cell for each.
        zz_idx = next(i for i, c in enumerate(cells)
                      if c.cell_type == "markdown" and c.source.lstrip().startswith("## Species: ZZspecies"))
        aa_idx = next(i for i, c in enumerate(cells)
                      if c.cell_type == "markdown" and c.source.lstrip().startswith("## TS: AATS"))
        self.assertLess(zz_idx, aa_idx)

    def test_shared_setup_cell_only_once(self):
        s2 = _make_two_term_section(label="OH", paths=self.section.sub_job_paths)
        kwargs = {**self.kwargs, "sections": [self.section, s2]}
        write_composite_notebook(**kwargs)
        cells = self._cells()
        setup_cells = [c for c in cells
                       if c.cell_type == "code"
                       and "import arc.parser" in c.source
                       and "CompositeProtocol" in c.source]
        self.assertEqual(len(setup_cells), 1)

    def test_toc_present(self):
        s2 = _make_two_term_section(label="OH", paths=self.section.sub_job_paths)
        kwargs = {**self.kwargs, "sections": [self.section, s2]}
        write_composite_notebook(**kwargs)
        toc_cells = [c for c in self._cells()
                     if c.cell_type == "markdown"
                     and ("table of contents" in c.source.lower() or "contents" in c.source.lower())]
        self.assertGreaterEqual(len(toc_cells), 1)
        # TOC should mention every section label.
        toc_joined = "\n".join(c.source for c in toc_cells)
        self.assertIn("H2O", toc_joined)
        self.assertIn("OH", toc_joined)

    def test_references_section_present(self):
        write_composite_notebook(**self.kwargs)
        ref_cells = [c for c in self._cells()
                     if c.cell_type == "markdown" and "References" in c.source]
        self.assertGreaterEqual(len(ref_cells), 1)

    def test_references_deduplicated(self):
        # Two sections citing the same DOI → one entry in the References block.
        s2 = _make_two_term_section(label="OH", paths=self.section.sub_job_paths)  # same DOI
        kwargs = {**self.kwargs, "sections": [self.section, s2]}
        write_composite_notebook(**kwargs)
        ref_cell = next(c for c in self._cells()
                        if c.cell_type == "markdown" and c.source.lstrip().startswith("## References"))
        self.assertEqual(ref_cell.source.count("DOI: 10.0/test"), 1)

    def test_section_contains_required_cell_sequence(self):
        write_composite_notebook(**self.kwargs)
        cells = self._cells()
        # Locate the H2O section header.
        hdr = next(i for i, c in enumerate(cells)
                   if c.cell_type == "markdown" and "## Species: H2O" in c.source)
        # The 7 cells immediately after the header are: protocol summary (md),
        # recipe + build (code), paths map (code), parse + evaluate (code),
        # breakdown (code), final print (code), interpretation (md).
        seq = cells[hdr + 1 : hdr + 8]
        self.assertEqual(seq[0].cell_type, "markdown")
        self.assertIn("DOI: 10.0/test", seq[0].source)
        self.assertEqual(seq[1].cell_type, "code")
        self.assertIn("CompositeProtocol.from_user_input", seq[1].source)
        self.assertEqual(seq[2].cell_type, "code")
        self.assertIn("paths", seq[2].source)
        self.assertEqual(seq[3].cell_type, "code")
        self.assertIn("parse_e_elect", seq[3].source)
        self.assertEqual(seq[4].cell_type, "code")  # term breakdown
        self.assertEqual(seq[5].cell_type, "code")  # final print
        self.assertIn("FINAL", seq[5].source)
        self.assertEqual(seq[6].cell_type, "markdown")  # interpretation

    def test_section_markdown_includes_formula(self):
        write_composite_notebook(**self.kwargs)
        md_cells = [c for c in self._cells() if c.cell_type == "markdown"]
        has_formula = any("$" in c.source and "E_" in c.source for c in md_cells)
        self.assertTrue(has_formula, "No LaTeX-style formula found in any markdown cell.")

    # --- path rendering ----------------------------------------------------- #

    def test_paths_kept_absolute_and_wrapped_in_resolve_path(self):
        """Absolute paths survive into the paths-dict cell and each is wrapped
        in ``_resolve_path(...)`` so they're rebased through
        ``arc.common.globalize_path`` at notebook execution time. This makes
        the notebook robust to the user moving the project directory or
        running on a different machine — paths under ``<project>/calcs/Species/``
        or ``<project>/calcs/TSs/`` get auto-rebased to the new project root."""
        write_composite_notebook(**self.kwargs)
        paths_cell = next(c for c in self._cells()
                          if c.cell_type == "code" and "paths = " in c.source
                          and "parse_e_elect" not in c.source)
        # Each path value must be wrapped in _resolve_path(...).
        self.assertIn("_resolve_path(", paths_cell.source)
        # Paths are stored as absolute strings so globalize_path can recognise
        # the project-directory prefix and rewrite it. The rewrite is a no-op
        # if the project hasn't moved.
        self.assertIn(self.base_path, paths_cell.source)
        self.assertIn(self.hi_path, paths_cell.source)
        self.assertIn(self.lo_path, paths_cell.source)

    def test_paths_outside_project_survive_resolve_unchanged(self):
        """Paths that don't match the ``/calcs/Species|TSs/`` pattern flow
        through ``globalize_path`` unchanged (it only rebases project-tree
        QM-output paths). They must still appear as absolutes wrapped in
        ``_resolve_path(...)`` — the notebook can't pre-judge what's
        inside vs outside the project."""
        outside_dir = tempfile.mkdtemp()
        outside_path = os.path.join(outside_dir, "far.out")
        _write_gaussian_fixture(outside_path, -1.0)
        section = _make_two_term_section(
            paths={"base": outside_path,
                   "delta_T__high": self.hi_path,
                   "delta_T__low": self.lo_path},
        )
        kwargs = {**self.kwargs, "sections": [section]}
        try:
            write_composite_notebook(**kwargs)
            paths_cell = next(c for c in self._cells()
                              if c.cell_type == "code" and "paths = " in c.source
                              and "parse_e_elect" not in c.source)
            self.assertIn(outside_path, paths_cell.source)
            self.assertIn("_resolve_path(", paths_cell.source)
        finally:
            os.unlink(outside_path)
            os.rmdir(outside_dir)

    def test_setup_cell_defines_resolve_path_and_imports_arc_common(self):
        """The setup cell must define ``_resolve_path`` and import
        ``arc.common`` so the per-section paths-dict cells can call
        ``_resolve_path(...)`` on each absolute path."""
        write_composite_notebook(**self.kwargs)
        # Setup cell is the first code cell after the title/banner markdown.
        setup_cells = [c for c in self._cells()
                       if c.cell_type == "code" and "_resolve_path" in c.source]
        self.assertGreaterEqual(len(setup_cells), 1,
                                "Expected at least one code cell defining _resolve_path.")
        setup_src = setup_cells[0].source
        self.assertIn("arc.common", setup_src)
        self.assertIn("globalize_path", setup_src)
        self.assertIn("def _resolve_path", setup_src)

    # --- determinism -------------------------------------------------------- #

    def test_deterministic_byte_identical_across_runs(self):
        write_composite_notebook(**self.kwargs)
        with open(self.nb_path, "rb") as fh:
            first = fh.read()
        os.unlink(self.nb_path)
        write_composite_notebook(**self.kwargs)
        with open(self.nb_path, "rb") as fh:
            second = fh.read()
        self.assertEqual(first, second)

    def test_cell_ids_are_stable_across_runs(self):
        write_composite_notebook(**self.kwargs)
        ids1 = [c.get("id") for c in self._cells()]
        os.unlink(self.nb_path)
        write_composite_notebook(**self.kwargs)
        ids2 = [c.get("id") for c in self._cells()]
        self.assertEqual(ids1, ids2)

    # --- incremental growth -------------------------------------------------- #

    def test_incremental_growth_preserves_existing_sections(self):
        # First write: one section.
        write_composite_notebook(**self.kwargs)
        one_section_cells = len(self._cells())
        # Second write: same section + a new one. Must include both.
        s2 = _make_two_term_section(label="OH", paths=self.section.sub_job_paths)
        kwargs2 = {**self.kwargs, "sections": [self.section, s2]}
        write_composite_notebook(**kwargs2)
        two_section_cells = len(self._cells())
        self.assertGreater(two_section_cells, one_section_cells)
        # Both species labels are present.
        joined = "\n".join(c.source for c in self._cells())
        self.assertIn("H2O", joined)
        self.assertIn("OH", joined)

    # --- end-to-end executability -------------------------------------------- #

    def test_notebook_executes_and_recomputes_expected_final_value(self):
        """The generated notebook, executed via nbclient, prints the expected final e_elect."""
        write_composite_notebook(**self.kwargs)
        nb = self._read()
        # Ensure the kernel subprocess resolves `arc` from THIS worktree, not any
        # other `arc` package that happens to be on the user's sys.path.
        arc_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))  # .../ARC-wt3
        prior_pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = arc_root + (os.pathsep + prior_pp if prior_pp else "")
        try:
            client = nbclient.NotebookClient(
                nb, timeout=60, resources={"metadata": {"path": self.tmp}}
            )
            client.execute()
        finally:
            if prior_pp:
                os.environ["PYTHONPATH"] = prior_pp
            else:
                os.environ.pop("PYTHONPATH", None)
        expected_kjmol = (
            -76.345678 + (-76.345600 - (-76.346500))
        ) * E_h_kJmol
        stdouts = [
            out.get("text", "")
            for cell in nb.cells
            if cell.cell_type == "code"
            for out in cell.get("outputs", [])
            if out.get("output_type") == "stream" and out.get("name") == "stdout"
        ]
        all_text = "\n".join(stdouts)
        self.assertIn("FINAL", all_text)
        # Compare to 2 decimal places (the print format is :,.3f kJ/mol).
        self.assertIn(f"{expected_kjmol:,.3f}", all_text)


# --------------------------------------------------------------------------- #
#  build_species_report_dict + write_species_report_yaml                      #
# --------------------------------------------------------------------------- #


import yaml  # noqa: E402  (import after the other module-level imports for grouping)


class TestSpeciesReportDict(unittest.TestCase):
    """``build_species_report_dict`` produces the consumable per-species summary.

    The notebook (``sp_composite.ipynb``) is for *independent verification* via
    Run-All; this YAML report is for *consumption* — readable in plain text,
    parseable by tooling, one file per species with every term's contribution
    spelled out next to the QM-output paths backing it.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base_path = os.path.join(self.tmp, "base.out")
        self.hi_path = os.path.join(self.tmp, "delta_T__high.out")
        self.lo_path = os.path.join(self.tmp, "delta_T__low.out")
        _write_gaussian_fixture(self.base_path, -76.345678)
        _write_gaussian_fixture(self.hi_path, -76.346500)   # lower (more negative) → contribution = high - low < 0
        _write_gaussian_fixture(self.lo_path, -76.345600)
        self.section = _make_two_term_section(
            paths={"base": self.base_path,
                   "delta_T__high": self.hi_path,
                   "delta_T__low": self.lo_path},
        )

    def tearDown(self):
        # Tests create nested directories (e.g. Species/H2O/...) — use rmtree
        # to clean them up wholesale rather than enumerating fixture files.
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_top_level_fields(self):
        d = build_species_report_dict(
            section=self.section,
            e_elect_kj_per_mol=-200000.0,
            timestamp="2026-04-30T13:10:32Z",
            arc_version="1.1.0",
            arc_commit="74fc4fa5",
        )
        self.assertEqual(d["species"], "H2O")
        self.assertEqual(d["kind"], "species")
        self.assertEqual(d["generated_at"], "2026-04-30T13:10:32Z")
        self.assertEqual(d["arc_version"], "1.1.0")
        self.assertEqual(d["arc_commit"], "74fc4fa5")

    def test_protocol_block(self):
        d = build_species_report_dict(
            section=self.section,
            e_elect_kj_per_mol=-200000.0,
            timestamp="2026-04-30T13:10:32Z",
            arc_version="1.1.0",
            arc_commit="abc",
        )
        self.assertIsNone(d["protocol"]["preset"])  # explicit recipe in fixture
        self.assertIn("DOI", d["protocol"]["reference"])
        # Formula spells out the sum the protocol evaluates.
        self.assertIn("E_base", d["protocol"]["formula"])
        self.assertIn("delta_T", d["protocol"]["formula"])

    def test_units_block(self):
        d = build_species_report_dict(
            section=self.section,
            e_elect_kj_per_mol=-200000.0,
            timestamp="t",
            arc_version="v",
            arc_commit="c",
        )
        self.assertEqual(d["units"]["energy"], "kJ/mol")
        self.assertEqual(d["units"]["energy_alt"], "Hartree")

    def test_base_block(self):
        d = build_species_report_dict(
            section=self.section,
            e_elect_kj_per_mol=-200000.0,
            timestamp="t",
            arc_version="v",
            arc_commit="c",
        )
        base = d["base"]
        self.assertEqual(base["sub_label"], "base")
        self.assertEqual(base["path"], self.base_path)
        # Energy parsed via arc.parser; cross-check Hartree↔kJ/mol consistency.
        self.assertAlmostEqual(
            base["energy_kj_per_mol"] / E_h_kJmol,
            base["energy_hartree"],
            places=6,
        )

    def test_terms_block_has_one_entry_per_correction(self):
        d = build_species_report_dict(
            section=self.section,
            e_elect_kj_per_mol=-200000.0,
            timestamp="t",
            arc_version="v",
            arc_commit="c",
        )
        self.assertEqual(len(d["terms"]), 1)  # fixture has exactly one correction
        term = d["terms"][0]
        self.assertEqual(term["label"], "delta_T")
        self.assertEqual(term["type"], "DeltaTerm")
        self.assertEqual(len(term["sub_jobs"]), 2)
        self.assertEqual({sj["sub_label"] for sj in term["sub_jobs"]},
                         {"delta_T__high", "delta_T__low"})
        # Contribution = E[high] - E[low] = -76.346500 - (-76.345600) = -0.000900 Ha
        # = -0.000900 × E_h_kJmol ≈ -2.363 kJ/mol
        self.assertAlmostEqual(term["contribution_hartree"], -0.000900, places=6)
        self.assertAlmostEqual(term["contribution_kj_per_mol"],
                               -0.000900 * E_h_kJmol, places=3)

    def test_final_block_uses_caller_supplied_e_elect(self):
        d = build_species_report_dict(
            section=self.section,
            e_elect_kj_per_mol=-200000.123,
            timestamp="t",
            arc_version="v",
            arc_commit="c",
        )
        self.assertEqual(d["final"]["e_elect_kj_per_mol"], -200000.123)
        self.assertAlmostEqual(d["final"]["e_elect_hartree"],
                               -200000.123 / E_h_kJmol, places=6)
        self.assertEqual(d["final"]["e_elect_source"], "sp_composite")

    def test_flags_propagated(self):
        section = _make_two_term_section(
            paths={"base": self.base_path,
                   "delta_T__high": self.hi_path,
                   "delta_T__low": self.lo_path},
            flags=["MRCC degenerate-system fallback for delta_Q__high"],
        )
        d = build_species_report_dict(
            section=section,
            e_elect_kj_per_mol=-200000.0,
            timestamp="t",
            arc_version="v",
            arc_commit="c",
        )
        self.assertEqual(len(d["flags"]), 1)
        self.assertIn("MRCC", d["flags"][0])

    def test_yaml_round_trips(self):
        out = os.path.join(self.tmp, "sp_composite_report.yml")
        write_species_report_yaml(
            path=out,
            section=self.section,
            e_elect_kj_per_mol=-200000.0,
            timestamp="2026-04-30T13:10:32Z",
            arc_version="1.1.0",
            arc_commit="74fc4fa5",
        )
        self.assertTrue(os.path.exists(out))
        with open(out) as fh:
            loaded = yaml.safe_load(fh)
        self.assertEqual(loaded["species"], "H2O")
        self.assertEqual(loaded["kind"], "species")
        self.assertIn("base", loaded)
        self.assertEqual(len(loaded["terms"]), 1)
        self.assertEqual(loaded["terms"][0]["label"], "delta_T")

    def test_yaml_writer_creates_parent_directory(self):
        nested = os.path.join(self.tmp, "Species", "H2O", "sp_composite_report.yml")
        write_species_report_yaml(
            path=nested,
            section=self.section,
            e_elect_kj_per_mol=-200000.0,
            timestamp="t",
            arc_version="v",
            arc_commit="c",
        )
        self.assertTrue(os.path.exists(nested))

    def test_writer_is_deterministic(self):
        """Two writes with the same inputs produce byte-identical files."""
        out_a = os.path.join(self.tmp, "a.yml")
        out_b = os.path.join(self.tmp, "b.yml")
        for out in (out_a, out_b):
            write_species_report_yaml(
                path=out,
                section=self.section,
                e_elect_kj_per_mol=-200000.0,
                timestamp="2026-04-30T13:10:32Z",
                arc_version="1.1.0",
                arc_commit="74fc4fa5",
            )
        with open(out_a, "rb") as fa, open(out_b, "rb") as fb:
            self.assertEqual(fa.read(), fb.read())


# --------------------------------------------------------------------------- #
#  Import placement (module-level, per project guidelines)                    #
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    unittest.main()
