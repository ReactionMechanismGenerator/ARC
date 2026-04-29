"""Tests for the ARC.py end-of-run TCKDB upload sweep dispatcher.

These tests focus on the wiring between ``tckdb.upload_mode`` and the
adapter method that gets called per species. They use a stub adapter so
no network or live ARC objects are required.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

# ARC.py is the top-level entry script; import its sweep helper directly.
import importlib.util
_ARC_PY = Path(__file__).parent / "ARC.py"
_spec = importlib.util.spec_from_file_location("arc_entry", _ARC_PY)
_arc_entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_arc_entry)

from arc.tckdb.adapter import UploadOutcome
from arc.tckdb.config import TCKDBConfig


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class _StubAdapter:
    """Records which adapter method was called per species, no network."""

    def __init__(self, *, conformer_outcome=None, bundle_outcome=None,
                 conformer_raises=None, bundle_raises=None):
        self.conformer_calls = []
        self.bundle_calls = []
        self.artifact_calls = []
        self._conformer_outcome = conformer_outcome
        self._bundle_outcome = bundle_outcome
        self._conformer_raises = conformer_raises
        self._bundle_raises = bundle_raises

    def submit_from_output(self, *, output_doc, species_record):
        self.conformer_calls.append(species_record.get("label"))
        if self._conformer_raises is not None:
            raise self._conformer_raises
        return self._conformer_outcome

    def submit_computed_species_from_output(self, *, output_doc, species_record):
        self.bundle_calls.append(species_record.get("label"))
        if self._bundle_raises is not None:
            raise self._bundle_raises
        return self._bundle_outcome

    def submit_artifacts_for_calculation(self, **kwargs):
        self.artifact_calls.append(kwargs)
        return None


def _outcome(status, *, label="ethanol", error=None,
             primary=None, additional=None):
    """Build a stand-in UploadOutcome with the fields the sweep reads."""
    return UploadOutcome(
        status=status,
        payload_path=Path(f"/tmp/{label}.payload.json"),
        sidecar_path=Path(f"/tmp/{label}.meta.json"),
        idempotency_key=f"arc:test:{label}:k:abc1234567890def",
        error=error,
        primary_calculation=primary,
        additional_calculations=additional or [],
    )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _write_output_yml(project_dir: str, *, species_labels=("CCO",), with_ts=False):
    """Write a minimal ``output.yml`` matching what the sweep reads."""
    out_dir = os.path.join(project_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    doc = {
        "schema_version": "1.0",
        "project": "test_project",
        "arc_version": "0.0.0",
        "opt_level": {"method": "wb97xd", "basis": "def2-tzvp", "software": "gaussian"},
        "species": [
            {
                "label": label,
                "smiles": "CCO",
                "charge": 0,
                "multiplicity": 1,
                "is_ts": False,
                "converged": True,
                "xyz": "C 0.0 0.0 0.0\nH 1.0 0.0 0.0",
                "opt_n_steps": 12,
                "opt_final_energy_hartree": -154.0,
                "ess_versions": {"opt": "Gaussian 16, Revision A.03"},
            }
            for label in species_labels
        ],
        "transition_states": [
            {"label": "TS0", "is_ts": True, "converged": True}
        ] if with_ts else [],
    }
    with open(os.path.join(out_dir, "output.yml"), "w") as f:
        yaml.safe_dump(doc, f)
    return doc


# --------------------------------------------------------------------------
# Dispatch behavior
# --------------------------------------------------------------------------


class TestRunTckdbUploadSweepDispatch(unittest.TestCase):
    """Wiring tests: which adapter method gets called per upload_mode."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-sweep-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        _write_output_yml(self.tmp)
        self.arc_object = SimpleNamespace(project_directory=self.tmp)

    def _cfg(self, **overrides):
        defaults = dict(
            enabled=True,
            base_url="http://localhost:8000/api/v1",
            api_key_env="X_TCKDB_API_KEY",
        )
        defaults.update(overrides)
        return TCKDBConfig(**defaults)

    # ---------------- 1: missing upload_mode → conformer (default)
    def test_default_mode_uses_legacy_conformer_path(self):
        cfg = self._cfg()  # upload_mode defaults to "conformer"
        adapter = _StubAdapter(conformer_outcome=_outcome("uploaded"))
        _arc_entry._run_tckdb_upload_sweep(self.arc_object, adapter, cfg)
        self.assertEqual(adapter.conformer_calls, ["CCO"])
        self.assertEqual(adapter.bundle_calls, [])

    # ---------------- 2: explicit conformer
    def test_explicit_conformer_mode_uses_legacy_path(self):
        cfg = self._cfg(upload_mode="conformer")
        adapter = _StubAdapter(conformer_outcome=_outcome("uploaded"))
        _arc_entry._run_tckdb_upload_sweep(self.arc_object, adapter, cfg)
        self.assertEqual(adapter.conformer_calls, ["CCO"])
        self.assertEqual(adapter.bundle_calls, [])

    # ---------------- 3: computed_species → bundle path
    def test_computed_species_mode_dispatches_bundle(self):
        cfg = self._cfg(upload_mode="computed_species")
        adapter = _StubAdapter(bundle_outcome=_outcome("uploaded"))
        _arc_entry._run_tckdb_upload_sweep(self.arc_object, adapter, cfg)
        self.assertEqual(adapter.bundle_calls, ["CCO"])
        self.assertEqual(adapter.conformer_calls, [])

    # ---------------- 4: bundle path never calls legacy
    def test_computed_species_does_not_call_legacy_submit(self):
        # Multiple species so we'd notice any leak across iterations.
        _write_output_yml(self.tmp, species_labels=("CCO", "CO", "CC"))
        cfg = self._cfg(upload_mode="computed_species")
        adapter = _StubAdapter(bundle_outcome=_outcome("uploaded"))
        _arc_entry._run_tckdb_upload_sweep(self.arc_object, adapter, cfg)
        self.assertEqual(adapter.bundle_calls, ["CCO", "CO", "CC"])
        self.assertEqual(adapter.conformer_calls, [])
        # And no per-artifact sweep call: bundles inline artifacts.
        self.assertEqual(adapter.artifact_calls, [])

    # ---------------- 5: failure in bundle mode is recorded; sweep continues
    def test_computed_species_failure_continues_to_next_species(self):
        _write_output_yml(self.tmp, species_labels=("CCO", "CO"))
        cfg = self._cfg(upload_mode="computed_species")
        # First species: outcome with status=failed (non-strict path).
        # Second species: outcome with status=uploaded.
        # We achieve "different per call" by mutating the stub's outcome
        # mid-sweep, since the stub returns the same outcome each call by
        # default. Use a side-effect via a wrapper instead.
        outcomes = iter([
            _outcome("failed", label="CCO", error="HTTP 503"),
            _outcome("uploaded", label="CO"),
        ])
        adapter = _StubAdapter()
        adapter.submit_computed_species_from_output = (
            lambda *, output_doc, species_record: (
                adapter.bundle_calls.append(species_record.get("label"))
                or next(outcomes)
            )
        )
        _arc_entry._run_tckdb_upload_sweep(self.arc_object, adapter, cfg)
        # Both species processed; first failed, second uploaded.
        self.assertEqual(adapter.bundle_calls, ["CCO", "CO"])

    # ---------------- 5b: an unhandled exception in bundle mode is caught
    def test_computed_species_exception_is_caught_and_logged(self):
        _write_output_yml(self.tmp, species_labels=("CCO", "CO"))
        cfg = self._cfg(upload_mode="computed_species")
        # Simulate an unhandled exception on the FIRST species; second
        # should still be attempted (matches conformer-mode behavior).
        call_log = []
        def fake_submit(*, output_doc, species_record):
            label = species_record.get("label")
            call_log.append(label)
            if label == "CCO":
                raise RuntimeError("boom")
            return _outcome("uploaded", label=label)
        adapter = _StubAdapter()
        adapter.submit_computed_species_from_output = fake_submit
        _arc_entry._run_tckdb_upload_sweep(self.arc_object, adapter, cfg)
        self.assertEqual(call_log, ["CCO", "CO"])

    # ---------------- 6: sidecar written before live upload failure (bundle)
    def test_bundle_mode_sidecar_written_before_upload_failure(self):
        # This is fundamentally an adapter-level guarantee, but we verify
        # the wiring preserves it: a "failed" outcome carrying real
        # payload_path and sidecar_path values means the sweep still
        # passes those upward to the user.
        cfg = self._cfg(upload_mode="computed_species")
        sentinel_payload = Path("/tmp/sentinel.payload.json")
        sentinel_sidecar = Path("/tmp/sentinel.meta.json")
        outcome = UploadOutcome(
            status="failed",
            payload_path=sentinel_payload,
            sidecar_path=sentinel_sidecar,
            idempotency_key="arc:t:CCO:c:abc1234567890def",
            error="HTTP 503",
        )
        adapter = _StubAdapter(bundle_outcome=outcome)
        # Capture stdout to confirm the failure summary is printed
        # (don't assert on exact text — assert on key tokens).
        with mock.patch("builtins.print") as mock_print:
            _arc_entry._run_tckdb_upload_sweep(self.arc_object, adapter, cfg)
        printed = "\n".join(str(c.args[0]) for c in mock_print.call_args_list)
        self.assertIn("computed-species bundle", printed)
        self.assertIn("failed: 1", printed)
        self.assertIn("HTTP 503", printed)


# --------------------------------------------------------------------------
# Summary-print mode awareness
# --------------------------------------------------------------------------


class TestSweepSummaryByMode(unittest.TestCase):
    """The summary line names the mode; bundle mode omits the artifact line."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-sweep-summary-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        _write_output_yml(self.tmp)
        self.arc_object = SimpleNamespace(project_directory=self.tmp)

    def _run_with_mode(self, *, upload_mode, artifacts_upload=False):
        from arc.tckdb.config import TCKDBArtifactConfig
        cfg = TCKDBConfig(
            enabled=True, base_url="http://x", api_key_env="X",
            upload_mode=upload_mode,
            artifacts=TCKDBArtifactConfig(upload=artifacts_upload),
        )
        adapter = _StubAdapter(
            conformer_outcome=_outcome("uploaded"),
            bundle_outcome=_outcome("uploaded"),
        )
        with mock.patch("builtins.print") as mock_print:
            _arc_entry._run_tckdb_upload_sweep(self.arc_object, adapter, cfg)
        return "\n".join(str(c.args[0]) for c in mock_print.call_args_list)

    def test_conformer_mode_summary_says_conformer(self):
        out = self._run_with_mode(upload_mode="conformer")
        self.assertIn("conformer/calculation", out)
        self.assertNotIn("computed-species bundle", out)

    def test_bundle_mode_summary_says_bundle(self):
        out = self._run_with_mode(upload_mode="computed_species")
        self.assertIn("computed-species bundle", out)
        self.assertNotIn("conformer/calculation", out)

    def test_bundle_mode_omits_artifact_line_even_when_enabled(self):
        # Inline artifacts mean the standalone artifact tally would mislead.
        out = self._run_with_mode(upload_mode="computed_species", artifacts_upload=True)
        self.assertNotIn("artifacts: uploaded", out)

    def test_conformer_mode_emits_artifact_line_when_enabled(self):
        out = self._run_with_mode(upload_mode="conformer", artifacts_upload=True)
        self.assertIn("artifacts:", out)


# --------------------------------------------------------------------------
# _resolve_artifact_path: prefer recorded <calc>_input over derivation
# --------------------------------------------------------------------------


class TestResolveArtifactPath(unittest.TestCase):
    """The legacy artifact sweep prefers ``output.yml``'s ``<calc>_input``
    field, falling back to settings-based derivation only when absent."""

    def test_input_kind_prefers_recorded_field(self):
        """When ``opt_input`` is on the record, it wins over the derived path."""
        species_record = {
            "opt_log": "calcs/CH4/opt/input.log",
            "opt_input": "calcs/CH4/opt/explicit_input.gjf",   # NEW field
        }
        output_doc = {"opt_level": {"software": "gaussian"}}
        path = _arc_entry._resolve_artifact_path(
            kind="input", calc_type="opt",
            species_record=species_record, output_doc=output_doc,
        )
        self.assertEqual(path, "calcs/CH4/opt/explicit_input.gjf")

    def test_input_kind_falls_back_to_settings_when_field_absent(self):
        """Older output.yml without ``<calc>_input`` still resolves via settings."""
        species_record = {"opt_log": "/abs/calcs/CH4/opt/input.log"}
        output_doc = {"opt_level": {"software": "gaussian"}}
        path = _arc_entry._resolve_artifact_path(
            kind="input", calc_type="opt",
            species_record=species_record, output_doc=output_doc,
        )
        # Derived sibling: input.gjf next to the log.
        self.assertEqual(path, "/abs/calcs/CH4/opt/input.gjf")

    def test_input_kind_falls_back_when_recorded_field_is_none(self):
        """Explicit ``None`` in the record (deck wasn't kept) → fallback."""
        species_record = {
            "opt_log": "/abs/calcs/CH4/opt/input.log",
            "opt_input": None,
        }
        output_doc = {"opt_level": {"software": "gaussian"}}
        path = _arc_entry._resolve_artifact_path(
            kind="input", calc_type="opt",
            species_record=species_record, output_doc=output_doc,
        )
        self.assertEqual(path, "/abs/calcs/CH4/opt/input.gjf")

    def test_input_kind_per_job_picks_correct_recorded_field(self):
        """Different calcs hit different ``<calc>_input`` fields, not all opt's."""
        species_record = {
            "opt_log": "/abs/opt.log", "opt_input": "/abs/opt_deck.gjf",
            "freq_log": "/abs/freq.log", "freq_input": "/abs/freq_deck.gjf",
            "sp_log": "/abs/sp.log", "sp_input": "/abs/sp_deck.in",  # cross-software run
        }
        output_doc = {"opt_level": {"software": "gaussian"}}
        for calc, expected in (
            ("opt", "/abs/opt_deck.gjf"),
            ("freq", "/abs/freq_deck.gjf"),
            ("sp", "/abs/sp_deck.in"),  # NOT input.gjf — sp uses its own software
        ):
            path = _arc_entry._resolve_artifact_path(
                kind="input", calc_type=calc,
                species_record=species_record, output_doc=output_doc,
            )
            self.assertEqual(path, expected,
                             msg=f"{calc}: expected {expected}, got {path}")


if __name__ == "__main__":
    unittest.main()
