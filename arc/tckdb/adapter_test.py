#!/usr/bin/env python3
# encoding: utf-8

"""Unit tests for arc.tckdb.adapter.

These tests do not require a live TCKDB server. The TCKDBClient is
replaced by a stub via the adapter's ``client_factory`` parameter.
"""

import json
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

from arc.tckdb.adapter import (
    ARTIFACTS_ENDPOINT_TEMPLATE,
    ArtifactUploadOutcome,
    CONFORMER_UPLOAD_ENDPOINT,
    PAYLOAD_KIND,
    TCKDBAdapter,
    UploadOutcome,
)
from arc.tckdb.config import TCKDBArtifactConfig, TCKDBConfig


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, data, status_code=201, replayed=False):
        self.data = data
        self.status_code = status_code
        self.idempotency_replayed = replayed


class _StubClient:
    """Minimal TCKDBClient lookalike for adapter tests."""

    def __init__(self, *, response=None, raise_exc=None):
        self._response = response
        self._raise_exc = raise_exc
        self.calls = []
        self.closed = False

    def request_json(self, method, path, *, json=None, idempotency_key=None):
        self.calls.append(
            dict(method=method, path=path, json=json, idempotency_key=idempotency_key)
        )
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response

    def close(self):
        self.closed = True


def _fake_record(label="ethanol", smiles="CCO", charge=0, multiplicity=1, is_ts=False):
    """Mimics one entry from output.yml's `species:` (or `transition_states:`) list.

    Note ``ess_versions`` is keyed by job type (``opt``/``freq``/``sp``/``neb``),
    matching ``arc/output.py::_get_ess_versions``.
    """
    return {
        "label": label,
        "smiles": smiles,
        "charge": charge,
        "multiplicity": multiplicity,
        "is_ts": is_ts,
        "converged": True,
        "xyz": "C 0.0 0.0 0.0\nH 1.0 0.0 0.0",
        "opt_n_steps": 12,
        "opt_final_energy_hartree": -154.123,
        "ess_versions": {"opt": "Gaussian 16, Revision A.03"},
    }


def _fake_output_doc():
    """Mimics the top-level fields of output.yml that the adapter reads."""
    return {
        "schema_version": "1.0",
        "project": "test_project",
        "arc_version": "1.2.3",
        "arc_git_commit": "deadbeef",
        "opt_level": {"method": "wb97xd", "basis": "def2-tzvp", "software": "gaussian"},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdapterDisabled(unittest.TestCase):
    """Test 1: disabled config does nothing."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_disabled_returns_none_and_writes_nothing(self):
        cfg = TCKDBConfig(enabled=False, base_url="http://x", payload_dir=self.tmp)
        client = _StubClient(response=_StubResponse({"ok": True}))
        adapter = TCKDBAdapter(cfg, client_factory=lambda c, k: client)
        outcome = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())
        self.assertIsNone(outcome)
        self.assertEqual(os.listdir(self.tmp), [])
        self.assertEqual(client.calls, [])


class TestAdapterPayloadAndUpload(unittest.TestCase):
    """Tests 2, 3, 9, 10, 11: payload + sidecar + replay metadata + no DB IDs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cfg = TCKDBConfig(
            enabled=True,
            base_url="http://localhost:8000/api/v1",
            payload_dir=self.tmp,
            api_key_env="X_TCKDB_API_KEY",
            project_label="proj-A",
        )

    def _adapter(self, client):
        return TCKDBAdapter(self.cfg, client_factory=lambda c, k: client)

    def test_payload_written_before_failed_upload(self):
        """Test 2: payload written even when upload fails."""
        client = _StubClient(raise_exc=RuntimeError("network down"))
        adapter = self._adapter(client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())

        self.assertEqual(outcome.status, "failed")
        self.assertTrue(outcome.payload_path.exists())
        self.assertTrue(outcome.sidecar_path.exists())
        sc = json.loads(outcome.sidecar_path.read_text())
        self.assertEqual(sc["status"], "failed")
        self.assertIn("network down", sc["last_error"])
        # Payload file is intact and parseable
        json.loads(outcome.payload_path.read_text())

    def test_upload_success_updates_sidecar(self):
        """Test 3: upload success -> sidecar status = uploaded."""
        client = _StubClient(response=_StubResponse({"conformer_observation_id": 42}))
        adapter = self._adapter(client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())

        self.assertEqual(outcome.status, "uploaded")
        sc = json.loads(outcome.sidecar_path.read_text())
        self.assertEqual(sc["status"], "uploaded")
        self.assertIsNotNone(sc["uploaded_at"])
        self.assertEqual(sc["response_status_code"], 201)
        self.assertEqual(sc["idempotency_replayed"], False)
        # Test 11: replay-ready metadata present
        self.assertEqual(sc["endpoint"], CONFORMER_UPLOAD_ENDPOINT)
        self.assertEqual(sc["payload_kind"], PAYLOAD_KIND)
        self.assertEqual(sc["payload_file"], str(outcome.payload_path))
        self.assertTrue(sc["idempotency_key"])
        self.assertEqual(sc["base_url"], self.cfg.base_url)

    def test_upload_records_idempotency_replay(self):
        client = _StubClient(
            response=_StubResponse({"conformer_observation_id": 42}, replayed=True)
        )
        adapter = self._adapter(client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())
        sc = json.loads(outcome.sidecar_path.read_text())
        self.assertTrue(sc["idempotency_replayed"])

    def test_payload_contains_no_db_ids(self):
        """Test 9: payload must not include raw TCKDB DB IDs."""
        client = _StubClient(response=_StubResponse({"id": 1}))
        adapter = self._adapter(client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())
        flat = outcome.payload_path.read_text()
        for forbidden in (
            '"species_id"',
            '"species_entry_id"',
            '"calculation_id"',
            '"conformer_observation_id"',
            '"literature_id"',
            '"software_release_id"',
            '"workflow_tool_release_id"',
        ):
            self.assertNotIn(forbidden, flat, msg=f"raw DB id in payload: {forbidden}")

    def test_payload_validates_against_expected_shape(self):
        """Test 10: payload has the conformer-upload top-level shape."""
        client = _StubClient(response=_StubResponse({"id": 1}))
        adapter = self._adapter(client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())
        payload = json.loads(outcome.payload_path.read_text())
        self.assertIn("species_entry", payload)
        self.assertIn("geometry", payload)
        self.assertIn("calculation", payload)
        self.assertEqual(payload["species_entry"]["smiles"], "CCO")
        self.assertEqual(
            payload["geometry"]["xyz_text"],
            "2\nethanol\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0",
        )
        self.assertEqual(payload["calculation"]["type"], "opt")
        self.assertEqual(payload["calculation"]["software_release"]["name"], "gaussian")
        self.assertEqual(
            payload["calculation"]["software_release"]["version"],
            "Gaussian 16, Revision A.03",
        )
        self.assertEqual(payload["calculation"]["level_of_theory"]["method"], "wb97xd")
        self.assertEqual(payload["calculation"]["level_of_theory"]["basis"], "def2-tzvp")


class TestAdapterSkipped(unittest.TestCase):
    """Test 4: upload=false -> payload written, sidecar skipped, no network."""

    def test_upload_skipped_writes_payload_no_call(self):
        tmp = tempfile.mkdtemp(prefix="arc-tckdb-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        cfg = TCKDBConfig(
            enabled=True,
            base_url="http://x",
            payload_dir=tmp,
            upload=False,
        )
        client = _StubClient(response=_StubResponse({"ok": True}))
        adapter = TCKDBAdapter(cfg, client_factory=lambda c, k: client)
        outcome = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())
        self.assertEqual(outcome.status, "skipped")
        self.assertTrue(outcome.payload_path.exists())
        sc = json.loads(outcome.sidecar_path.read_text())
        self.assertEqual(sc["status"], "skipped")
        self.assertEqual(client.calls, [])


class TestAdapterStrict(unittest.TestCase):
    """Tests 5, 6: strict raises; non-strict swallows."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_strict_mode_raises_and_records(self):
        cfg = TCKDBConfig(
            enabled=True,
            base_url="http://x",
            payload_dir=self.tmp,
            api_key_env="X_TCKDB_API_KEY",
            strict=True,
        )
        client = _StubClient(raise_exc=RuntimeError("503"))
        adapter = TCKDBAdapter(cfg, client_factory=lambda c, k: client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            with self.assertRaises(RuntimeError):
                adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())
        # Sidecar still written, status failed
        files = os.listdir(os.path.join(self.tmp, "conformer_calculation"))
        sidecar = [f for f in files if f.endswith(".meta.json")][0]
        sc = json.loads(open(os.path.join(self.tmp, "conformer_calculation", sidecar)).read())
        self.assertEqual(sc["status"], "failed")

    def test_non_strict_does_not_raise(self):
        cfg = TCKDBConfig(
            enabled=True,
            base_url="http://x",
            payload_dir=self.tmp,
            api_key_env="X_TCKDB_API_KEY",
            strict=False,
        )
        client = _StubClient(raise_exc=RuntimeError("503"))
        adapter = TCKDBAdapter(cfg, client_factory=lambda c, k: client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())
        self.assertEqual(outcome.status, "failed")


class TestAdapterIdempotency(unittest.TestCase):
    """Test 7: same logical input -> same key; changed input -> different key."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cfg = TCKDBConfig(
            enabled=True,
            base_url="http://x",
            payload_dir=self.tmp,
            api_key_env="X_TCKDB_API_KEY",
            project_label="proj-A",
        )

    def test_idempotency_key_stable_and_distinct(self):
        client_a = _StubClient(response=_StubResponse({"id": 1}))
        adapter = TCKDBAdapter(self.cfg, client_factory=lambda c, k: client_a)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            o1 = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())
            o2 = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())
            o3 = adapter.submit_from_output(
                output_doc=_fake_output_doc(),
                species_record=_fake_record(label="methanol", smiles="CO"),
            )
        self.assertEqual(o1.idempotency_key, o2.idempotency_key)
        self.assertNotEqual(o1.idempotency_key, o3.idempotency_key)
        # The header sent must match the recorded key.
        sent_keys = {call["idempotency_key"] for call in client_a.calls}
        self.assertEqual(sent_keys, {o1.idempotency_key, o3.idempotency_key})


class TestAdapterApiKey(unittest.TestCase):
    """Test 8: missing API key with upload=true produces failed sidecar (no raise)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_missing_api_key_records_failure(self):
        cfg = TCKDBConfig(
            enabled=True,
            base_url="http://x",
            payload_dir=self.tmp,
            api_key_env="DEFINITELY_NOT_SET_X_X",
        )
        client = _StubClient(response=_StubResponse({"ok": True}))
        adapter = TCKDBAdapter(cfg, client_factory=lambda c, k: client)
        os.environ.pop("DEFINITELY_NOT_SET_X_X", None)
        outcome = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(client.calls, [])  # never called the network
        sc = json.loads(outcome.sidecar_path.read_text())
        self.assertIn("DEFINITELY_NOT_SET_X_X", sc["last_error"])

    def test_missing_api_key_strict_raises(self):
        cfg = TCKDBConfig(
            enabled=True,
            base_url="http://x",
            payload_dir=self.tmp,
            api_key_env="DEFINITELY_NOT_SET_X_X",
            strict=True,
        )
        client = _StubClient(response=_StubResponse({"ok": True}))
        adapter = TCKDBAdapter(cfg, client_factory=lambda c, k: client)
        os.environ.pop("DEFINITELY_NOT_SET_X_X", None)
        with self.assertRaises(ValueError):
            adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())


# ---------------------------------------------------------------------------
# Artifact upload tests
# ---------------------------------------------------------------------------


_GAUSSIAN_LOG_HEADER = (
    b" Gaussian, Inc.,  Pittsburgh PA, All Rights Reserved.\n"
    b" Cite this work as:\n Gaussian 16, Revision A.03\n"
)


class TestArtifactUpload(unittest.TestCase):
    """Tests for TCKDBAdapter.submit_artifacts_for_calculation."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-art-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.project_dir = os.path.join(self.tmp, "project")
        os.makedirs(self.project_dir)
        self.payload_dir = "tckdb_payloads"   # relative -> resolved under project_dir
        self.log_path = os.path.join(self.project_dir, "calcs", "Species", "ethanol", "opt", "output.log")
        os.makedirs(os.path.dirname(self.log_path))
        with open(self.log_path, "wb") as fh:
            fh.write(_GAUSSIAN_LOG_HEADER)

    def _cfg(self, *, artifacts=None, strict=False, api_key_env="X_TCKDB_API_KEY"):
        return TCKDBConfig(
            enabled=True,
            base_url="http://localhost:8000/api/v1",
            payload_dir=self.payload_dir,
            api_key_env=api_key_env,
            project_label="proj-A",
            strict=strict,
            artifacts=artifacts or TCKDBArtifactConfig(upload=True, kinds=("output_log",)),
        )

    def _adapter(self, client, cfg=None):
        return TCKDBAdapter(
            cfg or self._cfg(),
            project_directory=self.project_dir,
            client_factory=lambda c, k: client,
        )

    def _submit(
        self,
        adapter,
        *,
        log_path=None,
        calculation_id=42,
        calculation_type="opt",
        kind="output_log",
    ):
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            return adapter.submit_artifacts_for_calculation(
                output_doc=_fake_output_doc(),
                species_record=_fake_record(),
                calculation_id=calculation_id,
                calculation_type=calculation_type,
                file_path=log_path or self.log_path,
                kind=kind,
            )

    def test_disabled_artifact_upload_returns_skipped_no_call(self):
        cfg = self._cfg(artifacts=TCKDBArtifactConfig(upload=False))
        client = _StubClient(response=_StubResponse({}, status_code=201))
        adapter = self._adapter(client, cfg=cfg)
        outcome = self._submit(adapter)
        self.assertEqual(outcome.status, "skipped")
        self.assertEqual(client.calls, [])
        self.assertIn("artifacts.upload", outcome.skip_reason)

    def test_kind_not_in_config_returns_skipped(self):
        cfg = self._cfg(artifacts=TCKDBArtifactConfig(upload=True, kinds=("input",)))
        client = _StubClient(response=_StubResponse({}, status_code=201))
        adapter = self._adapter(client, cfg=cfg)
        outcome = self._submit(adapter)
        self.assertEqual(outcome.status, "skipped")
        self.assertEqual(client.calls, [])
        self.assertIn("not in config.kinds", outcome.skip_reason)

    def test_missing_log_file_returns_skipped(self):
        client = _StubClient(response=_StubResponse({}, status_code=201))
        adapter = self._adapter(client)
        outcome = self._submit(adapter, log_path=os.path.join(self.project_dir, "no_such_log.log"))
        self.assertEqual(outcome.status, "skipped")
        self.assertEqual(client.calls, [])
        self.assertIn("file missing", outcome.skip_reason)

    def test_oversized_log_returns_skipped(self):
        cfg = self._cfg(
            artifacts=TCKDBArtifactConfig(upload=True, kinds=("output_log",), max_size_mb=1)
        )
        # Write a 2 MB file
        big_log = os.path.join(self.project_dir, "big.log")
        with open(big_log, "wb") as fh:
            fh.write(b"\x00" * (2 * 1024 * 1024))
        client = _StubClient(response=_StubResponse({}, status_code=201))
        adapter = self._adapter(client, cfg=cfg)
        outcome = self._submit(adapter, log_path=big_log)
        self.assertEqual(outcome.status, "skipped")
        self.assertEqual(client.calls, [])
        self.assertIn(">1 MB cap", outcome.skip_reason)

    def test_successful_upload_writes_sidecar_and_posts(self):
        client = _StubClient(response=_StubResponse(
            {"calculation_id": 42, "artifacts": [{"id": 7}]},
            status_code=201,
        ))
        adapter = self._adapter(client)
        outcome = self._submit(adapter)
        self.assertEqual(outcome.status, "uploaded")
        self.assertEqual(outcome.calculation_id, 42)
        self.assertEqual(outcome.kind, "output_log")
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["path"], "/calculations/42/artifacts")
        # Body shape
        artifacts = call["json"]["artifacts"]
        self.assertEqual(len(artifacts), 1)
        a = artifacts[0]
        self.assertEqual(a["kind"], "output_log")
        self.assertEqual(a["filename"], "output.log")
        self.assertEqual(a["bytes"], len(_GAUSSIAN_LOG_HEADER))
        self.assertEqual(len(a["sha256"]), 64)
        self.assertRegex(a["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(a["content_base64"])
        # Sidecar
        sc = json.loads(outcome.sidecar_path.read_text())
        self.assertEqual(sc["status"], "uploaded")
        self.assertEqual(sc["calculation_id"], 42)
        self.assertEqual(sc["kind"], "output_log")
        self.assertEqual(sc["bytes"], len(_GAUSSIAN_LOG_HEADER))

    def test_strict_failure_raises(self):
        cfg = self._cfg(strict=True)
        client = _StubClient(raise_exc=RuntimeError("422 ESS signature missing"))
        adapter = self._adapter(client, cfg=cfg)
        with self.assertRaises(RuntimeError):
            self._submit(adapter)
        # Sidecar still recorded as failed
        sidecar_files = [
            f for f in os.listdir(os.path.join(self.project_dir, self.payload_dir, "calculation_artifacts"))
            if f.endswith(".artifact.meta.json")
        ]
        self.assertEqual(len(sidecar_files), 1)
        sc = json.loads(open(os.path.join(self.project_dir, self.payload_dir,
                                          "calculation_artifacts", sidecar_files[0])).read())
        self.assertEqual(sc["status"], "failed")
        self.assertIn("422", sc["last_error"])

    def test_non_strict_failure_returns_outcome(self):
        client = _StubClient(raise_exc=RuntimeError("422 ESS signature missing"))
        adapter = self._adapter(client)
        outcome = self._submit(adapter)
        self.assertEqual(outcome.status, "failed")
        self.assertIn("422", outcome.error)
        # Sidecar marks failure
        sc = json.loads(outcome.sidecar_path.read_text())
        self.assertEqual(sc["status"], "failed")

    def test_idempotency_replay_recorded(self):
        client = _StubClient(response=_StubResponse(
            {"calculation_id": 42, "artifacts": [{"id": 7}]},
            status_code=201,
            replayed=True,
        ))
        adapter = self._adapter(client)
        outcome = self._submit(adapter)
        self.assertEqual(outcome.status, "uploaded")
        sc = json.loads(outcome.sidecar_path.read_text())
        self.assertTrue(sc["idempotency_replayed"])

    def test_idempotency_key_stable_for_same_inputs(self):
        client = _StubClient(response=_StubResponse(
            {"calculation_id": 42, "artifacts": []}, status_code=201
        ))
        adapter = self._adapter(client)
        o1 = self._submit(adapter)
        o2 = self._submit(adapter)
        self.assertEqual(o1.idempotency_key, o2.idempotency_key)
        # Both calls sent the same key header
        keys = {c["idempotency_key"] for c in client.calls}
        self.assertEqual(keys, {o1.idempotency_key})

    def test_idempotency_key_distinct_for_different_calc(self):
        client = _StubClient(response=_StubResponse(
            {"calculation_id": 99, "artifacts": []}, status_code=201
        ))
        adapter = self._adapter(client)
        o1 = self._submit(adapter, calculation_id=42)
        o2 = self._submit(adapter, calculation_id=99)
        self.assertNotEqual(o1.idempotency_key, o2.idempotency_key)

    def test_endpoint_template_matches_artifact_request(self):
        client = _StubClient(response=_StubResponse(
            {"calculation_id": 314, "artifacts": []}, status_code=201
        ))
        adapter = self._adapter(client)
        outcome = self._submit(adapter, calculation_id=314)
        self.assertEqual(client.calls[0]["path"], ARTIFACTS_ENDPOINT_TEMPLATE.format(calculation_id=314))
        self.assertEqual(outcome.status, "uploaded")

    def test_input_kind_uploads_with_correct_marshalling(self):
        # Input deck file (input.gjf) gets uploaded with kind="input".
        # Use a config that allows both kinds, then upload an input file.
        cfg = self._cfg(
            artifacts=TCKDBArtifactConfig(
                upload=True,
                kinds=("output_log", "input"),
            )
        )
        input_path = os.path.join(self.project_dir, "calcs", "Species", "ethanol", "opt", "input.gjf")
        with open(input_path, "wb") as fh:
            fh.write(b"%mem=42GB\n# wb97xd/def2tzvp opt\n")
        client = _StubClient(response=_StubResponse(
            {"calculation_id": 42, "artifacts": []}, status_code=201
        ))
        adapter = self._adapter(client, cfg=cfg)
        outcome = self._submit(adapter, log_path=input_path, kind="input")
        self.assertEqual(outcome.status, "uploaded")
        self.assertEqual(outcome.kind, "input")
        self.assertEqual(client.calls[0]["json"]["artifacts"][0]["kind"], "input")
        self.assertEqual(client.calls[0]["json"]["artifacts"][0]["filename"], "input.gjf")

    def test_unimplemented_kind_defensive_skip(self):
        # Even if the user somehow gets `checkpoint` into config.kinds
        # AND a caller passes kind="checkpoint", the adapter refuses
        # rather than uploading bytes that may not match the kind's
        # semantic contract.
        cfg = self._cfg(
            artifacts=TCKDBArtifactConfig(
                upload=True,
                kinds=("output_log", "checkpoint"),  # parse-time warning only
            )
        )
        client = _StubClient(response=_StubResponse({}, status_code=201))
        adapter = self._adapter(client, cfg=cfg)
        outcome = self._submit(adapter, kind="checkpoint")
        self.assertEqual(outcome.status, "skipped")
        self.assertIn("no upload path yet", outcome.skip_reason)
        self.assertEqual(client.calls, [])


class TestUploadOutcomeCalcRefs(unittest.TestCase):
    """Conformer upload exposes primary/additional calc refs from response."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_calc_refs_extracted_from_response(self):
        cfg = TCKDBConfig(
            enabled=True,
            base_url="http://localhost:8000/api/v1",
            payload_dir=self.tmp,
            api_key_env="X_TCKDB_API_KEY",
            project_label="proj-A",
        )
        client = _StubClient(response=_StubResponse({
            "id": 1,
            "primary_calculation": {"calculation_id": 10, "type": "opt", "request_index": None},
            "additional_calculations": [
                {"calculation_id": 11, "type": "freq", "request_index": 0},
                {"calculation_id": 12, "type": "sp", "request_index": 1},
            ],
        }))
        adapter = TCKDBAdapter(cfg, client_factory=lambda c, k: client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())
        self.assertEqual(outcome.status, "uploaded")
        self.assertEqual(outcome.primary_calculation["calculation_id"], 10)
        self.assertEqual(outcome.primary_calculation["type"], "opt")
        self.assertEqual(len(outcome.additional_calculations), 2)
        self.assertEqual(outcome.additional_calculations[0]["calculation_id"], 11)
        self.assertEqual(outcome.additional_calculations[1]["type"], "sp")

    def test_calc_refs_default_when_response_omits_them(self):
        cfg = TCKDBConfig(
            enabled=True, base_url="http://x", payload_dir=self.tmp,
            api_key_env="X_TCKDB_API_KEY",
        )
        client = _StubClient(response=_StubResponse({"id": 1}))
        adapter = TCKDBAdapter(cfg, client_factory=lambda c, k: client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_from_output(output_doc=_fake_output_doc(), species_record=_fake_record())
        self.assertIsNone(outcome.primary_calculation)
        self.assertEqual(outcome.additional_calculations, [])


class TestAdditionalCalculations(unittest.TestCase):
    """opt/freq/sp chain: payload must carry additional_calculations for freq+sp."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cfg = TCKDBConfig(
            enabled=True,
            base_url="http://localhost:8000/api/v1",
            payload_dir=self.tmp,
            api_key_env="X_TCKDB_API_KEY",
            project_label="proj-A",
        )

    def _adapter(self, client):
        return TCKDBAdapter(self.cfg, client_factory=lambda c, k: client)

    def _submit(self, *, output_doc, record):
        client = _StubClient(response=_StubResponse({"id": 1}))
        adapter = self._adapter(client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_from_output(output_doc=output_doc, species_record=record)
        return outcome, client, json.loads(outcome.payload_path.read_text())

    def test_opt_only_record_has_no_additional_calculations(self):
        """1. opt-only record produces no additional_calculations key."""
        record = _fake_record()  # no freq_*, no sp_energy_hartree
        _, _, payload = self._submit(output_doc=_fake_output_doc(), record=record)
        self.assertNotIn("additional_calculations", payload)

    def test_opt_plus_freq_record_yields_one_freq_additional(self):
        """2. opt+freq record produces one freq additional calculation."""
        record = _fake_record()
        record["freq_n_imag"] = 0
        record["zpe_hartree"] = 0.024131
        _, _, payload = self._submit(output_doc=_fake_output_doc(), record=record)
        additional = payload["additional_calculations"]
        self.assertEqual(len(additional), 1)
        freq = additional[0]
        self.assertEqual(freq["type"], "freq")
        self.assertEqual(freq["freq_result"]["n_imag"], 0)
        self.assertAlmostEqual(freq["freq_result"]["zpe_hartree"], 0.024131)
        self.assertNotIn("imag_freq_cm1", freq["freq_result"])
        # No sp data → no sp calc
        self.assertEqual([c["type"] for c in additional], ["freq"])

    def test_opt_plus_sp_record_yields_one_sp_additional(self):
        """3. opt+sp record produces one sp additional calculation."""
        record = _fake_record()
        record["sp_energy_hartree"] = -154.987
        _, _, payload = self._submit(output_doc=_fake_output_doc(), record=record)
        additional = payload["additional_calculations"]
        self.assertEqual(len(additional), 1)
        sp = additional[0]
        self.assertEqual(sp["type"], "sp")
        self.assertAlmostEqual(sp["sp_result"]["electronic_energy_hartree"], -154.987)

    def test_opt_freq_sp_record_yields_two_additional_calculations(self):
        """4. opt+freq+sp record produces two additional calculations."""
        record = _fake_record()
        record["freq_n_imag"] = 1
        record["imag_freq_cm1"] = -512.3
        record["zpe_hartree"] = 0.0399
        record["sp_energy_hartree"] = -155.111
        _, _, payload = self._submit(output_doc=_fake_output_doc(), record=record)
        additional = payload["additional_calculations"]
        self.assertEqual([c["type"] for c in additional], ["freq", "sp"])
        freq, sp = additional
        self.assertEqual(freq["freq_result"]["n_imag"], 1)
        self.assertAlmostEqual(freq["freq_result"]["imag_freq_cm1"], -512.3)
        self.assertAlmostEqual(freq["freq_result"]["zpe_hartree"], 0.0399)
        self.assertAlmostEqual(sp["sp_result"]["electronic_energy_hartree"], -155.111)

    def test_distinct_levels_preserved_per_calculation(self):
        """5. freq/sp use their own level-of-theory when output_doc has them."""
        doc = _fake_output_doc()
        doc["freq_level"] = {"method": "wb97xd", "basis": "6-31g*", "software": "gaussian"}
        doc["sp_level"] = {"method": "ccsd(t)-f12a", "basis": "cc-pvtz-f12", "software": "molpro"}
        record = _fake_record()
        record["freq_n_imag"] = 0
        record["zpe_hartree"] = 0.024131
        record["sp_energy_hartree"] = -154.987
        # Job-type-keyed ess_versions: freq from gaussian, sp from molpro
        record["ess_versions"] = {
            "opt": "Gaussian 16, Revision A.03",
            "freq": "Gaussian 16, Revision A.03",
            "sp": "Molpro 2022.3",
        }
        _, _, payload = self._submit(output_doc=doc, record=record)
        primary = payload["calculation"]
        freq, sp = payload["additional_calculations"]
        # opt uses opt_level + ess_versions['opt']
        self.assertEqual(primary["level_of_theory"]["method"], "wb97xd")
        self.assertEqual(primary["level_of_theory"]["basis"], "def2-tzvp")
        self.assertEqual(primary["software_release"]["version"], "Gaussian 16, Revision A.03")
        # freq uses freq_level (distinct basis) + ess_versions['freq']
        self.assertEqual(freq["level_of_theory"]["method"], "wb97xd")
        self.assertEqual(freq["level_of_theory"]["basis"], "6-31g*")
        self.assertEqual(freq["software_release"]["name"], "gaussian")
        self.assertEqual(freq["software_release"]["version"], "Gaussian 16, Revision A.03")
        # sp uses sp_level (different method+software) + ess_versions['sp']
        self.assertEqual(sp["level_of_theory"]["method"], "ccsd(t)-f12a")
        self.assertEqual(sp["level_of_theory"]["basis"], "cc-pvtz-f12")
        self.assertEqual(sp["software_release"]["name"], "molpro")
        self.assertEqual(sp["software_release"]["version"], "Molpro 2022.3")

    def test_freq_sp_levels_fall_back_to_opt_level_when_missing(self):
        """Option B: missing freq_level/sp_level falls back to opt_level."""
        doc = _fake_output_doc()
        # freq_level / sp_level absent (the common ARC case)
        record = _fake_record()
        record["freq_n_imag"] = 0
        record["sp_energy_hartree"] = -154.5
        _, _, payload = self._submit(output_doc=doc, record=record)
        freq, sp = payload["additional_calculations"]
        self.assertEqual(freq["level_of_theory"]["method"], "wb97xd")
        self.assertEqual(freq["level_of_theory"]["basis"], "def2-tzvp")
        self.assertEqual(sp["level_of_theory"]["method"], "wb97xd")
        # ess_versions has only 'opt' → both freq and sp fall back to that
        self.assertEqual(freq["software_release"]["version"], "Gaussian 16, Revision A.03")
        self.assertEqual(sp["software_release"]["version"], "Gaussian 16, Revision A.03")

    def test_ess_versions_uses_job_type_key_not_software_name(self):
        """7. ess_versions lookup must use job-type keys ('opt'/'freq'/'sp')."""
        # Record with job-type-keyed ess_versions (matches arc/output.py).
        # If the adapter were still using software-name keys, version would be missing.
        record = _fake_record()
        record["freq_n_imag"] = 0
        record["sp_energy_hartree"] = -154.5
        record["ess_versions"] = {
            "opt": "OPT_VER",
            "freq": "FREQ_VER",
            "sp": "SP_VER",
        }
        _, _, payload = self._submit(output_doc=_fake_output_doc(), record=record)
        self.assertEqual(payload["calculation"]["software_release"]["version"], "OPT_VER")
        freq, sp = payload["additional_calculations"]
        self.assertEqual(freq["software_release"]["version"], "FREQ_VER")
        self.assertEqual(sp["software_release"]["version"], "SP_VER")

    def test_idempotency_key_changes_when_freq_sp_added(self):
        """Adding freq/sp to a previously opt-only record must change the idempotency key."""
        record_opt_only = _fake_record()
        record_full = _fake_record()
        record_full["freq_n_imag"] = 0
        record_full["zpe_hartree"] = 0.024
        record_full["sp_energy_hartree"] = -154.5
        o1, _, _ = self._submit(output_doc=_fake_output_doc(), record=record_opt_only)
        o2, _, _ = self._submit(output_doc=_fake_output_doc(), record=record_full)
        self.assertNotEqual(o1.idempotency_key, o2.idempotency_key)

    # ------------------------------------------------------------------
    # SP origin metadata: reused-from-opt vs independently executed.
    # ------------------------------------------------------------------

    def test_sp_distinct_level_has_no_reused_marker(self):
        """sp_level differing from opt_level → no tckdb_origin marker on SP."""
        doc = _fake_output_doc()
        doc["sp_level"] = {"method": "ccsd(t)-f12a", "basis": "cc-pvtz-f12", "software": "molpro"}
        record = _fake_record()
        record["sp_energy_hartree"] = -154.987
        record["ess_versions"] = {"opt": "Gaussian 16, Revision A.03", "sp": "Molpro 2022.3"}
        _, _, payload = self._submit(output_doc=doc, record=record)
        sp = next(c for c in payload["additional_calculations"] if c["type"] == "sp")
        self.assertNotIn(
            "parameters_json", sp,
            msg="distinct sp_level must not produce a tckdb_origin marker",
        )

    def test_sp_equal_level_marks_reused(self):
        """sp_level structurally equal to opt_level → SP carries tckdb_origin=reused_result."""
        doc = _fake_output_doc()
        # Same dict contents as opt_level — explicit duplication, not just absence.
        doc["sp_level"] = dict(doc["opt_level"])
        record = _fake_record()
        record["sp_energy_hartree"] = -154.123
        _, _, payload = self._submit(output_doc=doc, record=record)
        sp = next(c for c in payload["additional_calculations"] if c["type"] == "sp")
        origin = sp["parameters_json"]["tckdb_origin"]
        self.assertEqual(origin["origin_kind"], "reused_result")
        self.assertFalse(origin["independent_ess_job"])
        self.assertEqual(origin["reused_from"]["calculation_type"], "opt")
        self.assertEqual(origin["producer"], "ARC")

    def test_sp_missing_level_falls_back_and_marks_reused(self):
        """Missing sp_level (the common ARC case) is treated as equal-to-opt → reused."""
        doc = _fake_output_doc()  # no sp_level key
        record = _fake_record()
        record["sp_energy_hartree"] = -154.5
        _, _, payload = self._submit(output_doc=doc, record=record)
        sp = next(c for c in payload["additional_calculations"] if c["type"] == "sp")
        origin = sp["parameters_json"]["tckdb_origin"]
        self.assertEqual(origin["origin_kind"], "reused_result")
        self.assertEqual(origin["reused_from"]["calculation_type"], "opt")
        self.assertFalse(origin["independent_ess_job"])

    def test_opt_freq_reused_sp_keeps_per_calc_invariants(self):
        """opt + freq + reused SP: primary unchanged, each row has only its own result."""
        doc = _fake_output_doc()  # only opt_level → sp falls back, freq falls back
        record = _fake_record()
        record["freq_n_imag"] = 0
        record["zpe_hartree"] = 0.024131
        record["sp_energy_hartree"] = -154.5
        _, _, payload = self._submit(output_doc=doc, record=record)

        # Primary opt unchanged: still type=opt, has opt_result, no parameters_json.
        primary = payload["calculation"]
        self.assertEqual(primary["type"], "opt")
        self.assertIn("opt_result", primary)
        self.assertNotIn("freq_result", primary)
        self.assertNotIn("sp_result", primary)
        self.assertNotIn("parameters_json", primary)

        freq = next(c for c in payload["additional_calculations"] if c["type"] == "freq")
        sp = next(c for c in payload["additional_calculations"] if c["type"] == "sp")

        # freq has only freq_result and no origin marker.
        self.assertIn("freq_result", freq)
        self.assertNotIn("sp_result", freq)
        self.assertNotIn("opt_result", freq)
        self.assertNotIn("parameters_json", freq)

        # sp has only sp_result and the reused-result marker.
        self.assertIn("sp_result", sp)
        self.assertNotIn("freq_result", sp)
        self.assertNotIn("opt_result", sp)
        self.assertEqual(
            sp["parameters_json"]["tckdb_origin"]["origin_kind"], "reused_result"
        )

    def test_no_sp_energy_no_sp_additional_calculation(self):
        """No sp_energy_hartree on the record → no SP row, regardless of sp_level."""
        doc = _fake_output_doc()
        doc["sp_level"] = {"method": "ccsd(t)-f12a", "basis": "cc-pvtz-f12", "software": "molpro"}
        record = _fake_record()  # no sp_energy_hartree
        _, _, payload = self._submit(output_doc=doc, record=record)
        types = [c["type"] for c in payload.get("additional_calculations", [])]
        self.assertNotIn("sp", types)


class TestMalformedAdditionalCalcFields(unittest.TestCase):
    """6. Malformed optional freq/sp fields skip the calc with a warning."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cfg = TCKDBConfig(
            enabled=True,
            base_url="http://localhost:8000/api/v1",
            payload_dir=self.tmp,
            api_key_env="X_TCKDB_API_KEY",
            project_label="proj-A",
        )

    def _record_with_opt_freq_sp(self):
        record = _fake_record()
        record["freq_n_imag"] = 0
        record["zpe_hartree"] = 0.0399
        record["sp_energy_hartree"] = -154.5
        return record

    def _submit(self, record):
        client = _StubClient(response=_StubResponse({"id": 1}))
        adapter = TCKDBAdapter(self.cfg, client_factory=lambda c, k: client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_from_output(
                output_doc=_fake_output_doc(), species_record=record
            )
        return outcome, json.loads(outcome.payload_path.read_text())

    def test_malformed_freq_field_skips_freq_but_keeps_opt_and_sp(self):
        """Malformed freq value drops the freq calc only; opt + sp survive."""
        record = self._record_with_opt_freq_sp()
        record["zpe_hartree"] = "not_a_number"

        with self.assertLogs("arc", level="WARNING") as logs:
            outcome, payload = self._submit(record)

        self.assertEqual(outcome.status, "uploaded")
        # Primary opt calculation present and intact
        self.assertEqual(payload["calculation"]["type"], "opt")
        self.assertIn("opt_result", payload["calculation"])
        # freq dropped, sp preserved
        additional = payload.get("additional_calculations", [])
        types = [c["type"] for c in additional]
        self.assertNotIn("freq", types)
        self.assertIn("sp", types)
        # Warning names the calc type and the offending field
        self.assertTrue(
            any("freq" in m and "zpe_hartree" in m for m in logs.output),
            msg=f"expected warning naming freq/zpe_hartree; got {logs.output}",
        )

    def test_malformed_sp_field_skips_sp_but_keeps_opt_and_freq(self):
        """Malformed sp value drops the sp calc only; opt + freq survive."""
        record = self._record_with_opt_freq_sp()
        record["sp_energy_hartree"] = "not-a-float"

        with self.assertLogs("arc", level="WARNING") as logs:
            outcome, payload = self._submit(record)

        self.assertEqual(outcome.status, "uploaded")
        self.assertEqual(payload["calculation"]["type"], "opt")
        self.assertIn("opt_result", payload["calculation"])
        additional = payload.get("additional_calculations", [])
        types = [c["type"] for c in additional]
        self.assertIn("freq", types)
        self.assertNotIn("sp", types)
        self.assertTrue(
            any("sp" in m and "sp_energy_hartree" in m for m in logs.output),
            msg=f"expected warning naming sp/sp_energy_hartree; got {logs.output}",
        )

    def test_both_malformed_yields_two_warnings_and_no_additional_calcs(self):
        """Malformed freq AND sp: two warnings, payload has only opt."""
        record = self._record_with_opt_freq_sp()
        record["zpe_hartree"] = "not_a_number"
        record["sp_energy_hartree"] = "not-a-float"

        with self.assertLogs("arc", level="WARNING") as logs:
            outcome, payload = self._submit(record)

        self.assertEqual(outcome.status, "uploaded")
        self.assertEqual(payload["calculation"]["type"], "opt")
        # Either no key, or empty list — both are "no additional calcs"
        self.assertEqual(payload.get("additional_calculations", []), [])

        freq_warnings = [m for m in logs.output if "freq" in m and "zpe_hartree" in m]
        sp_warnings = [m for m in logs.output if "sp" in m and "sp_energy_hartree" in m]
        self.assertEqual(
            len(freq_warnings), 1,
            msg=f"expected exactly one freq warning; got {logs.output}",
        )
        self.assertEqual(
            len(sp_warnings), 1,
            msg=f"expected exactly one sp warning; got {logs.output}",
        )


# ---------------------------------------------------------------------------
# Computed-species bundle (POST /uploads/computed-species)
# ---------------------------------------------------------------------------


_FORBIDDEN_BUNDLE_ID_FIELDS = {
    "existing_calculation_id",
    "existing_conformer_id",
    "existing_conformer_observation_id",
    "existing_species_entry_id",
    "source_calculation_id",
    "source_conformer_observation_id",
}


def _walk_for_keys(value, keys):
    """Yield every (path, key) pair in a JSON-like value matching ``keys``."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k in keys:
                yield k
            yield from _walk_for_keys(v, keys)
    elif isinstance(value, list):
        for v in value:
            yield from _walk_for_keys(v, keys)


def _full_record():
    """A species record with opt+freq+sp results plus thermo populated."""
    record = _fake_record()
    record["freq_n_imag"] = 0
    record["zpe_hartree"] = 0.024131
    record["sp_energy_hartree"] = -154.987
    record["thermo"] = {
        "h298_kj_mol": -235.1,
        "s298_j_mol_k": 282.6,
        "tmin_k": 100.0,
        "tmax_k": 5000.0,
        "nasa_low": {
            "tmin_k": 100.0,
            "tmax_k": 1000.0,
            "coeffs": [4.0, -1e-3, 2e-6, -1e-9, 4e-13, -29000.0, 1.0],
        },
        "nasa_high": {
            "tmin_k": 1000.0,
            "tmax_k": 5000.0,
            "coeffs": [3.5, 1e-3, -2e-7, 1e-11, -3e-15, -28500.0, 5.0],
        },
        "cp_data": [
            {"temperature_k": 300.0, "cp_j_mol_k": 33.6},
            {"temperature_k": 400.0, "cp_j_mol_k": 35.2},
            {"temperature_k": 500.0, "cp_j_mol_k": 37.0},
        ],
    }
    return record


class TestComputedSpeciesBundle(unittest.TestCase):
    """Producer-side tests for the /uploads/computed-species bundle path."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-bundle-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cfg = TCKDBConfig(
            enabled=True,
            base_url="http://localhost:8000/api/v1",
            payload_dir=self.tmp,
            api_key_env="X_TCKDB_API_KEY",
            project_label="proj-A",
            upload_mode="computed_species",
        )

    def _adapter(self, client, *, project_directory=None, cfg=None):
        return TCKDBAdapter(
            cfg or self.cfg,
            project_directory=project_directory,
            client_factory=lambda c, k: client,
        )

    def _submit(self, *, output_doc=None, record=None, client=None,
                project_directory=None, cfg=None):
        client = client or _StubClient(response=_StubResponse({
            "species_entry_id": 7,
            "conformers": [{
                "key": "conf0",
                "conformer_group_id": 3,
                "conformer_observation_id": 11,
                "primary_calculation": {"key": "opt", "calculation_id": 100, "type": "opt", "role": "primary"},
                "additional_calculations": [
                    {"key": "freq", "calculation_id": 101, "type": "freq", "role": "additional"},
                    {"key": "sp", "calculation_id": 102, "type": "sp", "role": "additional"},
                ],
            }],
            "thermo": {"thermo_id": 9},
        }))
        adapter = self._adapter(client, project_directory=project_directory, cfg=cfg)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_computed_species_from_output(
                output_doc=output_doc or _fake_output_doc(),
                species_record=record or _full_record(),
            )
        return outcome, client, json.loads(outcome.payload_path.read_text())

    # ---------------- 1: payload contains species_entry + one conformer
    def test_payload_contains_species_entry_and_one_conformer(self):
        _, _, payload = self._submit()
        self.assertIn("species_entry", payload)
        self.assertEqual(payload["species_entry"]["smiles"], "CCO")
        self.assertEqual(payload["species_entry"]["charge"], 0)
        self.assertEqual(payload["species_entry"]["multiplicity"], 1)
        self.assertEqual(len(payload["conformers"]), 1)
        self.assertEqual(payload["conformers"][0]["key"], "conf0")

    # ---------------- 2: primary opt maps correctly
    def test_primary_opt_calculation_maps_correctly(self):
        _, _, payload = self._submit()
        primary = payload["conformers"][0]["primary_calculation"]
        self.assertEqual(primary["key"], "opt")
        self.assertEqual(primary["type"], "opt")
        self.assertEqual(primary["quality"], "raw")
        self.assertEqual(primary["level_of_theory"]["method"], "wb97xd")
        self.assertEqual(primary["software_release"]["name"], "gaussian")
        self.assertIn("opt_result", primary)
        self.assertEqual(primary["opt_result"]["n_steps"], 12)

    # ---------------- 3: freq+sp included when fields exist
    def test_freq_and_sp_included_when_fields_exist(self):
        _, _, payload = self._submit()
        keys = [c["key"] for c in payload["conformers"][0]["additional_calculations"]]
        self.assertEqual(keys, ["freq", "sp"])
        freq = payload["conformers"][0]["additional_calculations"][0]
        sp = payload["conformers"][0]["additional_calculations"][1]
        self.assertEqual(freq["freq_result"]["n_imag"], 0)
        self.assertAlmostEqual(freq["freq_result"]["zpe_hartree"], 0.024131)
        self.assertAlmostEqual(sp["sp_result"]["electronic_energy_hartree"], -154.987)

    # ---------------- 4: dependencies point to opt by local key
    def test_dependencies_point_to_opt_by_local_key(self):
        _, _, payload = self._submit()
        freq, sp = payload["conformers"][0]["additional_calculations"]
        self.assertEqual(
            freq["depends_on"],
            [{"parent_calculation_key": "opt", "role": "freq_on"}],
        )
        self.assertEqual(
            sp["depends_on"],
            [{"parent_calculation_key": "opt", "role": "single_point_on"}],
        )
        # Primary opt has no dependencies in the bundle namespace.
        self.assertNotIn("depends_on", payload["conformers"][0]["primary_calculation"])

    # ---------------- 5: missing freq/sp omits calc + dep
    def test_missing_freq_sp_omits_calc_and_dep(self):
        record = _fake_record()  # no freq_*, no sp_*
        _, _, payload = self._submit(record=record)
        self.assertEqual(payload["conformers"][0]["additional_calculations"], [])
        # Thermo also has no freq/sp source links, since neither calc was included.
        self.assertNotIn("thermo", payload)

    # ---------------- 6: thermo scalar fields
    def test_thermo_scalar_fields_map_correctly(self):
        _, _, payload = self._submit()
        thermo = payload["thermo"]
        self.assertAlmostEqual(thermo["h298_kj_mol"], -235.1)
        self.assertAlmostEqual(thermo["s298_j_mol_k"], 282.6)
        self.assertAlmostEqual(thermo["tmin_k"], 100.0)
        self.assertAlmostEqual(thermo["tmax_k"], 5000.0)

    # ---------------- 7: NASA coefficients
    def test_nasa_coefficients_map_correctly(self):
        _, _, payload = self._submit()
        nasa = payload["thermo"]["nasa"]
        self.assertAlmostEqual(nasa["t_low"], 100.0)
        self.assertAlmostEqual(nasa["t_mid"], 1000.0)
        self.assertAlmostEqual(nasa["t_high"], 5000.0)
        self.assertAlmostEqual(nasa["a1"], 4.0)
        self.assertAlmostEqual(nasa["a7"], 1.0)
        self.assertAlmostEqual(nasa["b1"], 3.5)
        self.assertAlmostEqual(nasa["b7"], 5.0)

    # ---------------- 8: malformed NASA skips block, keeps scalar thermo
    def test_malformed_nasa_skips_block_keeps_scalar(self):
        record = _full_record()
        # Drop one coefficient → not 7 → NASA block must be skipped.
        record["thermo"]["nasa_low"]["coeffs"] = [1.0] * 6
        with self.assertLogs("arc", level="WARNING") as logs:
            _, _, payload = self._submit(record=record)
        self.assertNotIn("nasa", payload["thermo"])
        # Scalar thermo retained.
        self.assertAlmostEqual(payload["thermo"]["h298_kj_mol"], -235.1)
        self.assertAlmostEqual(payload["thermo"]["s298_j_mol_k"], 282.6)
        # Cp points retained.
        self.assertEqual(len(payload["thermo"]["points"]), 3)
        self.assertTrue(any("NASA block skipped" in m for m in logs.output))

    # ---------------- 9: Cp points
    def test_cp_points_map_correctly(self):
        _, _, payload = self._submit()
        points = payload["thermo"]["points"]
        self.assertEqual([p["temperature_k"] for p in points], [300.0, 400.0, 500.0])
        self.assertAlmostEqual(points[0]["cp_j_mol_k"], 33.6)

    # ---------------- 10: thermo source links use local keys
    def test_thermo_source_calculations_use_local_keys(self):
        _, _, payload = self._submit()
        sources = payload["thermo"]["source_calculations"]
        self.assertEqual(
            sources,
            [
                {"calculation_key": "freq", "role": "freq"},
                {"calculation_key": "sp", "role": "sp"},
            ],
        )

    # ---------------- 11: payload has no DB ids anywhere
    def test_payload_has_no_existing_or_source_calculation_id(self):
        _, _, payload = self._submit()
        forbidden_hits = list(_walk_for_keys(payload, _FORBIDDEN_BUNDLE_ID_FIELDS))
        self.assertEqual(forbidden_hits, [], msg=f"forbidden DB id keys present: {forbidden_hits}")

    # ---------------- 12: artifacts attach under correct calc when enabled
    def test_artifacts_attach_under_correct_calc_when_enabled(self):
        # Create a project dir with three real log files, configure artifacts on.
        proj = pathlib.Path(self.tmp) / "project"
        proj.mkdir()
        for name in ("opt.log", "freq.log", "sp.log"):
            (proj / name).write_bytes(
                # Fake but valid Gaussian header so any future signature check passes;
                # the producer doesn't validate, but real fakes are cheaper than mocks.
                b" Entering Gaussian System, Link 0\n" + b"x" * 256
            )
        record = _full_record()
        record["opt_log"] = "opt.log"
        record["freq_log"] = "freq.log"
        record["sp_log"] = "sp.log"
        cfg_with_art = TCKDBConfig(
            enabled=True,
            base_url=self.cfg.base_url,
            payload_dir=str(proj / "tckdb_payloads"),
            api_key_env=self.cfg.api_key_env,
            project_label=self.cfg.project_label,
            upload_mode="computed_species",
            artifacts=TCKDBArtifactConfig(upload=True, kinds=("output_log",), max_size_mb=50),
        )
        _, _, payload = self._submit(
            record=record,
            project_directory=str(proj),
            cfg=cfg_with_art,
        )
        primary = payload["conformers"][0]["primary_calculation"]
        freq, sp = payload["conformers"][0]["additional_calculations"]
        self.assertEqual(len(primary["artifacts"]), 1)
        self.assertEqual(primary["artifacts"][0]["kind"], "output_log")
        self.assertEqual(primary["artifacts"][0]["filename"], "opt.log")
        self.assertEqual(len(freq["artifacts"]), 1)
        self.assertEqual(freq["artifacts"][0]["filename"], "freq.log")
        self.assertEqual(len(sp["artifacts"]), 1)
        self.assertEqual(sp["artifacts"][0]["filename"], "sp.log")

    # ---------------- 13: artifacts disabled = no artifact lists emitted
    def test_artifact_disabled_produces_no_artifacts(self):
        _, _, payload = self._submit()  # default cfg has artifacts.upload=False
        primary = payload["conformers"][0]["primary_calculation"]
        self.assertNotIn("artifacts", primary)
        for calc in payload["conformers"][0]["additional_calculations"]:
            self.assertNotIn("artifacts", calc)

    # ---------------- 12b: input-deck artifacts inline alongside output_log
    def test_input_artifacts_inline_when_kind_enabled_and_path_recorded(self):
        """``<calc>_input`` from output.yml drives the inline ``input`` artifact."""
        proj = pathlib.Path(self.tmp) / "project_inputs"
        proj.mkdir()
        # Produce all six artifact files: opt log + deck, freq log + deck, sp log + deck.
        # output_log files need a fake ESS header so any future signature check
        # would pass; deck contents are arbitrary text.
        for name in ("opt.log", "freq.log", "sp.log"):
            (proj / name).write_bytes(
                b" Entering Gaussian System, Link 0\n" + b"x" * 256
            )
        for name in ("opt.gjf", "freq.gjf", "sp.gjf"):
            (proj / name).write_text("# opt freq=hpmodes wb97xd/def2tzvp\n")
        record = _full_record()
        record["opt_log"] = "opt.log"
        record["freq_log"] = "freq.log"
        record["sp_log"] = "sp.log"
        # The new schema-extension keys from arc/output.py.
        record["opt_input"] = "opt.gjf"
        record["freq_input"] = "freq.gjf"
        record["sp_input"] = "sp.gjf"
        cfg = TCKDBConfig(
            enabled=True,
            base_url=self.cfg.base_url,
            payload_dir=str(proj / "tckdb_payloads"),
            api_key_env=self.cfg.api_key_env,
            project_label=self.cfg.project_label,
            upload_mode="computed_species",
            artifacts=TCKDBArtifactConfig(
                upload=True, kinds=("output_log", "input"), max_size_mb=50,
            ),
        )
        _, _, payload = self._submit(record=record, project_directory=str(proj), cfg=cfg)
        primary = payload["conformers"][0]["primary_calculation"]
        freq, sp = payload["conformers"][0]["additional_calculations"]
        # Each calc should now have BOTH kinds inlined.
        for calc, expected_files in (
            (primary, ("opt.log", "opt.gjf")),
            (freq,    ("freq.log", "freq.gjf")),
            (sp,      ("sp.log", "sp.gjf")),
        ):
            self.assertEqual(len(calc["artifacts"]), 2,
                             msg=f"{calc['key']}: expected log+input, got {calc['artifacts']}")
            kinds = [a["kind"] for a in calc["artifacts"]]
            self.assertEqual(sorted(kinds), ["input", "output_log"])
            filenames = {a["filename"] for a in calc["artifacts"]}
            self.assertEqual(filenames, set(expected_files))

    def test_input_artifact_omitted_when_kind_not_configured(self):
        """``input`` not in config.artifacts.kinds → only output_log emitted, even if path recorded."""
        proj = pathlib.Path(self.tmp) / "project_no_input_kind"
        proj.mkdir()
        (proj / "opt.log").write_bytes(b" Entering Gaussian System, Link 0\n" + b"x" * 256)
        (proj / "opt.gjf").write_text("# opt\n")
        record = _full_record()
        record["opt_log"] = "opt.log"
        record["opt_input"] = "opt.gjf"   # path is set, but kind isn't enabled
        record.pop("freq_log", None); record.pop("sp_log", None)
        record["freq_n_imag"] = None      # drop freq/sp so we only test the opt calc
        record["zpe_hartree"] = None
        record.pop("sp_energy_hartree", None)
        record.pop("thermo", None)        # avoid thermo's source-link assertions
        cfg = TCKDBConfig(
            enabled=True,
            base_url=self.cfg.base_url,
            payload_dir=str(proj / "tckdb_payloads"),
            api_key_env=self.cfg.api_key_env,
            project_label=self.cfg.project_label,
            upload_mode="computed_species",
            artifacts=TCKDBArtifactConfig(
                upload=True, kinds=("output_log",), max_size_mb=50,
            ),
        )
        _, _, payload = self._submit(record=record, project_directory=str(proj), cfg=cfg)
        primary = payload["conformers"][0]["primary_calculation"]
        self.assertEqual(len(primary["artifacts"]), 1)
        self.assertEqual(primary["artifacts"][0]["kind"], "output_log")

    def test_input_artifact_omitted_when_path_field_null(self):
        """``input`` enabled but record has no ``<calc>_input`` → just output_log emitted."""
        proj = pathlib.Path(self.tmp) / "project_input_null"
        proj.mkdir()
        (proj / "opt.log").write_bytes(b" Entering Gaussian System, Link 0\n" + b"x" * 256)
        record = _full_record()
        record["opt_log"] = "opt.log"
        record["opt_input"] = None        # explicitly None — file wasn't kept
        record.pop("freq_log", None); record.pop("sp_log", None)
        record["freq_n_imag"] = None
        record["zpe_hartree"] = None
        record.pop("sp_energy_hartree", None)
        record.pop("thermo", None)
        cfg = TCKDBConfig(
            enabled=True,
            base_url=self.cfg.base_url,
            payload_dir=str(proj / "tckdb_payloads"),
            api_key_env=self.cfg.api_key_env,
            project_label=self.cfg.project_label,
            upload_mode="computed_species",
            artifacts=TCKDBArtifactConfig(
                upload=True, kinds=("output_log", "input"), max_size_mb=50,
            ),
        )
        _, _, payload = self._submit(record=record, project_directory=str(proj), cfg=cfg)
        primary = payload["conformers"][0]["primary_calculation"]
        self.assertEqual(len(primary["artifacts"]), 1)
        self.assertEqual(primary["artifacts"][0]["kind"], "output_log")

    # ---------------- 14/15: sidecar contract
    def test_sidecar_payload_kind_and_endpoint(self):
        outcome, _, _ = self._submit()
        sc = json.loads(outcome.sidecar_path.read_text())
        self.assertEqual(sc["payload_kind"], "computed_species")
        self.assertEqual(sc["endpoint"], "/uploads/computed-species")
        self.assertEqual(sc["bundle_format_version"], "0")
        # Sidecar must live under the dedicated computed_species subdir
        # so replay can sweep it without colliding with conformer sidecars.
        self.assertIn("computed_species", str(outcome.sidecar_path))

    # ---------------- 16: stable idempotency key
    def test_idempotency_key_stable_for_identical_payload(self):
        out1, _, _ = self._submit()
        out2, _, _ = self._submit()
        self.assertEqual(out1.idempotency_key, out2.idempotency_key)
        # And distinct from a payload-altering change.
        modified_record = _full_record()
        modified_record["sp_energy_hartree"] = -200.0
        out3, _, _ = self._submit(record=modified_record)
        self.assertNotEqual(out1.idempotency_key, out3.idempotency_key)

    # ---------------- 17: payload written before live upload
    def test_payload_written_before_live_upload(self):
        client = _StubClient(raise_exc=RuntimeError("network down"))
        adapter = self._adapter(client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_computed_species_from_output(
                output_doc=_fake_output_doc(),
                species_record=_full_record(),
            )
        self.assertEqual(outcome.status, "failed")
        self.assertTrue(outcome.payload_path.exists(),
                        "payload must hit disk before any network call")
        # Sidecar exists and records the failure.
        self.assertTrue(outcome.sidecar_path.exists())
        sc = json.loads(outcome.sidecar_path.read_text())
        self.assertEqual(sc["status"], "failed")
        self.assertIn("network down", sc["last_error"])

    # ---------------- 18: live upload success marks sidecar uploaded
    def test_live_upload_success_marks_sidecar_uploaded(self):
        outcome, client, _ = self._submit()
        self.assertEqual(outcome.status, "uploaded")
        sc = json.loads(outcome.sidecar_path.read_text())
        self.assertEqual(sc["status"], "uploaded")
        self.assertEqual(sc["response_status_code"], 201)
        # And the call really went to the bundle endpoint.
        self.assertEqual(client.calls[0]["path"], "/uploads/computed-species")
        # Idempotency-Key was sent.
        self.assertEqual(client.calls[0]["idempotency_key"], outcome.idempotency_key)

    # ---------------- 19: live upload failure preserves payload, marks failed
    def test_live_upload_failure_marks_failed_and_preserves_payload(self):
        client = _StubClient(raise_exc=RuntimeError("HTTP 503"))
        adapter = self._adapter(client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_computed_species_from_output(
                output_doc=_fake_output_doc(),
                species_record=_full_record(),
            )
        # Payload still on disk verbatim — replay-ready.
        self.assertTrue(outcome.payload_path.exists())
        replay_payload = json.loads(outcome.payload_path.read_text())
        self.assertIn("species_entry", replay_payload)
        self.assertIn("conformers", replay_payload)
        self.assertEqual(outcome.status, "failed")
        self.assertIn("HTTP 503", outcome.error)

    # ---------------- 20: producer reads only output_doc + species_record (mapping arg shape)
    def test_producer_consumes_only_output_doc_and_record_mappings(self):
        # Pass plain dicts (no ARC class instances). If the adapter ever
        # reaches into ARC live objects, this test would surface that
        # via a missing-attribute crash.
        plain_doc = dict(_fake_output_doc())
        plain_record = dict(_full_record())
        outcome, _, _ = self._submit(output_doc=plain_doc, record=plain_record)
        self.assertEqual(outcome.status, "uploaded")


if __name__ == "__main__":
    unittest.main()
