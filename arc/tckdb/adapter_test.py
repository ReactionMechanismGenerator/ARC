#!/usr/bin/env python3
# encoding: utf-8

"""Unit tests for arc.tckdb.adapter.

These tests do not require a live TCKDB server. The TCKDBClient is
replaced by a stub via the adapter's ``client_factory`` parameter.
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from arc.tckdb.adapter import (
    CONFORMER_UPLOAD_ENDPOINT,
    PAYLOAD_KIND,
    TCKDBAdapter,
    UploadOutcome,
)
from arc.tckdb.config import TCKDBConfig


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


def _fake_species(label="ethanol", smiles="CCO", charge=0, multiplicity=1, is_ts=False):
    return SimpleNamespace(
        label=label,
        smiles=smiles,
        charge=charge,
        multiplicity=multiplicity,
        is_ts=is_ts,
        final_xyz="C 0.0 0.0 0.0\nH 1.0 0.0 0.0",
    )


def _fake_level():
    return SimpleNamespace(
        method="wb97xd",
        basis="def2-tzvp",
        auxiliary_basis=None,
        cabs=None,
        dispersion=None,
        solvation_method=None,
        solvent=None,
        software="gaussian",
        software_version="16",
    )


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
        outcome = adapter.submit_conformer(species=_fake_species(), level=_fake_level())
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
            outcome = adapter.submit_conformer(species=_fake_species(), level=_fake_level())

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
            outcome = adapter.submit_conformer(species=_fake_species(), level=_fake_level())

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
            outcome = adapter.submit_conformer(species=_fake_species(), level=_fake_level())
        sc = json.loads(outcome.sidecar_path.read_text())
        self.assertTrue(sc["idempotency_replayed"])

    def test_payload_contains_no_db_ids(self):
        """Test 9: payload must not include raw TCKDB DB IDs."""
        client = _StubClient(response=_StubResponse({"id": 1}))
        adapter = self._adapter(client)
        with mock.patch.dict(os.environ, {"X_TCKDB_API_KEY": "tck_x"}):
            outcome = adapter.submit_conformer(species=_fake_species(), level=_fake_level())
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
            outcome = adapter.submit_conformer(species=_fake_species(), level=_fake_level())
        payload = json.loads(outcome.payload_path.read_text())
        self.assertIn("species_entry", payload)
        self.assertIn("geometry", payload)
        self.assertIn("calculation", payload)
        self.assertEqual(payload["species_entry"]["smiles"], "CCO")
        self.assertEqual(payload["geometry"]["xyz_text"], "C 0.0 0.0 0.0\nH 1.0 0.0 0.0")
        self.assertEqual(payload["calculation"]["type"], "opt")
        self.assertEqual(payload["calculation"]["software_release"]["name"], "gaussian")
        self.assertEqual(payload["calculation"]["software_release"]["version"], "16")
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
        outcome = adapter.submit_conformer(species=_fake_species(), level=_fake_level())
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
                adapter.submit_conformer(species=_fake_species(), level=_fake_level())
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
            outcome = adapter.submit_conformer(species=_fake_species(), level=_fake_level())
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
            o1 = adapter.submit_conformer(species=_fake_species(), level=_fake_level())
            o2 = adapter.submit_conformer(species=_fake_species(), level=_fake_level())
            o3 = adapter.submit_conformer(
                species=_fake_species(label="methanol", smiles="CO"), level=_fake_level()
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
        outcome = adapter.submit_conformer(species=_fake_species(), level=_fake_level())
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
            adapter.submit_conformer(species=_fake_species(), level=_fake_level())


if __name__ == "__main__":
    unittest.main()
