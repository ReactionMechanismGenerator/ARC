#!/usr/bin/env python3
# encoding: utf-8

"""Unit tests for arc.tckdb.payload_writer."""

import json
import os
import shutil
import tempfile
import unittest

from arc.tckdb.payload_writer import (
    ArtifactSidecarMetadata,
    PayloadWriter,
    SidecarMetadata,
)


class TestPayloadWriter(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-writer-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.writer = PayloadWriter(self.tmp)

    def test_write_creates_payload_and_sidecar(self):
        result = self.writer.write(
            label="ethanol.conf0",
            payload={"species_entry": {"smiles": "CCO"}},
            endpoint="/uploads/conformers",
            idempotency_key="arc:proj:ethanol:conf0:conformer_calculation:abc1234567890def",
        )
        self.assertTrue(result.payload_path.exists())
        self.assertTrue(result.sidecar_path.exists())
        with open(result.payload_path) as fh:
            payload = json.load(fh)
        self.assertEqual(payload["species_entry"]["smiles"], "CCO")
        with open(result.sidecar_path) as fh:
            sc = json.load(fh)
        self.assertEqual(sc["status"], "pending")
        self.assertEqual(sc["endpoint"], "/uploads/conformers")
        self.assertIsNone(sc["uploaded_at"])
        self.assertIn("created_at", sc)
        # Replay-tool contract: every conformer sidecar must declare its
        # payload_kind (for dispatch) and bundle_format_version (for the
        # version gate). Missing either makes the sidecar unreplayable.
        self.assertEqual(sc["payload_kind"], "conformer_calculation")
        self.assertEqual(sc["bundle_format_version"], "0")

    def test_write_sanitizes_label(self):
        result = self.writer.write(
            label="ethanol/conf 0?!",
            payload={"x": 1},
            endpoint="/uploads/conformers",
            idempotency_key="arc:proj:ethanol:conf0:conformer_calculation:abc1234567890def",
        )
        # Final filename has no slashes/spaces/punctuation
        name = result.payload_path.name
        self.assertNotIn("/", name)
        self.assertNotIn(" ", name)
        self.assertNotIn("?", name)
        self.assertTrue(name.endswith(".payload.json"))

    def test_update_sidecar_in_place(self):
        result = self.writer.write(
            label="ethanol",
            payload={"x": 1},
            endpoint="/uploads/conformers",
            idempotency_key="arc:proj:ethanol:conf0:conformer_calculation:abc1234567890def",
        )
        sc = result.sidecar
        sc.status = "uploaded"
        sc.uploaded_at = "2026-04-26T12:00:00Z"
        sc.response_status_code = 200
        sc.response_body = {"id": 7}
        self.writer.update_sidecar(result.sidecar_path, sc)
        with open(result.sidecar_path) as fh:
            on_disk = json.load(fh)
        self.assertEqual(on_disk["status"], "uploaded")
        self.assertEqual(on_disk["response_status_code"], 200)
        self.assertEqual(on_disk["response_body"], {"id": 7})

    def test_payload_unchanged_after_sidecar_update(self):
        result = self.writer.write(
            label="ethanol",
            payload={"x": 42},
            endpoint="/uploads/conformers",
            idempotency_key="arc:proj:ethanol:conf0:conformer_calculation:abc1234567890def",
        )
        before = result.payload_path.read_bytes()
        sc = result.sidecar
        sc.status = "failed"
        sc.last_error = "boom"
        self.writer.update_sidecar(result.sidecar_path, sc)
        after = result.payload_path.read_bytes()
        self.assertEqual(before, after)


class TestArtifactSidecar(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-artifact-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.writer = PayloadWriter(self.tmp)

    def _write(self, **overrides):
        defaults = dict(
            species_label="ethanol",
            calculation_id=42,
            kind="output_log",
            filename="opt.log",
            sha256="a" * 64,
            bytes_=1024,
            endpoint="/calculations/42/artifacts",
            idempotency_key="arc:proj:ethanol:artifact:42:output_log:abcdef0123456789",
            source_path="/runs/proj/calcs/Species/ethanol/opt_a0/output.log",
            base_url="http://localhost:8000/api/v1",
        )
        defaults.update(overrides)
        return self.writer.write_artifact_sidecar(**defaults)

    def test_artifact_sidecar_initialized_pending(self):
        result = self._write()
        self.assertTrue(result.sidecar_path.exists())
        on_disk = json.loads(result.sidecar_path.read_text())
        self.assertEqual(on_disk["status"], "pending")
        self.assertEqual(on_disk["calculation_id"], 42)
        self.assertEqual(on_disk["kind"], "output_log")
        self.assertEqual(on_disk["sha256"], "a" * 64)
        self.assertEqual(on_disk["bytes"], 1024)
        self.assertIn("created_at", on_disk)
        self.assertIsNone(on_disk["uploaded_at"])
        self.assertIsNone(on_disk["last_error"])
        # Replay-tool contract: artifact sidecars must declare their
        # payload_kind so the replay dispatcher can route them to the
        # artifact handler instead of the __unknown__ bucket.
        self.assertEqual(on_disk["payload_kind"], "calculation_artifact")
        self.assertEqual(on_disk["bundle_format_version"], "0")

    def test_artifact_sidecar_filename_includes_calc_and_kind(self):
        result = self._write(calculation_id=99, kind="input")
        name = result.sidecar_path.name
        self.assertIn("ethanol", name)
        self.assertIn("calc99", name)
        self.assertIn("input", name)
        self.assertTrue(name.endswith(".artifact.meta.json"))

    def test_distinct_sidecars_for_distinct_kinds(self):
        a = self._write(kind="output_log")
        b = self._write(kind="input")
        self.assertNotEqual(a.sidecar_path, b.sidecar_path)
        self.assertTrue(a.sidecar_path.exists())
        self.assertTrue(b.sidecar_path.exists())

    def test_distinct_sidecars_for_distinct_calculations(self):
        a = self._write(calculation_id=1)
        b = self._write(calculation_id=2)
        self.assertNotEqual(a.sidecar_path, b.sidecar_path)

    def test_update_artifact_sidecar_in_place(self):
        result = self._write()
        sc = result.sidecar
        sc.status = "uploaded"
        sc.uploaded_at = "2026-04-27T12:00:00Z"
        sc.response_status_code = 201
        sc.response_body = {"calculation_id": 42, "artifacts": [{"id": 7}]}
        self.writer.update_artifact_sidecar(result.sidecar_path, sc)
        on_disk = json.loads(result.sidecar_path.read_text())
        self.assertEqual(on_disk["status"], "uploaded")
        self.assertEqual(on_disk["response_status_code"], 201)
        self.assertEqual(on_disk["response_body"]["calculation_id"], 42)


if __name__ == "__main__":
    unittest.main()
