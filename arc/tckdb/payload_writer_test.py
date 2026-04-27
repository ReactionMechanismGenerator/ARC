#!/usr/bin/env python3
# encoding: utf-8

"""Unit tests for arc.tckdb.payload_writer."""

import json
import os
import shutil
import tempfile
import unittest

from arc.tckdb.payload_writer import PayloadWriter, SidecarMetadata


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


if __name__ == "__main__":
    unittest.main()
