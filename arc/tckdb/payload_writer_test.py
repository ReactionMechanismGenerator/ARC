#!/usr/bin/env python3
# encoding: utf-8

"""Unit tests for arc.tckdb.payload_writer."""

import json
import shutil
import tempfile
import unittest

from arc.tckdb.payload_writer import (
    PayloadWriter,
    SidecarMetadata,
    should_replay_sidecar,
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
        sc.public_refs = {"calculation_refs": ["calc_7"]}
        sc.request_ids = [{"operation": "upload", "request_id": "req-7", "status_code": 200}]
        sc.preflight = {"ready": True, "status_code": 200, "request_id": "req-ready"}
        self.writer.update_sidecar(result.sidecar_path, sc)
        with open(result.sidecar_path) as fh:
            on_disk = json.load(fh)
        self.assertEqual(on_disk["status"], "uploaded")
        self.assertEqual(on_disk["response_status_code"], 200)
        self.assertEqual(on_disk["response_body"], {"id": 7})
        self.assertEqual(on_disk["public_refs"], {"calculation_refs": ["calc_7"]})
        self.assertEqual(on_disk["request_ids"][0]["request_id"], "req-7")
        self.assertEqual(on_disk["preflight"]["request_id"], "req-ready")

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


    def test_write_default_is_not_partial(self):
        # Existing callers (no is_partial) must keep their pre-phase-1
        # filenames and metadata exactly. Regression guard.
        result = self.writer.write(
            label="ethanol",
            payload={"x": 1},
            endpoint="/uploads/conformers",
            idempotency_key="arc:proj:ethanol:conf0:conformer_calculation:abc1234567890def",
        )
        self.assertNotIn(".partial.", result.payload_path.name)
        self.assertNotIn(".partial.", result.sidecar_path.name)
        with open(result.sidecar_path) as fh:
            sc = json.load(fh)
        self.assertFalse(sc["is_partial"])

    def test_write_partial_marks_filenames_and_metadata(self):
        result = self.writer.write(
            label="r0_p0",
            payload={"species": []},
            endpoint="/uploads/computed-reaction",
            idempotency_key="arc:proj:rxn0:noTS:computed_reaction:abc1234567890def",
            payload_kind="computed_reaction",
            is_partial=True,
        )
        # Both filenames must carry the .partial. infix so the on-disk
        # bucket can be greppable by humans, not just by parsed metadata.
        self.assertTrue(result.payload_path.name.endswith(".partial.payload.json"))
        self.assertTrue(result.sidecar_path.name.endswith(".partial.meta.json"))
        with open(result.sidecar_path) as fh:
            sc = json.load(fh)
        self.assertTrue(sc["is_partial"])
        self.assertEqual(sc["status"], "pending")


class TestPayloadWriterArchive(unittest.TestCase):
    """Re-run archiving: prior payload+sidecar preserved under archive/."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="arc-tckdb-writer-arch-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.writer = PayloadWriter(self.tmp)

    def _archive_dir(self, result):
        return result.payload_path.parent / PayloadWriter.ARCHIVE_SUBDIR

    def test_first_write_creates_no_archive(self):
        result = self.writer.write(
            label="ethanol",
            payload={"x": 1},
            endpoint="/uploads/conformers",
            idempotency_key="key-first-run",
        )
        self.assertFalse(self._archive_dir(result).exists())

    def test_second_write_archives_prior_pair_keyed_on_uploaded_at(self):
        first = self.writer.write(
            label="ethanol",
            payload={"run": 1},
            endpoint="/uploads/conformers",
            idempotency_key="key-run-1",
        )
        # Resolve the first upload so the sidecar carries an uploaded_at
        # timestamp — the preferred archive key.
        sc = first.sidecar
        sc.status = "uploaded"
        sc.uploaded_at = "2026-07-14T09:15:30Z"
        self.writer.update_sidecar(first.sidecar_path, sc)

        second = self.writer.write(
            label="ethanol",
            payload={"run": 2},
            endpoint="/uploads/conformers",
            idempotency_key="key-run-2",
        )
        # Canonical path is unchanged and holds the NEW content.
        self.assertEqual(second.payload_path, first.payload_path)
        with open(second.payload_path) as fh:
            self.assertEqual(json.load(fh)["run"], 2)

        archive_dir = self._archive_dir(second)
        self.assertTrue(archive_dir.exists())
        # Self-describing name keyed off the prior uploaded_at (colons
        # stripped for filesystem safety).
        archived_payloads = sorted(archive_dir.glob("*.payload.json"))
        archived_sidecars = sorted(archive_dir.glob("*.meta.json"))
        self.assertEqual(len(archived_payloads), 1)
        self.assertEqual(len(archived_sidecars), 1)
        name = archived_payloads[0].name
        self.assertIn("2026-07-14T09-15-30Z", name)
        self.assertNotIn(":", name)
        # Archived pair holds the PRIOR content + sidecar.
        with open(archived_payloads[0]) as fh:
            self.assertEqual(json.load(fh)["run"], 1)
        with open(archived_sidecars[0]) as fh:
            self.assertEqual(json.load(fh)["idempotency_key"], "key-run-1")

    def test_second_write_keys_on_idempotency_when_no_uploaded_at(self):
        self.writer.write(
            label="ethanol",
            payload={"run": 1},
            endpoint="/uploads/conformers",
            idempotency_key="idem-key-abc123",
        )
        # No update_sidecar call -> sidecar has no uploaded_at; archive
        # name must fall back to the idempotency key.
        second = self.writer.write(
            label="ethanol",
            payload={"run": 2},
            endpoint="/uploads/conformers",
            idempotency_key="idem-key-def456",
        )
        archived = sorted(self._archive_dir(second).glob("*.payload.json"))
        self.assertEqual(len(archived), 1)
        self.assertIn("idem-key-abc123", archived[0].name)

    def test_corrupt_prior_sidecar_falls_back_gracefully(self):
        first = self.writer.write(
            label="ethanol",
            payload={"run": 1},
            endpoint="/uploads/conformers",
            idempotency_key="key-run-1",
        )
        # Corrupt the sidecar so no key can be extracted.
        first.sidecar_path.write_text("{ this is not valid json", encoding="utf-8")

        second = self.writer.write(
            label="ethanol",
            payload={"run": 2},
            endpoint="/uploads/conformers",
            idempotency_key="key-run-2",
        )
        archive_dir = self._archive_dir(second)
        # No crash; both prior files preserved (corrupt sidecar moved too).
        archived_payloads = sorted(archive_dir.glob("*.payload.json"))
        archived_sidecars = sorted(archive_dir.glob("*.meta.json"))
        self.assertEqual(len(archived_payloads), 1)
        self.assertEqual(len(archived_sidecars), 1)
        with open(archived_payloads[0]) as fh:
            self.assertEqual(json.load(fh)["run"], 1)
        # New canonical payload intact.
        with open(second.payload_path) as fh:
            self.assertEqual(json.load(fh)["run"], 2)

    def test_missing_prior_sidecar_does_not_crash(self):
        first = self.writer.write(
            label="ethanol",
            payload={"run": 1},
            endpoint="/uploads/conformers",
            idempotency_key="key-run-1",
        )
        first.sidecar_path.unlink()  # only the payload survives
        second = self.writer.write(
            label="ethanol",
            payload={"run": 2},
            endpoint="/uploads/conformers",
            idempotency_key="key-run-2",
        )
        archived_payloads = sorted(self._archive_dir(second).glob("*.payload.json"))
        self.assertEqual(len(archived_payloads), 1)
        with open(archived_payloads[0]) as fh:
            self.assertEqual(json.load(fh)["run"], 1)

    def test_archive_name_collision_is_disambiguated(self):
        # Three runs all sharing the same idempotency key (so the same
        # archive key) must not clobber each other in archive/.
        for run in (1, 2, 3):
            self.writer.write(
                label="ethanol",
                payload={"run": run},
                endpoint="/uploads/conformers",
                idempotency_key="same-key",
            )
        archive_dir = self.tmp_archive()
        archived_payloads = sorted(archive_dir.glob("*.payload.json"))
        # Runs 1 and 2 were archived (run 3 is the live canonical file);
        # both survive under distinct names.
        self.assertEqual(len(archived_payloads), 2)
        names = {p.name for p in archived_payloads}
        self.assertEqual(len(names), 2)
        runs = set()
        for p in archived_payloads:
            with open(p) as fh:
                runs.add(json.load(fh)["run"])
        self.assertEqual(runs, {1, 2})

    def tmp_archive(self):
        from pathlib import Path
        return Path(self.tmp) / PayloadWriter.SUBDIR / PayloadWriter.ARCHIVE_SUBDIR

    def test_archive_previous_disabled_overwrites_in_place(self):
        writer = PayloadWriter(self.tmp, archive_previous=False)
        first = writer.write(
            label="ethanol",
            payload={"run": 1},
            endpoint="/uploads/conformers",
            idempotency_key="key-run-1",
        )
        writer.write(
            label="ethanol",
            payload={"run": 2},
            endpoint="/uploads/conformers",
            idempotency_key="key-run-2",
        )
        self.assertFalse(self._archive_dir(first).exists())

    def test_partial_infix_preserved_in_archive_name(self):
        self.writer.write(
            label="r0_p0",
            payload={"run": 1},
            endpoint="/uploads/computed-reaction",
            idempotency_key="idem-partial-1",
            payload_kind="computed_reaction",
            is_partial=True,
        )
        second = self.writer.write(
            label="r0_p0",
            payload={"run": 2},
            endpoint="/uploads/computed-reaction",
            idempotency_key="idem-partial-2",
            payload_kind="computed_reaction",
            is_partial=True,
        )
        archived = sorted(self._archive_dir(second).glob("*.payload.json"))
        self.assertEqual(len(archived), 1)
        # The .partial infix survives into the archived name.
        self.assertIn(".partial.", archived[0].name)


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
        sc.public_refs = {"artifact_refs": ["artifact_7"]}
        sc.request_ids = [{"operation": "artifact_upload", "request_id": "req-art", "status_code": 201}]
        sc.preflight = {"ready": True, "status_code": 200, "request_id": "req-ready"}
        self.writer.update_artifact_sidecar(result.sidecar_path, sc)
        on_disk = json.loads(result.sidecar_path.read_text())
        self.assertEqual(on_disk["status"], "uploaded")
        self.assertEqual(on_disk["response_status_code"], 201)
        self.assertEqual(on_disk["response_body"]["calculation_id"], 42)
        self.assertEqual(on_disk["public_refs"], {"artifact_refs": ["artifact_7"]})
        self.assertEqual(on_disk["request_ids"][0]["request_id"], "req-art")
        self.assertEqual(on_disk["preflight"]["request_id"], "req-ready")


class TestShouldReplaySidecar(unittest.TestCase):
    """The replay-tool contract: partial sidecars must not auto-replay.

    These tests pin the contract that any in-tree or external replay
    code (e.g. ``tckdb-client``) imports/relies on. If this gate ever
    weakens, partial computed-reaction bundles could be POSTed as
    complete records — which is exactly the failure mode phase-1's
    sidecar-only policy is designed to prevent.
    """

    def _meta(self, **overrides):
        base = SidecarMetadata(
            payload_file="/tmp/x.payload.json",
            endpoint="/uploads/computed-reaction",
            idempotency_key="arc:proj:rxn0:noTS:computed_reaction:abc1234567890def",
            payload_kind="computed_reaction",
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    def test_complete_sidecar_replays(self):
        self.assertTrue(should_replay_sidecar(self._meta()))

    def test_partial_sidecar_skipped_by_default(self):
        self.assertFalse(should_replay_sidecar(self._meta(is_partial=True)))

    def test_partial_sidecar_opt_in_with_explicit_flag(self):
        # The opt-in path is reserved for once the TCKDB server formally
        # accepts transition_state=null bundles. Keeping the kwarg
        # working today means the contract is greppable.
        self.assertTrue(
            should_replay_sidecar(self._meta(is_partial=True), include_partial=True)
        )

    def test_dict_input_partial_skipped(self):
        # Replay daemons re-read sidecars as JSON; the helper must
        # accept the on-disk dict shape too.
        on_disk = {
            "payload_file": "/tmp/x.payload.json",
            "endpoint": "/uploads/computed-reaction",
            "is_partial": True,
        }
        self.assertFalse(should_replay_sidecar(on_disk))

    def test_dict_input_complete_replays(self):
        on_disk = {
            "payload_file": "/tmp/x.payload.json",
            "endpoint": "/uploads/computed-reaction",
            "is_partial": False,
        }
        self.assertTrue(should_replay_sidecar(on_disk))

    def test_dict_missing_is_partial_treated_as_complete(self):
        # Pre-phase-1 sidecars on disk lack the field entirely. They
        # are by definition complete bundles (the partial path didn't
        # exist when they were written). Replay should proceed.
        on_disk = {
            "payload_file": "/tmp/x.payload.json",
            "endpoint": "/uploads/computed-reaction",
        }
        self.assertTrue(should_replay_sidecar(on_disk))

    def test_none_input_does_not_replay(self):
        # Defensive: a missing/unparseable sidecar must not crash the
        # caller and must not be replayed.
        self.assertFalse(should_replay_sidecar(None))

    def test_round_trip_partial_sidecar_from_disk(self):
        # End-to-end: write a partial sidecar via PayloadWriter, parse
        # the on-disk JSON back, verify the helper rejects it. Catches
        # any regression where the writer stops emitting is_partial.
        tmp = tempfile.mkdtemp(prefix="arc-tckdb-replay-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        writer = PayloadWriter(tmp)
        result = writer.write(
            label="rxn0",
            payload={"species": []},
            endpoint="/uploads/computed-reaction",
            idempotency_key="arc:proj:rxn0:noTS:computed_reaction:abc1234567890def",
            payload_kind="computed_reaction",
            is_partial=True,
        )
        on_disk = json.loads(result.sidecar_path.read_text())
        self.assertTrue(on_disk["is_partial"])
        self.assertFalse(should_replay_sidecar(on_disk))
        self.assertTrue(
            should_replay_sidecar(on_disk, include_partial=True)
        )


if __name__ == "__main__":
    unittest.main()
