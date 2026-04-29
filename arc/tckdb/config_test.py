#!/usr/bin/env python3
# encoding: utf-8

"""Unit tests for arc.tckdb.config."""

import logging
import os
import unittest
from unittest import mock

from arc.tckdb.config import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_ARTIFACT_KINDS,
    DEFAULT_ARTIFACT_MAX_SIZE_MB,
    DEFAULT_PAYLOAD_DIR,
    DEFAULT_TIMEOUT_SECONDS,
    IMPLEMENTED_ARTIFACT_KINDS,
    TCKDBArtifactConfig,
    TCKDBConfig,
    VALID_ARTIFACT_KINDS,
)


class TestTCKDBConfig(unittest.TestCase):

    def test_from_dict_returns_none_when_missing(self):
        self.assertIsNone(TCKDBConfig.from_dict(None))
        self.assertIsNone(TCKDBConfig.from_dict({}))

    def test_from_dict_returns_none_when_disabled(self):
        self.assertIsNone(
            TCKDBConfig.from_dict({"enabled": False, "base_url": "http://x"})
        )

    def test_from_dict_requires_base_url_when_enabled(self):
        with self.assertRaises(ValueError):
            TCKDBConfig.from_dict({"enabled": True})

    def test_from_dict_uses_defaults(self):
        cfg = TCKDBConfig.from_dict(
            {"enabled": True, "base_url": "http://localhost:8000/api/v1"}
        )
        self.assertIsNotNone(cfg)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.api_key_env, DEFAULT_API_KEY_ENV)
        self.assertEqual(cfg.payload_dir, DEFAULT_PAYLOAD_DIR)
        self.assertEqual(cfg.timeout_seconds, DEFAULT_TIMEOUT_SECONDS)
        self.assertTrue(cfg.upload)
        self.assertFalse(cfg.strict)

    def test_from_dict_overrides(self):
        cfg = TCKDBConfig.from_dict(
            {
                "enabled": True,
                "base_url": "http://srv/api/v1",
                "api_key_env": "MY_KEY",
                "payload_dir": "/tmp/payloads",
                "upload": False,
                "strict": True,
                "timeout_seconds": 5,
                "project_label": "proj-A",
            }
        )
        self.assertEqual(cfg.api_key_env, "MY_KEY")
        self.assertEqual(cfg.payload_dir, "/tmp/payloads")
        self.assertFalse(cfg.upload)
        self.assertTrue(cfg.strict)
        self.assertEqual(cfg.timeout_seconds, 5.0)
        self.assertEqual(cfg.project_label, "proj-A")

    def test_resolve_api_key_from_env(self):
        cfg = TCKDBConfig(enabled=True, base_url="http://x", api_key_env="X_TEST_KEY")
        with mock.patch.dict(os.environ, {"X_TEST_KEY": "secret"}, clear=False):
            self.assertEqual(cfg.resolve_api_key(), "secret")

    def test_resolve_api_key_missing(self):
        cfg = TCKDBConfig(enabled=True, base_url="http://x", api_key_env="DOES_NOT_EXIST_X")
        os.environ.pop("DOES_NOT_EXIST_X", None)
        self.assertIsNone(cfg.resolve_api_key())

    def test_artifacts_defaults_when_omitted(self):
        cfg = TCKDBConfig.from_dict({"enabled": True, "base_url": "http://x"})
        self.assertIsNotNone(cfg)
        self.assertFalse(cfg.artifacts.upload)
        self.assertEqual(cfg.artifacts.kinds, DEFAULT_ARTIFACT_KINDS)
        self.assertEqual(cfg.artifacts.max_size_mb, DEFAULT_ARTIFACT_MAX_SIZE_MB)

    def test_artifacts_full_block(self):
        cfg = TCKDBConfig.from_dict({
            "enabled": True,
            "base_url": "http://x",
            "artifacts": {
                "upload": True,
                "kinds": ["output_log", "input"],
                "max_size_mb": 25,
            },
        })
        self.assertTrue(cfg.artifacts.upload)
        self.assertEqual(cfg.artifacts.kinds, ("output_log", "input"))
        self.assertEqual(cfg.artifacts.max_size_mb, 25)

    def test_artifacts_partial_block_uses_defaults(self):
        cfg = TCKDBConfig.from_dict({
            "enabled": True,
            "base_url": "http://x",
            "artifacts": {"upload": True},
        })
        self.assertTrue(cfg.artifacts.upload)
        self.assertEqual(cfg.artifacts.kinds, DEFAULT_ARTIFACT_KINDS)
        self.assertEqual(cfg.artifacts.max_size_mb, DEFAULT_ARTIFACT_MAX_SIZE_MB)

    def test_artifacts_unknown_kind_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            TCKDBConfig.from_dict({
                "enabled": True,
                "base_url": "http://x",
                "artifacts": {"kinds": ["output_log", "bogus"]},
            })
        self.assertIn("bogus", str(ctx.exception))

    def test_artifacts_max_size_zero_rejected(self):
        with self.assertRaises(ValueError):
            TCKDBConfig.from_dict({
                "enabled": True,
                "base_url": "http://x",
                "artifacts": {"max_size_mb": 0},
            })

    def test_artifacts_max_size_negative_rejected(self):
        with self.assertRaises(ValueError):
            TCKDBConfig.from_dict({
                "enabled": True,
                "base_url": "http://x",
                "artifacts": {"max_size_mb": -5},
            })

    def test_artifacts_kinds_string_normalized_to_tuple(self):
        cfg = TCKDBConfig.from_dict({
            "enabled": True,
            "base_url": "http://x",
            "artifacts": {"kinds": "output_log"},
        })
        self.assertEqual(cfg.artifacts.kinds, ("output_log",))

    def test_valid_artifact_kinds_matches_server_enum(self):
        # Sanity check: keep the ARC-side allowlist in sync with the
        # server's ArtifactKind enum.
        self.assertEqual(
            VALID_ARTIFACT_KINDS,
            frozenset({"input", "output_log", "checkpoint",
                       "formatted_checkpoint", "ancillary"}),
        )


class TestTCKDBArtifactConfig(unittest.TestCase):

    def test_dataclass_defaults(self):
        c = TCKDBArtifactConfig()
        self.assertFalse(c.upload)
        self.assertEqual(c.kinds, DEFAULT_ARTIFACT_KINDS)
        self.assertEqual(c.max_size_mb, DEFAULT_ARTIFACT_MAX_SIZE_MB)


class TestImplementedKinds(unittest.TestCase):
    """Server-accepted vs ARC-implemented kinds split."""

    def test_implemented_is_subset_of_valid(self):
        self.assertTrue(IMPLEMENTED_ARTIFACT_KINDS.issubset(VALID_ARTIFACT_KINDS))

    def test_implemented_kinds_today(self):
        # Pin the current implementation surface so adding a kind is a
        # deliberate, test-failing change.
        self.assertEqual(
            IMPLEMENTED_ARTIFACT_KINDS,
            frozenset({"output_log", "input"}),
        )

    def test_unimplemented_kind_warns_but_does_not_raise(self):
        with self.assertLogs("arc", level="WARNING") as cm:
            cfg = TCKDBConfig.from_dict({
                "enabled": True,
                "base_url": "http://x",
                "artifacts": {"kinds": ["output_log", "checkpoint"]},
            })
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.artifacts.kinds, ("output_log", "checkpoint"))
        joined = "\n".join(cm.output)
        self.assertIn("checkpoint", joined)
        self.assertIn("doesn't yet produce", joined)

    def test_implemented_only_kinds_no_warning(self):
        # No warning when every requested kind is in IMPLEMENTED.
        # We assert by checking the warning logger was either silent or
        # only emitted unrelated lines (no "doesn't yet produce" phrase).
        logger_name = "arc"
        with self.assertLogs(logger_name, level="WARNING") as cm:
            # Force at least one log line so assertLogs doesn't raise:
            logging.getLogger(logger_name).warning("sentinel")
            TCKDBConfig.from_dict({
                "enabled": True,
                "base_url": "http://x",
                "artifacts": {"kinds": ["output_log", "input"]},
            })
        joined = "\n".join(cm.output)
        self.assertNotIn("doesn't yet produce", joined)


if __name__ == "__main__":
    import logging
    unittest.main()
