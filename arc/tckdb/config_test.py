#!/usr/bin/env python3
# encoding: utf-8

"""Unit tests for arc.tckdb.config."""

import os
import unittest
from unittest import mock

from arc.tckdb.config import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_PAYLOAD_DIR,
    DEFAULT_TIMEOUT_SECONDS,
    TCKDBConfig,
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


if __name__ == "__main__":
    unittest.main()
