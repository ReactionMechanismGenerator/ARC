#!/usr/bin/env python3
# encoding: utf-8

"""Unit tests for arc.tckdb.config."""

import os
import tempfile
import unittest
from pathlib import Path

from arc.common import get_logger
from arc.exceptions import InputError
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
    _read_tckdb_api_key_from_env_file,
    resolve_tckdb_api_key,
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
        self.assertTrue(cfg.preflight)
        self.assertFalse(cfg.strict)

    def test_from_dict_overrides(self):
        cfg = TCKDBConfig.from_dict(
            {
                "enabled": True,
                "base_url": "http://srv/api/v1",
                "api_key_env": "MY_KEY",
                "payload_dir": "/tmp/payloads",
                "upload": False,
                "preflight": False,
                "strict": True,
                "timeout_seconds": 5,
                "project_label": "proj-A",
            }
        )
        self.assertEqual(cfg.api_key_env, "MY_KEY")
        self.assertEqual(cfg.payload_dir, "/tmp/payloads")
        self.assertFalse(cfg.upload)
        self.assertFalse(cfg.preflight)
        self.assertTrue(cfg.strict)
        self.assertEqual(cfg.timeout_seconds, 5.0)
        self.assertEqual(cfg.project_label, "proj-A")

    def test_allow_partial_uploads_defaults_to_true(self):
        # Phase-1 default-on partial sidecars: a tckdb block that omits
        # allow_partial_uploads still ends up with sidecar generation
        # for failed-TS reactions enabled. Live POST of those sidecars
        # is gated separately inside the adapter.
        cfg = TCKDBConfig.from_dict(
            {"enabled": True, "base_url": "http://localhost:8000/api/v1"}
        )
        self.assertIsNotNone(cfg)
        self.assertTrue(cfg.allow_partial_uploads)

    def test_allow_partial_uploads_can_be_disabled(self):
        cfg = TCKDBConfig.from_dict({
            "enabled": True,
            "base_url": "http://localhost:8000/api/v1",
            "allow_partial_uploads": False,
        })
        self.assertIsNotNone(cfg)
        self.assertFalse(cfg.allow_partial_uploads)

    def test_resolve_api_key_from_env(self):
        cfg = TCKDBConfig(enabled=True, base_url="http://x", api_key_env="X_TEST_KEY")
        with unittest.mock.patch.dict(os.environ, {"X_TEST_KEY": "secret"}, clear=False):
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
            get_logger().warning("sentinel")
            TCKDBConfig.from_dict({
                "enabled": True,
                "base_url": "http://x",
                "artifacts": {"kinds": ["output_log", "input"]},
            })
        joined = "\n".join(cm.output)
        self.assertNotIn("doesn't yet produce", joined)


class TestResolveTckdbApiKey(unittest.TestCase):
    """Resolution of the API key from env vs configured local files."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _write(self, name: str, content: str) -> Path:
        p = self.tmp / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_env_var_wins_over_api_key_file(self):
        path = self._write("key.txt", "from_file_key")
        with unittest.mock.patch.dict(os.environ, {"X_T_KEY": "from_env_key"}, clear=False):
            got = resolve_tckdb_api_key(
                api_key_env="X_T_KEY",
                api_key_file=str(path),
            )
        self.assertEqual(got, "from_env_key")

    def test_api_key_file_returns_raw_key(self):
        path = self._write("key.txt", "tck_abcdef")
        os.environ.pop("X_T_KEY", None)
        got = resolve_tckdb_api_key(api_key_env="X_T_KEY", api_key_file=str(path))
        self.assertEqual(got, "tck_abcdef")

    def test_api_key_file_strips_trailing_newline(self):
        path = self._write("key.txt", "tck_abcdef\n")
        os.environ.pop("X_T_KEY", None)
        got = resolve_tckdb_api_key(api_key_env="X_T_KEY", api_key_file=str(path))
        self.assertEqual(got, "tck_abcdef")

    def test_api_key_file_strips_surrounding_whitespace(self):
        path = self._write("key.txt", "   tck_abcdef \n\n")
        os.environ.pop("X_T_KEY", None)
        got = resolve_tckdb_api_key(api_key_env="X_T_KEY", api_key_file=str(path))
        self.assertEqual(got, "tck_abcdef")

    def test_missing_api_key_file_raises(self):
        os.environ.pop("X_T_KEY", None)
        with self.assertRaises(InputError) as ctx:
            resolve_tckdb_api_key(
                api_key_env="X_T_KEY",
                api_key_file=str(self.tmp / "does_not_exist"),
            )
        self.assertIn("does not exist", str(ctx.exception))

    def test_empty_api_key_file_raises(self):
        path = self._write("key.txt", "   \n\n")
        os.environ.pop("X_T_KEY", None)
        with self.assertRaises(InputError) as ctx:
            resolve_tckdb_api_key(
                api_key_env="X_T_KEY",
                api_key_file=str(path),
            )
        self.assertIn("empty", str(ctx.exception))

    def test_no_sources_returns_none(self):
        os.environ.pop("X_T_KEY", None)
        self.assertIsNone(resolve_tckdb_api_key(api_key_env="X_T_KEY"))

    def test_user_home_expansion(self):
        # Paths starting with ~ should be expanded.
        with unittest.mock.patch.dict(os.environ, {"HOME": str(self.tmp)}, clear=False):
            self._write("key.txt", "tck_home")
            os.environ.pop("X_T_KEY", None)
            got = resolve_tckdb_api_key(
                api_key_env="X_T_KEY",
                api_key_file="~/key.txt",
            )
        self.assertEqual(got, "tck_home")

    # api_key_env_file paths -------------------------------------------------

    def test_env_file_supports_unquoted(self):
        path = self._write("auth.env", "TCKDB_API_KEY=tck_unq\n")
        os.environ.pop("TCKDB_API_KEY", None)
        got = resolve_tckdb_api_key(api_key_env_file=str(path))
        self.assertEqual(got, "tck_unq")

    def test_env_file_supports_single_quoted(self):
        path = self._write("auth.env", "TCKDB_API_KEY='tck_sq'\n")
        os.environ.pop("TCKDB_API_KEY", None)
        got = resolve_tckdb_api_key(api_key_env_file=str(path))
        self.assertEqual(got, "tck_sq")

    def test_env_file_supports_double_quoted(self):
        path = self._write("auth.env", 'TCKDB_API_KEY="tck_dq"\n')
        os.environ.pop("TCKDB_API_KEY", None)
        got = resolve_tckdb_api_key(api_key_env_file=str(path))
        self.assertEqual(got, "tck_dq")

    def test_env_file_supports_export_prefix(self):
        path = self._write("auth.env", "export TCKDB_API_KEY='tck_exp'\n")
        os.environ.pop("TCKDB_API_KEY", None)
        got = resolve_tckdb_api_key(api_key_env_file=str(path))
        self.assertEqual(got, "tck_exp")

    def test_env_file_ignores_comments_and_blank_lines(self):
        body = (
            "# top comment\n"
            "\n"
            "OTHER_VAR=irrelevant\n"
            "   # indented comment\n"
            "\n"
            "export TCKDB_API_KEY='tck_real'\n"
            "TRAILING=stuff\n"
        )
        path = self._write("auth.env", body)
        os.environ.pop("TCKDB_API_KEY", None)
        got = resolve_tckdb_api_key(api_key_env_file=str(path))
        self.assertEqual(got, "tck_real")

    def test_env_file_missing_var_raises(self):
        path = self._write("auth.env", "OTHER_VAR=not_the_one\n")
        os.environ.pop("TCKDB_API_KEY", None)
        with self.assertRaises(InputError) as ctx:
            resolve_tckdb_api_key(api_key_env_file=str(path))
        self.assertIn("does not define", str(ctx.exception))

    def test_env_file_missing_path_raises(self):
        os.environ.pop("TCKDB_API_KEY", None)
        with self.assertRaises(InputError) as ctx:
            resolve_tckdb_api_key(
                api_key_env_file=str(self.tmp / "no_such_auth.env"),
            )
        self.assertIn("does not exist", str(ctx.exception))

    def test_env_file_does_not_execute_shell(self):
        # If the parser were sourcing the file, the $(...) would run
        # and the value would equal "SHOULD_NOT_RUN" (echo's output).
        # Asserting we get the literal, untransformed token proves the
        # subshell never ran.
        body = "TCKDB_API_KEY='$(echo SHOULD_NOT_RUN)'\n"
        path = self._write("auth.env", body)
        os.environ.pop("TCKDB_API_KEY", None)
        got = resolve_tckdb_api_key(api_key_env_file=str(path))
        self.assertEqual(got, "$(echo SHOULD_NOT_RUN)")
        self.assertNotEqual(got, "SHOULD_NOT_RUN")

    def test_env_file_does_not_interpolate_dollar_vars(self):
        # POSIX shlex does not expand $VAR. Confirm we hand back the
        # literal "$HOME" rather than its expanded value.
        path = self._write("auth.env", "TCKDB_API_KEY='$HOME'\n")
        os.environ.pop("TCKDB_API_KEY", None)
        got = resolve_tckdb_api_key(api_key_env_file=str(path))
        self.assertEqual(got, "$HOME")

    def test_env_file_uses_configured_var_name(self):
        path = self._write("auth.env", "MY_KEY='tck_custom'\n")
        os.environ.pop("MY_KEY", None)
        got = resolve_tckdb_api_key(
            api_key_env="MY_KEY",
            api_key_env_file=str(path),
        )
        self.assertEqual(got, "tck_custom")

    # Resolution-order priority ---------------------------------------------

    def test_api_key_file_takes_precedence_over_env_file(self):
        kf = self._write("key.txt", "tck_from_keyfile")
        ef = self._write("auth.env", "TCKDB_API_KEY='tck_from_envfile'\n")
        os.environ.pop("TCKDB_API_KEY", None)
        got = resolve_tckdb_api_key(
            api_key_file=str(kf),
            api_key_env_file=str(ef),
        )
        self.assertEqual(got, "tck_from_keyfile")


class TestReadEnvFileHelper(unittest.TestCase):
    """Direct tests of the env-file parser."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "auth.env"

    def _write(self, content: str) -> Path:
        self.path.write_text(content, encoding="utf-8")
        return self.path

    def test_returns_none_when_var_absent(self):
        self._write("OTHER=xx\n")
        self.assertIsNone(_read_tckdb_api_key_from_env_file(self.path, "TCKDB_API_KEY"))

    def test_first_match_wins(self):
        self._write("TCKDB_API_KEY=first\nTCKDB_API_KEY=second\n")
        self.assertEqual(
            _read_tckdb_api_key_from_env_file(self.path, "TCKDB_API_KEY"),
            "first",
        )


class TestTCKDBConfigKeyFileFields(unittest.TestCase):
    """from_dict + integration with resolve_api_key()."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_from_dict_parses_file_paths(self):
        cfg = TCKDBConfig.from_dict({
            "enabled": True,
            "base_url": "http://x",
            "api_key_file": "/etc/tckdb/key",
            "api_key_env_file": "/etc/tckdb/auth.env",
        })
        self.assertEqual(cfg.api_key_file, "/etc/tckdb/key")
        self.assertEqual(cfg.api_key_env_file, "/etc/tckdb/auth.env")

    def test_from_dict_rejects_non_string_paths(self):
        with self.assertRaises(ValueError):
            TCKDBConfig.from_dict({
                "enabled": True,
                "base_url": "http://x",
                "api_key_file": ["/a", "/b"],
            })

    def test_resolve_api_key_uses_api_key_file(self):
        kf = self.tmp / "key.txt"
        kf.write_text("tck_via_cfg\n", encoding="utf-8")
        cfg = TCKDBConfig.from_dict({
            "enabled": True,
            "base_url": "http://x",
            "api_key_env": "DOES_NOT_EXIST_X_X",
            "api_key_file": str(kf),
        })
        os.environ.pop("DOES_NOT_EXIST_X_X", None)
        self.assertEqual(cfg.resolve_api_key(), "tck_via_cfg")

    def test_resolve_api_key_env_var_overrides_config_file(self):
        kf = self.tmp / "key.txt"
        kf.write_text("from_file\n", encoding="utf-8")
        cfg = TCKDBConfig.from_dict({
            "enabled": True,
            "base_url": "http://x",
            "api_key_env": "X_OVERRIDE_KEY",
            "api_key_file": str(kf),
        })
        with unittest.mock.patch.dict(os.environ, {"X_OVERRIDE_KEY": "from_env"}, clear=False):
            self.assertEqual(cfg.resolve_api_key(), "from_env")

    def test_describe_api_key_sources_terse_default(self):
        cfg = TCKDBConfig(enabled=True, base_url="http://x", api_key_env="X_KEY")
        self.assertEqual(cfg.describe_api_key_sources(), "env var 'X_KEY'")

    def test_describe_api_key_sources_with_files(self):
        cfg = TCKDBConfig(
            enabled=True,
            base_url="http://x",
            api_key_env="X_KEY",
            api_key_file="/etc/tckdb/key",
            api_key_env_file="/etc/tckdb/auth.env",
        )
        desc = cfg.describe_api_key_sources()
        self.assertIn("env var 'X_KEY'", desc)
        self.assertIn("api_key_file=/etc/tckdb/key", desc)
        self.assertIn("api_key_env_file=/etc/tckdb/auth.env", desc)


if __name__ == "__main__":
    unittest.main()
