"""Tests for ARC's parser-neutral TCKDB evidence producer."""

import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from arc.tckdb_evidence import (
    EVIDENCE_SCHEMA_NAME,
    EVIDENCE_SCHEMA_VERSION,
    _build_gsm,
    _build_hessian,
    _build_irc,
    build_tckdb_evidence,
    write_tckdb_evidence_atomic,
)


DOC_ID = "0123456789abcdef0123456789abcdef"
XYZ_DICT = {
    "symbols": ("H", "H"),
    "isotopes": (1, 1),
    "coords": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.7)),
}
ATOM_LINES = "H       0.00000000    0.00000000    0.00000000\nH       0.00000000    0.00000000    0.70000000"


class TestEvidenceProducer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _touch(self, relative="calcs/freq/output.log"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
        return path

    def test_exact_top_level_shape_and_no_timestamp(self):
        output = {
            "schema_version": "1.1", "arc_version": "1.1.0", "arc_git_commit": "abc123",
            "species": [], "transition_states": [],
        }
        result = build_tckdb_evidence(output_doc=output, project_directory=self.root, document_id=DOC_ID)
        self.assertEqual(set(result), {
            "schema_name", "schema_version", "document_id", "output_schema_version", "producer", "records"
        })
        self.assertEqual(result["schema_name"], EVIDENCE_SCHEMA_NAME)
        self.assertEqual(result["schema_version"], EVIDENCE_SCHEMA_VERSION)
        self.assertNotIn("timestamp", json.dumps(result))

    @patch("arc.tckdb_evidence.ess_factory")
    @patch("arc.tckdb_evidence.determine_ess", return_value="gaussian")
    def test_gaussian_hessian_shape_and_triangle(self, determine, factory):
        self._touch()
        parser = MagicMock()
        parser.parse_cartesian_hessian_lower_triangle.return_value = [float(i) for i in range(21)]
        factory.return_value = parser
        envelope = _build_hessian(
            {"label": "H2", "freq_log": "calcs/freq/output.log", "xyz": ATOM_LINES}, self.root
        )
        self.assertEqual(envelope["status"], "available")
        value = envelope["value"]
        self.assertEqual(value["source"], "parsed_log")
        self.assertEqual(value["matrix_dimension"], 6)
        self.assertEqual(value["lower_triangle"], [float(i) for i in range(21)])
        self.assertEqual(value["geometry_xyz_text"].splitlines()[1], "H2")
        self.assertFalse(value["geometry_xyz_text"].endswith("\n"))

    @patch("arc.tckdb_evidence.ess_factory")
    @patch("arc.tckdb_evidence.determine_ess", return_value="orca")
    def test_orca_hessian_source(self, determine, factory):
        self._touch()
        factory.return_value.parse_cartesian_hessian_lower_triangle.return_value = [0.0] * 21
        envelope = _build_hessian(
            {"label": "H2", "freq_log": "calcs/freq/output.log", "xyz": ATOM_LINES}, self.root
        )
        self.assertEqual(envelope["value"]["source"], "parsed_hess")

    @patch("arc.tckdb_evidence.ess_factory")
    @patch("arc.tckdb_evidence.determine_ess", return_value="gaussian")
    def test_bad_hessian_length_and_monatomic_are_unavailable(self, determine, factory):
        self._touch()
        factory.return_value.parse_cartesian_hessian_lower_triangle.return_value = [0.0]
        bad = _build_hessian(
            {"label": "H2", "freq_log": "calcs/freq/output.log", "xyz": ATOM_LINES}, self.root
        )
        mono = _build_hessian(
            {"label": "H", "freq_log": "calcs/freq/output.log", "xyz": "H 0 0 0"}, self.root
        )
        self.assertEqual(bad["reason"], "parse_failed")
        self.assertEqual(mono["reason"], "unsupported_source")

    @patch("arc.tckdb_evidence.parse_irc_path")
    def test_rich_irc_fields_and_direction(self, rich):
        self._touch("calcs/irc/forward.log")
        rich.return_value = [{
            "point_number": 2, "direction": "forward", "xyz": XYZ_DICT,
            "electronic_energy_hartree": -1.2, "reaction_coordinate": 0.4,
            "max_gradient": 0.01, "rms_gradient": 0.005,
        }]
        envelope = _build_irc(
            {"label": "TS0", "irc_logs": ["calcs/irc/forward.log"], "irc_log_directions": ["forward"]},
            self.root,
        )
        point = envelope["value"]["trajectories"][0]["points"][0]
        self.assertEqual(point["source_point_index"], 2)
        self.assertEqual(point["reaction_coordinate_sqrt_amu_bohr"], 0.4)
        self.assertEqual(point["max_gradient_hartree_per_bohr"], 0.01)
        self.assertEqual(point["geometry_xyz_text"].splitlines()[1], "")
        self.assertFalse(point["geometry_xyz_text"].endswith("\n"))

    @patch("arc.tckdb_evidence.parse_irc_traj", return_value=[XYZ_DICT])
    @patch("arc.tckdb_evidence.parse_irc_path", return_value=None)
    def test_geometry_only_irc_fallback(self, rich, geometry):
        self._touch("calcs/irc/reverse.log")
        envelope = _build_irc(
            {"label": "TS0", "irc_logs": ["calcs/irc/reverse.log"], "irc_log_directions": []}, self.root
        )
        point = envelope["value"]["trajectories"][0]["points"][0]
        self.assertEqual(point["source_point_index"], 0)
        self.assertEqual(point["direction"], "reverse")

    @patch("arc.tckdb_evidence.parse_irc_path")
    def test_partial_irc_records_omitted_source(self, rich):
        self._touch("calcs/irc/good.log")
        rich.return_value = [{"point_number": 1, "direction": None, "xyz": XYZ_DICT}]
        envelope = _build_irc(
            {"label": "TS0", "irc_logs": ["missing.log", "calcs/irc/good.log"],
             "irc_log_directions": ["forward", "reverse"]}, self.root
        )
        self.assertEqual(envelope["status"], "available")
        self.assertEqual(envelope["value"]["omitted_source_paths"], ["missing.log"])

    @patch("arc.tckdb_evidence.parse_gsm_stringfile_energies", return_value=[0.0, 2.0, 0.0])
    @patch("arc.tckdb_evidence.parse_trajectory", return_value=[XYZ_DICT, XYZ_DICT, XYZ_DICT])
    @patch("arc.tckdb_evidence.kabsch", return_value=0.25)
    def test_gsm_frames_coordinates_selected_and_node_priority(self, kabsch_mock, trajectory, energies):
        stringfile = self._touch("calcs/gsm/stringfile.xyz0000")
        outputs = stringfile.parent / "gsm_node_outputs"
        outputs.mkdir()
        (outputs / "0000.01.energy").write_text("$energy\n 1 -2.0 0 0\n$end\n")
        (outputs / "0000.01.xtbout").write_text("TOTAL ENERGY -9.0 Eh")
        (outputs / "0000.02.xtbout").write_text("TOTAL ENERGY -1.0 Eh")
        envelope = _build_gsm({"label": "TS0", "gsm_log": "calcs/gsm/stringfile.xyz0000"}, self.root)
        value = envelope["value"]
        self.assertEqual(value["selected_source_point_index"], 2)
        self.assertEqual([p["path_coordinate_angstrom"] for p in value["points"]], [0.0, 0.25, 0.5])
        self.assertEqual(value["points"][1]["electronic_energy_hartree"], -2.0)
        self.assertEqual(value["points"][2]["electronic_energy_hartree"], -1.0)
        self.assertEqual(value["points"][1]["geometry_xyz_text"].splitlines()[1], "gsm_point_1")
        self.assertFalse(value["points"][1]["geometry_xyz_text"].endswith("\n"))

    @patch("arc.tckdb_evidence.parse_irc_path")
    def test_malformed_and_nonfinite_irc_logs_are_isolated(self, rich):
        self._touch("calcs/irc/bad.log")
        self._touch("calcs/irc/good.log")
        rich.side_effect = [
            [{"point_number": 0, "xyz": XYZ_DICT, "electronic_energy_hartree": math.nan}],
            [{"point_number": 1, "xyz": XYZ_DICT, "electronic_energy_hartree": -1.0}],
        ]
        envelope = _build_irc(
            {"label": "TS0", "irc_logs": ["calcs/irc/bad.log", "calcs/irc/good.log"]},
            self.root,
        )
        self.assertEqual(envelope["status"], "available")
        self.assertEqual(envelope["value"]["omitted_source_paths"], ["calcs/irc/bad.log"])
        self.assertEqual(len(envelope["value"]["trajectories"]), 1)

    @patch("arc.tckdb_evidence.parse_irc_traj", return_value=None)
    @patch("arc.tckdb_evidence.parse_irc_path", return_value=[{"xyz": None}])
    def test_all_malformed_irc_logs_are_unavailable(self, rich, geometry):
        self._touch("calcs/irc/bad.log")
        envelope = _build_irc(
            {"label": "TS0", "irc_logs": ["calcs/irc/bad.log"]}, self.root,
        )
        self.assertEqual(envelope["status"], "unavailable")
        self.assertEqual(envelope["source_paths"], ["calcs/irc/bad.log"])

    @patch("arc.tckdb_evidence._build_irc", side_effect=RuntimeError("unexpected"))
    @patch("arc.tckdb_evidence._build_gsm")
    @patch("arc.tckdb_evidence._build_hessian")
    def test_unexpected_kind_failure_preserves_other_evidence(self, hessian, gsm, irc):
        available = {"status": "available", "value": {}}
        hessian.return_value = available
        gsm.return_value = available
        output = {
            "schema_version": "1.1", "arc_version": "x", "arc_git_commit": "abc",
            "species": [],
            "transition_states": [{
                "label": "TS0", "freq_log": "freq.log", "irc_logs": ["irc.log"],
                "chosen_ts_method": "gsm", "gsm_log": "gsm.log",
            }],
        }
        record = build_tckdb_evidence(
            output_doc=output, project_directory=self.root, document_id=DOC_ID,
        )["records"][0]
        self.assertEqual(record["freq_hessian"], available)
        self.assertEqual(record["gsm"], available)
        self.assertEqual(record["irc"]["status"], "unavailable")

    @patch("arc.tckdb_evidence.parse_gsm_stringfile_energies", return_value=[0.0, 0.0, 0.0])
    @patch("arc.tckdb_evidence.parse_trajectory", return_value=[XYZ_DICT, XYZ_DICT, XYZ_DICT])
    @patch("arc.tckdb_evidence.kabsch", return_value=0.0)
    def test_gsm_all_zero_comment_energies_are_preserved(self, kabsch_mock, trajectory, energies):
        self._touch("calcs/gsm/stringfile.xyz0000")
        envelope = _build_gsm({"label": "TS0", "gsm_log": "calcs/gsm/stringfile.xyz0000"}, self.root)
        self.assertEqual(
            [point["stringfile_relative_energy_kcal_mol"] for point in envelope["value"]["points"]],
            [0.0, 0.0, 0.0],
        )

    def test_atomic_json_format_and_no_temporary_files(self):
        document = {
            "schema_name": EVIDENCE_SCHEMA_NAME, "schema_version": "1.0", "document_id": DOC_ID,
            "output_schema_version": "1.1", "producer": {"name": "ARC", "version": "x", "git_commit": None},
            "records": [],
        }
        path = write_tckdb_evidence_atomic(evidence_doc=document, output_directory=self.root)
        text = path.read_text()
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(json.loads(text), document)
        self.assertFalse(any(item.name.endswith(".tmp") for item in self.root.iterdir()))

    def test_non_finite_cannot_be_serialized_and_temp_is_removed(self):
        with self.assertRaises(ValueError):
            write_tckdb_evidence_atomic(evidence_doc={"bad": math.nan}, output_directory=self.root)
        self.assertFalse(any(item.name.endswith(".tmp") for item in self.root.iterdir()))

    def test_paths_are_relative_and_operational_fields_do_not_leak(self):
        output = {
            "schema_version": "1.1", "arc_version": "1.1.0", "arc_git_commit": None,
            "species": [], "transition_states": [],
        }
        document = build_tckdb_evidence(output_doc=output, project_directory=self.root, document_id=DOC_ID)
        serialized = json.dumps(document)
        for forbidden in ("server", "job_id", "credential", str(self.root)):
            self.assertNotIn(forbidden, serialized)

    def test_cross_repository_golden_contract(self):
        """Producer assembly is exactly the sidecar consumed by tckdb-adapters."""
        fixture_dir = Path(__file__).resolve().parents[2] / "tckdb-adapters" / \
            "tckdb_arc" / "tests" / "fixtures" / "golden"
        if not fixture_dir.is_dir():
            self.skipTest("tckdb-adapters golden fixture checkout is not adjacent")
        output = yaml.safe_load((fixture_dir / "phase3_output.yml").read_text())
        expected = json.loads((fixture_dir / "tckdb_evidence.json").read_text())
        expected_by_label = {record["label"]: record for record in expected["records"]}

        with patch(
            "arc.tckdb_evidence._build_hessian",
            side_effect=lambda record, _root: expected_by_label[record["label"]]["freq_hessian"],
        ), patch(
            "arc.tckdb_evidence._build_irc",
            side_effect=lambda record, _root: expected_by_label[record["label"]]["irc"],
        ), patch(
            "arc.tckdb_evidence._build_gsm",
            side_effect=lambda record, _root: expected_by_label[record["label"]]["gsm"],
        ):
            actual = build_tckdb_evidence(
                output_doc=output,
                project_directory=self.root,
                document_id=DOC_ID,
            )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
