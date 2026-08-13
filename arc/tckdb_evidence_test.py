"""Tests for ARC's parser-neutral TCKDB evidence producer."""

import json
import math
from collections.abc import Mapping
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from arc.common import ARC_PATH
from arc.constants import bohr_to_angstrom
from arc.parser.adapters.gaussian import GaussianParser
from arc.parser.adapters.hessian_test import vibrational_frequencies_cm1
from arc.species.converter import get_element_mass_from_xyz, str_to_xyz
from arc.tckdb_evidence import (
    EVIDENCE_SCHEMA_NAME,
    EVIDENCE_SCHEMA_VERSION,
    _build_gsm,
    _build_hessian,
    _build_irc,
    _parse_gradient,
    build_tckdb_evidence,
    write_tckdb_evidence_atomic,
)


ARC_TESTING_PATH = os.path.join(ARC_PATH, "arc", "testing")
GAUSSIAN_FREQ_LOG = os.path.join(ARC_TESTING_PATH, "restart", "3_restart_bde", "calcs", "Species",
                                 "anilino_radical", "freq_a4345", "output.out")
ORCA_FREQ_LOG = os.path.join(ARC_TESTING_PATH, "freq", "orca_hessian_h2o", "output.out")
ORCA_HESS = os.path.join(ARC_TESTING_PATH, "freq", "orca_hessian_h2o", "input.hess")
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

    def _stage(self, *sources, into="calcs/freq"):
        """Copy real ESS fixture files into the temp project and return the first one's run-relative path."""
        destination = self.root / into
        destination.mkdir(parents=True, exist_ok=True)
        for source in sources:
            shutil.copy(source, destination / os.path.basename(source))
        return f"{into}/{os.path.basename(sources[0])}"

    def test_gaussian_hessian_shape_and_triangle(self):
        """A real 13-atom Gaussian freq log yields a full, frame-stamped Hessian record."""
        relative = self._stage(GAUSSIAN_FREQ_LOG)
        envelope = _build_hessian(
            {"label": "anilino_radical", "freq_log": relative, "xyz": ATOM_LINES}, self.root
        )
        self.assertEqual(envelope["status"], "available")
        value = envelope["value"]
        self.assertEqual(value["source"], "parsed_log")
        self.assertEqual(value["atom_count"], 13)
        self.assertEqual(value["matrix_dimension"], 39)
        self.assertEqual(len(value["lower_triangle"]), 39 * 40 // 2)
        self.assertEqual(value["frame"], "gaussian_input_orientation")
        self.assertEqual(value["parser_version"], "arc-hessian-2")
        self.assertEqual(value["geometry_xyz_text"].splitlines()[1], "anilino_radical")
        self.assertFalse(value["geometry_xyz_text"].endswith("\n"))

    def test_hessian_geometry_ignores_the_records_exported_xyz(self):
        """The exported ``xyz`` is not the Hessian's frame, so the builder must not use it.

        The record's ``xyz`` here is a 2-atom placeholder; if the builder still
        paired it with the 13-atom Hessian the record would be incoherent.
        """
        relative = self._stage(GAUSSIAN_FREQ_LOG)
        value = _build_hessian(
            {"label": "anilino_radical", "freq_log": relative, "xyz": ATOM_LINES}, self.root
        )["value"]
        self.assertEqual(int(value["geometry_xyz_text"].splitlines()[0]), 13)
        self.assertNotIn("0.70000000", value["geometry_xyz_text"])

    def test_exported_hessian_and_geometry_reproduce_the_logs_frequencies(self):
        """The central physical invariant: the exported pair must reconstruct the ESS's own spectrum.

        This is what makes the Hessian evidence trustworthy. A transpose, a unit
        slip, a mis-paged column block, or a frame mismatch all leave the
        triangle's length and finiteness intact and are invisible to a shape
        assertion; only re-deriving the spectrum catches them.
        """
        relative = self._stage(GAUSSIAN_FREQ_LOG)
        value = _build_hessian(
            {"label": "anilino_radical", "freq_log": relative, "xyz": ATOM_LINES}, self.root
        )["value"]
        xyz = str_to_xyz("\n".join(value["geometry_xyz_text"].splitlines()[2:]))
        reported = np.sort(np.array(GaussianParser(GAUSSIAN_FREQ_LOG).parse_frequencies()))
        reconstructed = vibrational_frequencies_cm1(value["lower_triangle"], xyz, len(reported))
        self.assertLess(np.abs(reconstructed - reported).max(), 1.0)

    def test_orca_hessian_source(self):
        """An Orca freq log with a sibling .hess is sourced as ``parsed_hess`` and frame-stamped."""
        relative = self._stage(ORCA_FREQ_LOG, ORCA_HESS)
        envelope = _build_hessian({"label": "H2O", "freq_log": relative, "xyz": ATOM_LINES}, self.root)
        self.assertEqual(envelope["value"]["source"], "parsed_hess")
        self.assertEqual(envelope["value"]["frame"], "orca_hess_atoms")
        self.assertEqual(envelope["value"]["atom_count"], 3)

    def test_monatomic_and_unparseable_sources_are_unavailable(self):
        """Sources that cannot yield a coherent Hessian degrade to an envelope, never raise."""
        # A real log with no force-constant block at all.
        no_block = self._stage(os.path.join(ARC_TESTING_PATH, "freq", "dual_freq_output.out"),
                               into="calcs/none")
        self.assertEqual(
            _build_hessian({"label": "no_block", "freq_log": no_block, "xyz": ATOM_LINES}, self.root)["reason"],
            "empty_result",
        )
        # A file no ESS claims.
        self._touch("calcs/freq/unknown.log")
        self.assertEqual(
            _build_hessian({"label": "X", "freq_log": "calcs/freq/unknown.log", "xyz": ATOM_LINES},
                           self.root)["reason"],
            "unsupported_source",
        )
        self.assertEqual(
            _build_hessian({"label": "X", "freq_log": "", "xyz": ATOM_LINES}, self.root)["reason"],
            "missing_source",
        )

    @patch("arc.tckdb_evidence.ess_factory")
    @patch("arc.tckdb_evidence.determine_ess", return_value="gaussian")
    def test_triangle_inconsistent_with_geometry_is_unavailable(self, determine, factory):
        """A triangle whose length contradicts the frame geometry's atom count is rejected.

        No real log produces this pairing, so the parser is stubbed here
        deliberately — the assertion is about the builder's guard, not about
        parsing.
        """
        relative = self._stage(GAUSSIAN_FREQ_LOG)
        parser = MagicMock()
        # 10 entries cannot pack a 2-atom (3N = 6) triangle, which needs 21.
        parser.parse_cartesian_hessian_lower_triangle.return_value = [0.0] * 10
        parser.parse_cartesian_hessian_geometry.return_value = (XYZ_DICT, "gaussian_input_orientation")
        factory.return_value = parser
        envelope = _build_hessian(
            {"label": "mismatch", "freq_log": relative, "xyz": ATOM_LINES}, self.root
        )
        self.assertEqual(envelope["reason"], "parse_failed")

    @patch("arc.tckdb_evidence.ess_factory")
    @patch("arc.tckdb_evidence.determine_ess", return_value="gaussian")
    def test_missing_frame_geometry_is_unavailable(self, determine, factory):
        """A Hessian whose frame-matched geometry cannot be recovered must not ship.

        Emitting the triangle with the species' exported geometry instead is
        exactly the silent corruption ``frame`` exists to prevent.
        """
        relative = self._stage(GAUSSIAN_FREQ_LOG)
        parser = MagicMock()
        parser.parse_cartesian_hessian_lower_triangle.return_value = [0.0] * 21
        parser.parse_cartesian_hessian_geometry.return_value = (None, None)
        factory.return_value = parser
        envelope = _build_hessian(
            {"label": "no_frame", "freq_log": relative, "xyz": ATOM_LINES}, self.root
        )
        self.assertEqual(envelope["reason"], "hessian_frame_unavailable")

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
    def test_gsm_frames_coordinates_and_selected_point(self, kabsch_mock, trajectory, energies):
        self._touch("calcs/gsm/stringfile.xyz0000")
        envelope = _build_gsm({"label": "TS0", "gsm_log": "calcs/gsm/stringfile.xyz0000"}, self.root)
        value = envelope["value"]
        self.assertEqual(value["selected_source_point_index"], 2)
        self.assertEqual(value["parser_version"], "arc-gsm-stringfile-3")
        self.assertEqual(
            [point["cumulative_com_superposed_displacement_angstrom"] for point in value["points"]],
            [0.0, 0.25, 0.5],
        )
        self.assertEqual([point["stringfile_relative_energy_kcal_mol"] for point in value["points"]],
                         [0.0, 2.0, 0.0])
        self.assertEqual(value["points"][1]["geometry_xyz_text"].splitlines()[1], "gsm_point_1")
        self.assertFalse(value["points"][1]["geometry_xyz_text"].endswith("\n"))

    @patch("arc.tckdb_evidence.parse_gsm_stringfile_energies", return_value=[0.0, 0.0, 0.0])
    @patch("arc.tckdb_evidence.parse_trajectory")
    def test_gsm_cumulative_displacement_sums_consecutive_pairs_about_the_center_of_mass(
            self, trajectory, energies):
        """The coordinate sums consecutive frames and superposes at the mass center.

        Three water-shaped frames with the real ``kabsch``: an O at the origin
        and two H at ``(+-0.8, 0.6 s, 0)`` for ``s`` of 1, 1.5 and 2. Each
        frame's second-moment matrix is diagonal, so scaling ``y`` leaves the
        optimal rotation the identity and each pair's displacement is
        ``|ds| * L``, where ``L`` is the norm of the frames' ``y`` coordinates
        about the mass center. The two segments are therefore ``L/2`` each and
        the sum at the last frame is ``L``.

        Three assertions separate that from its two nearest wrong answers.
        Summing every frame against the *first* would give ``1.5 L``, which
        only shows up with three or more frames. Superposing about the
        geometric centroid instead of the center of mass would give
        ``0.4899`` per unit ``ds`` instead of ``0.7566``, which only shows up
        with unequal masses. The RMSD over the same pairs is ``L/sqrt(3)``.
        """
        def water(scale: float) -> dict:
            return {"symbols": ("O", "H", "H"), "isotopes": (16, 1, 1),
                    "coords": ((0.0, 0.0, 0.0), (0.8, 0.6 * scale, 0.0), (-0.8, 0.6 * scale, 0.0))}

        frames = [water(1.0), water(1.5), water(2.0)]
        trajectory.return_value = frames
        masses = get_element_mass_from_xyz(frames[0])
        ordinates = [coord[1] for coord in frames[0]["coords"]]
        mass_center = sum(mass * y for mass, y in zip(masses, ordinates)) / sum(masses)
        length = math.sqrt(sum((y - mass_center) ** 2 for y in ordinates))
        centroid = sum(ordinates) / len(ordinates)
        centroid_length = math.sqrt(sum((y - centroid) ** 2 for y in ordinates))

        self._touch("calcs/gsm/stringfile.xyz0000")
        points = _build_gsm(
            {"label": "TS0", "gsm_log": "calcs/gsm/stringfile.xyz0000"}, self.root
        )["value"]["points"]

        coordinates = [point["cumulative_com_superposed_displacement_angstrom"] for point in points]
        self.assertEqual(coordinates[0], 0.0)
        self.assertAlmostEqual(coordinates[1], 0.5 * length, places=10)
        self.assertAlmostEqual(coordinates[2], length, places=10)
        self.assertNotAlmostEqual(coordinates[2], 1.5 * length, places=3)
        self.assertNotAlmostEqual(coordinates[2], centroid_length, places=3)
        self.assertNotAlmostEqual(coordinates[2], length / math.sqrt(3.0), places=3)

    @patch("arc.tckdb_evidence.parse_gsm_stringfile_energies", return_value=[0.0, 2.0, 0.0])
    @patch("arc.tckdb_evidence.parse_trajectory", return_value=[XYZ_DICT, XYZ_DICT, XYZ_DICT])
    def test_gsm_node_outputs_are_reported_by_invocation_id(self, trajectory, energies):
        """Archived ``ograd`` results are listed verbatim under their own ids.

        The one gradient file here holds an H2 at 1.8 bohr while every frame is
        an H2 at 0.7 Angstrom, so no id matches a frame by content and no point
        is enriched. The ids are ``01``-``03`` against three frames -- the
        arrangement whose *count* matches, and which a frame-indexing parser
        would happily consume.
        """
        stringfile = self._touch("calcs/gsm/stringfile.xyz0000")
        outputs = stringfile.parent / "gsm_node_outputs"
        outputs.mkdir()
        (outputs / "0000.01.energy").write_text("$energy\n 1 -2.0 0 0\n$end\n")
        (outputs / "0000.01.xtbout").write_text("TOTAL ENERGY -9.0 Eh")
        (outputs / "0000.02.xtbout").write_text("TOTAL ENERGY -1.0 Eh")
        (outputs / "0000.03.energy").write_text("$energy\n 1 -3.0 0 0\n$end\n")
        (outputs / "0000.03.gradient").write_text(
            "$grad\n  cycle =      1    SCF energy =    -3.0\n"
            "   0.0  0.0  0.0   h\n   0.0  0.0  1.8   h\n"
            "   0.3  0.0  0.0\n  -0.4  0.0  0.0\n$end\n"
        )
        value = _build_gsm({"label": "TS0", "gsm_log": "calcs/gsm/stringfile.xyz0000"}, self.root)["value"]

        for point in value["points"]:
            self.assertEqual(set(point), {"source_point_index", "geometry_xyz_text",
                                          "cumulative_com_superposed_displacement_angstrom",
                                          "stringfile_relative_energy_kcal_mol"})
        self.assertEqual([item["invocation_id"] for item in value["ograd_invocations"]],
                         ["0000.01", "0000.02", "0000.03"])
        # The ``energy`` file wins over an ``xtbout`` log archived under the same id.
        self.assertEqual([item["electronic_energy_hartree"] for item in value["ograd_invocations"]],
                         [-2.0, -1.0, -3.0])
        self.assertAlmostEqual(value["ograd_invocations"][2]["max_gradient_hartree_per_bohr"], 0.4)
        self.assertAlmostEqual(value["ograd_invocations"][2]["rms_gradient_hartree_per_bohr"],
                               math.sqrt(0.25 / 6))

    @patch("arc.tckdb_evidence.parse_gsm_stringfile_energies", return_value=[0.0, 2.0, 0.0])
    @patch("arc.tckdb_evidence.parse_trajectory", return_value=[XYZ_DICT, XYZ_DICT, XYZ_DICT])
    @patch("arc.tckdb_evidence.kabsch", return_value=0.25)
    def test_gsm_invocations_without_a_gradient_file_reach_no_point(
            self, kabsch_mock, trajectory, energies):
        """An invocation that archived no ``gradient`` file has no geometry to verify, so it stays off the points."""
        stringfile = self._touch("calcs/gsm/stringfile.xyz0000")
        outputs = stringfile.parent / "gsm_node_outputs"
        outputs.mkdir()
        (outputs / "0000.00.energy").write_text("$energy\n 1 -7.0 0 0\n$end\n")
        (outputs / "0000.02.energy").write_text("$energy\n 1 -2.0 0 0\n$end\n")
        (outputs / "0000.05.energy").write_text("not an energy file")
        value = _build_gsm({"label": "TS0", "gsm_log": "calcs/gsm/stringfile.xyz0000"}, self.root)["value"]

        self.assertEqual([item["invocation_id"] for item in value["ograd_invocations"]],
                         ["0000.00", "0000.02"])
        for point in value["points"]:
            self.assertNotIn("electronic_energy_hartree", point)
            self.assertNotIn("geometry_matched_ograd_invocation_id", point)
            self.assertNotIn("node_label", point)

    @patch("arc.tckdb_evidence.parse_gsm_stringfile_energies", return_value=[0.0, 0.0])
    @patch("arc.tckdb_evidence.parse_trajectory")
    def test_gsm_energies_attach_to_the_frame_the_gradient_geometry_is(self, trajectory, energies):
        """A gradient file's own geometry, not its id, decides which point gets its energy.

        Two H2 frames at 0.7 and 0.9525 Angstrom; the single archived id
        ``0000.09`` holds the 1.8-bohr (0.9525 Angstrom) geometry, which is the
        *second* frame. Any arithmetic on the id would land elsewhere.
        """
        stretched = {"symbols": ("H", "H"), "isotopes": (1, 1),
                     "coords": ((0.0, 0.0, 0.0), (0.0, 0.0, 1.8 * bohr_to_angstrom))}
        trajectory.return_value = [XYZ_DICT, stretched]
        stringfile = self._touch("calcs/gsm/stringfile.xyz0000")
        outputs = stringfile.parent / "gsm_node_outputs"
        outputs.mkdir()
        (outputs / "0000.09.energy").write_text("$energy\n 1 -3.0 0 0\n$end\n")
        (outputs / "0000.09.gradient").write_text(
            "$grad\n  cycle =      1    SCF energy =    -3.0\n"
            "   0.0  0.0  0.0   h\n   0.0  0.0  1.8   h\n"
            "   0.3  0.0  0.0\n  -0.4  0.0  0.0\n$end\n"
        )
        points = _build_gsm(
            {"label": "TS0", "gsm_log": "calcs/gsm/stringfile.xyz0000"}, self.root
        )["value"]["points"]

        self.assertNotIn("geometry_matched_ograd_invocation_id", points[0])
        self.assertEqual(points[1]["geometry_matched_ograd_invocation_id"], "0000.09")
        self.assertEqual(points[1]["electronic_energy_hartree"], -3.0)
        self.assertAlmostEqual(points[1]["max_gradient_hartree_per_bohr"], 0.4)
        self.assertLess(points[1]["geometry_match_displacement_angstrom"], 1e-6)

    @patch("arc.tckdb_evidence.parse_gsm_stringfile_energies", return_value=[0.0, 0.0])
    @patch("arc.tckdb_evidence.parse_trajectory")
    def test_gsm_contested_and_multiply_matching_geometries_are_dropped(self, trajectory, energies):
        """A geometry that is two frames, and two invocations that are one frame, both attach nothing.

        ``0000.07`` and ``0000.08`` archive the same 1.8-bohr geometry, so the
        frame they both are is contested. ``0000.09`` archives the 0.7-Angstrom
        geometry, which the two identical first frames both are.
        """
        stretched = {"symbols": ("H", "H"), "isotopes": (1, 1),
                     "coords": ((0.0, 0.0, 0.0), (0.0, 0.0, 1.8 * bohr_to_angstrom))}
        trajectory.return_value = [XYZ_DICT, XYZ_DICT, stretched]
        energies.return_value = [0.0, 0.0, 0.0]
        stringfile = self._touch("calcs/gsm/stringfile.xyz0000")
        outputs = stringfile.parent / "gsm_node_outputs"
        outputs.mkdir()
        for invocation_id, z_coordinate in (("0000.07", "1.8"), ("0000.08", "1.8"), ("0000.09", "1.32280829")):
            (outputs / f"{invocation_id}.energy").write_text("$energy\n 1 -3.0 0 0\n$end\n")
            (outputs / f"{invocation_id}.gradient").write_text(
                f"$grad\n  cycle =      1    SCF energy =    -3.0\n"
                f"   0.0  0.0  0.0   h\n   0.0  0.0  {z_coordinate}   h\n"
                f"   0.3  0.0  0.0\n  -0.4  0.0  0.0\n$end\n"
            )
        value = _build_gsm({"label": "TS0", "gsm_log": "calcs/gsm/stringfile.xyz0000"}, self.root)["value"]

        self.assertEqual([item["invocation_id"] for item in value["ograd_invocations"]],
                         ["0000.07", "0000.08", "0000.09"])
        for point in value["points"]:
            self.assertNotIn("geometry_matched_ograd_invocation_id", point)
            self.assertNotIn("electronic_energy_hartree", point)

    @patch("arc.tckdb_evidence.parse_gsm_stringfile_energies", return_value=[0.0, 2.0, 0.0])
    @patch("arc.tckdb_evidence.parse_trajectory", return_value=[XYZ_DICT, XYZ_DICT, XYZ_DICT])
    @patch("arc.tckdb_evidence.kabsch", return_value=0.25)
    def test_gsm_invocations_sort_numerically_and_non_ids_are_skipped(
            self, kabsch_mock, trajectory, energies):
        """Slots order by number, not by string, and a file whose name is not an id is not an invocation."""
        stringfile = self._touch("calcs/gsm/stringfile.xyz0000")
        outputs = stringfile.parent / "gsm_node_outputs"
        outputs.mkdir()
        for name in ("0000.9", "0000.11", "0000.100", "notes"):
            (outputs / f"{name}.energy").write_text("$energy\n 1 -1.0 0 0\n$end\n")
        (outputs / "notes.xtbout").write_text("TOTAL ENERGY -1.0 Eh")
        value = _build_gsm({"label": "TS0", "gsm_log": "calcs/gsm/stringfile.xyz0000"}, self.root)["value"]

        self.assertEqual([item["invocation_id"] for item in value["ograd_invocations"]],
                         ["0000.9", "0000.11", "0000.100"])

    @patch("arc.tckdb_evidence.parse_gsm_stringfile_energies", return_value=[0.0, 0.0, 0.0])
    @patch("arc.tckdb_evidence.parse_trajectory", return_value=[XYZ_DICT, XYZ_DICT, XYZ_DICT])
    @patch("arc.tckdb_evidence.kabsch", return_value=0.25)
    def test_gsm_without_archived_node_outputs_omits_the_invocation_list(
            self, kabsch_mock, trajectory, energies):
        self._touch("calcs/gsm/stringfile.xyz0000")
        value = _build_gsm({"label": "TS0", "gsm_log": "calcs/gsm/stringfile.xyz0000"}, self.root)["value"]
        self.assertNotIn("ograd_invocations", value)

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

    @patch("arc.tckdb_evidence._build_gsm")
    def test_geometry_winner_with_merged_gsm_source_emits_gsm_evidence(self, gsm):
        """Sanitized chosen-guess sources retain path-search attribution."""
        available = {"status": "available", "value": {"method": "gsm"}}
        gsm.return_value = available
        source_record = {
            "label": "TS_merged",
            "chosen_ts_method": "gcn",
            "gsm_log": "calcs/gsm/stringfile.xyz0000",
            "irc_logs": [],
            "ts_guesses": [{
                "index": 7,
                "chosen": True,
                "method": "gcn",
                "method_sources": ["gcn", "xtb-gsm"],
            }],
        }
        output = {
            "schema_version": "1.1", "arc_version": "x", "arc_git_commit": "abc",
            "species": [], "transition_states": [source_record],
        }

        record = build_tckdb_evidence(
            output_doc=output, project_directory=self.root, document_id=DOC_ID,
        )["records"][0]

        self.assertEqual(record["gsm"], available)
        gsm.assert_called_once_with(source_record, self.root)

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

    def test_atomic_write_fsyncs_the_parent_directory(self):
        """The rename is only durable once the containing directory is synced."""
        document = {
            "schema_name": EVIDENCE_SCHEMA_NAME, "schema_version": "1.0", "document_id": DOC_ID,
            "output_schema_version": "1.1", "producer": {"name": "ARC", "version": "x", "git_commit": None},
            "records": [],
        }
        synced = []
        real_fsync = os.fsync

        def _recording_fsync(descriptor):
            synced.append(os.fstat(descriptor).st_ino)
            return real_fsync(descriptor)

        with patch("arc.tckdb_evidence.os.fsync", side_effect=_recording_fsync):
            write_tckdb_evidence_atomic(evidence_doc=document, output_directory=self.root)
        self.assertIn(os.stat(self.root).st_ino, synced)

    def test_atomic_write_survives_a_platform_that_cannot_fsync_directories(self):
        """A directory sync that is unsupported must not fail a completed write."""
        document = {
            "schema_name": EVIDENCE_SCHEMA_NAME, "schema_version": "1.0", "document_id": DOC_ID,
            "output_schema_version": "1.1", "producer": {"name": "ARC", "version": "x", "git_commit": None},
            "records": [],
        }
        real_open = os.open

        def _refuse_directories(path, flags, *args, **kwargs):
            if os.path.isdir(path):
                raise PermissionError("no directory handles")
            return real_open(path, flags, *args, **kwargs)

        with patch("arc.tckdb_evidence.os.open", side_effect=_refuse_directories):
            path = write_tckdb_evidence_atomic(evidence_doc=document, output_directory=self.root)
        self.assertEqual(json.loads(path.read_text()), document)

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

    # Real ESS fixtures staged at the run-relative paths the golden output.yml
    # names. Every builder runs against a real file; nothing here is stubbed.
    # The GSM pair -- stringfile and archived node outputs -- comes from one
    # real completed xtb_gsm run, so the two are mutually consistent and the
    # geometry match runs against files a run actually produced.
    GOLDEN_STAGE = {
        "calcs/Species/H2O/freq/output.out": (ARC_TESTING_PATH, "freq", "orca_hessian_h2o", "output.out"),
        "calcs/Species/H2O/freq/input.hess": (ARC_TESTING_PATH, "freq", "orca_hessian_h2o", "input.hess"),
        "calcs/TSs/TS0/irc/irc_f/output.out": (ARC_TESTING_PATH, "irc", "rxn_1_irc_1.out"),
        "calcs/TSs/TS0/irc/irc_r/output.out": (ARC_TESTING_PATH, "irc", "rxn_1_irc_2.out"),
        "calcs/TSs/TS0/gsm/stringfile.xyz0000": (ARC_TESTING_PATH, "tckdb_evidence", "gsm",
                                                 "stringfile.xyz0000"),
    }
    GOLDEN_STAGE_TREES = {
        "calcs/TSs/TS0/gsm/gsm_node_outputs": (ARC_TESTING_PATH, "tckdb_evidence", "gsm", "gsm_node_outputs"),
    }

    def _stage_golden(self):
        """Copy every golden fixture into the temp project at the path the golden output.yml names."""
        for relative, source in self.GOLDEN_STAGE.items():
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(os.path.join(*source), destination)
        for relative, source in self.GOLDEN_STAGE_TREES.items():
            shutil.copytree(os.path.join(*source), self.root / relative)

    def test_gsm_energies_attach_to_real_frames_by_geometry_on_a_real_run(self):
        """On a real xtb_gsm run the archived geometries pick their own frames, and the ids do not.

        Slot ``0000.01`` holds the geometry of the *last* of the 15 frames --
        GSM's last overwrite of that id -- so reading the id as a frame index
        would put the run's final energy on its first point. ``0000.04``
        archived no gradient file and so reaches no point at all. Every comment
        line in this real stringfile carries ``0.000000``, so the attachment is
        the points' only energy.
        """
        self._stage_golden()
        value = _build_gsm(
            {"label": "TS0", "gsm_log": "calcs/TSs/TS0/gsm/stringfile.xyz0000"}, self.root
        )["value"]
        points = value["points"]

        self.assertEqual(len(points), 15)
        self.assertEqual([item["invocation_id"] for item in value["ograd_invocations"]],
                         ["0000.01", "0000.02", "0000.03", "0000.04"])
        matched = {point["source_point_index"]: point["geometry_matched_ograd_invocation_id"]
                   for point in points if "geometry_matched_ograd_invocation_id" in point}
        self.assertEqual(matched, {14: "0000.01", 1: "0000.02", 2: "0000.03"})
        self.assertNotIn("electronic_energy_hartree", points[0])
        self.assertAlmostEqual(points[14]["electronic_energy_hartree"], -9.89904684440)
        self.assertLess(points[14]["geometry_match_displacement_angstrom"], 1e-4)
        self.assertIn("max_gradient_hartree_per_bohr", points[14])
        self.assertEqual({point["stringfile_relative_energy_kcal_mol"] for point in points}, {0.0})

    def test_parse_gradient_max_and_rms(self):
        """``_parse_gradient`` returns the max |component| and the RMS over all 3N components.

        All other gradient coverage reaches this through a stubbed
        ``parse_irc_path``, so the arithmetic here — in particular the ``/ N``
        RMS normalization, whose omission inflates every reported RMS by sqrt(N)
        — was previously unexercised.
        """
        path = self.root / "gradient"
        # Two atoms; components chosen so max and RMS are exact and distinct.
        path.write_text(
            "$grad          cartesian gradients\n"
            "  cycle =      1    SCF energy =    -1.0   |dE/dxyz| = 0.1\n"
            "   0.0  0.0  0.0   o\n"
            "   0.0  0.0  1.8   h\n"
            "   3.0D-01  0.0  0.0\n"
            "  -0.4  0.0  0.0\n"
            "$end\n",
            encoding="utf-8",
        )
        max_gradient, rms = _parse_gradient(path)
        self.assertAlmostEqual(max_gradient, 0.4)
        # sqrt((0.3^2 + 0.4^2) / 6) -- normalized over all six components.
        self.assertAlmostEqual(rms, math.sqrt(0.25 / 6))

    def test_parse_gradient_uses_the_final_cycle(self):
        """xTB appends cycles; only the last block describes the reported geometry."""
        path = self.root / "gradient"
        path.write_text(
            "$grad          cartesian gradients\n"
            "  cycle =      1    SCF energy =    -1.0   |dE/dxyz| = 0.9\n"
            "   0.0  0.0  0.0   o\n"
            "   0.0  0.0  1.8   h\n"
            "   9.0  0.0  0.0\n"
            "   9.0  0.0  0.0\n"
            "  cycle =      2    SCF energy =    -1.1   |dE/dxyz| = 0.1\n"
            "   0.0  0.0  0.0   o\n"
            "   0.0  0.0  1.8   h\n"
            "   0.1  0.0  0.0\n"
            "   0.0  0.0  0.0\n"
            "$end\n",
            encoding="utf-8",
        )
        max_gradient, _ = _parse_gradient(path)
        self.assertAlmostEqual(max_gradient, 0.1)

    def test_parse_gradient_malformed_returns_none(self):
        """Unreadable or malformed gradient files degrade, never raise."""
        missing = self.root / "does_not_exist"
        self.assertEqual(_parse_gradient(missing), (None, None))
        truncated = self.root / "gradient_truncated"
        truncated.write_text("$grad\n  cycle = 1\n   0.0 0.0 0.0   o\n$end\n", encoding="utf-8")
        self.assertEqual(_parse_gradient(truncated), (None, None))
        no_cycle = self.root / "gradient_no_cycle"
        no_cycle.write_text("$grad\n   0.1 0.2 0.3\n$end\n", encoding="utf-8")
        self.assertEqual(_parse_gradient(no_cycle), (None, None))

    def _assert_matches_golden(self, actual, expected, path="") -> None:
        """Assert ``actual`` equals ``expected``, comparing floats within tolerance.

        Structure, key sets, types, string values and integers must match
        exactly. Floats are compared with ``math.isclose`` because some values
        in the document reach the sidecar through ``kabsch``, whose SVD is
        performed by the platform's LAPACK — near-zero geometry-match residuals
        of order 1e-5 differ in their low bits between a developer machine and
        CI, which an exact comparison reports as a contract violation.
        """
        self.assertEqual(type(actual), type(expected), f"type differs at {path or 'root'}")
        if isinstance(expected, Mapping):
            self.assertEqual(sorted(actual), sorted(expected), f"key set differs at {path or 'root'}")
            for key in expected:
                self._assert_matches_golden(actual[key], expected[key], f"{path}/{key}")
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected), f"length differs at {path}")
            for index, item in enumerate(expected):
                self._assert_matches_golden(actual[index], item, f"{path}[{index}]")
        elif isinstance(expected, float):
            self.assertTrue(
                math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12),
                f"{path}: {actual!r} is not close to {expected!r}",
            )
        else:
            self.assertEqual(actual, expected, f"value differs at {path}")

    def test_golden_contract(self):
        """The assembled sidecar reproduces the committed golden document.

        This runs in CI unconditionally and exercises ``_build_hessian``,
        ``_build_irc`` and ``_build_gsm`` for real against real ESS logs. An
        earlier version of this test skipped unless a sibling consumer checkout
        happened to be on disk, and stubbed those three builders with the very
        values it then asserted — so it could not fail on a defect in any of
        them, including a wrong Hessian frame.
        """
        golden_dir = Path(ARC_TESTING_PATH) / "tckdb_evidence" / "golden"
        output = yaml.safe_load((golden_dir / "output.yml").read_text())
        expected = json.loads((golden_dir / "tckdb_evidence.json").read_text())
        self._stage_golden()

        actual = build_tckdb_evidence(
            output_doc=output, project_directory=self.root, document_id=DOC_ID
        )
        self._assert_matches_golden(actual, expected)

    def test_golden_records_are_all_available(self):
        """Guard the golden itself: a fixture of ``unavailable`` envelopes would assert nothing.

        Without this, a regression that made every builder degrade would be
        silently absorbed by regenerating the golden.
        """
        expected = json.loads(
            (Path(ARC_TESTING_PATH) / "tckdb_evidence" / "golden" / "tckdb_evidence.json").read_text()
        )
        kinds = set()
        attached = 0
        for record in expected["records"]:
            for kind in ("freq_hessian", "irc", "gsm"):
                if kind in record:
                    self.assertEqual(record[kind]["status"], "available", f"{record['label']}/{kind}")
                    kinds.add(kind)
            if "gsm" in record:
                attached += sum("geometry_matched_ograd_invocation_id" in point
                                for point in record["gsm"]["value"]["points"])
        self.assertEqual(kinds, {"freq_hessian", "irc", "gsm"})
        self.assertEqual(attached, 3)


if __name__ == "__main__":
    unittest.main()
