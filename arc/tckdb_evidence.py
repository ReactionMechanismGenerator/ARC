"""Produce parser-neutral scientific evidence alongside ARC's output.yml."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from arc.common import get_logger
from arc.parser.factory import ess_factory
from arc.parser.parser import (
    determine_ess,
    parse_gsm_stringfile_energies,
    parse_irc_path,
    parse_irc_traj,
    parse_trajectory,
)
from arc.species.converter import kabsch, xyz_to_str


logger = get_logger()

EVIDENCE_SCHEMA_NAME = "arc-tckdb-evidence"
EVIDENCE_SCHEMA_VERSION = "1.0"
HESSIAN_PARSER_VERSION = "arc-hessian-1"
IRC_PARSER_VERSION = "arc-irc-path-1"
GSM_PARSER_VERSION = "arc-gsm-stringfile-1"
EVIDENCE_FILENAME = "tckdb_evidence.json"

_XTBOUT_TOTAL_ENERGY_RE = re.compile(
    r"total\s+energy\s+(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)\s*Eh", re.IGNORECASE,
)


def _run_relative(path: str | Path, project_directory: str | Path) -> str:
    return os.path.relpath(str(path), str(project_directory))


def _resolve(path: str | Path, project_directory: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(project_directory) / candidate


def _canonical_xyz(xyz: Mapping[str, Any] | str, label: str) -> str:
    atom_lines = xyz if isinstance(xyz, str) else xyz_to_str(dict(xyz))
    if not atom_lines:
        raise ValueError("empty geometry")
    lines = str(atom_lines).strip().splitlines()
    if lines and lines[0].strip().isdigit():
        count = int(lines[0])
        if len(lines) != count + 2:
            raise ValueError("malformed canonical XYZ")
        return "\n".join(lines) + "\n"
    return f"{len(lines)}\n{label}\n" + "\n".join(lines) + "\n"


def _unavailable(reason: str, paths: list[str] | None = None) -> dict:
    envelope = {"status": "unavailable", "reason": reason}
    if paths:
        envelope["source_paths"] = paths
    return envelope


def _finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("evidence contains a non-finite float")
    if isinstance(value, Mapping):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_json(item)


def _build_hessian(record: Mapping[str, Any], project_directory: str | Path) -> dict:
    source_log = str(record.get("freq_log") or "")
    paths = [source_log] if source_log else None
    if not source_log:
        return _unavailable("missing_source")
    resolved = _resolve(source_log, project_directory)
    if not resolved.is_file():
        return _unavailable("missing_source", paths)
    try:
        geometry = _canonical_xyz(record.get("xyz") or "", f"{record.get('label')} freq")
        atom_count = int(geometry.splitlines()[0])
        if atom_count <= 1:
            return _unavailable("unsupported_source", paths)
        ess_name = determine_ess(str(resolved), raise_error=False)
        if ess_name not in {"gaussian", "orca"}:
            return _unavailable("unsupported_source", paths)
        parser = ess_factory(str(resolved), ess_name)
        parse = getattr(parser, "parse_cartesian_hessian_lower_triangle", None)
        triangle = parse() if parse else None
        dimension = 3 * atom_count
        if not triangle:
            return _unavailable("empty_result", paths)
        triangle = [float(value) for value in triangle]
        if len(triangle) != dimension * (dimension + 1) // 2:
            return _unavailable("parse_failed", paths)
        value = {
            "source_log": source_log,
            "geometry_xyz_text": geometry,
            "atom_count": atom_count,
            "matrix_dimension": dimension,
            "packing": "lower_triangle_row_major_including_diagonal",
            "units": "hartree_per_bohr_squared",
            "source": "parsed_hess" if ess_name == "orca" else "parsed_log",
            "parser_version": HESSIAN_PARSER_VERSION,
            "lower_triangle": triangle,
        }
        _finite_json(value)
        return {"status": "available", "value": value}
    except Exception:
        logger.debug("TCKDB evidence Hessian parse failed for %s", record.get("label"), exc_info=True)
        return _unavailable("parse_failed", paths)


def _direction(log_path: str, declared: Any) -> str | None:
    if declared in {"forward", "reverse"}:
        return declared
    name = Path(log_path).name.lower()
    if "irc_f" in name or "forward" in name:
        return "forward"
    if "irc_r" in name or "reverse" in name:
        return "reverse"
    return None


def _build_irc(record: Mapping[str, Any], project_directory: str | Path) -> dict:
    logs = list(record.get("irc_logs") or [])
    directions = list(record.get("irc_log_directions") or [])
    trajectories = []
    omitted = []
    for position, log_path in enumerate(logs):
        source_log = str(log_path)
        resolved = _resolve(source_log, project_directory)
        declared = _direction(source_log, directions[position] if position < len(directions) else None)
        if not resolved.is_file():
            omitted.append(source_log)
            continue
        points = []
        try:
            rich = parse_irc_path(log_file_path=str(resolved))
        except Exception:
            rich = None
        if rich:
            for fallback_index, point in enumerate(rich):
                xyz = point.get("xyz")
                if xyz is None:
                    continue
                item = {
                    "source_point_index": int(point.get("point_number", fallback_index)),
                    "direction": point.get("direction") or declared,
                    "geometry_xyz_text": _canonical_xyz(xyz, f"{record.get('label')} IRC point"),
                }
                for source_key, evidence_key in (
                    ("electronic_energy_hartree", "electronic_energy_hartree"),
                    ("reaction_coordinate", "reaction_coordinate_sqrt_amu_bohr"),
                    ("max_gradient", "max_gradient_hartree_per_bohr"),
                    ("rms_gradient", "rms_gradient_hartree_per_bohr"),
                ):
                    if point.get(source_key) is not None:
                        item[evidence_key] = float(point[source_key])
                points.append(item)
        else:
            try:
                geometries = parse_irc_traj(log_file_path=str(resolved)) or []
            except Exception:
                geometries = []
            for index, xyz in enumerate(geometries):
                points.append({
                    "source_point_index": index,
                    "direction": declared,
                    "geometry_xyz_text": _canonical_xyz(xyz, f"{record.get('label')} IRC point {index}"),
                })
        if points:
            trajectories.append({"source_log": source_log, "declared_direction": declared, "points": points})
        else:
            omitted.append(source_log)
    if not trajectories:
        return _unavailable("empty_result" if logs else "missing_source", [str(p) for p in logs] or None)
    value = {"parser_version": IRC_PARSER_VERSION, "trajectories": trajectories}
    if omitted:
        value["omitted_source_paths"] = omitted
    _finite_json(value)
    return {"status": "available", "value": value}


def _parse_energy(path: Path) -> float | None:
    try:
        return float(path.read_text().splitlines()[-2].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _parse_gradient(path: Path) -> tuple[float | None, float | None]:
    try:
        lines = path.read_text().splitlines(True)
        atom_count = int((len(lines) - 3) / 2)
        components = [
            float(token.replace("D", "E"))
            for line in lines[2 + atom_count:2 + 2 * atom_count]
            for token in line.split()[:3]
        ]
        return max(abs(x) for x in components), math.sqrt(sum(x * x for x in components) / len(components))
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        return None, None


def _parse_xtbout(path: Path) -> float | None:
    try:
        matches = _XTBOUT_TOTAL_ENERGY_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
        return float(matches[-1].replace("D", "E").replace("d", "e")) if matches else None
    except (OSError, ValueError):
        return None


def _node_outputs(directory: Path) -> dict[int, dict[str, float]]:
    result = {}
    if not directory.is_dir():
        return result
    label_of = lambda path: int(path.stem.split(".")[-1])
    for path in sorted(directory.glob("*.energy")):
        try:
            label = label_of(path)
        except (ValueError, IndexError):
            continue
        energy = _parse_energy(path)
        if energy is None:
            continue
        item = {"electronic_energy_hartree": energy}
        max_g, rms_g = _parse_gradient(path.with_suffix(".gradient"))
        if max_g is not None:
            item["max_gradient_hartree_per_bohr"] = max_g
        if rms_g is not None:
            item["rms_gradient_hartree_per_bohr"] = rms_g
        result[label] = item
    for path in sorted(directory.glob("*.xtbout")):
        try:
            label = label_of(path)
        except (ValueError, IndexError):
            continue
        if label not in result:
            energy = _parse_xtbout(path)
            if energy is not None:
                result[label] = {"electronic_energy_hartree": energy}
    return result


def _path_search_method(record: Mapping[str, Any]) -> str | None:
    aliases = {"xtb_gsm": "gsm", "xtb-gsm": "gsm", "gsm": "gsm", "orca_neb": "neb", "orca-neb": "neb"}
    method = record.get("chosen_ts_method")
    if isinstance(method, str) and method.strip().lower() in aliases:
        return aliases[method.strip().lower()]
    candidates = []
    for guess in record.get("ts_guesses") or []:
        if isinstance(guess, Mapping) and guess.get("chosen"):
            for source in guess.get("method_sources") or []:
                if isinstance(source, str) and source.strip().lower() in aliases:
                    candidate = aliases[source.strip().lower()]
                    if candidate not in candidates:
                        candidates.append(candidate)
            break
    for candidate in candidates:
        if record.get("gsm_log" if candidate == "gsm" else "neb_log"):
            return candidate
    return candidates[0] if candidates else None


def _build_gsm(record: Mapping[str, Any], project_directory: str | Path) -> dict:
    source = str(record.get("gsm_log") or "")
    if not source:
        return _unavailable("missing_source")
    path = _resolve(source, project_directory)
    if not path.is_file():
        return _unavailable("missing_source", [source])
    try:
        frames = parse_trajectory(str(path)) or []
        if not frames:
            return _unavailable("empty_result", [source])
        selected = int((len(frames) - 1) / 2) + 1
        if selected >= len(frames):
            return _unavailable("parse_failed", [source])
        coordinates = None
        if len(frames) >= 2:
            try:
                coordinates = [0.0]
                for index in range(1, len(frames)):
                    coordinates.append(float(coordinates[-1] + kabsch(frames[index], frames[index - 1])))
            except Exception:
                coordinates = None
        relative = parse_gsm_stringfile_energies(str(path))
        metadata = _node_outputs(path.parent / "gsm_node_outputs")
        points = []
        for index, frame in enumerate(frames):
            point = {
                "source_point_index": index,
                "node_label": index if index else None,
                "geometry_xyz_text": _canonical_xyz(frame, f"{record.get('label')} GSM point {index}"),
            }
            if coordinates is not None:
                point["path_coordinate_angstrom"] = coordinates[index]
            if relative is not None and index < len(relative):
                point["stringfile_relative_energy_kcal_mol"] = float(relative[index])
            point.update(metadata.get(index, {}))
            points.append(point)
        value = {
            "source_stringfile": source,
            "parser_version": GSM_PARSER_VERSION,
            "method": "gsm",
            "selected_source_point_index": selected,
            "points": points,
        }
        _finite_json(value)
        return {"status": "available", "value": value}
    except Exception:
        logger.debug("TCKDB evidence GSM parse failed for %s", record.get("label"), exc_info=True)
        return _unavailable("parse_failed", [source])


def build_tckdb_evidence(*, output_doc: Mapping[str, Any], project_directory: str | Path, document_id: str) -> dict:
    """Build one complete evidence document, isolating each optional entry."""
    records = []
    for section, record_kind in (("species", "species"), ("transition_states", "transition_state")):
        for source_record in output_doc.get(section) or []:
            record = {"record_kind": record_kind, "label": source_record["label"]}
            if source_record.get("freq_log"):
                record["freq_hessian"] = _build_hessian(source_record, project_directory)
            if record_kind == "transition_state" and source_record.get("irc_logs"):
                record["irc"] = _build_irc(source_record, project_directory)
            if record_kind == "transition_state" and _path_search_method(source_record) == "gsm" and source_record.get("gsm_log"):
                record["gsm"] = _build_gsm(source_record, project_directory)
            for evidence_kind in ("freq_hessian", "irc", "gsm"):
                envelope = record.get(evidence_kind)
                if envelope and envelope.get("status") == "unavailable":
                    logger.debug(
                        "TCKDB evidence unavailable kind=%s label=%s reason=%s source_paths=%s",
                        evidence_kind,
                        source_record.get("label"),
                        envelope.get("reason"),
                        envelope.get("source_paths") or [],
                    )
            if len(record) > 2:
                records.append(record)
    evidence = {
        "schema_name": EVIDENCE_SCHEMA_NAME,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "document_id": document_id,
        "output_schema_version": "1.1",
        "producer": {
            "name": "ARC",
            "version": output_doc.get("arc_version"),
            "git_commit": output_doc.get("arc_git_commit"),
        },
        "records": records,
    }
    _finite_json(evidence)
    return evidence


def write_tckdb_evidence_atomic(*, evidence_doc: Mapping[str, Any], output_directory: str | Path) -> Path:
    """Write deterministic strict JSON, fsync, and atomically replace the sidecar."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output_directory, suffix=".tckdb_evidence.json.tmp")
    target = output_directory / EVIDENCE_FILENAME
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(evidence_doc, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            logger.debug("Failed to remove temporary evidence file %s", temporary, exc_info=True)
        raise
    return target
