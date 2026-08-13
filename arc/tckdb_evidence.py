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
from arc.constants import bohr_to_angstrom
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
EVIDENCE_SCHEMA_VERSION = "1.1"
# The output.yml contract version this evidence schema is bound to. Defined
# here rather than in arc/output.py because output.py imports this module (the
# reverse would be circular), and stated once so the sidecar's
# ``output_schema_version`` cannot drift from the value output.yml actually
# carries -- a consumer gating on it would otherwise mis-gate after a bump.
OUTPUT_SCHEMA_VERSION = "1.1"
# Bumped from ``arc-hessian-1``: ``geometry_xyz_text`` now carries the geometry
# in the Hessian's own Cartesian frame (named by the new ``frame`` field)
# rather than the species record's exported geometry, which for Gaussian was a
# rotation away and reconstructed the wrong spectrum. A consumer must not treat
# ``arc-hessian-1`` records as frame-consistent.
HESSIAN_PARSER_VERSION = "arc-hessian-2"
IRC_PARSER_VERSION = "arc-irc-path-1"
# Bumped from ``arc-gsm-stringfile-1`` (and again from ``arc-gsm-stringfile-2``)
# for three incompatible changes to the GSM record.
#
# 1. ``path_coordinate_angstrom`` is renamed
#    ``cumulative_com_superposed_displacement_angstrom``. The value is
#    unchanged -- ``kabsch`` returns SciPy's RSSD, the Frobenius norm of the
#    superposed coordinate difference (``sqrt(N_atoms)`` times the RMSD), so
#    the running sum was already a Cartesian path length -- but the old name
#    invited reading it as a reaction coordinate commensurable with the IRC
#    record's mass-weighted ``reaction_coordinate_sqrt_amu_bohr``, which it is
#    not. The name also has to say *which* superposition: ``kabsch`` translates
#    each geometry to its center of mass before rotating, so the translation is
#    constrained to align mass centers rather than freely optimized, the value
#    is an upper bound on the minimal superposed displacement (6% high on the
#    golden's final point), and it moves under isotopic substitution even
#    though the norm itself is unweighted.
# 2. Per-node energies and gradient norms no longer ride on the points by id
#    arithmetic, and ``node_label`` is gone. GSM invokes ``ograd <run>.<slot>
#    <ncpu>`` and the wrapper archives each result under that id; ``slot`` is
#    ``runend`` (1) plus the ICoord array index, so it is one greater than any
#    frame index it could describe, ICoords that are not string nodes use
#    ``runend - 1`` and ``runend`` itself, and ``cp -p`` overwrites, leaving one
#    file per id holding whichever ICoord evaluated last. On a real 15-frame run
#    this puts the *last* frame under id ``0000.01``, which id arithmetic reads
#    as the first.
# 3. The energies are nevertheless recoverable, because the archived Turbomole
#    ``gradient`` file carries the geometry the gradient was evaluated at. That
#    geometry superposes onto its stringfile frame to ~1e-5 Angstrom while the
#    nearest other frame is ~0.3 Angstrom away, so a frame is identified by
#    content instead of by id, and the attachment is stamped on the point with
#    the id and the measured displacement. This matters because
#    ``stringfile_relative_energy_kcal_mol`` is identically zero in every real
#    xTB/GSM run -- the ORCA-format writer emits ``0.000000`` into every comment
#    line -- so without the attachment the points carry no energy at all.
#
# A consumer must not read ``arc-gsm-stringfile-1`` per-point energies or
# gradient norms as belonging to the point that carries them, and must not read
# an ``arc-gsm-stringfile-2`` cumulative coordinate as a freely superposed
# displacement.
GSM_PARSER_VERSION = "arc-gsm-stringfile-3"
EVIDENCE_FILENAME = "tckdb_evidence.json"
# The largest superposed displacement, in Angstrom, at which an archived
# gradient geometry is accepted as being a stringfile frame. Observed matches
# on a real run are ~1e-5 Angstrom and the nearest non-matching frame is
# ~0.3 Angstrom away, so the threshold is four orders of magnitude clear of
# both, absorbing the stringfile's six-decimal printing without admitting a
# neighbouring frame.
GRADIENT_MATCH_TOLERANCE_ANGSTROM = 1e-3

_XTBOUT_TOTAL_ENERGY_RE = re.compile(
    r"total\s+energy\s+(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)\s*Eh", re.IGNORECASE,
)
_INVOCATION_ID_RE = re.compile(r"^\d+\.\d+$")


def _resolve(path: str | Path, project_directory: str | Path) -> Path:
    """Return ``path`` as an absolute path, resolving run-relative paths against the project."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(project_directory) / candidate


def _canonical_xyz(xyz: Mapping[str, Any] | str, label: str) -> str:
    """Return ``xyz`` as canonical XYZ text: atom count, comment line, then atom lines.

    Accepts either an ARC xyz dict or bare atom-line text. Text that is already
    canonical is passed through unchanged; bare atom lines get a count and
    ``label`` as the comment. Raises ``ValueError`` on empty or malformed input
    rather than emitting a geometry a consumer cannot parse.
    """
    atom_lines = xyz if isinstance(xyz, str) else xyz_to_str(dict(xyz))
    if not atom_lines:
        raise ValueError("empty geometry")
    lines = str(atom_lines).strip().splitlines()
    if lines and lines[0].strip().isdigit():
        count = int(lines[0])
        if len(lines) != count + 2:
            raise ValueError("malformed canonical XYZ")
        return "\n".join(lines)
    return f"{len(lines)}\n{label}\n" + "\n".join(lines)


def _unavailable(reason: str, paths: list[str] | None = None) -> dict:
    """Return the ``unavailable`` envelope carrying ``reason`` and any source paths.

    Every builder degrades to this shape instead of raising or omitting the key,
    so a consumer can always distinguish "not attempted" from "attempted and
    failed, for this reason".
    """
    envelope = {"status": "unavailable", "reason": reason}
    if paths:
        envelope["source_paths"] = paths
    return envelope


def _finite_json(value: Any) -> None:
    """Raise ``ValueError`` if ``value`` contains a non-finite float, recursively.

    ``NaN`` and ``Infinity`` are not valid JSON, and ``json.dump`` emits them as
    bare literals that strict parsers reject. Checking before writing keeps a
    single bad parse from producing an unreadable sidecar.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("evidence contains a non-finite float")
    if isinstance(value, Mapping):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_json(item)


def _build_hessian(record: Mapping[str, Any], project_directory: str | Path) -> dict:
    """Build the Cartesian Hessian evidence record for one species.

    The geometry is read from the Hessian's own source rather than from the
    species record's exported ``xyz``. A Hessian is only meaningful alongside
    the geometry it was evaluated at, in the same frame: Gaussian prints force
    constants in the input orientation while ARC exports the standard
    orientation, a pure rotation apart. Pairing the two silently reconstructs a
    different spectrum -- invented low-frequency modes and errors of hundreds of
    wavenumbers -- and no size or finiteness check can detect it, because a
    rotation changes neither. ``frame`` names the frame both fields share.

    Returns an ``unavailable`` envelope rather than raising for every failure
    mode, including a Hessian whose frame-matched geometry cannot be recovered.
    """
    source_log = str(record.get("freq_log") or "")
    paths = [source_log] if source_log else None
    if not source_log:
        return _unavailable("missing_source")
    resolved = _resolve(source_log, project_directory)
    if not resolved.is_file():
        return _unavailable("missing_source", paths)
    try:
        ess_name = determine_ess(str(resolved), raise_error=False)
        if ess_name not in {"gaussian", "orca"}:
            return _unavailable("unsupported_source", paths)
        parser = ess_factory(str(resolved), ess_name)
        parse = getattr(parser, "parse_cartesian_hessian_lower_triangle", None)
        triangle = parse() if parse else None
        if not triangle:
            return _unavailable("empty_result", paths)
        parse_frame = getattr(parser, "parse_cartesian_hessian_geometry", None)
        frame_xyz, frame = parse_frame() if parse_frame else (None, None)
        if frame_xyz is None or frame is None:
            return _unavailable("hessian_frame_unavailable", paths)
        geometry = _canonical_xyz(xyz_to_str(frame_xyz), str(record.get("label") or ""))
        atom_count = int(geometry.splitlines()[0])
        if atom_count <= 1:
            return _unavailable("unsupported_source", paths)
        dimension = 3 * atom_count
        triangle = [float(value) for value in triangle]
        if len(triangle) != dimension * (dimension + 1) // 2:
            return _unavailable("parse_failed", paths)
        value = {
            "source_log": source_log,
            "geometry_xyz_text": geometry,
            "frame": frame,
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
    """Return the IRC branch direction, preferring the value ARC recorded.

    Falls back to the log file name (ARC names IRC jobs ``irc_f``/``irc_r``)
    when the run carried no explicit direction, and ``None`` when neither is
    conclusive -- an unlabeled branch is better than a guessed one.
    """
    if declared in {"forward", "reverse"}:
        return declared
    name = Path(log_path).name.lower()
    if "irc_f" in name or "forward" in name:
        return "forward"
    if "irc_r" in name or "reverse" in name:
        return "reverse"
    return None


def _build_irc(record: Mapping[str, Any], project_directory: str | Path) -> dict:
    """Build the IRC evidence record: one trajectory per reachable IRC log.

    Prefers the rich per-point parse (energy, reaction coordinate, gradients)
    and falls back to a geometry-only trajectory when the log yields no
    structured points. Logs that are missing on disk are reported in
    ``omitted_source_paths`` rather than dropped silently.
    """
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
        try:
            try:
                rich = parse_irc_path(log_file_path=str(resolved))
            except Exception:
                rich = None
            points = []
            if rich:
                for fallback_index, point in enumerate(rich):
                    xyz = point.get("xyz")
                    if xyz is None:
                        raise ValueError("IRC point has no geometry")
                    item = {
                        "source_point_index": int(point.get("point_number", fallback_index)),
                        "direction": point.get("direction") or declared,
                        "geometry_xyz_text": _canonical_xyz(xyz, ""),
                    }
                    for source_key, evidence_key in (
                        ("electronic_energy_hartree", "electronic_energy_hartree"),
                        ("reaction_coordinate", "reaction_coordinate_sqrt_amu_bohr"),
                        ("max_gradient", "max_gradient_hartree_per_bohr"),
                        ("rms_gradient", "rms_gradient_hartree_per_bohr"),
                    ):
                        if point.get(source_key) is not None:
                            item[evidence_key] = float(point[source_key])
                    _finite_json(item)
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
                        "geometry_xyz_text": _canonical_xyz(xyz, ""),
                    })
            if not points:
                raise ValueError("IRC source produced no valid points")
            trajectory = {"source_log": source_log, "declared_direction": declared, "points": points}
            _finite_json(trajectory)
            trajectories.append(trajectory)
        except Exception:
            logger.debug("TCKDB evidence IRC source omitted: %s", source_log, exc_info=True)
            omitted.append(source_log)
    if not trajectories:
        return _unavailable("empty_result" if logs else "missing_source", [str(p) for p in logs] or None)
    value = {"parser_version": IRC_PARSER_VERSION, "trajectories": trajectories}
    if omitted:
        value["omitted_source_paths"] = omitted
    _finite_json(value)
    return {"status": "available", "value": value}


def _parse_energy(path: Path) -> float | None:
    """Return the electronic energy in hartree from an xTB ``energy`` file, or ``None``."""
    try:
        return float(path.read_text().splitlines()[-2].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _parse_gradient(path: Path) -> tuple[float | None, float | None]:
    """Return ``(max_component, rms)`` of the Cartesian gradient, in hartree/bohr.

    Reads a Turbomole-style ``gradient`` file: a ``$grad`` header, one ``cycle``
    line, ``N`` atom-coordinate lines, then ``N`` gradient lines, then ``$end``.
    xTB rewrites this file each time it evaluates a gradient and *appends* a new
    cycle block rather than truncating, so only the final block describes the
    geometry we are reporting; slicing positionally from the top of the file
    would mix an early cycle's gradient into a late cycle's point.

    Returns ``(None, None)`` for any unreadable or malformed file — this is
    best-effort evidence, not a hard requirement of the record.
    """
    try:
        lines = path.read_text().splitlines()
        cycle_positions = [index for index, line in enumerate(lines) if "cycle" in line]
        if not cycle_positions:
            return None, None
        start = cycle_positions[-1] + 1
        block = [line for line in lines[start:] if not line.strip().startswith("$")]
        if len(block) < 2 or len(block) % 2:
            return None, None
        atom_count = len(block) // 2
        components = [
            float(token.replace("D", "E").replace("d", "E"))
            for line in block[atom_count:]
            for token in line.split()[:3]
        ]
        if len(components) != 3 * atom_count:
            return None, None
        return (max(abs(x) for x in components),
                math.sqrt(sum(x * x for x in components) / len(components)))
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        return None, None


def _parse_xtbout(path: Path) -> float | None:
    """Return the last total energy in hartree from an ``xtbout`` log, or ``None``."""
    try:
        matches = _XTBOUT_TOTAL_ENERGY_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
        return float(matches[-1].replace("D", "E").replace("d", "e")) if matches else None
    except (OSError, ValueError):
        return None


def _invocation_sort_key(invocation_id: str) -> tuple[int, int]:
    """Return the numeric ``(run, slot)`` ordering key for a validated invocation id."""
    run, _, slot = invocation_id.partition(".")
    return int(run), int(slot)


def _node_outputs(directory: Path) -> list[dict[str, Any]]:
    """Return one record per archived GSM ``ograd`` invocation, ordered by invocation id.

    Reads the ``gsm_node_outputs`` directory the queue wrapper archives. Each
    file is named for the id GSM passes to ``ograd`` (``<run>.<slot>``, e.g.
    ``0000.01``) and holds the last evaluation copied under that id. The id is
    reported verbatim as ``invocation_id``, and files whose name is not a
    ``<digits>.<digits>`` id are skipped. Records are ordered by the numeric
    run and slot the id names. Invocations with no readable energy are omitted;
    a Turbomole ``gradient`` file supplies the gradient norms when it is
    present, and an ``xtbout`` log supplies the energy for ids that archived no
    ``energy`` file.
    """
    outputs: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return []
    for path in sorted(directory.glob("*.energy")):
        if not _INVOCATION_ID_RE.match(path.stem):
            logger.debug("Skipping GSM node output with a non-id name: %s", path)
            continue
        energy = _parse_energy(path)
        if energy is None:
            continue
        item = {"invocation_id": path.stem, "electronic_energy_hartree": energy}
        max_g, rms_g = _parse_gradient(path.with_suffix(".gradient"))
        if max_g is not None:
            item["max_gradient_hartree_per_bohr"] = max_g
        if rms_g is not None:
            item["rms_gradient_hartree_per_bohr"] = rms_g
        outputs[path.stem] = item
    for path in sorted(directory.glob("*.xtbout")):
        if not _INVOCATION_ID_RE.match(path.stem):
            logger.debug("Skipping GSM node output with a non-id name: %s", path)
            continue
        if path.stem not in outputs:
            energy = _parse_xtbout(path)
            if energy is not None:
                outputs[path.stem] = {"invocation_id": path.stem, "electronic_energy_hartree": energy}
    return [outputs[invocation_id] for invocation_id in sorted(outputs, key=_invocation_sort_key)]


def _gradient_geometry(path: Path) -> dict[str, Any] | None:
    """Return the geometry a Turbomole ``gradient`` file's final cycle was evaluated at.

    The coordinate block carries the element symbol after the three Cartesian
    components and is written in bohr; the returned xyz is in Angstrom and
    carries no isotopes. Returns ``None`` for a missing, unreadable, or
    malformed file.
    """
    try:
        lines = path.read_text().splitlines()
        cycle_positions = [index for index, line in enumerate(lines) if "cycle" in line]
        if not cycle_positions:
            return None
        block = [line for line in lines[cycle_positions[-1] + 1:] if not line.strip().startswith("$")]
        if len(block) < 2 or len(block) % 2:
            return None
        symbols, coords = [], []
        for line in block[:len(block) // 2]:
            tokens = line.split()
            if len(tokens) < 4:
                return None
            coords.append(tuple(float(token.replace("D", "E").replace("d", "E")) * bohr_to_angstrom
                                for token in tokens[:3]))
            symbols.append(tokens[3].capitalize())
        return {"symbols": tuple(symbols), "coords": tuple(coords)}
    except (OSError, ValueError, IndexError):
        logger.debug("Could not read a geometry from gradient file %s", path, exc_info=True)
        return None


def _superposed_displacement(geometry: Mapping[str, Any], frame: Mapping[str, Any]) -> float | None:
    """Return the ``kabsch`` displacement between ``geometry`` and ``frame``, or ``None``.

    ``frame``'s isotopes are adopted for the comparison so both geometries are
    centered with the same masses. ``None`` when the two do not share an atom
    ordering ``kabsch`` accepts.
    """
    if tuple(geometry["symbols"]) != tuple(frame["symbols"]):
        return None
    candidate = {"symbols": tuple(geometry["symbols"]),
                 "isotopes": frame.get("isotopes"),
                 "coords": tuple(geometry["coords"])}
    try:
        return float(kabsch(candidate, dict(frame)))
    except Exception:
        logger.debug("Kabsch comparison of an archived gradient geometry failed", exc_info=True)
        return None


def _frame_attachments(directory: Path,
                       frames: list[Mapping[str, Any]],
                       invocations: list[dict[str, Any]],
                       ) -> dict[int, dict[str, Any]]:
    """Return, per frame index, the archived invocation whose geometry is that frame.

    Each invocation's geometry is read from its ``gradient`` file and superposed
    onto every frame. A frame is claimed only when its displacement is within
    ``GRADIENT_MATCH_TOLERANCE_ANGSTROM`` and no other frame is, and the
    attachment is kept only when exactly one invocation claims that frame.
    Invocations with no gradient file, no match, an ambiguous match, or a
    contested frame yield no attachment.
    """
    claims: dict[int, list[tuple[dict[str, Any], float]]] = {}
    for item in invocations:
        geometry = _gradient_geometry(directory / f"{item['invocation_id']}.gradient")
        if geometry is None:
            continue
        within = []
        for index, frame in enumerate(frames):
            displacement = _superposed_displacement(geometry, frame)
            if displacement is not None and displacement <= GRADIENT_MATCH_TOLERANCE_ANGSTROM:
                within.append((index, displacement))
        if len(within) != 1:
            logger.debug("GSM invocation %s matched %d frames by geometry; not attached",
                         item["invocation_id"], len(within))
            continue
        claims.setdefault(within[0][0], []).append((item, within[0][1]))
    attachments = {}
    for index, claimants in claims.items():
        if len(claimants) != 1:
            logger.debug("GSM frame %d is claimed by %d invocations by geometry; not attached",
                         index, len(claimants))
            continue
        item, displacement = claimants[0]
        attachment = {
            "geometry_matched_ograd_invocation_id": item["invocation_id"],
            "geometry_match_displacement_angstrom": displacement,
        }
        for key in ("electronic_energy_hartree",
                    "max_gradient_hartree_per_bohr",
                    "rms_gradient_hartree_per_bohr"):
            if key in item:
                attachment[key] = item[key]
        attachments[index] = attachment
    return attachments


def _path_search_method(record: Mapping[str, Any]) -> str | None:
    """Return ``'gsm'``, ``'neb'``, or ``None`` for the TS path-search method used.

    Prefers the record's explicit ``chosen_ts_method`` and falls back to the
    method sources of the chosen TS guess. ``None`` when the record does not
    identify a path-search method, which gates GSM evidence off entirely.
    """
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
    """Build the GSM evidence record from the selected stringfile and node outputs.

    Points carry the stringfile geometry, the stringfile's own relative energy,
    and ``cumulative_com_superposed_displacement_angstrom``: the running sum,
    zero at the first frame, over consecutive frames, of the Frobenius norm of
    the coordinate difference after each pair is translated to its center of
    mass and optimally rotated onto the other. The norm is unweighted --
    ``sqrt(sum over atoms of |dr|^2)`` in Angstrom, which is ``sqrt(N_atoms)``
    times the RMSD -- while the superposition frame is mass-dependent, so the
    value changes under isotopic substitution. It shares neither units, sign
    convention, nor origin with the IRC record's
    ``reaction_coordinate_sqrt_amu_bohr``.

    Every archived ``ograd`` invocation is reported verbatim under
    ``ograd_invocations``, keyed by GSM's own invocation id. An id does not
    identify a frame by arithmetic, but a frame is identified by content: the
    archived ``gradient`` file records the geometry it was evaluated at, and an
    invocation's energy and gradient norms are attached to a point only when
    that geometry superposes onto exactly one frame within
    ``GRADIENT_MATCH_TOLERANCE_ANGSTROM`` and that frame is claimed by no other
    invocation. An attached point carries
    ``geometry_matched_ograd_invocation_id`` and
    ``geometry_match_displacement_angstrom`` alongside the values.
    """
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
                logger.debug("GSM cumulative displacement unavailable for %s", source, exc_info=True)
                coordinates = None
        relative = parse_gsm_stringfile_energies(str(path))
        node_directory = path.parent / "gsm_node_outputs"
        invocations = _node_outputs(node_directory)
        attachments = _frame_attachments(node_directory, frames, invocations)
        points = []
        for index, frame in enumerate(frames):
            point = {
                "source_point_index": index,
                "geometry_xyz_text": _canonical_xyz(frame, f"gsm_point_{index}"),
            }
            if coordinates is not None:
                point["cumulative_com_superposed_displacement_angstrom"] = coordinates[index]
            if relative is not None and index < len(relative):
                point["stringfile_relative_energy_kcal_mol"] = float(relative[index])
            point.update(attachments.get(index) or {})
            points.append(point)
        value = {
            "source_stringfile": source,
            "parser_version": GSM_PARSER_VERSION,
            "method": "gsm",
            "selected_source_point_index": selected,
            "points": points,
        }
        if invocations:
            value["ograd_invocations"] = invocations
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
            attempts = []
            if source_record.get("freq_log"):
                attempts.append(("freq_hessian", _build_hessian, [str(source_record["freq_log"])]))
            if record_kind == "transition_state" and source_record.get("irc_logs"):
                attempts.append(("irc", _build_irc, [str(path) for path in source_record["irc_logs"]]))
            if (record_kind == "transition_state"
                    and _path_search_method(source_record) == "gsm"
                    and source_record.get("gsm_log")):
                attempts.append(("gsm", _build_gsm, [str(source_record["gsm_log"])]))
            for evidence_kind, builder, source_paths in attempts:
                try:
                    record[evidence_kind] = builder(source_record, project_directory)
                except Exception:
                    logger.debug(
                        "Unexpected TCKDB evidence builder failure kind=%s label=%s",
                        evidence_kind, source_record.get("label"), exc_info=True,
                    )
                    record[evidence_kind] = _unavailable("parse_failed", source_paths)
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
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "producer": {
            "name": "ARC",
            "version": output_doc.get("arc_version"),
            "git_commit": output_doc.get("arc_git_commit"),
        },
        "records": records,
    }
    _finite_json(evidence)
    return evidence


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry so a completed rename survives a crash.

    ``os.replace`` is atomic, but the new directory entry is not durable until
    the containing directory is itself synced. Not every platform or filesystem
    permits opening or syncing a directory (notably Windows), so a failure here
    is logged and skipped rather than failing a write that already succeeded.
    """
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        logger.debug("Could not open directory %s to fsync it", directory, exc_info=True)
        return
    try:
        os.fsync(descriptor)
    except OSError:
        logger.debug("Could not fsync directory %s", directory, exc_info=True)
    finally:
        os.close(descriptor)


def write_tckdb_evidence_atomic(*, evidence_doc: Mapping[str, Any], output_directory: str | Path) -> Path:
    """Write deterministic strict JSON, fsync, and atomically replace the sidecar.

    Both the file contents and the parent directory entry are synced, so a
    successfully returned path is durable across a crash.
    """
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
        _fsync_directory(output_directory)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            logger.debug("Failed to remove temporary evidence file %s", temporary, exc_info=True)
        raise
    return target
