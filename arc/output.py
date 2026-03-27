"""
Module for writing the consolidated output.yml at the end of an ARC run.

output.yml supersedes output/status.yml and <project>.info — it consolidates
all result data into a single file with run-relative paths so downstream
consumers (TCKDB, analysis scripts) need only one file.

Written atomically at the very end of a run. If the run is interrupted, the
file will not exist rather than be partially written.
"""

import datetime
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from arc.common import ARC_PATH, VERSION, get_git_commit, get_logger, read_yaml_file, save_yaml_file
from arc.constants import E_h_kJmol
from arc.species.converter import xyz_to_str


logger = get_logger()

_RMG_PY_PATH = '/home/calvin/code/RMG-Py'


def write_output_yml(
    project: str,
    project_directory: str,
    species_dict: Dict,
    reactions: List,
    output_dict: Dict,
    opt_level=None,
    freq_level=None,
    sp_level=None,
    freq_scale_factor: Optional[float] = None,
    freq_scale_factor_user_provided: bool = False,
    bac_type: Optional[str] = None,
) -> None:
    """
    Write the consolidated output.yml to <project_directory>/output/output.yml.

    Non-converged species appear with ``converged: false`` and null result fields.
    Monoatomic species have null for all freq/statmech fields (not absent).

    Args:
        project (str): ARC project name.
        project_directory (str): Root directory of this ARC project.
        species_dict (dict): {label: ARCSpecies} for all species and TSs.
        reactions (list): List of ARCReaction objects.
        output_dict (dict): {label: {convergence, paths, job_types, ...}}.
        opt_level (Level, optional): Level of theory for geometry optimization.
        freq_level (Level, optional): Level of theory for frequency calculations.
        sp_level (Level, optional): Level of theory for single-point energies.
        freq_scale_factor (float, optional): The harmonic frequencies scaling factor used.
        freq_scale_factor_user_provided (bool): Whether the user explicitly set the scale factor.
        bac_type (str, optional): The BAC type ('p', 'm', or None).
    """
    doc: Dict[str, Any] = {}

    # ---- header ----------------------------------------------------------------
    arc_git, _ = get_git_commit(ARC_PATH)
    doc['project'] = project
    doc['arc_version'] = VERSION
    doc['arc_git_commit'] = arc_git or None
    doc['arkane_git_commit'] = _get_arkane_git_commit()
    doc['datetime_completed'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    # ---- levels of theory -------------------------------------------------------
    doc['opt_level'] = _level_to_dict(opt_level)
    doc['freq_level'] = _level_to_dict(freq_level)
    doc['sp_level'] = _level_to_dict(sp_level)
    doc['freq_scale_factor'] = freq_scale_factor
    doc['freq_scale_factor_source'] = (
        None if freq_scale_factor_user_provided
        else _resolve_freq_scale_factor_source(freq_level)
    )
    doc['energy_corrections_applied'] = bac_type is not None
    doc['energy_correction_note'] = None

    # ---- species and TSs --------------------------------------------------------
    point_groups = _compute_point_groups(species_dict, project_directory)
    doc['species'] = []
    doc['transition_states'] = []
    for spc in species_dict.values():
        d = _spc_to_dict(spc, output_dict, project_directory, point_groups)
        if spc.is_ts:
            doc['transition_states'].append(d)
        else:
            doc['species'].append(d)

    # ---- reactions --------------------------------------------------------------
    doc['reactions'] = [_rxn_to_dict(rxn) for rxn in reactions]

    # ---- atomic write -----------------------------------------------------------
    out_dir = os.path.join(project_directory, 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'output.yml')
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix='.output.yml.tmp')
    try:
        os.close(fd)
        save_yaml_file(path=tmp_path, content=doc)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    logger.info(f'Wrote consolidated results to {out_path}')


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_arkane_git_commit() -> Optional[str]:
    """Return the HEAD commit hash of the RMG-Py repo (Arkane lives there), or None."""
    try:
        head, _ = get_git_commit(_RMG_PY_PATH)
        return head or None
    except Exception:
        return None


def _level_to_dict(level) -> Optional[Dict]:
    """Convert a Level object to {method, basis, software}, or None."""
    if level is None:
        return None
    return {
        'method': getattr(level, 'method', None),
        'basis': getattr(level, 'basis', None),
        'software': getattr(level, 'software', None),
    }


def _resolve_freq_scale_factor_source(freq_level) -> Optional[str]:
    """
    Return the full literature citation for the freq scale factor, or None.

    Reads data/freq_scale_factors.yml as raw text, parses the header comment
    block to build [N] → citation, then finds the [N] tag for freq_level.
    Returns None when the level is not in the database (scale factor was
    user-supplied or computed).
    """
    if freq_level is None:
        return None
    yml_path = os.path.join(ARC_PATH, 'data', 'freq_scale_factors.yml')
    try:
        with open(yml_path, 'r', encoding='utf-8') as f:
            raw = f.read()
    except OSError:
        return None

    # Build {ref_number: full_citation_text} from header comment lines
    citations: Dict[str, str] = {}
    for m in re.finditer(r'#\s*\[(\d+)\]\s+(.+)', raw):
        citations[m.group(1)] = m.group(2).strip()

    level_key = str(freq_level) if not isinstance(freq_level, str) else freq_level
    # Match the entry line and extract the LAST [N] tag in its trailing comment.
    # Handles both simple "# [4]" and compound "# 0.915 * 1.014, [1] Table 7" forms.
    pattern = rf"'{re.escape(level_key)}':[^\n]+#[^\n]*\[(\d+)\]"
    m = re.search(pattern, raw)
    if not m:
        return None
    return citations.get(m.group(1))


def _make_rel_path(path: Optional[str], project_directory: str) -> Optional[str]:
    """Convert an absolute path to one relative to project_directory, or None."""
    if not path:
        return None
    try:
        return os.path.relpath(path, project_directory)
    except ValueError:
        return path  # Windows: relpath can fail across drives


def _safe(fn, default=None):
    """Call fn() and return *default* if any exception is raised."""
    try:
        return fn()
    except Exception:
        return default


def _compute_point_groups(species_dict: Dict, project_directory: str) -> Dict[str, Optional[str]]:
    """
    Compute point groups for all species via the ``symmetry`` binary in the RMG env.

    Calls ``arc/scripts/get_point_groups.py`` as a subprocess in the RMG conda
    environment (same pattern as save_arkane_thermo.py).  Returns a dict mapping
    each species label to its point group string (e.g. ``'C2v'``) or ``None``.
    On any failure the function returns an empty dict so callers get ``None`` for
    every species rather than crashing the run.
    """
    from arc.imports import settings
    from arc.job.local import execute_command

    rmg_env = settings.get('RMG_ENV_NAME', 'rmg_env')
    script_path = os.path.join(ARC_PATH, 'arc', 'scripts', 'get_point_groups.py')

    # Build input dict: {label: {symbols: [...], coords: [...]}}
    pg_input: Dict[str, Any] = {}
    for label, spc in species_dict.items():
        xyz = spc.final_xyz if spc.final_xyz is not None else spc.initial_xyz
        if xyz is None:
            continue
        symbols = list(xyz.get('symbols', []))
        coords = [list(row) for row in xyz.get('coords', [])]
        if symbols and coords:
            pg_input[label] = {'symbols': symbols, 'coords': coords}

    if not pg_input:
        return {}

    tmp_dir = os.path.join(project_directory, 'output')
    os.makedirs(tmp_dir, exist_ok=True)
    fd_in, tmp_in = tempfile.mkstemp(dir=tmp_dir, suffix='.pg_input.yml')
    fd_out, tmp_out = tempfile.mkstemp(dir=tmp_dir, suffix='.pg_output.yml')
    try:
        os.close(fd_in)
        os.close(fd_out)
        save_yaml_file(path=tmp_in, content=pg_input)

        commands = [
            'bash -lc "set -euo pipefail; '
            'if command -v micromamba >/dev/null 2>&1; then '
            f'    micromamba run -n {rmg_env} python {script_path} {tmp_in} {tmp_out}; '
            'elif command -v conda >/dev/null 2>&1 || command -v mamba >/dev/null 2>&1; then '
            f'    conda run -n {rmg_env} python {script_path} {tmp_in} {tmp_out}; '
            'else '
            '    echo \'micromamba/conda required\' >&2; exit 1; '
            'fi"',
        ]
        _, stderr = execute_command(command=commands, executable='/bin/bash')
        if stderr:
            logger.warning(f'get_point_groups.py stderr: {stderr}')

        result = read_yaml_file(tmp_out) or {}
        return {str(k): (str(v) if v is not None else None) for k, v in result.items()}
    except Exception as e:
        logger.warning(f'Could not compute point groups: {e}')
        return {}
    finally:
        for p in (tmp_in, tmp_out):
            try:
                os.unlink(p)
            except OSError:
                pass


def _spc_to_dict(spc, output_dict: Dict, project_directory: str, point_groups: Optional[Dict] = None) -> Dict:
    """Build the per-species/TS section for output.yml."""
    label = spc.label
    entry = output_dict.get(label, {})
    converged = entry.get('convergence', False)
    paths = entry.get('paths', {})

    d: Dict[str, Any] = {
        'label': label,
        'original_label': spc.original_label,
        'charge': spc.charge,
        'multiplicity': spc.multiplicity,
        'converged': converged,
        'is_ts': spc.is_ts,
    }

    # ── molecular identity (non-TS only) ────────────────────────────────────
    if not spc.is_ts and spc.mol is not None:
        d['smiles'] = _safe(lambda: spc.mol.copy(deep=True).to_smiles())
        d['inchi'] = _safe(lambda: spc.mol.copy(deep=True).to_inchi())
        d['inchi_key'] = _safe(lambda: spc.mol.copy(deep=True).to_inchi_key())
        d['formula'] = _safe(lambda: spc.mol.get_formula())
    else:
        d['smiles'] = None
        d['inchi'] = None
        d['inchi_key'] = None
        d['formula'] = None

    # ── final geometry ──────────────────────────────────────────────────────
    xyz = spc.final_xyz if spc.final_xyz is not None else spc.initial_xyz
    d['xyz'] = xyz_to_str(xyz) if xyz is not None else None

    # ── is monoatomic? (drives null-vs-value for freq/statmech) ─────────────
    is_mono = spc.is_monoatomic() is True

    # ── energies ────────────────────────────────────────────────────────────
    if converged and spc.e_elect is not None:
        d['sp_energy_hartree'] = spc.e_elect / E_h_kJmol
    else:
        d['sp_energy_hartree'] = None

    if converged and not is_mono and spc.e0 is not None and spc.e_elect is not None:
        d['zpe_hartree'] = (spc.e0 - spc.e_elect) / E_h_kJmol
    else:
        d['zpe_hartree'] = None

    d['opt_converged'] = entry.get('job_types', {}).get('opt') if converged else None
    # opt_n_steps and opt_final_energy_hartree are not retained in memory
    d['opt_n_steps'] = None
    d['opt_final_energy_hartree'] = None

    # ── freq results ────────────────────────────────────────────────────────
    if is_mono:
        d['freq_n_imag'] = None
        d['imag_freq_cm1'] = None
    elif converged and not spc.is_ts:
        d['freq_n_imag'] = 0
        d['imag_freq_cm1'] = None          # never meaningful for stable species
    elif converged and spc.is_ts:
        d['freq_n_imag'] = 1
        d['imag_freq_cm1'] = _get_ts_imag_freq(spc)
    else:
        d['freq_n_imag'] = None
        d['imag_freq_cm1'] = None

    # ── log file paths (run-relative) ────────────────────────────────────────
    d['freq_log'] = _make_rel_path(paths.get('freq') or None, project_directory)
    # When SP level == opt level the SP log IS the opt log — store it anyway.
    d['sp_log'] = _make_rel_path(paths.get('sp') or None, project_directory)

    if spc.is_ts:
        d['irc_logs'] = [_make_rel_path(p, project_directory) for p in (paths.get('irc') or [])]
        d['rxn_label'] = spc.rxn_label

    # ── thermochemistry (non-TS converged species only) ──────────────────────
    if not spc.is_ts and converged and spc.thermo is not None and spc.thermo.H298 is not None:
        d['thermo'] = _thermo_to_dict(spc.thermo)
    else:
        d['thermo'] = None

    # ── statistical mechanics ────────────────────────────────────────────────
    if not is_mono and converged:
        pg = (point_groups or {}).get(label)
        d['statmech'] = _statmech_to_dict(spc, project_directory, point_group=pg)
    else:
        d['statmech'] = None

    return d


def _get_ts_imag_freq(spc) -> Optional[float]:
    """Return the imaginary frequency (cm⁻¹) of the chosen TS conformer, or None."""
    try:
        chosen = spc.chosen_ts
        if chosen is not None and spc.ts_guesses and chosen < len(spc.ts_guesses):
            freqs = spc.ts_guesses[chosen].imaginary_freqs
            if freqs:
                return float(freqs[0])
    except Exception:
        pass
    return None


def _thermo_to_dict(thermo) -> Dict:
    """Convert a ThermoData object to a plain, unit-labelled dict."""
    def _scalar(x):
        """Extract the numeric value from a (value, units) tuple or a plain number."""
        if isinstance(x, (list, tuple)) and len(x) >= 1:
            return x[0]
        return x

    t: Dict[str, Any] = {
        'h298_kj_mol': thermo.H298,
        's298_j_mol_k': thermo.S298,
        'tmin_k': _scalar(thermo.Tmin),
        'tmax_k': _scalar(thermo.Tmax),
    }

    # ── Cp tabulation ────────────────────────────────────────────────────────
    cp = getattr(thermo, 'cp_data', None)
    if cp is not None:
        t['cp_data'] = cp
    elif thermo.Tdata is not None and thermo.Cpdata is not None:
        T_list = thermo.Tdata[0] if isinstance(thermo.Tdata, (list, tuple)) else thermo.Tdata
        Cp_list = thermo.Cpdata[0] if isinstance(thermo.Cpdata, (list, tuple)) else thermo.Cpdata
        t['cp_data'] = [{'temperature_k': float(T), 'cp_j_mol_k': float(Cp)}
                        for T, Cp in zip(T_list, Cp_list)]
    else:
        t['cp_data'] = None

    # ── NASA polynomials ─────────────────────────────────────────────────────
    t['nasa_low'] = getattr(thermo, 'nasa_low', None)
    t['nasa_high'] = getattr(thermo, 'nasa_high', None)

    return t


def _statmech_to_dict(spc, project_directory: str, point_group: Optional[str] = None) -> Dict:
    """Build the statmech sub-section for a non-monoatomic converged species/TS."""
    # Use the cached private attribute to avoid triggering a geometry re-read
    is_linear = spc._is_linear

    if spc.is_monoatomic() is True:
        rotor_kind = 'atom'
    elif is_linear:
        rotor_kind = 'linear'
    else:
        rotor_kind = 'asymmetric_top'

    freqs = getattr(spc, 'freqs', None)
    if freqs is not None and spc.is_ts:
        # Exclude the imaginary mode (most negative frequency)
        freqs = [f for f in freqs if f >= 0]

    return {
        'e0_kj_mol': spc.e0,
        'spin_multiplicity': spc.multiplicity,
        'optical_isomers': spc.optical_isomers,
        'is_linear': is_linear,
        'external_symmetry': spc.external_symmetry,
        'point_group': point_group,
        'rigid_rotor_kind': rotor_kind,
        'harmonic_frequencies_cm1': [float(f) for f in freqs] if freqs is not None else None,
        'torsions': _get_torsions(spc, project_directory),
    }


def _get_torsions(spc, project_directory: str) -> List[Dict]:
    """Build the torsions list from spc.rotors_dict."""
    if not getattr(spc, 'rotors_dict', None):
        return []
    torsions = []
    for rotor in spc.rotors_dict.values():
        if rotor.get('success') is not True:
            continue
        scan = rotor.get('scan')       # 4-atom dihedral defining atoms, 1-indexed
        pivots = rotor.get('pivots')   # 2-atom rotation axis, 1-indexed
        symmetry = rotor.get('symmetry', 1)
        rotor_type = rotor.get('type', 'HinderedRotor')
        treatment = 'free_rotor' if 'Free' in str(rotor_type) else 'hindered_rotor'
        torsions.append({
            'symmetry_number': symmetry,
            'treatment': treatment,
            'atom_indices': scan,
            'pivot_atoms': pivots,
            'barrier_kj_mol': _get_rotor_barrier(rotor, project_directory),
        })
    return torsions


def _get_rotor_barrier(rotor: Dict, project_directory: str) -> Optional[float]:
    """
    Return max(V) - min(V) in kJ/mol from the 1D scan output file.

    parse_1d_scan_energies already zeroes the minimum, so max(energies) is the
    barrier height directly.
    """
    scan_path = rotor.get('scan_path', '')
    if not scan_path:
        return None
    if not os.path.isabs(scan_path):
        scan_path = os.path.join(project_directory, scan_path)
    if not os.path.isfile(scan_path):
        return None
    try:
        from arc.parser.parser import parse_1d_scan_energies
        energies, _ = parse_1d_scan_energies(log_file_path=scan_path)
        if energies:
            return float(max(energies))
    except Exception:
        pass
    return None


def _rxn_to_dict(rxn) -> Dict:
    """Convert an ARCReaction to a plain dict for output.yml."""
    kinetics = rxn.kinetics
    kin_dict: Optional[Dict] = None
    if kinetics is not None:
        A = kinetics.get('A')
        Ea = kinetics.get('Ea')
        Tmin = kinetics.get('Tmin')
        Tmax = kinetics.get('Tmax')
        kin_dict = {
            'A': A[0] if isinstance(A, (tuple, list)) else A,
            'A_units': A[1] if isinstance(A, (tuple, list)) else None,
            'n': kinetics.get('n'),
            'Ea': Ea[0] if isinstance(Ea, (tuple, list)) else Ea,
            'Ea_units': Ea[1] if isinstance(Ea, (tuple, list)) else None,
            'Tmin_k': Tmin[0] if isinstance(Tmin, (tuple, list)) else Tmin,
            'Tmax_k': Tmax[0] if isinstance(Tmax, (tuple, list)) else Tmax,
            'dA': kinetics.get('dA'),
            'dn': kinetics.get('dn'),
            'dEa': kinetics.get('dEa'),
        }

    return {
        'label': rxn.label,
        'reactant_labels': list(rxn.reactants),
        'product_labels': list(rxn.products),
        'family': rxn.family,
        'multiplicity': rxn.multiplicity,
        'ts_label': rxn.ts_label,
        'kinetics': kin_dict,
    }
