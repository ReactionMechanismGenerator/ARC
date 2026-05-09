"""
Build mockter fixture YAMLs from real ARC project directories.

Walks an ARC project's calcs/ tree and output/output.yml to extract:
- Per-conformer xyz + e_elect (from conf_opt_N/input.log).
- opt + fine_opt geometries + energies (from output.yml's coarse_opt_log / opt_log).
- freq geometry, freqs, ZPE (from output.yml).
- Raw force-constant matrix block (text-scraped verbatim from the freq log).
- sp e_elect + T1 diagnostic (from output.yml + sp/input.log).
- Rotor scan energies + per-point geometries (from scan_*/input.log).
- IRC trajectories (from TSs/<label>/irc_*/input.log).
- Transition-state opt/freq/sp/IRC.

Outputs one fixture YAML per scenario in our v1 schema, plus inline
provenance tying back to the source ARC project commit and DFT levels.

Designed for one-shot use: when the source calcs/ tree is about to be
deleted, this script captures everything mockter will need to replay it.
"""

import datetime
import os
import re
import sys

import numpy as np

from arc.common import read_yaml_file, save_yaml_file
from arc.parser.parser import (
    parse_1d_scan_coords,
    parse_1d_scan_energies,
    parse_e_elect,
    parse_geometry,
    parse_irc_traj,
    parse_t1,
)
from arc.species.converter import xyz_to_str


FIXTURES_DIR = '/home/alon/Code/ARC/arc/testing/mockter_fixtures'
COMPUTED_DIR = os.path.join(FIXTURES_DIR, 'computed')

KJ_PER_HARTREE = 2625.4996394798


def kjmol_to_hartree(value):
    """
    Convert an energy in kJ/mol (ARC parser convention) to Hartree (fixture convention).

    Args:
        value: Energy in kJ/mol or None.

    Returns:
        float | None: Energy in Hartree, or None.
    """
    if value is None:
        return None
    return float(value) / KJ_PER_HARTREE


def scrape_force_constant_block(log_path: str) -> str | None:
    """
    Extract the verbatim 'Force constants in Cartesian coordinates' block
    from a Gaussian log file. Returns the multi-line text (header line plus
    all rows) or None if the block is absent.

    The block is preserved as a YAML literal so mockter can re-emit it
    byte-identically when forging logs.

    Args:
        log_path (str): Absolute path to the Gaussian log file.

    Returns:
        str | None: The verbatim block as a single string, or None if absent.
    """
    if not os.path.isfile(log_path):
        return None
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    block_start = None
    for idx, line in enumerate(lines):
        if 'Force constants in Cartesian coordinates:' in line:
            block_start = idx
    if block_start is None:
        return None
    block_lines = [lines[block_start]]
    pattern = re.compile(r'^\s*\d+\s+([\d\.\-DE\+]+\s*)+$')
    for line in lines[block_start + 1:]:
        if pattern.match(line) or re.match(r'^\s*\d+(\s+\d+)+\s*$', line):
            block_lines.append(line)
        else:
            break
    return ''.join(block_lines)


def parse_conformer_dir(conf_dir: str) -> dict | None:
    """
    Extract xyz + electronic energy from a conf_opt_N directory.

    Args:
        conf_dir (str): Path to the conformer directory.

    Returns:
        dict | None: {'xyz': str, 'e_elect': float | None}, or None on failure.
    """
    log_path = os.path.join(conf_dir, 'input.log')
    if not os.path.isfile(log_path):
        return None
    geom = parse_geometry(log_path)
    if geom is None:
        return None
    e = parse_e_elect(log_path)
    return {'xyz': xyz_to_str(geom), 'e_elect': kjmol_to_hartree(e)}


def parse_scan_dir(scan_dir: str) -> dict | None:
    """
    Extract dihedral indices, energies (Hartree relative to first), and
    per-point geometries from a scan_aN directory.

    Args:
        scan_dir (str): Path to the scan directory.

    Returns:
        dict | None: {'torsions': list[list[int]], 'energies': list[float], 'xyzs': list[str]} or None.
    """
    log_path = os.path.join(scan_dir, 'input.log')
    if not os.path.isfile(log_path):
        return None
    scan = parse_1d_scan_energies(log_path)
    if scan is None:
        return None
    energies, _angles = scan
    coords = parse_1d_scan_coords(log_path)
    torsions = _torsion_indices_from_log(log_path)
    return {
        'torsions': torsions,
        'energies': [kjmol_to_hartree(e) for e in energies] if energies is not None else None,
        'xyzs': [xyz_to_str(c) for c in coords] if coords is not None else None,
    }


def _torsion_indices_from_log(log_path: str) -> list[list[int]] | None:
    """
    Pull the scanned dihedral indices from a Gaussian scan log's route section.

    Args:
        log_path (str): Path to the Gaussian log.

    Returns:
        list[list[int]] | None: One torsion as [[a, b, c, d]], or None.
    """
    pattern = re.compile(r'D\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+S\s+\d+\s+[\d\.\-]+', re.IGNORECASE)
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return [[int(g) for g in m.groups()]]
    return None


def parse_irc_dir(irc_dir: str) -> dict | None:
    """
    Extract IRC trajectory xyzs from a single irc_aN directory.

    Args:
        irc_dir (str): Path to the IRC directory.

    Returns:
        dict | None: {'xyzs': list[str]} or None.
    """
    log_path = os.path.join(irc_dir, 'input.log')
    if not os.path.isfile(log_path):
        return None
    traj = parse_irc_traj(log_path)
    if traj is None:
        return None
    return {'xyzs': [xyz_to_str(p) for p in traj]}


def species_block_from_output(spc_yml: dict, calcs_root: str) -> dict:
    """
    Build the per-species fixture block from an output.yml species entry,
    augmenting with conformer/scan harvesting from disk.

    Args:
        spc_yml (dict): Single species dict from output.yml (already parsed).
        calcs_root (str): Path to the scenario's calcs/ root.

    Returns:
        dict: Per-species fixture content following our schema.
    """
    label = spc_yml['label']
    species_calcs = os.path.join(calcs_root, 'Species', label)
    block: dict = {
        'smiles': spc_yml.get('smiles'),
        'multiplicity': spc_yml.get('multiplicity'),
        'charge': spc_yml.get('charge'),
        'inchi': spc_yml.get('inchi'),
    }

    conformers = []
    if os.path.isdir(species_calcs):
        for entry in sorted(os.listdir(species_calcs)):
            if entry.startswith('conf_opt_'):
                conf = parse_conformer_dir(os.path.join(species_calcs, entry))
                if conf is not None:
                    conf['isomorphic'] = True
                    conformers.append(conf)
    if conformers:
        block['conformers'] = conformers

    coarse_opt_log = spc_yml.get('coarse_opt_log')
    if coarse_opt_log:
        coarse_path = os.path.join(os.path.dirname(calcs_root), coarse_opt_log)
        coarse_geom = parse_geometry(coarse_path) if os.path.isfile(coarse_path) else None
        block['opt'] = {
            'xyz': xyz_to_str(coarse_geom) if coarse_geom is not None else None,
            'e_elect': spc_yml.get('coarse_opt_final_energy_hartree'),
        }

    opt_log = spc_yml.get('opt_log')
    if opt_log:
        block['fine_opt'] = {
            'xyz': spc_yml.get('xyz'),
            'e_elect': spc_yml.get('opt_final_energy_hartree'),
        }

    freq_log_rel = spc_yml.get('freq_log')
    freq_block: dict = {
        'freqs': spc_yml.get('statmech', {}).get('harmonic_frequencies_cm1') if spc_yml.get('statmech') else None,
        'zpe': spc_yml.get('zpe_hartree'),
        'imag_freq_cm1': spc_yml.get('imag_freq_cm1'),
        'n_imag': spc_yml.get('freq_n_imag'),
    }
    if freq_log_rel:
        freq_log = os.path.join(os.path.dirname(calcs_root), freq_log_rel)
        fc_block = scrape_force_constant_block(freq_log)
        freq_block['hessian_block'] = fc_block
    block['freq'] = freq_block

    sp_e = spc_yml.get('sp_energy_hartree')
    sp_log_rel = spc_yml.get('sp_log')
    t1 = None
    is_composite_sp = bool(sp_log_rel and 'composite_' in sp_log_rel)
    if sp_log_rel:
        sp_log = os.path.join(os.path.dirname(calcs_root), sp_log_rel)
        if os.path.isfile(sp_log):
            t1 = parse_t1(sp_log)
    block['sp'] = {'e_elect': sp_e, 't1_diagnostic': float(t1) if t1 is not None else None}

    if is_composite_sp:
        composite_log = os.path.join(os.path.dirname(calcs_root), sp_log_rel)
        composite_geom = parse_geometry(composite_log) if os.path.isfile(composite_log) else None
        block['composite'] = {
            'xyz': xyz_to_str(composite_geom) if composite_geom is not None else None,
            'e_elect': sp_e,
            'hessian_block': scrape_force_constant_block(composite_log),
        }

    scans = []
    if os.path.isdir(species_calcs):
        for entry in sorted(os.listdir(species_calcs)):
            if entry.startswith('scan_'):
                scan = parse_scan_dir(os.path.join(species_calcs, entry))
                if scan is not None and scan.get('energies'):
                    scans.append(scan)
    if scans:
        block['scans'] = scans

    if spc_yml.get('statmech') and spc_yml['statmech'].get('torsions'):
        block['rotors_meta'] = spc_yml['statmech']['torsions']

    return block


def transition_state_block(ts_yml: dict, calcs_root: str) -> dict:
    """
    Build the per-TS fixture block. Tolerates partial / unconverged TS
    runs by recording whatever data is on disk and flagging convergence.

    Args:
        ts_yml (dict): Single TS dict from output.yml.
        calcs_root (str): Path to the scenario's calcs/ root.

    Returns:
        dict: Per-TS fixture content.
    """
    label = ts_yml['label']
    ts_calcs = os.path.join(calcs_root, 'TSs', label)
    block: dict = {
        'multiplicity': ts_yml.get('multiplicity'),
        'charge': ts_yml.get('charge'),
        'rxn_label': ts_yml.get('rxn_label'),
        'converged': ts_yml.get('converged'),
        'chosen_ts_method': ts_yml.get('chosen_ts_method'),
    }

    if not os.path.isdir(ts_calcs):
        return block

    opt_dirs = sorted([d for d in os.listdir(ts_calcs) if d.startswith('opt_')])
    freq_dirs = sorted([d for d in os.listdir(ts_calcs) if d.startswith('freq_')])
    sp_dirs = sorted([d for d in os.listdir(ts_calcs) if d.startswith('sp_')])
    scan_dirs = sorted([d for d in os.listdir(ts_calcs) if d.startswith('scan_')])
    irc_dirs = sorted([d for d in os.listdir(ts_calcs) if d.startswith('irc_')])

    if opt_dirs:
        first_opt = os.path.join(ts_calcs, opt_dirs[0])
        last_opt = os.path.join(ts_calcs, opt_dirs[-1])
        first_geom = parse_geometry(os.path.join(first_opt, 'input.log'))
        last_geom = parse_geometry(os.path.join(last_opt, 'input.log'))
        block['opt'] = {
            'xyz': xyz_to_str(first_geom) if first_geom is not None else None,
            'e_elect': kjmol_to_hartree(parse_e_elect(os.path.join(first_opt, 'input.log'))),
        }
        if len(opt_dirs) > 1:
            block['fine_opt'] = {
                'xyz': xyz_to_str(last_geom) if last_geom is not None else None,
                'e_elect': kjmol_to_hartree(parse_e_elect(os.path.join(last_opt, 'input.log'))),
            }

    if freq_dirs:
        freq_log = os.path.join(ts_calcs, freq_dirs[-1], 'input.log')
        from arc.parser.parser import parse_frequencies, parse_zpe_correction
        freqs = parse_frequencies(freq_log)
        zpe = parse_zpe_correction(freq_log)
        block['freq'] = {
            'freqs': [float(f) for f in freqs] if freqs is not None else None,
            'zpe': kjmol_to_hartree(zpe),
            'hessian_block': scrape_force_constant_block(freq_log),
        }

    if sp_dirs:
        sp_dir = os.path.join(ts_calcs, sp_dirs[-1])
        candidates = [
            os.path.join(sp_dir, 'input.log'),
            os.path.join(sp_dir, 'input.out'),
            os.path.join(sp_dir, 'out.txt'),
        ]
        sp_log = next((c for c in candidates if os.path.isfile(c)), None)
        block['sp'] = {
            'e_elect': kjmol_to_hartree(parse_e_elect(sp_log)) if sp_log else None,
            't1_diagnostic': parse_t1(sp_log) if sp_log else None,
            'log_file': os.path.basename(sp_log) if sp_log else None,
        }

    scans = []
    for sd in scan_dirs:
        scan = parse_scan_dir(os.path.join(ts_calcs, sd))
        if scan is not None and scan.get('energies'):
            scans.append(scan)
    if scans:
        block['scans'] = scans

    irc = {}
    for idx, ircd in enumerate(irc_dirs):
        traj = parse_irc_dir(os.path.join(ts_calcs, ircd))
        if traj is not None:
            key = 'forward' if idx == 0 else 'reverse'
            irc[key] = traj
    if irc:
        block['irc'] = irc

    return block


def _to_jsonable(obj):
    """Recursively convert numpy types to plain Python for clean YAML output."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def build_scenario_fixture(scenario_dir: str, scenario_name: str) -> dict:
    """
    Assemble a complete fixture dict for one scenario.

    Args:
        scenario_dir (str): Path to the scenario directory (containing calcs/ and output/).
        scenario_name (str): Short name (e.g. 'mockter1').

    Returns:
        dict: Fixture content ready for YAML dump.
    """
    calcs_root = os.path.join(scenario_dir, 'calcs')
    output_yml_path = os.path.join(scenario_dir, 'output', 'output.yml')
    output_yml = read_yaml_file(output_yml_path) if os.path.isfile(output_yml_path) else {}

    fixture = {
        'schema_version': 1,
        'provenance': {
            'generated_at': datetime.date.today().isoformat(),
            'source_project': output_yml.get('project'),
            'arc_version': output_yml.get('arc_version'),
            'arc_git_commit': output_yml.get('arc_git_commit'),
            'arkane_git_commit': output_yml.get('arkane_git_commit'),
            'datetime_started': str(output_yml.get('datetime_started')),
            'datetime_completed': str(output_yml.get('datetime_completed')),
            'opt_level': output_yml.get('opt_level'),
            'freq_level': output_yml.get('freq_level'),
            'sp_level': output_yml.get('sp_level'),
            'composite_method': output_yml.get('composite_method'),
            'arkane_level_of_theory': output_yml.get('arkane_level_of_theory'),
            'freq_scale_factor': output_yml.get('freq_scale_factor'),
        },
        'species': {},
        'reactions': {},
        'ts': {},
    }

    for spc in output_yml.get('species', []) or []:
        fixture['species'][spc['label']] = species_block_from_output(spc, calcs_root)

    for rxn in output_yml.get('reactions', []) or []:
        fixture['reactions'][rxn['label']] = {
            'family': rxn.get('family'),
            'multiplicity': rxn.get('multiplicity'),
            'reactants': rxn.get('reactant_labels'),
            'products': rxn.get('product_labels'),
            'ts_label': rxn.get('ts_label'),
        }

    for ts in output_yml.get('transition_states', []) or []:
        fixture['ts'][ts['label']] = transition_state_block(ts, calcs_root)

    return _to_jsonable(fixture)


SCENARIOS = [
    ('s1', 'mockter1'),
    ('s2', 'mockter2'),
    ('s3', 'mockter3'),
    ('s4', 'mockter4'),
    ('s5', 'mockter5'),
    ('s6', 'mockter6'),
]


def main() -> int:
    """Main entry: build all six fixtures and write them next to the script's source dir."""
    out_dir = FIXTURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    for src, name in SCENARIOS:
        scenario_dir = os.path.join(COMPUTED_DIR, src)
        if not os.path.isdir(scenario_dir):
            print(f'[skip] {src}: not found at {scenario_dir}', file=sys.stderr)
            continue
        try:
            fixture = build_scenario_fixture(scenario_dir, name)
        except Exception as exc:
            print(f'[error] {src}: {exc.__class__.__name__}: {exc}', file=sys.stderr)
            continue
        out_path = os.path.join(out_dir, f'{name}.yml')
        save_yaml_file(path=out_path, content=fixture)
        n_species = len(fixture.get('species', {}))
        n_ts = len(fixture.get('ts', {}))
        print(f'[ok] {src} -> {out_path}: {n_species} species, {n_ts} TS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
