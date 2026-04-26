#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to run RitS (Right into the Saddle) and emit TS guesses
as a YAML file consumable by ARC's RitSAdapter.

This script must be invoked from inside the ``rits_env`` conda environment
(it does NOT import ``megalodon`` directly — RitS's own
``scripts/sample_transition_state.py`` does that). The parent ARC process
shells out to this script via ``subprocess.run`` so that ARC's main env
does not have to carry the heavy ML dependency stack.

Input file (``input.yml``)
--------------------------
Required keys:
    reactant_xyz_path : str   absolute path to a plain XYZ file (atom-mapped)
    product_xyz_path  : str   absolute path to the matching product XYZ
    rits_repo_path    : str   absolute path to the RitS source checkout
    ckpt_path         : str   absolute path to the pretrained ``rits.ckpt``
    output_xyz_path   : str   absolute path RitS should write its raw output to
    yml_out_path      : str   absolute path this script writes the parsed TSGuess list to

Optional keys (with defaults):
    config_path  : str   defaults to ``<rits_repo_path>/scripts/conf/rits.yaml``
    n_samples    : int   default 10
    batch_size   : int   default 32
    charge       : int   default 0
    device       : str   default 'auto' (RitS picks GPU if visible, else CPU)
    add_stereo   : bool  default False
    num_steps    : int   default None (use config value)

Output (``yml_out_path``)
-------------------------
A YAML *list* of TSGuess dictionaries. Each entry has:
    method           : 'RitS'
    method_direction : 'F'
    method_index     : int      (0-based sample index)
    initial_xyz      : str      (XYZ-format coordinate block, no header lines)
    success          : bool
    execution_time   : str      (str(datetime.timedelta))

If RitS fails to produce any usable output, the script writes a list with a
single failed-guess entry instead of raising — the parent adapter then logs
the failure but continues running other TS methods.
"""

import argparse
import datetime
import os
import subprocess
import sys
import traceback
from typing import List, Optional

import yaml


def read_yaml_file(path: str) -> dict:
    """Read a YAML file and return its contents as a dict."""
    with open(path, 'r') as f:
        return yaml.load(stream=f, Loader=yaml.FullLoader)


def string_representer(dumper, data):
    """YAML representer that uses block literals for multi-line strings."""
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)


def save_yaml_file(path: str, content) -> None:
    """Save ``content`` to a YAML file at ``path``."""
    yaml.add_representer(str, string_representer)
    with open(path, 'w') as f:
        f.write(yaml.dump(data=content))


def parse_multi_frame_xyz(xyz_path: str) -> List[str]:
    """
    Parse a (possibly multi-frame) XYZ file into a list of coordinate-block strings.

    RitS writes a single XYZ file when ``--n_samples == 1`` and a multi-frame
    XYZ when ``n_samples > 1`` (frames concatenated, each prefixed by an atom
    count line and a blank/comment line). This parser handles both.

    Args:
        xyz_path (str): Path to the XYZ file emitted by RitS.

    Returns:
        List[str]: One coordinate block per frame, suitable for passing to
        ``arc.species.converter.str_to_xyz`` (atom symbols + xyz only — no
        header / comment lines).
    """
    if not os.path.isfile(xyz_path):
        return list()
    with open(xyz_path, 'r') as f:
        raw_lines = [line.rstrip('\n') for line in f]
    frames = list()
    i, n = 0, len(raw_lines)
    while i < n:
        # Skip blank lines between frames
        while i < n and not raw_lines[i].strip():
            i += 1
        if i >= n:
            break
        # First non-blank line of a frame should be the atom count
        try:
            n_atoms = int(raw_lines[i].strip())
        except ValueError:
            # Not a frame header — bail on this row to avoid an infinite loop
            i += 1
            continue
        i += 1
        # Comment / energy line (may be blank)
        if i < n:
            i += 1
        # The next n_atoms lines are coordinates
        coord_lines = list()
        for _ in range(n_atoms):
            if i >= n:
                break
            coord_lines.append(raw_lines[i])
            i += 1
        if len(coord_lines) == n_atoms:
            frames.append('\n'.join(coord_lines))
    return frames


def run_rits(input_dict: dict) -> List[dict]:
    """
    Invoke ``scripts/sample_transition_state.py`` from the RitS source tree
    and parse the resulting XYZ frames into a list of TSGuess dictionaries.

    Args:
        input_dict (dict): The parsed contents of ``input.yml``.

    Returns:
        List[dict]: One TSGuess-shaped dict per generated sample. Always at
        least one entry — a failed sentinel if RitS produced nothing.
    """
    repo = input_dict['rits_repo_path']
    sample_script = os.path.join(repo, 'scripts', 'sample_transition_state.py')
    config_path = input_dict.get('config_path') or os.path.join(repo, 'scripts', 'conf', 'rits.yaml')
    output_xyz = input_dict['output_xyz_path']
    n_samples = int(input_dict.get('n_samples', 10))
    batch_size = int(input_dict.get('batch_size', 32))
    charge = int(input_dict.get('charge', 0))
    device = str(input_dict.get('device', 'auto'))
    add_stereo = bool(input_dict.get('add_stereo', False))
    num_steps = input_dict.get('num_steps')

    cmd = [
        sys.executable, sample_script,
        '--reactant_xyz', input_dict['reactant_xyz_path'],
        '--product_xyz', input_dict['product_xyz_path'],
        '--config', config_path,
        '--ckpt', input_dict['ckpt_path'],
        '--output', output_xyz,
        '--n_samples', str(n_samples),
        '--batch_size', str(batch_size),
        '--charge', str(charge),
        '--device', device,
    ]
    if add_stereo:
        cmd.append('--add_stereo')
    if num_steps is not None:
        cmd.extend(['--num_steps', str(num_steps)])

    t0 = datetime.datetime.now()
    print(f'[rits_script] running: {" ".join(cmd)}', flush=True)
    completed = subprocess.run(cmd, cwd=repo)
    elapsed = datetime.datetime.now() - t0

    if completed.returncode != 0:
        print(f'[rits_script] sample_transition_state.py exited with code {completed.returncode}', flush=True)
        return [_failed_guess(elapsed, index=0)]

    frames = parse_multi_frame_xyz(output_xyz)
    if not frames:
        print(f'[rits_script] no frames parsed from {output_xyz}', flush=True)
        return [_failed_guess(elapsed, index=0)]

    tsgs = list()
    for i, coord_block in enumerate(frames):
        tsgs.append({
            'method': 'RitS',
            'method_direction': 'F',
            'method_index': i,
            'initial_xyz': coord_block,
            'success': True,
            'execution_time': str(elapsed),
        })
    return tsgs


def _failed_guess(elapsed: datetime.timedelta, index: int = 0) -> dict:
    """Return a failed-TSGuess sentinel dict."""
    return {
        'method': 'RitS',
        'method_direction': 'F',
        'method_index': index,
        'initial_xyz': None,
        'success': False,
        'execution_time': str(elapsed),
    }


def parse_command_line_arguments(command_line_args: Optional[list] = None) -> argparse.Namespace:
    """Parse the script's command-line arguments."""
    parser = argparse.ArgumentParser(description='Run RitS to generate TS guesses for an ARC reaction.')
    parser.add_argument('--yml_in_path', metavar='input', type=str, default='input.yml',
                        help='Path to the input YAML file (default: ./input.yml).')
    return parser.parse_args(command_line_args)


def main():
    """Entry point: read input.yml, run RitS, write output YAML."""
    args = parse_command_line_arguments()
    yml_in_path = str(args.yml_in_path)
    if not os.path.isfile(yml_in_path):
        print(f'[rits_script] input file not found: {yml_in_path}', file=sys.stderr)
        sys.exit(1)
    input_dict = read_yaml_file(yml_in_path)

    try:
        tsgs = run_rits(input_dict)
    except Exception:
        traceback.print_exc()
        tsgs = [_failed_guess(datetime.timedelta(0), index=0)]

    save_yaml_file(path=input_dict['yml_out_path'], content=tsgs)
    n_ok = sum(1 for tsg in tsgs if tsg.get('success'))
    print(f'[rits_script] wrote {len(tsgs)} TSGuess entries ({n_ok} successful) to {input_dict["yml_out_path"]}',
          flush=True)


if __name__ == '__main__':
    main()
