#!/usr/bin/env python3
# encoding: utf-8

"""
Batch point group calculation using Patchkovskii's symmetry program.

Run from the RMG conda environment — the ``symmetry`` binary must be on PATH.

Usage::

    python get_point_groups.py input.yaml output.yaml

Input YAML format::

    species_label:
        symbols: [C, H, H, H]
        coords:
            - [0.0, 0.0, 0.0]
            - [0.63, 0.63, 0.63]
            ...

Output YAML format::

    species_label: C3v        # or null if calculation failed
"""

import os
import subprocess
import sys
import tempfile

from common import read_yaml_file, save_yaml_file
from rdkit import Chem

_pt = Chem.GetPeriodicTable()

# Patchkovskii symmetry program tries these tolerances in order.
_TOLERANCES = ['0.02', '0.1', '0.0']


def _point_group_for_species(symbols, coords):
    """
    Return the point group string (e.g. ``'C2v'``) for a species, or ``None``.

    Monoatomic species get ``'Kh'`` (infinite spherical symmetry) without calling
    the binary.  All others use the symmetry binary with a temp input file.
    """
    if not symbols or not coords:
        return None
    if len(symbols) == 1:
        return 'Kh'

    # Build the symmetry binary input:  N\n  Z  x  y  z\n  ...
    lines = [str(len(symbols))]
    for sym, (x, y, z) in zip(symbols, coords):
        try:
            an = _pt.GetAtomicNumber(sym)
        except RuntimeError:
            return None   # unknown element — skip rather than guess
        lines.append(f'{an} {x:.8f} {y:.8f} {z:.8f}')
    geom_text = '\n'.join(lines) + '\n'

    fd, tmp_path = tempfile.mkstemp(suffix='.symm')
    try:
        os.close(fd)
        with open(tmp_path, 'w') as fh:
            fh.write(geom_text)

        for tol in _TOLERANCES:
            try:
                result = subprocess.run(
                    ['symmetry', tmp_path, '-final', tol],
                    capture_output=True, text=True, timeout=30,
                )
                for line in result.stdout.splitlines():
                    if line.startswith('It seems to be the '):
                        # "It seems to be the C2v point group"
                        parts = line.split()
                        if len(parts) >= 7:
                            return parts[5]
            except Exception as exc:
                print(f'Warning: symmetry call failed at tolerance {tol}: {exc}', file=sys.stderr)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError as exc:
            print(f'Warning: failed to remove temp file {tmp_path}: {exc}', file=sys.stderr)
    return None


def main(input_path, output_path):
    """Compute point groups for all species in *input_path*, write to *output_path*."""
    data = read_yaml_file(input_path) or {}
    results = {}
    for label, item in data.items():
        symbols = item.get('symbols', [])
        coords = item.get('coords', [])
        results[label] = _point_group_for_species(symbols, coords)
    save_yaml_file(output_path, results)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} input.yaml output.yaml', file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
