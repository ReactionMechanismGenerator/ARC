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

# Patchkovskii symmetry program tries these tolerances in order.
_TOLERANCES = ['0.02', '0.1', '0.0']

# Atomic number lookup — covers elements commonly encountered in combustion chemistry.
_ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Mo': 42, 'Ru': 44,
    'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52,
    'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Pb': 82,
}


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
        an = _ATOMIC_NUMBERS.get(sym)
        if an is None:
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
            except Exception:
                pass
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
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
