#!/usr/bin/env python3
# encoding: utf-8

"""
Look up atom energy corrections (AEC) and bond additivity corrections (BAC)
from the RMG quantum corrections database for a given level of theory.

Run from the RMG conda environment so that Arkane is importable.

Usage::

    python get_qm_corrections.py input.yaml output.yaml

Input YAML format::

    matched_key: "LevelOfTheory(method='cbsqb3',software='gaussian')"
    bac_type: p          # 'p', 'm', or null

Output YAML format, Petersson (``bac_type: p``)::

    aec:
        H: -0.499818
        C: -37.78794
    bac:
        C-H: -0.06

Output YAML format, Melius (``bac_type: m``)::

    aec:
        H: -0.499818
        C: -37.78794
    bac:
        atom_corr: {C: -0.6, H: -0.05}
        bond_corr_length: {C: 57.6, H: 12.4}
        bond_corr_neighbor: {C: -0.027, H: -0.011}
        mol_corr: 0.306

The two sections are looked up independently: a failure in one is reported on
stderr and leaves that section null, while the other section is still written.
"""

import re
import sys
import traceback
from typing import Any, Optional

from arkane.encorr.data import atom_energies, pbac, mbac

from arkane_levels import lot_from_string
from common import read_yaml_file, save_yaml_file


def _to_float_tree(value: Any) -> Any:
    """Convert a correction table to plain floats, preserving its nesting.

    A flat Petersson table stays a ``{bond: float}`` map; a Melius table keeps
    its ``atom_corr``/``bond_corr_length``/``bond_corr_neighbor`` sub-maps and
    its scalar ``mol_corr``.
    """
    if isinstance(value, dict):
        return {str(key): _to_float_tree(item) for key, item in value.items()}
    return float(value)


def _lookup_aec(aec_key: str) -> Optional[dict]:
    """The atom energy corrections for ``aec_key``, or None when unlisted."""
    aec = atom_energies.get(lot_from_string(aec_key))
    return _to_float_tree(aec) if aec is not None else None


def _lookup_bac(bac_key: str, bac_type: str) -> Optional[dict]:
    """The bond additivity corrections for ``bac_key``, or None when unlisted."""
    bac = (pbac if bac_type == 'p' else mbac).get(lot_from_string(bac_key))
    return _to_float_tree(bac) if bac is not None else None


def main(input_path: str, output_path: str) -> None:
    """Look up AEC and BAC for the given level of theory key."""
    params = read_yaml_file(input_path) or {}
    bac_type = params.get('bac_type')

    result = {'aec': None, 'bac': None}

    aec_key = params.get('aec_key') or params.get('matched_key')
    bac_key = params.get('bac_key') or params.get('matched_key')

    if aec_key:
        try:
            result['aec'] = _lookup_aec(aec_key)
        except Exception:
            print(f'Failed to look up atom energy corrections for {aec_key!r}:\n'
                  f'{traceback.format_exc()}', file=sys.stderr)

    if bac_key and bac_type in ('p', 'm'):
        try:
            result['bac'] = _lookup_bac(bac_key, bac_type)
        except Exception:
            print(f'Failed to look up {bac_type!r} bond additivity corrections for {bac_key!r}:\n'
                  f'{traceback.format_exc()}', file=sys.stderr)

    save_yaml_file(output_path, result)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} input.yaml output.yaml', file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
