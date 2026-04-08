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

Output YAML format::

    aec:
        H: -0.499818
        C: -37.78794
    bac:
        C-H: -0.06
"""

import re
import sys

from arkane.encorr.data import atom_energies, pbac, mbac
from arkane.modelchem import LevelOfTheory

from common import read_yaml_file, save_yaml_file


def _lot_from_string(lot_str):
    """Construct a LevelOfTheory from its repr string."""
    # Extract keyword arguments from e.g. "LevelOfTheory(method='cbsqb3',software='gaussian')"
    kwargs = dict(re.findall(r"(\w+)\s*=\s*'([^']*)'", lot_str))
    return LevelOfTheory(**kwargs)


def main(input_path, output_path):
    """Look up AEC and BAC for the given level of theory key."""
    params = read_yaml_file(input_path) or {}
    bac_type = params.get('bac_type')

    result = {'aec': None, 'bac': None}

    # Support both old format (single matched_key) and new format (separate aec_key/bac_key)
    aec_key = params.get('aec_key') or params.get('matched_key')
    bac_key = params.get('bac_key')

    if aec_key:
        lot = _lot_from_string(aec_key)
        aec = atom_energies.get(lot)
        if aec is not None:
            result['aec'] = {str(k): float(v) for k, v in aec.items()}

    if bac_key and bac_type in ('p', 'm'):
        bac_lot = _lot_from_string(bac_key)
        bac_dict = pbac if bac_type == 'p' else mbac
        bac = bac_dict.get(bac_lot)
        if bac is not None:
            result['bac'] = {str(k): float(v) for k, v in bac.items()}

    save_yaml_file(output_path, result)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} input.yaml output.yaml', file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
