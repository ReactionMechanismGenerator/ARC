#!/usr/bin/env python3
# encoding: utf-8

"""
Compute per-species applied AEC (atom energy correction) and BAC (bond
additivity correction) totals by delegating to Arkane's public correction
functions. Run from the RMG conda environment so that Arkane is importable.

Inputs and outputs are exchanged via YAML files so the caller (in arc_env)
does not need to import Arkane.

Usage::

    python get_species_corrections.py input.yaml output.yaml

Input YAML::

    level_of_theory: "LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian')"
    bac_type: p          # 'p', 'm', or null
    species:
      - label: CH4
        atoms: {C: 1, H: 4}
        bonds: {C-H: 4}
        coords: [[0, 0, 0], [0.63, 0.63, 0.63], ...]
        nums: [6, 1, 1, 1, 1]
        multiplicity: 1

Output YAML::

    species:
      - label: CH4
        aec:
          value: -0.0234     # in Hartree
          value_unit: hartree
          components:
            - component_kind: atom
              key: C
              multiplicity: 1
              parameter_value: -37.84706210301937
              parameter_unit: hartree
              contribution_value: -0.0153
            - ...
        bac:
          value: -0.7         # in kcal/mol (Petersson) or kcal/mol (Melius)
          value_unit: kcal_mol
          bac_type: p
          components:        # only for bac_type == 'p'; absent for 'm'
            - component_kind: bond
              key: C-H
              multiplicity: 4
              parameter_value: -0.1735
              parameter_unit: kcal_mol
              contribution_value: -0.694

Per-species failure (e.g. missing parameters for that level of theory) is
reported by omitting the failing block (``aec``, ``bac``, or both) from
that species' output; the species entry is still present.
"""

import re
import sys
import traceback

import rmgpy.constants as constants

from arkane.encorr.corr import get_atom_correction, get_bac
from arkane.encorr.data import atom_energies, pbac
from arkane.modelchem import LevelOfTheory

from common import read_yaml_file, save_yaml_file


HARTREE_TO_J_MOL = constants.E_h * constants.Na      # ~2625499.858
KCAL_TO_J_MOL = 4184.0


def _lot_from_string(lot_str):
    """Construct a LevelOfTheory from its repr string."""
    kwargs = dict(re.findall(r"(\w+)\s*=\s*'([^']*)'", lot_str))
    return LevelOfTheory(**kwargs)


def _aec_for(lot, atoms):
    """Total AEC in Hartree, plus atom components (key, multiplicity,
    parameter_value [Hartree], contribution_value [Hartree]).

    Per-element contribution is computed by calling Arkane on a singleton
    ``{element: count}`` dict and converting J/mol → Hartree. The sum of
    component contributions equals the total (modulo float roundoff)
    because Arkane's atom-correction formula is linear in atom counts.
    """
    total_j_mol = get_atom_correction(lot, atoms)
    total_hartree = total_j_mol / HARTREE_TO_J_MOL

    # Look up the per-element parameter (Hartree) from Arkane's atom_energies
    # table via the same fallback Arkane uses internally.
    energy_level = getattr(lot, 'energy', lot)
    params = atom_energies.get(energy_level) or atom_energies.get(energy_level.simple())

    components = []
    for symbol, count in atoms.items():
        per_element_j_mol = get_atom_correction(lot, {symbol: count})
        components.append({
            'component_kind': 'atom',
            'key': symbol,
            'multiplicity': int(count),
            'parameter_value': float(params[symbol]) if params and symbol in params else None,
            'parameter_unit': 'hartree',
            'contribution_value': per_element_j_mol / HARTREE_TO_J_MOL,
        })
    return {
        'value': total_hartree,
        'value_unit': 'hartree',
        'components': components,
    }


def _pbac_for(lot, bonds, coords, nums, multiplicity):
    """Total Petersson BAC in kcal/mol, plus bond components.

    Per-bond contribution is ``count * pbac[bond_key]`` (kcal/mol). The
    parameter dict is looked up via Arkane's level-of-theory fallbacks
    (full → simple → energy → energy.simple). Bond keys absent from the
    parameter table are reported with ``parameter_value: null`` so the
    caller can decide whether to keep the components.
    """
    total_j_mol = get_bac(lot, bonds, coords, nums,
                          bac_type='p', multiplicity=multiplicity)
    total_kcal = total_j_mol / KCAL_TO_J_MOL

    # Find the matching pbac parameter dict. Mirrors the fallback chain
    # used by ``arkane.encorr.bac.BAC`` so component lookups match the
    # parameters the total was computed with.
    candidates = [lot, lot.simple()]
    energy_level = getattr(lot, 'energy', None)
    if energy_level is not None:
        candidates.extend([energy_level, energy_level.simple()])
    params = None
    for cand in candidates:
        if cand in pbac:
            params = pbac[cand]
            break

    components = []
    for bond_key, count in bonds.items():
        param_value = None
        if params:
            if bond_key in params:
                param_value = float(params[bond_key])
            else:
                # Arkane accepts reversed bond keys (e.g. 'H-C' for 'C-H').
                flipped = ''.join(re.findall(r'[a-zA-Z]+|[^a-zA-Z]+', bond_key)[::-1])
                if flipped in params:
                    param_value = float(params[flipped])
        components.append({
            'component_kind': 'bond',
            'key': bond_key,
            'multiplicity': int(count),
            'parameter_value': param_value,
            'parameter_unit': 'kcal_mol',
            'contribution_value': (param_value * count) if param_value is not None else None,
        })
    return {
        'value': total_kcal,
        'value_unit': 'kcal_mol',
        'bac_type': 'p',
        'components': components,
    }


def _mbac_for(lot, bonds, coords, nums, multiplicity):
    """Total Melius BAC in kcal/mol, no components.

    Melius BAC is a pairwise atom-pair function plus a multiplicity
    correction; there is no clean per-bond decomposition, so components
    are deliberately omitted.
    """
    total_j_mol = get_bac(lot, bonds, coords, nums,
                          bac_type='m', multiplicity=multiplicity)
    return {
        'value': total_j_mol / KCAL_TO_J_MOL,
        'value_unit': 'kcal_mol',
        'bac_type': 'm',
    }


def _process_species(spc, lot, bac_type):
    """Compute AEC and BAC for one species; failure of either branch is
    isolated so a partial result is still emitted."""
    out = {'label': spc['label']}

    atoms = spc.get('atoms') or {}
    if atoms:
        try:
            out['aec'] = _aec_for(lot, atoms)
        except Exception:
            out['aec_error'] = traceback.format_exc(limit=2).strip().splitlines()[-1]

    bonds = spc.get('bonds') or {}
    coords = spc.get('coords')
    nums = spc.get('nums')
    multiplicity = spc.get('multiplicity', 1)
    if bac_type in ('p', 'm') and coords and nums:
        try:
            if bac_type == 'p':
                out['bac'] = _pbac_for(lot, bonds, coords, nums, multiplicity)
            else:
                out['bac'] = _mbac_for(lot, bonds, coords, nums, multiplicity)
        except Exception:
            out['bac_error'] = traceback.format_exc(limit=2).strip().splitlines()[-1]

    return out


def main(input_path, output_path):
    """Compute per-species totals + components for the given level of theory."""
    params = read_yaml_file(input_path) or {}

    lot_str = params.get('level_of_theory')
    bac_type = params.get('bac_type')
    species = params.get('species') or []

    result = {'species': []}
    if not lot_str:
        save_yaml_file(output_path, result)
        return

    lot = _lot_from_string(lot_str)
    for spc in species:
        result['species'].append(_process_species(spc, lot, bac_type))

    save_yaml_file(output_path, result)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} input.yaml output.yaml', file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
