#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to read an Arknane thermo job and save in the same folder a YAML file with the thermo data.
"""

import os
import re
import sys

from common import save_yaml_file

from rmgpy.data.thermo import ThermoLibrary
from rmgpy.thermo import NASAPolynomial, NASA, ThermoData, Wilhoit

RT = 298.15  # Room temperature in K


_CP_TEMPS = [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0, 2000.0, 2400.0]


def _extract_nasa(thermo_data):
    """Return (nasa_low, nasa_high) dicts from a NASA thermo object, or (None, None)."""
    if not isinstance(thermo_data, NASA):
        return None, None
    polys = sorted(thermo_data.polynomials, key=lambda p: p.Tmax.value_si)
    if len(polys) < 2:
        return None, None
    low, high = polys[0], polys[1]
    return (
        {'tmin_k': float(low.Tmin.value_si), 'tmax_k': float(low.Tmax.value_si),
         'coeffs': [float(c) for c in low.coeffs]},
        {'tmin_k': float(high.Tmin.value_si), 'tmax_k': float(high.Tmax.value_si),
         'coeffs': [float(c) for c in high.coeffs]},
    )


def _extract_thermo_points(thermo_data):
    """Return a list of per-temperature thermochemistry dicts, or None.

    Each entry carries the full set of TCKDB-shaped fields that
    ``thermo_point`` accepts:

        ``temperature_k``  - the evaluation temperature in K
        ``cp_j_mol_k``     - heat capacity        (J/(mol*K))
        ``h_kj_mol``       - enthalpy             (kJ/mol)
        ``s_j_mol_k``      - entropy              (J/(mol*K))
        ``g_kj_mol``       - Gibbs free energy    (kJ/mol)

    RMG's NASA / ThermoData accessors return SI units (J/mol for energies,
    J/(mol*K) for capacities/entropies); enthalpy and free energy are
    converted to kJ/mol at the boundary because TCKDB persists them in
    those units.

    Any per-temperature evaluation that raises (e.g., the polynomial is
    not valid at that T) is skipped silently — the goal is best-effort
    enrichment, not failing the whole library extraction over one out-
    of-range point.
    """
    try:
        tmin = thermo_data.Tmin.value_si
        tmax = thermo_data.Tmax.value_si
    except Exception:
        return None

    points = []
    for T in _CP_TEMPS:
        if not (tmin <= T <= tmax):
            continue
        try:
            cp = float(thermo_data.get_heat_capacity(T))
            h_kj = float(thermo_data.get_enthalpy(T)) / 1000.0
            s = float(thermo_data.get_entropy(T))
            g_kj = float(thermo_data.get_free_energy(T)) / 1000.0
        except Exception:
            continue
        points.append({
            'temperature_k': T,
            'cp_j_mol_k': cp,
            'h_kj_mol': h_kj,
            's_j_mol_k': s,
            'g_kj_mol': g_kj,
        })
    return points or None


def _iter_thermo_calls(content):
    """Return the source text of each top-level ``thermo(...)`` call in ``content``.

    Uses balanced-parenthesis scanning rather than a single regex because a
    ``thermo(...)`` call spans multiple lines and contains nested calls
    (``NASA(NASAPolynomial(...), ...)``). The lookbehind ensures we only match a
    standalone ``thermo(`` call, not the ``thermo=`` keyword inside it.
    """
    calls = []
    for m in re.finditer(r'(?<![A-Za-z0-9_.])thermo\s*\(', content):
        idx = m.end() - 1  # position of the opening '('
        depth = 0
        matched = False
        while idx < len(content):
            char = content[idx]
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    calls.append(content[m.start():idx + 1])
                    matched = True
                    break
            idx += 1
        if not matched:
            # Unbalanced parentheses (e.g. an unpaired '(' inside a NASA comment) — warn rather
            # than silently drop the block, which would recreate the very null-thermo symptom
            # this fallback exists to prevent.
            sys.stderr.write('Could not extract a balanced thermo() block from an Arkane output.py '
                             f'starting at offset {m.start()}; skipping it.\n')
    return calls


def _load_thermo_entries_from_output_py(output_path, local_context):
    """Reconstruct a ``{label: NASA}`` mapping directly from an Arkane ``output.py``.

    Used as a fallback when ``RMG_libraries/thermo.py`` is absent because Arkane's
    ``save_thermo_lib`` crashed *after* writing ``output.py`` (e.g. it rejects the two
    identical reactants of an A+A reaction such as OH + OH, or a singlet-carbene
    multiplicity clash). ``output.py`` holds one ``thermo(label=..., thermo=NASA(...))``
    call per species; each is evaluated with the real rmgpy thermo classes in scope —
    exactly the context Arkane itself uses to read these files back — so no thermo data
    is lost to the library-save failure.
    """
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
    entries = dict()

    def _capture(label=None, thermo=None, *args, **kwargs):
        if label is not None:
            entries[label] = thermo

    eval_context = dict(local_context)
    eval_context['thermo'] = _capture
    for call_src in _iter_thermo_calls(content):
        try:
            eval(call_src, {'__builtins__': {}}, eval_context)
        except Exception as e:  # noqa: BLE001 - log which block failed, keep parsing the rest
            sys.stderr.write(f'Could not parse an Arkane thermo() block from {output_path}: {e}\n')
    return entries


def main():
    """
    Run this script from an Arkane project folder.
    In ARC this is under calcs/statmech/thermo.
    It loads the computed thermo (from the RMG thermo library Arkane wrote, or — when
    that library save failed — straight from ``output.py``), extracts H298, S298, NASA
    polynomial coefficients, and tabulated thermo points, saving the results in a YAML file.
    """
    cwd = os.getcwd()
    thermo_lib_path = os.path.join(cwd, 'RMG_libraries', 'thermo.py')
    output_path = os.path.join(cwd, 'output.py')
    local_context = {'ThermoData': ThermoData,
                     'Wilhoit': Wilhoit,
                     'NASAPolynomial': NASAPolynomial,
                     'NASA': NASA}
    entries = dict()  # label -> rmgpy thermo object (NASA / ThermoData / Wilhoit)
    if os.path.isfile(thermo_lib_path):
        library = ThermoLibrary()
        library.load(thermo_lib_path, local_context, {})
        for entry in library.entries.values():
            entries[entry.label] = entry.data
    elif os.path.isfile(output_path):
        entries = _load_thermo_entries_from_output_py(output_path, local_context)
    else:
        return
    result = dict()
    for label, thermo_data in entries.items():
        if thermo_data is None:
            continue
        H298 = thermo_data.get_enthalpy(RT) / 1000.0
        S298 = thermo_data.get_entropy(RT)
        data = str(thermo_data)
        nasa_low, nasa_high = _extract_nasa(thermo_data)
        thermo_points = _extract_thermo_points(thermo_data)
        result[label] = {
            'H298': H298,
            'S298': S298,
            'data': data,
            'nasa_low': nasa_low,
            'nasa_high': nasa_high,
            'thermo_points': thermo_points,
        }
    if result:
        result_path = os.path.join(cwd, 'thermo.yaml')
        save_yaml_file(path=result_path, content=result)


if __name__ == '__main__':
    main()
