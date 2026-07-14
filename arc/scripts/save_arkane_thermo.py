#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to read an Arknane thermo job and save in the same folder a YAML file with the thermo data.
"""

import ast
import os
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


def _extract_cp(thermo_data):
    """Return a list of {temperature_k, cp_j_mol_k} dicts, or None."""
    try:
        tmin = thermo_data.Tmin.value_si
        tmax = thermo_data.Tmax.value_si
        return [
            {'temperature_k': T, 'cp_j_mol_k': float(thermo_data.get_heat_capacity(T))}
            for T in _CP_TEMPS
            if tmin <= T <= tmax
        ]
    except Exception:
        return None


def _iter_thermo_calls(content):
    """Return the :class:`ast.Call` node of each ``thermo(...)`` call in ``content``.

    Selects calls to the bare name ``thermo`` only, so the ``thermo=`` keyword nested inside each
    call is not mistaken for one. If ``content`` is not parseable Python, a message is written to
    stderr and an empty list is returned.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        sys.stderr.write(f'Could not parse an Arkane output.py as Python: {e}\n')
        return []
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'thermo']


def _load_thermo_entries_from_output_py(output_path, local_context):
    """Reconstruct a ``{label: NASA}`` mapping directly from an Arkane ``output.py``.

    Used as a fallback when ``RMG_libraries/thermo.py`` is absent because Arkane's
    ``save_thermo_lib`` crashed *after* writing ``output.py`` (e.g. it rejects the two
    identical reactants of an A+A reaction such as OH + OH, or a singlet-carbene
    multiplicity clash). ``output.py`` holds one ``thermo(label=..., thermo=NASA(...))``
    call per species; each is evaluated with the real rmgpy thermo classes in scope —
    exactly the context Arkane itself uses to read these files back — so no thermo data
    is lost to the library-save failure. A block that fails to evaluate is reported on
    stderr and the remaining blocks are still parsed.
    """
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
    entries = dict()

    def _capture(label=None, thermo=None, *args, **kwargs):
        if label is not None:
            entries[label] = thermo

    eval_context = dict(local_context)
    eval_context['thermo'] = _capture
    for node in _iter_thermo_calls(content):
        try:
            eval(compile(ast.Expression(body=node), '<arkane_output>', 'eval'), eval_context)
        except Exception as e:
            sys.stderr.write(f'Could not parse an Arkane thermo() block from {output_path}: {e}\n')
    return entries


def main():
    """
    Run this script from an Arkane project folder.
    In ARC this is under calcs/statmech/thermo.
    It loads the computed thermo (from the RMG thermo library Arkane wrote, or — when
    that library save failed — straight from ``output.py``), extracts H298, S298, NASA
    polynomial coefficients, and tabulated Cp data, saving the results in a YAML file.
    A species whose thermo cannot be evaluated is reported on stderr and skipped, so the
    remaining species are still written.
    """
    cwd = os.getcwd()
    thermo_lib_path = os.path.join(cwd, 'RMG_libraries', 'thermo.py')
    output_path = os.path.join(cwd, 'output.py')
    local_context = {'ThermoData': ThermoData,
                     'Wilhoit': Wilhoit,
                     'NASAPolynomial': NASAPolynomial,
                     'NASA': NASA}
    entries = dict()
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
        try:
            H298 = thermo_data.get_enthalpy(RT) / 1000.0
            S298 = thermo_data.get_entropy(RT)
            data = str(thermo_data)
            nasa_low, nasa_high = _extract_nasa(thermo_data)
            cp_data = _extract_cp(thermo_data)
        except Exception as e:
            sys.stderr.write(f'Could not evaluate the computed thermo of {label}: {e}\n')
            continue
        result[label] = {
            'H298': H298,
            'S298': S298,
            'data': data,
            'nasa_low': nasa_low,
            'nasa_high': nasa_high,
            'cp_data': cp_data,
        }
    if result:
        result_path = os.path.join(cwd, 'thermo.yaml')
        save_yaml_file(path=result_path, content=result)


if __name__ == '__main__':
    main()
