#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to read an Arknane thermo job and save in the same folder a YAML file with the thermo data.
"""

import os

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


def main():
    """
    Run this script from an Arkane project folder.
    In ARC this is under calcs/statmech/thermo.
    It loads the RMG thermo library, extracts H298, S298, NASA polynomial coefficients, and
    tabulated Cp data, saving the results in a YAML file.
    """
    thermo_lib_path = os.path.join(os.getcwd(), 'RMG_libraries', 'thermo.py')
    if not os.path.isfile(thermo_lib_path):
        return
    result = dict()
    local_context = {'ThermoData': ThermoData,
                     'Wilhoit': Wilhoit,
                     'NASAPolynomial': NASAPolynomial,
                     'NASA': NASA}
    global_context = {}
    library = ThermoLibrary()
    library.load(thermo_lib_path, local_context, global_context)
    for entry in library.entries.values():
        thermo_data = entry.data
        H298 = thermo_data.get_enthalpy(RT) / 1000.0
        S298 = thermo_data.get_entropy(RT)
        data = str(thermo_data)
        nasa_low, nasa_high = _extract_nasa(thermo_data)
        cp_data = _extract_cp(thermo_data)
        result[entry.label] = {
            'H298': H298,
            'S298': S298,
            'data': data,
            'nasa_low': nasa_low,
            'nasa_high': nasa_high,
            'cp_data': cp_data,
        }
    if result:
        result_path = os.path.join(os.getcwd(), 'thermo.yaml')
        save_yaml_file(path=result_path, content=result)


if __name__ == '__main__':
    main()
