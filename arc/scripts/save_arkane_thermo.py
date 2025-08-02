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


def main():
    """
    Run this script from an Arkane project folder.
    In ARC this is under calcs/statmech/thermo.
    It loads the RMG thermo library, extracts H298 and S298, and saves the results in kJ/mol and J/mol/K, respectively,
    in a YAML file in the format::

    label:
        H298: 0.0  # kJ/mol
        S298: 0.0  # J/mol/K
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
        result[entry.label] = {'H298': H298, 'S298': S298, 'data': data}
    if result:
        result_path = os.path.join(os.getcwd(), 'thermo.yaml')
        save_yaml_file(path=result_path, content=result)


if __name__ == '__main__':
    main()
