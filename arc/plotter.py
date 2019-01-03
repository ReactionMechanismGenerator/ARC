#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import logging
import numpy as np
import math
import matplotlib.pyplot as plt

import py3Dmol
from rdkit import Chem

from rmgpy.exceptions import AtomTypeError

from arc.species import mol_from_xyz, rdkit_conf_from_mol


##################################################################


def plot_rotor_scan(angle, v_list):
    """
    plots a 1D rotor PES for v_list vs. angle
    """
    angle = angle * 180 / math.pi  # convert radians to degree
    v_list = np.array(v_list, np.float64)  # in kJ/mol
    plt.figure(figsize=(4, 3), dpi=120)
    plt.subplot(1, 1, 1)
    plt.plot(angle, v_list, 'g.')
    plt.xlabel('dihedral (deg)')
    plt.xlim = (0, 360)
    plt.xticks(np.arange(0, 361, step=60))
    plt.ylabel('V (kJ/mol)')
    plt.tight_layout()
    plt.show()


def show_sticks(xyz):
    """
    Draws the molecule in a "sticks" style according to supplied xyz coordinates
    Returns whether successful of not
    """
    try:
        mol, coordinates = mol_from_xyz(xyz)
    except AtomTypeError:
        return False
    try:
        _, rd_mol, _ = rdkit_conf_from_mol(mol, coordinates)
    except ValueError:
        return False
    mb = Chem.MolToMolBlock(rd_mol)
    p = py3Dmol.view(width=400, height=400)
    p.addModel(mb, 'sdf')
    p.setStyle({'stick': {}})
    # p.setBackgroundColor('0xeeeeee')
    p.zoomTo()
    p.show()
    return True


def log_thermo(label, path):
    """
    Logging thermodata from an Arkane output file
    """
    logging.info('\n\n')
    logging.debug('Thermodata for species {0}'.format(label))
    thermo_block = ''
    log = False
    with open(path, 'r') as f:
        line = f.readline()
        while line != '':
            if 'Thermodynamics for' in line:
                thermo_block = ''
                log = True
            elif 'thermo(' in line:
                log = False
            if log:
                thermo_block += line[2:]
            line = f.readline()
    logging.info(thermo_block)
    logging.info('\n')
