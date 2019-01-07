#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import logging
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import py3Dmol as p3d
from rdkit import Chem

from rmgpy.exceptions import AtomTypeError

from arc.species import mol_from_xyz, rdkit_conf_from_mol
from arc.settings import arc_path


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
    p = p3d.view(width=400, height=400)
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


def draw_thermo_parity_plot(species_list, path=None):
    """
    Plots a parity plot of calculated thermo and RMG's best values for species in species_list
    """
    logging.info('Thermo parity plots (labels can be dragged around if they overlap):')
    if path is not None:
        path = os.path.join(path, str('thermo_parity_plots.pdf'))
        if os.path.exists(path):
            os.remove(path)
        pp = PdfPages(path)
    labels, comments, h298_arc, h298_rmg, s298_arc, s298_rmg = [], [], [], [], [], []
    for spc in species_list:
        labels.append(spc.label)
        h298_arc.append(spc.thermo.getEnthalpy(298) * 0.001)  # convered to kJ/mol
        h298_rmg.append(spc.rmg_thermo.getEnthalpy(298) * 0.001)  # convered to kJ/mol
        s298_arc.append(spc.thermo.getEntropy(298))  # in J/mol*K
        s298_rmg.append(spc.rmg_thermo.getEntropy(298))  # in J/mol*K
        comments.append(spc.rmg_thermo.comment)
    min_h = min(h298_arc + h298_rmg)
    max_h = max(h298_arc + h298_rmg)
    min_s = min(s298_arc + s298_rmg)
    max_s = max(s298_arc + s298_rmg)
    plt.figure(figsize=(5, 4), dpi=120)
    plt.title('H298 parity plot')
    plt.plot(h298_arc, h298_rmg, 'go')
    plt.plot([min_h, max_h], [min_h, max_h], 'b-', linewidth=0.5)
    plt.xlabel('H298 calculated by ARC (kJ / mol)')
    plt.ylabel('H298 determined by RMG (kJ / mol)')
    plt.xlim = (min_h, max_h)
    plt.ylim = (min_h, max_h)
    for i, label in enumerate(labels):
        a = plt.annotate(label, xy=(h298_arc[i], h298_rmg[i]), size=10)
        a.draggable()
    plt.tight_layout()
    if path is not None:
        plt.savefig(pp, format='pdf')
    plt.show()

    plt.figure(figsize=(5, 4), dpi=120)
    plt.title('S298 parity plot')
    plt.plot(s298_arc, s298_rmg, 'go')
    plt.plot([min_s, max_s], [min_s, max_s], 'b-', linewidth=0.5)
    plt.xlabel('S298 calculated by ARC (J / mol * K)')
    plt.ylabel('S298 determined by RMG (J / mol * K)')
    plt.xlim = (min_s, max_s)
    plt.ylim = (min_s, max_s)
    for i, label in enumerate(labels):
        b = plt.annotate(label, xy=(h298_arc[i], h298_rmg[i]), size=10)
        b.draggable()
    plt.tight_layout()
    if path is not None:
        plt.savefig(pp, format='pdf')
        pp.close()
    plt.show()

    logging.info('\nSources of thermoproperties determined by RMG for the parity plots:')
    max_label_len = max([len(label) for label in labels])
    for i, label in enumerate(labels):
        logging.info('   {0}: {1}{2}'.format(label, ' '*(max_label_len - len(label)), comments[i]))


def save_geo(species, project):
    """
    Save the geometry in several forms for an ARC Species object in the project's output folder under the species name
    """
    if species.is_ts:
        folder_name = 'TSs'
    else:
        folder_name = 'Species'
    geo_path = os.path.join(arc_path, 'Projects', project, 'output', folder_name, species.label, 'geometry')
    if os.path.exists(geo_path):
        # clean working folder from all previous output
        for file in os.listdir(geo_path):
            file_path = os.path.join(geo_path, file)
            os.remove(file_path)
    else:
        os.makedirs(geo_path)

    # xyz
    xyz = '{0}\n'.format(species.number_of_atoms)
    xyz += '{0} optimized at {1}\n'.format(species.label, species.opt_level)
    xyz += '{0}\n'.format(species.final_xyz)
    with open(os.path.join(geo_path, '{0}.xyz'.format(species.label)), 'w') as f:
        f.write(xyz)

    # GaussView file
    gv = '# hf/3-21g\n\n{0} optimized at {1}\n'.format(
        species.label, species.opt_level)
    gv += '{0} {1}\n'.format(species.charge, species.multiplicity)
    gv += '{0}\n'.format(species.final_xyz)
    with open(os.path.join(geo_path, '{0}.gjf'.format(species.label)), 'w') as f:
        f.write(gv)
