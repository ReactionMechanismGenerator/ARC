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


def draw_thermo_parity_plots(species_list, path=None):
    """
    Plots a parity plot of calculated thermo and RMG's best values for species in species_list
    """
    pp = None
    if path is not None:
        path = os.path.join(path, str('thermo_parity_plots.pdf'))
        if os.path.exists(path):
            os.remove(path)
        pp = PdfPages(path)
    labels, comments, h298_arc, h298_rmg, s298_arc, s298_rmg = [], [], [], [], [], []
    for spc in species_list:
        labels.append(spc.label)
        h298_arc.append(spc.thermo.getEnthalpy(298) * 0.001)  # converted to kJ/mol
        h298_rmg.append(spc.rmg_thermo.getEnthalpy(298) * 0.001)  # converted to kJ/mol
        s298_arc.append(spc.thermo.getEntropy(298))  # in J/mol*K
        s298_rmg.append(spc.rmg_thermo.getEntropy(298))  # in J/mol*K
        comments.append(spc.rmg_thermo.comment)
    draw_parity_plot(var_arc=h298_arc, var_rmg=h298_rmg, var_label='H298', var_units='kJ / mol', labels=labels, pp=pp)
    logging.info('Thermo parity plot for S:')
    draw_parity_plot(var_arc=s298_arc, var_rmg=s298_rmg, var_label='S298', var_units='J / mol * K', labels=labels, pp=pp)
    pp.close()
    logging.info('\nSources of thermoproperties determined by RMG for the parity plots:')
    max_label_len = max([len(label) for label in labels])
    for i, label in enumerate(labels):
        logging.info('   {0}: {1}{2}'.format(label, ' '*(max_label_len - len(label)), comments[i]))


def draw_parity_plot(var_arc, var_rmg, var_label, var_units, labels, pp):
    min_var = min(var_arc + var_rmg)
    max_var = max(var_arc + var_rmg)
    fig = plt.figure(figsize=(5, 4), dpi=120)
    ax = fig.add_subplot(111)
    plt.title('{0} parity plot'.format(var_label))
    plt.plot(var_arc, var_rmg, 'go')
    plt.plot([min_var, max_var], [min_var, max_var], 'b-', linewidth=0.5)
    plt.xlabel('{0} calculated by ARC ({1})'.format(var_label, var_units))
    plt.ylabel('{0} determined by RMG ({1})'.format(var_label, var_units))
    plt.xlim = (min_var, max_var * 1.1)
    plt.ylim = (min_var, max_var)
    txt_height = 0.04 * (plt.ylim[1] - plt.ylim[0])  # plt.ylim and plt.xlim return a tuple
    txt_width = 0.02 * (plt.xlim[1] - plt.xlim[0])
    text_positions = get_text_positions(var_arc, var_rmg, txt_width, txt_height)
    text_plotter(var_arc, var_rmg, labels, text_positions, ax, txt_width, txt_height)
    plt.tight_layout()
    if pp is not None:
        plt.savefig(pp, format='pdf')
    plt.show()


def get_text_positions(x_data, y_data, txt_width, txt_height):
    """
    Get the positions of plot annotations to avoid overlapping
    Source: https://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations
    """
    a = zip(y_data, x_data)
    text_positions = y_data
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height)
                                and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height:  # True means collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    # j is the vertical distance between words
                    if j > txt_height * 1.5:  # if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions


def text_plotter(x_data, y_data, labels, text_positions, axis, txt_width, txt_height):
    """
    Annotate a plot and add an arrow
    Source: https://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations
    """
    for x, y, l, t in zip(x_data, y_data, labels, text_positions):
        axis.text(x - .03, 1.02 * t, '{0}'.format(l), rotation=0, color='black', fontsize=10)
        if y != t:
            axis.arrow(x, t + 20, 0, y-t, color='blue', alpha=0.2, width=txt_width*0.0,
                       head_width=.02, head_length=txt_height*0.5,
                       zorder=0, length_includes_head=True)


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
