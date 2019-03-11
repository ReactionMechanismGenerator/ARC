#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import logging
import numpy as np
import os
import shutil
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

import py3Dmol as p3d
from rdkit import Chem
from IPython.display import display
from ase.visualize import view
from ase import Atom, Atoms
from ase.io import write as ase_write

from rmgpy.data.thermo import ThermoLibrary
from rmgpy.data.kinetics.library import KineticsLibrary
from rmgpy.data.base import Entry
from rmgpy.quantity import ScalarQuantity
from rmgpy.species import Species

from arc.species.species import ARCSpecies
from arc.species.converter import get_xyz_matrix, rdkit_conf_from_mol, molecules_from_xyz
from arc.arc_exceptions import InputError


##################################################################


def draw_3d(xyz=None, species=None, project_directory=None, save_only=False):
    """
    Draws the molecule in a "3D-balls" style
    If xyz ig given, it will be used, otherwise the function looks for species.final_xyz
    Input coordinates are in string format
    Saves an image if a species and `project_directory` are provided
    If `save_only` is ``True``, then don't plot, only save the image
    """
    xyz = check_xyz_species_for_drawing(xyz, species)
    _, atoms, x, y, z = get_xyz_matrix(xyz)
    atoms = [str(a) for a in atoms]
    ase_atoms = list()
    for i, atom in enumerate(atoms):
        ase_atoms.append(Atom(symbol=atom, position=(x[i], y[i], z[i])))
    ase_mol = Atoms(ase_atoms)
    if not save_only:
        display(view(ase_mol, viewer='x3d'))
    if project_directory is not None and species is not None:
        folder_name = 'rxns' if species.is_ts else 'Species'
        geo_path = os.path.join(project_directory, 'output', folder_name, species.label, 'geometry')
        if not os.path.exists(geo_path):
            os.makedirs(geo_path)
        ase_write(filename=os.path.join(geo_path, 'geometry.png'), images=ase_mol, scale=100)


def show_sticks(xyz=None, species=None, project_directory=None):
    """
    Draws the molecule in a "sticks" style according to the supplied xyz coordinates
    Returns whether successful of not
    If successful, save an image using draw_3d
    """
    xyz = check_xyz_species_for_drawing(xyz, species)
    coordinates, _, _, _, _ = get_xyz_matrix(xyz)
    s_mol, b_mol = molecules_from_xyz(xyz)
    mol = b_mol if b_mol is not None else s_mol
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
    draw_3d(xyz=xyz, species=species, project_directory=project_directory, save_only=True)
    return True


def check_xyz_species_for_drawing(xyz, species):
    """A helper function to avoid repetative code"""
    if species is not None and xyz is None:
        xyz = xyz if xyz is not None else species.final_xyz
    if species is not None and not isinstance(species, ARCSpecies):
        raise InputError('Species must be an ARCSpecies instance. Got {0}.'.format(type(species)))
    if species is not None and not species.final_xyz:
        raise InputError('Species {0} has an empty final_xyz attribute.'.format(species.label))
    return xyz


def plot_3d_mol_as_scatter(xyz, path=None, plot_h=True, show_plot=True):
    """
    Draws the molecule as scattered balls in space according to the supplied xyz coordinates
    `xyz` is in string form
    `path` is the species output path to save the image
    """
    xyz, atoms, x, y, z = get_xyz_matrix(xyz)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    colors = []
    sizes = []
    for i, atom in enumerate(atoms):
        size = 500
        if atom == 'H':
            if plot_h:
                colors.append('gray')
                size = 250
            else:
                colors.append('white')
                atoms[i] = ''
                x[i], y[i], z[i] = 0, 0, 0
        elif atom == 'C':
            colors.append('k')
        elif atom == 'N':
            colors.append('b')
        elif atom == 'O':
            colors.append('r')
        elif atom == 'S':
            colors.append('orange')
        else:
            colors.append('g')
        sizes.append(size)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xs=x, ys=y, zs=z, s=sizes, c=colors, depthshade=True)
    for i, atom in enumerate(atoms):
        ax.text(x[i]+0.01, y[i]+0.01, z[i]+0.01, str(atom), size=7)
    plt.axis('off')
    if show_plot:
        plt.show()
    if path is not None:
        image_path = os.path.join(path, "scattered_balls_structure.png")
        plt.savefig(image_path, bbox_inches='tight')


def plot_rotor_scan(angle, v_list, path=None, pivots=None, comment=''):
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
    if path is not None and pivots is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        fig_path = os.path.join(path, '{0}.png'.format(pivots))
        plt.savefig(fig_path, dpi=120, facecolor='w', edgecolor='w', orientation='portrait', papertype=None,
                    format=str('png'), transparent=False, bbox_inches=None, pad_inches=0.1, frameon=False, metadata=None)
        if comment:
            txt_path = os.path.join(path, 'rotor comments.txt')
            if os.path.isfile(txt_path):
                with open(txt_path, 'a') as f:
                    f.write('\n\nPivots: {0}\nComment: {1}'.format(pivots, comment))
            else:
                with open(txt_path, 'w') as f:
                    f.write('Pivots: {0}\nComment: {1}'.format(pivots, comment))


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


def log_kinetics(label, path):
    """
    Logging kinetics from an Arkane output file
    """
    logging.info('\n\n')
    logging.debug('Kinetics for species {0}'.format(label))
    kinetics_block = ''
    log = False
    with open(path, 'r') as f:
        line = f.readline()
        while line != '':
            if 'kinetics(' in line:
                kinetics_block = ''
                log = True
            elif 'thermo(' in line:
                log = False
            if log:
                kinetics_block += line
            line = f.readline()
    logging.info(kinetics_block)
    logging.info('\n')


def draw_thermo_parity_plots(species_list, path=None):
    """
    Draws parity plots of calculated thermo and RMG's best values for species in species_list
    """
    pp = None
    if path is not None:
        thermo_path = os.path.join(path, str('thermo_parity_plots.pdf'))
        if os.path.exists(thermo_path):
            os.remove(thermo_path)
        pp = PdfPages(thermo_path)
    labels, comments, h298_arc, h298_rmg, s298_arc, s298_rmg = [], [], [], [], [], []
    for spc in species_list:
        labels.append(spc.label)
        h298_arc.append(spc.thermo.getEnthalpy(298) * 0.001)  # converted to kJ/mol
        h298_rmg.append(spc.rmg_thermo.getEnthalpy(298) * 0.001)  # converted to kJ/mol
        s298_arc.append(spc.thermo.getEntropy(298))  # in J/mol*K
        s298_rmg.append(spc.rmg_thermo.getEntropy(298))  # in J/mol*K
        comments.append(spc.rmg_thermo.comment)
    draw_parity_plot(var_arc=h298_arc, var_rmg=h298_rmg, var_label='H298', var_units='kJ / mol', labels=labels, pp=pp)
    draw_parity_plot(var_arc=s298_arc, var_rmg=s298_rmg, var_label='S298', var_units='J / mol * K', labels=labels, pp=pp)
    pp.close()
    thermo_sources = '\nSources of thermoproperties determined by RMG for the parity plots:\n'
    max_label_len = max([len(label) for label in labels])
    for i, label in enumerate(labels):
        thermo_sources += '   {0}: {1}{2}\n'.format(label, ' '*(max_label_len - len(label)), comments[i])
    logging.info(thermo_sources)
    if path is not None:
        with open(os.path.join(path, str('thermo.info')), 'w') as f:
            f.write(thermo_sources)


def draw_parity_plot(var_arc, var_rmg, var_label, var_units, labels, pp):
    height = max(len(var_arc) / 3.5, 4)
    width = 8
    min_var = min(var_arc + var_rmg)
    max_var = max(var_arc + var_rmg)
    fig = plt.figure(figsize=(width, height), dpi=120)
    ax = fig.add_subplot(111)
    plt.title('{0} parity plot'.format(var_label))
    for i, label in enumerate(labels):
        plt.plot(var_arc[i], var_rmg[i], 'o', label=label)
    plt.plot([min_var, max_var], [min_var, max_var], 'b-', linewidth=0.5)
    plt.xlabel('{0} calculated by ARC ({1})'.format(var_label, var_units))
    plt.ylabel('{0} determined by RMG ({1})'.format(var_label, var_units))
    plt.xlim = (min_var, max_var * 1.1)
    plt.ylim = (min_var, max_var)
    plt.legend(shadow=False, frameon=False, loc='best')
    # txt_height = 0.04 * (plt.ylim[1] - plt.ylim[0])  # plt.ylim and plt.xlim return a tuple
    # txt_width = 0.02 * (plt.xlim[1] - plt.xlim[0])
    # text_positions = get_text_positions(var_arc, var_rmg, txt_width, txt_height)
    # text_plotter(var_arc, var_rmg, labels, text_positions, ax, txt_width, txt_height)
    plt.tight_layout()
    if pp is not None:
        plt.savefig(pp, format='pdf')
    plt.show()


def draw_kinetics_plots(rxn_list, path=None, t_min=(300, 'K'), t_max=(3000, 'K'), t_count=50):
    """
    Draws plots of calculated rates and RMG's best values for reaction rates in rxn_list
    `rxn_list` has a .kinetics attribute calculated by ARC and an .rmg_reactions list with RMG rates
    """
    plt.style.use(str('seaborn-talk'))
    t_min = ScalarQuantity(value=t_min[0], units=str(t_min[1]))
    t_max = ScalarQuantity(value=t_max[0], units=str(t_max[1]))
    temperature = np.linspace(t_min.value_si, t_max.value_si, t_count)
    pressure = 1e7  # Pa  (=100 bar)

    pp = None
    if path is not None:
        path = os.path.join(path, str('rate_plots.pdf'))
        if os.path.exists(path):
            os.remove(path)
        pp = PdfPages(path)

    for rxn in rxn_list:
        reaction_order = len(rxn.reactants)
        units = ''
        conversion_factor = {1: 1, 2: 1e6, 3: 1e12}
        if reaction_order == 1:
            units = r' (s$^-1$)'
        elif reaction_order == 2:
            units = r' (cm$^3$/(mol s))'
        elif reaction_order == 3:
            units = r' (cm$^6$/(mol$^2$ s))'
        arc_k = list()
        for t in temperature:
            arc_k.append(rxn.kinetics.getRateCoefficient(t, pressure) * conversion_factor[reaction_order])
        rmg_rxns = list()
        for rmg_rxn in rxn.rmg_reactions:
            rmg_rxn_dict = dict()
            rmg_rxn_dict['rmg_rxn'] = rmg_rxn
            rmg_rxn_dict['t_min'] = rmg_rxn.kinetics.Tmin if rmg_rxn.kinetics.Tmin is not None else t_min
            rmg_rxn_dict['t_max'] = rmg_rxn.kinetics.Tmax if rmg_rxn.kinetics.Tmax is not None else t_max
            k = list()
            temp = np.linspace(rmg_rxn_dict['t_min'].value_si, rmg_rxn_dict['t_max'].value_si, t_count)
            for t in temp:
                k.append(rmg_rxn.kinetics.getRateCoefficient(t, pressure) * conversion_factor[reaction_order])
            rmg_rxn_dict['k'] = k
            rmg_rxn_dict['T'] = temp
            if rmg_rxn.kinetics.isPressureDependent():
                rmg_rxn.comment += str(' (at {0} bar)'.format(int(pressure / 1e5)))
            rmg_rxn_dict['label'] = rmg_rxn.comment
            rmg_rxns.append(rmg_rxn_dict)
        _draw_kinetics_plots(rxn.label, arc_k, temperature, rmg_rxns, units, pp)
    pp.close()


def _draw_kinetics_plots(rxn_label, arc_k, temperature, rmg_rxns, units, pp, max_rmg_rxns=5):
    kinetics_library_priority = ['BurkeH2O2inN2', 'Klippenstein_Glarborg2016', 'primaryNitrogenLibrary',
                                 'primarySulfurLibrary', 'N-S_interactions', 'NOx2018',
                                 'Nitrogen_Dean_and_Bozzelli', 'FFCM1(-)', 'JetSurF2.0']
    fig = plt.figure(figsize=(8, 6), dpi=120)
    # plt.rcParams.update({'font.size': 12})
    ax = fig.add_subplot(111)
    plt.title(rxn_label)
    inverse_temperature = [1000 / t for t in temperature]
    ax.semilogy(inverse_temperature, arc_k, 'k--', linewidth=2.5, label='ARC')
    plotted_rmg_rxns = 0
    remaining_rmg_rxns = list()
    for rmg_rxn in rmg_rxns:
        if 'family' in rmg_rxn['label'].lower():
            inverse_temp = [1000 / t for t in rmg_rxn['T']]
            ax.semilogy(inverse_temp, rmg_rxn['k'], label=rmg_rxn['label'])
            plotted_rmg_rxns += 1
        else:
            remaining_rmg_rxns.append(rmg_rxn)
    for priority_lib in kinetics_library_priority:
        for rmg_rxn in remaining_rmg_rxns:
            if priority_lib.lower() in rmg_rxn['label'].lower() and plotted_rmg_rxns <= max_rmg_rxns:
                inverse_temp = [1000 / t for t in rmg_rxn['T']]
                ax.semilogy(inverse_temp, rmg_rxn['k'], label=rmg_rxn['label'])
                plotted_rmg_rxns += 1
    for rmg_rxn in rmg_rxns:
        if plotted_rmg_rxns <= max_rmg_rxns:
            inverse_temp = [1000 / t for t in rmg_rxn['T']]
            ax.semilogy(inverse_temp, rmg_rxn['k'], label=rmg_rxn['label'])
            plotted_rmg_rxns += 1
    plt.xlabel(r'1000 / T (K$^-$$^1$)')
    plt.ylabel('Rate coefficient{0}'.format(units))
    plt.legend()
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
                                and (abs(i[1] - x) < txt_width * 2) and i != (y, x)]
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


def save_geo(species, project_directory):
    """
    Save the geometry in several forms for an ARC Species object in the project's output folder under the species name
    """
    folder_name = 'rxns' if species.is_ts else 'Species'
    geo_path = os.path.join(project_directory, 'output', folder_name, species.label, 'geometry')
    if os.path.exists(geo_path):
        # clean working folder from all previous output
        for file0 in os.listdir(geo_path):
            file_path = os.path.join(geo_path, file0)
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
    gv = '# hf/3-21g\n\n{0} optimized at {1}\n\n'.format(species.label, species.opt_level)
    gv += '{0} {1}\n'.format(species.charge, species.multiplicity)
    gv += '{0}\n'.format(species.final_xyz)
    with open(os.path.join(geo_path, '{0}.gjf'.format(species.label)), 'w') as f:
        f.write(gv)


def save_thermo_lib(species_list, path, name, lib_long_desc):
    """
    Save an RMG thermo library of all species in `species_list` in the supplied `path`
    `name` is the library's name (or project's name)
    `long_desc` is a multiline string with level of theory description
    """
    if species_list:
        lib_path = os.path.join(path, 'thermo', '{0}.py'.format(name))
        thermo_library = ThermoLibrary(name=name, longDesc=lib_long_desc)
        for i, spc in enumerate(species_list):
            if spc.thermo is not None:
                spc.long_thermo_description += '\nExternal symmetry: {0}, optical isomers: {1}\n'.format(
                    spc.external_symmetry, spc.optical_isomers)
                spc.long_thermo_description += '\nGeometry:\n{0}'.format(spc.final_xyz)
                thermo_library.loadEntry(index=i+1,
                                         label=spc.label,
                                         molecule=spc.mol_list[0].toAdjacencyList(),
                                         thermo=spc.thermo,
                                         shortDesc=spc.thermo.comment,
                                         longDesc=spc.long_thermo_description)
            else:
                logging.warning('Species {0} did not contain any thermo data and was omitted from the thermo'
                                ' library.'.format(str(spc)))

        thermo_library.save(lib_path)


def save_kinetics_lib(rxn_list, path, name, lib_long_desc):
    """
    Save an RMG kinetics library of all reactions in `rxn_list` in the supplied `path`
    `rxn_list` is a list of ARCReaction objects
    `name` is the library's name (or project's name)
    `long_desc` is a multiline string with level of theory description
    """
    entries = dict()
    if rxn_list:
        for i, rxn in enumerate(rxn_list):
            if rxn.kinetics is not None:
                if len(rxn.rmg_reaction.reactants):
                    reactants = rxn.rmg_reaction.reactants
                    products = rxn.rmg_reaction.products
                elif rxn.r_species.mol_list is not None:
                    reactants = [Species(molecule=arc_spc.mol_list) for arc_spc in rxn.r_species]
                    products = [Species(molecule=arc_spc.mol_list) for arc_spc in rxn.p_species]
                elif rxn.r_species.mol is not None:
                    reactants = [Species(molecule=[arc_spc.mol]) for arc_spc in rxn.r_species]
                    products = [Species(molecule=[arc_spc.mol]) for arc_spc in rxn.p_species]
                else:
                    reactants = [Species(molecule=[arc_spc.xyz_mol]) for arc_spc in rxn.r_species]
                    products = [Species(molecule=[arc_spc.xyz_mol]) for arc_spc in rxn.p_species]
                rxn.rmg_reaction.reactants = reactants
                rxn.rmg_reaction.products = products
                entry = Entry(
                    index=i+1,
                    item=rxn.rmg_reaction,
                    data=rxn.kinetics,
                    label=rxn.label)
                rxn.ts_species.make_ts_report()
                entry.longDesc = rxn.ts_species.ts_report + '\n\nOptimized TS geometry:\n' + rxn.ts_species.final_xyz
                rxn.rmg_reaction.kinetics = rxn.kinetics
                rxn.rmg_reaction.kinetics.comment = str('')
                entries[i+1] = entry
            else:
                logging.warning('Reaction {0} did not contain any kinetic data and was omitted from the kinetics'
                                ' library.'.format(rxn.label))
        kinetics_library = KineticsLibrary(name=name, longDesc=lib_long_desc, autoGenerated=True)
        kinetics_library.entries = entries
        lib_path = os.path.join(path, 'kinetics', '')
        if os.path.exists(lib_path):
            shutil.rmtree(lib_path)
        try:
            os.makedirs(lib_path)
        except OSError:
            pass
        kinetics_library.save(os.path.join(lib_path, 'reactions.py'))
        kinetics_library.saveDictionary(os.path.join(lib_path, 'dictionary.txt'))
