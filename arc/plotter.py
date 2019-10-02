#!/usr/bin/env python3
# encoding: utf-8

"""
A module for plotting and saving output files such as RMG libraries.
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
# This must be called before pylab, matplotlib.pyplot, or matplotlib.backends is imported.
# Do not warn if the backend has already been set, e.g., when running from an IPython notebook.
matplotlib.use('Agg', warn=False, force=False)
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

import py3Dmol as p3D
from ase import Atom, Atoms
from ase.io import write as ase_write
from ase.visualize import view
from IPython.display import display
from rdkit import Chem

from rmgpy.data.base import Entry
from rmgpy.data.kinetics.library import KineticsLibrary
from rmgpy.data.thermo import ThermoLibrary
from rmgpy.data.transport import TransportLibrary
from rmgpy.quantity import ScalarQuantity
from rmgpy.species import Species

from arc.common import get_logger, min_list, save_yaml_file
from arc.exceptions import InputError, SanitizationError
from arc.species.converter import rdkit_conf_from_mol, molecules_from_xyz, check_xyz_dict, str_to_xyz, xyz_to_str, \
    xyz_to_x_y_z, xyz_from_data
from arc.species.species import ARCSpecies


logger = get_logger()


# *** Drawings species ***

def draw_structure(xyz=None, species=None, project_directory=None, method='show_sticks'):
    """
    A helper function for drawing a molecular structure using either show_sticks or draw_3d.

    Args:
        xyz (str, optional): The xyz coordinates to plot in string format.
        species (ARCSpecies, optional): A species from which to extract the xyz coordinates to plot.
        project_directory (str, optional): A directory for saving the image (only supported for draw_3d).
        method (str, optional): The method to use, either show_sticks or draw_3d.
    """
    success = False
    if method == 'show_sticks':
        try:
            success = show_sticks(xyz=xyz, species=species, project_directory=project_directory)
        except (IndexError, InputError):
            pass
    if not success or method == 'draw_3d':
        draw_3d(xyz=xyz, species=species, project_directory=project_directory)


def show_sticks(xyz=None, species=None, project_directory=None):
    """
    Draws the molecule in a "sticks" style according to the supplied xyz coordinates.
    Returns whether successful of not. If successful, saves the image using draw_3d.
    Either ``xyz`` or ``species`` must be specified.

    Args:
        xyz (str, dict, optional): The coordinates to display.
        species (ARCSpecies, optional): xyz coordinates will be taken from the species.
        project_directory (str): ARC's project directory to save a draw_3d image in.

    Returns:
        bool: Whether the show_sticks drawing was successfull. ``True`` if it was.
    """
    xyz = check_xyz_species_for_drawing(xyz, species)
    if species is None:
        s_mol, b_mol = molecules_from_xyz(xyz)
        mol = b_mol if b_mol is not None else s_mol
    else:
        mol = species.mol
    try:
        rd_mol = rdkit_conf_from_mol(mol, xyz)[1]
    except (ValueError, AttributeError):
        return False
    mb = Chem.MolToMolBlock(rd_mol)
    p = p3D.view(width=400, height=400)
    p.addModel(mb, 'sdf')
    p.setStyle({'stick': {}})
    # p.setBackgroundColor('0xeeeeee')
    p.zoomTo()
    p.show()
    if project_directory is not None:
        draw_3d(xyz=xyz, species=species, project_directory=project_directory, save_only=True)
    return True


def draw_3d(xyz=None, species=None, project_directory=None, save_only=False):
    """
    Draws the molecule in a "3D-balls" style.
    Saves an image if a species and ``project_directory`` are provided.

    Args:
        xyz (str, dict, optional): The coordinates to display.
        species (ARCSpecies, optional): xyz coordinates will be taken from the species.
        project_directory (str): ARC's project directory to save the image in.
        save_only (bool): Whether to only save an image without plotting it, ``True`` to only save.
    """
    xyz = check_xyz_species_for_drawing(xyz, species)
    ase_atoms = list()
    for symbol, coord in zip(xyz['symbols'], xyz['coords']):
        ase_atoms.append(Atom(symbol=symbol, position=coord))
    ase_mol = Atoms(ase_atoms)
    if not save_only:
        display(view(ase_mol, viewer='x3d'))
    if project_directory is not None and species is not None:
        folder_name = 'rxns' if species.is_ts else 'Species'
        geo_path = os.path.join(project_directory, 'output', folder_name, species.label, 'geometry')
        if not os.path.exists(geo_path):
            os.makedirs(geo_path)
        ase_write(filename=os.path.join(geo_path, 'geometry.png'), images=ase_mol, scale=100)


def plot_3d_mol_as_scatter(xyz, path=None, plot_h=True, show_plot=True, name=''):
    """
    Draws the molecule as scattered balls in space according to the supplied xyz coordinates.

    Args:
        xyz (dict, str): The xyz coordinates.
        path (str, optional): A directory path to save the generated figure in.
        plot_h (bool, optional): Whether to plot hydrogen atoms as well. ``True`` to plot them.
        show_plot (bool, optional): Whether to show the plot. ``True`` to show.
        name (str, optional): A name to be added to the saved file name.
    """
    xyz = check_xyz_species_for_drawing(xyz=xyz)
    coords, symbols, colors, sizes = list(), list(), list(), list()
    for symbol, coord in zip(xyz['symbols'], xyz['coords']):
        size = 500
        if symbol == 'H':
            color = 'gray'
            size = 250
        elif symbol == 'C':
            color = 'k'
        elif symbol == 'N':
            color = 'b'
        elif symbol == 'O':
            color = 'r'
        elif symbol == 'S':
            color = 'orange'
        else:
            color = 'g'
        if not (symbol == 'H' and not plot_h):
            # we do want to plot this atom
            coords.append(coord)
            symbols.append(symbol)
            colors.append(color)
            sizes.append(size)

    xyz_ = xyz_from_data(coords=coords, symbols=symbols)
    x, y, z = xyz_to_x_y_z(xyz_)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xs=x, ys=y, zs=z, s=sizes, c=colors, depthshade=True)
    for i, symbol in enumerate(symbols):
        ax.text(x[i]+0.01, y[i]+0.01, z[i]+0.01, symbol, size=7)
    plt.axis('off')
    if show_plot:
        plt.show()
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)
        image_path = os.path.join(path, 'scattered_balls_structure{0}.png'.format(name))
        plt.savefig(image_path, bbox_inches='tight')


def check_xyz_species_for_drawing(xyz=None, species=None):
    """
    A helper function for checking the coordinates before drawing them.
    Either ``xyz`` or ``species`` must be given. If both are given, ``xyz`` gets precedence.
    If ``species`` is given, xys will be taken from it or cheaply generated for it.

    Args:
        xyz (dict, str, optional): The 3D coordinates in any form.
        species (ARCSpecies, optional): A species to take the coordinates from.

    Returns:
        xyz (dict): The coordinates to plot.

    Raises:
        InputError: If neither ``xyz`` nor ``species`` are given.
        TypeError: If ``species`` is of wrong type.
    """
    if xyz is None and species is None:
        raise InputError('Either xyz or species must be given.')
    if species is not None and not isinstance(species, ARCSpecies):
        raise TypeError('Species must be an ARCSpecies instance. Got {0}.'.format(type(species)))
    if xyz is not None:
        if isinstance(xyz, str):
            xyz = str_to_xyz(xyz)
    else:
        xyz = species.get_xyz(generate=True)
    return check_xyz_dict(xyz)


# *** Logging output ***

def log_thermo(label, path):
    """
    Logging thermodata from an Arkane output file.

    Args:
        label (str): The species label.
        path (str): The path to the folder containing the relevant Arkane output file.
    """
    logger.info('\n\n')
    logger.debug('Thermodata for species {0}'.format(label))
    thermo_block = ''
    log = False
    with open(os.path.join(path, 'output.py'), 'r') as f:
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
    logger.info(thermo_block)
    logger.info('\n')


def log_kinetics(label, path):
    """
    Logging kinetics from an Arkane output file.

    Args:
        label (str): The species label.
        path (str): The path to the folder containing the relevant Arkane output file.
    """
    logger.info('\n\n')
    logger.debug('Kinetics for species {0}'.format(label))
    kinetics_block = ''
    log = False
    with open(os.path.join(path, 'output.py'), 'r') as f:
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
    logger.info(kinetics_block)
    logger.info('\n')


def log_bde_report(path, bde_report):
    """
    Prettify the report for bond dissociation energies. Log and save to file.

    Args:
        path (str): The file path.
        bde_report (dict): The BDE report dictionary. Keys are species labels, values are species BDE dictionaries.
                           In the second level dict, keys are pivot tuples, values are energies in kJ/mol.
    """
    with open(path, 'w') as f:
        content = ''
        for label, bde_dict in bde_report.items():
            content += ' BDE report for {0}:\n'.format(label)
            content += ' Pivots        BDE (kJ/mol)\n'
            content += ' ------        ------------\n'
            for pivots, bde in bde_dict.items():
                pivots_str = f'({pivots[0]}, {pivots[1]})'
                if isinstance(bde, str):
                    content += ' {0:15} {1:13}\n'.format(pivots_str, bde)
                elif isinstance(bde, float):
                    content += ' {0:15} {1:10.2f}\n'.format(pivots_str, bde)
            content += '\n\n'
        logger.info('\n\n')
        logger.info(content)
        f.write(content)


# *** Parity and kinetic plots ***

def draw_thermo_parity_plots(species_list, path=None):
    """
    Draws parity plots of calculated thermo and RMG's best values for species in species_list.
    """
    pp = None
    if path is not None:
        thermo_path = os.path.join(path, 'thermo_parity_plots.pdf')
        if os.path.exists(thermo_path):
            os.remove(thermo_path)
        pp = PdfPages(thermo_path)
    labels, comments, h298_arc, h298_rmg, s298_arc, s298_rmg = [], [], [], [], [], []
    for spc in species_list:
        labels.append(spc.label)
        h298_arc.append(spc.thermo.get_enthalpy(298) * 0.001)  # converted to kJ/mol
        h298_rmg.append(spc.rmg_thermo.get_enthalpy(298) * 0.001)  # converted to kJ/mol
        s298_arc.append(spc.thermo.get_entropy(298))  # in J/mol*K
        s298_rmg.append(spc.rmg_thermo.get_entropy(298))  # in J/mol*K
        comments.append(spc.rmg_thermo.comment)
    draw_parity_plot(var_arc=h298_arc, var_rmg=h298_rmg, var_label='H298', var_units='kJ / mol', labels=labels, pp=pp)
    draw_parity_plot(var_arc=s298_arc, var_rmg=s298_rmg, var_label='S298', var_units='J / mol * K', labels=labels,
                     pp=pp)
    pp.close()
    thermo_sources = '\nSources of thermoproperties determined by RMG for the parity plots:\n'
    max_label_len = max([len(label) for label in labels])
    for i, label in enumerate(labels):
        thermo_sources += '   {0}: {1}{2}\n'.format(label, ' '*(max_label_len - len(label)), comments[i])
    logger.info(thermo_sources)
    if path is not None:
        with open(os.path.join(path, 'thermo.info'), 'w') as f:
            f.write(thermo_sources)


def draw_parity_plot(var_arc, var_rmg, var_label, var_units, labels, pp):
    """
    Draw a parity plot.
    """
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
    plt.legend(shadow=False, loc='best')
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
    `rxn_list` has a .kinetics attribute calculated by ARC and an .rmg_reactions list with RMG rates.
    """
    plt.style.use('seaborn-talk')
    t_min = ScalarQuantity(value=t_min[0], units=t_min[1])
    t_max = ScalarQuantity(value=t_max[0], units=t_max[1])
    temperature = np.linspace(t_min.value_si, t_max.value_si, t_count)
    pressure = 1e7  # Pa  (=100 bar)

    pp = None
    if path is not None:
        path = os.path.join(path, 'rate_plots.pdf')
        if os.path.exists(path):
            os.remove(path)
        pp = PdfPages(path)

    for rxn in rxn_list:
        if rxn.kinetics is not None:
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
                arc_k.append(rxn.kinetics.get_rate_coefficient(t, pressure) * conversion_factor[reaction_order])
            rmg_rxns = list()
            for rmg_rxn in rxn.rmg_reactions:
                rmg_rxn_dict = dict()
                rmg_rxn_dict['rmg_rxn'] = rmg_rxn
                rmg_rxn_dict['t_min'] = rmg_rxn.kinetics.Tmin if rmg_rxn.kinetics.Tmin is not None else t_min
                rmg_rxn_dict['t_max'] = rmg_rxn.kinetics.Tmax if rmg_rxn.kinetics.Tmax is not None else t_max
                k = list()
                temp = np.linspace(rmg_rxn_dict['t_min'].value_si, rmg_rxn_dict['t_max'].value_si, t_count)
                for t in temp:
                    k.append(rmg_rxn.kinetics.get_rate_coefficient(t, pressure) * conversion_factor[reaction_order])
                rmg_rxn_dict['k'] = k
                rmg_rxn_dict['T'] = temp
                if rmg_rxn.kinetics.is_pressure_dependent():
                    rmg_rxn.comment += ' (at {0} bar)'.format(int(pressure / 1e5))
                rmg_rxn_dict['label'] = rmg_rxn.comment
                rmg_rxns.append(rmg_rxn_dict)
            _draw_kinetics_plots(rxn.label, arc_k, temperature, rmg_rxns, units, pp)

    if path is not None:
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
    Get the positions of plot annotations to avoid overlapping.
    Source: `stackoverflow <https://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations>`_.
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
    Annotate a plot and add an arrow.
    Source: `stackoverflow <https://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations>`_.
    """
    for x, y, l, t in zip(x_data, y_data, labels, text_positions):
        axis.text(x - .03, 1.02 * t, '{0}'.format(l), rotation=0, color='black', fontsize=10)
        if y != t:
            axis.arrow(x, t + 20, 0, y-t, color='blue', alpha=0.2, width=txt_width*0.0,
                       head_width=.02, head_length=txt_height*0.5,
                       zorder=0, length_includes_head=True)


def save_geo(species, project_directory):
    """
    Save the geometry in several forms for an ARC Species object in the project's output folder under the species name.
    """
    folder_name = 'rxns' if species.is_ts else 'Species'
    geo_path = os.path.join(project_directory, 'output', folder_name, species.label, 'geometry')
    if not os.path.exists(geo_path):
        os.makedirs(geo_path)

    xyz_str = xyz_to_str(species.final_xyz)

    # xyz
    xyz = '{0}\n'.format(species.number_of_atoms)
    xyz += '{0} optimized at {1}\n'.format(species.label, species.opt_level)
    xyz += '{0}\n'.format(xyz_str)
    with open(os.path.join(geo_path, '{0}.xyz'.format(species.label)), 'w') as f:
        f.write(xyz)

    # GaussView file
    gv = '# hf/3-21g\n\n{0} optimized at {1}\n\n'.format(species.label, species.opt_level)
    gv += '{0} {1}\n'.format(species.charge, species.multiplicity)
    gv += '{0}\n'.format(xyz_str)
    with open(os.path.join(geo_path, '{0}.gjf'.format(species.label)), 'w') as f:
        f.write(gv)


# *** Files (libraries, xyz, conformers) ***

def save_thermo_lib(species_list, path, name, lib_long_desc):
    """
    Save an RMG thermo library of all species in `species_list` in the supplied `path`.
    `name` is the library's name (or project's name).
    `long_desc` is a multiline string with level of theory description.
    """
    if species_list:
        lib_path = os.path.join(path, 'thermo', '{0}.py'.format(name))
        thermo_library = ThermoLibrary(name=name, long_desc=lib_long_desc)
        for i, spc in enumerate(species_list):
            if spc.thermo is not None:
                spc.long_thermo_description += '\nExternal symmetry: {0}, optical isomers: {1}\n'.format(
                    spc.external_symmetry, spc.optical_isomers)
                spc.long_thermo_description += '\nGeometry:\n{0}'.format(spc.final_xyz)
                thermo_library.load_entry(index=i,
                                          label=spc.label,
                                          molecule=spc.mol_list[0].to_adjacency_list(),
                                          thermo=spc.thermo,
                                          shortDesc=spc.thermo.comment,
                                          longDesc=spc.long_thermo_description)
            else:
                logger.warning('Species {0} did not contain any thermo data and was omitted from the thermo '
                               'library.'.format(spc.label))

        thermo_library.save(lib_path)


def save_transport_lib(species_list, path, name, lib_long_desc=''):
    """
    Save an RMG transport library of all species in `species_list` in the supplied `path`.
    `name` is the library's name (or project's name).
    `long_desc` is a multiline string with level of theory description.
    """
    if species_list:
        lib_path = os.path.join(path, 'transport', '{0}.py'.format(name))
        transport_library = TransportLibrary(name=name, long_desc=lib_long_desc)
        for i, spc in enumerate(species_list):
            if spc.transport_data is not None:
                description = '\nGeometry:\n{0}'.format(spc.final_xyz)
                transport_library.load_entry(index=i,
                                             label=spc.label,
                                             molecule=spc.mol_list[0].to_adjacency_list(),
                                             transport=spc.transport_data,
                                             shortDesc=spc.thermo.comment,
                                             longDesc=description)
                logger.info('\n\nTransport properties for {0}:'.format(spc.label))
                logger.info('  Shape index: {0}'.format(spc.transport_data.shapeIndex))
                logger.info('  Epsilon: {0}'.format(spc.transport_data.epsilon))
                logger.info('  Sigma: {0}'.format(spc.transport_data.sigma))
                logger.info('  Dipole moment: {0}'.format(spc.transport_data.dipoleMoment))
                logger.info('  Polarizability: {0}'.format(spc.transport_data.polarizability))
                logger.info('  Rotational relaxation collision number: {0}'.format(spc.transport_data.rotrelaxcollnum))
                logger.info('  Comment: {0}'.format(spc.transport_data.comment))
            else:
                logger.warning('Species {0} did not contain any thermo data and was omitted from the thermo '
                               'library.'.format(spc.label))

        transport_library.save(lib_path)


def save_kinetics_lib(rxn_list, path, name, lib_long_desc):
    """
    Save an RMG kinetics library of all reactions in `rxn_list` in the supplied `path`.
    `rxn_list` is a list of ARCReaction objects.
    `name` is the library's name (or project's name).
    `long_desc` is a multiline string with level of theory description.
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
                    index=i,
                    item=rxn.rmg_reaction,
                    data=rxn.kinetics,
                    label=rxn.label)
                rxn.ts_species.make_ts_report()
                entry.long_desc = rxn.ts_species.ts_report + '\n\nOptimized TS geometry:\n' + rxn.ts_species.final_xyz
                rxn.rmg_reaction.kinetics = rxn.kinetics
                rxn.rmg_reaction.kinetics.comment = ''
                entries[i+1] = entry
            else:
                logger.warning('Reaction {0} did not contain any kinetic data and was omitted from the kinetics'
                               ' library.'.format(rxn.label))
        kinetics_library = KineticsLibrary(name=name, long_desc=lib_long_desc, auto_generated=True)
        kinetics_library.entries = entries
        lib_path = os.path.join(path, 'kinetics', '')
        if os.path.exists(lib_path):
            shutil.rmtree(lib_path)
        try:
            os.makedirs(lib_path)
        except OSError:
            pass
        kinetics_library.save(os.path.join(lib_path, 'reactions.py'))
        kinetics_library.save_dictionary(os.path.join(lib_path, 'dictionary.txt'))


def save_conformers_file(project_directory, label, xyzs, level_of_theory, multiplicity=None, charge=None, is_ts=False,
                         energies=None, ts_methods=None):
    """
    Save the conformers before or after optimization.
    If energies are given, the conformers are considered to be optimized.

    Args:
        project_directory (str): The path to the project's directory.
        label (str): The species label.
        xyzs (list): Entries are dict-format xyz coordinates of conformers.
        level_of_theory (str): The level of theory used for the conformers optimization.
        multiplicity (int, optional): The species multiplicity, used for perceiving the molecule.
        charge (int, optional): The species charge, used for perceiving the molecule.
        is_ts (bool, optional): Whether the species represents a TS. True if it does.
        energies (list, optional): Entries are energies corresponding to the conformer list in kJ/mol.
                                   If not given (None) then the Species.conformer_energies are used instead.
        ts_methods (list, optional): Entries are method names used to generate the TS guess.
    """
    spc_dir = 'rxns' if is_ts else 'Species'
    geo_dir = os.path.join(project_directory, 'output', spc_dir, label, 'geometry', 'conformers')
    if not os.path.exists(geo_dir):
        os.makedirs(geo_dir)
    if energies is not None and any(e is not None for e in energies):
        optimized = True
        min_e = min_list(energies)
        conf_path = os.path.join(geo_dir, 'conformers_after_optimization.txt')
    else:
        optimized = False
        conf_path = os.path.join(geo_dir, 'conformers_before_optimization.txt')
    with open(conf_path, 'w') as f:
        content = ''
        if optimized:
            content += 'Conformers for {0}, optimized at the {1} level:\n\n'.format(label, level_of_theory)
        for i, xyz in enumerate(xyzs):
            content += 'conformer {0}:\n'.format(i)
            if xyz is not None:
                content += xyz_to_str(xyz) + '\n'
                if not is_ts:
                    try:
                        b_mol = molecules_from_xyz(xyz, multiplicity=multiplicity, charge=charge)[1]
                    except SanitizationError:
                        b_mol = None
                    smiles = b_mol.to_smiles() if b_mol is not None else 'Could not perceive molecule'
                    content += '\nSMILES: {0}\n'.format(smiles)
                elif ts_methods is not None:
                    content += 'TS guess method: {0}\n'.format(ts_methods[i])
                if optimized:
                    if energies[i] == min_e:
                        content += 'Relative Energy: 0 kJ/mol (lowest)'
                    elif energies[i] is not None:
                        content += 'Relative Energy: {0:.3f} kJ/mol'.format(energies[i] - min_e)
            else:
                # Failed to converge
                if is_ts and ts_methods is not None:
                    content += 'TS guess method: ' + ts_methods[i] + '\n'
                content += 'Failed to converge'
            content += '\n\n\n'
        f.write(content)


# *** Torsions ***

def plot_torsion_angles(torsion_angles, torsions_sampling_points=None, wells_dict=None, e_conformers=None,
                        de_threshold=5.0, plot_path=None):
    """
    Plot the torsion angles of the generated conformers.

    Args:
        torsion_angles (dict): Keys are torsions, values are lists of corresponding angles.
        torsions_sampling_points (dict, optional): Keys are torsions, values are sampling points.
        wells_dict (dict, optional): Keys are torsions, values are lists of wells.
                                     Each entry in such a list is a well dictionary with the following keys:
                                     ``start_idx``, ``end_idx``, ``start_angle``, ``end_angle``, and ``angles``.
        e_conformers (list, optional): Entries are conformers corresponding to the sampling points with FF energies.
        de_threshold (float, optional): Energy threshold, plotted as a dashed horizontal line.
        plot_path (str, optional): The path for saving the plot.
    """
    num_comb = None
    torsions = list(torsion_angles.keys()) if torsions_sampling_points is None \
        else list(torsions_sampling_points.keys())
    ticks = [0, 60, 120, 180, 240, 300, 360]
    sampling_points = dict()
    if torsions_sampling_points is not None:
        for tor, points in torsions_sampling_points.items():
            sampling_points[tor] = [point if point <= 360 else point - 360 for point in points]
    if not torsions:
        return
    if len(torsions) == 1:
        torsion = torsions[0]
        fig, axs = plt.subplots(nrows=len(torsions), ncols=1, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
        fig.dpi = 120
        axs.plot(np.array(torsion_angles[tuple(torsion)]),
                 np.zeros_like(np.arange(len(torsion_angles[tuple(torsion)]))), 'g.')
        if torsions_sampling_points is not None:
            axs.plot(np.array(sampling_points[tuple(torsion)]),
                     np.zeros_like(np.arange(len(sampling_points[tuple(torsion)]))), 'ro', alpha=0.35, ms=7)
        axs.frameon = False
        axs.set_ylabel(str(torsion), labelpad=10)
        axs.set_yticklabels(['' for _ in range(len(torsions))])
        axs.tick_params(axis='y',         # changes apply to the x-axis
                        which='both',     # both major and minor ticks are affected
                        left=False,       # ticks along the bottom edge are off
                        right=False,      # ticks along the top edge are off
                        labelleft=False)  # labels along the bottom edge are off
        axs.set_title('Dihedral angle (degrees)')
        axs.axes.xaxis.set_ticks(ticks=ticks)
        fig.set_size_inches(8, 2)
    else:
        fig, axs = plt.subplots(nrows=len(torsions), ncols=1, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
        fig.dpi = 120
        num_comb = 1
        for i, torsion in enumerate(torsions):
            axs[i].plot(np.array(torsion_angles[tuple(torsion)]),
                        np.zeros_like(np.arange(len(torsion_angles[tuple(torsion)]))), 'g.')
            if wells_dict is not None:
                for well in wells_dict[torsion]:
                    axs[i].plot(well['start_angle'] if well['start_angle'] <= 360 else well['start_angle'] - 360, 0,
                                'b|', alpha=0.5)
                    axs[i].plot(well['end_angle'] if well['end_angle'] <= 360 else well['end_angle'] - 360, 0,
                                'k|', alpha=0.5)
            if torsions_sampling_points is not None:
                x, y = list(), list()
                h_line = False
                if e_conformers is not None:
                    for dihedral in sampling_points[tuple(torsion)]:
                        for e_conformer in e_conformers[tuple(torsion)]:
                            if 'FF energy' in e_conformer and e_conformer['FF energy'] is not None \
                                    and 'dihedral' in e_conformer and e_conformer['dihedral'] is not None \
                                    and (abs(dihedral - e_conformer['dihedral']) < 0.1
                                         or abs(dihedral - e_conformer['dihedral'] + 360) < 0.1):
                                x.append(dihedral)
                                y.append(e_conformer['FF energy'])
                                break
                    min_y = min(y)
                    y = [round(yi - min_y, 3) for yi in y]
                    num_comb *= len([yi for yi in y if yi < de_threshold])
                    if any([yi > de_threshold for yi in y]):
                        h_line = True
                else:
                    x = sampling_points[torsion]
                    y = [0.0] * len(sampling_points[tuple(torsion)])
                axs[i].plot(x, y, 'ro', alpha=0.35, ms=7)
                if h_line:
                    x_h = [0, 360]
                    y_h = [de_threshold, de_threshold]
                    axs[i].plot(x_h, y_h, '--k', alpha=0.30, linewidth=0.8)
            axs[i].frameon = False
            axs[i].set_ylabel(str(torsion), labelpad=10)
            # axs[i].yaxis.label.set_rotation(0)
            if e_conformers is None:
                axs[i].set_yticklabels(['' for _ in range(len(torsions))])
                axs[i].tick_params(axis='y',         # changes apply to the x-axis
                                   which='both',     # both major and minor ticks are affected
                                   left=False,       # ticks along the bottom edge are off
                                   right=False,      # ticks along the top edge are off
                                   labelleft=False)  # labels along the bottom edge are off
        axs[0].set_title('Dihedral angle (degrees)')
        # Hide x labels and tick labels for all but bottom plot.
        # for ax in axs:
        #     ax.label_outer()
        axs[0].axes.xaxis.set_ticks(ticks=ticks)
        fig.set_size_inches(8, len(torsions) * 1.5)
    plt.show()
    if plot_path is not None:
        if not os.path.isdir(plot_path):
            os.makedirs(plot_path)
        file_names = list()
        for (_, _, files) in os.walk(plot_path):
            file_names.extend(files)
            break  # don't continue to explore subdirectories
        i = 0
        for file_ in file_names:
            if 'conformer torsions' in file_:
                i += 1
        image_path = os.path.join(plot_path, 'conformer torsions {0}.png'.format(i))
        plt.savefig(image_path, bbox_inches='tight')
    return num_comb


def plot_1d_rotor_scan(angles=None, energies=None, results=None, path=None, pivots=None, comment='', units='degrees'):
    """
    Plots a 1D rotor PES for energy vs. angles. Either ``angles`` and ``energies`` or ``results`` must be given.

    Args:
        angles (list, tuple, np.array, optional): Dihedral angles.
        energies (list, tuple, np.array, optional): The energies in kJ/mol.
        results (dict, optional): The results dictionary, dihedrals are assumed to be in degrees (not radians).
        path (str, optional): The folder path for saving the rotor scan image and comments.
        pivots (list, tuple, optional): The pivotal atoms of the scan.
        comment (str, optional): Reason for invalidating this rotor.
        units (str, optional): The ``angle`` units, either 'degrees' or 'radians'.

    Raises:
        InputError: If neither `angles`` and ``energies`` nor ``results`` were given.
    """
    if (angles is None or energies is None) and results is None:
        raise InputError('Either angles and energies or results must be given')
    if results is not None:
        energies = np.zeros(shape=(len(results['directed_scan'].keys())), dtype=np.float64)
        for i, key in enumerate(results['directed_scan'].keys()):
            energies[i] = results['directed_scan'][key]['energy']
        if len(list(results['directed_scan'].keys())[0]) == 1:
            # keys represent a single dihedral
            angles = [float(key[0]) for key in results['directed_scan'].keys()]
        else:
            angles = list(range(len(list(results['directed_scan'].keys()))))
    else:
        if units == 'radians':
            angles = angles * 180 / np.pi  # convert radians to degree
        energies = np.array(energies, np.float64)  # in kJ/mol
    marker_color, line_color = plt.cm.viridis([0.1, 0.9])
    plt.figure(figsize=(4, 3), dpi=120)
    plt.subplot(1, 1, 1)
    plt.plot(angles, energies, '.-', markerfacecolor=marker_color,
             markeredgecolor=marker_color, color=line_color)
    plt.xlabel('Dihedral angle (degrees)')
    min_angle = int(np.ceil(min(angles) / 10.0)) * 10
    plt.xlim = (min_angle, min_angle + 360)
    plt.xticks(np.arange(min_angle, min_angle + 361, step=60))
    plt.ylabel('Electronic energy (kJ/mol)')
    plt.tight_layout()
    plt.show()

    if path is not None and pivots is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        if comment:
            fig_name = '{0}-invalid.png'.format(pivots)
            txt_path = os.path.join(path, 'rotor comments.txt')
            if os.path.isfile(txt_path):
                with open(txt_path, 'a') as f:
                    f.write('\n\nPivots: {0}\nComment: {1}'.format(pivots, comment))
            else:
                with open(txt_path, 'w') as f:
                    f.write('Pivots: {0}\nComment: {1}'.format(pivots, comment))
        else:
            fig_name = '{0}.png'.format(pivots)
        fig_path = os.path.join(path, fig_name)
        plt.savefig(fig_path, dpi=120, facecolor='w', edgecolor='w', orientation='portrait', papertype=None,
                    format='png', transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)


def plot_2d_rotor_scan(results, path=None, label='', cmap='Blues', resolution=90):
    """
    Plot a 2D rotor scan.

    Args:
        results (dict): The results dictionary, dihedrals are assumed to be in degrees (not radians).
        path (str, optional): The folder path to save this 2D image.
        label (str, optional): The species label.
        cmap (str, optional): The color map to use. See optional arguments below.
        resolution (int, optional): The image resolution to produce.

    Raises:
        TypeError: If ``results`` if of wrong type.
        InputError: If ``results`` does not represent a 2D rotor.

    Optional arguments for cmap::

        Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r,
        GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired,
        Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r,
        PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r,
        Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu,
        YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r,
        bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r,
        cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r,
        gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot,
        gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma,
        magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r,
        rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r,
        tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r
    """
    if not isinstance(results, dict):
        raise TypeError('results must be a dictionary, got {0}'.format(type(results)))
    if len(results['scans']) != 2:
        raise InputError('results must represent a 2D rotor, got {0}D'.format(len(results['scans'])))

    # phis0 and phis1 correspond to columns and rows in energies, respectively
    phis0 = np.array(sorted(list(set([float(key[0]) for key in results['directed_scan'].keys()]))), np.float64)
    phis1 = np.array(sorted(list(set([float(key[1]) for key in results['directed_scan'].keys()]))), np.float64)
    # If the last phi equals to the first, it is removed by the abive set() call. Bring it back:
    if phis0.size < 360 / (phis0[1] - phis0[0]) + 1:
        phis0 = np.append(phis0, phis0[0])
    if phis1.size < 360 / (phis1[1] - phis1[0]) + 1:
        phis1 = np.append(phis1, phis1[0])
    zero_phi0, zero_phi1 = list(), list()
    energies = np.zeros(shape=(phis0.size, phis1.size), dtype=np.float64)
    for i, phi0 in enumerate(phis0):
        for j, phi1 in enumerate(phis1):
            key = tuple('{0:.2f}'.format(dihedral) for dihedral in [phi0, phi1])
            energies[i, j] = results['directed_scan'][key]['energy']
            if energies[i, j] == 0:
                zero_phi0.append(phi0)
                zero_phi1.append(phi1)

    plt.figure(num=None, figsize=(12, 8), dpi=resolution, facecolor='w', edgecolor='k')

    plt.contourf(phis0, phis1, energies, 20, cmap=cmap)
    plt.colorbar()
    contours = plt.contour(phis0, phis1, energies, 4, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)

    plt.xlabel('Dihedral 1 for {scan} (degrees)'.format(scan=results['scans'][0]))
    plt.ylabel('Dihedral 2 for {scan} (degrees)'.format(scan=results['scans'][1]))
    label = ' for ' + label if label else ''
    plt.title('2D scan energies (kJ/mol){label}'.format(label=label))
    min_x = int(np.ceil(np.min(phis0) / 10.0)) * 10
    plt.xlim = (min_x, min_x + 360)
    plt.xticks(np.arange(min_x, min_x + 361, step=60))
    min_y = int(np.ceil(np.min(phis1) / 10.0)) * 10
    plt.ylim = (min_y, min_y + 360)
    plt.yticks(np.arange(min_y, min_y + 361, step=60))

    plt.plot(zero_phi0, zero_phi1, color='k', marker='D', markersize=12, linewidth=0)  # mark the lowest conformations

    if path is not None:
        fig_name = '{0}_{1}.png'.format(results['directed_scan_type'], results['scans'])
        fig_path = os.path.join(path, fig_name)
        plt.savefig(fig_path, dpi=120, facecolor='w', edgecolor='w', orientation='portrait', papertype=None,
                    format='png', transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)


def save_rotor_text_file(angles, energies, path):
    """
    Save a text file summarizing a rotor scan, useful for brute force scans.

    Args:
        angles (list): The dihedral angles in degrees.
        energies (list): The respective scan energies in kJ/mol.
        path (str): The path of the file to be saved.

    Raises:
        InputError: If energies and angles are not the same length.
    """
    if len(energies) != len(angles):
        raise InputError('energies and angles must be the same length')
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    min_angle = min_list(angles)
    angles = [angle - min_angle for angle in angles]  # set first angle to 0
    if energies:
        lines = ['Angle (degrees)        Energy (kJ/mol)\n']
        for angle, energy in zip(angles, energies):
            lines.append('{0:12.2f} {1:24.3f}\n'.format(angle, energy))
        with open(path, 'w') as f:
            f.writelines(lines)


def save_nd_rotor_yaml(results, path):
    """
    Save a text file summarizing a rotor scan, useful for brute force scans.

    Args:
        results (dict): The respective scan dictionary to save.
        path (str): The path of the file to be saved.
    """
    print('save_nd_rotor_yaml')
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    modified_results = results.copy()  # don't dump floats into a YAML file, it's buggy
    for dihedral_tuple, dihedral_dict in results['directed_scan'].items():
        for key, val in dihedral_dict.items():
            if key == 'energy':
                modified_results['directed_scan'][dihedral_tuple][key] = '{:.2f}'.format(val)
            elif key == 'xyz':
                modified_results['directed_scan'][dihedral_tuple][key] = xyz_to_str(val)
    save_yaml_file(path=path, content=modified_results)
