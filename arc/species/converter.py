#!/usr/bin/env python
# encoding: utf-8

"""
A module for performing various species-related format conversions.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np

import pybel
from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdMT

from arkane.common import symbol_by_number, mass_by_symbol, get_element_mass
from rmgpy.exceptions import AtomTypeError
from rmgpy.molecule.molecule import Atom, Bond, Molecule
from rmgpy.molecule.element import getElement
from rmgpy.species import Species

from arc.common import get_logger
from arc.arc_exceptions import SpeciesError, SanitizationError, InputError
from arc.species.xyz_to_2d import MolGraph


logger = get_logger()


def str_to_xyz(xyz_str):
    """
    Convert a string xyz format with optional Gaussian-style isotope specification, e.g.::

        C(Iso=13)    0.6616514836    0.4027481525   -0.4847382281
        N           -0.6039793084    0.6637270105    0.0671637135
        H           -1.4226865648   -0.4973210697   -0.2238712255
        H           -0.4993010635    0.6531020442    1.0853092315
        H           -2.2115796924   -0.4529256762    0.4144516252
        H           -1.8113671395   -0.3268900681   -1.1468957003

    into the ARC xyz dictionary format, e.g.::

        {'symbols': ('C', 'N', 'H', 'H', 'H', 'H'),
         'isotopes': (13, 14, 1, 1, 1, 1),
         'coords': ((0.6616514836, 0.4027481525, -0.4847382281),
                    (-0.6039793084, 0.6637270105, 0.0671637135),
                    (-1.4226865648, -0.4973210697, -0.2238712255),
                    (-0.4993010635, 0.6531020442, 1.0853092315),
                    (-2.2115796924, -0.4529256762, 0.4144516252),
                    (-1.8113671395, -0.3268900681, -1.1468957003))}

    Args:
        xyz_str (str): The string xyz format to be converted.

    Returns:
        xyz_dict (dict): The ARC xyz format.

    Raises:
        InputError: If input is not a string or does not have four space-separated entries per non empty line.
    """
    if not isinstance(xyz_str, (str, unicode)):
        raise InputError('Expected a string input, got {0}'.format(type(xyz_str)))
    xyz_dict = {'symbols': tuple(), 'isotopes': tuple(), 'coords': tuple()}
    if all([len(line.split()) == 6 for line in xyz_str.splitlines() if line.strip()]):
        # Convert Gaussian output format, e.g., "      1          8           0        3.132319    0.769111   -0.080869"
        # not considering isotopes in this method!
        for line in xyz_str.splitlines():
            if line.strip():
                splits = line.split()
                symbol = symbol_by_number[int(splits[1])]
                coord = (float(splits[3]), float(splits[4]), float(splits[5]))
                xyz_dict['symbols'] += (symbol,)
                xyz_dict['isotopes'] += (get_most_common_isotope_for_element(symbol),)
                xyz_dict['coords'] += (coord,)
    else:
        # this is a "regular" string xyz format, if it has isotope information it will be preserved
        for line in xyz_str.strip().splitlines():
            if line.strip():
                splits = line.split()
                if len(splits) != 4:
                    raise InputError('xyz_str has an incorrect format, expected 4 elements in each line, '
                                     'got "{0}" in:\n{1}'.format(line, xyz_str))
                symbol = splits[0]
                if '(iso=' in symbol.lower():
                    isotope = int(symbol.split('=')[1].strip(')'))
                    symbol = symbol.split('(')[0]
                else:
                    # no specific isotope is specified in str_xyz
                    isotope = get_most_common_isotope_for_element(symbol)
                coord = (float(splits[1]), float(splits[2]), float(splits[3]))
                xyz_dict['symbols'] += (symbol,)
                xyz_dict['isotopes'] += (isotope,)
                xyz_dict['coords'] += (coord,)
    return xyz_dict


def xyz_to_str(xyz_dict, isotope_format=None):
    """
    Convert an ARC xyz dictionary format, e.g.::

        {'symbols': ('C', 'N', 'H', 'H', 'H', 'H'),
         'isotopes': (13, 14, 1, 1, 1, 1),
         'coords': ((0.6616514836, 0.4027481525, -0.4847382281),
                    (-0.6039793084, 0.6637270105, 0.0671637135),
                    (-1.4226865648, -0.4973210697, -0.2238712255),
                    (-0.4993010635, 0.6531020442, 1.0853092315),
                    (-2.2115796924, -0.4529256762, 0.4144516252),
                    (-1.8113671395, -0.3268900681, -1.1468957003))}

    to a string xyz format with optional Gaussian-style isotope specification, e.g.::

        C(Iso=13)    0.6616514836    0.4027481525   -0.4847382281
        N           -0.6039793084    0.6637270105    0.0671637135
        H           -1.4226865648   -0.4973210697   -0.2238712255
        H           -0.4993010635    0.6531020442    1.0853092315
        H           -2.2115796924   -0.4529256762    0.4144516252
        H           -1.8113671395   -0.3268900681   -1.1468957003

    Args:
        xyz_dict (dict): The ARC xyz format to be converted.
        isotope_format (str, optional): The format for specifying the isotope if it is not the most abundant one.
                                        By default, isotopes will not be specified. Currently the only supported
                                        option is 'gaussian'.

    Returns:
        str: The string xyz format.

    Raises:
        InputError: If input is not a dict or does not have all attributes.
    """
    recognized_isotope_formats = ['gaussian']
    if not isinstance(xyz_dict, dict):
        raise InputError('Expected a dictionary input, got {0}'.format(type(xyz_dict)))
    if any([key not in list(xyz_dict.keys()) for key in ['symbols', 'isotopes', 'coords']]):
        raise InputError('Missing keys in the xyz dictionary. Expected to find "symbols", "isotopes", and "coords", '
                         'but got {0} in\n{1}'.format(list(xyz_dict.keys()), xyz_dict))
    if any([len(xyz_dict['isotopes']) != len(xyz_dict['symbols']),
            len(xyz_dict['coords']) != len(xyz_dict['symbols'])]):
        raise InputError('Got different lengths for "symbols", "isotopes", and "coords": {0}, {1}, and {2}, '
                         'respectively, in xyz:\n{3}'.format(len(xyz_dict['symbols']), len(xyz_dict['isotopes']),
                                                             len(xyz_dict['coords']), xyz_dict))
    if any([len(xyz_dict['coords'][i]) != 3 for i in range(len(xyz_dict['coords']))]):
        raise InputError('Expected 3 coordinates for each atom (x, y, and z), got:\n{0}'.format(xyz_dict))
    xyz_list = list()
    for symbol, isotope, coord in zip(xyz_dict['symbols'], xyz_dict['isotopes'], xyz_dict['coords']):
        common_isotope = get_most_common_isotope_for_element(symbol)
        if isotope_format is not None and common_isotope != isotope:
            # consider the isotope number
            if isotope_format == 'gaussian':
                element_with_isotope = '{0}(Iso={1})'.format(symbol, isotope)
                row = '{0:14}'.format(element_with_isotope)
            else:
                raise InputError('Recognized isotope formats for printing are {0}, got: {1}'.format(
                                  recognized_isotope_formats, isotope_format))
        else:
            # don't consider the isotope number
            row = '{0:4}'.format(symbol)
        row += '{0:14.8f}{1:14.8f}{2:14.8f}'.format(*coord)
        xyz_list.append(row)
    return '\n'.join(xyz_list) + '\n'


def xyz_to_x_y_z(xyz_dict):
    """
    Get the X, Y, and Z coordinates separately from the ARC xyz dictionary format.

    Args:
        xyz_dict (dict): The ARC xyz format.

    Returns:
        x (tuple): The X coordinates.
    Returns:
        y (tuple): The Y coordinates.
    Returns:
        z (tuple): The Z coordinates.
    """
    x, y, z = tuple(), tuple(), tuple()
    for coord in xyz_dict['coords']:
        x += (coord[0],)
        y += (coord[1],)
        z += (coord[2],)
    return x, y, z


def xyz_to_xyz_file_format(xyz_dict, comment=''):
    """
    Get the `XYZ file format <https://en.wikipedia.org/wiki/XYZ_file_format>`_ representation
    from the ARC xyz dictionary format.
    This function does not consider isotopes.

    Args:
        xyz_dict (dict): The ARC xyz format.
        comment (str, optional): A comment to be shown in the output's 2nd line.

    Returns:
        str: The XYZ file format.

    Raises:
        InputError: If ``xyz_dict`` is of wrong format or ``comment`` is a multiline string.
    """
    if not isinstance(xyz_dict, dict):
        raise InputError('Expected a dictionary input, got {0} which is a {1}'.format(xyz_dict, type(xyz_dict)))
    if len(comment.splitlines()) > 1:
        raise InputError('The comment attribute cannot be a multiline string, got:\n{0}'.format(list(comment)))
    return str(len(xyz_dict['symbols'])) + '\n' + comment.strip() + '\n' + xyz_to_str(xyz_dict) + '\n'


def xyz_file_format_to_xyz(xyz_file):
    """
    Get the ARC xyz dictionary format from an
    `XYZ file format <https://en.wikipedia.org/wiki/XYZ_file_format>`_ representation.

    Args:
        xyz_file (str): The content of an XYZ file

    Returns:
        dict: The ARC dictionary xyz format.

    Raises:
        InputError: If cannot identify the number of atoms entry, of if it is different that the actual number.
    """
    lines = xyz_file.strip().splitlines()
    if not lines[0].isdigit():
        raise InputError('Cannot identify the number of atoms from the XYZ file format representation. '
                         'Expected a number, got: {0} of type {1}'.format(lines[0], type(lines[0])))
    number_of_atoms = int(lines[0])
    lines = lines[2:]
    if len(lines) != number_of_atoms:
        raise InputError('The actual number of atoms ({0}) does not match the expected number parsed ({1}).'.format(
                          len(lines), number_of_atoms))
    xyz_str = '\n'.join(lines)
    return str_to_xyz(xyz_str)


def xyz_from_data(coords, numbers=None, symbols=None, isotopes=None):
    """
    Get the ARC xyz dictionary format from raw data.
    Either ``numbers`` or ``symbols`` must be specified.
    If ``isotopes`` isn't specified, the most common isotopes will be assumed for all elements.

    Args:
        coords (tuple, list): The xyz coordinates.
        numbers (tuple, list, optional): Element nuclear charge numbers.
        symbols (tuple, list, optional): Element symbols.
        isotopes (tuple, list, optional): Element isotope numbers.

    Returns:
        dict: The ARC dictionary xyz format.

    Raises:
        InputError: If neither ``numbers`` nor ``symbols`` are specified, if both are specified,
        or if the input lengths aren't consistent.
    """
    if isinstance(coords, (list, np.ndarray)):
        coords = tuple(tuple(coord) for coord in coords)
    if numbers is not None and isinstance(numbers, (list, np.ndarray)):
        numbers = tuple(numbers)
    if symbols is not None and isinstance(symbols, list):
        symbols = tuple(symbols)
    if isotopes is not None and isinstance(isotopes, list):
        isotopes = tuple(isotopes)
    if not isinstance(coords, tuple):
        raise InputError('Expected coords to be a tuple, got {0} which is a {1}'.format(coords, type(coords)))
    if numbers is not None and not isinstance(numbers, tuple):
        raise InputError('Expected numbers to be a tuple, got {0} which is a {1}'.format(numbers, type(numbers)))
    if symbols is not None and not isinstance(symbols, tuple):
        raise InputError('Expected symbols to be a tuple, got {0} which is a {1}'.format(symbols, type(symbols)))
    if isotopes is not None and not isinstance(isotopes, tuple):
        raise InputError('Expected isotopes to be a tuple, got {0} which is a {1}'.format(isotopes, type(isotopes)))
    if numbers is None and symbols is None:
        raise InputError('Must set either "numbers" or "symbols". Got neither.')
    if numbers is not None and symbols is not None:
        raise InputError('Must set either "numbers" or "symbols". Got both.')
    if numbers is not None:
        symbols = tuple(symbol_by_number[number] for number in numbers)
    if len(coords) != len(symbols):
        raise InputError('The length of the coordinates ({0}) is different than the length of the numbers/symbols '
                         '({1}).'.format(len(coords), len(symbols)))
    if isotopes is not None and len(coords) != len(isotopes):
        raise InputError('The length of the coordinates ({0}) is different than the length of isotopes '
                         '({1}).'.format(len(coords), len(isotopes)))
    if isotopes is None:
        isotopes = tuple(get_most_common_isotope_for_element(symbol) for symbol in symbols)
    xyz_dict = {'symbols': symbols, 'isotopes': isotopes, 'coords': coords}
    return xyz_dict


def standardize_xyz_string(xyz_str, isotope_format=None):
    """
    A helper function to correct xyz string format input (** string to string **).
    Usually empty lines are added by the user either in the beginning or the end,
    here we remove them along with other common issues.

    Args:
        xyz_str (str): The string xyz format, or a Gaussian output format.
        isotope_format (str, optional): The format for specifying the isotope if it is not the most abundant one.
                                        By default, isotopes will not be specified. Currently the only supported
                                        option is 'gaussian'.

    Returns:
        xyz (str): The string xyz format in standardized format.
    """
    xyz_dict = str_to_xyz(xyz_str)
    return xyz_to_str(xyz_dict=xyz_dict, isotope_format=isotope_format)


def check_xyz_dict(xyz):
    """
    Check that the xyz dictionary entered is valid. If it is a string, correct it.
    If isotopes are not in xyz_dict, common values will be added.

    Args:
        xyz (dict, str): The xyz dictionary.

    Raises:
        TypeError: If xyz_dict is not a dictionary.
        ValueError: If xyz_dict is missing symbols or coords.
    """
    xyz_dict = str_to_xyz(xyz) if isinstance(xyz, (str, unicode)) else xyz
    if not isinstance(xyz_dict, dict):
        raise TypeError('Expected a dictionary, got {0}'.format(type(xyz_dict)))
    if 'symbols' not in list(xyz_dict.keys()):
        raise ValueError('XYZ dictionary is missing symbols. Got:\n{0}'.format(xyz_dict))
    if 'coords' not in list(xyz_dict.keys()):
        raise ValueError('XYZ dictionary is missing coords. Got:\n{0}'.format(xyz_dict))
    if len(xyz_dict['symbols']) != len(xyz_dict['coords']):
        raise ValueError('Got {0} symbols and {1} coordinates:\n{2}'.format(
            len(xyz_dict['symbols']), len(xyz_dict['coords']), xyz_dict))
    if 'isotopes' not in list(xyz_dict.keys()):
        xyz_dict = xyz_from_data(coords=xyz_dict['coords'], symbols=xyz_dict['symbols'])
    if len(xyz_dict['symbols']) != len(xyz_dict['isotopes']):
        raise ValueError('Got {0} symbols and {1} isotopes:\n{2}'.format(
            len(xyz_dict['symbols']), len(xyz_dict['isotopes']), xyz_dict))
    return xyz_dict


def get_most_common_isotope_for_element(element_symbol):
    """
    Get the most common isotope for a given element symbol.

    Args:
        element_symbol (str): The element symbol.

    Returns:
        int: The most common isotope number for the element.
    """
    mass_list = mass_by_symbol[element_symbol]
    if len(mass_list[0]) == 2:
        # isotope contribution is unavailable, just get the first entry
        isotope = mass_list[0][0]
    else:
        # isotope contribution is unavailable, get the most common isotope
        isotope, isotope_contribution = mass_list[0][0], mass_list[0][2]
        for iso in mass_list:
            if iso[2] > isotope_contribution:
                isotope_contribution = iso[2]
                isotope = iso[0]
    return isotope


def xyz_to_pybel_mol(xyz):
    """
    Convert xyz into an Open Babel molecule object.

    Args:
        xyz (dict): ARC's xyz dictionary format.

    Returns:
        pybel_mol (OBmol): An Open Babel molecule.

    Raises:
        InputError: If xyz has a wrong format.
    """
    if not isinstance(xyz, dict):
        raise InputError('xyz must be in the ARC dictionary format, got type: {0}'.format(type(xyz)))
    try:
        pybel_mol = pybel.readstring('xyz', xyz_to_xyz_file_format(xyz))
    except IOError:
        return None
    return pybel_mol


def pybel_to_inchi(pybel_mol):
    """
    Convert an Open Babel molecule object to InChI

    Args:
        pybel_mol (OBmol): An Open Babel molecule.

    Returns:
        inchi (str): The respective InChI representation of the molecule.
    """
    inchi = pybel_mol.write('inchi', opt={'F': None}).strip()  # Add fixed H layer
    return inchi


def rmg_mol_from_inchi(inchi):
    """
    Generate an RMG Molecule object from InChI.

    Args:
        inchi (str): The InChI string.

    Returns:
        Molecule: The respective RMG Molecule object.
    """
    try:
        rmg_mol = Molecule().fromInChI(str(inchi))
    except (AtomTypeError, ValueError, KeyError) as e:
        logger.warning('Got the following Error when trying to create an RMG Molecule object from InChI:'
                       '\n{0}'.format(e.message))
        return None
    return rmg_mol


def elementize(atom):
    """
    Convert the atomType of an RMG:Atom object into its general parent element atomType (e.g., `S4d` into `S`).

    Args:
        atom (Atom): The atom to process.
    """
    atom_type = atom.atomType
    atom_type = [at for at in atom_type.generic if at.label != 'R' and at.label != 'R!H' and 'Val' not in at.label]
    if atom_type:
        atom.atomType = atom_type[0]


def molecules_from_xyz(xyz, multiplicity=None, charge=0):
    """
    Creating RMG:Molecule objects from xyz with correct atom labeling.
    Based on the MolGraph.perceive_smiles method.
    If `multiplicity` is given, the returned species multiplicity will be set to it.

    Args:
        xyz (dict): The ARC dict format xyz coordinates of the species.
        multiplicity (int, optional): The species spin multiplicity.
        charge (int, optional): The species net charge.

    Returns:
        s_mol (Molecule): The respective Molecule object with only single bonds.
    Returns:
        b_mol (Molecule): The respective Molecule object with perceived bond orders.
                          Returns None if unsuccessful to infer bond orders.

    Raises:
        InputError: If xyz has incorrect format.
    """
    if xyz is None:
        return None, None
    if not isinstance(xyz, dict):
        raise InputError('xyz must be a dictionary, got: {0}'.format(type(xyz)))
    xyz = check_xyz_dict(xyz)
    mol_graph = MolGraph(symbols=xyz['symbols'], coords=xyz['coords'])
    inferred_connections = mol_graph.infer_connections()
    if inferred_connections:
        mol_s1 = mol_graph.to_rmg_mol()  # An RMG Molecule with single bonds, atom order corresponds to xyz
    else:
        mol_s1 = s_bonds_mol_from_xyz(xyz)
    if mol_s1 is None:
        logger.error('Could not create a 2D graph representation from xyz:\n{0}'.format(xyz))
        return None, None
    mol_s1_updated = update_molecule(mol_s1, to_single_bonds=True)
    pybel_mol = xyz_to_pybel_mol(xyz)
    if pybel_mol is not None:
        inchi = pybel_to_inchi(pybel_mol)
        mol_bo = rmg_mol_from_inchi(inchi)  # An RMG Molecule with bond orders, but without preserved atom order
        if mol_bo is not None:
            if multiplicity is not None:
                try:
                    set_multiplicity(mol_bo, multiplicity, charge)
                except SpeciesError as e:
                    logger.warning('Cannot infer 2D graph connectivity, failed to set species multiplicity with the '
                                   'following error:\n{0}'.format(e.message))
                    return None, None
            mol_s1_updated.multiplicity = mol_bo.multiplicity
            try:
                order_atoms(ref_mol=mol_s1_updated, mol=mol_bo)
            except SanitizationError:
                logger.warning('Could not order atoms for {0}!'.format(mol_s1_updated.toSMILES()))
            try:
                set_multiplicity(mol_s1_updated, mol_bo.multiplicity, charge, radical_map=mol_bo)
            except SpeciesError as e:
                logger.warning('Cannot infer 2D graph connectivity, failed to set species multiplicity with the '
                               'following error:\n{0}'.format(e.message))
                return mol_s1_updated, None
    else:
        mol_bo = None
    s_mol, b_mol = mol_s1_updated, mol_bo
    return s_mol, b_mol


def set_multiplicity(mol, multiplicity, charge, radical_map=None):
    """
    Set the multiplicity of ``mol`` and change radicals as needed.
    If a `radical_map`, which is an RMG Molecule object with the same atom order, is given,
    it'll be used to set radicals (useful if bond orders aren't known for a molecule).

    Args:
        mol (Molecule): The RMG Molecule object.
        multiplicity (int) The spin multiplicity.
        charge (int): THe species net charge.
        radical_map (Molecule, optional): An RMG Molecule object with the same atom order to be used as a radical map.
    """
    mol.multiplicity = multiplicity
    if radical_map is not None:
        if not isinstance(radical_map, Molecule):
            raise TypeError('radical_map sent to set_multiplicity() has to be a Molecule object. Got {0}'.format(
                type(radical_map)))
        set_radicals_by_map(mol, radical_map)
    radicals = mol.getRadicalCount()
    if mol.multiplicity != radicals + 1:
        # this is not the trivial "multiplicity = number of radicals + 1" case
        # either the number of radicals was not identified correctly from the 3D structure (i.e., should be lone pairs),
        # or their spin isn't determined correctly
        if mol.multiplicity > radicals + 1:
            # there are sites that should have radicals, but were'nt identified as such.
            # try adding radicals according to missing valances
            add_rads_by_atom_valance(mol)
            if mol.multiplicity > radicals + 1:
                # still problematic, currently there's no automated solution to this case, raise an error
                raise SpeciesError('A multiplicity of {0} was given, but only {1} radicals were identified. '
                                   'Cannot infer 2D graph representation for this species.\nMore info:{2}\n{3}'.format(
                                    mol.multiplicity, radicals, mol.toSMILES(), mol.toAdjacencyList()))
        if len(mol.atoms) == 1 and mol.multiplicity == 1 and mol.atoms[0].radicalElectrons == 4:
            # This is a singlet atomic C or Si
            mol.atoms[0].radicalElectrons = 0
            mol.atoms[0].lonePairs = 2
        if mol.multiplicity < radicals + 1:
            # make sure all carbene and nitrene sites, if exist, have lone pairs rather than two unpaired electrons
            for atom in mol.atoms:
                if atom.radicalElectrons == 2:
                    atom.radicalElectrons = 0
                    atom.lonePairs += 1
    # final check: an even number of radicals results in an odd multiplicity, and vice versa
    if divmod(mol.multiplicity, 2)[1] == divmod(radicals, 2)[1]:
        if not charge:
            raise SpeciesError('Number of radicals ({0}) and multiplicity ({1}) for {2} do not match.\n{3}'.format(
                radicals, mol.multiplicity, mol.toSMILES(), mol.toAdjacencyList()))
        else:
            logger.warning('Number of radicals ({0}) and multiplicity ({1}) for {2} do not match. It might be OK since'
                           ' this species is charged and charged molecules are currently not perceived well in ARC.'
                           '\n{3}'.format(radicals, mol.multiplicity, mol.toSMILES(), mol.toAdjacencyList()))


def add_rads_by_atom_valance(mol):
    """
    A helper function for assigning radicals if not identified automatically,
    and they missing according to the given multiplicity.
    We assume here that all partial charges were already set, but this assumption could be wrong.
    Note: This implementation might also be problematic for aromatic species with undefined bond orders.

    Args:
        mol (Molecule): The Molecule object to process.
    """
    for atom in mol.atoms:
        if atom.isNonHydrogen():
            atomic_orbitals = atom.lonePairs + atom.radicalElectrons + atom.getBondOrdersForAtom()
            missing_electrons = 4 - atomic_orbitals
            if missing_electrons:
                atom.radicalElectrons = missing_electrons


def set_radicals_by_map(mol, radical_map):
    """
    Set radicals in ``mol`` by ``radical_map``.

    Args:
        mol (Molecule): THe RMG Molecule object to process.
        radical_map (Molecule): An RMG Molecule object with the same atom order to be used as a radical map.

    Raises:
        ValueError: If atom order does not match.
    """
    for i, atom in enumerate(mol.atoms):
        if atom.element.number != radical_map.atoms[i].element.number:
            raise ValueError('Atom order in mol and radical_map in set_radicals_by_map() do not match. '
                             '{0} is not {1}.'.format(atom.element.symbol, radical_map.atoms[i].symbol))
        atom.radicalElectrons = radical_map.atoms[i].radicalElectrons


def order_atoms_in_mol_list(ref_mol, mol_list):
    """
    Order the atoms in all molecules of ``mol_list`` by the atom order in ``ref_mol``.

    Args:
        ref_mol (Molecule): The reference Molecule object.
        mol_list (list): Entries are Molecule objects whos atoms will be reordered according to the reference.

    Raises:
        SanitizationError: If atoms could not be re-ordered.
    """
    if mol_list is not None:
        for mol in mol_list:
            try:  # TODO: flag as unordered (or solve)
                order_atoms(ref_mol, mol)
            except SanitizationError as e:
                logger.warning('Could not order atoms in\n{0}\nGot the following error:'
                               '\n{1}'.format(mol.toAdjacencyList, e))


def order_atoms(ref_mol, mol):
    """
    Order the atoms in ``mol`` by the atom order in ``ref_mol``.

    Args:
        ref_mol (Molecule): The reference Molecule object.
        mol (Molecule): The Molecule object to process.

    Raises:
        SanitizationError: If atoms could not be re-ordered.
    """
    if ref_mol is not None and mol is not None:
        ref_mol_is_iso_copy = ref_mol.copy(deep=True)
        mol_is_iso_copy = mol.copy(deep=True)
        ref_mol_find_iso_copy = ref_mol.copy(deep=True)
        mol_find_iso_copy = mol.copy(deep=True)

        ref_mol_is_iso_copy = update_molecule(ref_mol_is_iso_copy, to_single_bonds=True)
        mol_is_iso_copy = update_molecule(mol_is_iso_copy, to_single_bonds=True)
        ref_mol_find_iso_copy = update_molecule(ref_mol_find_iso_copy, to_single_bonds=True)
        mol_find_iso_copy = update_molecule(mol_find_iso_copy, to_single_bonds=True)

        if mol_is_iso_copy.isIsomorphic(ref_mol_is_iso_copy, saveOrder=True):
            mapping = mol_find_iso_copy.findIsomorphism(ref_mol_find_iso_copy, saveOrder=True)
            if len(mapping):
                if isinstance(mapping, list):
                    mapping = mapping[0]
                index_map = {ref_mol_find_iso_copy.atoms.index(val): mol_find_iso_copy.atoms.index(key)
                             for key, val in mapping.items()}
                mol.atoms = [mol.atoms[index_map[i]] for i, _ in enumerate(mol.atoms)]
            else:
                raise SanitizationError('Could not map molecules {0}, {1}:\n\n{2}\n\n{3}'.format(
                    ref_mol.toSMILES(), mol.toSMILES(), ref_mol.toAdjacencyList(), mol.toAdjacencyList()))
        else:
            raise SanitizationError('Could not map non isomorphic molecules {0}, {1}:\n\n{2}\n\n{3}'.format(
                ref_mol.toSMILES(), mol.toSMILES(), ref_mol.toAdjacencyList(), mol.toAdjacencyList()))


def update_molecule(mol, to_single_bonds=False):
    """
    Updates the molecule, useful for isomorphism comparison.

    Args:
        mol (Molecule): THe RMG Molecule object to process.
        to_single_bonds (bool, optional): Whether to convert all bonds to single bonds. True to convert.

    Returns:
        new_mol (Molecule): The updated molecule..
    """
    new_mol = Molecule()
    try:
        atoms = mol.atoms
    except AttributeError:
        return None
    atom_mapping = dict()
    for atom1 in atoms:
        new_atom = new_mol.addAtom(Atom(atom1.element))
        atom_mapping[atom1] = new_atom
    for atom1 in atoms:
        for atom2 in atom1.bonds.keys():
            bond_order = 1.0 if to_single_bonds else atom1.bonds[atom2].getOrderNum()
            bond = Bond(atom_mapping[atom1], atom_mapping[atom2], bond_order)
            new_mol.addBond(bond)
    try:
        new_mol.updateAtomTypes()
    except (AtomTypeError, KeyError):
        pass
    new_mol.multiplicity = mol.multiplicity
    return new_mol


def s_bonds_mol_from_xyz(xyz):
    """
    Create a single bonded molecule from xyz using RMG's connectTheDots() method.

    Args:
        xyz (dict): THe xyz coordinates.

    Returns:
        Molecule: THe single bonded molecule.

    Raises:
        InputError: If xyz is in a wrong format.
    """
    mol = Molecule()
    if not isinstance(xyz, dict):
        raise SpeciesError('xyz must be a dictionary, got: {0}'.format(type(xyz)))
    for symbol, coord in zip(xyz['symbols'], xyz['coords']):
        atom = Atom(element=str(symbol))
        atom.coords = np.array([coord[0], coord[1], coord[2]], np.float64)
        mol.addAtom(atom)
    mol.connectTheDots()  # only adds single bonds, but we don't care
    return mol


def to_rdkit_mol(mol, remove_h=False, return_mapping=True, sanitize=True):
    """
    Convert a molecular structure to a RDKit rdmol object. Uses
    `RDKit <http://rdkit.org/>`_ to perform the conversion.
    Perceives aromaticity.
    Adopted from rmgpy/molecule/converter.py

    Args:
        mol (Molecule): An RMG Molecule object for the conversion.
        remove_h (bool, optional): Whether to remove hydrogen atoms from the molecule, True to remove.
        return_mapping (bool, optional): Whether to return the atom mapping, True to return.
        sanitize (bool, optional): Whether to sanitize the RDKit molecule, True to sanitize.

    Returns:
        RDMol: An RDKit molecule object corresponding to the input RMG Molecule object.
    Returns:
        dict: An atom mapping dictionary. Keys are Atom objects of 'mol', values are atom indices in the RDKit Mol.
    """
    mol_copy = mol.copy(deep=True)
    if not mol_copy.atomIDValid():
        mol_copy.assignAtomIDs()
    atom_id_map = dict()
    for i, atom in enumerate(mol_copy.atoms):
        atom_id_map[atom.id] = i
    # Sort the atoms before converting to ensure output is consistent between different runs
    mol_copy.sortAtoms()
    atoms = mol_copy.vertices
    rd_atom_indices = {}  # dictionary of RDKit atom indices
    rdkitmol = Chem.rdchem.EditableMol(Chem.rdchem.Mol())
    for index, atom in enumerate(mol_copy.vertices):
        if atom.element.symbol == 'X':
            rd_atom = Chem.rdchem.Atom('Pt')  # not sure how to do this with linear scaling when this might not be Pt
        else:
            rd_atom = Chem.rdchem.Atom(atom.element.symbol)
        if atom.element.isotope != -1:
            rd_atom.SetIsotope(atom.element.isotope)
        rd_atom.SetNumRadicalElectrons(atom.radicalElectrons)
        rd_atom.SetFormalCharge(atom.charge)
        if atom.element.symbol == 'C' and atom.lonePairs == 1 and mol_copy.multiplicity == 1:
            rd_atom.SetNumRadicalElectrons(2)
        rdkitmol.AddAtom(rd_atom)
        if not (remove_h and atom.symbol == 'H'):
            rd_atom_indices[mol.atoms[atom_id_map[atom.id]]] = index

    rd_bonds = Chem.rdchem.BondType
    orders = {'S': rd_bonds.SINGLE, 'D': rd_bonds.DOUBLE, 'T': rd_bonds.TRIPLE, 'B': rd_bonds.AROMATIC,
              'Q': rd_bonds.QUADRUPLE}
    # Add the bonds
    for atom1 in mol_copy.vertices:
        for atom2, bond in atom1.edges.items():
            if bond.isHydrogenBond():
                continue
            index1 = atoms.index(atom1)
            index2 = atoms.index(atom2)
            if index1 < index2:
                order_string = bond.getOrderStr()
                order = orders[order_string]
                rdkitmol.AddBond(index1, index2, order)

    # Make editable mol and rectify the molecule
    rdkitmol = rdkitmol.GetMol()
    if sanitize:
        Chem.SanitizeMol(rdkitmol)
    if remove_h:
        rdkitmol = Chem.RemoveHs(rdkitmol, sanitize=sanitize)
    if return_mapping:
        return rdkitmol, rd_atom_indices
    return rdkitmol


def rdkit_conf_from_mol(mol, xyz):
    """
     Generate an RDKit Conformer object from an RMG Molecule object.

    Args:
        mol (Molecule): The RMG Molecule object.
        xyz (dict): The xyz coordinates (of the conformer, atoms must be ordered as in ``mol``.

    Returns:
        Conformer: An RDKit Conformer object.
    Returns:
        RDMol: An RDKit Molecule object.
    Returns:
        dict: Atom index map. Keys are atom indices in the RMG Molecule, values are atom indices in the RDKit Molecule.
    """
    if not isinstance(xyz, dict):
        raise InputError('The xyz argument seem to be of wrong type. Expected a dictionary, '
                         'got\n{0}\nwhich is a {1}'.format(xyz, type(xyz)))
    rd_mol, rd_indices = to_rdkit_mol(mol=mol, remove_h=False, return_mapping=True)
    Chem.AllChem.EmbedMolecule(rd_mol)
    index_map = dict()
    for xyz_index, atom in enumerate(mol.atoms):  # generate an atom index mapping dictionary
        index_map[xyz_index] = rd_indices[atom]
    conf = rd_mol.GetConformer(id=0)
    for i in range(rd_mol.GetNumAtoms()):
        conf.SetAtomPosition(index_map[i], xyz['coords'][i])  # reset atom coordinates
    return conf, rd_mol, index_map


def set_rdkit_dihedrals(conf, rd_mol, index_map, rd_scan, deg_increment=None, deg_abs=None):
    """
    A helper function for setting dihedral angles using RDKit.
    Either ``deg_increment`` or ``deg_abs`` must be specified.

    Args:
        conf: The RDKit conformer with the current xyz information.
        rd_mol: The respective RDKit molecule.
        index_map (dict): An atom index mapping dictionary, keys are xyz_index, values are rd_index.
        rd_scan (list): The four-atom torsion scan indices corresponding to the RDKit conformer indices.
        deg_increment (float, optional): The required dihedral increment in degrees.
        deg_abs (float, optional): The required dihedral in degrees.

    Returns:
        new_xyz (dict): The xyz with the new dihedral, ordered according to the map,
    """
    if deg_increment is None and deg_abs is None:
        raise SpeciesError('Cannot set dihedral without either a degree increment or an absolute degree')
    if deg_increment is not None:
        deg0 = rdMT.GetDihedralDeg(conf, rd_scan[0], rd_scan[1], rd_scan[2], rd_scan[3])  # get original dihedral
        deg = deg0 + deg_increment
    else:
        deg = deg_abs
    rdMT.SetDihedralDeg(conf, rd_scan[0], rd_scan[1], rd_scan[2], rd_scan[3], deg)
    coords = list()
    symbols = list()
    for i, atom in enumerate(list(rd_mol.GetAtoms())):
        coords.append([conf.GetAtomPosition(index_map[i]).x, conf.GetAtomPosition(index_map[i]).y,
                       conf.GetAtomPosition(index_map[i]).z])
        symbols.append(atom.GetSymbol())
    new_xyz = xyz_from_data(coords=coords, symbols=symbols)
    return new_xyz


def check_isomorphism(mol1, mol2, filter_structures=True):
    """
    Convert ``mol1`` and ``mol2`` to RMG Species objects, and generate resonance structures.
    Then check Species isomorphism.
    This function first makes copies of the molecules, since isIsomorphic() changes atom orders.

    Args:
        mol1 (Molecule): An RMG Molecule object.
        mol2 (Molecule): An RMG Molecule object.
        filter_structures (bool, optional): Whether to apply the filtration algorithm when generating
                                            resonance structures. ``True`` to apply.

    Returns:
        bool: Whether one of the molecules in the Species derived from ``mol1``
              is isomorphic to one of the molecules in the Species derived from ``mol2``. ``True`` if it is.
    """
    mol1.reactive, mol2.reactive = True, True
    mol1_copy = mol1.copy(deep=True)
    mol2_copy = mol2.copy(deep=True)
    spc1 = Species(molecule=[mol1_copy])
    spc2 = Species(molecule=[mol2_copy])

    try:
        spc1.generate_resonance_structures(keep_isomorphic=False, filter_structures=filter_structures)
    except AtomTypeError:
        pass
    try:
        spc2.generate_resonance_structures(keep_isomorphic=False, filter_structures=filter_structures)
    except AtomTypeError:
        pass

    for molecule1 in spc1.molecule:
        for molecule2 in spc2.molecule:
            if molecule1.isIsomorphic(molecule2, saveOrder=True):
                return True
    return False


def get_center_of_mass(xyz):
    """
    Get the center of mass of xyz coordinates.
    Assumes arc.converter.standardize_xyz_string() was already called for xyz.
    Note that xyz from ESS output is usually already centered at the center of mass (to some precision).
    Either xyz or coords and symbols must be given.

    Args:
        xyz (dict): The xyz coordinates.

    Returns:
        tuple: The center of mass coordinates.
    """
    masses = [get_element_mass(str(symbol), isotope)[0] for symbol, isotope in zip(xyz['symbols'], xyz['isotopes'])]
    cm_x, cm_y, cm_z = 0, 0, 0
    for coord, mass in zip(xyz['coords'], masses):
        cm_x += coord[0] * mass
        cm_y += coord[1] * mass
        cm_z += coord[2] * mass
    cm_x /= sum(masses)
    cm_y /= sum(masses)
    cm_z /= sum(masses)
    return cm_x, cm_y, cm_z


def translate_to_center_of_mass(xyz):
    """
    Translate coordinates to their center of mass.

    Args:
        xyz (dict): The 3D coordinates.

    Returns:
        dict: The translated coordinates.
    """
    cm_x, cm_y, cm_z = get_center_of_mass(xyz)
    x = [coord[0] - cm_x for coord in xyz['coords']]
    y = [coord[1] - cm_y for coord in xyz['coords']]
    z = [coord[2] - cm_z for coord in xyz['coords']]
    translated_coords = tuple((xi, yi, zi) for xi, yi, zi in zip(x, y, z))
    return xyz_from_data(coords=translated_coords, symbols=xyz['symbols'], isotopes=xyz['isotopes'])
