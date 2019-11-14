#!/usr/bin/env python3
# encoding: utf-8

"""
A module for performing various species-related format conversions.
"""

import numpy as np
import os

import pybel
from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdMT

from arkane.common import symbol_by_number, mass_by_symbol, get_element_mass
from rmgpy.exceptions import AtomTypeError
from rmgpy.molecule.molecule import Atom, Bond, Molecule
from rmgpy.species import Species

from arc.common import get_logger
from arc.exceptions import SpeciesError, SanitizationError, InputError
from arc.species.xyz_to_2d import MolGraph


logger = get_logger()


def str_to_xyz(xyz_str):
    """
    Convert a string xyz format to the ARC dict xyz style.
    Note: The ``xyz_str`` argument could also direct to a file path to parse the data from.
    The xyz string format may have optional Gaussian-style isotope specification, e.g.::

        C(Iso=13)    0.6616514836    0.4027481525   -0.4847382281
        N           -0.6039793084    0.6637270105    0.0671637135
        H           -1.4226865648   -0.4973210697   -0.2238712255
        H           -0.4993010635    0.6531020442    1.0853092315
        H           -2.2115796924   -0.4529256762    0.4144516252
        H           -1.8113671395   -0.3268900681   -1.1468957003

    which will also be parsed into the ARC xyz dictionary format, e.g.::

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
    if not isinstance(xyz_str, str):
        raise InputError('Expected a string input, got {0}'.format(type(xyz_str)))
    if os.path.isfile(xyz_str):
        from arc.parser import parse_xyz_from_file
        return parse_xyz_from_file(xyz_str)
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
    xyz_dict = check_xyz_dict(xyz_dict)
    if xyz_dict is None:
        logger.warning('Got None for xyz_dict')
        return None
    recognized_isotope_formats = ['gaussian']
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
    return '\n'.join(xyz_list)


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
    xyz_dict = check_xyz_dict(xyz_dict)
    x, y, z = tuple(), tuple(), tuple()
    for coord in xyz_dict['coords']:
        x += (coord[0],)
        y += (coord[1],)
        z += (coord[2],)
    return x, y, z


def xyz_to_coords_list(xyz_dict):
    """
    Get the coords part of an xyz dict as a (mutable) list of lists (rather than a tuple of tuples)

    Args:
        xyz_dict (dict): The ARC xyz format.

    Returns:
        list: The coordinates.
    """
    xyz_dict = check_xyz_dict(xyz_dict)
    coords_tuple = xyz_dict['coords']
    coords_list = list()
    for coords_tup in coords_tuple:
        coords_list.append([coords_tup[0], coords_tup[1], coords_tup[2]])
    return coords_list


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
    xyz_dict = check_xyz_dict(xyz_dict)
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
    if not isinstance(xyz_str, str):
        raise TypeError('Expected a string format, got {0}'.format(type(xyz_str)))
    xyz_dict = str_to_xyz(xyz_str)
    return xyz_to_str(xyz_dict=xyz_dict, isotope_format=isotope_format)


def check_xyz_dict(xyz):
    """
    Check that the xyz dictionary entered is valid. If it is a string, convert it.
    If isotopes are not in xyz_dict, common values will be added.

    Args:
        xyz (dict, str): The xyz dictionary.

    Raises:
        TypeError: If xyz_dict is not a dictionary.
        ValueError: If xyz_dict is missing symbols or coords.
    """
    xyz_dict = str_to_xyz(xyz) if isinstance(xyz, str) else xyz
    if not isinstance(xyz_dict, dict):
        raise TypeError(f'Expected a dictionary, got {type(xyz_dict)}')
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
    xyz = check_xyz_dict(xyz)
    try:
        pybel_mol = pybel.readstring('xyz', xyz_to_xyz_file_format(xyz))
    except (IOError, InputError):
        return None
    return pybel_mol


def pybel_to_inchi(pybel_mol, has_h=True):
    """
    Convert an Open Babel molecule object to InChI

    Args:
        pybel_mol (OBmol): An Open Babel molecule.
        has_h (bool): Whether the molecule has hydrogen atoms. ``True`` if it does.

    Returns:
        inchi (str): The respective InChI representation of the molecule.
    """
    if has_h:
        inchi = pybel_mol.write('inchi', opt={'F': None}).strip()  # Add fixed H layer
    else:
        inchi = pybel_mol.write('inchi').strip()
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
        rmg_mol = Molecule().from_inchi(inchi)
    except (AtomTypeError, ValueError, KeyError, TypeError) as e:
        logger.warning('Got the following Error when trying to create an RMG Molecule object from InChI:'
                       '\n{0}'.format(e))
        return None
    return rmg_mol


def elementize(atom):
    """
    Convert the atom type of an RMG:Atom object into its general parent element atom type (e.g., `S4d` into `S`).

    Args:
        atom (Atom): The atom to process.
    """
    atom_type = atom.atomtype
    atom_type = [at for at in atom_type.generic if at.label != 'R' and at.label != 'R!H' and 'Val' not in at.label]
    if atom_type:
        atom.atomtype = atom_type[0]


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
        Molecule: The respective Molecule object with only single bonds.
    Returns:
        Molecule: The respective Molecule object with perceived bond orders.
                  Returns None if unsuccessful to infer bond orders.
    """
    if xyz is None:
        return None, None
    xyz = check_xyz_dict(xyz)
    mol_bo = None

    # 1. Generate a molecule with no bond order information with atoms ordered as in xyz
    mol_graph = MolGraph(symbols=xyz['symbols'], coords=xyz['coords'])
    inferred_connections = mol_graph.infer_connections()
    if inferred_connections:
        mol_s1 = mol_graph.to_rmg_mol()  # An RMG Molecule with single bonds, atom order corresponds to xyz
    else:
        mol_s1 = s_bonds_mol_from_xyz(xyz)  # An RMG Molecule with single bonds, atom order corresponds to xyz
    if mol_s1 is None:
        logger.error(f'Could not create a 2D graph representation from xyz:\n{xyz_to_str(xyz)}')
        return None, None
    if multiplicity is not None:
        mol_s1.multiplicity = multiplicity
    mol_s1_updated = update_molecule(mol_s1, to_single_bonds=True)

    # 2. A. Generate a molecule with bond order information using pybel:
    pybel_mol = xyz_to_pybel_mol(xyz)
    if pybel_mol is not None:
        inchi = pybel_to_inchi(pybel_mol, has_h=bool(len([atom.is_hydrogen() for atom in mol_s1_updated.atoms])))
        mol_bo = rmg_mol_from_inchi(inchi)  # An RMG Molecule with bond orders, but without preserved atom order

    # TODO 2. B. Deduce bond orders from xyz distances (fallback method)
    # else:
    #     mol_bo = deduce_bond_orders_from_distances(xyz)

    if mol_bo is not None:
        if multiplicity is not None:
            try:
                set_multiplicity(mol_bo, multiplicity, charge)
            except SpeciesError as e:
                logger.warning('Cannot infer 2D graph connectivity, failed to set species multiplicity with the '
                               'following error:\n{0}'.format(e))
                return mol_s1_updated, None
        mol_s1_updated.multiplicity = mol_bo.multiplicity
        try:
            order_atoms(ref_mol=mol_s1_updated, mol=mol_bo)
        except SanitizationError:
            logger.warning('Could not order atoms for {0}!'.format(mol_s1_updated.copy(deep=True).to_smiles()))
        try:
            set_multiplicity(mol_s1_updated, mol_bo.multiplicity, charge, radical_map=mol_bo)
        except SpeciesError as e:
            logger.warning('Cannot infer 2D graph connectivity, failed to set species multiplicity with the '
                           'following error:\n{0}'.format(e))
            return mol_s1_updated, mol_bo

    return mol_s1_updated, mol_bo


def set_multiplicity(mol, multiplicity, charge, radical_map=None):
    """
    Set the multiplicity and charge of a molecule.
    If a `radical_map`, which is an RMG Molecule object with the same atom order, is given,
    it'll be used to set radicals (useful if bond orders aren't known for a molecule).

    Args:
        mol (Molecule): The RMG Molecule object.
        multiplicity (int) The spin multiplicity.
        charge (int): The species net charge.
        radical_map (Molecule, optional): An RMG Molecule object with the same atom order to be used as a radical map.
    """
    mol.multiplicity = multiplicity
    if radical_map is not None:
        if not isinstance(radical_map, Molecule):
            raise TypeError('radical_map sent to set_multiplicity() has to be a Molecule object. Got {0}'.format(
                type(radical_map)))
        set_radicals_by_map(mol, radical_map)
    radicals = mol.get_radical_count()
    if mol.multiplicity != radicals + 1:
        # this is not the trivial "multiplicity = number of radicals + 1" case
        # either the number of radicals was not identified correctly from the 3D structure (i.e., should be lone pairs),
        # or their spin isn't determined correctly
        if mol.multiplicity > radicals + 1:
            # there are sites that should have radicals, but weren't identified as such.
            # try adding radicals according to missing valances
            add_rads_by_atom_valance(mol)
            if mol.multiplicity > radicals + 1:
                # still problematic, currently there's no automated solution to this case, raise an error
                raise SpeciesError('A multiplicity of {0} was given, but only {1} radicals were identified. '
                                   'Cannot infer 2D graph representation for this species.\nMore info:{2}\n{3}'.format(
                                    mol.multiplicity, radicals, mol.to_smiles(), mol.to_adjacency_list()))
    add_lone_pairs_by_atom_valance(mol)
    # final check: an even number of radicals results in an odd multiplicity, and vice versa
    if divmod(mol.multiplicity, 2)[1] == divmod(radicals, 2)[1]:
        if not charge:
            raise SpeciesError('Number of radicals ({0}) and multiplicity ({1}) for {2} do not match.\n{3}'.format(
                radicals, mol.multiplicity, mol.to_smiles(), mol.to_adjacency_list()))
        else:
            logger.warning('Number of radicals ({0}) and multiplicity ({1}) for {2} do not match. It might be OK since '
                           'this species is charged and charged molecules are currently not perceived well in ARC.'
                           '\n{3}'.format(radicals, mol.multiplicity, mol.copy(deep=True).to_smiles(),
                                          mol.copy(deep=True).to_adjacency_list()))


def add_rads_by_atom_valance(mol):
    """
    A helper function for assigning radicals if not identified automatically,
    and they are missing according to the given multiplicity.
    We assume here that all partial charges were already set, but this assumption could be wrong.
    Note: This implementation might also be problematic for aromatic species with undefined bond orders.

    Args:
        mol (Molecule): The Molecule object to process.
    """
    for atom in mol.atoms:
        if atom.is_non_hydrogen():
            atomic_orbitals = atom.lone_pairs + atom.radical_electrons + atom.get_total_bond_order()
            missing_electrons = 4 - atomic_orbitals
            if missing_electrons:
                atom.radical_electrons = missing_electrons


def add_lone_pairs_by_atom_valance(mol):
    """
     helper function for assigning lone pairs instead of carbenes/nitrenes if not identified automatically,
    and they are missing according to the given multiplicity.

    Args:
        mol (Molecule): The Molecule object to process.
    """
    radicals = mol.get_radical_count()
    if mol.multiplicity < radicals + 1:
        carbenes, nitrenes = 0, 0
        for atom in mol.atoms:
            if atom.is_carbon() and atom.radical_electrons >= 2:
                carbenes += 1
            elif atom.is_nitrogen() and atom.radical_electrons >= 2:
                nitrenes += 1
        if 2 * (carbenes + nitrenes) + mol.multiplicity == radicals + 1:
            # this issue can be solved by converting carbenes/nitrenes to lone pairs:
            if carbenes:
                for i in range(len(mol.atoms)):
                    atom = mol.atoms[i]
                    if atom.is_carbon() and atom.radical_electrons >= 2:
                        atom.lone_pairs += 1
                        atom.radical_electrons -= 2
            if nitrenes:
                for i in range(len(mol.atoms)):
                    atom = mol.atoms[i]
                    if atom.is_nitrogen() and atom.radical_electrons >= 2:
                        for atom2, bond12 in atom.edges.items():
                            if atom2.is_sulfur() and atom2.lone_pairs >= 2 and bond12.is_single():
                                bond12.set_order_num(3)
                                atom2.lone_pairs -= 1
                                break
                            elif atom2.is_sulfur() and atom2.lone_pairs == 1 and bond12.is_single():
                                bond12.set_order_num(2)
                                atom2.lone_pairs -= 1
                                atom2.charge += 1
                                atom.charge -= 1
                                break
                            elif atom2.is_nitrogen() and atom2.lone_pairs == 1 and bond12.is_single():
                                bond12.set_order_num(2)
                                atom2.lone_pairs -= 1
                                atom.lone_pairs += 1
                                atom2.charge += 1
                                atom.charge -= 1
                                break
                        else:
                            atom.lone_pairs += 1
                        atom.radical_electrons -= 2
    if len(mol.atoms) == 1 and mol.multiplicity == 1 and mol.atoms[0].radical_electrons == 4:
        # This is a singlet atomic C or Si, convert all radicals to lone pairs
        mol.atoms[0].radical_electrons = 0
        mol.atoms[0].lone_pairs = 2


def set_radicals_by_map(mol, radical_map):
    """
    Set radicals in ``mol`` by ``radical_map``.

    Args:
        mol (Molecule): The RMG Molecule object to process.
        radical_map (Molecule): An RMG Molecule object with the same atom order to be used as a radical map.

    Raises:
        ValueError: If atom order does not match.
    """
    for i, atom in enumerate(mol.atoms):
        if atom.element.number != radical_map.atoms[i].element.number:
            raise ValueError('Atom order in mol and radical_map in set_radicals_by_map() do not match. '
                             '{0} is not {1}.'.format(atom.element.symbol, radical_map.atoms[i].symbol))
        atom.radical_electrons = radical_map.atoms[i].radical_electrons


def order_atoms_in_mol_list(ref_mol, mol_list):
    """
    Order the atoms in all molecules of ``mol_list`` by the atom order in ``ref_mol``.

    Args:
        ref_mol (Molecule): The reference Molecule object.
        mol_list (list): Entries are Molecule objects whos atoms will be reordered according to the reference.

    Returns:
        bool: Whether the reordering was successful, ``True`` if it was.

    Raises:
        SanitizationError: If atoms could not be re-ordered.
    """
    if mol_list is not None:
        for mol in mol_list:
            try:  # TODO: flag as unordered (or solve)
                order_atoms(ref_mol, mol)
            except SanitizationError as e:
                logger.warning('Could not order atoms in\n{0}\nGot the following error:'
                               '\n{1}'.format(mol.to_adjacency_list, e))
                return False
    else:
        logger.warning('Could not order atoms')
        return False
    return True


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

        if mol_is_iso_copy.is_isomorphic(ref_mol_is_iso_copy, save_order=True, strict=False):
            mapping = mol_find_iso_copy.find_isomorphism(ref_mol_find_iso_copy, save_order=True)
            if len(mapping):
                if isinstance(mapping, list):
                    mapping = mapping[0]
                index_map = {ref_mol_find_iso_copy.atoms.index(val): mol_find_iso_copy.atoms.index(key)
                             for key, val in mapping.items()}
                mol.atoms = [mol.atoms[index_map[i]] for i, _ in enumerate(mol.atoms)]
            else:
                # logger.debug('Could not map molecules {0}, {1}:\n\n{2}\n\n{3}'.format(
                #     ref_mol.copy(deep=True).to_smiles(), mol.copy(deep=True).to_smiles(),
                #     ref_mol.copy(deep=True).to_adjacency_list(), mol.copy(deep=True).to_adjacency_list()))
                raise SanitizationError('Could not map molecules')
        else:
            # logger.debug('Could not map non isomorphic molecules {0}, {1}:\n\n{2}\n\n{3}'.format(
            #     ref_mol.copy(deep=True).to_smiles(), mol.copy(deep=True).to_smiles(),
            #     ref_mol.copy(deep=True).to_adjacency_list(), mol.copy(deep=True).to_adjacency_list()))
            raise SanitizationError('Could not map non isomorphic molecules')


def update_molecule(mol, to_single_bonds=False):
    """
    Updates the molecule, useful for isomorphism comparison.

    Args:
        mol (Molecule): The RMG Molecule object to process.
        to_single_bonds (bool, optional): Whether to convert all bonds to single bonds. ``True`` to convert.

    Returns:
        new_mol (Molecule): The updated molecule.
    """
    new_mol = Molecule()
    try:
        atoms = mol.atoms
    except AttributeError:
        return None
    atom_mapping = dict()
    for atom in atoms:
        new_atom = new_mol.add_atom(Atom(atom.element))
        atom_mapping[atom] = new_atom
    for atom1 in atoms:
        for atom2 in atom1.bonds.keys():
            bond_order = 1.0 if to_single_bonds else atom1.bonds[atom2].get_order_num()
            bond = Bond(atom_mapping[atom1], atom_mapping[atom2], bond_order)
            new_mol.add_bond(bond)
    try:
        new_mol.update_atomtypes()
    except (AtomTypeError, KeyError):
        pass
    new_mol.multiplicity = mol.multiplicity
    return new_mol


def s_bonds_mol_from_xyz(xyz):
    """
    Create a single bonded molecule from xyz using RMG's connect_the_dots() method.

    Args:
        xyz (dict): The xyz coordinates.

    Returns:
        Molecule: The respective molecule with only single bonds.
    """
    xyz = check_xyz_dict(xyz)
    mol = Molecule()
    for symbol, coord in zip(xyz['symbols'], xyz['coords']):
        atom = Atom(element=symbol)
        atom.coords = np.array([coord[0], coord[1], coord[2]], np.float64)
        mol.add_atom(atom)
    mol.connect_the_dots()  # only adds single bonds, but we don't care
    return mol


def to_rdkit_mol(mol, remove_h=False, sanitize=True):
    """
    Convert a molecular structure to an RDKit RDMol object. Uses
    `RDKit <http://rdkit.org/>`_ to perform the conversion.
    Perceives aromaticity.
    Adopted from rmgpy/molecule/converter.py

    Args:
        mol (Molecule): An RMG Molecule object for the conversion.
        remove_h (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
        sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.

    Returns:
        RDMol: An RDKit molecule object corresponding to the input RMG Molecule object.
    """
    atom_id_map = dict()

    # only manipulate a copy of ``mol``
    mol_copy = mol.copy(deep=True)
    if not mol_copy.atom_ids_valid():
        mol_copy.assign_atom_ids()
    for i, atom in enumerate(mol_copy.atoms):
        atom_id_map[atom.id] = i  # keeps the original atom order before sorting
    # mol_copy.sort_atoms()  # Sort the atoms before converting to ensure output is consistent between different runs
    atoms_copy = mol_copy.vertices

    rd_mol = Chem.rdchem.EditableMol(Chem.rdchem.Mol())
    for rmg_atom in atoms_copy:
        rd_atom = Chem.rdchem.Atom(rmg_atom.element.symbol)
        if rmg_atom.element.isotope != -1:
            rd_atom.SetIsotope(rmg_atom.element.isotope)
        rd_atom.SetNumRadicalElectrons(rmg_atom.radical_electrons)
        rd_atom.SetFormalCharge(rmg_atom.charge)
        if rmg_atom.element.symbol == 'C' and rmg_atom.lone_pairs == 1 and mol_copy.multiplicity == 1:
            # hard coding for carbenes
            rd_atom.SetNumRadicalElectrons(2)
        if not (remove_h and rmg_atom.symbol == 'H'):
            rd_mol.AddAtom(rd_atom)

    rd_bonds = Chem.rdchem.BondType
    orders = {'S': rd_bonds.SINGLE, 'D': rd_bonds.DOUBLE, 'T': rd_bonds.TRIPLE, 'B': rd_bonds.AROMATIC,
              'Q': rd_bonds.QUADRUPLE}
    # Add the bonds
    for atom1 in atoms_copy:
        for atom2, bond12 in atom1.edges.items():
            if bond12.is_hydrogen_bond():
                continue
            if atoms_copy.index(atom1) < atoms_copy.index(atom2):
                rd_mol.AddBond(atom_id_map[atom1.id], atom_id_map[atom2.id], orders[bond12.get_order_str()])

    # Make editable mol and rectify the molecule
    rd_mol = rd_mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(rd_mol)
    if remove_h:
        rd_mol = Chem.RemoveHs(rd_mol, sanitize=sanitize)
    return rd_mol


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
    """
    if not isinstance(xyz, dict):
        raise InputError('The xyz argument seem to be of wrong type. Expected a dictionary, '
                         'got\n{0}\nwhich is a {1}'.format(xyz, type(xyz)))
    rd_mol = to_rdkit_mol(mol=mol, remove_h=False)
    Chem.AllChem.EmbedMolecule(rd_mol)
    conf = None
    if rd_mol.GetNumConformers():
        conf = rd_mol.GetConformer(id=0)
        for i in range(rd_mol.GetNumAtoms()):
            conf.SetAtomPosition(i, xyz['coords'][i])  # reset atom coordinates
    return conf, rd_mol


def set_rdkit_dihedrals(conf, rd_mol, torsion, deg_increment=None, deg_abs=None):
    """
    A helper function for setting dihedral angles using RDKit.
    Either ``deg_increment`` or ``deg_abs`` must be specified.

    Args:
        conf: The RDKit conformer with the current xyz information.
        rd_mol: The respective RDKit molecule.
        torsion (list, tuple): The 0-indexed atom indices of the four atoms defining the torsion.
        deg_increment (float, optional): The required dihedral increment in degrees.
        deg_abs (float, optional): The required dihedral in degrees.

    Returns:
        new_xyz (dict): The xyz with the new dihedral, ordered according to the map,
    """
    if deg_increment is None and deg_abs is None:
        raise SpeciesError('Cannot set dihedral without either a degree increment or an absolute degree')
    if deg_increment is not None:
        deg0 = rdMT.GetDihedralDeg(conf, torsion[0], torsion[1], torsion[2], torsion[3])  # get original dihedral
        deg = deg0 + deg_increment
    else:
        deg = deg_abs
    rdMT.SetDihedralDeg(conf, torsion[0], torsion[1], torsion[2], torsion[3], deg)
    coords = list()
    symbols = list()
    for i, atom in enumerate(list(rd_mol.GetAtoms())):
        coords.append([conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z])
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
            if molecule1.is_isomorphic(molecule2, save_order=True):
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
    masses = [get_element_mass(symbol, isotope)[0] for symbol, isotope in zip(xyz['symbols'], xyz['isotopes'])]
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
    for i in range(len(x)):
        x[i] = x[i] if abs(x[i]) > 1e-10 else 0.0
        y[i] = y[i] if abs(y[i]) > 1e-10 else 0.0
        z[i] = z[i] if abs(z[i]) > 1e-10 else 0.0
    translated_coords = tuple((xi, yi, zi) for xi, yi, zi in zip(x, y, z))
    return xyz_from_data(coords=translated_coords, symbols=xyz['symbols'], isotopes=xyz['isotopes'])
