#!/usr/bin/env python
# encoding: utf-8

"""
A module for various conversions
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdMT
import pybel

from rmgpy.species import Species
from rmgpy.molecule.molecule import Atom, Bond, Molecule
from rmgpy.molecule.element import getElement
from rmgpy.exceptions import AtomTypeError
from arkane.common import symbol_by_number, get_element_mass

from arc.common import get_logger
from arc.arc_exceptions import SpeciesError, SanitizationError, InputError
from arc.species.xyz_to_2d import MolGraph

##################################################################

logger = get_logger()


def get_xyz_string(coords, mol=None, numbers=None, symbols=None):
    """
    Convert list of lists xyz form:
    [[0.6616514836, 0.4027481525, -0.4847382281],
    [-0.6039793084, 0.6637270105, 0.0671637135],
    [-1.4226865648, -0.4973210697, -0.2238712255],
    [-0.4993010635, 0.6531020442, 1.0853092315],
    [-2.2115796924, -0.4529256762, 0.4144516252],
    [-1.8113671395, -0.3268900681, -1.1468957003]]
    into a geometry form read by ESS:
    C    0.6616514836    0.4027481525   -0.4847382281
    N   -0.6039793084    0.6637270105    0.0671637135
    H   -1.4226865648   -0.4973210697   -0.2238712255
    H   -0.4993010635    0.6531020442    1.0853092315
    H   -2.2115796924   -0.4529256762    0.4144516252
    H   -1.8113671395   -0.3268900681   -1.1468957003
    The atom symbols are derived from either an RMG Molecule object (`mol`) or atom numbers ('number`)
    or explicitly given (`symbol`).
    `number` and `symbol` are lists (optional parameters)
    `xyz` is an array of arrays, as shown in the example above.
    This function isn't defined as a method of ARCSpecies since it is also used when parsing opt geometry in Scheduler
    """
    if isinstance(coords, (str, unicode)):
        logger.debug('Cannot convert string format xyz into a string...')
        return coords
    result = ''
    if symbols is not None:
        elements = symbols
    elif numbers is not None:
        elements = []
        for num in numbers:
            elements.append(getElement(int(num)).symbol)
    elif mol is not None:
        elements = []
        for atom in mol.atoms:
            elements.append(atom.element.symbol)
    else:
        raise ValueError("Must have either an RMG:Molecule object input as `mol`, or atomic numbers / symbols.")
    for i, coordinate in enumerate(coords):
        result += elements[i] + ' ' * (4 - len(elements[i]))
        for c in coordinate:
            result += '{0:14.8f}'.format(c)
        result += '\n'
    return result


def get_xyz_matrix(xyz):
    """
    Convert a string xyz form:
    C    0.6616514836    0.4027481525   -0.4847382281
    N   -0.6039793084    0.6637270105    0.0671637135
    H   -1.4226865648   -0.4973210697   -0.2238712255
    H   -0.4993010635    0.6531020442    1.0853092315
    H   -2.2115796924   -0.4529256762    0.4144516252
    H   -1.8113671395   -0.3268900681   -1.1468957003
    into a list of lists xyz form:
    [[0.6616514836, 0.4027481525, -0.4847382281],
    [-0.6039793084, 0.6637270105, 0.0671637135],
    [-1.4226865648, -0.4973210697, -0.2238712255],
    [-0.4993010635, 0.6531020442, 1.0853092315],
    [-2.2115796924, -0.4529256762, 0.4144516252],
    [-1.8113671395, -0.3268900681, -1.1468957003]]

    Args:
        xyz (str, unicode): The xyz coordinates to conver.

    Returns:
        list: Array-style coordinates.
        list: Chemical symbols of the elements in xyz, order preserved.
        list: X axis values.
        list: Y axis values.
        list: Z axis values.
    """
    if not isinstance(xyz, (str, unicode)):
        raise InputError('Can only convert sting or unicode to list, got: {0}'.format(xyz))
    xyz = standardize_xyz_string(xyz)
    x, y, z, symbols = list(), list(), list(), list()
    for line in xyz.split('\n'):
        if line:
            line_split = line.split()
            if len(line_split) != 4:
                raise InputError('Expecting each line in an xyz string to have 4 values, e.g.:\n'
                                 'C    0.1000    0.2000   -0.3000\nbut got {0} values:\n{1}'.format(
                                  len(line_split), line))
            atom, xx, yy, zz = line.split()
            x.append(float(xx))
            y.append(float(yy))
            z.append(float(zz))
            symbols.append(atom)
    coords = list()
    for i, _ in enumerate(x):
        coords.append([x[i], y[i], z[i]])
    return coords, symbols, x, y, z


def xyz_string_to_xyz_file_format(xyz, comment=''):
    """
    Convert the ARC xyz string format into the XYZ file format: https://en.wikipedia.org/wiki/XYZ_file_format
    """
    if xyz is not None:
        xyz = standardize_xyz_string(xyz)
        num = int(len(xyz.split()) / 4)
        return str(num) + '\n' + comment + '\n' + xyz + '\n'
    else:
        return None


def standardize_xyz_string(xyz):
    """
    A helper function to correct xyz string format input.
    Usually empty lines are added by the user either in the beginning or the end,
    here we remove them along with other common issues.
    """
    xyz = os.linesep.join([s.lstrip() for s in xyz.splitlines() if s and any(c != ' ' for c in s)])
    lines = xyz.splitlines()
    if all([len(line.split()) == 6 for line in lines if len(line)]):
        # Convert Gaussian output format, e.g., "      1          8           0        3.132319    0.769111   -0.080869"
        new_lines = list()
        for line in lines:
            if line:
                split = line.split()
                new_lines.append(' '.join([symbol_by_number[int(split[1])], split[3], split[4], split[5]]))
        lines = new_lines
    return os.linesep.join(line for line in lines if (line and any([char != ' ' for char in line])))


def xyz_to_pybel_mol(xyz):
    """
    Convert xyz in string format into an Open Babel molecule object
    """
    if not isinstance(xyz, (str, unicode)):
        raise SpeciesError('xyz must be a string format, got: {0}'.format(type(xyz)))
    try:
        pybel_mol = pybel.readstring('xyz', xyz_string_to_xyz_file_format(xyz))
    except IOError:
        return None
    return pybel_mol


def pybel_to_inchi(pybel_mol):
    """
    Convert an Open Babel molecule object to InChI
    """
    inchi = pybel_mol.write('inchi', opt={'F': None}).strip()  # Add fixed H layer
    return inchi


def rmg_mol_from_inchi(inchi):
    """
    Generate an RMG Molecule object from InChI
    """
    try:
        rmg_mol = Molecule().fromInChI(str(inchi))
    except (AtomTypeError, ValueError) as e:
        logger.warning('Got the following Error when trying to create an RMG Molecule object from InChI:'
                       '\n{0}'.format(e.message))
        return None
    return rmg_mol


def elementize(atom):
    """
    Convert the atomType of an RMG:Atom object into its general parent element atomType (e.g., `S4d` into `S`)
    `atom` is an RMG:Atom object
    Written by Matt Johnson
    """
    atom_type = atom.atomType
    atom_type = [at for at in atom_type.generic if at.label != 'R' and at.label != 'R!H' and 'Val' not in at.label]
    if atom_type:
        atom.atomType = atom_type[0]


def molecules_from_xyz(xyz, multiplicity=None, charge=0):
    """
    Creating RMG:Molecule objects from xyz with correct atom labeling
    `xyz` is in a string format
    returns `s_mol` (with only single bonds) and `b_mol` (with best guesses for bond orders)
    This function is based on the MolGraph.perceive_smiles method
    Returns None for b_mol is unsuccessful to infer bond orders
    If `multiplicity` is given, the returned species multiplicity will be set to it.
    """
    if xyz is None:
        return None, None
    if not isinstance(xyz, (str, unicode)):
        raise SpeciesError('xyz must be a string format, got: {0}'.format(type(xyz)))
    xyz = standardize_xyz_string(xyz)
    coords, symbols, _, _, _ = get_xyz_matrix(xyz)
    mol_graph = MolGraph(symbols=symbols, coords=coords)
    inferred_connections = mol_graph.infer_connections()
    if inferred_connections:
        mol_s1 = mol_graph.to_rmg_mol()  # An RMG Molecule with single bonds, atom order corresponds to xyz
    else:
        mol_s1, _ = s_bonds_mol_from_xyz(xyz)
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
            order_atoms(ref_mol=mol_s1_updated, mol=mol_bo)
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
    Set the multiplicity of `mol` to `multiplicity` and change radicals as needed
    if a `radical_map`, which is an RMG Molecule object with the same atom order, is given,
    it'll be used to set radicals (useful if bond orders aren't known for a molecule)
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
    A helper function for assigning radicals if not identified automatically
    and missing according to the given multiplicity
    We assume here that all partial charges are already set, but this assumption could be wrong
    This implementation might also be problematic for aromatic species with undefined bond orders
    """
    for atom in mol.atoms:
        if atom.isNonHydrogen():
            atomic_orbitals = atom.lonePairs + atom.radicalElectrons + atom.getBondOrdersForAtom()
            missing_electrons = 4 - atomic_orbitals
            if missing_electrons:
                atom.radicalElectrons = missing_electrons


def set_radicals_by_map(mol, radical_map):
    """Set radicals in `mol` by `radical_map`, bot are RMG Molecule objects with the same atom order"""
    for i, atom in enumerate(mol.atoms):
        if atom.element.number != radical_map.atoms[i].element.number:
            raise ValueError('Atom order in mol and radical_map in set_radicals_by_map() do not match. '
                             '{0} is not {1}.'.format(atom.element.symbol, radical_map.atoms[i].symbol))
        atom.radicalElectrons = radical_map.atoms[i].radicalElectrons


def order_atoms_in_mol_list(ref_mol, mol_list):
    """Order the atoms in all molecules of mol_list by the atom order in ref_mol"""
    if mol_list is not None:
        for mol in mol_list:
            try:  # TODO: flag as unordered (or solve)
                order_atoms(ref_mol, mol)
            except SanitizationError as e:
                logger.warning('Could not order atoms in\n{0}\nGot the following error:'
                               '\n{1}'.format(mol.toAdjacencyList, e))


def order_atoms(ref_mol, mol):
    """Order the atoms in `mol` by the atom order in ref_mol"""
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
    Returns a copy of the current molecule with updated atomTypes
    if to_single_bonds is True, the returned mol contains only single bonds.
    This is useful for isomorphism comparison
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
    except AtomTypeError:
        pass
    new_mol.multiplicity = mol.multiplicity
    return new_mol


def s_bonds_mol_from_xyz(xyz):
    """Create a single bonded molecule from xyz using RMG's connectTheDots()"""
    mol = Molecule()
    coordinates = list()
    if not isinstance(xyz, (str, unicode)):
        raise SpeciesError('xyz must be a string format, got: {0}'.format(type(xyz)))
    for line in xyz.split('\n'):
        if line:
            atom = Atom(element=str(line.split()[0]))
            coordinates.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
            atom.coords = np.array(coordinates[-1], np.float64)
            mol.addAtom(atom)
    mol.connectTheDots()  # only adds single bonds, but we don't care
    return mol, coordinates


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


def rdkit_conf_from_mol(mol, coordinates):
    """
     Generate an RDKit Conformer object from an RMG Molecule object

    Args:
        mol (Molecule): The RMG Molecule object.
        coordinates (list, str, unicode): The coordinates (in any format) of the conformer,
                                          atoms must be ordered as in the molecule.

    Returns:
        Conformer: An RDKit Conformer object.
        RDMol: An RDKit Molecule object.
        dict: Atom index map. Keys are atom indices in the RMG Molecule, values are atom indices in the RDKit Molecule.
    """
    if coordinates is None or not coordinates:
        raise InputError('Cannot process empty coordinates, got: {0}'.format(coordinates))
    if isinstance(coordinates[0], (str, unicode)) and isinstance(coordinates, list):
        raise InputError('The coordinates argument seem to be of wrong type. Got a list of strings:\n{0}'.format(
            coordinates))
    if isinstance(coordinates, (str, unicode)):
        coordinates = get_xyz_matrix(xyz=coordinates)[0]
    rd_mol, rd_indices = to_rdkit_mol(mol=mol, remove_h=False, return_mapping=True)
    Chem.AllChem.EmbedMolecule(rd_mol)
    index_map = dict()
    for xyz_index, atom in enumerate(mol.atoms):  # generate an atom index mapping dictionary
        index_map[xyz_index] = rd_indices[atom]
    conf = rd_mol.GetConformer(id=0)
    for i in range(rd_mol.GetNumAtoms()):
        conf.SetAtomPosition(index_map[i], coordinates[i])  # reset atom coordinates
    return conf, rd_mol, index_map


def set_rdkit_dihedrals(conf, rd_mol, index_map, rd_scan, deg_increment=None, deg_abs=None):
    """
    A helper function for setting dihedral angles
    `conf` is the RDKit conformer with the current xyz information
    `rd_mol` is the RDKit molecule
    `indx_map` is an atom index mapping dictionary, keys are xyz_index, values are rd_index
    `rd_scan` is the torsion scan atom indices corresponding to the RDKit conformer indices
    Either `deg_increment` or `deg_abs` must be specified for the dihedral increment
    Returns xyz in an array format ordered according to the map,
    the elements in the xyz should be identified by the calling function from the context
    """
    if deg_increment is None and deg_abs is None:
        raise SpeciesError('Cannot set dihedral without either a degree increment or an absolute degree')
    if deg_increment is not None:
        deg0 = rdMT.GetDihedralDeg(conf, rd_scan[0], rd_scan[1], rd_scan[2], rd_scan[3])  # get original dihedral
        deg = deg0 + deg_increment
    else:
        deg = deg_abs
    rdMT.SetDihedralDeg(conf, rd_scan[0], rd_scan[1], rd_scan[2], rd_scan[3], deg)
    new_xyz = list()
    for i in range(rd_mol.GetNumAtoms()):
        new_xyz.append([conf.GetAtomPosition(index_map[i]).x, conf.GetAtomPosition(index_map[i]).y,
                        conf.GetAtomPosition(index_map[i]).z])
    return new_xyz


def check_isomorphism(mol1, mol2, filter_structures=True):
    """
    Converts `mol1` and `mol2` which are RMG:Molecule objects into RMG:Species object
    and generate resonance structures. Then check Species isomorphism.
    Return True if one of the molecules in the Species derived from `mol1`
    is isomorphic to  one of the molecules in the Species derived from `mol2`.
    `filter_structures` is being passes to Species.generate_resonance_structures().
    make copies of the molecules, since isIsomorphic() changes atom orders
    """
    mol1.reactive, mol2.reactive = True, True
    mol1_copy = mol1.copy(deep=True)
    mol2_copy = mol2.copy(deep=True)
    spc1 = Species(molecule=[mol1_copy])
    spc1.generate_resonance_structures(keep_isomorphic=False, filter_structures=filter_structures)
    spc2 = Species(molecule=[mol2_copy])
    spc2.generate_resonance_structures(keep_isomorphic=False, filter_structures=filter_structures)
    return spc1.isIsomorphic(spc2)


def get_center_of_mass(xyz=None, coords=None, symbols=None):
    """
    Get the center of mass of xyz coordinates.
    Assumes arc.converter.standardize_xyz_string() was already called for xyz.
    Note that xyz from ESS output is usually already centered at the center of mass (to some precision).
    Either xyz or coords and symbols must be given.

    Args:
        xyz (string, unicode, optional): The xyz coordinates in a string format.
        coords (list, optional): The xyz coordinates in an array format.
        symbols (list, optional): The chemical element symbols corresponding to `coords`.

    Returns:
        tuple: The center of mass coordinates.
    """
    if xyz is not None:
        masses, coords = list(), list()
        for line in xyz.splitlines():
            if line.strip():
                splits = line.split()
                masses.append(get_element_mass(str(splits[0]))[0])
                coords.append([float(splits[1]), float(splits[2]), float(splits[3])])
    elif coords is not None and symbols is not None:
        masses = [get_element_mass(str(symbol))[0] for symbol in symbols]
    else:
        raise InputError('Either xyz or coords and symbols must be given')
    cm_x, cm_y, cm_z = 0, 0, 0
    for coord, mass in zip(coords, masses):
        cm_x += coord[0] * mass
        cm_y += coord[1] * mass
        cm_z += coord[2] * mass
    cm_x /= sum(masses)
    cm_y /= sum(masses)
    cm_z /= sum(masses)
    return cm_x, cm_y, cm_z


def translate_to_center_of_mass(xyz=None, coords=None, symbols=None):
    """
    Translate coordinates to their center of mass.
    Must give either xyz or coords along with symbols.

    Args:
        xyz (str, unicode, optional): A molecule's coordinates in string-format.
        coords (list): A molecule's coordinates in array-format.
        symbols (list): The matching elemental symbols for the coordinates.

    Returns:
        list or str or unicode: The translated coordinates in the input format.
    """
    if xyz is not None:
        coords, symbols, _, _, _ = get_xyz_matrix(xyz)
    if coords is None or symbols is None:
        raise InputError('Could not translate coordinates to center of mass. Got coords = {0} and '
                         'symbols = {1}.'.format(coords, symbols))
    cm_x, cm_y, cm_z = get_center_of_mass(coords=coords, symbols=symbols)
    x = [coord[0] - cm_x for coord in coords]
    y = [coord[1] - cm_y for coord in coords]
    z = [coord[2] - cm_z for coord in coords]
    translated_coords = [[xi, yi, zi] for xi, yi, zi in zip(x, y, z)]
    if xyz is not None:
        return get_xyz_string(coords=translated_coords, symbols=symbols)
    else:
        return translated_coords
