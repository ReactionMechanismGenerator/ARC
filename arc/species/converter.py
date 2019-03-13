#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import numpy as np
import logging

from rdkit import Chem
import pybel

from rmgpy.species import Species
from rmgpy.molecule.molecule import Atom, Bond, Molecule
from rmgpy.molecule.element import getElement
from rmgpy.exceptions import AtomTypeError
from arkane.common import symbol_by_number

from arc.arc_exceptions import SpeciesError, SanitizationError
from arc.species.xyz_to_2d import MolGraph

##################################################################


def get_xyz_string(xyz, mol=None, number=None, symbol=None):
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
    result = ''
    if symbol is not None:
        elements = symbol
    elif number is not None:
        elements = []
        for num in number:
            elements.append(getElement(int(num)).symbol)
    elif mol is not None:
        elements = []
        for atom in mol.atoms:
            elements.append(atom.element.symbol)
    else:
        raise ValueError("Must have either an RMG:Molecule object input as `mol`, or atomic numbers \ symbols.")
    for i, coord in enumerate(xyz):
        result += elements[i] + ' ' * (4 - len(elements[i]))
        for c in coord:
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

    Returns xyz as well as atom symbols, x, y, and z seperately
    """
    xyz = standardize_xyz_string(xyz)
    x, y, z, symbols = [], [], [], []
    for line in xyz.split('\n'):
        if line:
            atom, xx, yy, zz = line.split()
            x.append(float(xx))
            y.append(float(yy))
            z.append(float(zz))
            symbols.append(atom)
    xyz = []
    for i, _ in enumerate(x):
        xyz.append([x[i], y[i], z[i]])
    return xyz, symbols, x, y, z


def xyz_string_to_xyz_file_format(xyz, comment=''):
    """
    Convert the ARC xyz string format into the XYZ file format: https://en.wikipedia.org/wiki/XYZ_file_format
    """
    xyz = standardize_xyz_string(xyz)
    num = int(len(xyz.split()) / 4)
    return str(num) + '\n' + comment + '\n' + xyz + '\n'


def standardize_xyz_string(xyz):
    """
    A helper function to correct xyz string format input
    Usually empty lines are added by the user either in the beginning or the end, and we'd like to remove them
    """
    xyz = os.linesep.join([s for s in xyz.splitlines() if s and any(c != ' ' for c in s)])
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
    except AtomTypeError:
        return None
    return rmg_mol


def elementize(atom):
    """
    Convert the atomType of an RMG:Atom object into its general parent element atomType (e.g., `S4d` into `S`)
    `atom` is an RMG:Atom object
    Written by Matt Johnson
    """
    atom_type = atom.atomType
    atom_type = [at for at in atom_type.generic if at.label != 'R' and at.label != 'R!H' and not 'Val' in at.label]
    if atom_type:
        atom.atomType = atom_type[0]


def molecules_from_xyz(xyz):
    """
    Creating RMG:Molecule objects from xyz with correct atom labeling
    `xyz` is in a string format
    returns `s_mol` (with only single bonds) and `b_mol` (with best guesses for bond orders)
    This function is based on the MolGraph.perceive_smiles method
    Returns None for b_mol is unsuccessful to infer bond orders
    """
    if xyz is None:
        return None, None
    if not isinstance(xyz, (str, unicode)):
        raise SpeciesError('xyz must be a string format, got: {0}'.format(type(xyz)))
    xyz = standardize_xyz_string(xyz)
    coords, symbols, _, _, _ = get_xyz_matrix(xyz)
    mol_graph = MolGraph(symbols=symbols, coords=coords)
    infered_connections = mol_graph.infer_connections()
    if infered_connections:
        mol_s1 = mol_graph.to_rmg_mol()  # An RMG Molecule with single bonds, atom order corresponds to xyz
    else:
        mol_s1, _ = s_bonds_mol_from_xyz(xyz)
    if mol_s1 is None:
        logging.error('Could not create a 2D graph representation from xyz:\n{0}'.format(xyz))
        return None, None
    mol_s1_updated = update_molecule(mol_s1, to_single_bonds=True)
    pybel_mol = xyz_to_pybel_mol(xyz)
    if pybel_mol is not None:
        inchi = pybel_to_inchi(pybel_mol)
        mol_bo = rmg_mol_from_inchi(inchi)  # An RMG Molecule with bond orders, but without preserved atom order
    else:
        mol_bo = None
    order_atoms(ref_mol=mol_s1_updated, mol=mol_bo)
    s_mol, b_mol = mol_s1_updated, mol_bo
    return s_mol, b_mol


def order_atoms_in_mol_list(ref_mol, mol_list):
    """Order the atoms in all molecules of mol_list by the atom order in ref_mol"""
    for mol in mol_list:
        order_atoms(ref_mol, mol)


def order_atoms(ref_mol, mol):
    """Order the atoms in `mol` by the atom order in ref_mol"""
    if mol is not None:
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
    new_mol.updateAtomTypes()
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


def rdkit_conf_from_mol(mol, coordinates):
    """A helper function generating an RDKit:Conformer object from an RMG:Molecule object"""
    rd_mol, rd_inds = mol.toRDKitMol(removeHs=False, returnMapping=True)
    Chem.AllChem.EmbedMolecule(rd_mol)  # unfortunately, this mandatory embedding changes the coordinates
    indx_map = dict()
    for xyz_index, atom in enumerate(mol.atoms):  # generate an atom index mapping dictionary
        rd_index = rd_inds[atom]
        indx_map[xyz_index] = rd_index
    conf = rd_mol.GetConformer(id=0)
    for i in range(rd_mol.GetNumAtoms()):  # reset atom coordinates
        conf.SetAtomPosition(indx_map[i], coordinates[i])
    return conf, rd_mol, indx_map


def check_isomorphism(mol1, mol2, filter_structures=True):
    """
    Converts `mol1` and `mol2` which are RMG:Molecule objects into RMG:Species object
    and generate resonance structures. Then check Species isomorphism.
    Return True if one of the molecules in the Species derived from `mol1`
    is isomorphic to  one of the molecules in the Species derived from `mol2`.
    `filter_structures` is being passes to Species.generate_resonance_structures().
    make copies of the molecules, since isIsomorphic() changes atom orders
    """
    mol1_copy = mol1.copy(deep=True)
    mol2_copy = mol2.copy(deep=True)
    spc1 = Species(molecule=[mol1_copy])
    spc1.generate_resonance_structures(keep_isomorphic=False, filter_structures=filter_structures)
    spc2 = Species(molecule=[mol2_copy])
    spc2.generate_resonance_structures(keep_isomorphic=False, filter_structures=filter_structures)
    return spc1.isIsomorphic(spc2)
