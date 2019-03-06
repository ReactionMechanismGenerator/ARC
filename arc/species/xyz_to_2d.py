#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Written by Colin Grambow

from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np

import pybel
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

from arc.arc_exceptions import SanitizationError

_rdkit_periodic_table = GetPeriodicTable()


class Atom(object):
    """
    Represents an atopm in a molecular graph.
    """

    def __init__(self, symbol=None, idx=None, coords=np.array([]), frozen=False):
        self.symbol = symbol
        self.idx = idx
        self.coords = coords
        self.frozen = frozen
        self.connections = {}

    def __str__(self):
        return '{}: {}'.format(self.idx, self.symbol)

    def __repr__(self):
        return '<Atom "{}">'.format(str(self))

    def copy(self):
        return Atom(
            symbol=self.symbol,
            idx=self.idx,
            coords=self.coords.copy(),
            frozen=self.frozen,
        )

    def get_atomicnum(self):
        return _rdkit_periodic_table.GetAtomicNumber(self.symbol)

    def get_cov_rad(self):
        return _rdkit_periodic_table.GetRcovalent(self.symbol)


class Connection(object):
    """
    Represents a connection in a molecular graph.

    Note: Equality and hash are only based on atom symbols and indices.
    """

    def __init__(self, atom1, atom2):
        self._atom1 = atom1
        self._atom2 = atom2
        self._make_order_invariant()

    def __str__(self):
        return '({})--({})'.format(str(self.atom1), str(self.atom2))

    def __repr__(self):
        return '<Connection "{}">'.format(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(str(self))

    def _make_order_invariant(self):
        # Ensure that atom ordering is consistent
        atoms = [self._atom1, self._atom2]
        atoms.sort(key=lambda a: a.symbol)
        if self._atom1.idx is not None or self._atom2.idx is not None:
            atoms.sort(key=lambda a: a.idx)
        self._atom1, self._atom2 = atoms

    @property
    def atom1(self):
        return self._atom1

    @property
    def atom2(self):
        return self._atom2

    @atom1.setter
    def atom1(self, val):
        self._atom1 = val
        self._make_order_invariant()

    @atom2.setter
    def atom2(self, val):
        self._atom2 = val
        self._make_order_invariant()

    def copy(self):
        return Connection(self.atom1, self.atom2)


class MolGraph(object):
    """
    Class to convert coordinates to a molecular graph w/o bond order information
    and to generate driving coordinates.

    Note: Atom indices start at 1.
    """

    def __init__(self, atoms=None, symbols=None, coords=None, energy=None):
        self.atoms = atoms or []
        self.energy = energy

        if not self.atoms and symbols is not None:
            for idx, symbol in enumerate(symbols):
                atom = Atom(symbol=symbol, idx=idx+1)
                self.add_atom(atom)

        if coords is not None:
            self.set_coords(coords)

    def __iter__(self):
        for atom in self.atoms:
            yield atom

    def get_formula(self):
        """
        Return the molecular formula corresponding to the graph.
        """
        # Count the numbers of each element
        elements = {}
        for atom in self:
            symbol = atom.symbol
            elements[symbol] = elements.get(symbol, 0) + 1

        # Carbon and hydrogen come first if carbon is present, other
        # atoms come in alphabetical order (also hydrogen if there is no
        # carbon)
        formula = ''
        if 'C' in elements.keys():
            count = elements['C']
            formula += 'C{:d}'.format(count) if count > 1 else 'C'
            del elements['C']
            if 'H' in elements.keys():
                count = elements['H']
                formula += 'H{:d}'.format(count) if count > 1 else 'H'
                del elements['H']
        keys = elements.keys()
        keys.sort()
        for key in keys:
            count = elements[key]
            formula += '{}{:d}'.format(key, count) if count > 1 else key

        return formula

    def to_rmg_mol(self):
        import rmgpy.molecule.molecule as rmg_molecule

        rmg_atoms = [rmg_molecule.Atom(element=str(atom.symbol), coords=atom.coords) for atom in self]
        mapping = {atom: rmg_atom for atom, rmg_atom in zip(self.atoms, rmg_atoms)}
        rmg_bonds = [rmg_molecule.Bond(mapping[connection.atom1], mapping[connection.atom2])
                     for connection in self.get_all_connections()]
        rmg_mol = rmg_molecule.Molecule(atoms=rmg_atoms)
        for bond in rmg_bonds:
            rmg_mol.addBond(bond)

        return rmg_mol

    def to_rdkit_mol(self):
        """
        Convert the graph to an RDKit molecule with atom map numbers set
        by the indices of the atoms.
        """
        assert all(atom.idx is not None for atom in self)

        rd_mol = Chem.rdchem.EditableMol(Chem.rdchem.Mol())
        for atom in self:
            rd_atom = Chem.rdchem.Atom(atom.symbol)
            rd_atom.SetAtomMapNum(atom.idx)
            rd_mol.AddAtom(rd_atom)

        for atom1 in self:
            for atom2 in atom1.connections.keys():
                idx1 = self.atoms.index(atom1)  # This is the index in the atoms list
                idx2 = self.atoms.index(atom2)
                if idx1 < idx2:
                    rd_mol.AddBond(idx1, idx2, Chem.rdchem.BondType.SINGLE)

        rd_mol = rd_mol.GetMol()
        return rd_mol

    def to_pybel_mol(self, from_coords=True):
        """
        Convert the graph to a Pybel molecule. Currently only supports
        creating the molecule from 3D coordinates.
        """
        if from_coords:
            xyz = self.to_xyz()
            try:
                mol = pybel.readstring('xyz', xyz)
            except IOError:
                return None
            return mol
        else:
            raise NotImplementedError('Can only create Pybel molecules from 3D structure')

    def to_xyz(self, comment=''):
        """
        Convert the graph to an XYZ-format string. Optionally, add
        comment on the second line.
        """
        for atom in self:
            assert len(atom.coords) != 0
        symbols, coords = self.get_geometry()
        cblock = ['{0}  {1[0]: .10f}  {1[1]: .10f}  {1[2]: .10f}'.format(s, c) for s, c in zip(symbols, coords)]
        return str(len(symbols)) + '\n' + comment + '\n' + '\n'.join(cblock)

    def perceive_smiles(self, atommap=True):
        """
        Using the geometry, perceive the corresponding SMILES with bond
        orders using Open Babel and RDKit. In order to create a sensible
        SMILES, first infer the connectivity from the 3D coordinates
        using Open Babel, then convert to InChI to saturate unphysical
        multi-radical structures, then convert to RDKit and match the
        atoms to the ones in self in order to return a SMILES with atom
        mapping corresponding to the order given by the values of
        atom.idx for all atoms in self.

        This method requires Open Babel version >=2.4.1
        """

        # Get dict of atomic numbers for later comparison.
        atoms_in_mol_true = {}
        for atom in self:
            anum = atom.get_atomicnum()
            atoms_in_mol_true[anum] = atoms_in_mol_true.get(anum, 0) + 1

        # There seems to be no particularly simple way in RDKit to read
        # in 3D structures, so use Open Babel for this part. RMG doesn't
        # recognize some single bonds, so we can't use that.
        # We've probably called to_pybel_mol at some previous time to set
        # connections, but it shouldn't be too expensive to do it again.
        pybel_mol = self.to_pybel_mol()

        # Open Babel will often make single bonds and generate Smiles
        # that have multiple radicals, which would probably correspond
        # to double bonds. To get around this, convert to InChI (which
        # does not consider bond orders) and then convert to Smiles.
        inchi = pybel_mol.write('inchi', opt={'F': None}).strip()  # Add fixed H layer

        # Use RDKit to convert back to Smiles
        mol_sanitized = Chem.MolFromInchi(inchi)

        # RDKit doesn't like some hypervalent atoms
        if mol_sanitized is None:
            raise SanitizationError(
                'Could not convert \n{}\nto Smiles. Unsanitized Smiles: {}'.format(self.to_xyz(),
                                                                                   pybel_mol.write('smi').strip()))

        # RDKit adds unnecessary hydrogens in some cases. If
        # this happens, give up and return an error.
        mol_sanitized = Chem.AddHs(mol_sanitized)
        atoms_in_mol_sani = {}
        for atom in mol_sanitized.GetAtoms():
            atoms_in_mol_sani[atom.GetAtomicNum()] = atoms_in_mol_sani.get(atom.GetAtomicNum(), 0) + 1
        if atoms_in_mol_sani != atoms_in_mol_true:
            raise SanitizationError(
                'Could not convert \n{}\nto Smiles. Wrong Smiles: {}'.format(self.to_xyz(),
                                                                             Chem.MolToSmiles(mol_sanitized)))

        if not atommap:
            return Chem.MolToSmiles(mol_sanitized)

        # Because we went through InChI, we lost atom mapping
        # information. Restore it by matching the original molecule.
        # There should only be one unique map.
        mol_with_map = self.to_rdkit_mol()  # This only has single bonds
        mol_sani_sb = Chem.Mol(mol_sanitized)  # Make copy with single bonds only
        for bond in mol_sani_sb.GetBonds():
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        match = mol_sani_sb.GetSubstructMatch(mol_with_map)  # Isomorphism mapping
        assert mol_with_map.GetNumAtoms() == len(match)  # Make sure we match all atoms
        for atom in mol_with_map.GetAtoms():
            idx = match[atom.GetIdx()]
            map_num = atom.GetAtomMapNum()
            mol_sanitized.GetAtomWithIdx(idx).SetAtomMapNum(map_num)

        # If everything succeeded up to here, we hopefully have a
        # sensible Smiles string with atom mappings for all atoms.
        return Chem.MolToSmiles(mol_sanitized)

    def add_atom(self, atom):
        self.atoms.append(atom)
        atom.connections = {}
        return atom

    def add_connection(self, connection=None, atom1=None, atom2=None):
        """
        Either add a connection directly or first create one from two
        atoms and then add it.
        """
        if connection is None:
            connection = Connection(atom1, atom2)
        if connection.atom1 not in self.atoms or connection.atom2 not in self.atoms:
            raise Exception('Cannot add connection between atoms not in the graph')
        else:
            connection.atom1.connections[connection.atom2] = connection
            connection.atom2.connections[connection.atom1] = connection
            return connection

    def get_all_connections(self):
        return {connection for atom in self.atoms for connection in atom.connections.values()}

    def get_connection(self, atom1, atom2):
        if atom1 not in self.atoms or atom2 not in self.atoms:
            raise Exception('One or both of the specified atoms are not in this graph')

        try:
            return atom1.connections[atom2]
        except KeyError:
            raise Exception('The specified atoms are not connected in this graph')

    def remove_atom(self, atom):
        for atom2 in atom.connections:
            del atom2.connections[atom]
        atom.connections = {}
        self.atoms.remove(atom)

    def remove_connection(self, connection):
        if connection.atom1 not in self.atoms or connection.atom2 not in self.atoms:
            raise Exception('Cannot remove connection between atoms not in the graph')
        del connection.atom1.connections[connection.atom2]
        del connection.atom2.connections[connection.atom1]

    def copy(self, deep=False):
        other = MolGraph(energy=self.energy)
        atoms = self.atoms
        mapping = {}
        for atom in atoms:
            if deep:
                atom2 = other.add_atom(atom.copy())
                mapping[atom] = atom2
            else:
                connections = atom.connections
                other.add_atom(atom)
                atom.connections = connections
        if deep:
            for atom1 in atoms:
                for atom2 in atom1.connections:
                    connection = atom1.connections[atom2]
                    connection = connection.copy()
                    connection.atom1 = mapping[atom1]
                    connection.atom2 = mapping[atom2]
                    other.add_connection(connection)
        return other

    def merge(self, other):
        new = MolGraph()
        for atom in self.atoms:
            connections = atom.connections
            new.add_atom(atom)
            atom.connections = connections
        for atom in other.atoms:
            connections = atom.connections
            new.add_atom(atom)
            atom.connections = connections
        new.energy = self.energy + other.energy
        return new

    def split(self):
        new1 = self.copy()
        new2 = MolGraph()

        if len(self.atoms) == 0:
            return [new1]

        atoms_to_move = [self.atoms[-1]]
        idx = 0
        while idx < len(atoms_to_move):
            for atom2 in atoms_to_move[idx].connections:
                if atom2 not in atoms_to_move:
                    atoms_to_move.append(atom2)
            idx += 1

        if len(new1.atoms) == len(atoms_to_move):
            return [new1]

        for atom in atoms_to_move:
            new2.atoms.append(atom)
            new1.atoms.remove(atom)

        new = [new2]
        new.extend(new1.split())
        new.energy = None
        return new

    def sort_atoms(self):
        self.atoms.sort(key=lambda a: a.idx)

    def is_radical(self):
        """
        Determine whether or not the molecule is a radical based on the number
        of valence electrons for each atom. If the total number of valence
        electrons is odd, then it is a radical. This assumes that molecules
        with an even number of electrons are singlets. This method also assumes
        that none of the atoms are charged.
        """
        valence_electrons = {'H': 1, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 7, 'I': 7}
        symbols = [atom.symbol for atom in self]
        total_valence_electrons = sum(valence_electrons[s] for s in symbols)
        return bool(total_valence_electrons % 2)

    def is_isomorphic(self, other):
        """
        Test if self is isomorphic with other, ignoring atom indices.
        Requires RMG to do the isomorphism check.
        """
        self_rmg = self.to_rmg_mol()
        other_rmg = other.to_rmg_mol()
        return self_rmg.isIsomorphic(other_rmg)

    def set_coords(self, coords):
        """
        Set atom coordinates. Assumes coords are in same order as self.atoms.
        """
        try:
            coords = np.reshape(coords, (-1,3))
        except ValueError:
            raise Exception('Coordinates cannot be reshaped into matrix of size Nx3')
        assert len(coords) == len(self.atoms)

        for atom, xyz in zip(self.atoms, coords):
            atom.coords = xyz

    def get_coords(self):
        """
        Get coordinates in the order specified by the atom indices.
        """
        assert all(atom.idx is not None for atom in self)
        atoms = self.atoms[:]
        atoms.sort(key=lambda a: a.idx)
        return np.array([atom.coords for atom in atoms])

    def get_symbols(self):
        """
        Get symbols in the order specified by the atom indices.
        """
        assert all(atom.idx is not None for atom in self)
        atoms = self.atoms[:]
        atoms.sort(key=lambda a: a.idx)
        return [atom.symbol for atom in atoms]

    def get_geometry(self):
        """
        Get symbols and coordinates in the order specified by the atom
        indices.
        """
        assert all(atom.idx is not None for atom in self)
        atoms = self.atoms[:]
        atoms.sort(key=lambda a: a.idx)
        return [atom.symbol for atom in atoms], np.array([atom.coords for atom in atoms])

    def infer_connections(self, use_ob=True):
        """
        Delete connections and set them again based on coordinates.

        Note: By default this uses Open Babel, which is better than a
        simple covalent radii check.
        """
        atoms = self.atoms

        for atom in atoms:
            assert len(atom.coords) != 0

        for atom in atoms:
            for connection in atom.connections:
                self.remove_connection(connection)

        if use_ob:
            pybel_mol = self.to_pybel_mol()  # Should be sorted by atom indices
            if pybel_mol is None:
                return False
            assert all(ap.idx == a.idx for ap, a in zip(pybel_mol, self))  # Check to be sure
            mapping = {ap.idx: a for ap, a in zip(pybel_mol, self)}
            for bond in pybel.ob.OBMolBondIter(pybel_mol.OBMol):
                atom1 = mapping[bond.GetBeginAtomIdx()]
                atom2 = mapping[bond.GetEndAtomIdx()]
                connection = Connection(atom1, atom2)
                self.add_connection(connection)
        else:
            sorted_atoms = sorted(atoms, key=lambda a: a.coords[2])
            for i, atom1 in enumerate(sorted_atoms):
                for atom2 in sorted_atoms[(i+1):]:
                    crit_dist = (atom1.get_cov_rad() + atom2.get_cov_rad() + 0.45)**2
                    z_boundary = (atom1.coords[2] - atom2.coords[2])**2
                    if z_boundary > 16.0:
                        break
                    dist_sq = sum((atom1.coords - atom2.coords)**2)
                    if dist_sq > crit_dist or dist_sq < 0.4:
                        continue
                    else:
                        connection = Connection(atom1, atom2)
                        self.add_connection(connection)
        return True

    def is_atom_in_cycle(self, atom):
        return self._is_chain_in_cycle([atom])

    def _is_chain_in_cycle(self, chain):
        atom1 = chain[-1]
        for atom2 in atom1.connections:
            if atom2 is chain[0] and len(chain) > 2:
                return True
            elif atom2 not in chain:
                chain.append(atom2)
                if self._is_chain_in_cycle(chain):
                    return True
                else:
                    chain.remove(atom2)
        return False

    def label_equivalent_hydrogens(self):
        """
        Mark all equivalent hydrogens as frozen. For now, this assumes that the
        carbons they are attached to have 4 connections, which means this
        method does not yet work for radicals.
        """
        if self.is_radical():
            raise NotImplementedError('Cannot yet label equivalent hydrogens for radicals')
        for atom in self:
            if (atom.symbol.upper() == 'C'
                    and len(atom.connections) == 4
                    and not self.is_atom_in_cycle(atom)):
                first_hydrogen = True
                for atom2 in atom.connections:
                    if atom2.symbol.upper() == 'H':
                        if first_hydrogen:
                            first_hydrogen = False
                        else:
                            atom2.frozen = True
