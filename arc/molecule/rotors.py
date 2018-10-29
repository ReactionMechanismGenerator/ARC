#!/usr/bin/env python
# encoding: utf-8

##################################################################


class Rotors(object):
    def __init__(self, mol):
        self.mol = mol
        self.rotors = []
        self.find_internal_rotors()

    def find_internal_rotors(self):
        """
        Locates the sets of indices corresponding to every internal rotor.
        Returns for each rotors the gaussian scan coordinates, the pivots and the top.
        """
        for atom1 in self.mol.vertices:
            for atom2, bond in atom1.edges.items():
                if self.mol.vertices.index(atom1) < self.mol.vertices.index(atom2) \
                        and (bond.isSingle() or bond.isHydrogenBond()) and not self.mol.isBondInCycle(bond):
                    if len(atom1.edges) > 1 and len(
                            atom2.edges) > 1:  # none of the pivotal atoms is terminal (nor hydrogen)
                        rotor = dict()
                        # pivots:
                        rotor['pivots'] = [self.mol.vertices.index(atom1) + 1, self.mol.vertices.index(atom2) + 1]
                        # top:
                        top1, top2 = [], []
                        top1_has_heavy_atoms, top2_has_heavy_atoms = False, False
                        atom_list = [atom1]
                        while len(atom_list):
                            for atom in atom_list:
                                top1.append(self.mol.vertices.index(atom) + 1)
                                for atom3, bond3 in atom.edges.items():
                                    if atom3.isHydrogen():
                                        top1.append(self.mol.vertices.index(atom3) + 1)  # append H's
                                    elif atom3 is not atom2:
                                        top1_has_heavy_atoms = True
                                        if not bond3.isSingle():
                                            top1.append(
                                                self.mol.vertices.index(atom3) + 1)  # append non-single-bonded heavy atoms
                                            atom_list.append(atom3)
                                atom_list.pop(atom_list.index(atom))
                        atom_list = [atom2]
                        while len(atom_list):
                            for atom in atom_list:
                                top2.append(self.mol.vertices.index(atom) + 1)
                                for atom3, bond3 in atom.edges.items():
                                    if atom3.isHydrogen():
                                        top2.append(self.mol.vertices.index(atom3) + 1)  # append H's
                                    elif atom3 is not atom1:
                                        top2_has_heavy_atoms = True
                                        if not bond3.isSingle():
                                            top2.append(
                                                self.mol.vertices.index(atom3) + 1)  # append non-single-bonded heavy atoms
                                            atom_list.append(atom3)
                                atom_list.pop(atom_list.index(atom))
                        if top1_has_heavy_atoms and not top2_has_heavy_atoms:
                            rotor['top'] = top2
                        elif top2_has_heavy_atoms and not top1_has_heavy_atoms:
                            rotor['top'] = top1
                        else:
                            rotor['top'] = top1 if len(top1) < len(top2) else top2
                        # scan:
                        rotor['scan'] = []
                        heavy_atoms = []
                        hydrogens = []
                        for atom3, bond13 in atom1.edges.items():
                            if atom3.isHydrogen():
                                hydrogens.append(self.mol.vertices.index(atom3))
                            elif atom3 is not atom2:
                                heavy_atoms.append(self.mol.vertices.index(atom3))
                        smallest_index = len(self.mol.vertices)
                        if len(heavy_atoms):
                            for i in heavy_atoms:
                                if i < smallest_index:
                                    smallest_index = i
                        else:
                            for i in hydrogens:
                                if i < smallest_index:
                                    smallest_index = i
                        rotor['scan'].append(smallest_index + 1)
                        rotor['scan'].extend([self.mol.vertices.index(atom1) + 1, self.mol.vertices.index(atom2) + 1])
                        heavy_atoms = []
                        hydrogens = []
                        for atom3, bond3 in atom2.edges.items():
                            if atom3.isHydrogen():
                                hydrogens.append(self.mol.vertices.index(atom3))
                            elif atom3 is not atom1:
                                heavy_atoms.append(self.mol.vertices.index(atom3))
                        smallest_index = len(self.mol.vertices)
                        if len(heavy_atoms):
                            for i in heavy_atoms:
                                if i < smallest_index:
                                    smallest_index = i
                        else:
                            for i in hydrogens:
                                if i < smallest_index:
                                    smallest_index = i
                        rotor['scan'].append(smallest_index + 1)
                        self.rotors.append(rotor)

# TODO: determine rotor symmetry (in Cantherm?)
