#!/usr/bin/env python
# encoding: utf-8

import logging
from rdkit.Chem import AllChem
import openbabel as ob
import pybel as pyb

from rmgpy.molecule.converter import toOBMol
from rmgpy.molecule.element import getElement

##################################################################


class ConformerSearch(object):
    """
    ConformerSearch object. Generates conformers for `mol` which is an ``RMG.Molecule`` object using the method/s
    specified in `method`: 'rdkit', 'openbabel', or 'all'. result/s saved to self.xyzs
    """
    def __init__(self, mol, method='all'):
        self.mol = mol
        self.xyzs = list()
        self.method = method
        self.search()

    def search(self):
        rdkit = False
        ob = False
        if self.method == 'all':
            rdkit = True
            ob = True
        elif self.method.lower() == 'rdkit':
            rdkit = True
        elif self.method.lower() in ['ob', 'openbabel']:
            ob = True

        if rdkit:
            rd_xyzs, rd_energies = self.get_possible_conformers_rdkit()
            if rd_xyzs:
                rd_xyz = self.get_min_energy_conformer(xyzs=rd_xyzs, energies=rd_energies)
                self.xyzs.append(get_xyz_matrix(xyz=rd_xyz, conformer_search=self))
        if ob:
            ob_xyzs, ob_energies = self.get_possible_conformers_openbabel()
            ob_xyz = self.get_min_energy_conformer(xyzs=ob_xyzs, energies=ob_energies)
            self.xyzs.append(get_xyz_matrix(xyz=ob_xyz, conformer_search=self))

    def get_possible_conformers_rdkit(self):
        """
        Uses rdkit to automatically generate a set of len(mol.atoms)-3)*30 initial geometries, optimizes
        these geometries using MMFF94s, calculates the energies using MMFF94s
        and converts them back in terms of the RMG atom ordering
        Returns the coordinates and energies
        """
        rdmol, rdInds = self.mol.toRDKitMol(removeHs=False, returnMapping=True)
        rdIndMap = dict()
        for k, atm in enumerate(self.mol.atoms):
            ind = rdInds[atm]
            rdIndMap[ind] = k
        if len(self.mol.atoms) > 5:
            AllChem.EmbedMultipleConfs(rdmol, numConfs=(len(self.mol.atoms) - 3) * 30, randomSeed=1)
        else:
            AllChem.EmbedMultipleConfs(rdmol, numConfs=120, randomSeed=1)
        energies = []
        xyzs = []
        for i in xrange(rdmol.GetNumConformers()):
            v = 1
            while v == 1:
                v = AllChem.MMFFOptimizeMolecule(rdmol, mmffVariant='MMFF94s', confId=i,
                                                 maxIters=500, ignoreInterfragInteractions=False)
            mp = AllChem.MMFFGetMoleculeProperties(rdmol, mmffVariant='MMFF94s')
            if mp is not None:
                ff = AllChem.MMFFGetMoleculeForceField(rdmol, mp, confId=i)
                E = ff.CalcEnergy()
                energies.append(E)
                cf = rdmol.GetConformer(i)
                xyz = []
                for j in xrange(cf.GetNumAtoms()):
                    pt = cf.GetAtomPosition(j)
                    xyz.append([pt.x, pt.y, pt.z])
                xyz = [xyz[rdIndMap[i]] for i in xrange(len(xyz))]  # reorder
                xyzs.append(xyz)
        return xyzs, energies

    def get_possible_conformers_openbabel(self):
        """
        Uses OpenBabel to automatically generate set of len(mol.atoms)*10-3 initial geometries,
        optimizes these geometries using MMFF94s, calculates the energies using MMFF94s
        Returns the coordinates and energies
        """
        energies = []
        xyzs = []
        obmol, obInds = toOBMol(self.mol, returnMapping=True)
        pybmol = pyb.Molecule(obmol)
        pybmol.make3D()
        obmol = pybmol.OBMol

        ff = ob.OBForceField.FindForceField("mmff94s")
        ff.Setup(obmol)
        if len(self.mol.atoms) > 5:
            ff.WeightedRotorSearch(len(self.mol.atoms) * 10 - 3, 2000)
        else:
            ff.WeightedRotorSearch(120, 2000)
        ff.GetConformers(obmol)
        for n in xrange(obmol.NumConformers()):
            xyz = []
            obmol.SetConformer(n)
            ff.Setup(obmol)
            # ff.ConjugateGradientsTakeNSteps(1000)
            energies.append(ff.Energy())
            for atm in pybmol.atoms:
                xyz.append(list(atm.coords))
            xyzs.append(xyz)
        return xyzs, energies

    def get_min_energy_conformer(self, xyzs, energies):
        minval = min(energies)
        minind = energies.index(minval)
        return xyzs[minind]


def get_xyz_matrix(xyz, conformer_search=None, from_arkane=False, number=None):
    if conformer_search is None and not from_arkane:
        logging.error("Must have either a ConformerSearch object as input, or 'from_arkane' set to True.")
        raise ValueError("Must have either a ConformerSearch object as input, or 'from_arkane' set to True.")
    result = ''
    longest_xyz_coord = 1
    for coord in xyz:
        for c in coord:
            if len(str(c)) > longest_xyz_coord:
                longest_xyz_coord = len(str(c))
    for i, coord in enumerate(xyz):
        if from_arkane:
            element_label = getElement(number[i]).symbol
        else:
            element_label = conformer_search.mol.atoms[i].element.symbol
        result += element_label + ' ' * (4 - len(element_label))
        for j, c in enumerate(coord):
            if c > 0:  # add space for positive numbers
                result += ' '
            result += str(c)
            if j < 2:  # add trailing spaces only for x, y (not z)
                result += ' ' * (longest_xyz_coord - len(str(abs(c))) + 2)
        result += '\n'
    return result

# TODO: isomorphism check for final conformer
# TODO: solve chirality issues? How can I get the N4 isomer?
