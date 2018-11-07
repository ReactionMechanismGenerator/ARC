#!/usr/bin/env python
# encoding: utf-8

import logging

from rdkit.Chem import AllChem
import openbabel as ob
import pybel as pyb

from rmgpy.molecule.converter import toOBMol
from rmgpy.molecule.element import getElement
from rmgpy.species import Species

from arc.exceptions import SpeciesError

##################################################################


class ARCSpecies(object):
    """
    ARCSpecies class

    ====================== =================== =========================================================================
    Attribute              Type                Description
    ====================== =================== =========================================================================
    `label`                 ``str``            The species' label
    `multiplicity`          ``int``            The species' multiplicity
    'charge'                ``int''            The species' net charge
    `is_ts`                 ``bool``           Whether or not the species represents a transition state
    'number_of_rotors'      ``int``            The number of potential rotors to scan
    'rotors_dict'           ``dict``           A dictionary of rotors. structure given below.
    `conformers`            ``list``           A list of all conformers XYZs
    `conformer_energies`    ``list``           A list of conformers E0 (Hartree)
    'initial_xyz'           ``string``         The initial geometry guess
    'final_xyz'             ``string``         The optimized species geometry
    'monoatomic'            ``bool``           Whether the species has only one atom or not
    ====================== =================== =========================================================================

    Dictionary structure:

*   rotors_dict: {1: {'pivots': pivots_list,
                      'top': top_list,
                      'scan': scan_list,
                      'success: ''bool''},
                  2: {}, ...
                 }
    """
    def __init__(self, is_ts, rmg_species=None, label=None, xyz=None, multiplicity=None, charge=None):
        """
        All parameters get precedence over their respective rmg_species values if the latter is given.
        'is_ts' is a mandatory parameter.
        If 'rmg_species' is given, all other parameters are optional.
        Note that if an xyz guess is given directly, localized (resonance) structures won't be generated
        """

        self.is_ts = is_ts

        self.rmg_species = rmg_species

        if self.rmg_species is None:
            if xyz is None:
                raise SpeciesError('xyz must be specified if an RMG Species isnt given.')
            if multiplicity is None:
                raise SpeciesError('multiplicity must be specified if an RMG Species isnt given.')
            if charge is None:
                raise SpeciesError('charge must be specified if an RMG Species isnt given.')
            if label is None:
                raise SpeciesError('label must be specified if an RMG Species isnt given.')
        else:
            if not isinstance(self.rmg_species, Species):
                raise SpeciesError('The rmg_species parameter has to be a valid RMG Species object.'
                                   ' Got: {0}'.format(type(self.rmg_species)))
            if not self.rmg_species.molecule:
                raise SpeciesError('If an RMG Species given, it must have a non-empty molecule list')
            if not self.rmg_species.label and not label:
                raise SpeciesError('If an RMG Species given, it must have a label or a label must be given separately')

        if self.rmg_species is not None:
            self.multiplicity = self.rmg_species.molecule[0].multiplicity
            self.charge = self.rmg_species.molecule[0].getNetCharge()
            self.label = self.rmg_species.label

        if multiplicity is not None:
            self.multiplicity = multiplicity

        if charge is not None:
            self.multiplicity = charge

        if label is not None:
            self.label = label

        if xyz is not None:
            self.initial_xyz = xyz
            self.monoatomic = len(xyz.split('\n')) == 1
        else:
            self.initial_xyz = ''

        self.final_xyz = ''

        self.number_of_rotors = 0
        self.rotors_dict = dict()

        self.conformers = list()
        self.conformer_energies = list()

        if self.rmg_species.molecule:
            self.molecule = self.rmg_species.molecule
            self.monoatomic = len(self.molecule[0].atoms) == 1
        else:
            self.molecule = list()

        self.xyzs = list()  # used for conformer search

    def find_conformers(self, mol, method='all'):
        """
        Generates conformers for `mol` which is an ``RMG.Molecule`` object using the method/s
        specified in `method`: 'rdkit', 'openbabel', or 'all'. result/s saved to self.xyzs
        """
        rdkit = False
        ob = False
        if method == 'all':
            rdkit = True
            ob = True
        elif method.lower() == 'rdkit':
            rdkit = True
        elif method.lower() in ['ob', 'openbabel']:
            ob = True

        if rdkit:
            rd_xyzs, rd_energies = self._get_possible_conformers_rdkit(mol)
            if rd_xyzs:
                rd_xyz = self.get_min_energy_conformer(xyzs=rd_xyzs, energies=rd_energies)
                self.xyzs.append(get_xyz_matrix(xyz=rd_xyz, mol=mol))
        if ob:
            ob_xyzs, ob_energies = self._get_possible_conformers_openbabel(mol)
            ob_xyz = self.get_min_energy_conformer(xyzs=ob_xyzs, energies=ob_energies)
            self.xyzs.append(get_xyz_matrix(xyz=ob_xyz, mol=mol))

    def _get_possible_conformers_rdkit(self, mol):
        """
        A helper function for conformer search
        Uses rdkit to automatically generate a set of len(mol.atoms)-3)*30 initial geometries, optimizes
        these geometries using MMFF94s, calculates the energies using MMFF94s
        and converts them back in terms of the RMG atom ordering
        Returns the coordinates and energies
        """
        rdmol, rdInds = mol.toRDKitMol(removeHs=False, returnMapping=True)
        rdIndMap = dict()
        for k, atm in enumerate(mol.atoms):
            ind = rdInds[atm]
            rdIndMap[ind] = k
        if len(mol.atoms) > 5:
            AllChem.EmbedMultipleConfs(rdmol, numConfs=(len(mol.atoms) - 3) * 30, randomSeed=1)
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

    def _get_possible_conformers_openbabel(self, mol):
        """
        A helper function for conformer search
        Uses OpenBabel to automatically generate set of len(mol.atoms)*10-3 initial geometries,
        optimizes these geometries using MMFF94s, calculates the energies using MMFF94s
        Returns the coordinates and energies
        """
        energies = []
        xyzs = []
        obmol, obInds = toOBMol(mol, returnMapping=True)
        pybmol = pyb.Molecule(obmol)
        pybmol.make3D()
        obmol = pybmol.OBMol

        ff = ob.OBForceField.FindForceField("mmff94s")
        ff.Setup(obmol)
        if len(mol.atoms) > 5:
            ff.WeightedRotorSearch(len(mol.atoms) * 10 - 3, 2000)
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

    def generate_localized_structures(self):
        """
        Generate localized (resonance) structures
        """
        if len(self.rmg_species.molecule) == 1:
            self.rmg_species.generate_resonance_structures(keep_isomorphic=False, filter_structures=True)
            self.molecule = self.rmg_species.molecule

    def generate_conformers(self):
        """
        Generate conformers using RDKit and OpenBabel for all representative localized structures of each species
        """
        if not self.initial_xyz:
            if self.molecule:
                for mol in self.molecule:
                    self.find_conformers(mol)
                    for xyz in self.xyzs:
                        self.conformers.append(xyz)
                        self.conformer_energies.append(0.0)  # a placeholder (lists are synced)
            else:
                logging.info(self.molecule)
                raise SpeciesError('Cannot generate conformers without a molecule list')
        else:
            logging.debug('Not generating conformers for species {0},'
                          ' since it already has an initial xyz'.format(self.label))

    def determine_rotors(self):
        """
        Determine possible unique rotors in the species to be treated as hindered rotors,
        taking into account all localized structures.
        The resulting rotors are saved in {'pivots': [1, 3], 'top': [3, 7], 'scan': [2, 1, 3, 7]} format
        in self.species_dict[species.label]['rotors_dict']. Also updates 'number_of_rotors'.
        """
        for mol in self.molecule:
            rotors = find_internal_rotors(mol)
            for new_rotor in rotors:
                for key, existing_rotor in self.rotors_dict.iteritems():
                    if existing_rotor['pivots'] == new_rotor['pivots']:
                        break
                else:
                    self.rotors_dict[self.number_of_rotors] = new_rotor
                    self.number_of_rotors += 1
        if self.number_of_rotors == 1:
            logging.info('Found 1 rotor for {0}'.format(self.label))
        elif self.number_of_rotors > 1:
            logging.info('Found {0} rotors for {1}'.format(self.number_of_rotors, self.label))
        if self.number_of_rotors > 0:
            logging.info('Pivot list(s) for {0}: {1}\n'.format(self.label,
                                                [self.rotors_dict[i]['pivots'] for i in xrange(self.number_of_rotors)]))


def find_internal_rotors(mol):
    """
    Locates the sets of indices corresponding to every internal rotor.
    Returns for each rotors the gaussian scan coordinates, the pivots and the top.
    """
    rotors = []
    for atom1 in mol.vertices:
        for atom2, bond in atom1.edges.items():
            if mol.vertices.index(atom1) < mol.vertices.index(atom2) \
                    and (bond.isSingle() or bond.isHydrogenBond()) and not mol.isBondInCycle(bond):
                if len(atom1.edges) > 1 and len(
                        atom2.edges) > 1:  # none of the pivotal atoms is terminal (nor hydrogen)
                    rotor = dict()
                    # pivots:
                    rotor['pivots'] = [mol.vertices.index(atom1) + 1, mol.vertices.index(atom2) + 1]
                    # top:
                    top1, top2 = [], []
                    top1_has_heavy_atoms, top2_has_heavy_atoms = False, False
                    atom_list = [atom1]
                    while len(atom_list):
                        for atom in atom_list:
                            top1.append(mol.vertices.index(atom) + 1)
                            for atom3, bond3 in atom.edges.items():
                                if atom3.isHydrogen():
                                    top1.append(mol.vertices.index(atom3) + 1)  # append H's
                                elif atom3 is not atom2:
                                    top1_has_heavy_atoms = True
                                    if not bond3.isSingle():
                                        top1.append(
                                            mol.vertices.index(atom3) + 1)  # append non-single-bonded heavy atoms
                                        atom_list.append(atom3)
                            atom_list.pop(atom_list.index(atom))
                    atom_list = [atom2]
                    while len(atom_list):
                        for atom in atom_list:
                            top2.append(mol.vertices.index(atom) + 1)
                            for atom3, bond3 in atom.edges.items():
                                if atom3.isHydrogen():
                                    top2.append(mol.vertices.index(atom3) + 1)  # append H's
                                elif atom3 is not atom1:
                                    top2_has_heavy_atoms = True
                                    if not bond3.isSingle():
                                        top2.append(
                                            mol.vertices.index(atom3) + 1)  # append non-single-bonded heavy atoms
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
                            hydrogens.append(mol.vertices.index(atom3))
                        elif atom3 is not atom2:
                            heavy_atoms.append(mol.vertices.index(atom3))
                    smallest_index = len(mol.vertices)
                    if len(heavy_atoms):
                        for i in heavy_atoms:
                            if i < smallest_index:
                                smallest_index = i
                    else:
                        for i in hydrogens:
                            if i < smallest_index:
                                smallest_index = i
                    rotor['scan'].append(smallest_index + 1)
                    rotor['scan'].extend([mol.vertices.index(atom1) + 1, mol.vertices.index(atom2) + 1])
                    heavy_atoms = []
                    hydrogens = []
                    for atom3, bond3 in atom2.edges.items():
                        if atom3.isHydrogen():
                            hydrogens.append(mol.vertices.index(atom3))
                        elif atom3 is not atom1:
                            heavy_atoms.append(mol.vertices.index(atom3))
                    smallest_index = len(mol.vertices)
                    if len(heavy_atoms):
                        for i in heavy_atoms:
                            if i < smallest_index:
                                smallest_index = i
                    else:
                        for i in hydrogens:
                            if i < smallest_index:
                                smallest_index = i
                    rotor['scan'].append(smallest_index + 1)
                    rotor['success'] = None
                    rotors.append(rotor)
    return rotors


def get_xyz_matrix(xyz, mol=None, from_arkane=False, number=None):
    """
    This function isn't defined as a method of ARCSpecies since it is also used when parsing opt geometry in Scheduler
    (using the from_arkane=True keyword)
    """
    if mol is None and not from_arkane:
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
            element_label = mol.atoms[i].element.symbol
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
