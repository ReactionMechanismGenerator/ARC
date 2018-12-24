#!/usr/bin/env python
# encoding: utf-8

import os
import logging
import string
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdmt
from rdkit.Chem.rdchem import EditableMol as RDMol
import openbabel as ob
import pybel as pyb

from rmgpy.molecule.converter import toOBMol
from rmgpy.molecule.element import getElement
from rmgpy.species import Species
from rmgpy.molecule.molecule import Atom, Molecule
from rmgpy.qm.qmdata import QMData
from rmgpy.qm.symmetry import PointGroupCalculator

from arc.exceptions import SpeciesError, RotorError
from arc.settings import arc_path

##################################################################


class ARCSpecies(object):
    """
    ARCSpecies class

    ====================== ============= ===============================================================================
    Attribute              Type          Description
    ====================== ============= ===============================================================================
    `label`                 ``str``      The species' label
    `multiplicity`          ``int``      The species' multiplicity
    'charge'                ``int''      The species' net charge
    `is_ts`                 ``bool``     Whether or not the species represents a transition state
    'number_of_rotors'      ``int``      The number of potential rotors to scan
    'rotors_dict'           ``dict``     A dictionary of rotors. structure given below.
    `conformers`            ``list``     A list of selected conformers XYZs
    `conformer_energies`    ``list``     A list of conformers E0 (Hartree)
    'initial_xyz'           ``string``   The initial geometry guess
    'final_xyz'             ``string``   The optimized species geometry
    'number_of_atoms'       ``int``      The number of atoms in the species/TS
    'smiles'                ``str``      The SMILES structure. Either SMILES, adjList, or mol is required for BAC.
    'adjlist'               ``str``      The Adjacency List structure.
    'mol'                   ``Molecule`` An RMG:Molecule object used for BAC determination.
                                           Does not correctly describe bond orders.
    'bond_corrections'      ``dict``     The bond additivity corrections (BAC) to be used. Determined from the structure
                                           if not directly given.
    't0'                    ``float``    Initial time when the first species job was spawned
    `neg_freqs_trshed`      ``list``     A list of negative frequencies this species was troubleshooted for
    ====================== ============= ===============================================================================

    Dictionary structure:

*   rotors_dict: {1: {'pivots': pivots_list,
                      'top': top_list,
                      'scan': scan_list,
                      'success: ''bool''.
                      'times_dihedral_set': ``int``,
                      'scan_path': <path to scan output file>},
                  2: {}, ...
                 }
    """
    def __init__(self, is_ts=False, rmg_species=None, mol=None, label=None, xyz=None, multiplicity=None, charge=None,
                 smiles='', adjlist='', bond_corrections=None,):
        self.is_ts = is_ts
        self.t0 = None

        self.rmg_species = rmg_species
        if bond_corrections is None:
            self.bond_corrections = dict()
        else:
            self.bond_corrections = bond_corrections

        self.mol = mol
        self.mol_no_bond_info = None
        self.initial_xyz = xyz

        if self.rmg_species is None:
            # parameters were entered directly, not via an RMG:Species object
            if label is None:
                raise SpeciesError('label must be specified for an ARCSpecies object.')
            else:
                self.label = label
            if multiplicity is None:
                raise SpeciesError('No multiplicity was specified for {0}.'.format(self.label))
            if charge is None:
                raise SpeciesError('No charge was specified for {0}.'.format(self.label))
            if adjlist and not mol:
                self.mol = Molecule().fromAdjacencyList(adjlist=adjlist)
            elif smiles and not mol:
                self.mol = Molecule(SMILES=smiles)
            if not self.is_ts and not smiles and not adjlist and not mol:
                logging.warn('No structure (SMILES, adjList, or an RMG:Species object) was given for species {0},'
                             ' NOT using bond additivity corrections (BAC) for thermo computation'.format(label))
            if multiplicity < 1:
                raise SpeciesError('Multiplicity for species {0} is lower than 1 (got {1})'.format(
                    self.label, multiplicity))
            if not isinstance(multiplicity, int):
                raise SpeciesError('Multiplicity for species {0} is not an integer (got {1}, a {2})'.format(
                    self.label, multiplicity, type(multiplicity)))
            if not isinstance(charge, int):
                raise SpeciesError('Charge for species {0} is not an integer (got {1}, a {2})'.format(
                    self.label, charge, type(charge)))
            self.multiplicity = multiplicity
            self.charge = charge
        else:
            # an RMG Species was given
            if not isinstance(self.rmg_species, Species):
                raise SpeciesError('The rmg_species parameter has to be a valid RMG Species object.'
                                   ' Got: {0}'.format(type(self.rmg_species)))
            if not self.rmg_species.molecule:
                raise SpeciesError('If an RMG Species given, it must have a non-empty molecule list')
            if not self.rmg_species.label and not label:
                raise SpeciesError('If an RMG Species given, it must have a label or a label must be given separately')
            else:
                if self.rmg_species.label:
                    self.label = self.rmg_species.label
                else:
                    self.label = label
            if not self.mol:
                self.mol = self.rmg_species.molecule[0]
            if len(self.rmg_species.molecule) > 1:
                logging.info('Using localized structure {0} of species {1} for BAC determination. To use a different'
                             ' structure, pass the RMG:Molecule object in the `mol` parameter'.format(
                                self.mol.toSMILES(), self.label))
            self.multiplicity = self.rmg_species.molecule[0].multiplicity
            self.charge = self.rmg_species.molecule[0].getNetCharge()

        # Check `label` is valid, since it is used for folder names
        valid_chars = "-_()<=>%s%s" % (string.ascii_letters, string.digits)
        for char in self.label:
            if char not in valid_chars:
                raise SpeciesError('Species label {0} contains an invalid character: "{1}"'.format(self.label, char))

        if self.mol is None:
            mol, _ = mol_from_xyz(self.initial_xyz)
            self.mol_list = [mol]
            self.number_of_atoms = len(mol.atoms)
        else:
            self.mol_list = self.mol.generate_resonance_structures(keep_isomorphic=False, filter_structures=True)
            if not bond_corrections:
                self.determine_bond_corrections()
            self.number_of_atoms = len(self.mol.atoms)

        self.final_xyz = ''
        self.number_of_rotors = 0
        self.rotors_dict = dict()
        self.conformers = list()
        self.conformer_energies = list()
        if self.initial_xyz is not None:
            # consider the initial guess as one of the conformers if generating others.
            # otherwise, just consider it as the conformer.
            self.conformers.append(self.initial_xyz)
            self.conformer_energies.append(0.0)  # dummy

        self.xyzs = list()  # used for conformer search
        self.external_symmetry = 1
        self.optical_isomers = 1
        self.neg_freqs_trshed = list()

    def generate_conformers(self):
        """
        Generate conformers using RDKit and OpenBabel for all representative localized structures of each species
        """
        if self.mol:
            for mol in self.mol_list:
                self.find_conformers(mol)
            for xyz in self.xyzs:
                self.conformers.append(xyz)
                self.conformer_energies.append(0.0)  # a placeholder (lists are synced)
        else:
            logging.warn('Generating conformers for species {0}, without bond order information (using coordinates'
                         ' only).'.format(self.label))
            mol, coordinates = mol_from_xyz(self.initial_xyz)
            rd_mol, rd_inds = mol.toRDKitMol(removeHs=False, returnMapping=True)
            Chem.AllChem.EmbedMolecule(rd_mol)  # unfortunately, this mandatory embedding changes the coordinates
            indx_map = dict()
            for xyz_index, atom in enumerate(mol.atoms):  # generate an atom index mapping dictionary
                rd_index = rd_inds[atom]
                indx_map[xyz_index] = rd_index
            conf = rd_mol.GetConformer(id=0)
            for i in xrange(rd_mol.GetNumAtoms()):  # reset atom coordinates
                conf.SetAtomPosition(indx_map[i], coordinates[i])
            self.find_conformers(mol, method='rdkit')
            for xyz in self.xyzs:
                self.conformers.append(xyz)
                self.conformer_energies.append(0.0)  # a placeholder (lists are synced)

    def find_conformers(self, mol, method='all'):
        """
        Generates conformers for `mol` which is an ``RMG.Molecule`` object using the method/s
        specified in `method`: 'rdkit', 'openbabel', or 'all'. result/s saved to self.xyzs
        """
        rdkit, ob = False, False
        rd_xyzs, ob_xyzs = list(), list()
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
        logging.debug('Considering {actual} conformers for {label} out of {total} total ran using a force field'.format(
            actual=len(self.xyzs), total=len(rd_xyzs+ob_xyzs), label=self.label))

    def _get_possible_conformers_rdkit(self, mol):
        """
        A helper function for conformer search
        Uses rdkit to automatically generate a set of len(mol.atoms)-3)*30 initial geometries, optimizes
        these geometries using MMFF94s, calculates the energies using MMFF94s
        and converts them back in terms of the RMG atom ordering
        Returns the coordinates and energies
        """
        if not isinstance(mol, (Molecule, RDMol)):
            raise SpeciesError('Can generate conformers to either an RDKit or RMG molecule. Got {0}'.format(type(mol)))
        if isinstance(mol, RDMol):
            rd_mol = mol
        else:
            rd_mol, rd_inds = mol.toRDKitMol(removeHs=False, returnMapping=True)
            rd_indx_map = dict()
            for k, atom in enumerate(mol.atoms):
                ind = rd_inds[atom]
                rd_indx_map[ind] = k
        if len(mol.atoms) > 50:
            Chem.AllChem.EmbedMultipleConfs(rd_mol, numConfs=(len(mol.atoms)) * 3, randomSeed=1)
        elif len(mol.atoms) > 5:
            Chem.AllChem.EmbedMultipleConfs(rd_mol, numConfs=(len(mol.atoms) - 3) * 30, randomSeed=1)
        else:
            Chem.AllChem.EmbedMultipleConfs(rd_mol, numConfs=120, randomSeed=1)
        energies = []
        xyzs = []
        for i in xrange(rd_mol.GetNumConformers()):
            v = 1
            while v == 1:
                v = Chem.AllChem.MMFFOptimizeMolecule(rd_mol, mmffVariant='MMFF94s', confId=i,
                                                 maxIters=500, ignoreInterfragInteractions=False)
            mp = Chem.AllChem.MMFFGetMoleculeProperties(rd_mol, mmffVariant='MMFF94s')
            if mp is not None:
                ff = Chem.AllChem.MMFFGetMoleculeForceField(rd_mol, mp, confId=i)
                E = ff.CalcEnergy()
                energies.append(E)
                cf = rd_mol.GetConformer(i)
                xyz = []
                for j in xrange(cf.GetNumAtoms()):
                    pt = cf.GetAtomPosition(j)
                    xyz.append([pt.x, pt.y, pt.z])
                if isinstance(mol, Molecule):
                    xyz = [xyz[rd_indx_map[i]] for i in xrange(len(xyz))]  # reorder
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

    def determine_rotors(self):
        """
        Determine possible unique rotors in the species to be treated as hindered rotors,
        taking into account all localized structures.
        The resulting rotors are saved in {'pivots': [1, 3], 'top': [3, 7], 'scan': [2, 1, 3, 7]} format
        in self.species_dict[species.label]['rotors_dict']. Also updates 'number_of_rotors'.
        """
        for mol in self.mol_list:
            rotors = find_internal_rotors(mol)
            for new_rotor in rotors:
                for key, existing_rotor in self.rotors_dict.iteritems():
                    if existing_rotor['pivots'] == new_rotor['pivots']:
                        break
                else:
                    self.rotors_dict[self.number_of_rotors] = new_rotor
                    self.number_of_rotors += 1
        if self.number_of_rotors == 1:
            logging.info('\nFound 1 rotor for {0}'.format(self.label))
        elif self.number_of_rotors > 1:
            logging.info('\nFound {0} rotors for {1}'.format(self.number_of_rotors, self.label))
        if self.number_of_rotors > 0:
            logging.info('Pivot list(s) for {0}: {1}\n'.format(self.label,
                                                [self.rotors_dict[i]['pivots'] for i in xrange(self.number_of_rotors)]))

    def set_dihedral(self, scan, pivots, deg_increment):
        """
        Generated an RDKit molecule object from the given self.final_xyz.
        Increments the current dihedral angle between atoms i, j, k, l in the `scan` list by 'deg_increment` in degrees.
        All bonded atoms are moved accordingly. The result is saved in self.initial_xyz.
        `pivots` is used to identify the rotor.
        """
        # TODO: show 3D structure before and after the change
        if deg_increment == 0:
            logging.warning('set_dihedral was called with zero increment for {label} with pivots {pivots}'.format(
                label=self.label, pivots=pivots))
            for rotor in self.rotors_dict.itervalues():  # penalize this rotor to avoid inf. looping
                if rotor['pivots'] == pivots:
                    rotor['times_dihedral_set'] += 1
                    break
        else:
            for rotor in self.rotors_dict.itervalues():
                if rotor['pivots'] == pivots and rotor['times_dihedral_set'] <= 10:
                    rotor['times_dihedral_set'] += 1
                    break
            else:
                logging.info('\n\n')
                for i, rotor in self.rotors_dict.iteritems():
                    logging.error('Rotor {i} with pivots {pivots} was set {times} times'.format(
                        i=i, pivots=rotor['pivots'], times=rotor['times_dihedral_set']))
                raise RotorError('Rotors for {0} were set beyond the maximal number of times without converging')
            for i in xrange(len(scan)):
                scan[i] -= 1  # atom indices start from 0, but atom labels (as in scan) start from 1
            mol = Molecule()
            coordinates = list()
            for line in self.final_xyz.split('\n'):
                if line:
                    atom = Atom(element=line.split()[0])
                    coordinates.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
                    atom.coords = np.array(coordinates[-1], np.float64)
                    mol.addAtom(atom)
            mol.connectTheDots()  # only adds single bonds, but we don't care
            rd_mol, rd_inds = mol.toRDKitMol(removeHs=False, returnMapping=True)
            Chem.AllChem.EmbedMolecule(rd_mol)  # unfortunately, this mandatory embedding changes the coordinates
            indx_map = dict()
            for xyz_index, atom in enumerate(mol.atoms):  # generate an atom index mapping dictionary
                rd_index = rd_inds[atom]
                indx_map[xyz_index] = rd_index
            conf = rd_mol.GetConformer(id=0)
            for i in xrange(rd_mol.GetNumAtoms()):  # reset atom coordinates
                conf.SetAtomPosition(indx_map[i], coordinates[i])

            rd_scan = [indx_map[scan[i]] for i in xrange(4)]  # convert the atom indices in `scan` to RDkit indices

            deg0 = rdmt.GetDihedralDeg(conf, rd_scan[0], rd_scan[1], rd_scan[2], rd_scan[3])  # get the original dihedral
            deg = deg0 + deg_increment
            rdmt.SetDihedralDeg(conf, rd_scan[0], rd_scan[1], rd_scan[2], rd_scan[3], deg)
            new_xyz = list()
            for i in xrange(rd_mol.GetNumAtoms()):
                new_xyz.append([conf.GetAtomPosition(indx_map[i]).x, conf.GetAtomPosition(indx_map[i]).y,
                            conf.GetAtomPosition(indx_map[i]).z])
            self.initial_xyz = get_xyz_matrix(new_xyz, mol=mol)

    def determine_symmetry(self):
        """
        Determine external symmetry and optical isomers
        """
        # TODO: test this on several benchmark species
        atom_numbers = list()  # List of atomic numbers
        coordinates = list()
        for line in self.final_xyz.split('\n'):
            if line:
                atom_numbers.append(getElement(line.split()[0]).number)
                coordinates.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        coordinates = np.array(coordinates, np.float64)  # N x 3 numpy.ndarray of atomic coordinates
        #  in the same order as `atom_numbers`
        unique_id = '0'  # Just some name that the SYMMETRY code gives to one of its jobs
        scr_dir = os.path.join(arc_path, 'scratch')  # Scratch directory that the SYMMETRY code writes its files in
        if not os.path.exists(scr_dir):
            os.makedirs(scr_dir)
        symmetry = optical_isomers = 1
        qmdata = QMData(
            groundStateDegeneracy=1,  # Only needed to check if valid QMData
            numberOfAtoms=len(atom_numbers),
            atomicNumbers=atom_numbers,
            atomCoords=(coordinates, 'angstrom'),
            energy=(0.0, 'kcal/mol')  # Only needed to avoid error
        )
        settings = type("", (), dict(symmetryPath='symmetry', scratchDirectory=scr_dir))()  # Creates anonymous class
        pgc = PointGroupCalculator(settings, unique_id, qmdata)
        pg = pgc.calculate()
        if pg is not None:
            symmetry = pg.symmetryNumber
            optical_isomers = 2 if pg.chiral else optical_isomers
        self.optical_isomers = optical_isomers
        self.external_symmetry = symmetry

    def determine_bond_corrections(self):
        """
        A helper function to determine bond types for applying BAC
        """
        explored_atoms = []
        for atom1 in self.mol.vertices:
            for atom2, bond12 in atom1.edges.items():
                if atom2 not in explored_atoms:
                    bac = atom1.symbol
                    if bond12.isSingle():
                        bac += '-'
                    elif bond12.isDouble():
                        bac += '='
                    elif bond12.isTriple():
                        bac += '#'
                    else:
                        break
                    bac += atom2.symbol
                    if bac in self.bond_corrections:
                        self.bond_corrections[bac] += 1
                    elif bac[::-1] in self.bond_corrections:  # check in reverse
                        self.bond_corrections[bac] += 1
                    else:
                        self.bond_corrections[bac] = 1
            explored_atoms.append(atom1)
        logging.debug('Using the following BAC for {0}: {1}'.format(self.label, self.bond_corrections))


def find_internal_rotors(mol):
    """
    Locates the sets of indices corresponding to every internal rotor.
    Returns for each rotors the gaussian scan coordinates, the pivots and the top.
    """
    # TODO: find rotors for xyz input
    rotors = []
    for atom1 in mol.vertices:
        if atom1.isNonHydrogen():
            for atom2, bond in atom1.edges.items():
                if atom2.isNonHydrogen() and mol.vertices.index(atom1) < mol.vertices.index(atom2) \
                        and (bond.isSingle() or bond.isHydrogenBond()) and not mol.isBondInCycle(bond):
                    if len(atom1.edges) > 1 and len(atom2.edges) > 1:  # none of the pivotal atoms are terminal
                        rotor = dict()
                        # pivots:
                        rotor['pivots'] = [mol.vertices.index(atom1) + 1, mol.vertices.index(atom2) + 1]
                        # top:
                        top1, top2 = [], []
                        top1_has_heavy_atoms, top2_has_heavy_atoms = False, False
                        explored_atom_list = [atom2]
                        atom_list_to_explore = [atom1]
                        while len(atom_list_to_explore):
                            for atom in atom_list_to_explore:
                                top1.append(mol.vertices.index(atom) + 1)
                                for atom3, bond3 in atom.edges.items():
                                    if atom3.isHydrogen():
                                        # append H w/o further exploring
                                        top1.append(mol.vertices.index(atom3) + 1)
                                    elif atom3 not in explored_atom_list:
                                        top1_has_heavy_atoms = True
                                        atom_list_to_explore.append(atom3)  # explore it further
                                atom_list_to_explore.pop(atom_list_to_explore.index(atom))
                                explored_atom_list.append(atom)  # mark as explored
                        explored_atom_list, atom_list_to_explore = [atom1, atom2], [atom2]
                        while len(atom_list_to_explore):
                            for atom in atom_list_to_explore:
                                top2.append(mol.vertices.index(atom) + 1)
                                for atom3, bond3 in atom.edges.items():
                                    if atom3.isHydrogen():
                                        # append H w/o further exploring
                                        top2.append(mol.vertices.index(atom3) + 1)
                                    elif atom3 not in explored_atom_list:
                                        top2_has_heavy_atoms = True
                                        atom_list_to_explore.append(atom3)  # explore it further
                                atom_list_to_explore.pop(atom_list_to_explore.index(atom))
                                explored_atom_list.append(atom)  # mark as explored
                        if top1_has_heavy_atoms and not top2_has_heavy_atoms:
                            rotor['top'] = top2
                        elif top2_has_heavy_atoms and not top1_has_heavy_atoms:
                            rotor['top'] = top1
                        else:
                            rotor['top'] = top1 if len(top1) <= len(top2) else top2
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
                        rotor['times_dihedral_set'] = 0
                        rotor['scan_path'] = ''
                        rotors.append(rotor)
    return rotors


def get_xyz_matrix(xyz, mol=None, number=None):
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
    The atom symbol is derived from either an RMG Molecule object (`mol`) or atom numbers ('number`).
    This function isn't defined as a method of ARCSpecies since it is also used when parsing opt geometry in Scheduler
    """
    if mol is None and number is None:
        raise ValueError("Must have either an RMG:Molecule object input as `mol`, or atomic numbers.")
    result = ''
    for i, coord in enumerate(xyz):
        if mol is not None:
            element_label = mol.atoms[i].element.symbol
        else:
            element_label = getElement(int(number[i])).symbol
        result += element_label + ' ' * (4 - len(element_label))
        for j, c in enumerate(coord):
            result += '{0:14.8f}'.format(c)
        result += '\n'
    return result


def determine_occ(label, xyz, charge):
    """
    Determines the number of occupied orbitals for an MRCI calculation
    """
    electrons = 0
    for line in xyz.split('\n'):
        if line:
            atom = Atom(element=line.split()[0])
            electrons += atom.number
    electrons -= charge


def mol_from_xyz(xyz):
    """
    A helper function for creating an `RMG:Molecule` object from xyz
    """
    mol = Molecule()
    coordinates = list()
    for line in xyz.split('\n'):
        atom = Atom(element=line.split()[0])
        coordinates.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        atom.coords = np.array(coordinates[-1], np.float64)
        mol.addAtom(atom)
    mol.connectTheDots()  # only adds single bonds, but we don't care
    return mol, coordinates


# TODO: isomorphism check for final conformer
# TODO: solve chirality issues? How can I get the N4 isomer?
#  RDkit has 'FindMolChiralCenters'
# and rdkit.Chem.rdmolops.AssignAtomChiralTagsFromStructure, rdkit.Chem.rdmolops.AssignStereochemistry
# TODO: parse spin contamination and multireference characteristic from sp file in output dictionary
# TODO: spin contamination (in molpro, this is in the **output.log** file(!) as
#  "Spin contamination <S**2-Sz**2-Sz>     0.00000000")

