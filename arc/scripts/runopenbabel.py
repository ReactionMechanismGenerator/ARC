#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to execute OpenBabel
to generate conformers or to optimize conformers using force fields.
"""

import argparse
import os
from typing import Optional, Tuple

from openbabel import openbabel as ob
from openbabel import pybel as pyb

from rmgpy.molecule.converter import to_ob_mol

from arc.common import read_yaml_file, rmg_mol_from_dict_repr, save_yaml_file
from arc.species.conformers import embed_rdkit
from arc.species.converter import str_to_xyz


ob.obErrorLog.SetOutputLevel(0)


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.
    This uses the :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """
    parser = argparse.ArgumentParser(description='OpenBabel')
    parser.add_argument('file', metavar='FILE', type=str, nargs=1,
                        help='a YAML input file describing the job to execute')
    parser.add_argument('num', metavar='OUTPUT_NUM', type=str, nargs=1,
                        help='an identifier for the output number')

    args = parser.parse_args(command_line_args)

    # Process args to set correct default values and format.
    args.file = args.file[0]
    args.num = args.num[0]

    return args


def main():
    """
    Run OpenBabel wrapper.
    """
    # Parse command-line arguments
    args = parse_command_line_arguments()
    input_file = args.file
    output_num = args.num

    function_dict = {'openbabel_force_field_on_rdkit_conformers': openbabel_force_field_on_rdkit_conformers,
                     'openbabel_force_field': openbabel_force_field,
                     }

    # Execute.
    try:
        input_dict = read_yaml_file(path=input_file)
        function = function_dict[input_dict['function']]
        del input_dict['function']
        xyzs, energies = function(**input_dict)
    except Exception as e:
        print(f'Could not execute OpenBabel, got the following error:\n{e}')
        return None

    # Process output.
    output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), f'output_{output_num}.yml')
    save_yaml_file(path=output_path, content={'xyzs': xyzs, 'energies': energies})


def openbabel_force_field_on_rdkit_conformers(label: str,
                                              mol_dict: dict,
                                              num_confs: Optional[int] = None,
                                              xyz: Optional[dict] = None,
                                              force_field: str = 'MMFF94s',
                                              optimize: bool = True,
                                              ) -> Tuple[list, list]:
    """
    Optimize RDKit conformers by OpenBabel using a force field (MMFF94 or MMFF94s are recommended).
    This is a fall back method when RDKit fails to generate force field optimized conformers.

    Args:
        label (str): The species' label.
        mol_dict (dict): A dict representation of an RMG molecule object with connectivity and bond order information.
        num_confs (int, optional): The number of random 3D conformations to generate.
        xyz (dict, optional): The 3D coordinates guess.
        force_field (str, optional): The type of force field to use.
        optimize (bool, optional): Whether to first optimize the conformer using FF. True to optimize.

    Returns:
        Tuple[list, list]:
            - Entries are optimized xyz's in a dictionary format.
            - Entries are float numbers representing the energies (in kJ/mol).
    """
    mol = rmg_mol_from_dict_repr(mol_dict)
    rd_mol = embed_rdkit(label, mol, num_confs=num_confs, xyz=xyz)
    xyzs, energies = list(), list()
    # Set up Openbabel input and output format
    obconversion = ob.OBConversion()
    obconversion.SetInAndOutFormats('xyz', 'xyz')
    # Set up Openbabel force field
    ff = ob.OBForceField.FindForceField(force_field)
    symbols = [rd_atom.GetSymbol() for rd_atom in rd_mol.GetAtoms()]
    for i in range(rd_mol.GetNumConformers()):
        # Convert RDKit conformer to xyz string
        conf = rd_mol.GetConformer(i)
        xyz_str = f'{conf.GetNumAtoms()}\n\n'
        for j in range(conf.GetNumAtoms()):
            xyz_str += symbols[j] + '      '
            pt = conf.GetAtomPosition(j)
            xyz_str += '   '.join([str(pt.x), str(pt.y), str(pt.z)]) + '\n'
        # Build OpenBabel molecule from xyz string
        ob_mol = ob.OBMol()
        obconversion.ReadString(ob_mol, xyz_str)
        ff.Setup(ob_mol)
        # Optimize the molecule if needed
        if optimize:
            ff.ConjugateGradients(2000)
        # Export xyzs and energies
        ob_mol.GetCoordinates()
        ff.GetCoordinates(ob_mol)
        energies.append(ff.Energy())
        xyz_str = '\n'.join(obconversion.WriteString(ob_mol).splitlines()[2:])
        xyzs.append(str_to_xyz(xyz_str))
    return xyzs, energies


def openbabel_force_field(label,
                          mol_dict,
                          num_confs: Optional[int] = None,
                          xyz: Optional[dict] = None,
                          force_field: str = 'GAFF',
                          method: str = 'diverse',
                          ) -> Tuple[list, list]:
    """
    Optimize conformers using a force field (GAFF, MMFF94s, MMFF94, UFF, Ghemical).

    Args:
        label (str): The species' label.
        mol_dict (dict): A dict representation of an RMG molecule object with connectivity and bond order information.
        num_confs (int, optional): The number of random 3D conformations to generate.
        xyz (dict, optional): The 3D coordinates.
        force_field (str, optional): The type of force field to use.
        method (str, optional): The conformer searching method to use in OpenBabel.
                                For method description, see http://openbabel.org/dev-api/group__conformer.shtml

    Returns:
        Tuple[list, list]:
            - Entries are optimized xyz's in a list format.
            - Entries are float numbers representing the energies in kJ/mol.
    """
    mol = rmg_mol_from_dict_repr(mol_dict)
    xyzs, energies = list(), list()
    ff = ob.OBForceField.FindForceField(force_field)

    if xyz is not None:
        # generate an OpenBabel molecule
        obmol = ob.OBMol()
        atoms = mol.vertices
        ob_atom_ids = dict()  # dictionary of OB atom IDs
        for i, atom in enumerate(atoms):
            a = obmol.NewAtom()
            a.SetAtomicNum(atom.number)
            a.SetVector(xyz['coords'][i][0], xyz['coords'][i][1], xyz['coords'][i][2])
            if atom.element.isotope != -1:
                a.SetIsotope(atom.element.isotope)
            a.SetFormalCharge(atom.charge)
            ob_atom_ids[atom] = a.GetId()
        orders = {1: 1, 2: 2, 3: 3, 4: 4, 1.5: 5}
        for atom1 in mol.vertices:
            for atom2, bond in atom1.edges.items():
                if bond.is_hydrogen_bond():
                    continue
                index1 = atoms.index(atom1)
                index2 = atoms.index(atom2)
                if index1 < index2:
                    obmol.AddBond(index1 + 1, index2 + 1, orders[bond.order])

        # optimize
        ff.Setup(obmol)
        ff.SetLogLevel(0)
        ff.SetVDWCutOff(6.0)  # The VDW cut-off distance (default=6.0)
        ff.SetElectrostaticCutOff(10.0)  # The Electrostatic cut-off distance (default=10.0)
        ff.SetUpdateFrequency(10)  # The frequency to update the non-bonded pairs (default=10)
        ff.EnableCutOff(False)  # Use cut-off (default=don't use cut-off)
        # ff.SetLineSearchType('Newton2Num')
        ff.SteepestDescentInitialize()  # ConjugateGradientsInitialize
        v = 1
        while v:
            v = ff.SteepestDescentTakeNSteps(1)  # ConjugateGradientsTakeNSteps
            if ff.DetectExplosion():
                raise ValueError(f'Force field {force_field} exploded with method SteepestDescent for {label}')
        ff.GetCoordinates(obmol)

    elif num_confs is not None:
        obmol, ob_atom_ids = to_ob_mol(mol, return_mapping=True)
        pybmol = pyb.Molecule(obmol)
        pybmol.make3D()
        obmol = pybmol.OBMol
        ff.Setup(obmol)

        if method.lower() == 'weighted':
            ff.WeightedRotorSearch(num_confs, 2000)
        elif method.lower() == 'random':
            ff.RandomRotorSearch(num_confs, 2000)
        elif method.lower() == 'diverse':
            rmsd_cutoff = 0.5
            energy_cutoff = 50.
            confab_verbose = False
            ff.DiverseConfGen(rmsd_cutoff, num_confs, energy_cutoff, confab_verbose)
        elif method.lower() == 'systematic':
            ff.SystematicRotorSearch(num_confs)
        else:
            raise ValueError(f'Could not identify method {method} for {label}')
    else:
        raise ValueError(f'Either num_confs or xyz should be given for {label}')

    ff.GetConformers(obmol)
    obconversion = ob.OBConversion()
    obconversion.SetOutFormat('xyz')

    for i in range(obmol.NumConformers()):
        obmol.SetConformer(i)
        ff.Setup(obmol)
        xyz_str = '\n'.join(obconversion.WriteString(obmol).splitlines()[2:])
        xyz_dict = str_to_xyz(xyz_str)
        # reorder:
        xyz_dict['coords'] = tuple(xyz_dict['coords'][ob_atom_ids[mol.atoms[j]]]
                                   for j in range(len(xyz_dict['coords'])))
        xyzs.append(xyz_dict)
        energies.append(ff.Energy())
    return xyzs, energies


if __name__ == '__main__':
    main()
