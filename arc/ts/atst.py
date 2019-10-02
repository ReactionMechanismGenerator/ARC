#!/usr/bin/env python3
# encoding: utf-8

"""
A module for calling AutoTST
"""

import os

from arc.exceptions import TSError
from arc.settings import arc_path
from arc.species.converter import str_to_xyz


def autotst(reaction_label=None, rmg_reaction=None, reaction_family=None):
    """
    Run AutoTST to generate a TS guess. Currently only works for H Abstraction reactions.
    Either `reaction_label` or `rmg_reaction` has to be given (ARC sends rmg_reaction).
    If reaction_family isn't specified, it is assumed to be H_Abstraction.

    Args:
        reaction_label (str): AutoTST's string format reaction, e.g., CCCC+[O]O_[CH2]CCC+OO
        rmg_reaction (Reaction): An RMG Reaction object.
        reaction_family (str): The RMG family corresponding to the RMG Reaction object

    Returns:
        xyz (dict): The TS guess.

    Raises:
        TSError: if neither ``rmg_reaction`` nor ``reaction_family`` were specified.
    """
    xyz_str = ''
    xyz_path = os.path.join(arc_path, 'arc', 'ts', 'auto_tst.xyz')
    run_autotst_path = os.path.join(arc_path, 'arc', 'ts', 'run_autotst.py')

    reaction_family = str('H_Abstraction') if reaction_family is None else str(reaction_family)

    if os.path.isfile(xyz_path):
        os.remove(xyz_path)
    if rmg_reaction is not None and reaction_label is None:
        reaction_label = get_reaction_label(rmg_reaction)
    elif reaction_label is None:
        raise TSError('Must get either reaction_label or rmg_reaction')
    os.system('python {run_autotst_path} {reaction_label} {reaction_family}'.format(
        run_autotst_path=run_autotst_path, reaction_label=reaction_label, reaction_family=reaction_family))

    if os.path.isfile(xyz_path):
        with open(xyz_path, 'r') as f:
            lines = f.readlines()
            xyz_str = ''.join([str(line) for line in lines])
        os.remove(xyz_path)
    if not xyz_str or xyz_str == '\n':
        return None
    return str_to_xyz(xyz_str)


def get_reaction_label(rmg_reaction):
    """
    Returns the AutoTST reaction string in the form of r1+r2_p1+p2 (e.g., `CCC+[O]O_[CH2]CC+OO`).
    `reactants` and `products` are lists of class:`Molecule`s.
    """
    reactants = rmg_reaction.reactants
    products = rmg_reaction.products
    if len(reactants) > 1:
        reactants_string = '+'.join([reactant.molecule[0].to_smiles() for reactant in reactants])
    else:
        reactants_string = reactants[0].molecule[0].to_smiles()
    if len(products) > 1:
        products_string = '+'.join([product.molecule[0].to_smiles() for product in products])
    else:
        products_string = products[0].molecule[0].to_smiles()
    reaction_label = '_'.join([reactants_string, products_string])
    return reaction_label
