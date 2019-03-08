#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import logging
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdmt
from rdkit.Chem.rdchem import EditableMol as RDMol
import openbabel as ob
import pybel as pyb

from arc.arc_exceptions import ZMatrixError
from arc.settings import arc_path, default_ts_methods, valid_chars, minimum_barrier
from arc.parser import parse_xyz_from_file
from arc.species.converter import get_xyz_matrix, check_xyz
from arc.ts import atst

##################################################################


class ZMatrix(object):
    """
    ZMatrix class

    ====================== ============= ===============================================================================
    Attribute              Type          Description
    ====================== ============= ===============================================================================
    `label`                 ``str``      The species' label
    `multiplicity`          ``int``      The species' multiplicity. Can be determined from adjlist/smiles/xyz
    ====================== ============= ===============================================================================

    """
    def __init__(self, xyz):
        self.xyz = check_xyz(xyz)
        self.xyz_matrix, self.symbols, self.x, self.y, self.z = get_xyz_matrix(xyz)



class ZMatrixAtom(object):
    """
    ZMatrixAtom class

    ====================== ============= ===============================================================================
    Attribute              Type          Description
    ====================== ============= ===============================================================================
    `index`                 ``int``      The atom index of a molecule (starting at 1)
    `symbol`                ``str``      The chemical symbol of atom
    `distance`              ``float``    Distance between two atoms (in Angstrom)
    `angle`                 ``float``    Angle between three atoms (in degrees)
    `dihedral`              ``float``    Dihedral angle between four atoms (in degrees)
    `indices_list`          ``list``     A list of relevant atom indices.
                                          [self.index, reference (r, a, d), reference (a, d), reference (d)]
                                          r = distance, a = angle, d = dihedral
                                          Usage: to get distance, use indices_list[:1]
                                          Usage: to get angle, use indices_list[:2]
                                          Usage: to get dihedral, use indices_list[:3]
    ====================== ============= ===============================================================================

    """
    def __init__(self, symbol, distance, angle, dihedral, indices_list, index=None):
        self.index = index if index is not None else indices_list[0]
        self.symbol = symbol
        self.distance = distance
        self.angle = angle
        self.dihedral = dihedral
        self.indices_list = indices_list

        if self.index != self.indices_list[0]:
            raise ZMatrixError('Index and first entry of indices_list must be identical.')






