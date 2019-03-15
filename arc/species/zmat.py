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
from arc.species.converter import get_xyz_matrix, standardize_xyz_string
from arc.ts import atst

##################################################################


class ZMatrix(object):
    """
    ZMatrix class

    ====================== ============= ===============================================================================
    Attribute              Type          Description
    ====================== ============= ===============================================================================
    `xyz`                   ``str``      Species xyz representation
    `zmat_str`              ``str``      Species zmat representation
    `zmat`                  ``list``      List of zmat atom items
    ====================== ============= ===============================================================================

    """
    def __init__(self, xyz=None, zmat_str=None):
        self.xyz = standardize_xyz_string(xyz)
        self.zmat_str = zmat_str
        self.zmat = []
        if self.xyz is None and self.zmat_str is None:
            raise ZMatrixError('The ZMatrix object must have at least xyz or zmat_str as input')
        if self.zmat_str is not None:
            self.from_zmat_str()
        else:
            self.from_xyz()

    def from_xyz(self):
        """
        Construct self.zmat from input self.xyz
        """
        xyz_matrix, symbols, x, y, z = get_xyz_matrix(self.xyz)

    def from_zmat_str(self):
        """
        Construct self.zmat from input self.zmat_str
        """
        for line in self.zmat_str.splitlines():
            distance, angle, dihedral = None, None, None





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
    def __init__(self, symbol, indices_list, distance=None, angle=None, dihedral=None):
        self.symbol = symbol
        self.distance = distance
        self.angle = angle
        self.dihedral = dihedral
        self.indices_list = indices_list
        if not len(indices_list):
            raise ZMatrixError('Length of indices_list has to be at least 1')
        if len(indices_list) > 4:
            raise ZMatrixError('Length of indices_list has to be at most 4')
        self.index = indices_list[0]






