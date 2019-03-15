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
    `zmat`                  ``list``     List of zmat atom items
    ====================== ============= ===============================================================================

    """

    def __init__(self, xyz=None, zmat_str=None):
        self.xyz = standardize_xyz_string(xyz) if xyz is not None else None
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

        for index, line in enumerate(self.zmat_str.splitlines()):
            symbol = line.split()[0]
            if index == 0:
                indices_list = [index]
                self.zmat.append(ZMatrixAtom(symbol, indices_list))
            elif index == 1:
                distance_reference_atom_index = line.split()[1]
                indices_list = [index, distance_reference_atom_index]
                distance = line.split()[2]
                self.zmat.append(ZMatrixAtom(symbol, indices_list, distance))
            elif index == 2:
                distance_reference_atom_index = line.split()[1]
                angle_reference_atom_index = line.split()[3]
                indices_list = [index, distance_reference_atom_index, angle_reference_atom_index]
                distance = line.split()[2]
                angle = line.split()[4]
                self.zmat.append(ZMatrixAtom(symbol, indices_list, distance, angle))
            elif index >= 3:
                distance_reference_atom_index = line.split()[1]
                angle_reference_atom_index = line.split()[3]
                dihedral_reference_atom_index = line.split()[5]
                indices_list = [index, distance_reference_atom_index, angle_reference_atom_index,
                                dihedral_reference_atom_index]
                distance = line.split()[2]
                angle = line.split()[4]
                dihedral = line.split()[6]
                self.zmat.append(ZMatrixAtom(symbol, indices_list, distance, angle, dihedral))

    def to_zmat_str(self):
        """
        Convert self.zmat to a Z-matrix list
        :return: zmat_list
        """
        zmat_list_tmp = [obj.to_str() for obj in self.zmat]
        zmat_list = "\n".join(zmat_list_tmp)
        return zmat_list

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

    def to_str(self):
        """
        Convert a ZMatrixAtom to a string
        :return: zmat_atom_str
        """

        indices_list_full = [None]*4
        indices_list_full[:len(self.indices_list)] = self.indices_list
        zmat_atom_list = [self.symbol,
                          indices_list_full[1], self.distance,
                          indices_list_full[2], self.angle,
                          indices_list_full[3], self.dihedral]
        zmat_atom_list_tmp = [entry for entry in zmat_atom_list if entry is not None]
        zmat_atom_str = " ".join(str(entry) for entry in zmat_atom_list_tmp)
        return zmat_atom_str
