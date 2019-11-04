#!/usr/bin/env python3
# encoding: utf-8

"""
A module for manipulating vectors
"""

import math
import numpy as np

from rmgpy.molecule.molecule import Molecule

from arc.common import logger
from arc.exceptions import VectorsError
from arc.species import converter


def get_normal(v1, v2):
    """
    Calculate a normal vector using cross multiplication.

    Args:
         v1 (list): Vector 1.
         v2 (list): Vector 2.

    Returns:
        list: A normal unit vector to v1 and v2.
    """
    normal = [v1[1] * v2[2] - v2[1] * v1[2], - v1[0] * v2[2] + v2[0] * v1[2], v1[0] * v2[1] - v2[0] * v1[1]]
    return unit_vector(normal)


def get_angle(v1, v2, units='rads'):
    """
    Calculate the angle in radians between two vectors.

    Args:
         v1 (list): Vector 1.
         v2 (list): Vector 2.
         units (str): The desired units, either 'rads' for radians, or 'degs' for degrees.

    Returns:
        float: The angle in radians between ``v1`` and ``v2`` in the desired units.

    Raises:
        VectorsError: If ``v1`` and ``v2`` are of different lengths.
    """
    if len(v1) != len(v2):
        raise VectorsError(f'v1 and v2 must be the same length, got {len(v1)} and {len(v2)}.')
    v1_u, v2_u = unit_vector(v1), unit_vector(v2)
    conversion = 180 / math.pi if 'degs' in units else 1
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * conversion


def unit_vector(vector):
    """
    Calculate a unit vector in the same direction as the input vector.

    Args:
        vector (list): The input vector.

    Returns:
        list: The unit vector.
    """
    length = get_vector_length(vector)
    return [vi / length for vi in vector]


def set_vector_length(vector, length):
    """
    Set the length of a 3D vector.

    Args:
        vector (list): The vector to process.
        length (float): The desired length to set.

    Returns:
        list: A vector with the desired length.
    """
    u = unit_vector(vector)
    return [u[0] * length, u[1] * length, u[2] * length]


def rotate_vector(point_a, point_b, normal, theta):
    """
    Rotate a vector in 3D space around a given axis by a certain angle.

    Inspired by https://stackoverflow.com/questions/6802577/rotation-of-3d-vector

    Args:
        point_a (list): The 3D coordinates of the starting point (point A) of the vector to be rotated.
        point_b (list): The 3D coordinates of the ending point (point B) of the vector to be rotated.
        normal (list): The axis to be rotated around.
        theta (float): The degree in radians by which to rotate.

    Returns:
        list: The rotated vector (the new coordinates for point B).
    """
    normal = np.asarray(normal)
    normal = normal / math.sqrt(np.dot(normal, normal))  # should *not* be replaced by an augmented assignment
    a = math.cos(theta / 2.0)
    b, c, d = -normal * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    vector = [b - a for b, a in zip(point_b, point_a)]
    new_vector = np.dot(rotation_matrix, vector).tolist()
    new_vector = [n_v + a for n_v, a in zip(new_vector, point_a)]  # root the vector at the original starting point
    return new_vector


def get_vector(pivot, anchor, xyz):
    """
    Get a vector between two atoms in the molecule (pointing from pivot to anchor).

    Args:
        pivot (int): The 0-index of the pivotal atom around which groups are to be translated.
        anchor (int): The 0-index of an additional atom in the molecule.
        xyz (dict): The 3D coordinates of the molecule with the same atom order as in mol.

    Returns:
         list: A vector pointing from the pivotal atom towards the anchor atom.
    """
    x, y, z = converter.xyz_to_x_y_z(xyz)
    dx = x[anchor] - x[pivot]
    dy = y[anchor] - y[pivot]
    dz = z[anchor] - z[pivot]
    return [dx, dy, dz]


def get_lp_vector(label, mol, xyz, pivot):
    """
    Get a vector from the pivotal atom in the molecule towards its lone electron pair (lp).
    The approach is to reverse the average of the three unit vectors between the pivotal atom and its neighbors.

    Args:
        label (str): The species' label.
        mol (Molecule): The 2D graph representation of the molecule.
        xyz (dict): The 3D coordinates of the molecule with the same atom order as in mol.
        pivot (int): The 0-index of the pivotal atom of interest.

    Returns:
        list: A unit vector pointing from the pivotal (nitrogen) atom towards its lone electron pairs orbital.

    Raises:
        VectorsError: If the lp vector cannot be attained.
    """
    neighbors, vectors = list(), list()
    if not mol.atoms[pivot].is_nitrogen():
        logger.warning(f'The get_lp_vector specializes in nitrogen atoms, got {mol.atoms[pivot].symbol} '
                       f'(atom number {pivot}) in species {label}.')
    for atom in mol.atoms[pivot].edges.keys():
        neighbors.append(mol.atoms.index(atom))
    if len(neighbors) < 3:
        # N will have 3, S may have more.
        raise VectorsError(f'Can only get lp vector if the pivotal atom has at least three neighbors. '
                           f'Atom {mol.atoms[pivot]} in {label} has only {len(neighbors)}.')
    for neighbor in neighbors:
        # already taken in "reverse", pointing towards the lone pair
        vectors.append(unit_vector(get_vector(pivot=neighbor, anchor=pivot, xyz=xyz)))
    x = sum(v[0] for v in vectors)
    y = sum(v[1] for v in vectors)
    z = sum(v[2] for v in vectors)
    return unit_vector([x, y, z])


def get_vector_length(v):
    """
    Get the length of an ND vector

    Args:
        v (list): The vector.

    Returns:
        float: The vector's length.
    """
    return np.dot(v, v) ** 0.5
