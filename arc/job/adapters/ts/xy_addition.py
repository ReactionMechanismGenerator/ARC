"""
TS-guess seed builder for the ``XY_Addition_MultipleBond`` reaction family.

The family adds an X-Y bond across a multiple bond. Its RMG recipe is::

    ['BREAK_BOND',  '*3', 1,  '*4'],   # break the X-Y bond
    ['CHANGE_BOND', '*1', -1, '*2'],   # reduce the multiple bond
    ['FORM_BOND',   '*1', 1,  '*3'],   # form *1-X
    ['FORM_BOND',   '*2', 1,  '*4'],   # form *2-Y

so the transition state is a 4-center arrangement: X (``*3``) approaching one end of the
multiple bond (``*1``) and Y (``*4``) approaching the other (``*2``), while the X-Y bond
(``*3``-``*4``) breaks. This module builds that 4-center geometry from the reactant
geometries and the family's atom labels, to seed a downstream TS search (e.g. CREST
refinement followed by a saddle-point optimization).

The builder handles the bimolecular case (the multiple bond and the X-Y group are on
separate reactants); other topologies are skipped.
"""

from typing import TYPE_CHECKING

import numpy as np

from arc.common import get_logger

if TYPE_CHECKING:
    from arc.reaction import ARCReaction

logger = get_logger()

# Target forming-bond lengths for the seed (Angstrom). These are deliberately TS-like
# starting values; the seed is refined downstream, so they need only be reasonable.
FORMING_X_DISTANCE = 1.6   # *1 (multiple-bond atom) ... *3 (X)
FORMING_Y_DISTANCE = 2.2   # *2 (multiple-bond atom) ... *4 (Y)


def xy_addition(reaction: 'ARCReaction') -> list[dict]:
    """
    Generate 4-center TS-guess seeds for an ``XY_Addition_MultipleBond`` reaction.

    One seed is built per product dictionary that carries the four family labels
    (``*1``, ``*2``, ``*3``, ``*4``) and whose multiple bond and X-Y group sit on
    different reactants (the bimolecular case).

    Args:
        reaction (ARCReaction): The reaction. Must have ``product_dicts`` populated
                                (each with an ``r_label_map``) and reactant geometries.

    Returns:
        list[dict]: Seed entries, each ``{'xyz': <xyz dict>, 'method': 'Heuristics-XY'}``.
    """
    seeds: list[dict] = list()
    reactants, _ = reaction.get_reactants_and_products(return_copies=True)
    lengths = [spc.number_of_atoms for spc in reactants]
    offsets = [sum(lengths[:i]) for i in range(len(reactants))]
    total_atoms = sum(lengths)

    def which_reactant(global_index: int) -> tuple[int | None, int | None]:
        """Map a global reactant atom index to (reactant index, local atom index)."""
        for reactant_index, offset in enumerate(offsets):
            if offset <= global_index < offset + lengths[reactant_index]:
                return reactant_index, global_index - offset
        return None, None

    for product_dict in reaction.product_dicts:
        r_label_map = product_dict.get('r_label_map', dict())
        if not all(label in r_label_map for label in ('*1', '*2', '*3', '*4')):
            continue
        (r_mb, i1), (r_mb2, i2) = which_reactant(r_label_map['*1']), which_reactant(r_label_map['*2'])
        (r_xy, i3), (r_xy2, i4) = which_reactant(r_label_map['*3']), which_reactant(r_label_map['*4'])
        # The multiple bond (*1, *2) must share one reactant and the X-Y group (*3, *4)
        # the other. Only the bimolecular case is handled here.
        if r_mb is None or r_mb != r_mb2 or r_xy != r_xy2 or r_mb == r_xy:
            continue
        mb_xyz, xy_xyz = reactants[r_mb].get_xyz(), reactants[r_xy].get_xyz()
        mb_coords = np.array(mb_xyz['coords'], dtype=float)
        xy_coords = np.array(xy_xyz['coords'], dtype=float)
        placed_xy_coords = _build_4_center_geometry(mb_coords, i1, i2, xy_coords, i3, i4)
        if placed_xy_coords is None:
            continue
        # Reassemble a single geometry in the reaction's reactant (global) atom order.
        symbols, isotopes, coords = list(), list(), list()
        for global_index in range(total_atoms):
            reactant_index, local_index = which_reactant(global_index)
            source_xyz = mb_xyz if reactant_index == r_mb else xy_xyz
            source_coords = mb_coords if reactant_index == r_mb else placed_xy_coords
            symbols.append(source_xyz['symbols'][local_index])
            isotopes.append(source_xyz['isotopes'][local_index])
            coords.append(tuple(float(v) for v in source_coords[local_index]))
        seeds.append({'xyz': {'symbols': tuple(symbols),
                              'isotopes': tuple(isotopes),
                              'coords': tuple(coords)},
                      'method': 'Heuristics-XY'})
    return seeds


def _build_4_center_geometry(mb_coords: np.ndarray,
                             i1: int,
                             i2: int,
                             xy_coords: np.ndarray,
                             i3: int,
                             i4: int,
                             ) -> np.ndarray | None:
    """
    Position the X-Y fragment over the multiple bond in a 4-center arrangement.

    The multiple-bond fragment is kept fixed; the X-Y fragment is rigidly translated and
    rotated so that X (``*3``) sits over ``*1`` at :data:`FORMING_X_DISTANCE` and Y
    (``*4``) points toward ``*2`` at :data:`FORMING_Y_DISTANCE`, approaching the open
    face of the multiple bond.

    Args:
        mb_coords (np.ndarray): Coordinates of the multiple-bond reactant.
        i1, i2 (int): Local indices of the multiple-bond atoms (``*1``, ``*2``).
        xy_coords (np.ndarray): Coordinates of the X-Y reactant.
        i3, i4 (int): Local indices of the X-Y atoms (``*3`` = X, ``*4`` = Y).

    Returns:
        np.ndarray | None: The transformed X-Y coordinates, or ``None`` if the multiple
                           bond is degenerate.
    """
    p1, p2 = mb_coords[i1], mb_coords[i2]
    bond = p2 - p1
    bond_length = np.linalg.norm(bond)
    if bond_length < 1e-3:
        return None
    bond_axis = bond / bond_length
    # Approach normal: perpendicular to the *1-*2 axis, pointing away from the rest of the
    # multiple-bond fragment (its open face).
    midpoint = (p1 + p2) / 2
    to_centroid = mb_coords.mean(axis=0) - midpoint
    normal = to_centroid - np.dot(to_centroid, bond_axis) * bond_axis
    if np.linalg.norm(normal) < 1e-3:
        normal = np.cross(bond_axis, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(normal) < 1e-3:
            normal = np.cross(bond_axis, np.array([0.0, 1.0, 0.0]))
    normal = -normal / np.linalg.norm(normal)
    target_x = p1 + FORMING_X_DISTANCE * normal
    target_y = p2 + FORMING_Y_DISTANCE * normal
    # Translate the X-Y fragment so X lands on its target, then rotate so the X->Y vector
    # points toward the Y target. This preserves the fragment's internal geometry.
    translated = xy_coords - xy_coords[i3] + target_x
    rotation = _rotation_matrix_between(translated[i4] - target_x, target_y - target_x)
    return (rotation @ (translated - target_x).T).T + target_x


def _rotation_matrix_between(vector_from: np.ndarray, vector_to: np.ndarray) -> np.ndarray:
    """
    Return the rotation matrix that aligns ``vector_from`` onto ``vector_to`` (Rodrigues).

    Args:
        vector_from (np.ndarray): The source vector.
        vector_to (np.ndarray): The target vector.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    a = vector_from / np.linalg.norm(vector_from)
    b = vector_to / np.linalg.norm(vector_to)
    axis = np.cross(a, b)
    sine = np.linalg.norm(axis)
    cosine = float(np.dot(a, b))
    if sine < 1e-8:
        # Parallel (identity) or antiparallel (180-degree flip).
        return np.eye(3) if cosine > 0 else -np.eye(3)
    skew = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))
