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
separate reactants); other topologies, including reactions with more or fewer than two
reactants, are skipped.

Seed distances are set as multiples of the equilibrium single-bond length of the atom pair
involved rather than as absolute lengths, so that the seed is TS-like for any member of the
family (``*3`` is H or a halogen and ``*4`` is always a halogen, so their equilibrium bond
lengths span roughly 1.1-1.9 A). The module constants are:

* ``FORMING_X_FACTOR`` -- multiplier for the forming ``*1``...``*3`` bond.
* ``FORMING_Y_FACTOR`` -- multiplier for the forming ``*2``...``*4`` bond.
* ``BREAKING_XY_FACTOR`` -- multiplier for the breaking ``*3``-``*4`` bond. It is larger than
  one so the X-Y fragment is stretched past its equilibrium length; an unstretched fragment
  yields a van der Waals complex rather than a saddle-point candidate, since bond order is
  created at ``*1`` and ``*2`` without any being released at ``*3``-``*4``.
* ``LOCAL_PLANE_RADIUS`` -- radius in Angstrom around the multiple-bond midpoint within which
  atoms (the bond atoms and their substituents) define the local plane of the multiple bond.

The two forming-bond multipliers differ from each other, which relies on the family's group
definition: ``*4`` is restricted to ``[F1s,Cl1s,Br1s]`` while ``*3`` is ``[H,F1s,Cl1s,Br1s]``,
so RMG labels the hydrogen of an H-X fragment ``*3``. Their values reproduce the concerted
C2H4 + HCl four-centre transition state, in which the forming C...H bond is much further
advanced than the forming C...halogen bond. When both X and Y are halogens RMG enumerates both
``*3``/``*4`` assignments and a seed is built for each.
"""

from typing import TYPE_CHECKING

import numpy as np

from arc.common import get_logger, get_single_bond_length
from arc.species.species import colliding_atoms

if TYPE_CHECKING:
    from arc.reaction import ARCReaction

logger = get_logger()

FORMING_X_FACTOR = 1.31
FORMING_Y_FACTOR = 1.50
BREAKING_XY_FACTOR = 1.42
LOCAL_PLANE_RADIUS = 2.6


def xy_addition(reaction: 'ARCReaction') -> list[dict]:
    """
    Generate 4-center TS-guess seeds for an ``XY_Addition_MultipleBond`` reaction.

    One seed is built per product dictionary that carries the four family labels
    (``*1``, ``*2``, ``*3``, ``*4``) and whose multiple bond and X-Y group sit on
    different reactants (the bimolecular case).

    Reactions with a number of reactants other than two return no seeds: the seed geometry is
    reassembled from exactly two fragments, so any additional reactant would have its symbols
    and coordinates drawn from the X-Y fragment.

    Args:
        reaction (ARCReaction): The reaction. Must have ``product_dicts`` populated
                                (each with an ``r_label_map``) and reactant geometries.

    Returns:
        list[dict]: Seed entries with Cartesian coordinates and the explicit family-label
                    atom mapping used to build each geometry.
    """
    seeds: list[dict] = list()
    reactants, _ = reaction.get_reactants_and_products(return_copies=True)
    if len(reactants) != 2:
        logger.info(f'The XY-addition seed builder only handles bimolecular reactions, '
                    f'got {len(reactants)} reactant(s) for {reaction.label}.')
        return seeds
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
        placed_xy_coords = _build_4_center_geometry(mb_coords, i1, i2, xy_coords, i3, i4,
                                                    mb_symbols=tuple(mb_xyz['symbols']),
                                                    xy_symbols=tuple(xy_xyz['symbols']))
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
        seed_xyz = {'symbols': tuple(symbols),
                    'isotopes': tuple(isotopes),
                    'coords': tuple(coords)}
        if colliding_atoms(seed_xyz):
            # The X-Y fragment is placed rigidly, so it can land on a substituent. Every other seed
            # builder rejects colliding geometries rather than sending them into a TS optimization.
            logger.info(f'Discarding an XY-addition seed for {reaction.label}: it has colliding atoms.')
            continue
        seeds.append({'xyz': seed_xyz,
                      'method': 'Heuristics-XY',
                      'metadata': {
                          'reactive_atoms': {
                              label: r_label_map[label] for label in ('*1', '*2', '*3', '*4')
                          },
                      }})
    return seeds


def _build_4_center_geometry(mb_coords: np.ndarray,
                             i1: int,
                             i2: int,
                             xy_coords: np.ndarray,
                             i3: int,
                             i4: int,
                             mb_symbols: tuple,
                             xy_symbols: tuple,
                             ) -> np.ndarray | None:
    """
    Position the X-Y fragment over the multiple bond in a 4-center arrangement.

    The multiple-bond fragment is kept fixed. The X-Y fragment is stretched onto its breaking-bond
    distance, then translated and rotated so that *3 and *4 sit over *1 and *2 at their forming-bond
    distances, approaching the pi face. All three distances scale with the equilibrium single-bond
    length of the pair involved, so the seed is TS-like for any member of the family rather than only
    for those whose atoms happen to match a fixed constant.

    ``FORMING_X_FACTOR`` is applied to ``*1``...``*3`` and ``FORMING_Y_FACTOR`` to ``*2``...``*4``,
    which assumes the family's label convention that ``*4`` is a halogen while ``*3`` may be the
    hydrogen of an H-X fragment.

    Args:
        mb_coords (np.ndarray): Coordinates of the multiple-bond reactant.
        i1, i2 (int): Local indices of the multiple-bond atoms (``*1``, ``*2``).
        xy_coords (np.ndarray): Coordinates of the X-Y reactant.
        i3, i4 (int): Local indices of the X-Y atoms (``*3`` = X, ``*4`` = Y).
        mb_symbols (tuple): Chemical element symbols of the multiple-bond reactant.
        xy_symbols (tuple): Chemical element symbols of the X-Y reactant.

    Returns:
        np.ndarray | None: The transformed X-Y coordinates, or ``None`` if the multiple bond is
                           degenerate, if the X-Y bond is degenerate, or if the three seed
                           distances cannot close a ring.
    """
    p1, p2 = mb_coords[i1], mb_coords[i2]
    bond = p2 - p1
    bond_length = np.linalg.norm(bond)
    if bond_length < 1e-3:
        return None
    bond_axis = bond / bond_length
    midpoint = (p1 + p2) / 2
    # Approach normal: perpendicular to the local plane of the multiple bond and its substituents,
    # i.e. the pi face. Projecting a vector towards the centroid perpendicular to the bond axis
    # instead leaves the normal *inside* the molecular plane for any planar alkene, which sends the
    # X-Y fragment edge-on into the substituents.
    local = np.array([coord for coord in mb_coords
                      if np.linalg.norm(coord - midpoint) < LOCAL_PLANE_RADIUS])
    normal = np.zeros(3)
    if len(local) >= 3:
        # The smallest right singular vector of the centred local coordinates is the direction of
        # least variance, i.e. the normal of the plane those atoms lie in.
        normal = np.linalg.svd(local - local.mean(axis=0))[2][-1]
        normal = normal - np.dot(normal, bond_axis) * bond_axis
    if np.linalg.norm(normal) < 1e-3:
        # A linear or two-atom fragment has no plane; any direction perpendicular to the bond works.
        normal = np.cross(bond_axis, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(normal) < 1e-3:
            normal = np.cross(bond_axis, np.array([0.0, 1.0, 0.0]))
    normal = normal / np.linalg.norm(normal)
    d_13 = FORMING_X_FACTOR * get_single_bond_length(mb_symbols[i1], xy_symbols[i3])
    d_24 = FORMING_Y_FACTOR * get_single_bond_length(mb_symbols[i2], xy_symbols[i4])
    d_34 = BREAKING_XY_FACTOR * get_single_bond_length(xy_symbols[i3], xy_symbols[i4])
    # Both faces are chemically equivalent; approach the less hindered one.
    others = np.array([coord for i, coord in enumerate(mb_coords) if i not in (i1, i2)])
    if len(others):
        clearances = [min(np.linalg.norm(others - (p1 + d_13 * sign * normal), axis=1))
                      for sign in (1.0, -1.0)]
        if clearances[1] > clearances[0]:
            normal = -normal
    placed = _solve_ring_positions(p1=p1, p2=p2, bond_axis=bond_axis, normal=normal,
                                   d_13=d_13, d_24=d_24, d_34=d_34)
    if placed is None:
        return None
    target_x, target_y = placed
    # Stretch the X-Y fragment onto the breaking-bond distance before placing it, then translate and
    # rotate it rigidly. Members of this family are diatomic, so the stretch is exact for them.
    xy_axis = xy_coords[i4] - xy_coords[i3]
    xy_length = np.linalg.norm(xy_axis)
    if xy_length < 1e-3:
        return None
    stretched = xy_coords + (d_34 - xy_length) * (xy_axis / xy_length) * (
        np.arange(len(xy_coords)) != i3).astype(float)[:, None]
    translated = stretched - stretched[i3] + target_x
    rotation = _rotation_matrix_between(translated[i4] - target_x, target_y - target_x)
    return (rotation @ (translated - target_x).T).T + target_x


def _solve_ring_positions(p1: np.ndarray,
                          p2: np.ndarray,
                          bond_axis: np.ndarray,
                          normal: np.ndarray,
                          d_13: float,
                          d_24: float,
                          d_34: float,
                          ) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Place *3 and *4 in the ring plane so all three seed distances are satisfied simultaneously.

    Works in the 2D frame spanned by the multiple-bond axis and the approach normal, with *1 at the
    origin. *3 sits directly over *1; *4 is the intersection of the circle of radius ``d_24`` about
    *2 with the circle of radius ``d_34`` about *3, taking the solution on the same face as *3.

    Args:
        p1, p2 (np.ndarray): Coordinates of *1 and *2.
        bond_axis (np.ndarray): Unit vector from *1 to *2.
        normal (np.ndarray): Unit approach normal, perpendicular to ``bond_axis``.
        d_13, d_24, d_34 (float): Target *1-*3, *2-*4 and *3-*4 distances.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: Coordinates of *3 and *4, or ``None`` if the three
                                              distances cannot close a ring on this bond.
    """
    length = float(np.linalg.norm(p2 - p1))
    x_uv = np.array([0.0, d_13])
    centre_distance = float(np.linalg.norm(x_uv - np.array([length, 0.0])))
    if centre_distance > d_34 + d_24 or centre_distance < abs(d_34 - d_24) or centre_distance < 1e-6:
        return None
    # Standard two-circle intersection, expressed along and perpendicular to the centre-centre line.
    a = (d_34 ** 2 - d_24 ** 2 + centre_distance ** 2) / (2 * centre_distance)
    h_squared = d_34 ** 2 - a ** 2
    if h_squared < 0:
        return None
    along = (np.array([length, 0.0]) - x_uv) / centre_distance
    perpendicular = np.array([-along[1], along[0]])
    base = x_uv + a * along
    candidates = [base + np.sqrt(h_squared) * perpendicular, base - np.sqrt(h_squared) * perpendicular]
    y_uv = max(candidates, key=lambda candidate: candidate[1])
    to_3d = lambda uv: p1 + uv[0] * bond_axis + uv[1] * normal
    return to_3d(x_uv), to_3d(y_uv)


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
        if cosine > 0:
            return np.eye(3)
        # Antiparallel: rotate by 180 degrees about any axis perpendicular to a. Returning -I here
        # would be an improper transform (det = -1), i.e. the mirror image of the fragment.
        perpendicular = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(a, perpendicular))) > 0.9:
            perpendicular = np.array([0.0, 1.0, 0.0])
        axis_180 = np.cross(a, perpendicular)
        axis_180 = axis_180 / np.linalg.norm(axis_180)
        return 2.0 * np.outer(axis_180, axis_180) - np.eye(3)
    skew = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))
