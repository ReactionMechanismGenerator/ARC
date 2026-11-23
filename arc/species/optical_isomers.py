"""
A module for determining the number of optical isomers of a well or a TS.

The rule: n = 2 for a chiral structure, except n = 1 when the treated (converged, used) rotors reach its mirror image.
A ring twist is never a rotor, and a stereocenter always gives 2.

A rotor integral over 0..2*pi already contains the mirror-image well along its torsion, so only mirror wells that no
treated rotor reaches are counted by n. Reach is decided geometrically: the treated tops are rotated to every
candidate angle that maps a graph-equivalent atom pair onto its mirror position, the result is aligned onto the mirror
image with atom permutation, and D, the largest signed-torsion deviation between the two, is compared with
``MIRROR_TORSION_TOLERANCE``. A geometry with D <= tolerance and no rotation is achiral.
"""

import itertools
import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from arc.common import get_logger
from arc.mapping.cluster import ComplexGraph, core_automorphisms
from arc.species.conformers import (get_graph_stereocenters, get_neighbors, get_reachable, get_ts_bonds,
                                    is_ring_twist, is_ts_graph_out_of_order)
from arc.species.species import USER_OPTICAL_ISOMERS_CAUSE
from arc.species.vectors import (apply_rodrigues_rotation, calculate_angle, calculate_dihedral_angles, get_normal,
                                 unit_vector)

if TYPE_CHECKING:
    from arc.reaction import ARCReaction
    from arc.species.species import ARCSpecies

logger = get_logger()

# The largest signed-torsion deviation (degrees) between a structure and its aligned mirror image for the two to count
# as the same well.
MIRROR_TORSION_TOLERANCE = 25.0

# Torsions through a bond angle above this (degrees) are not compared.
LINEAR_ANGLE_THRESHOLD = 165.0

# The smallest sum of the three bond angles (degrees) at a planar three-coordinate centre, and the margin (degrees)
# around it within which a centre is reported as near-planar.
PLANAR_ANGLE_SUM = 350.0
NEAR_PLANAR_MARGIN = 5.0

# The largest bond-length difference (Angstrom) between the bonds a symmetry map pairs.
BOND_LENGTH_TOLERANCE = 0.05

# The largest number of graph-symmetry matches enumerated when aligning a structure onto its mirror image.
MAX_SYMMETRY_MATCHES = 10000

# The largest number of torsion comparisons (one per candidate structure and map) a mirror-reach search may make.
MAX_SEARCH_EVALUATIONS = 50000

OPTICAL_ISOMER_CAUSES = {
    'user': USER_OPTICAL_ISOMERS_CAUSE,
    'stereocentre': 'stereocentre at {atoms}',
    'achiral': 'achiral optimised geometry (D = {deviation:.1f} deg)',
    'reached': 'mirror well reached by the {rotors} rotor{plural} (D = {deviation:.1f} deg)',
    'not_reached': '{parts}, mirror well counted by n (D = {deviation:.1f} deg)',
    'undetermined': 'not determined: {reason}',
}

NEAR_PLANAR_NOTE = 'near-planar {atom} (angle sum {angle_sum:.1f} deg)'

OPTICAL_ISOMER_CAUSE_PARTS = {
    'harmonic': '{bond} torsion harmonic: {why}',
    'ring_twist': 'ring twist about {bonds}',
    'configuration': 'configuration at {atoms}',
    'torsion': 'no treated rotor reaches the {bond} torsion',
    'unnamed': 'no treated rotor reaches it (the cause could not be named: {error})',
}


class MirrorSearchError(Exception):
    """The mirror-reach search exceeded its budget"""


class OpticalIsomers(NamedTuple):
    """
    The number of optical isomers of a well or a TS, and why.

    Attributes:
        value (int | None): The number of optical isomers, ``None`` if it could not be determined.
        kind (str): The key of the cause in ``OPTICAL_ISOMER_CAUSES``.
        cause (str): The cause in words.
    """
    value: int | None
    kind: str
    cause: str


def format_cause(kind: str, **fields) -> str:
    """
    Format the cause of an optical-isomer decision.

    Args:
        kind (str): A key of ``OPTICAL_ISOMER_CAUSES``.
        **fields: The template fields.

    Returns:
        str: The cause in words.
    """
    return ' '.join(OPTICAL_ISOMER_CAUSES[kind].format(**fields).split())


def _result(value: int | None, kind: str, **fields) -> OpticalIsomers:
    """
    Build an ``OpticalIsomers`` result whose cause is its template formatted with ``fields``.

    Args:
        value (int | None): The number of optical isomers, ``None`` if it could not be determined.
        kind (str): A key of ``OPTICAL_ISOMER_CAUSES``.
        **fields: The template fields.

    Returns:
        OpticalIsomers: The result.
    """
    return OpticalIsomers(value=value, kind=kind, cause=format_cause(kind, **fields))


def determine_optical_isomers(species: 'ARCSpecies',
                              rotors: list[dict],
                              reaction: 'ARCReaction | None' = None,
                              untreated: list[tuple[tuple[int, int, int, int], str]] | None = None,
                              max_matches: int = MAX_SYMMETRY_MATCHES,
                              ) -> OpticalIsomers:
    """
    Determine the number of optical isomers of a well or a TS, given the rotors its Arkane file treats.
    Any exception raised while deciding gives an undetermined result.

    Args:
        species (ARCSpecies): The well or TS.
        rotors (list[dict]): The treated rotors (``rotors_dict`` entries written into the Arkane file).
        reaction (ARCReaction, optional): The reaction whose TS is ``species``, used for the TS graph.
        untreated (list[tuple], optional): The 0-indexed torsions a rotor could have treated and did not, each with
                                           why, used only to name the cause.
        max_matches (int, optional): The largest number of graph-symmetry matches to enumerate.

    Returns:
        OpticalIsomers: The number of optical isomers and its cause.
    """
    user_value = species.optical_isomers if species.has_user_optical_isomers() else None
    try:
        result = _determine(species, rotors, reaction, untreated or list(), max_matches)
    except Exception as e:
        result = _result(None, 'undetermined', reason=f'{type(e).__name__}: {e}')
    if user_value is not None:
        if isinstance(user_value, int) and not isinstance(user_value, bool) and user_value > 0:
            if result.value is not None and result.value != user_value:
                logger.warning(f'Species {species.label} was given optical_isomers = {user_value}, while ARC '
                               f'determined {result.value} ({result.cause}). Using {user_value}.')
            return _result(user_value, 'user')
        logger.warning(f'Ignoring the invalid optical_isomers value {user_value!r} of species {species.label}.')
    return result


def _determine(species: 'ARCSpecies',
               rotors: list[dict],
               reaction: 'ARCReaction | None',
               untreated: list[tuple[tuple[int, int, int, int], str]],
               max_matches: int,
               ) -> OpticalIsomers:
    """
    Apply the optical-isomer rule, without the user value and the exception guard of ``determine_optical_isomers``.
    Each three-coordinate centre whose bond angles sum to within ``NEAR_PLANAR_MARGIN`` of ``PLANAR_ANGLE_SUM`` is
    logged and appended to the cause with its angle sum.

    Args:
        species (ARCSpecies): The well or TS.
        rotors (list[dict]): The treated rotors.
        reaction (ARCReaction | None): The reaction whose TS is ``species``.
        untreated (list[tuple]): The 0-indexed untreated torsions, each with why.
        max_matches (int): The largest number of graph-symmetry matches to enumerate.

    Returns:
        OpticalIsomers: The number of optical isomers and its cause.
    """
    xyz = species.get_xyz(generate=False)
    if xyz is None:
        return _result(None, 'undetermined', reason='no optimised geometry')
    symbols = list(xyz['symbols'])
    if is_ts_graph_out_of_order(mol=species.mol, xyz=xyz):
        return _result(None, 'undetermined', reason='the atom order of the 2D graph and the geometry differ')
    if species.is_ts:
        bonds = get_ts_bonds(mol=species.mol, xyz=xyz, reaction=reaction)[0]
    elif species.mol is None:
        return _result(None, 'undetermined', reason='no 2D graph')
    else:
        index = {atom: i for i, atom in enumerate(species.mol.atoms)}
        bonds = {tuple(sorted((index[a], index[b]))) for a in species.mol.atoms for b in a.edges}
    angle_sums = get_angle_sums([list(coords) for coords in xyz['coords']], get_neighbors(bonds, len(symbols)))
    result = _decide(species, symbols, xyz, bonds, angle_sums, rotors, untreated, max_matches)
    near_planar = [NEAR_PLANAR_NOTE.format(atom=atom_label(symbols, i), angle_sum=angle_sum)
                   for i, angle_sum in sorted(angle_sums.items())
                   if abs(angle_sum - PLANAR_ANGLE_SUM) <= NEAR_PLANAR_MARGIN]
    if near_planar:
        logger.info(f'Optical isomers of {species.label}: {", ".join(near_planar)}, planar from an angle sum of '
                    f'{PLANAR_ANGLE_SUM:.0f} deg')
        result = result._replace(cause=f'{result.cause}; {"; ".join(near_planar)}')
    return result


def _decide(species: 'ARCSpecies',
            symbols: list[str],
            xyz: dict,
            bonds: set[tuple[int, int]],
            angle_sums: dict[int, float],
            rotors: list[dict],
            untreated: list[tuple[tuple[int, int, int, int], str]],
            max_matches: int,
            ) -> OpticalIsomers:
    """
    Decide the number of optical isomers of a well or a TS whose geometry and graph bonds are known.
    An N stereocentre of a well whose bond angles sum to at least ``PLANAR_ANGLE_SUM`` is planar, not a stereocentre.

    Args:
        species (ARCSpecies): The well or TS.
        symbols (list[str]): The element symbols.
        xyz (dict): The geometry.
        bonds (set[tuple[int, int]]): The graph bonds (the TS graph for a TS).
        angle_sums (dict[int, float]): The bond angle sum of each three-coordinate atom.
        rotors (list[dict]): The treated rotors.
        untreated (list[tuple]): The 0-indexed untreated torsions, each with why.
        max_matches (int): The largest number of graph-symmetry matches to enumerate.

    Returns:
        OpticalIsomers: The number of optical isomers and its cause.
    """
    if not species.is_ts:
        chiral, centers, reason = get_graph_stereocenters(species.label, species.mol, xyz)
        if chiral is None:
            return _result(None, 'undetermined', reason=reason)
        planar_n = [i for i in centers if symbols[i] == 'N' and angle_sums.get(i, 0.0) >= PLANAR_ANGLE_SUM]
        if chiral and (not centers or len(planar_n) < len(centers)):
            return _result(2, 'stereocentre', atoms=', '.join(atom_label(symbols, i) for i in centers
                                                              if i not in planar_n) or 'an axis')
    if len(symbols) < 3:
        return _result(1, 'achiral', deviation=0.0)
    if not is_connected(bonds, len(symbols)):
        return _result(None, 'undetermined', reason='the bond graph is not connected, so its fragments\' relative '
                                                    'orientation cannot be compared with the mirror image')
    reach = MirrorReach(xyz, bonds, max_matches)
    if reach.too_many_matches:
        return _result(None, 'undetermined', reason=f'more than {max_matches} graph symmetry matches')
    if not reach.maps:
        return _result(None, 'undetermined', reason='no graph symmetry map')
    deviation, _ = reach.best([])
    if deviation <= MIRROR_TORSION_TOLERANCE:
        return _result(1, 'achiral', deviation=deviation)
    torsions = [t for t in get_rotor_torsions(rotors) if not is_ring_twist(bonds, (t[1], t[2]))]
    deviation, worst = reach.best(torsions)
    if deviation <= MIRROR_TORSION_TOLERANCE:
        try:
            needed = reach.needed(torsions)
        except MirrorSearchError:
            needed = torsions
        return _result(1, 'reached', deviation=deviation,
                       rotors=', '.join(bond_label(symbols, t[1], t[2]) for t in needed),
                       plural='s' if len(needed) > 1 else '')
    try:
        parts = explain_unreached(symbols, bonds, reach, torsions, worst, untreated)
    except Exception as e:
        parts = [OPTICAL_ISOMER_CAUSE_PARTS['unnamed'].format(error=f'{type(e).__name__}: {e}')]
    return _result(2, 'not_reached', deviation=deviation, parts='; '.join(parts))


def explain_unreached(symbols: list[str],
                      bonds: set[tuple[int, int]],
                      reach: 'MirrorReach',
                      torsions: list[tuple[int, int, int, int]],
                      worst: dict,
                      untreated: list[tuple[tuple[int, int, int, int], str]],
                      ) -> list[str]:
    """
    Name what keeps a structure from reaching its mirror image: a twist about a bond in a ring of four or more
    atoms, the untreated torsions whose rotors would reach it, or else the configuration at the atoms whose improper
    torsions deviate, or the bond whose torsion deviates most.

    Args:
        symbols (list[str]): The element symbols.
        bonds (set[tuple[int, int]]): The graph bonds.
        reach (MirrorReach): The mirror-reach search of the structure.
        torsions (list[tuple]): The treated rotor torsions.
        worst (dict): The torsion deviations of the best search with the treated rotors.
        untreated (list[tuple]): The 0-indexed untreated torsions, each with why.

    Returns:
        list[str]: The causes in words.
    """
    parts = list()
    ring_bonds = sorted({tuple(sorted((t[1], t[2]))) for t, dev in worst.items() if dev > MIRROR_TORSION_TOLERANCE
                         and t not in reach.impropers and tuple(sorted((t[1], t[2]))) in bonds
                         and is_ring_twist(bonds, (t[1], t[2]))
                         and not reach.neighbors[t[1]] & reach.neighbors[t[2]]})
    if ring_bonds:
        parts.append(OPTICAL_ISOMER_CAUSE_PARTS['ring_twist'].format(
            bonds=', '.join(bond_label(symbols, *bond) for bond in ring_bonds)))
    treated_bonds = {tuple(sorted((t[1], t[2]))) for t in torsions}
    untreated = [(t, why) for t, why in untreated
                 if tuple(sorted((t[1], t[2]))) not in treated_bonds
                 and not is_ring_twist(bonds, (t[1], t[2]))]
    if not ring_bonds and untreated:
        all_torsions = torsions + [t for t, _ in untreated]
        if reach.best(all_torsions)[0] <= MIRROR_TORSION_TOLERANCE:
            needed = reach.needed(all_torsions, keep=torsions)
            for t, why in untreated:
                if t in needed:
                    parts.append(OPTICAL_ISOMER_CAUSE_PARTS['harmonic'].format(bond=bond_label(symbols, t[1], t[2]),
                                                                               why=why))
            return parts
    if not ring_bonds:
        centers = sorted({t[1] for t, dev in worst.items() if dev > MIRROR_TORSION_TOLERANCE and t in reach.impropers})
        if centers:
            parts.append(OPTICAL_ISOMER_CAUSE_PARTS['configuration'].format(
                atoms=', '.join(atom_label(symbols, i) for i in centers)))
        else:
            t = max(worst, key=worst.get)
            parts.append(OPTICAL_ISOMER_CAUSE_PARTS['torsion'].format(bond=bond_label(symbols, t[1], t[2])))
    return parts


def get_rotor_torsions(rotors: list[dict]) -> list[tuple[int, int, int, int]]:
    """
    Get the 0-indexed torsions of rotors, one per dimension.
    A rotor without a ``torsion``, as in old restart files, falls back to its 1-indexed ``scan``.

    Args:
        rotors (list[dict]): ``rotors_dict`` entries.

    Returns:
        list[tuple[int, int, int, int]]: The torsions.
    """
    torsions = list()
    for rotor in rotors:
        torsion, offset = rotor.get('torsion'), 0
        if not torsion:
            torsion, offset = rotor.get('scan'), 1
        if not torsion:
            continue
        for t in (torsion if isinstance(torsion[0], (list, tuple)) else [torsion]):
            torsions.append(tuple(int(i) - offset for i in t))
    return torsions


class MirrorReach(object):
    """
    Search for rotor angles that turn a structure into its mirror image.

    A three-coordinate centre whose bond angles sum to at least ``PLANAR_ANGLE_SUM`` is planar: it is moved into the
    plane of its three neighbors before the search, so its improper torsions match those of the mirror image.

    Args:
        xyz (dict): The coordinates.
        bonds (set[tuple[int, int]]): The graph bonds, 0-indexed.
        max_matches (int): The largest number of graph-symmetry matches to enumerate.
    """

    def __init__(self, xyz: dict, bonds: set[tuple[int, int]], max_matches: int):
        self.symbols = list(xyz['symbols'])
        self.n = len(self.symbols)
        self.bonds = {tuple(sorted(bond)) for bond in bonds}
        self.neighbors = get_neighbors(self.bonds, self.n)
        self.coords = np.array(xyz['coords'], dtype=float)
        self.planar = {i for i, angle_sum in get_angle_sums(self.coords.tolist(), self.neighbors).items()
                       if angle_sum >= PLANAR_ANGLE_SUM}
        self.coords = flatten_centers(self.coords, self.neighbors, self.planar)
        coords = self.coords.tolist()
        self.mirror = self.coords * np.array([-1.0, 1.0, 1.0])
        self.max_matches = max_matches
        self.evaluations = 0
        self.isotopes = [int(i) for i in xyz['isotopes']] if xyz.get('isotopes') else [0] * self.n
        self.linear = get_linear_atoms(coords, self.neighbors)
        propers, impropers = get_all_torsions(self.neighbors, self.bonds)
        propers += get_axis_torsions(self.neighbors, self.linear)
        self.impropers = set(impropers)
        self.torsions = [t for t in propers + impropers
                         if max(calculate_angle(coords, list(t[:3])), calculate_angle(coords, list(t[1:])))
                         <= LINEAR_ANGLE_THRESHOLD]
        self.terminal_h = {i: next(iter(self.neighbors[i])) for i in range(self.n)
                           if self.symbols[i] == 'H' and len(self.neighbors[i]) == 1}
        self.skeleton = [i for i in range(self.n) if i not in self.terminal_h]
        self.hydrogens = {i: sorted(h for h, parent in self.terminal_h.items() if parent == i) for i in self.skeleton}
        self.maps, self.too_many_matches = get_symmetry_maps(self.symbols, self.isotopes, self.bonds, self.skeleton,
                                                             self.hydrogens, self.coords, max_matches)

    def candidates(self, torsion: tuple[int, int, int, int], symmetry_map: dict[int, int]) -> list[float]:
        """
        Get the candidate values of a rotor dihedral that put it at its mirror-image value under a symmetry map.
        The map may send the rotor bond onto another, equivalent bond (e.g., one methyl of a gem-dimethyl group onto
        the other). A terminal H atom of the torsion may go to any terminal H atom of its pivot's image.

        Args:
            torsion (tuple[int, int, int, int]): The rotor torsion a-b-c-d.
            symmetry_map (dict[int, int]): Maps skeleton (or all) atoms onto their mirror-image atoms.

        Returns:
            list[float]: The candidate dihedral values in degrees.
        """
        a, b, c, d = torsion
        b2, c2 = symmetry_map[b], symmetry_map[c]
        fixed = [symmetry_map[a]] if a in symmetry_map \
            else [h for h in self.hydrogens[b2] if self.isotopes[h] == self.isotopes[a]]
        moving = [symmetry_map[d]] if d in symmetry_map \
            else [h for h in self.hydrogens[c2] if self.isotopes[h] == self.isotopes[d]]
        coords = self.coords.tolist()
        values = {-get_signed_dihedral(coords, (a2, b2, c2, d2)) for a2 in fixed for d2 in moving}
        return sorted({round(value, 6) for value in values if not math.isnan(value)})

    def defined_torsion(self, torsion: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """
        Get a defined torsion that turns with a rotor: a rotor torsion whose pivot is a linear chain atom (e.g., an
        sp C of a C#C bond) is extended along the chain to its first non-linear atom on that side, and an off-chain
        neighbor of that atom replaces the end atom.

        Args:
            torsion (tuple[int, int, int, int]): The rotor torsion a-b-c-d.

        Returns:
            tuple[int, int, int, int]: The torsion, or its extension across the linear chains at its pivots
                                       (the torsion itself if a chain ends at an atom with no off-chain neighbor).
        """
        a, b, c, d = torsion
        if b in self.linear:
            b, previous = walk_linear_chain(self.neighbors, self.linear, b, c)
            if not self.neighbors[b] - {previous}:
                return torsion
            a = min(self.neighbors[b] - {previous})
        if c in self.linear:
            c, previous = walk_linear_chain(self.neighbors, self.linear, c, torsion[1])
            if not self.neighbors[c] - {previous}:
                return torsion
            d = min(self.neighbors[c] - {previous})
        return a, b, c, d

    def atom_map(self, coords: np.ndarray, symmetry_map: dict[int, int]) -> dict[int, int]:
        """
        Complete a skeleton symmetry map with the terminal H atoms: those of each skeleton atom go to those of its
        image, by distance after a best fit of the skeleton onto the mirror image.

        Args:
            coords (np.ndarray): The structure's coordinates.
            symmetry_map (dict[int, int]): Maps skeleton (or all) atoms onto their mirror-image atoms.

        Returns:
            dict[int, int]: Maps every atom onto a mirror-image atom.
        """
        if len(symmetry_map) == self.n:
            return symmetry_map
        fitted = fit_onto(coords, self.mirror, self.skeleton, [symmetry_map[i] for i in self.skeleton])
        atom_map = dict(symmetry_map)
        for atom, hydrogens in self.hydrogens.items():
            if hydrogens:
                targets = self.hydrogens[symmetry_map[atom]]
                distances = np.linalg.norm(fitted[hydrogens][:, None, :] - self.mirror[targets][None, :, :], axis=2)
                distances[np.array([self.isotopes[h] for h in hydrogens])[:, None]
                          != np.array([self.isotopes[t] for t in targets])[None, :]] = np.inf
                order = min(itertools.permutations(range(len(targets))),
                            key=lambda o: sum(distances[k, j] for k, j in enumerate(o)))
                atom_map.update({h: targets[j] for h, j in zip(hydrogens, order)})
        return atom_map

    def compare(self, coords: np.ndarray, symmetry_map: dict[int, int]) -> tuple[float, dict]:
        """
        Measure D between a structure and the mirror image under a symmetry map, within the search budget.

        Args:
            coords (np.ndarray): The structure's coordinates.
            symmetry_map (dict[int, int]): Maps skeleton (or all) atoms onto their mirror-image atoms.

        Raises:
            MirrorSearchError: If the search made more than ``MAX_SEARCH_EVALUATIONS`` comparisons.

        Returns:
            tuple[float, dict]: D in degrees, and the deviation of every compared torsion.
        """
        self.evaluations += 1
        if self.evaluations > MAX_SEARCH_EVALUATIONS:
            raise MirrorSearchError(f'the mirror search exceeded {MAX_SEARCH_EVALUATIONS} comparisons')
        return get_mirror_torsion_deviation(coords, self.mirror, self.atom_map(coords, symmetry_map), self.torsions)

    def best(self, torsions: list[tuple[int, int, int, int]]) -> tuple[float, dict]:
        """
        Find the rotor dihedrals that bring the structure closest to its mirror image.
        For each symmetry map, every rotor is set to its mirror-image value under that map. A rotor alone sets the
        torsions about its own bond, so each one is chosen among its candidates by the deviation of those torsions,
        in two passes (the second with every other rotor already set, so the fit that assigns H atoms is good).

        Args:
            torsions (list[tuple]): The rotor torsions.

        Raises:
            MirrorSearchError: If there is no symmetry map, or if the search runs out of comparisons before any map
                               brings D within ``MIRROR_TORSION_TOLERANCE``.

        Returns:
            tuple[float, dict]: The smallest D in degrees, and the torsion deviations at it, or, if the comparisons
                                ran out after a map brought D within the tolerance, the smallest D found so far.
        """
        if not self.maps:
            raise MirrorSearchError('there is no graph symmetry map to search')
        defined = {torsion: self.defined_torsion(torsion) for torsion in torsions}
        best = None
        for symmetry_map in self.maps:
            try:
                coords = self.coords
                for _ in range(2 if torsions else 0):
                    for torsion in torsions:
                        bond = {defined[torsion][1], defined[torsion][2]}
                        trials = list()
                        for value in self.candidates(defined[torsion], symmetry_map):
                            trial = rotate_top(coords, self.neighbors, torsion, value, dihedral=defined[torsion])
                            _, deviations = self.compare(trial, symmetry_map)
                            local = max((dev for t, dev in deviations.items()
                                         if t not in self.impropers and {t[1], t[2]} == bond), default=0.0)
                            trials.append((local, trial))
                        if trials:
                            coords = min(trials, key=lambda r: r[0])[1]
                result = self.compare(coords, symmetry_map)
            except MirrorSearchError:
                if best is not None and best[0] <= MIRROR_TORSION_TOLERANCE:
                    return best
                raise
            if best is None or result[0] < best[0]:
                best = result
        return best

    def needed(self,
               torsions: list[tuple[int, int, int, int]],
               keep: list[tuple[int, int, int, int]] | None = None,
               ) -> list[tuple[int, int, int, int]]:
        """
        Drop the rotors that are not needed to reach the mirror image, one at a time.

        Args:
            torsions (list[tuple]): Rotor torsions that reach the mirror image together.
            keep (list[tuple], optional): Torsions that are kept without testing.

        Returns:
            list[tuple]: The torsions still needed, in their original order (``keep`` excluded).
        """
        keep = keep or list()
        needed = [t for t in torsions if t not in keep]
        for t in list(needed):
            trial = [u for u in needed if u != t]
            if self.best(keep + trial)[0] <= MIRROR_TORSION_TOLERANCE:
                needed = trial
        return needed


def get_symmetry_maps(symbols: list[str],
                      isotopes: list[int],
                      bonds: set[tuple[int, int]],
                      skeleton: list[int],
                      hydrogens: dict[int, list[int]],
                      coords: np.ndarray,
                      max_matches: int,
                      ) -> tuple[list[dict[int, int]], bool]:
    """
    Get the graph-symmetry maps (automorphisms) to try when matching a structure onto its mirror image.
    The maps are taken over the skeleton when it has at least three atoms off one line, and the terminal H atoms are
    assigned later; otherwise they are taken over all atoms. A map must keep the isotopes, the number and isotopes of
    the terminal H atoms of each skeleton atom, and the bond lengths (``keeps_bond_lengths``).

    Args:
        symbols (list[str]): The element symbols.
        isotopes (list[int]): The isotopes (a map never swaps atoms of different isotopes).
        bonds (set[tuple[int, int]]): The bonds.
        skeleton (list[int]): The atoms that are not terminal H atoms.
        hydrogens (dict[int, list[int]]): The terminal H atoms of each skeleton atom.
        coords (np.ndarray): The coordinates.
        max_matches (int): The largest number of maps to enumerate.

    Returns:
        tuple[list[dict[int, int]], bool]:
            The maps (over the skeleton atoms, or over all atoms), and whether there were more than ``max_matches``.
    """
    points = coords[skeleton] - coords[skeleton].mean(axis=0) if skeleton else np.zeros((0, 3))
    singular = np.linalg.svd(points, compute_uv=False) if len(skeleton) >= 3 else np.zeros(2)
    if len(skeleton) >= 3 and singular[1] > 1e-2 * max(singular[0], 1e-12):
        core = list(skeleton)
    else:
        core, hydrogens = list(range(len(symbols))), {i: list() for i in range(len(symbols))}
    core_atoms = set(core)
    adjacency = {i: dict() for i in core}
    for i, j in bonds:
        if i in core_atoms and j in core_atoms:
            adjacency[i][j] = adjacency[j][i] = 1
    graph = ComplexGraph(n_atoms=len(symbols), symbols=list(symbols),
                         bonds={frozenset(bond): 1 for bond in bonds}, core=core, parent=dict(), adj=adjacency,
                         invariant={i: (symbols[i], 0, 0, 0, len(hydrogens[i])) for i in core})
    automorphisms, _ = core_automorphisms(graph, max_count=max_matches + 1)
    maps = [m for m in automorphisms
            if all(isotopes[i] == isotopes[m[i]] and sorted(isotopes[h] for h in hydrogens[i])
                   == sorted(isotopes[h] for h in hydrogens[m[i]]) for i in core)
            and keeps_bond_lengths(m, bonds, coords)]
    return maps, len(automorphisms) > max_matches


def keeps_bond_lengths(symmetry_map: dict[int, int], bonds: set[tuple[int, int]], coords: np.ndarray) -> bool:
    """
    Check that a symmetry map pairs every bond it covers with a bond of the same length, within BOND_LENGTH_TOLERANCE.

    Args:
        symmetry_map (dict[int, int]): Maps atoms onto atoms.
        bonds (set[tuple[int, int]]): The bonds.
        coords (np.ndarray): The coordinates.

    Returns:
        bool: Whether the map keeps every bond length it covers.
    """
    pairs = [(i, j) for i, j in bonds if i in symmetry_map and j in symmetry_map]
    if not pairs:
        return True
    first, second = np.array(pairs).T
    images = np.array([symmetry_map[i] for i in first]), np.array([symmetry_map[j] for j in second])
    lengths = np.linalg.norm(coords[first] - coords[second], axis=1)
    mapped = np.linalg.norm(coords[images[0]] - coords[images[1]], axis=1)
    return bool(np.all(np.abs(lengths - mapped) <= BOND_LENGTH_TOLERANCE))


def fit_onto(coords: np.ndarray, target: np.ndarray, atoms: list[int], target_atoms: list[int]) -> np.ndarray:
    """
    Move a structure rigidly (rotation and translation, no reflection) to best fit some of its atoms onto a target.

    Args:
        coords (np.ndarray): The structure's coordinates.
        target (np.ndarray): The target coordinates.
        atoms (list[int]): The structure atoms to fit.
        target_atoms (list[int]): The target atoms they fit onto, in the same order.

    Returns:
        np.ndarray: The moved coordinates of the whole structure.
    """
    p, q = coords[atoms], target[target_atoms]
    p_center, q_center = p.mean(axis=0), q.mean(axis=0)
    u, _, vt = np.linalg.svd((p - p_center).T @ (q - q_center))
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return (rotation @ (coords - p_center).T).T + q_center


def get_mirror_torsion_deviation(coords: np.ndarray | list | tuple,
                                 mirror: np.ndarray | list | tuple,
                                 atom_map: dict[int, int],
                                 torsions: list[tuple[int, int, int, int]],
                                 ) -> tuple[float, dict]:
    """
    The metric D: the largest deviation between signed torsions of a structure and of an aligned mirror image.

    Args:
        coords: The structure's coordinates.
        mirror: The mirror image's coordinates.
        atom_map (dict[int, int]): Maps each structure atom onto its mirror-image atom.
        torsions (list[tuple]): The torsions to compare, as atom indices of the structure.

    Returns:
        tuple[float, dict]: D in degrees (0 if no torsion is defined in both), and the deviation of every torsion
                            (``NaN`` for a torsion through three collinear atoms, which is not compared).
    """
    if not torsions:
        return 0.0, dict()
    index = np.array(torsions, dtype=int)
    mapped = np.vectorize(atom_map.__getitem__)(index)
    difference = calculate_dihedral_angles(coords, index) - calculate_dihedral_angles(mirror, mapped)
    deviations = np.abs((difference + 180.0) % 360.0 - 180.0)
    compared = deviations[~np.isnan(deviations)]
    return float(compared.max()) if compared.size else 0.0, dict(zip(torsions, deviations.tolist()))


def get_all_torsions(neighbors: dict[int, set[int]],
                     bonds: set[tuple[int, int]],
                     ) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """
    Get every proper torsion a-b-c-d across a bond b-c, and every improper torsion n1-b-n2-n3 about an atom b
    with three or more neighbors.

    Args:
        neighbors (dict[int, set[int]]): The neighbors of each atom.
        bonds (set[tuple[int, int]]): The bonds.

    Returns:
        tuple[list, list]: The proper torsions, and the improper torsions.
    """
    torsions, impropers = list(), list()
    for b, c in sorted(bonds):
        for a in sorted(neighbors[b] - {c}):
            for d in sorted(neighbors[c] - {b}):
                if a != d:
                    torsions.append((a, b, c, d))
    for b in sorted(neighbors):
        if len(neighbors[b]) >= 3:
            for n1, n2, n3 in itertools.combinations(sorted(neighbors[b]), 3):
                impropers.append((n1, b, n2, n3))
    return torsions, impropers


def get_angle_sums(coords: list, neighbors: dict[int, set[int]]) -> dict[int, float]:
    """
    Get the sum of the three bond angles at each three-coordinate atom (360 deg for a planar centre).

    Args:
        coords (list): The coordinates.
        neighbors (dict[int, set[int]]): The neighbors of each atom.

    Returns:
        dict[int, float]: The angle sum in degrees of each atom with exactly three neighbors.
    """
    return {j: sum(calculate_angle(coords, [i, j, k]) for i, k in itertools.combinations(sorted(nbrs), 2))
            for j, nbrs in neighbors.items() if len(nbrs) == 3}


def flatten_centers(coords: np.ndarray, neighbors: dict[int, set[int]], centers: set[int]) -> np.ndarray:
    """
    Move each of some three-coordinate centres into the plane of its three neighbors.

    Args:
        coords (np.ndarray): The coordinates.
        neighbors (dict[int, set[int]]): The neighbors of each atom.
        centers (set[int]): The three-coordinate atoms to move.

    Returns:
        np.ndarray: New coordinates.
    """
    new_coords = np.array(coords, dtype=float)
    for center in sorted(centers):
        p0, p1, p2 = (coords[i] for i in sorted(neighbors[center]))
        normal = np.array(get_normal(p1 - p0, p2 - p0))
        new_coords[center] = coords[center] - np.dot(coords[center] - p0, normal) * normal
    return new_coords


def get_linear_atoms(coords: list, neighbors: dict[int, set[int]]) -> set[int]:
    """
    Get the atoms in the middle of a linear chain: atoms with two neighbors and a bond angle above
    ``LINEAR_ANGLE_THRESHOLD`` between them (e.g., the sp C atoms of C#C, or the central C of an allene).

    Args:
        coords (list): The coordinates.
        neighbors (dict[int, set[int]]): The neighbors of each atom.

    Returns:
        set[int]: The linear atoms.
    """
    return {j for j, nbrs in neighbors.items() if len(nbrs) == 2
            and calculate_angle(coords, [min(nbrs), j, max(nbrs)]) > LINEAR_ANGLE_THRESHOLD}


def walk_linear_chain(neighbors: dict[int, set[int]], linear: set[int], start: int, back: int) -> tuple[int, int]:
    """
    Walk from a linear atom along its chain, away from one of its neighbors, to the first non-linear atom.

    Args:
        neighbors (dict[int, set[int]]): The neighbors of each atom.
        linear (set[int]): The linear atoms (see ``get_linear_atoms``).
        start (int): The linear atom to start from.
        back (int): The neighbor of ``start`` to walk away from.

    Returns:
        tuple[int, int]: The first non-linear atom, and the chain atom just before it.
    """
    previous, current = back, start
    while current in linear:
        previous, current = current, next(iter(neighbors[current] - {previous}))
    return current, previous


def get_axis_torsions(neighbors: dict[int, set[int]], linear: set[int]) -> list[tuple[int, int, int, int]]:
    """
    Get the torsions across linear chains (e.g., the C=C=C of an allene), which proper torsions cannot see because
    every bond angle along the chain is straight: a-b-c-d for the two ends b and c of the chain, with a and d off it.

    Args:
        neighbors (dict[int, set[int]]): The neighbors of each atom.
        linear (set[int]): The linear atoms (see ``get_linear_atoms``).

    Returns:
        list[tuple[int, int, int, int]]: The torsions across each linear chain.
    """
    torsions = list()
    for b in sorted(set(neighbors) - linear):
        for first in sorted(neighbors[b] & linear):
            c, previous = walk_linear_chain(neighbors, linear, first, b)
            if c <= b:
                continue
            for a in sorted(neighbors[b] - {first}):
                for d in sorted(neighbors[c] - {previous}):
                    if a != d:
                        torsions.append((a, b, c, d))
    return torsions


def get_signed_dihedral(coords: list, torsion: tuple[int, int, int, int] | list[int]) -> float:
    """
    Get a signed dihedral angle, ``NaN`` if three of its atoms are collinear.

    Args:
        coords (list): The coordinates.
        torsion (tuple[int, int, int, int] | list[int]): Four 0-indexed atoms.

    Returns:
        float: The dihedral angle in degrees, in [-180, 180).
    """
    return (float(calculate_dihedral_angles(coords, [[int(i) for i in torsion]])[0]) + 180.0) % 360.0 - 180.0


def rotate_top(coords: np.ndarray,
               neighbors: dict[int, set[int]],
               torsion: tuple[int, int, int, int],
               value: float,
               dihedral: tuple[int, int, int, int] | None = None,
               ) -> np.ndarray:
    """
    Set a dihedral by rotating the atoms on the c side of the b-c bond of a torsion about that bond.
    The b-c bond must not lie on a ring (see ``is_ring_twist``).

    Args:
        coords (np.ndarray): The coordinates.
        neighbors (dict[int, set[int]]): The neighbors of each atom.
        torsion (tuple[int, int, int, int]): The torsion a-b-c-d whose b-c bond is the rotation axis.
        value (float): The new dihedral value in degrees.
        dihedral (tuple[int, int, int, int], optional): The torsion whose value is set, if not ``torsion``
                                                        (e.g., its extension across a linear chain).

    Returns:
        np.ndarray: New coordinates (unchanged if the dihedral is undefined).
    """
    _, b, c, _ = torsion
    current = get_signed_dihedral(coords.tolist(), dihedral or torsion)
    if math.isnan(current):
        return coords
    new_coords = [tuple(row) for row in coords.tolist()]
    apply_rodrigues_rotation(new_coords, axis_origin=coords[c].tolist(),
                             axis_unit=unit_vector((coords[c] - coords[b]).tolist()),
                             angle_rad=math.radians(value - current),
                             indices=sorted(get_reachable(neighbors, c, cut=(b, c)) - {c}))
    return np.array(new_coords, dtype=float)


def is_connected(bonds: set[tuple[int, int]], n: int) -> bool:
    """
    Determine whether a graph is connected.

    Args:
        bonds (set[tuple[int, int]]): The bonds.
        n (int): The number of atoms.

    Returns:
        bool: Whether every atom is reachable from atom 0.
    """
    return len(get_reachable(get_neighbors(bonds, n), 0)) == n


def atom_label(symbols: list[str], index: int) -> str:
    """
    Label an atom by its element and 1-indexed position, e.g. 'C2'.

    Args:
        symbols (list[str]): The element symbols.
        index (int): The 0-indexed atom.

    Returns:
        str: The label.
    """
    return f'{symbols[index]}{index + 1}'


def bond_label(symbols: list[str], i: int, j: int) -> str:
    """
    Label a bond by its atoms in index order, e.g. 'C2-C3'.

    Args:
        symbols (list[str]): The element symbols.
        i (int): A 0-indexed atom of the bond.
        j (int): The other 0-indexed atom.

    Returns:
        str: The label.
    """
    return '-'.join(atom_label(symbols, k) for k in sorted((i, j)))
