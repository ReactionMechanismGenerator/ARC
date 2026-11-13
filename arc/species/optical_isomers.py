"""
A module for determining the number of optical isomers of a well or a TS (ARC issue #413).

The rule: n = 2 for a chiral structure, except n = 1 when the treated (converged, used) rotors reach its mirror image.
A ring twist is never a rotor, and a stereocenter always gives 2.

A rotor integral over 0..2*pi already contains the mirror-image well along its torsion, so only mirror wells that no
treated rotor reaches are counted by n. Reach is decided geometrically: the treated tops are rotated to every
candidate angle that maps a graph-equivalent atom pair onto its mirror position, the result is aligned onto the mirror
image with atom permutation, and D, the largest signed-torsion deviation between the two, is compared with
``MIRROR_TORSION_TOLERANCE``. A geometry with D <= tolerance and no rotation is achiral.
"""

import itertools
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from arc.common import get_logger
from arc.species.conformers import find_internal_rotors, get_graph_stereocenters, get_ts_bonds, is_ring_twist
from arc.species.species import USER_OPTICAL_ISOMERS_CAUSE

if TYPE_CHECKING:
    from arc.reaction import ARCReaction
    from arc.species.species import ARCSpecies

logger = get_logger()

# The largest signed-torsion deviation (degrees) between a structure and its aligned mirror image for the two to count
# as the same well. Measured on wB97X-D geometries: true matches <= 12.0 deg, true non-matches >= 48.2 deg.
MIRROR_TORSION_TOLERANCE = 25.0

# Torsions through a bond angle above this (degrees) are undefined and are not compared (e.g., C...H...C of a TS).
LINEAR_ANGLE_THRESHOLD = 165.0

# A symmetry map may not pair bonds whose lengths differ by more than this (Angstrom), so it never swaps chemically
# distinct bonds the bond graph does not tell apart (e.g., the single and double bonds of cyclooctatetraene).
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

OPTICAL_ISOMER_CAUSE_PARTS = {
    'harmonic': '{bond} torsion harmonic ({why})',
    'ring_twist': 'ring twist about {bonds}',
    'configuration': 'configuration at {atoms}',
    'unnamed': 'no treated rotor reaches it (the cause could not be named: {error})',
}


class MirrorSearchError(Exception):
    """The mirror-reach search exceeded its budget"""
    pass


class OpticalIsomers(NamedTuple):
    """
    The number of optical isomers of a well or a TS, and why.

    Attributes:
        value (int | None): The number of optical isomers, ``None`` if it could not be determined.
        kind (str): The key of the cause in ``OPTICAL_ISOMER_CAUSES``.
        cause (str): The cause in words.
        deviation (float | None): D, the largest torsion deviation from the mirror image in degrees, if measured.
        fields (dict): The values the cause template was formatted with.
    """
    value: int | None
    kind: str
    cause: str
    deviation: float | None
    fields: dict


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


def _result(value: int | None, kind: str, deviation: float | None = None, **fields) -> OpticalIsomers:
    """Build an ``OpticalIsomers`` result whose cause is its template formatted with ``fields``"""
    if deviation is not None:
        fields['deviation'] = deviation
    return OpticalIsomers(value=value, kind=kind, cause=format_cause(kind, **fields), deviation=deviation,
                          fields=fields)


def determine_optical_isomers(species: 'ARCSpecies',
                              rotors: list[dict],
                              reaction: 'ARCReaction | None' = None,
                              max_matches: int = MAX_SYMMETRY_MATCHES,
                              ) -> OpticalIsomers:
    """
    Determine the number of optical isomers of a well or a TS, given the rotors its Arkane file treats.

    Args:
        species (ARCSpecies): The well or TS.
        rotors (list[dict]): The treated rotors (``rotors_dict`` entries written into the Arkane file).
        reaction (ARCReaction, optional): The reaction whose TS is ``species``, used for the TS graph.
        max_matches (int, optional): The largest number of graph-symmetry matches to enumerate.

    Returns:
        OpticalIsomers: The number of optical isomers and its cause.
    """
    user_value = species.optical_isomers if species.has_user_optical_isomers() else None
    try:
        result = _determine(species, rotors, reaction, max_matches)
    except Exception as e:  # The key is optional, so an unexpected failure must not stop the Arkane run.
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
               max_matches: int,
               ) -> OpticalIsomers:
    """The rule, without the user value and the exception guard; see ``determine_optical_isomers``"""
    xyz = species.get_xyz(generate=False)
    if xyz is None:
        return _result(None, 'undetermined', reason='no optimised geometry')
    symbols = list(xyz['symbols'])
    if species.mol is not None and [atom.element.symbol for atom in species.mol.atoms] != symbols:
        return _result(None, 'undetermined', reason='the atom order of the 2D graph and the geometry differ')
    if species.is_ts:
        bonds = get_ts_bonds(mol=species.mol, xyz=xyz, reaction=reaction)[0]
    else:
        if species.mol is None:
            return _result(None, 'undetermined', reason='no 2D graph')
        chiral, centers, reason = get_graph_stereocenters(species.label, species.mol, xyz)
        if chiral is None:
            return _result(None, 'undetermined', reason=reason)
        if chiral:
            return _result(2, 'stereocentre', atoms=', '.join(atom_label(symbols, i) for i in centers) or 'an axis')
        index = {atom: i for i, atom in enumerate(species.mol.atoms)}
        bonds = {tuple(sorted((index[a], index[b]))) for a in species.mol.atoms for b in a.edges}
    if len(symbols) < 3:
        return _result(1, 'achiral', deviation=0.0)
    if not is_connected(bonds, len(symbols)):
        return _result(None, 'undetermined', reason='the bond graph is not connected, so its fragments\' relative '
                                                    'orientation cannot be compared with the mirror image')
    reach = MirrorReach(xyz, bonds, max_matches)
    if reach.too_many_matches:
        return _result(None, 'undetermined', reason=f'more than {max_matches} graph symmetry matches')
    deviation, _ = reach.best([])
    if deviation <= MIRROR_TORSION_TOLERANCE:
        return _result(1, 'achiral', deviation=deviation)
    torsions = [t for t in get_rotor_torsions(rotors) if not is_ring_twist(bonds, (t[1], t[2]))]
    deviation, worst = reach.best(torsions)
    if deviation <= MIRROR_TORSION_TOLERANCE:
        try:
            needed = reach.needed(torsions)
        except MirrorSearchError:  # Naming the rotors must not turn a decided n into an undetermined one.
            needed = torsions
        return _result(1, 'reached', deviation=deviation,
                       rotors=', '.join(bond_label(symbols, t[1], t[2]) for t in needed),
                       plural='s' if len(needed) > 1 else '')
    try:
        parts = explain_unreached(species, symbols, bonds, reach, torsions, worst)
    except Exception as e:  # Naming the cause must not turn a decided n into an undetermined one.
        parts = [OPTICAL_ISOMER_CAUSE_PARTS['unnamed'].format(error=f'{type(e).__name__}: {e}')]
    return _result(2, 'not_reached', deviation=deviation, parts='; '.join(parts))


def explain_unreached(species: 'ARCSpecies',
                      symbols: list[str],
                      bonds: set[tuple[int, int]],
                      reach: 'MirrorReach',
                      torsions: list[tuple[int, int, int, int]],
                      worst: dict,
                      ) -> list[str]:
    """
    Name what keeps a structure from reaching its mirror image.

    Args:
        species (ARCSpecies): The well or TS.
        symbols (list[str]): The element symbols.
        bonds (set[tuple[int, int]]): The graph bonds.
        reach (MirrorReach): The mirror-reach search of the structure.
        torsions (list[tuple]): The treated rotor torsions.
        worst (dict): The torsion deviations of the best search with the treated rotors.

    Returns:
        list[str]: The causes in words.
    """
    parts = list()
    ring_bonds = sorted({tuple(sorted((t[1], t[2]))) for t, dev in worst.items() if dev > MIRROR_TORSION_TOLERANCE
                         and t not in reach.impropers and tuple(sorted((t[1], t[2]))) in bonds
                         and is_ring_twist(bonds, (t[1], t[2]))})
    if ring_bonds:
        parts.append(OPTICAL_ISOMER_CAUSE_PARTS['ring_twist'].format(
            bonds=', '.join(bond_label(symbols, *bond) for bond in ring_bonds)))
    treated_bonds = {tuple(sorted((t[1], t[2]))) for t in torsions}
    untreated = [(t, why) for t, why in get_untreated_torsions(species)
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
        if not centers:
            t = max(worst, key=worst.get)
            centers = sorted({t[1], t[2]})
        parts.append(OPTICAL_ISOMER_CAUSE_PARTS['configuration'].format(
            atoms=', '.join(atom_label(symbols, i) for i in centers)))
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


def get_untreated_torsions(species: 'ARCSpecies') -> list[tuple[tuple[int, int, int, int], str]]:
    """
    Get the torsions of a species that a rotor could have treated, with why each was not treated.
    These are its ``rotors_dict`` entries, plus, for a well, the rotors of its 2D graph that were not scanned.

    Args:
        species (ARCSpecies): The well or TS.

    Returns:
        list[tuple[tuple[int, int, int, int], str]]: The torsions and why they are not treated.
    """
    out, seen = list(), set()
    for rotor in (species.rotors_dict or dict()).values():
        if rotor.get('success'):
            why = 'rotor not used in the Arkane file'
        elif rotor.get('invalidation_reason'):
            why = f"scan failed: {rotor['invalidation_reason'].strip()}"
        else:
            why = 'scan not converged'
        for t in get_rotor_torsions([rotor]):
            out.append((t, why))
            seen.add(tuple(sorted((t[1], t[2]))))
    if not species.is_ts and species.mol is not None:
        for rotor in find_internal_rotors(species.mol):
            t = tuple(i - 1 for i in rotor['scan'])
            if tuple(sorted((t[1], t[2]))) not in seen:
                out.append((t, 'not scanned'))
    return out


class MirrorReach(object):
    """
    Search for rotor angles that turn a structure into its mirror image.

    Args:
        xyz (dict): The coordinates.
        bonds (set[tuple[int, int]]): The graph bonds, 0-indexed.
        max_matches (int): The largest number of graph-symmetry matches to enumerate.
    """

    def __init__(self, xyz: dict, bonds: set[tuple[int, int]], max_matches: int):
        self.symbols = list(xyz['symbols'])
        self.n = len(self.symbols)
        self.coords = np.array(xyz['coords'], dtype=float)
        self.mirror = self.coords * np.array([-1.0, 1.0, 1.0])
        self.bonds = {tuple(sorted(bond)) for bond in bonds}
        self.neighbors = {i: set() for i in range(self.n)}
        for i, j in self.bonds:
            self.neighbors[i].add(j)
            self.neighbors[j].add(i)
        self.max_matches = max_matches
        self.evaluations = 0
        self.isotopes = [int(i) for i in xyz['isotopes']] if xyz.get('isotopes') else [0] * self.n
        self.rd_mol = build_rd_mol(self.symbols, self.bonds, self.coords, self.isotopes)
        self.ranks = list(Chem.CanonicalRankAtoms(self.rd_mol, breakTies=False))
        propers, impropers = get_all_torsions(self.neighbors, self.bonds)
        propers += get_axis_torsions(self.coords, self.neighbors)
        self.impropers = set(impropers)
        self.torsions = [t for t in propers + impropers
                         if max(get_angle(self.coords, *t[:3]), get_angle(self.coords, *t[1:]))
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
        return sorted({round(-get_dihedral(self.coords, (a2, b2, c2, d2)), 6) for a2 in fixed for d2 in moving})

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
            MirrorSearchError: If the search runs out of comparisons before any map brings D within
                               ``MIRROR_TORSION_TOLERANCE``.

        Returns:
            tuple[float, dict]: The smallest D in degrees, and the torsion deviations at it, or, if the comparisons
                                ran out after a map brought D within the tolerance, the smallest D found so far.
        """
        best = None
        for symmetry_map in self.maps:
            try:
                coords = self.coords
                for _ in range(2 if torsions else 0):
                    for torsion in torsions:
                        bond = {torsion[1], torsion[2]}
                        trials = list()
                        for value in self.candidates(torsion, symmetry_map):
                            trial = rotate_top(coords, self.neighbors, torsion, value)
                            _, deviations = self.compare(trial, symmetry_map)
                            local = max((dev for t, dev in deviations.items()
                                         if t not in self.impropers and {t[1], t[2]} == bond), default=0.0)
                            trials.append((local, trial))
                        coords = min(trials, key=lambda r: r[0])[1]
                result = self.compare(coords, symmetry_map)
            except MirrorSearchError:
                # The mirror image was already reached by an earlier map; a later map can only lower D.
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
    Terminal H atoms multiply the automorphisms (6 per CH3 group) without changing the fit, so the maps are taken
    over the skeleton when it has at least three atoms off one line, and the terminal H atoms are assigned later.

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
        index = {atom: i for i, atom in enumerate(skeleton)}
        sub_bonds = {(index[i], index[j]) for i, j in bonds if i in index and j in index}
        mol = build_rd_mol([symbols[i] for i in skeleton], sub_bonds, coords[skeleton])
        matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=False, maxMatches=max_matches + 1)
        maps = [{skeleton[q]: skeleton[t] for q, t in enumerate(match)} for match in matches]
        maps = [m for m in maps if all(len(hydrogens[i]) == len(hydrogens[m[i]]) and isotopes[i] == isotopes[m[i]]
                                       and sorted(isotopes[h] for h in hydrogens[i])
                                       == sorted(isotopes[h] for h in hydrogens[m[i]]) for i in skeleton)
                and keeps_bond_lengths(m, bonds, coords)]
        return maps, len(matches) > max_matches
    mol = build_rd_mol(symbols, bonds, coords)
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=False, maxMatches=max_matches + 1)
    maps = [dict(enumerate(match)) for match in matches]
    maps = [m for m in maps
            if all(isotopes[i] == isotopes[j] for i, j in m.items()) and keeps_bond_lengths(m, bonds, coords)]
    return maps, len(matches) > max_matches


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
        tuple[float, dict]: D in degrees (0 if no torsion), and the deviation of every torsion.
    """
    if not torsions:
        return 0.0, dict()
    index = np.array(torsions, dtype=int)
    mapped = np.vectorize(atom_map.__getitem__)(index)
    difference = get_dihedrals(np.asarray(coords, dtype=float), index) \
        - get_dihedrals(np.asarray(mirror, dtype=float), mapped)
    deviations = np.abs((difference + 180.0) % 360.0 - 180.0)
    return float(deviations.max()), dict(zip(torsions, deviations.tolist()))


def get_dihedrals(coords: np.ndarray, torsions: np.ndarray) -> np.ndarray:
    """
    Get signed dihedral angles, vectorized.

    Args:
        coords (np.ndarray): The coordinates.
        torsions (np.ndarray): Rows of four atom indices.

    Returns:
        np.ndarray: The dihedral angles in degrees, in (-180, 180].
    """
    p0, p1, p2, p3 = (coords[torsions[:, k]] for k in range(4))
    b1 = p2 - p1
    b1 /= np.linalg.norm(b1, axis=1)[:, None]
    v = (p0 - p1) - np.sum((p0 - p1) * b1, axis=1)[:, None] * b1
    w = (p3 - p2) - np.sum((p3 - p2) * b1, axis=1)[:, None] * b1
    return np.degrees(np.arctan2(np.sum(np.cross(b1, v) * w, axis=1), np.sum(v * w, axis=1)))


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


def get_axis_torsions(coords: np.ndarray, neighbors: dict[int, set[int]]) -> list[tuple[int, int, int, int]]:
    """
    Get the torsions across linear chains (e.g., the C=C=C of an allene), which proper torsions cannot see because
    every bond angle along the chain is straight: a-b-c-d for the two ends b and c of the chain, with a and d off it.

    Args:
        coords (np.ndarray): The coordinates.
        neighbors (dict[int, set[int]]): The neighbors of each atom.

    Returns:
        list[tuple[int, int, int, int]]: The torsions across each linear chain.
    """
    linear = {j for j, nbrs in neighbors.items() if len(nbrs) == 2
              and get_angle(coords, min(nbrs), j, max(nbrs)) > LINEAR_ANGLE_THRESHOLD}
    torsions = list()
    for b in sorted(set(neighbors) - linear):
        for first in sorted(neighbors[b] & linear):
            previous, current = b, first
            while current in linear:
                previous, current = current, next(iter(neighbors[current] - {previous}))
            c = current
            if c <= b or c in linear:
                continue
            for a in sorted(neighbors[b] - {first}):
                for d in sorted(neighbors[c] - {previous}):
                    if a != d:
                        torsions.append((a, b, c, d))
    return torsions


def build_rd_mol(symbols: list[str],
                 bonds: set[tuple[int, int]],
                 coords: np.ndarray,
                 isotopes: list[int] | None = None,
                 ) -> Chem.rdchem.Mol:
    """
    Build an RDKit molecule with explicit atoms, single bonds, and one conformer.

    Args:
        symbols (list[str]): The element symbols.
        bonds (set[tuple[int, int]]): The bonds.
        coords (np.ndarray): The coordinates.
        isotopes (list[int], optional): The isotopes, so graph-symmetry ranks tell e.g. D from H.

    Returns:
        Chem.rdchem.Mol: The molecule.
    """
    rd_mol = Chem.RWMol()
    for i, symbol in enumerate(symbols):
        rd_atom = Chem.Atom(symbol)
        rd_atom.SetNoImplicit(True)
        if isotopes is not None:
            rd_atom.SetIsotope(int(isotopes[i]))
        rd_mol.AddAtom(rd_atom)
    for i, j in sorted(bonds):
        rd_mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    rd_mol.UpdatePropertyCache(strict=False)
    conformer = Chem.Conformer(len(symbols))
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    rd_mol.AddConformer(conformer)
    return rd_mol.GetMol()


def get_dihedral(coords: np.ndarray | list | tuple, torsion: tuple[int, int, int, int] | list[int]) -> float:
    """
    Get a signed dihedral angle.

    Args:
        coords: The coordinates.
        torsion: Four atom indices.

    Returns:
        float: The dihedral angle in degrees, in (-180, 180].
    """
    p0, p1, p2, p3 = (np.asarray(coords[i], dtype=float) for i in torsion)
    b1 = (p2 - p1) / np.linalg.norm(p2 - p1)
    v = (p0 - p1) - np.dot(p0 - p1, b1) * b1
    w = (p3 - p2) - np.dot(p3 - p2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def get_angle(coords: np.ndarray, i: int, j: int, k: int) -> float:
    """
    Get the angle i-j-k.

    Args:
        coords (np.ndarray): The coordinates.
        i (int): The first atom.
        j (int): The vertex atom.
        k (int): The third atom.

    Returns:
        float: The angle in degrees.
    """
    v1, v2 = coords[i] - coords[j], coords[k] - coords[j]
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1.0, 1.0))))


def get_top(neighbors: dict[int, set[int]], b: int, c: int) -> set[int]:
    """
    Get the atoms on the c side of the bond b-c, including c.

    Args:
        neighbors (dict[int, set[int]]): The neighbors of each atom.
        b (int): The fixed pivot.
        c (int): The moving pivot.

    Returns:
        set[int]: The atoms reachable from c without crossing b-c (includes b if b-c lies on a ring).
    """
    top, stack = {c}, [c]
    while stack:
        atom = stack.pop()
        for neighbor in neighbors[atom]:
            if neighbor not in top and not (atom == c and neighbor == b):
                top.add(neighbor)
                stack.append(neighbor)
    return top


def rotate_top(coords: np.ndarray,
               neighbors: dict[int, set[int]],
               torsion: tuple[int, int, int, int],
               value: float,
               ) -> np.ndarray:
    """
    Set a dihedral by rotating the atoms on the c side of its b-c bond about that bond.

    Args:
        coords (np.ndarray): The coordinates.
        neighbors (dict[int, set[int]]): The neighbors of each atom.
        torsion (tuple[int, int, int, int]): The torsion a-b-c-d.
        value (float): The new dihedral value in degrees.

    Returns:
        np.ndarray: New coordinates (unchanged if b-c lies on a ring).
    """
    _, b, c, _ = torsion
    top = get_top(neighbors, b, c)
    if b in top:
        return coords
    axis = (coords[c] - coords[b]) / np.linalg.norm(coords[c] - coords[b])
    theta = np.radians(value - get_dihedral(coords, torsion))
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rotation = np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * k @ k
    new_coords = np.array(coords, dtype=float)
    for i in top - {c}:
        new_coords[i] = coords[c] + rotation @ (coords[i] - coords[c])
    return new_coords


def set_dihedral(xyz: dict, bonds: set[tuple[int, int]], torsion: list[int], value: float) -> dict:
    """
    Set a dihedral of a geometry by rotating the side of its last atom.

    Args:
        xyz (dict): The coordinates.
        bonds (set[tuple[int, int]]): The bonds, 0-indexed.
        torsion (list[int]): The torsion, 0-indexed.
        value (float): The new dihedral value in degrees.

    Returns:
        dict: The new coordinates.
    """
    neighbors = {i: set() for i in range(len(xyz['symbols']))}
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)
    coords = rotate_top(np.array(xyz['coords'], dtype=float), neighbors, tuple(torsion), value)
    return {'symbols': tuple(xyz['symbols']), 'isotopes': tuple(xyz['isotopes']),
            'coords': tuple(tuple(float(x) for x in row) for row in coords)}


def is_connected(bonds: set[tuple[int, int]], n: int) -> bool:
    """
    Determine whether a graph is connected.

    Args:
        bonds (set[tuple[int, int]]): The bonds.
        n (int): The number of atoms.

    Returns:
        bool: Whether every atom is reachable from atom 0.
    """
    neighbors = {i: set() for i in range(n)}
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)
    seen, stack = {0}, [0]
    while stack:
        for neighbor in neighbors[stack.pop()] - seen:
            seen.add(neighbor)
            stack.append(neighbor)
    return len(seen) == n


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
