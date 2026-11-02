"""
An adapter for TS guesses from a nudged elastic band (NEB) run with ASE on an ASE calculator, typically an MLIP.

ASE's NEB, DyNEB or AutoNEB (``ase.mep``) relaxes a band between the reactant and product wells on the calculator of
the reaction's TS-guess level (or of ``ase_neb_settings['level']``), and the highest image becomes a TS guess that ARC
then optimizes and verifies like any other. The band runs in ``ase_script.py`` (job type 'neb'), through the ASE
adapter's submission, so it uses the level's environment, queue and persistent worker.

ASE NEB: https://ase-lib.org/ase/neb.html
"""

import os
from typing import TYPE_CHECKING

import numpy as np

from arc.common import get_element_mass, get_logger, save_yaml_file
from arc.exceptions import DependencyError
from arc.imports import settings
from arc.job.adapters.ase_adapter import ASEAdapter
from arc.job.adapters.common import has_one_molecule_side
from arc.job.factory import register_job_adapter
from arc.level import Level
from arc.species import ARCSpecies
from arc.species.converter import sort_xyz_using_indices

if TYPE_CHECKING:
    from arc.reaction import ARCReaction

logger = get_logger()

ase_neb_settings = settings.get('ase_neb_settings', dict())

# The smallest distance (Angstrom) between the fragments of a bimolecular endpoint.
FRAGMENT_SEPARATION = 3.0


def molecule_count(reaction: 'ARCReaction', well: int) -> int:
    """The number of molecules in a well (0: reactants, 1: products), counting a species once per occurrence."""
    side = reaction.r_species if well == 0 else reaction.p_species
    return sum(reaction.get_species_count(species=spc, well=well) for spc in side)


def kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The rotation that best overlays ``mobile`` onto ``target`` (both N x 3, the same atoms in the same order) after
    centering, and the two centers: ``(x - mobile_center) @ rotation.T + target_center`` maps ``mobile`` onto
    ``target``. A single point fixes no rotation (identity).
    """
    mobile_center, target_center = mobile.mean(axis=0), target.mean(axis=0)
    if len(mobile) == 1:
        return np.eye(3), mobile_center, target_center
    h = (mobile - mobile_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T, mobile_center, target_center


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``mobile`` rigidly rotated and translated onto ``target`` (both N x 3, the same atoms in the same order)."""
    return align_on(mobile, target, list(range(len(mobile))))


def align_on(mobile: np.ndarray, target: np.ndarray, atoms: list[int]) -> np.ndarray:
    """All of ``mobile`` moved rigidly so that its ``atoms`` best overlay the same atoms of ``target``."""
    rotation, mobile_center, target_center = kabsch_transform(mobile[atoms], target[atoms])
    return (mobile - mobile_center) @ rotation.T + target_center


def side_fragments(reaction: 'ARCReaction', well: int) -> list[list[int]]:
    """
    The atom indices of each molecule of a well, in the reactants' atom order (the order of both NEB endpoints).

    The reactants are listed molecule by molecule, so their fragments are contiguous. The products are put into the
    reactants' order by the atom map, so product atom ``atom_map[i]`` is reactant atom ``i``.
    """
    side = reaction.r_species if well == 0 else reaction.p_species
    molecules = [spc for spc in side for _ in range(reaction.get_species_count(species=spc, well=well))]
    bounds, start = list(), 0
    for spc in molecules:
        bounds.append((start, start + spc.number_of_atoms))
        start += spc.number_of_atoms
    if well == 0:
        return [list(range(lo, hi)) for lo, hi in bounds]
    return [[i for i, j in enumerate(reaction.atom_map) if lo <= j < hi] for lo, hi in bounds]


def fit_atoms(fragment: list[int], symbols: tuple, reactive: set[int]) -> list[int]:
    """
    The atoms of ``fragment`` to align it by: those whose bonds do not change (spectators), else its heavy atoms, else
    all of it. Fitting a migrating atom would drag the rest of the fragment toward where that atom sits on the other
    side, e.g. HO2 onto the O-O and a methyl H of C2H5O2.
    """
    for atoms in ([i for i in fragment if i not in reactive], [i for i in fragment if symbols[i] != 'H']):
        if len(atoms) >= min(3, len(fragment)):
            return atoms
    return list(fragment)


def place_fragments(joined: np.ndarray, separated: np.ndarray, fragments: list[list[int]], symbols: tuple,
                    reactive: set[int] | None = None, min_distance: float = FRAGMENT_SEPARATION) -> np.ndarray:
    """
    A bimolecular endpoint built around the unimolecular one: each fragment's own geometry (from ``separated``) is
    aligned onto its atoms in ``joined`` (by the atoms :func:`fit_atoms` picks), and the fragments are then moved
    apart, keeping their center of mass fixed, until no two atoms of different fragments are closer than
    ``min_distance``. The NEB therefore starts from the orientation of the reactive site, not from fragments ARC
    placed side by side along x, and a leaving atom or small group does the moving while a heavy fragment stays where
    it is in the unimolecular well.
    """
    coords = np.array(separated, dtype=float)
    for fragment in fragments:
        anchors = fit_atoms(fragment, symbols, reactive or set())
        local = [fragment.index(i) for i in anchors]
        coords[fragment] = align_on(coords[fragment], joined[fragment], local)
    masses = np.array([get_element_mass(symbol)[0] for symbol in symbols])
    fragment_masses = np.array([masses[fragment].sum() for fragment in fragments])
    centers = np.array([np.average(coords[fragment], axis=0, weights=masses[fragment]) for fragment in fragments])
    center = np.average(centers, axis=0, weights=fragment_masses)
    directions = list()
    for fragment_center in centers:
        d = fragment_center - center
        directions.append(d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-8 else np.array([1.0, 0.0, 0.0]))
    # Each step moves the fragments 0.05 A further apart, each by its share (1 - m_k / M) of it.
    steps = [0.05 * (1.0 - m / fragment_masses.sum()) * d for m, d in zip(fragment_masses, directions)]

    def closest():
        return min(np.min(np.linalg.norm(coords[a][:, None, :] - coords[b][None, :, :], axis=2))
                   for i, a in enumerate(fragments) for b in fragments[i + 1:])

    for _ in range(1000):
        if closest() >= min_distance:
            break
        for fragment, step in zip(fragments, steps):
            coords[fragment] += step
    return coords


def fold_is_sound(xyz: dict, folded: dict, species: ARCSpecies) -> bool:
    """
    Whether a folded geometry is still the same molecule at sensible distances: every bond of ``species`` keeps its
    length to 0.05 A (a fold only turns dihedrals), and no two atoms it does not bond, at least three bonds apart, are
    closer than 1.6 A (either is H) or 2.2 A (two heavy atoms). The near-attack helper can fail both ways: it can move
    a terminal group's atoms off their bonds (seen for n-C4H9O2, C-H 0.83-1.54 A), and it can turn a migrating H
    almost onto its acceptor, a TS-like shape rather than an endpoint.
    """
    before, after = np.array(xyz['coords'], dtype=float), np.array(folded['coords'], dtype=float)
    symbols = xyz['symbols']
    adjacency = {i: {species.mol.atoms.index(n) for n in atom.edges} for i, atom in enumerate(species.mol.atoms)}
    for i, neighbors in adjacency.items():
        for j in neighbors:
            if abs(np.linalg.norm(after[i] - after[j]) - np.linalg.norm(before[i] - before[j])) > 0.05:
                return False
    for i in range(len(symbols)):
        near = set(adjacency[i]) | {k for j in adjacency[i] for k in adjacency[j]}
        for j in range(i + 1, len(symbols)):
            if j not in near and np.linalg.norm(after[i] - after[j]) < (1.6 if 'H' in (symbols[i], symbols[j]) else 2.2):
                return False
    return True


def near_attack(xyz: dict, species: ARCSpecies, forming: list[tuple[int, int]]) -> dict:
    """
    ``xyz`` of a one-molecule well folded into its near-attack conformation for the bonds it forms (the ring-like
    shape of a cyclic TS; the helper of ARC's linear adapter), or unchanged if there is nothing to fold, the fold fails,
    or the folded geometry is not sound (see :func:`fold_is_sound`). Folding the wells of an intramolecular reaction keeps the
    band from also carrying a conformational change.
    """
    if not forming or species.mol is None:
        return xyz
    from arc.job.adapters.ts.linear_utils.isomerization import get_near_attack_xyz
    try:
        folded = get_near_attack_xyz(xyz=xyz, mol=species.mol, bonds=forming)
    except Exception as e:  # a fold that fails leaves a usable, unfolded endpoint
        logger.debug(f'No near-attack conformation for {species.label}: {e}')
        return xyz
    return folded if fold_is_sound(xyz, folded, species) else xyz


# X-H bond lengths (A) for a hydrogen moved onto its acceptor, by acceptor element.
XH_BOND = {'C': 1.09, 'N': 1.01, 'O': 0.97, 'S': 1.34}


def h_transfers(reaction: 'ARCReaction', formed: list[tuple[int, int]], broken: list[tuple[int, int]],
                symbols: tuple) -> list[tuple[int, int]] | None:
    """
    The (hydrogen, acceptor) pairs of a reaction whose only bond changes are hydrogens moving from one atom to another
    (e.g., intra_H_migration), or None for any other reaction.
    """
    pairs = list()
    for bond in formed:
        h = next((i for i in bond if symbols[i] == 'H'), None)
        if h is None or not any(h in b for b in broken):
            return None
        pairs.append((h, bond[0] if bond[1] == h else bond[1]))
    if len(broken) != len(pairs) or not all(any(h in b for h, _ in pairs) for b in broken):
        return None
    return pairs or None


def moved_h_product(r_xyz: dict, transfers: list[tuple[int, int]], neighbors: dict[int, set[int]]) -> dict | None:
    """
    A product endpoint for a hydrogen transfer built from the (folded) reactant: every atom stays where it is in the
    reactant, except that each migrating H moves onto its acceptor at the X-H bond length. It points from the acceptor
    toward where it came from, tilted away from the acceptor's other neighbors (``neighbors``, the reactant's bonds)
    until every neighbor-acceptor-H angle is at least 100 degrees; straight along that line, n-C4H9O2's 1,5-H shift
    put the H at 69 degrees to the O-O bond, 1.3 A from the other O. The band then carries the transfer and nothing
    else, where the product's own minimum can sit in another conformation altogether. None if a moved H lands within
    1.0 A of another atom.
    """
    coords = np.array(r_xyz['coords'], dtype=float)
    for h, acceptor in transfers:
        toward = coords[h] - coords[acceptor]
        toward /= np.linalg.norm(toward)
        others = [n for n in neighbors.get(acceptor, set()) if n != h]
        away = sum((coords[acceptor] - coords[n]) / np.linalg.norm(coords[acceptor] - coords[n]) for n in others) \
            if others else np.zeros(3)
        direction = toward
        if np.linalg.norm(away) > 1e-8:
            away /= np.linalg.norm(away)
            for weight in np.arange(0.0, 5.01, 0.1):
                direction = (toward + weight * away) / np.linalg.norm(toward + weight * away)
                if all(np.degrees(np.arccos(np.clip(np.dot(direction, (coords[n] - coords[acceptor])
                                                           / np.linalg.norm(coords[n] - coords[acceptor])), -1, 1)))
                       >= 100.0 for n in others):
                    break
        coords[h] = coords[acceptor] + XH_BOND.get(r_xyz['symbols'][acceptor], 1.0) * direction
    for h, acceptor in transfers:
        rest = [i for i in range(len(coords)) if i not in (h, acceptor)]
        if np.min(np.linalg.norm(coords[rest] - coords[h], axis=1)) < 1.0:
            return None
    return {'symbols': tuple(r_xyz['symbols']), 'isotopes': tuple(r_xyz['isotopes']),
            'coords': tuple(tuple(float(x) for x in row) for row in coords)}


def neb_endpoints(reaction: 'ARCReaction') -> tuple[dict, dict]:
    """
    The reactant and product endpoints of a NEB for ``reaction``, both in the reactants' atom order.

    One side must be a single molecule; it is folded into its near-attack conformation for the bonds it forms
    (:func:`near_attack`). A one-molecule product is folded the same way for the bonds it forms in the reverse
    reaction, and aligned onto the reactant by the atoms whose bonds do not change. A bimolecular side is built from
    the one-molecule side with :func:`place_fragments`.

    Raises:
        DependencyError: If neither side is a single molecule, or the reaction has no atom map.
    """
    atom_map = reaction.atom_map
    if atom_map is None:
        raise DependencyError(f'ase_neb needs an atom map for reaction {reaction.label}.')
    if not has_one_molecule_side(reaction):
        raise DependencyError(f'ase_neb builds its endpoints from a one-molecule side, and neither side of '
                              f'{reaction.label} is one molecule.')
    formed, broken = reaction.get_formed_and_broken_bonds()  # reactant atom indices
    reactive = {i for bond in formed + broken for i in bond}
    n_r, n_p = molecule_count(reaction, 0), molecule_count(reaction, 1)
    r_xyz = reaction.get_reactants_xyz(return_format=dict)
    if n_r == 1:
        r_xyz = near_attack(r_xyz, reaction.r_species[0], formed)
    if n_p == 1:
        # Fold the product in its own atom order (its bonds to form in reverse are the broken ones), then reorder.
        product = reaction.p_species[0]
        p_own = near_attack(product.get_xyz(), product, [(atom_map[i], atom_map[j]) for i, j in broken])
        p_xyz = sort_xyz_using_indices(xyz_dict=p_own, indices=atom_map)
    else:
        p_xyz = reaction.get_products_xyz(return_format=dict)
    r, p = np.array(r_xyz['coords'], dtype=float), np.array(p_xyz['coords'], dtype=float)
    symbols = tuple(r_xyz['symbols'])
    if n_r == 1 and n_p == 1:
        p = align_on(p, r, fit_atoms(list(range(len(symbols))), symbols, reactive))
    elif n_r == 1:
        p = place_fragments(joined=r, separated=p, fragments=side_fragments(reaction, 1), symbols=symbols,
                            reactive=reactive)
    else:
        r = place_fragments(joined=p, separated=r, fragments=side_fragments(reaction, 0), symbols=symbols,
                            reactive=reactive)
    as_xyz = lambda coords: {'symbols': symbols, 'isotopes': tuple(r_xyz['isotopes']),
                             'coords': tuple(tuple(float(x) for x in row) for row in coords)}
    return as_xyz(r), as_xyz(p)


def rmsd(coords_1, coords_2) -> float:
    """The root-mean-square distance between two N x 3 coordinate arrays (no alignment)."""
    return float(np.sqrt(((np.asarray(coords_1) - np.asarray(coords_2)) ** 2).sum(axis=1).mean()))


def candidate_endpoints(reaction: 'ARCReaction', n_bands: int = 3, distinct: float = 0.1) -> list[dict]:
    """
    Endpoints for up to ``n_bands`` bands, from different atom maps of ``reaction``.

    ARC's atom map is one of several equivalent ones (which methyl H migrates or leaves, which of two equivalent atoms
    goes where), and an unlucky one gives a band that crosses the wrong saddle or none, so that its guess optimizes to
    another saddle or slides into a well. Every map ARC can enumerate (:func:`arc.mapping.cluster.enumerate_atom_maps`,
    plus the reaction's own) is turned into endpoints with :func:`neb_endpoints`; for a hydrogen transfer, a second
    product endpoint is built from the reactant with :func:`moved_h_product`. The candidates are ranked by the
    distance between their two endpoints, a proxy for the length of the band. Candidates whose endpoints are within
    ``distinct`` A RMSD of a better one are dropped. Each band's guess is a separate TS guess, so ARC moves on to the
    next one if a guess fails its frequency or IRC check.

    Returns:
        list[dict]: Entries with 'xyz' (reactant), 'product_xyz', 'atom_map', 'moved_h' (whether the product endpoint
                    was built by :func:`moved_h_product`; at most one such band is kept, and it carries 'keep_bonds',
                    the (H, acceptor) pairs, for ase_script.py to relax that endpoint first) and 'path_rmsd', best
                    first.
    """
    original = reaction.atom_map
    maps = list()
    try:
        from arc.mapping.cluster import enumerate_atom_maps
        maps = [list(m) for m in enumerate_atom_maps(reaction) or list()]
    except Exception as e:  # enumeration is a bonus; the reaction's own map still makes a band
        logger.debug(f'Could not enumerate atom maps for {reaction.label}: {e}')
    if original is not None and list(original) not in maps:
        maps.insert(0, list(original))
    scored = list()
    try:
        for atom_map in maps:
            reaction.atom_map = atom_map
            try:
                r_xyz, p_xyz = neb_endpoints(reaction)
            except Exception as e:
                logger.debug(f'No NEB endpoints for {reaction.label} with atom map {atom_map}: {e}')
                continue
            variants = [(p_xyz, None)]
            if molecule_count(reaction, 0) == 1 and molecule_count(reaction, 1) == 1:
                formed, broken = reaction.get_formed_and_broken_bonds()
                transfers = h_transfers(reaction, formed, broken, tuple(r_xyz['symbols']))
                mol = reaction.r_species[0].mol
                neighbors = {i: {mol.atoms.index(n) for n in atom.edges} for i, atom in enumerate(mol.atoms)}
                moved = moved_h_product(r_xyz, transfers, neighbors) if transfers else None
                if moved is not None:
                    variants.append((moved, transfers))
            for variant, transfers in variants:
                candidate = {'xyz': r_xyz, 'product_xyz': variant, 'atom_map': list(atom_map),
                             'moved_h': transfers is not None, 'path_rmsd': rmsd(r_xyz['coords'], variant['coords'])}
                if transfers is not None:
                    candidate['keep_bonds'] = [list(pair) for pair in transfers]  # ase_script.py relaxes it first
                scored.append(candidate)
    finally:
        reaction.atom_map = original
    if not scored:
        raise DependencyError(f'ase_neb could not build NEB endpoints for {reaction.label}.')
    kept = list()
    for candidate in sorted(scored, key=lambda c: c['path_rmsd']):
        if candidate['moved_h'] and any(k['moved_h'] for k in kept):
            continue  # one band ending at a moved-H product; the others end at the product's own minimum
        if all(rmsd(candidate['xyz']['coords'], k['xyz']['coords']) > distinct
               or rmsd(candidate['product_xyz']['coords'], k['product_xyz']['coords']) > distinct for k in kept):
            kept.append(candidate)
        if len(kept) >= n_bands:
            break
    return kept


def resolve_level(level: Level | None) -> Level:
    """
    The level the band runs at: ``level`` (the TS-guess level the scheduler passes) if an ASE calculator runs it,
    else ``ase_neb_settings['level']``.

    Raises:
        DependencyError: If neither is an ASE level, so this run has nothing to relax a band on.
    """
    if level is not None and getattr(level, 'software', None) == 'ase':
        return level
    fallback = ase_neb_settings.get('level')
    if fallback:
        fallback = fallback if isinstance(fallback, Level) else Level(repr=fallback)
        if fallback.software == 'ase':
            return fallback
    raise DependencyError(f'ase_neb needs an ASE level (e.g., an MLIP) as the TS-guess level or as '
                          f"ase_neb_settings['level']; got {level}.")


class ASENEBAdapter(ASEAdapter):
    """
    A class for TS guesses from an ASE nudged elastic band. Takes the same arguments as :class:`ASEAdapter`, and needs
    ``reactions`` (one reaction).
    """

    def __init__(self,
                 project: str,
                 project_directory: str,
                 job_type: list[str] | str,
                 args: dict | None = None,
                 level: Level | None = None,
                 reactions: list['ARCReaction'] | None = None,
                 species: list[ARCSpecies] | None = None,
                 **kwargs,
                 ):
        if not reactions:
            raise ValueError('Cannot run an ASE NEB without an ARCReaction object.')
        level = resolve_level(level)
        reaction = reactions[0]
        if not has_one_molecule_side(reaction):
            raise DependencyError(f'ase_neb builds its endpoints from a one-molecule side, and neither side of '
                                  f'{reaction.label} is one molecule.')
        if reaction.ts_species is None:
            # A placeholder TS species, as the Orca NEB adapter makes: the job needs a species, and its guesses go to
            # reaction.ts_species.
            reaction.ts_species = ARCSpecies(label=reaction.ts_label or f'{reaction.label}_TS', is_ts=True,
                                             charge=reaction.charge, multiplicity=reaction.multiplicity,
                                             xyz=reaction.r_species[0].get_xyz())
        # The level's args (e.g., the calculator's python and queue), under any given explicitly. The scheduler
        # already merges them for the level it passes, but not for a level from ase_neb_settings.
        args = dict(args or dict())
        for key, value in (level.get_args() if level.args else dict()).items():
            args[key] = {**value, **(args.get(key) or dict())} if isinstance(value, dict) else args.get(key, value)
        super().__init__(project=project,
                         project_directory=project_directory,
                         job_type=job_type,
                         args=args,
                         level=level,
                         reactions=reactions,
                         species=[reaction.ts_species],
                         **kwargs,
                         )
        self.job_adapter = 'ase_neb'
        self.url = 'https://ase-lib.org/ase/neb.html'

    def write_input_file(self) -> None:
        """
        Write input.yml for ase_script.py: a 'neb' job with one band per atom map (see :func:`candidate_endpoints`;
        ``ase_neb_settings['keyword']['neb_bands']`` of them, 3 by default), the level's ASE settings, and the band
        settings of ``ase_neb_settings['keyword']``.
        """
        reaction = self.reactions[0]
        run_settings = self.determine_settings()
        for key, value in (ase_neb_settings.get('keyword') or dict()).items():
            run_settings.setdefault(key, value)
        bands = candidate_endpoints(reaction, n_bands=int(run_settings.pop('neb_bands', 3)))
        input_dict = {'job_type': 'neb',
                      'xyz': bands[0]['xyz'],
                      'product_xyz': bands[0]['product_xyz'],
                      'bands': bands,
                      'charge': reaction.charge,
                      'multiplicity': reaction.multiplicity,
                      'is_ts': False,
                      'settings': run_settings,
                      }
        save_yaml_file(os.path.join(self.local_path, 'input.yml'), input_dict)


register_job_adapter('ase_neb', ASENEBAdapter)
