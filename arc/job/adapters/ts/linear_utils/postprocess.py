"""
Validation filters, geometry fixers, and family-specific post-processing handlers
extracted from ``arc.job.adapters.ts.linear``.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from arc.common import get_logger, get_single_bond_length
from arc.species.species import colliding_atoms
from arc.species.zmat import xyz_to_zmat

from arc.job.adapters.ts.linear_utils.geom_utils import (
    dihedral_deg as _dihedral_deg,
    rotate_atoms as _rotate_atoms,
)

if TYPE_CHECKING:
    from arc.molecule import Molecule


logger = get_logger()

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FamilyPostprocessor = Callable[
    [dict, 'Molecule', List[Tuple[int, int]], List[Tuple[int, int]], Optional[Dict[str, int]]],
    Tuple[dict, Set[int]],
]
FamilyValidator = Callable[
    [dict, Set[int], List[Tuple[int, int]], 'Molecule', str],
    Tuple[bool, str],
]

# Maps family names → handler functions.  Populated after the handler functions
# are defined (see "Family handlers" section below).  Multiple families may
# point to the same handler when they share identical post-processing logic.
_FAMILY_POSTPROCESSORS: Dict[str, FamilyPostprocessor] = {}
_FAMILY_VALIDATORS: Dict[str, FamilyValidator] = {}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Equilibrium single-bond lengths to H, used in TS distance estimation.
_EQ_BOND_TO_H: Dict[str, float] = {
    'C': get_single_bond_length('C', 'H'),
    'N': get_single_bond_length('N', 'H'),
    'O': get_single_bond_length('O', 'H'),
    'S': get_single_bond_length('S', 'H'),
    'P': get_single_bond_length('P', 'H'),
    'Si': get_single_bond_length('Si', 'H'),
}
_EQ_BOND_TO_H_DEFAULT: float = 1.09

# At a symmetric TS (bond order n = 0.5), Pauling's equation gives
# d_TS = d0 - 0.6 * ln(n) = d0 + 0.6 * ln(2) ≈ d0 + 0.42 Å.
_PAULING_DELTA: float = 0.42


# ---------------------------------------------------------------------------
# Rejection filters
# ---------------------------------------------------------------------------

def _has_detached_hydrogen(xyz: dict,
                           max_h_heavy_dist: float = 3.0,
                           exempt_indices: Optional[Set[int]] = None,
                           ) -> bool:
    """
    Return ``True`` if any hydrogen in *xyz* is farther than *max_h_heavy_dist* Å
    from every heavy atom. Such geometries indicate that a C–H (or N–H / O–H)
    bond has been unphysically stretched during Z-matrix interpolation.

    The threshold is intentionally generous (3.0 Å) so that only completely
    unphysical detachments are caught while partial-bond distances in TS
    geometries (where H is between two heavy atoms) remain below the cut-off.

    Args:
        xyz (dict): XYZ coordinate dictionary.
        max_h_heavy_dist (float): Maximum acceptable H–heavy distance in Å. Default: 3.0.
        exempt_indices (Set[int], optional): Atom indices to skip (e.g. migrating H atoms
            that are intentionally between two heavy atoms). Default: None.

    Returns:
        bool: ``True`` if a detached hydrogen is detected.
    """
    symbols = xyz['symbols']
    coords_arr = np.array(xyz['coords'], dtype=float)
    heavy_coords = coords_arr[[i for i, s in enumerate(symbols) if s != 'H']]
    if not len(heavy_coords):
        return False
    for i, sym in enumerate(symbols):
        if sym == 'H' and (exempt_indices is None or i not in exempt_indices):
            if np.linalg.norm(heavy_coords - coords_arr[i], axis=1).min() > max_h_heavy_dist:
                return True
    return False


def _has_too_many_fragments(xyz: dict,
                            max_heavy_heavy: float = 2.0,
                            max_heavy_h: float = 1.5,
                            ) -> bool:
    """
    Return ``True`` if the geometry has three or more distance-based
    fragments.

    A transition state may legitimately have two fragments (e.g. a
    migrating group between two heavy-atom centres with stretched bonds),
    but three or more fragments indicates a failed interpolation where
    atoms have drifted into space.

    Args:
        xyz (dict): XYZ coordinate dictionary.
        max_heavy_heavy (float): Maximum distance (Å) for two heavy atoms
            to be considered bonded.  Default 2.0 Å.
        max_heavy_h (float): Maximum distance (Å) for a heavy–H bond.
            Default 1.5 Å.

    Returns:
        bool: ``True`` if the geometry has 3+ fragments.
    """
    symbols = xyz['symbols']
    coords_arr = np.array(xyz['coords'], dtype=float)
    n = len(symbols)
    if n <= 1:
        return False
    adj: List[Set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = symbols[i], symbols[j]
            if si == 'H' and sj == 'H':
                continue
            thresh = max_heavy_h if (si == 'H' or sj == 'H') else max_heavy_heavy
            if np.linalg.norm(coords_arr[i] - coords_arr[j]) < thresh:
                adj[i].add(j)
                adj[j].add(i)
    visited: Set[int] = set()
    num_components = 0
    for start in range(n):
        if start in visited:
            continue
        num_components += 1
        if num_components >= 3:
            return True
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nbr in adj[node]:
                if nbr not in visited:
                    stack.append(nbr)
    return False


def _has_h_close_contact(xyz: dict, threshold: float = 0.85) -> bool:
    """
    Return ``True`` if any atom pair involving at least one hydrogen is closer
    than *threshold* × the single-bond length for that element pair.

    This is a tighter version of :func:`colliding_atoms` (which uses 0.60)
    specifically targeting hydrogen-related close contacts that indicate a
    broken TS geometry.  Heavy-heavy pairs are not checked here since partial
    bonds in cyclic TS geometries can legitimately be shorter than the
    equilibrium single-bond length.

    Args:
        xyz (dict): XYZ coordinate dictionary.
        threshold (float): Fraction of single-bond length below which a pair
            is considered a close contact.  Default: 0.85.

    Returns:
        bool: ``True`` if a close contact is detected.
    """
    symbols = xyz['symbols']
    coords_arr = np.array(xyz['coords'], dtype=float)
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            if symbols[i] != 'H' and symbols[j] != 'H':
                continue
            d = float(np.linalg.norm(coords_arr[i] - coords_arr[j]))
            sbl = get_single_bond_length(symbols[i], symbols[j])
            if d < sbl * threshold:
                return True
    return False


def _has_misoriented_migrating_h(xyz: dict, forming_bonds: list, mol: 'Molecule') -> bool:
    """
    Return ``True`` if any migrating hydrogen is closer to an H on the
    acceptor atom than to the acceptor itself.

    This catches TS guesses where a terminal-group rotor (e.g. CH₂, CH₃) on
    the acceptor is oriented so that one of its H atoms blocks the incoming
    migrating H, wasting downstream DFT resources on a wrong guess.

    Only H atoms bonded to the acceptor in *mol* (the reactant topology) are
    considered, so this check is specific to rotor-orientation problems and
    does not affect reactions where the acceptor carries no hydrogens.

    Args:
        xyz (dict): XYZ coordinate dictionary.
        forming_bonds (list): List of (i, j) index pairs for bonds that form
            in the product.
        mol (Molecule): Reactant molecule providing bond topology.

    Returns:
        bool: ``True`` if a misoriented migrating H is detected.
    """
    symbols = xyz['symbols']
    coords_arr = np.array(xyz['coords'], dtype=float)
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    for i_atom, j_atom in forming_bonds:
        if symbols[i_atom] == 'H':
            h_idx, acceptor_idx = i_atom, j_atom
        elif symbols[j_atom] == 'H':
            h_idx, acceptor_idx = j_atom, i_atom
        else:
            continue
        acceptor_dist = float(np.linalg.norm(coords_arr[h_idx] - coords_arr[acceptor_idx]))
        for nbr in mol.atoms[acceptor_idx].bonds.keys():
            nbr_idx = atom_to_idx[nbr]
            if symbols[nbr_idx] != 'H' or nbr_idx == h_idx:
                continue
            hh_dist = float(np.linalg.norm(coords_arr[h_idx] - coords_arr[nbr_idx]))
            if hh_dist < acceptor_dist:
                return True
    return False


def _has_migrating_h_nearer_to_nonreactive(xyz: dict,
                                            forming_bonds: list,
                                            mol: 'Molecule',
                                            margin: float = 0.3,
                                            ) -> bool:
    """
    Return ``True`` if any migrating hydrogen is *significantly* closer to a
    non-reactive heavy atom than to both of its reactive heavy atoms.

    In a well-formed H-migration TS the migrating H sits between donor and
    acceptor.  When Z-matrix interpolation leaves the H bunched with
    another H on the same group, the distance to a non-reactive atom is
    much shorter than to the reactive pair, and this check rejects the
    guess.

    A ``margin`` (default 0.3 Å) prevents false rejections in compact ring
    topologies where non-reactive ring atoms are only slightly closer than
    the reactive sites.  Only non-reactive **heavy** atoms are considered
    (other H atoms are ignored since they can legitimately be close).

    Args:
        xyz: XYZ coordinate dictionary.
        forming_bonds: List of (i, j) index pairs for bonds that form.
        mol: Reactant molecule providing bond topology.
        margin: Distance margin in Å. A non-reactive heavy atom must be
            closer than ``min(d_donor, d_acceptor) - margin`` to trigger
            rejection.

    Returns:
        ``True`` if a migrating H is much nearer to a non-reactive heavy atom.
    """
    symbols = xyz['symbols']
    coords_arr = np.array(xyz['coords'], dtype=float)
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    for i_atom, j_atom in forming_bonds:
        if symbols[i_atom] == 'H':
            h_idx, acceptor_idx = i_atom, j_atom
        elif symbols[j_atom] == 'H':
            h_idx, acceptor_idx = j_atom, i_atom
        else:
            continue
        donor_idx = None
        for nbr in mol.atoms[h_idx].bonds.keys():
            nbr_idx = atom_to_idx[nbr]
            if symbols[nbr_idx] != 'H':
                donor_idx = nbr_idx
                break
        if donor_idx is None:
            continue
        d_donor = float(np.linalg.norm(coords_arr[h_idx] - coords_arr[donor_idx]))
        d_acceptor = float(np.linalg.norm(coords_arr[h_idx] - coords_arr[acceptor_idx]))
        threshold = min(d_donor, d_acceptor) - margin
        for k in range(len(symbols)):
            if k in (h_idx, donor_idx, acceptor_idx) or symbols[k] == 'H':
                continue
            d_k = float(np.linalg.norm(coords_arr[h_idx] - coords_arr[k]))
            if d_k < threshold:
                return True
    return False


def _has_bad_ts_motif(xyz: dict,
                      forming_bonds: list,
                      mol: 'Molecule',
                      d_xh_range: Tuple[float, float] = (1.0, 2.2),
                      d_da_range: Tuple[float, float] = (2.0, 3.8),
                      ) -> bool:
    """
    Return ``True`` if the donor–H–acceptor distances are outside reasonable
    TS windows for an H-migration reaction.

    Checks three distances for each forming bond involving hydrogen:

    * **d(D, H)**: must be within *d_xh_range* (default 1.0–2.2 Å).
    * **d(A, H)**: must be within *d_xh_range*.
    * **d(D, A)**: must be within *d_da_range* (default 2.0–3.8 Å).

    Args:
        xyz: XYZ coordinate dictionary.
        forming_bonds: List of (i, j) index pairs for bonds that form.
        mol: Reactant molecule providing bond topology.
        d_xh_range: Allowed (min, max) for donor–H and acceptor–H distances.
        d_da_range: Allowed (min, max) for donor–acceptor distance.

    Returns:
        ``True`` if any distance is outside its window.
    """
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    symbols = xyz['symbols']
    coords_arr = np.array(xyz['coords'], dtype=float)
    for i_atom, j_atom in forming_bonds:
        if symbols[i_atom] == 'H':
            h_idx, acceptor_idx = i_atom, j_atom
        elif symbols[j_atom] == 'H':
            h_idx, acceptor_idx = j_atom, i_atom
        else:
            continue
        donor_idx = None
        for nbr in mol.atoms[h_idx].bonds.keys():
            nbr_idx = atom_to_idx[nbr]
            if symbols[nbr_idx] != 'H':
                donor_idx = nbr_idx
                break
        if donor_idx is None:
            continue
        d_dh = float(np.linalg.norm(coords_arr[donor_idx] - coords_arr[h_idx]))
        d_ah = float(np.linalg.norm(coords_arr[acceptor_idx] - coords_arr[h_idx]))
        d_da = float(np.linalg.norm(coords_arr[donor_idx] - coords_arr[acceptor_idx]))
        if not (d_xh_range[0] <= d_dh <= d_xh_range[1]):
            return True
        if not (d_xh_range[0] <= d_ah <= d_xh_range[1]):
            return True
        donor_atom = mol.atoms[donor_idx]
        acceptor_atom = mol.atoms[acceptor_idx]
        donor_acceptor_bonded = acceptor_atom in donor_atom.bonds
        if not donor_acceptor_bonded and not (d_da_range[0] <= d_da <= d_da_range[1]):
            return True
    return False


# ---------------------------------------------------------------------------
# Geometry fixers
# ---------------------------------------------------------------------------

def fix_forming_bond_distances(xyz: dict,
                                mol: 'Molecule',
                                bonds: List[Tuple[int, int]],
                                ) -> dict:
    """
    Place each migrating hydrogen at its ideal TS position by triangulation.

    For each forming bond that involves hydrogen the function identifies the
    *donor* (heavy atom bonded to H in the reactant) and the *acceptor* (the
    other atom in the forming bond).  Target distances are computed with the
    Pauling bond-order equation at n = 0.5:

        d_TS(X–H) = d0(X–H) + 0.42 Å

    The hydrogen is placed at the intersection of the two spheres centred on
    the donor and acceptor with the respective target radii.  Among the
    resulting circle of solutions the point closest to the *current* H
    position is chosen, preserving the approach direction from the
    interpolation.  This produces a chemically reasonable (non-collinear)
    D–H–A transfer geometry.

    When the donor–acceptor distance is too large for the spheres to overlap
    the function falls back to collinear placement at d_DH from the donor
    along the D→A vector.

    Args:
        xyz: TS guess XYZ coordinate dictionary.
        mol: Reactant molecule providing bond topology to identify the donor.
        bonds: Forming bonds in reactant atom ordering.

    Returns:
        A new XYZ dictionary with migrating H atoms repositioned.
    """
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    symbols = xyz['symbols']
    coords = list(xyz['coords'])

    for (i_atom, j_atom) in bonds:
        if symbols[i_atom] == 'H':
            h_atom, acceptor_atom = i_atom, j_atom
        elif symbols[j_atom] == 'H':
            h_atom, acceptor_atom = j_atom, i_atom
        else:
            continue
        donor_idx = None
        for nbr in mol.atoms[h_atom].bonds.keys():
            nbr_idx = atom_to_idx[nbr]
            if symbols[nbr_idx] != 'H':
                donor_idx = nbr_idx
                break
        if donor_idx is None:
            continue

        donor_sym = symbols[donor_idx]
        acceptor_sym = symbols[acceptor_atom]
        d_DH = _EQ_BOND_TO_H.get(donor_sym, _EQ_BOND_TO_H_DEFAULT) + _PAULING_DELTA
        d_AH = _EQ_BOND_TO_H.get(acceptor_sym, _EQ_BOND_TO_H_DEFAULT) + _PAULING_DELTA

        d_pos = np.array(coords[donor_idx], dtype=float)
        a_pos = np.array(coords[acceptor_atom], dtype=float)
        h_pos = np.array(coords[h_atom], dtype=float)
        da_vec = a_pos - d_pos
        da_dist = float(np.linalg.norm(da_vec))
        if da_dist < 1e-8:
            continue
        da_unit = da_vec / da_dist

        if da_dist <= d_DH + d_AH:
            x = (da_dist ** 2 + d_DH ** 2 - d_AH ** 2) / (2.0 * da_dist)
            h_sq = d_DH ** 2 - x ** 2
            h_perp = np.sqrt(max(h_sq, 0.0))

            proj = d_pos + np.dot(h_pos - d_pos, da_unit) * da_unit
            perp = h_pos - proj
            perp_norm = float(np.linalg.norm(perp))
            if perp_norm > 1e-8:
                n_perp = perp / perp_norm
            else:
                ref = np.array([1.0, 0.0, 0.0]) if abs(da_unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                n_perp = np.cross(da_unit, ref)
                n_perp /= np.linalg.norm(n_perp)

            cand_plus = d_pos + x * da_unit + h_perp * n_perp
            cand_minus = d_pos + x * da_unit - h_perp * n_perp

            def _min_backbone_dist(pos):
                md = float('inf')
                for k_idx, sym_k in enumerate(symbols):
                    if k_idx in (h_atom, donor_idx, acceptor_atom) or sym_k == 'H':
                        continue
                    md = min(md, float(np.linalg.norm(pos - np.array(coords[k_idx], dtype=float))))
                return md

            new_h_pos = cand_plus if _min_backbone_dist(cand_plus) >= _min_backbone_dist(cand_minus) \
                else cand_minus

            for k_idx, sym_k in enumerate(symbols):
                if k_idx in (h_atom, donor_idx, acceptor_atom) or sym_k == 'H':
                    continue
                b_pos = np.array(coords[k_idx], dtype=float)
                bh_vec = new_h_pos - b_pos
                bh_dist = float(np.linalg.norm(bh_vec))
                sbl_bh = get_single_bond_length('H', sym_k)
                min_dist = max(sbl_bh * 1.5, d_AH + 0.3)
                if bh_dist < min_dist and bh_dist > 1e-8:
                    new_h_pos = b_pos + min_dist * (bh_vec / bh_dist)
        else:
            new_h_pos = d_pos + d_DH * da_unit

        clash = False
        for k, sym_k in enumerate(symbols):
            if k == h_atom or k == donor_idx or sym_k == 'H':
                continue
            if np.linalg.norm(np.array(coords[k], dtype=float) - new_h_pos) < 1.2:
                clash = True
                break
        if not clash:
            coords[h_atom] = tuple(float(v) for v in new_h_pos)

    new_xyz = dict(xyz)
    new_xyz['coords'] = tuple(tuple(float(v) for v in row) for row in coords)
    return new_xyz


def fix_nonreactive_h_distances(xyz: dict,
                                mol: 'Molecule',
                                migrating_h_indices: Set[int],
                                ) -> dict:
    """
    Push displaced non-migrating hydrogens back to their equilibrium bond length.

    For each hydrogen that is NOT in *migrating_h_indices*, find its bonded heavy atom
    in the reactant topology (*mol*) and, if the current distance deviates from the
    equilibrium single-bond length by more than 5 % (too far or too close), project
    the hydrogen back to the equilibrium distance along the same direction.
    The heavy-atom position is not changed.

    Args:
        xyz (dict): TS guess XYZ coordinate dictionary.
        mol (Molecule): Reactant molecule providing bond topology.
        migrating_h_indices (Set[int]): Atom indices of migrating hydrogen(s) to skip.

    Returns:
        dict: A new XYZ dictionary with non-migrating H atoms repositioned.
    """
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    symbols = xyz['symbols']
    coords = list(xyz['coords'])

    for i, sym in enumerate(symbols):
        if sym != 'H' or i in migrating_h_indices:
            continue
        bonded_heavy_idx = None
        for nbr in mol.atoms[i].bonds.keys():
            nbr_idx = atom_to_idx[nbr]
            if symbols[nbr_idx] != 'H':
                bonded_heavy_idx = nbr_idx
                break
        if bonded_heavy_idx is None:
            continue

        heavy_sym = symbols[bonded_heavy_idx]
        sbl = get_single_bond_length('H', heavy_sym)

        h_pos = np.array(coords[i], dtype=float)
        heavy_pos = np.array(coords[bonded_heavy_idx], dtype=float)
        vec = h_pos - heavy_pos
        dist = np.linalg.norm(vec)
        if dist < 1e-8:
            continue

        if dist > sbl * 1.05 or dist < sbl * 0.95:
            new_h_pos = heavy_pos + sbl * (vec / dist)
            coords[i] = tuple(float(v) for v in new_h_pos)

    new_xyz = dict(xyz)
    new_xyz['coords'] = tuple(tuple(float(v) for v in row) for row in coords)
    return new_xyz


def fix_crowded_h_atoms(xyz: dict,
                        mol: 'Molecule',
                        skip_h_indices: Optional[Set[int]] = None,
                        redistribute_ch2: bool = False,
                        ) -> dict:
    """
    Redistribute H atoms around a heavy atom when they are too close together.

    Z-matrix interpolation can bunch multiple H atoms on the same side of their
    bonded heavy atom, especially during group migration (e.g. a CH₃ 1,2-shift).
    ``fix_nonreactive_h_distances`` corrects radial distances but not angular
    positions.  This function detects heavy atoms whose bonded H atoms are closer
    than 1.3 Å to each other and redistributes them at evenly spaced dihedral
    angles around the heavy-atom → backbone-neighbour axis, preserving the
    equilibrium bond length and a tetrahedral (or appropriate) valence angle.

    Args:
        xyz: TS guess XYZ coordinate dictionary.
        mol: Reactant molecule providing bond topology.
        skip_h_indices: H atom indices to exclude (e.g. migrating H's whose
            positions were set by ``fix_forming_bond_distances``).
        redistribute_ch2: When ``True``, also redistribute CH₂ groups
            (heavy atoms with exactly 2 non-H neighbours).  Used for
            families where hybridisation changes (e.g. sp → sp2) make
            Z-matrix interpolation of CH₂ H positions unreliable.

    Returns:
        A new XYZ dictionary with redistributed H atoms.
    """
    if skip_h_indices is None:
        skip_h_indices = set()
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    symbols = xyz['symbols']
    coords = np.array(xyz['coords'], dtype=float)

    heavy_to_hs: Dict[int, List[int]] = {}
    for i, sym in enumerate(symbols):
        if sym == 'H' and i not in skip_h_indices:
            for nbr in mol.atoms[i].bonds.keys():
                nbr_idx = atom_to_idx[nbr]
                if symbols[nbr_idx] != 'H':
                    heavy_to_hs.setdefault(nbr_idx, []).append(i)
                    break

    for heavy_idx, h_indices in heavy_to_hs.items():
        if len(h_indices) < 2:
            continue

        non_h_nbr_indices: List[int] = []
        for nbr in mol.atoms[heavy_idx].bonds.keys():
            nbr_idx = atom_to_idx[nbr]
            if symbols[nbr_idx] != 'H':
                non_h_nbr_indices.append(nbr_idx)
        if len(non_h_nbr_indices) == 0 or len(non_h_nbr_indices) > 2:
            continue

        if len(non_h_nbr_indices) == 2:
            if not redistribute_ch2:
                continue
        else:
            crowded = False
            for a_i in range(len(h_indices)):
                for b_i in range(a_i + 1, len(h_indices)):
                    if np.linalg.norm(coords[h_indices[a_i]] - coords[h_indices[b_i]]) < 1.3:
                        crowded = True
                        break
                if crowded:
                    break
            if not crowded:
                continue

        heavy_pos = coords[heavy_idx]
        sbl = get_single_bond_length('H', symbols[heavy_idx])

        if len(non_h_nbr_indices) == 2:
            vec1 = coords[non_h_nbr_indices[0]] - heavy_pos
            vec2 = coords[non_h_nbr_indices[1]] - heavy_pos
            d1, d2 = float(np.linalg.norm(vec1)), float(np.linalg.norm(vec2))
            if d1 < 1e-8 or d2 < 1e-8:
                continue
            u1, u2 = vec1 / d1, vec2 / d2

            normal = np.cross(u1, u2)
            normal_len = float(np.linalg.norm(normal))
            if normal_len < 1e-8:
                ref = np.array([1.0, 0.0, 0.0]) if abs(u1[0]) < 0.9 \
                    else np.array([0.0, 1.0, 0.0])
                normal = np.cross(u1, ref)
                normal /= np.linalg.norm(normal)
            else:
                normal /= normal_len

            bisector = u1 + u2
            bisector_len = float(np.linalg.norm(bisector))
            if bisector_len < 1e-8:
                coords[h_indices[0]] = heavy_pos + sbl * normal
                if len(h_indices) > 1:
                    coords[h_indices[1]] = heavy_pos - sbl * normal
                continue
            anti_bisector = -bisector / bisector_len

            cos_bb = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
            beta = np.arccos(cos_bb) / 2.0
            cos_beta = np.cos(beta)
            if cos_beta > 1e-6:
                cos_psi = -np.cos(np.radians(109.47)) / cos_beta
                cos_psi = float(np.clip(cos_psi, -1.0, 1.0))
                psi = np.arccos(cos_psi)
            else:
                psi = np.pi / 2.0

            h_dir_plus = np.cos(psi) * anti_bisector + np.sin(psi) * normal
            h_dir_minus = np.cos(psi) * anti_bisector - np.sin(psi) * normal
            coords[h_indices[0]] = heavy_pos + sbl * h_dir_plus
            if len(h_indices) > 1:
                coords[h_indices[1]] = heavy_pos + sbl * h_dir_minus

        else:
            backbone_idx = non_h_nbr_indices[0]
            axis = heavy_pos - coords[backbone_idx]
            axis_len = float(np.linalg.norm(axis))
            if axis_len < 1e-8:
                continue
            axis = axis / axis_len

            ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 \
                else np.array([0.0, 1.0, 0.0])
            perp1 = np.cross(axis, ref)
            perp1 /= np.linalg.norm(perp1)
            perp2 = np.cross(axis, perp1)

            h0_vec = coords[h_indices[0]] - heavy_pos
            h0_proj = h0_vec - np.dot(h0_vec, axis) * axis
            h0_proj_norm = float(np.linalg.norm(h0_proj))
            start_angle = np.arctan2(
                np.dot(h0_proj, perp2), np.dot(h0_proj, perp1)) \
                if h0_proj_norm > 1e-8 else 0.0

            tet_angle = np.radians(109.47)

            n_h = len(h_indices)
            for k, h_idx in enumerate(h_indices):
                dih = start_angle + 2.0 * np.pi * k / n_h
                h_dir = (np.cos(tet_angle) * axis
                         + np.sin(tet_angle) * (np.cos(dih) * perp1
                                                + np.sin(dih) * perp2))
                coords[h_idx] = heavy_pos + sbl * h_dir

    new_xyz = dict(xyz)
    new_xyz['coords'] = tuple(tuple(float(v) for v in row) for row in coords)
    return new_xyz


def fix_h_nonbonded_clashes(xyz: dict,
                            mol: 'Molecule',
                            skip_h_indices: Optional[Set[int]] = None,
                            threshold: float = 1.10,
                            ) -> dict:
    """
    Fix H atoms that are unreasonably close to non-bonded heavy atoms.

    Z-matrix interpolation can misplace H atoms angularly so they end up
    closer to a neighbouring heavy atom than any physically reasonable
    distance (e.g. a ring H pushed into the ring plane).  This function
    detects such clashes and repositions the offending H perpendicular to
    the plane of its bonded heavy atom's backbone neighbours, preserving
    the equilibrium bond length.

    A default threshold of 1.10 Å catches H atoms that are essentially
    at bonding distance from a non-bonded heavy atom.  Migrating H atoms
    (which may legitimately sit between two heavy atoms) should be passed
    via ``skip_h_indices``.

    Args:
        xyz: TS guess XYZ coordinate dictionary.
        mol: Reactant molecule providing bond topology.
        skip_h_indices: H atom indices to exclude (e.g. migrating H's).
        threshold: Maximum H–(non-bonded-heavy) distance in Å below which
            a clash is declared.  Default 1.10 Å.

    Returns:
        A new XYZ dictionary with clashing H atoms repositioned.
    """
    skip = skip_h_indices or set()
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    symbols = xyz['symbols']
    coords = np.array(xyz['coords'], dtype=float)

    bonded_neighbours: Dict[int, Set[int]] = {}
    for atom in mol.atoms:
        idx = atom_to_idx[atom]
        bonded_neighbours[idx] = {atom_to_idx[nbr] for nbr in atom.bonds}

    for i, sym in enumerate(symbols):
        if sym != 'H' or i in skip:
            continue
        bonded_heavy = None
        for nbr in mol.atoms[i].bonds:
            n = atom_to_idx[nbr]
            if symbols[n] != 'H':
                bonded_heavy = n
                break
        if bonded_heavy is None:
            continue

        sbl = get_single_bond_length('H', symbols[bonded_heavy])
        heavy_pos = coords[bonded_heavy]
        h_pos = coords[i]

        has_clash = False
        for j, jsym in enumerate(symbols):
            if jsym == 'H' or j == bonded_heavy or j in bonded_neighbours.get(i, set()):
                continue
            if float(np.linalg.norm(h_pos - coords[j])) < threshold:
                has_clash = True
                break
        if not has_clash:
            continue

        non_h_nbrs = [atom_to_idx[nbr] for nbr in mol.atoms[bonded_heavy].bonds
                       if symbols[atom_to_idx[nbr]] != 'H']
        if len(non_h_nbrs) >= 2:
            v1 = coords[non_h_nbrs[0]] - heavy_pos
            v2 = coords[non_h_nbrs[1]] - heavy_pos
            normal = np.cross(v1, v2)
            normal_len = float(np.linalg.norm(normal))
            if normal_len > 1e-8:
                normal /= normal_len
            else:
                ref = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.9 \
                    else np.array([0.0, 1.0, 0.0])
                normal = np.cross(v1 / max(float(np.linalg.norm(v1)), 1e-8), ref)
                normal /= max(float(np.linalg.norm(normal)), 1e-8)
            h_vec = h_pos - heavy_pos
            sign = 1.0 if float(np.dot(h_vec, normal)) >= 0 else -1.0
            coords[i] = heavy_pos + sbl * sign * normal
        elif len(non_h_nbrs) == 1:
            axis = heavy_pos - coords[non_h_nbrs[0]]
            axis_len = float(np.linalg.norm(axis))
            if axis_len < 1e-8:
                continue
            axis /= axis_len
            h_vec = h_pos - heavy_pos
            h_perp = h_vec - np.dot(h_vec, axis) * axis
            h_perp_len = float(np.linalg.norm(h_perp))
            if h_perp_len > 1e-8:
                h_perp /= h_perp_len
            else:
                ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 \
                    else np.array([0.0, 1.0, 0.0])
                h_perp = np.cross(axis, ref)
                h_perp /= max(float(np.linalg.norm(h_perp)), 1e-8)
            rot_perp = np.cross(axis, h_perp)
            best_pos = None
            best_min_dist = -1.0
            tet = np.radians(109.47)
            for angle_offset in (np.pi / 3, 2 * np.pi / 3, np.pi,
                                 4 * np.pi / 3, 5 * np.pi / 3):
                candidate_dir = (np.cos(tet) * axis
                                 + np.sin(tet) * (np.cos(angle_offset) * h_perp
                                                   + np.sin(angle_offset) * rot_perp))
                candidate = heavy_pos + sbl * candidate_dir
                min_d = min(float(np.linalg.norm(candidate - coords[j]))
                            for j, js in enumerate(symbols)
                            if js != 'H' and j != bonded_heavy)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_pos = candidate
            if best_pos is not None:
                coords[i] = best_pos

    new_xyz = dict(xyz)
    new_xyz['coords'] = tuple(tuple(float(v) for v in row) for row in coords)
    return new_xyz


def _get_migrating_group_info(xyz: dict,
                              mol: 'Molecule',
                              breaking_bonds: List[Tuple[int, int]],
                              forming_bonds: List[Tuple[int, int]],
                              ) -> List[dict]:
    """
    Identify migrating heavy atoms and their non-reactive H substituents.

    A migrating heavy atom is one that appears in both a breaking and a forming
    bond (e.g., the C of a CH₃ group undergoing a 1,2-shift).  For each such
    atom we also collect the backbone partners (donor, acceptor) and the
    non-reactive H indices.

    Args:
        xyz: XYZ coordinate dictionary.
        mol: Reactant molecule providing bond topology.
        breaking_bonds: Bonds being broken ``[(i, j), ...]``.
        forming_bonds: Bonds being formed ``[(i, j), ...]``.

    Returns:
        A list of dicts, one per migrating heavy atom, with keys:
        ``'mig_idx'``, ``'h_indices'``, ``'backbone_indices'``.
        Empty list if no migrating heavy atom with non-reactive H's exists.
    """
    symbols = xyz['symbols']
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}

    bb_atoms: Set[int] = set()
    for a, b in breaking_bonds:
        bb_atoms.update((a, b))
    fb_atoms: Set[int] = set()
    for a, b in forming_bonds:
        fb_atoms.update((a, b))
    reactive_set = bb_atoms | fb_atoms
    migrating = bb_atoms & fb_atoms

    results: List[dict] = []
    for mig_idx in migrating:
        if symbols[mig_idx] == 'H':
            continue
        h_indices: List[int] = []
        for nbr in mol.atoms[mig_idx].bonds:
            n_idx = atom_to_idx[nbr]
            if symbols[n_idx] == 'H' and n_idx not in reactive_set:
                h_indices.append(n_idx)
        if not h_indices:
            continue
        backbone_indices: Set[int] = set()
        for a, b in breaking_bonds:
            other = b if a == mig_idx else (a if b == mig_idx else None)
            if other is not None:
                backbone_indices.add(other)
        for a, b in forming_bonds:
            other = b if a == mig_idx else (a if b == mig_idx else None)
            if other is not None:
                backbone_indices.add(other)
        if backbone_indices:
            results.append({'mig_idx': mig_idx,
                            'h_indices': h_indices,
                            'backbone_indices': backbone_indices})
    return results


def has_inward_migrating_group_h(xyz: dict,
                                 mol: 'Molecule',
                                 breaking_bonds: List[Tuple[int, int]],
                                 forming_bonds: List[Tuple[int, int]],
                                 ) -> bool:
    """
    Check whether any non-reactive H on a migrating heavy atom points toward the backbone.

    In a group-migration TS (e.g., 1,2-shift of CH₃), the migrating heavy atom
    sits between two backbone atoms (donor and acceptor).  Its non-reactive H
    substituents should point **away** from the backbone.  This function returns
    ``True`` if the average H direction from the migrating atom has a positive
    projection onto the backbone centroid direction — i.e., the H's point inward.

    Args:
        xyz: XYZ coordinate dictionary.
        mol: Reactant molecule providing bond topology.
        breaking_bonds: Bonds being broken ``[(i, j), ...]``.
        forming_bonds: Bonds being formed ``[(i, j), ...]``.

    Returns:
        ``True`` if any migrating group has inward-pointing H substituents.
    """
    coords = np.array(xyz['coords'], dtype=float)
    for info in _get_migrating_group_info(xyz, mol, breaking_bonds, forming_bonds):
        mig_pos = coords[info['mig_idx']]
        backbone_centroid = np.mean([coords[i] for i in info['backbone_indices']], axis=0)
        to_backbone = backbone_centroid - mig_pos
        tb_len = float(np.linalg.norm(to_backbone))
        if tb_len < 1e-8:
            continue
        to_backbone_unit = to_backbone / tb_len

        h_vecs = [coords[h] - mig_pos for h in info['h_indices']]
        avg_h_dir = np.mean(h_vecs, axis=0)
        if float(np.linalg.norm(avg_h_dir)) < 1e-8:
            continue
        if float(np.dot(avg_h_dir, to_backbone_unit)) > 0:
            return True
    return False


def fix_migrating_group_umbrella(xyz: dict,
                                 mol: 'Molecule',
                                 breaking_bonds: List[Tuple[int, int]],
                                 forming_bonds: List[Tuple[int, int]],
                                 ) -> dict:
    """
    Flip non-reactive H atoms on a migrating heavy atom so they point away from the backbone.

    In a group-migration TS (e.g., 1,2-shift of CH₃), Z-matrix interpolation can
    place the H substituents of the migrating atom on the backbone side (between
    donor and acceptor).  Physically, they should point outward.  This function
    detects the wrong orientation and applies an umbrella inversion: each H is
    reflected through the equatorial plane perpendicular to the umbrella axis.

    The algorithm:

    1. Identify migrating heavy atoms (present in both breaking and forming bonds).
    2. Collect their non-reactive H neighbours from the reactant topology.
    3. Compute the umbrella axis (average H direction from the migrating atom).
    4. Place a dummy point on the anti-backbone side (opposite the H average).
    5. For each H, compute ``angle(dummy, migrating_C, H)``.  If > 90° the H is
       on the backbone side: reflect it so that 90° + A° becomes 90° − A°,
       preserving the bond length.

    Args:
        xyz: TS guess XYZ coordinate dictionary.
        mol: Reactant molecule providing bond topology.
        breaking_bonds: Bonds being broken ``[(i, j), ...]``.
        forming_bonds: Bonds being formed ``[(i, j), ...]``.

    Returns:
        A new XYZ dictionary with corrected H positions.
    """
    coords = np.array(xyz['coords'], dtype=float)
    for info in _get_migrating_group_info(xyz, mol, breaking_bonds, forming_bonds):
        mig_idx = info['mig_idx']
        h_indices = info['h_indices']
        mig_pos = coords[mig_idx]

        backbone_centroid = np.mean([coords[i] for i in info['backbone_indices']], axis=0)
        to_backbone = backbone_centroid - mig_pos
        tb_len = float(np.linalg.norm(to_backbone))
        if tb_len < 1e-8:
            continue
        to_backbone_unit = to_backbone / tb_len

        h_vecs = [coords[h] - mig_pos for h in h_indices]
        avg_h_dir = np.mean(h_vecs, axis=0)
        avg_h_len = float(np.linalg.norm(avg_h_dir))
        if avg_h_len < 1e-8:
            continue

        if float(np.dot(avg_h_dir, to_backbone_unit)) <= 0:
            continue

        n = -avg_h_dir / avg_h_len

        for h_idx in h_indices:
            v_h = coords[h_idx] - mig_pos
            proj = float(np.dot(v_h, n))
            if proj < 0:
                coords[h_idx] = mig_pos + (v_h - 2.0 * proj * n)

    new_xyz = dict(xyz)
    new_xyz['coords'] = tuple(tuple(float(v) for v in row) for row in coords)
    return new_xyz


def stagger_donor_terminal_h(xyz: dict,
                              mol: 'Molecule',
                              bonds: List[Tuple[int, int]],
                              ) -> dict:
    """
    Re-stagger non-migrating H atoms on the donor carbon after H placement.

    For each forming bond involving hydrogen, this function identifies the
    donor heavy atom (bonded to the migrating H in the reactant) and its
    backbone neighbour.  Non-migrating H atoms on the donor are then rotated
    around the donor–backbone bond axis to sit at ±120° from the migrating H
    dihedral, choosing the direction that requires the smaller rotation.

    This corrects the stagger that was originally applied in the pre-
    interpolation step (``get_near_attack_xyz``) but invalidated when
    ``fix_forming_bond_distances`` moved the migrating H to its triangulated
    TS position.

    Args:
        xyz: TS guess XYZ coordinate dictionary.
        mol: Reactant molecule providing bond topology.
        bonds: Forming bonds in reactant atom ordering.

    Returns:
        A new XYZ dictionary with non-migrating H atoms staggered.
    """
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    symbols = xyz['symbols']
    coords = np.array(xyz['coords'], dtype=float)

    for (i_atom, j_atom) in bonds:
        if symbols[i_atom] == 'H':
            h_atom, acceptor = i_atom, j_atom
        elif symbols[j_atom] == 'H':
            h_atom, acceptor = j_atom, i_atom
        else:
            continue
        donor_idx = None
        for nbr in mol.atoms[h_atom].bonds.keys():
            nbr_idx = atom_to_idx[nbr]
            if symbols[nbr_idx] != 'H':
                donor_idx = nbr_idx
                break
        if donor_idx is None:
            continue
        backbone_idx = None
        for nbr in mol.atoms[donor_idx].bonds.keys():
            nbr_idx = atom_to_idx[nbr]
            if nbr_idx != h_atom and symbols[nbr_idx] != 'H':
                backbone_idx = nbr_idx
                break
        if backbone_idx is None:
            continue
        ref_idx = None
        for nbr in mol.atoms[backbone_idx].bonds.keys():
            nbr_idx = atom_to_idx[nbr]
            if nbr_idx != donor_idx:
                ref_idx = nbr_idx
                break
        if ref_idx is None:
            continue

        axis_raw = coords[backbone_idx] - coords[donor_idx]
        axis_norm = float(np.linalg.norm(axis_raw))
        if axis_norm < 1e-8:
            continue
        axis = axis_raw / axis_norm
        origin = coords[donor_idx].copy()

        dih_mig = _dihedral_deg(coords[h_atom], coords[donor_idx],
                                coords[backbone_idx], coords[ref_idx])
        for nbr in mol.atoms[donor_idx].bonds.keys():
            other_idx = atom_to_idx[nbr]
            if other_idx == backbone_idx or other_idx == h_atom or symbols[other_idx] != 'H':
                continue
            dih_h = _dihedral_deg(coords[other_idx], coords[donor_idx],
                                  coords[backbone_idx], coords[ref_idx])
            t_plus = dih_mig + 120.0
            t_minus = dih_mig - 120.0
            d_plus = (t_plus - dih_h) * np.pi / 180.0
            d_minus = (t_minus - dih_h) * np.pi / 180.0
            delta = d_plus if abs(d_plus) <= abs(d_minus) else d_minus
            if abs(delta) > 1e-6:
                _rotate_atoms(coords, origin, axis, {other_idx}, delta)

    new_xyz = dict(xyz)
    new_xyz['coords'] = tuple(tuple(float(v) for v in row) for row in coords)
    return new_xyz


def _build_zmat_with_retry(xyz: dict,
                           mol: 'Molecule',
                           anchors: Optional[list],
                           constraints: Optional[dict],
                           label: str = '',
                           ) -> Optional[dict]:
    """
    Build a Z-matrix from ``xyz`` using ``mol`` topology, retrying without
    constraints if the first attempt fails.

    Args:
        xyz: XYZ coordinate dictionary.
        mol: RMG Molecule whose bond topology matches ``xyz``.
        anchors: Preferred anchor atoms for the Z-matrix.
        constraints: ``R_atom`` constraints for reactive bonds.
        label: Logging label for debug messages.

    Returns:
        The Z-matrix dict, or ``None`` if both attempts fail.
    """
    try:
        return xyz_to_zmat(xyz=xyz, mol=mol, consolidate=False,
                           anchors=anchors, constraints=constraints)
    except Exception as e:
        logger.debug(f'Linear ({label}): xyz_to_zmat raised {type(e).__name__}: {e}; '
                     f'retrying without constraints.')
        try:
            return xyz_to_zmat(xyz=xyz, mol=mol, consolidate=False, anchors=anchors)
        except Exception as e2:
            logger.debug(f'Linear ({label}): xyz_to_zmat (no constraints) raised '
                         f'{type(e2).__name__}: {e2}; skipping.')
            return None


# ---------------------------------------------------------------------------
# Family-specific post-processing handlers
# ---------------------------------------------------------------------------

def _postprocess_generic(xyz: dict,
                         r_mol: 'Molecule',
                         forming_bonds: List[Tuple[int, int]],
                         breaking_bonds: List[Tuple[int, int]],
                         r_label_map: Optional[Dict[str, int]] = None,
                         ) -> Tuple[dict, Set[int]]:
    """
    Generic (no-op) post-processing fallback for families without a dedicated handler.

    Applies only universally safe fixes: non-reactive H distance correction and
    crowded H redistribution.  No family-specific geometry manipulation is performed.

    Args:
        xyz: Raw TS guess XYZ dictionary.
        r_mol: Reactant RMG Molecule (bond topology reference).
        forming_bonds: List of (i, j) index pairs for bonds that form.
        breaking_bonds: List of (i, j) index pairs for bonds that break.
        r_label_map: Family-labelled atom map (unused in generic handler).

    Returns:
        Tuple of (corrected XYZ dict, empty set of migrating H indices).
    """
    xyz = fix_nonreactive_h_distances(xyz, r_mol, migrating_h_indices=set())
    xyz = fix_crowded_h_atoms(xyz, r_mol, skip_h_indices=set())
    xyz = fix_h_nonbonded_clashes(xyz, r_mol, skip_h_indices=set())
    return xyz, set()


def _postprocess_h_migration(xyz: dict,
                              r_mol: 'Molecule',
                              forming_bonds: List[Tuple[int, int]],
                              breaking_bonds: List[Tuple[int, int]],
                              r_label_map: Optional[Dict[str, int]] = None,
                              ) -> Tuple[dict, Set[int]]:
    """
    Post-processing pipeline for intra H-migration and related H-transfer families.

    The pipeline (in order):

    1. Fix forming-bond distances (H triangulation) — only affects forming bonds with H.
    2. Identify migrating H atoms.
    3. Stagger donor terminal H atoms (only when migrating H is present).
    4. Fix non-reactive H distances (reset H atoms displaced by interpolation).
    5. Fix crowded H atoms (redistribute terminal H groups).
    6. Fix migrating-group umbrella orientation.

    Args:
        xyz: Raw TS guess XYZ dictionary.
        r_mol: Reactant RMG Molecule (bond topology reference).
        forming_bonds: List of (i, j) index pairs for bonds that form.
        breaking_bonds: List of (i, j) index pairs for bonds that break.
        r_label_map: Family-labelled atom map (unused in this handler).

    Returns:
        Tuple of (corrected XYZ dict, set of migrating H atom indices).
    """
    xyz = fix_forming_bond_distances(xyz, r_mol, forming_bonds)
    migrating_hs: Set[int] = set()
    for ia, ja in forming_bonds:
        if xyz['symbols'][ia] == 'H':
            migrating_hs.add(ia)
        if xyz['symbols'][ja] == 'H':
            migrating_hs.add(ja)
    if migrating_hs:
        xyz = stagger_donor_terminal_h(xyz, r_mol, forming_bonds)
    xyz = fix_nonreactive_h_distances(xyz, r_mol, migrating_hs)
    xyz = fix_crowded_h_atoms(xyz, r_mol, skip_h_indices=migrating_hs)
    xyz = fix_h_nonbonded_clashes(xyz, r_mol, skip_h_indices=migrating_hs)
    xyz = fix_migrating_group_umbrella(xyz, r_mol, breaking_bonds, forming_bonds)
    return xyz, migrating_hs


def _postprocess_group_shift(xyz: dict,
                              r_mol: 'Molecule',
                              forming_bonds: List[Tuple[int, int]],
                              breaking_bonds: List[Tuple[int, int]],
                              r_label_map: Optional[Dict[str, int]] = None,
                              ) -> Tuple[dict, Set[int]]:
    """
    Post-processing pipeline for group-migration and sigmatropic families.

    These families involve a non-H group migrating between heavy atoms
    (e.g. 1,2-shiftC, 1,2-shiftS, 1,3-sigmatropic rearrangements).
    The pipeline applies forming-bond distance correction, umbrella
    orientation, and universal H fixes, but omits H-transfer-specific
    steps (donor terminal H staggering) that are not applicable when
    the migrating atom is not hydrogen.

    Args:
        xyz: Raw TS guess XYZ dictionary.
        r_mol: Reactant RMG Molecule (bond topology reference).
        forming_bonds: List of (i, j) index pairs for bonds that form.
        breaking_bonds: List of (i, j) index pairs for bonds that break.
        r_label_map: Family-labelled atom map (unused in this handler).

    Returns:
        Tuple of (corrected XYZ dict, empty set — no migrating H atoms).
    """
    xyz = fix_forming_bond_distances(xyz, r_mol, forming_bonds)
    xyz = fix_nonreactive_h_distances(xyz, r_mol, migrating_h_indices=set())
    xyz = fix_crowded_h_atoms(xyz, r_mol, skip_h_indices=set())
    xyz = fix_h_nonbonded_clashes(xyz, r_mol, skip_h_indices=set())
    xyz = fix_migrating_group_umbrella(xyz, r_mol, breaking_bonds, forming_bonds)
    return xyz, set()


# Register H-migration families.
for _fam in ('intra_H_migration', 'Ketoenol', 'Intra_Disproportionation',
             'Intra_ene_reaction', 'intra_OH_migration',
             'Intra_RH_Add_Endocyclic', 'Intra_RH_Add_Exocyclic',
             'Concerted_Intra_Diels_alder_monocyclic_1,2_shiftH'):
    _FAMILY_POSTPROCESSORS[_fam] = _postprocess_h_migration

def _postprocess_cc_shift(xyz: dict,
                          r_mol: 'Molecule',
                          forming_bonds: List[Tuple[int, int]],
                          breaking_bonds: List[Tuple[int, int]],
                          r_label_map: Optional[Dict[str, int]] = None,
                          ) -> Tuple[dict, Set[int]]:
    """
    Post-processing pipeline for C-C shift families with hybridisation changes.

    These reactions convert triple bonds to double bonds (sp → sp2), creating
    CH₂ groups whose H positions are unreliable after Z-matrix interpolation.
    This handler mirrors ``_postprocess_group_shift`` but additionally
    redistributes CH₂ H atoms to symmetric tetrahedral positions.

    Args:
        xyz: Raw TS guess XYZ dictionary.
        r_mol: Reactant RMG Molecule (bond topology reference).
        forming_bonds: List of (i, j) index pairs for bonds that form.
        breaking_bonds: List of (i, j) index pairs for bonds that break.
        r_label_map: Family-labelled atom map (unused in this handler).

    Returns:
        Tuple of (corrected XYZ dict, empty set — no migrating H atoms).
    """
    xyz = fix_forming_bond_distances(xyz, r_mol, forming_bonds)
    xyz = fix_nonreactive_h_distances(xyz, r_mol, migrating_h_indices=set())
    xyz = fix_crowded_h_atoms(xyz, r_mol, skip_h_indices=set(),
                              redistribute_ch2=True)
    xyz = fix_h_nonbonded_clashes(xyz, r_mol, skip_h_indices=set())
    xyz = fix_migrating_group_umbrella(xyz, r_mol, breaking_bonds, forming_bonds)
    return xyz, set()


# Register group-migration and sigmatropic families.
for _fam in ('1,2_shiftC', '1,2_shiftS',
             '1,3_sigmatropic_rearrangement',
             'intra_substitutionCS_isomerization'):
    _FAMILY_POSTPROCESSORS[_fam] = _postprocess_group_shift

_FAMILY_POSTPROCESSORS['6_membered_central_C-C_shift'] = _postprocess_cc_shift


# ---------------------------------------------------------------------------
# Family-specific validation handlers
# ---------------------------------------------------------------------------

def _validate_h_migration(xyz: dict,
                           migrating_hs: Set[int],
                           forming_bonds: List[Tuple[int, int]],
                           r_mol: 'Molecule',
                           label: str = '',
                           ) -> Tuple[bool, str]:
    """
    H-transfer-specific validation filters.

    Applied after generic validation (collisions, fragments, detached H) has passed.
    Only active when ``migrating_hs`` is non-empty.

    Args:
        xyz: Postprocessed TS guess XYZ dictionary.
        migrating_hs: Atom indices of migrating hydrogen atoms.
        forming_bonds: Forming-bond index pairs.
        r_mol: Reactant RMG Molecule (bond topology reference).
        label: Logging label for debug messages.

    Returns:
        Tuple of (is_valid, rejection_reason).
    """
    if not migrating_hs:
        return True, ''
    if _has_h_close_contact(xyz):
        return False, 'H close contact'
    if _has_misoriented_migrating_h(xyz, forming_bonds, r_mol):
        return False, 'misoriented migrating H'
    if _has_migrating_h_nearer_to_nonreactive(xyz, forming_bonds, r_mol):
        return False, 'migrating H nearer to non-reactive atom'
    if _has_bad_ts_motif(xyz, forming_bonds, r_mol):
        return False, 'bad TS motif distances'
    return True, ''


# Register H-migration validator for families dominated by H transfer.
for _fam in ('intra_H_migration', 'Ketoenol', 'Intra_Disproportionation',
             'Intra_ene_reaction', 'intra_OH_migration',
             'Intra_RH_Add_Endocyclic', 'Intra_RH_Add_Exocyclic',
             'Concerted_Intra_Diels_alder_monocyclic_1,2_shiftH',
             '1,3_sigmatropic_rearrangement',
             'intra_substitutionCS_isomerization'):
    _FAMILY_VALIDATORS[_fam] = _validate_h_migration


def _validate_group_shift(xyz: dict,
                          migrating_hs: Set[int],
                          forming_bonds: List[Tuple[int, int]],
                          r_mol: 'Molecule',
                          label: str = '',
                          ) -> Tuple[bool, str]:
    """
    Group-shift-specific validation (1,2-shiftC, 1,2-shiftS, C-C shift).

    Checks forming-bond distances are within a reasonable TS window and
    H close contacts.  Falls through to the H-migration validator when
    migrating H atoms are also present.

    Args:
        xyz: Postprocessed TS guess XYZ dictionary.
        migrating_hs: Atom indices of migrating hydrogen atoms.
        forming_bonds: Forming-bond index pairs.
        r_mol: Reactant RMG Molecule (bond topology reference).
        label: Logging label for debug messages.

    Returns:
        Tuple of (is_valid, rejection_reason).
    """
    if _has_h_close_contact(xyz):
        return False, 'H close contact'
    coords_arr = np.array(xyz['coords'], dtype=float)
    for i_atom, j_atom in forming_bonds:
        d = float(np.linalg.norm(coords_arr[i_atom] - coords_arr[j_atom]))
        if d < 1.5 or d > 4.0:
            return False, f'forming bond distance {d:.2f} outside TS range'
    if migrating_hs:
        return _validate_h_migration(xyz, migrating_hs, forming_bonds, r_mol, label)
    return True, ''


# Register group-shift validator for heavy-atom migration families.
for _fam in ('1,2_shiftC', '1,2_shiftS', '6_membered_central_C-C_shift'):
    _FAMILY_VALIDATORS[_fam] = _validate_group_shift


# ---------------------------------------------------------------------------
# Dispatch wrappers
# ---------------------------------------------------------------------------

def _postprocess_ts_guess(xyz: dict,
                          r_mol: 'Molecule',
                          forming_bonds: List[Tuple[int, int]],
                          breaking_bonds: List[Tuple[int, int]],
                          family: Optional[str] = None,
                          r_label_map: Optional[Dict[str, int]] = None,
                          ) -> Tuple[dict, Set[int]]:
    """
    Dispatch to the appropriate family-specific post-processing handler.

    Three handler tiers are available:

    * :func:`_postprocess_h_migration` — for H-transfer families (intra_H_migration,
      Ketoenol, etc.).  Full pipeline: forming-bond triangulation, donor terminal H
      staggering, non-reactive H distance fix, crowded-H redistribution, umbrella flip.
    * :func:`_postprocess_group_shift` — for non-H group migrations (1,2_shiftC/S,
      sigmatropic, etc.).  Applies forming-bond fix, umbrella, and universal H fixes
      but omits H-transfer-specific donor staggering.
    * :func:`_postprocess_generic` — safe default for unknown families.  Only universal
      H fixes (distance and crowding).

    Looks up ``family`` in :data:`_FAMILY_POSTPROCESSORS`.  If no handler is
    registered, :func:`_postprocess_generic` is used.

    Args:
        xyz: Raw TS guess XYZ dictionary.
        r_mol: Reactant RMG Molecule (bond topology reference).
        forming_bonds: List of (i, j) index pairs for bonds that form.
        breaking_bonds: List of (i, j) index pairs for bonds that break.
        family: RMG reaction family name (e.g. ``'intra_H_migration'``).
        r_label_map: Family-labelled atom map from the product_dict
            (e.g. ``{'*1': 3, '*2': 7, '*3': 0}``).

    Returns:
        Tuple of (corrected XYZ dict, set of special atom indices such as
        migrating H atoms).
    """
    handler = _FAMILY_POSTPROCESSORS.get(family, _postprocess_generic)
    return handler(xyz, r_mol, forming_bonds, breaking_bonds, r_label_map)


def _has_excessive_backbone_drift(xyz: dict,
                                  anchor_xyz: dict,
                                  max_mean_heavy_disp: float = 2.0,
                                  reactive_indices: Optional[Set[int]] = None,
                                  ) -> bool:
    """
    Check whether spectator heavy atoms in the TS guess have drifted too far from the anchor.

    Atoms at reactive sites (forming/breaking bonds) are expected to move and are
    excluded from the check.  Only non-reactive heavy atoms are considered.
    When their mean displacement exceeds ``max_mean_heavy_disp`` Angstroms,
    the guess is likely garbage (e.g. from a degraded atom map or failed Z-mat round-trip).

    Args:
        xyz: TS guess XYZ dictionary.
        anchor_xyz: Reference geometry (typically the reactant).
        max_mean_heavy_disp: Maximum allowed mean displacement (Angstroms) for
            spectator heavy atoms.
        reactive_indices: Atom indices involved in forming/breaking bonds.
            These are excluded from the drift calculation.

    Returns:
        ``True`` if the backbone has drifted excessively.
    """
    coords = np.array(xyz['coords'], dtype=float)
    anchor_coords = np.array(anchor_xyz['coords'], dtype=float)
    symbols = xyz['symbols']
    skip = reactive_indices or set()
    spectator_mask = [i for i, s in enumerate(symbols)
                      if s != 'H' and i not in skip]
    if not spectator_mask:
        return False
    heavy_disp = [float(np.linalg.norm(coords[i] - anchor_coords[i]))
                  for i in spectator_mask]
    return float(np.mean(heavy_disp)) > max_mean_heavy_disp


def _validate_ts_guess(xyz: dict,
                       migrating_hs: Set[int],
                       forming_bonds: List[Tuple[int, int]],
                       r_mol: 'Molecule',
                       label: str = '',
                       family: Optional[str] = None,
                       anchor_xyz: Optional[dict] = None,
                       reactive_indices: Optional[Set[int]] = None,
                       ) -> Tuple[bool, str]:
    """
    Run generic rejection filters, then family-specific filters.

    Generic filters (collisions, detached H, fragment count, backbone drift)
    are always applied.  Family-specific filters are applied only if a handler
    is registered in :data:`_FAMILY_VALIDATORS`.

    Args:
        xyz: Postprocessed TS guess XYZ dictionary.
        migrating_hs: Atom indices of migrating hydrogen atoms (or other
            special atoms identified during post-processing).
        forming_bonds: Forming-bond index pairs.
        r_mol: Reactant RMG Molecule (bond topology reference).
        label: Logging label for debug messages.
        family: RMG reaction family name.
        anchor_xyz: Optional reference geometry (typically the reactant).
            When provided, the spectator-heavy-atom mean displacement between
            the TS guess and the anchor is computed; guesses that drift too
            far are rejected.
        reactive_indices: Atom indices involved in forming/breaking bonds,
            excluded from the backbone drift check.

    Returns:
        Tuple of (is_valid, rejection_reason).  ``reason`` is an empty string
        when the guess is valid.
    """
    if colliding_atoms(xyz):
        reason = 'colliding atoms'
    elif _has_detached_hydrogen(xyz, max_h_heavy_dist=3.0):
        reason = 'detached hydrogen'
    elif _has_too_many_fragments(xyz):
        reason = 'too many fragments (3+)'
    elif anchor_xyz is not None and _has_excessive_backbone_drift(
            xyz, anchor_xyz, max_mean_heavy_disp=3.0,
            reactive_indices=reactive_indices):
        reason = 'excessive backbone drift from anchor'
    else:
        family_validator = _FAMILY_VALIDATORS.get(family)
        if family_validator is not None:
            is_valid, reason = family_validator(xyz, migrating_hs, forming_bonds, r_mol, label)
            if not is_valid:
                logger.debug(f'Linear ({label}): discarded — {reason}.')
                return False, reason
        return True, ''
    logger.debug(f'Linear ({label}): discarded — {reason}.')
    return False, reason
