"""
Perceive 3D Cartesian coordinates and generate a representative resonance structure as a Molecule object.
"""

import heapq
from typing import Any, Iterable
from itertools import combinations
import numpy as np

from arc.common import get_logger, get_bonds_from_dmat, NUMBER_BY_SYMBOL
from arc.exceptions import ConverterError, InputError
from arc.species.converter import check_xyz_dict, xyz_to_dmat
from arc.molecule.filtration import get_octet_deviation
from arc.molecule.molecule import Atom, Bond, Molecule
from arc.molecule.resonance import generate_resonance_structures_safely

logger = get_logger()

# valence electrons for main‐group elements
_VALENCE_ELECTRONS = {
    'H': 1, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7,
    'P': 5, 'S': 6, 'Cl': 7, 'Br': 7, 'I': 7,
}


def perceive_molecule_from_xyz(
    xyz: dict[str, Any] | str,
    charge: int = 0,
    multiplicity: int | None = None,
) -> Molecule | None:
    xyz_dict = validate_xyz(xyz)
    if xyz_dict is None:
        return None

    symbols, coords = xyz_dict["symbols"], xyz_dict["coords"]
    multiplicity = multiplicity or infer_multiplicity(symbols, charge)

    try:
        dmat = xyz_to_dmat(xyz_dict)
        bonds = get_bonds_from_dmat(dmat, symbols, charges=[0] * len(symbols))
    except ValueError as e:
        logger.warning(f"Failed to perceive bonds from XYZ: {e}")
        return None

    atoms = [
        Atom(element=sym, radical_electrons=0, charge=0, lone_pairs=0, coords=np.array(coord))
        for sym, coord in zip(symbols, coords)
    ]
    mol0 = Molecule(atoms=atoms)
    for i, j in bonds:
        mol0.add_bond(Bond(mol0.atoms[i], mol0.atoms[j], order=1))

    rep = generate_lewis_structure(mol0, charge, multiplicity)
    if rep is None:
        logger.warning(
            "Could not assign bond orders/electrons for XYZ perception, falling back to all single bonds"
        )
        rep = mol0.copy(deep=True)
        rep.multiplicity = multiplicity
        adjust_atoms_for_octet(rep, multiplicity)
        assign_formal_charges(rep)

    # pick canonical resonance
    rep = get_representative_resonance_structure(rep)
    # now dial down any over‐charged form (esp. fixes SN2)
    if multiplicity == 1 and rep.get_net_charge() == 0:
        rep = _reduce_charge_separation(rep)
    # re‐assign formal charges & finalize
    assign_formal_charges(rep)
    rep.multiplicity = multiplicity
    return rep


def validate_xyz(xyz: dict[str, Any] | str) -> dict[str, Any] | None:
    if not xyz:
        logger.warning("Empty XYZ input provided.")
        return None
    try:
        return check_xyz_dict(xyz)
    except (InputError, ConverterError) as e:
        logger.warning(f"Invalid XYZ input: {e}")
        return None


def infer_multiplicity(symbols: Iterable[str], total_charge: int) -> int:
    total_e = sum(NUMBER_BY_SYMBOL[s] for s in symbols) - total_charge
    return 1 if total_e % 2 == 0 else 2


def generate_lewis_structure(
    base_mol: Molecule,
    total_charge: int,
    multiplicity: int,
) -> Molecule | None:
    max_bond = []
    for at in base_mol.atoms:
        sym = getattr(at.element, 'symbol', at.element)
        ge = _VALENCE_ELECTRONS.get(sym, NUMBER_BY_SYMBOL[sym] - 2)
        max_bond.append(ge)

    bond_pairs = [
        (base_mol.atoms.index(e.vertex1), base_mol.atoms.index(e.vertex2))
        for e in base_mol.get_all_edges()
    ]
    initial = tuple(1 for _ in bond_pairs)
    seen = {initial}
    queue: list[tuple[int, int, int, tuple[int, ...]]] = []
    counter = 0

    def heuristic(orders: tuple[int, ...]) -> int:
        sums = [0] * len(base_mol.atoms)
        for (i, j), o in zip(bond_pairs, orders):
            sums[i] += o
            sums[j] += o
        return sum(1 for idx, s in enumerate(sums) if s < max_bond[idx])

    heapq.heappush(queue, (heuristic(initial), 0, counter, initial))
    counter += 1

    while queue:
        f, increments, _, orders = heapq.heappop(queue)

        partial = [0] * len(base_mol.atoms)
        for (i, j), o in zip(bond_pairs, orders):
            partial[i] += o
            partial[j] += o
        if any(partial[k] > max_bond[k] for k in range(len(partial))):
            continue

        mol = base_mol.copy(deep=True)
        for (i, j), order in zip(bond_pairs, orders):
            mol.get_edge(mol.atoms[i], mol.atoms[j]).order = order
        mol.multiplicity = multiplicity

        if adjust_atoms_for_octet(mol, multiplicity):
            assign_formal_charges(mol)
            if mol.get_net_charge() == total_charge:
                return mol

        for idx in range(len(orders)):
            if orders[idx] < 3:
                new_orders = list(orders)
                new_orders[idx] += 1
                tpl = tuple(new_orders)
                if tpl not in seen:
                    seen.add(tpl)
                    g = increments + 1
                    heapq.heappush(queue, (g + heuristic(tpl), g, counter, tpl))
                    counter += 1

    return None


def adjust_atoms_for_octet(
    mol: Molecule,
    multiplicity: int,
) -> bool:
    """
    Try to assign lone_pairs and radical_electrons to match:
      - hydrogen’s duet (2),
      - each heavy atom’s octet (8), or S’s 10/12,
      - if a site is radical center, allow a 7-electron core,
      - exactly multiplicity-1 unpaired electrons,
    by minimizing the overall octet deviation.
    Returns True if a valid assignment was applied to `mol`, False otherwise.
    """
    atoms = mol.atoms
    target_rad = multiplicity - 1

    # trivial H• case
    if len(atoms) == 1 and atoms[0].is_hydrogen():
        a = atoms[0]
        a.lone_pairs = 0
        a.radical_electrons = target_rad
        return target_rad in (0, 1)

    # precompute bond‐order sum for each atom
    bond_sums = [sum(e.order for e in a.edges.values()) for a in atoms]

    def test_sites(sites: tuple[int, ...]) -> Molecule | None:
        cand = mol.copy(deep=True)
        cand.multiplicity = mol.multiplicity
        # reset
        for at in cand.atoms:
            at.lone_pairs = 0
            at.radical_electrons = 0
        # place radicals
        for idx in sites:
            cand.atoms[idx].radical_electrons = 1

        # assign lone pairs
        for i, at in enumerate(cand.atoms):
            rad = at.radical_electrons
            B = bond_sums[i]
            if at.is_hydrogen():
                allowed = {2}
            else:
                sym = getattr(at.element, 'symbol', at.element)
                allowed = {8}
                if sym == 'S':
                    allowed |= {10, 12}
                if rad:
                    allowed.add(7)
            placed = False
            for v in sorted(allowed):
                rem = v - rad - 2 * B
                if rem >= 0 and rem % 2 == 0:
                    lp = rem // 2
                    if lp <= 3:
                        at.lone_pairs = lp
                        placed = True
                        break
            if not placed:
                return None

        # must match exactly
        if sum(at.radical_electrons for at in cand.atoms) != target_rad:
            return None
        return cand

    # collect candidate structures
    candidates: list[tuple[float, Molecule]] = []
    heavy_idxs = [i for i, a in enumerate(atoms) if not a.is_hydrogen()]
    H_idxs     = [i for i, a in enumerate(atoms) if     a.is_hydrogen()]

    if target_rad == 0:
        # closed shell
        c = test_sites(())
        if c is not None:
            candidates.append((get_octet_deviation(c), c))
    elif target_rad == 1:
        # single‐radical: hetero first, then carbon, then H
        hetero = [i for i in heavy_idxs if getattr(atoms[i].element, 'symbol', atoms[i].element) not in ('C', 'H')]
        carbon = [i for i in heavy_idxs if getattr(atoms[i].element, 'symbol', atoms[i].element) == 'C']
        for pool in (hetero, carbon, H_idxs):
            for site in pool:
                c = test_sites((site,))
                if c is not None:
                    candidates.append((get_octet_deviation(c), c))
            if candidates:
                break
    else:
        # multi‐radical: place on heavy atoms combinations
        if len(heavy_idxs) >= target_rad:
            for combo in combinations(heavy_idxs, target_rad):
                c = test_sites(combo)
                if c is not None:
                    candidates.append((get_octet_deviation(c), c))

    if not candidates:
        return False

    # pick the minimal‐deviation candidate
    _, best = min(candidates, key=lambda x: x[0])
    for orig, new in zip(mol.atoms, best.atoms):
        orig.lone_pairs        = new.lone_pairs
        orig.radical_electrons = new.radical_electrons
    return True


def assign_formal_charges(mol: Molecule) -> None:
    for atom in mol.atoms:
        if atom.radical_electrons:
            atom.charge = 0
            continue
        sym = getattr(atom.element, 'symbol', atom.element)
        ge = _VALENCE_ELECTRONS.get(sym, NUMBER_BY_SYMBOL[sym] - 2)
        nonbond = 2 * atom.lone_pairs
        bonding = sum(b.order for b in atom.edges.values())
        atom.charge = ge - (nonbond + bonding)


def get_representative_resonance_structure(mol: Molecule) -> Molecule:
    variants = generate_resonance_structures_safely(mol)
    if not variants:
        return mol
    for v in variants:
        if mol.copy().is_isomorphic(v.copy()):
            return v
    return variants[0]


def _reduce_charge_separation(mol: Molecule) -> Molecule:
    """
    Iteratively bump bond orders to reduce total |formal_charge|
    without worsening octet deviation.
    """
    best = mol.copy(deep=True)
    best.multiplicity = mol.multiplicity
    assign_formal_charges(best)
    base_dev = get_octet_deviation(best)
    base_sep = sum(abs(a.charge) for a in best.atoms)

    if base_sep <= 1:
        return best

    bond_pairs = [
        (best.vertices.index(e.vertex1), best.vertices.index(e.vertex2))
        for e in best.get_all_edges()
    ]

    improved = True
    while improved:
        improved = False
        curr = best
        curr_dev = get_octet_deviation(curr)
        curr_sep = sum(abs(a.charge) for a in curr.atoms)

        for i, j in bond_pairs:
            edge = curr.get_edge(curr.atoms[i], curr.atoms[j])
            if edge.order >= 3:
                continue
            cand = curr.copy(deep=True)
            cand.multiplicity = curr.multiplicity
            ce = cand.get_edge(cand.atoms[i], cand.atoms[j])
            ce.order += 1
            if not adjust_atoms_for_octet(cand, cand.multiplicity):
                continue
            assign_formal_charges(cand)
            dev = get_octet_deviation(cand)
            sep = sum(abs(a.charge) for a in cand.atoms)
            if dev == curr_dev and sep < curr_sep:
                best = cand
                improved = True
                break

    return best
