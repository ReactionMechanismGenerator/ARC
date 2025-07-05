"""
Perceive 3D Cartesian coordinates and generate a representative resonance structure as a Molecule object.
"""

import heapq
from typing import Any, Iterable

import numpy as np

from arc.common import get_logger, get_bonds_from_dmat, NUMBER_BY_SYMBOL
from arc.exceptions import ConverterError, InputError
from arc.species.converter import check_xyz_dict, xyz_to_dmat
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
    """
    Infer a localized Lewis‐structure ARC Molecule from Cartesian coordinates.

    Args:
        xyz (dict or str): Either an ARC‐format xyz dict or an XYZ string.
        charge (int): Total formal charge.
        multiplicity (int, optional): Spin multiplicity (2S+1). If None, inferred.

    Returns:
        Molecule or None: The perceived Molecule, or None if perception failed.
    """
    xyz_dict = validate_xyz(xyz)
    if xyz_dict is None:
        return None

    symbols = xyz_dict["symbols"]
    coords = xyz_dict["coords"]
    multiplicity = multiplicity or infer_multiplicity(symbols, charge)

    # Build distance matrix & connectivity
    try:
        dmat = xyz_to_dmat(xyz_dict)
        bonds = get_bonds_from_dmat(dmat, symbols, charges=[0] * len(symbols))
    except ValueError as e:
        logger.warning(f"Failed to perceive bonds from XYZ: {e}")
        return None

    # Create atoms + single bonds
    atoms = [
        Atom(element=sym, radical_electrons=0, charge=0, lone_pairs=0, coords=np.array(coord))
        for sym, coord in zip(symbols, coords)
    ]
    mol0 = Molecule(atoms=atoms)
    for i, j in bonds:
        mol0.add_bond(Bond(mol0.atoms[i], mol0.atoms[j], order=1))

    # A* search for minimal‐increment Lewis structure
    rep = generate_lewis_structure(mol0, charge, multiplicity)
    if rep is None:
        logger.warning("Could not assign bond orders/electrons for XYZ perception.")
        return None

    # Pick a canonical resonance form
    rep = get_representative_resonance(rep)

    # Assign formal charges based on lone pairs / radicals / bond orders
    assign_formal_charges(rep)

    # Finalize
    rep.multiplicity = multiplicity
    return rep


def validate_xyz(xyz: dict[str, Any] | str) -> dict[str, Any] | None:
    """
    Ensure the input is a valid ARC xyz dict.
    """
    if not xyz:
        logger.warning("Empty XYZ input provided.")
        return None
    try:
        return check_xyz_dict(xyz)
    except (InputError, ConverterError) as e:
        logger.warning(f"Invalid XYZ input: {e}")
        return None


def infer_multiplicity(symbols: Iterable[str], total_charge: int) -> int:
    """
    Closed‐shell (even e⁻) → singlet (1); odd e⁻ → doublet (2).
    """
    total_e = sum(NUMBER_BY_SYMBOL[s] for s in symbols) - total_charge
    return 1 if total_e % 2 == 0 else 2


def generate_lewis_structure(
    base_mol: Molecule,
    total_charge: int,
    multiplicity: int,
) -> Molecule | None:
    """
    A* search over bond‐order increments (1→2→3) to satisfy octet, charge, and spin,
    returning the first valid, minimal‐increment structure.
    """
    edges = base_mol.get_all_edges()
    initial = tuple(1 for _ in edges)
    seen = {initial}
    queue: list[tuple[int, int, tuple[int, ...]]] = []
    counter = 0

    # (increments so far, tie‐breaker, orders)
    heapq.heappush(queue, (0, counter, initial))
    counter += 1

    while queue:
        increments, _, orders = heapq.heappop(queue)

        # Apply this candidate bond‐order assignment
        mol = base_mol.copy(deep=True)
        for edge, order in zip(edges, orders):
            mol.get_edge(edge.vertex1, edge.vertex2).order = order

        # Check octet/charge/spin
        if adjust_atoms_for_octet(mol, total_charge, multiplicity):
            # Now check total formal charge
            assign_formal_charges(mol)
            if mol.get_net_charge() == total_charge:
                return mol

        # Otherwise branch: increment each bond up to triple
        for idx in range(len(orders)):
            if orders[idx] < 3:
                new_orders = list(orders)
                new_orders[idx] += 1
                tpl = tuple(new_orders)
                if tpl not in seen:
                    seen.add(tpl)
                    heapq.heappush(queue, (increments + 1, counter, tpl))
                    counter += 1

    return None


def adjust_atoms_for_octet(
    mol: Molecule,
    charge: int,
    multiplicity: int,
) -> bool:
    """
    Assign lone pairs & radicals so that:
      - Each heavy atom meets an octet (or expanded octet for S),
      - Total unpaired electrons == multiplicity−1.

    NOTE: Does *not* set formal charges here; that'll be done afterwards.
    """
    atoms = mol.atoms
    target_rad = multiplicity - 1

    # trivial H• case
    if len(atoms) == 1 and atoms[0].is_hydrogen():
        a = atoms[0]
        a.lone_pairs = 0
        a.radical_electrons = target_rad
        return target_rad in (0, 1)

    # partition indices
    heavy_idxs = [i for i, a in enumerate(atoms) if not a.is_hydrogen()]
    H_idxs     = [i for i, a in enumerate(atoms) if     a.is_hydrogen()]

    # helper: get plain symbol
    def sym(a_: Atom) -> str:
        e = a_.element
        return getattr(e, 'symbol', e)

    # can this atom carry exactly `rad` unpaired electrons + some lone pairs?
    def can_assign(a_: Atom, rad_: int) -> bool:
        B = int(a_.get_total_bond_order())
        # base allowed valences
        allowed = {8}
        if a_.is_sulfur():
            allowed |= {10, 12}
        # if it's the radical center, also allow a 7-electron valence
        if rad_ == 1:
            allowed.add(7)
        for v in allowed:
            rem = v - rad_ - 2 * B
            if rem >= 0 and rem % 2 == 0:
                lp = rem // 2
                if 0 <= lp <= 3:
                    a_.lone_pairs        = lp
                    a_.radical_electrons = rad_
                    return True
        return False

    # reset everybody
    for a in atoms:
        a.lone_pairs = 0
        a.radical_electrons = 0

    # closed-shell
    if target_rad == 0:
        return all(can_assign(atoms[i], 0) for i in heavy_idxs)

    # 1 unpaired electron: try heteroatoms first
    hetero_idxs = [i for i in heavy_idxs if sym(atoms[i]) not in ('C', 'H')]
    for site in hetero_idxs:
        # reset
        for a in atoms:
            a.lone_pairs = 0
            a.radical_electrons = 0
        ok = True
        for i in heavy_idxs:
            rad = 1 if i == site else 0
            if not can_assign(atoms[i], rad):
                ok = False
                break
        if ok:
            return True

    # then try carbon sites
    carbon_idxs = [i for i in heavy_idxs if sym(atoms[i]) == 'C']
    for site in carbon_idxs:
        for a in atoms:
            a.lone_pairs = 0
            a.radical_electrons = 0
        ok = True
        for i in heavy_idxs:
            rad = 1 if i == site else 0
            if not can_assign(atoms[i], rad):
                ok = False
                break
        if ok:
            return True

    # finally allow H•
    for site in H_idxs:
        for a in atoms:
            a.lone_pairs = 0
            a.radical_electrons = 0
        if not all(can_assign(atoms[i], 0) for i in heavy_idxs):
            continue
        atoms[site].radical_electrons = 1
        return True

    return False


def assign_formal_charges(mol: Molecule) -> None:
    """
    Compute and set atom.charge from lone pairs and bond orders,
    but do *not* charge any radical center (atom.radical_electrons > 0).
      FC = group_valence_electrons
           − (nonbonding electrons + bonding electrons/2)
    where nonbonding = 2*lone_pairs, bonding_electrons/2 = sum(bond_order).
    """
    for atom in mol.atoms:
        # if this atom carries any unpaired electron, leave it neutral
        if atom.radical_electrons:
            atom.charge = 0
            continue

        # get a plain‐string symbol (in case atom.element is an Element object)
        elem = atom.element
        sym = getattr(elem, 'symbol', elem)

        # group valence electrons (fallback to atomic_number−2)
        ge = _VALENCE_ELECTRONS.get(sym, NUMBER_BY_SYMBOL[sym] - 2)

        nonbond = atom.lone_pairs * 2
        bonding = sum(b.order for b in atom.edges.values())
        atom.charge = ge - (nonbond + bonding)


def get_representative_resonance(mol: Molecule) -> Molecule:
    """
    Generate all safe resonance variants and pick the canonical one.
    """
    variants = generate_resonance_structures_safely(mol)
    if not variants:
        return mol
    for v in variants:
        if mol.copy().is_isomorphic(v.copy()):
            return v
    return variants[0]
