"""
Perceive 3D Cartesian coordinates and generate a representative resonance structure as a Molecule object.
"""

import heapq
from typing import Any, Iterable

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
    """
    Infer a valid localized Lewis‐structure ARC Molecule from Cartesian coordinates.

    Args:
        xyz (dict or str): Either an ARC‐format xyz dict or an XYZ string.
        charge (int): Total formal charge.
        multiplicity (int, optional): Spin multiplicity (2S+1). If None, inferred.

    Returns:
        Molecule | None: The perceived Molecule, or None if perception failed.
    """
    xyz_dict = validate_xyz(xyz)
    if xyz_dict is None:
        return None

    symbols, coords = xyz_dict["symbols"], xyz_dict["coords"]
    multiplicity = multiplicity or infer_multiplicity(symbols, charge)

    # Build distance matrix & connectivity
    try:
        dmat = xyz_to_dmat(xyz_dict)
        bonds = get_bonds_from_dmat(dmat, symbols, charges=[0] * len(symbols))
    except ValueError as e:
        logger.warning(f"Failed to perceive bonds from XYZ: {e}")
        return None

    # Create atoms + single bonds
    atoms = [Atom(element=sym, radical_electrons=0, charge=0, lone_pairs=0, coords=np.array(coord))
        for sym, coord in zip(symbols, coords)]
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
    A* search over bond‐order assignments (1→2→3) to satisfy octet, charge & spin;
    aggressively prunes any partial assignment exceeding atomic valence.
    """
    # precompute per-atom max bonding electrons (in bond‐order units)
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
    queue: list[tuple[int, int, tuple[int, ...]]] = []
    counter = 0

    # f = g + heuristic; here heuristic = number of atoms still below valence
    def heuristic(orders: tuple[int, ...]) -> int:
        # partial bond sums
        sums = [0]*len(base_mol.atoms)
        for (i, j), o in zip(bond_pairs, orders):
            sums[i] += o
            sums[j] += o
        # count how many atoms haven't reached max_bond yet
        return sum(1 for idx, s in enumerate(sums) if s < max_bond[idx])

    # push (f, g, counter, orders)
    heapq.heappush(queue, (heuristic(initial), 0, counter, initial))
    counter += 1

    while queue:
        f, increments, _, orders = heapq.heappop(queue)

        # FEASIBILITY PRUNE: no atom may exceed its valence
        partial = [0]*len(base_mol.atoms)
        for (i, j), o in zip(bond_pairs, orders):
            partial[i] += o
            partial[j] += o
        if any(partial[k] > max_bond[k] for k in range(len(partial))):
            continue

        # apply this bond‐order pattern
        mol = base_mol.copy(deep=True)
        for (i, j), order in zip(bond_pairs, orders):
            mol.get_edge(mol.atoms[i], mol.atoms[j]).order = order

        # try to satisfy octet + spin, and check net charge
        if adjust_atoms_for_octet(mol, multiplicity):
            assign_formal_charges(mol)
            if mol.get_net_charge() == total_charge:
                return mol

        # branch out by bumping each bond order
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
      - if site is radical center, allow a 7-electron core,
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

    def test_site(site: int) -> Molecule | None:
        """
        Copy `mol`, place exactly one radical at `site` (or none if site<0),
        compute lone_pairs algebraically. Return that candidate, or None.
        """
        cand = mol.copy(deep=True)
        # reset counts
        for at in cand.atoms:
            at.lone_pairs = 0
            at.radical_electrons = 0

        rad_count = 0
        for i, at in enumerate(cand.atoms):
            rad = 1 if i == site and target_rad == 1 else 0
            B = bond_sums[i]

            # build allowed electron totals
            if at.is_hydrogen():
                allowed = {2}
            else:
                sym = getattr(at.element, 'symbol', at.element)
                # heavy default octet
                allowed = {8}
                # sulfur can expand
                if sym == 'S':
                    allowed |= {10, 12}
                # if this is the radical site, also allow 7
                if rad == 1:
                    allowed.add(7)

            # pick the first v in allowed that yields an integer LP ≤3
            placed = False
            for v in allowed:
                rem = v - rad - 2 * B
                if rem >= 0 and rem % 2 == 0:
                    lp = rem // 2
                    if lp <= 3:
                        at.lone_pairs = lp
                        at.radical_electrons = rad
                        rad_count += rad
                        placed = True
                        break
            if not placed:
                return None

        if rad_count != target_rad:
            return None
        return cand

    # gather all candidates
    candidates: list[tuple[float, Molecule]] = []
    heavy_idxs = [i for i, a in enumerate(atoms) if not a.is_hydrogen()]
    H_idxs     = [i for i, a in enumerate(atoms) if     a.is_hydrogen()]

    # closed‐shell
    if target_rad == 0:
        cand = test_site(-1)
        if cand is not None:
            dev = get_octet_deviation(cand)
            candidates.append((dev, cand))
    else:
        # hetero first, then carbon, then H
        hetero = [i for i in heavy_idxs
                  if getattr(atoms[i].element, 'symbol', atoms[i].element) not in ('C', 'H')]
        carbon = [i for i in heavy_idxs
                  if getattr(atoms[i].element, 'symbol', atoms[i].element) == 'C']
        for pool in (hetero, carbon, H_idxs):
            for site in pool:
                cand = test_site(site)
                if cand is not None:
                    dev = get_octet_deviation(cand)
                    candidates.append((dev, cand))
            if candidates:
                break

    if not candidates:
        return False

    # pick the candidate with minimal deviation
    _, best = min(candidates, key=lambda x: x[0])
    # copy back onto original mol
    for orig, new in zip(mol.atoms, best.atoms):
        orig.lone_pairs = new.lone_pairs
        orig.radical_electrons = new.radical_electrons

    return True


def assign_formal_charges(mol: Molecule) -> None:
    """
    Compute and set atom.charge from lone pairs and bond orders,
    but do *not* charge any radical center (atom.radical_electrons > 0).
      FC = group_valence_electrons
           − (nonbonding electrons + bonding electrons)
    where nonbonding = 2*lone_pairs, bonding = sum(bond_order).
    """
    for atom in mol.atoms:
        if atom.radical_electrons:
            atom.charge = 0
            continue
        sym = getattr(atom.element, 'symbol', atom.element)
        ge = _VALENCE_ELECTRONS.get(sym, NUMBER_BY_SYMBOL[sym] - 2)
        nonbond = 2 * atom.lone_pairs
        bonding = sum(b.order for b in atom.edges.values())
        atom.charge = ge - (nonbond + bonding)


def get_representative_resonance(mol: Molecule) -> Molecule:
    """
    Generate all safe resonance variants and apply the canonical one
    *in place* to `mol` (preserving mol.atoms order & identity).
    """
    variants = generate_resonance_structures_safely(mol)
    if not variants:
        return mol

    # try to find the variant whose graph is isomorphic to mol
    for v in variants:
        # strict isomorphism just on connectivity + bond orders + electrons
        if mol.copy(deep=True).is_isomorphic(v.copy(deep=True), strict=True):
            chosen = v
            break
    else:
        chosen = variants[0]

    # get the mapping mol.atoms[i] -> chosen.atoms[j]
    iso_map = mol.find_isomorphism(chosen, strict=True)
    if not iso_map:
        return mol

    # update bond orders
    for a in mol.atoms:
        for b, edge in a.edges.items():
            # only update each edge once
            if mol.vertices.index(a) < mol.vertices.index(b):
                ae = mol.get_edge(a, b)
                ce = chosen.get_edge(iso_map[a], iso_map[b])
                ae.order = ce.order

    # update lone pairs & radicals on each atom
    for a in mol.atoms:
        ca = iso_map[a]
        a.lone_pairs        = ca.lone_pairs
        a.radical_electrons = ca.radical_electrons

    return mol
