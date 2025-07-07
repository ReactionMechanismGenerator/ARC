"""
Perceive 3D Cartesian coordinates and generate a representative resonance structure as a Molecule object.
"""

import heapq
import numpy as np
from itertools import combinations, product
from typing import Any, Iterable

from arc.common import NUMBER_BY_SYMBOL, distance_matrix, get_bonds_from_dmat, get_logger, get_single_bond_length
from arc.exceptions import AtomTypeError
from arc.molecule.filtration import get_octet_deviation
from arc.molecule.molecule import Atom, Bond, Molecule
from arc.molecule.resonance import generate_resonance_structures_safely

logger = get_logger()

# valence electrons for main-group elements
_VALENCE_ELECTRONS = {
    'H': 1, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7,
    'P': 5, 'S': 6, 'Cl': 7, 'Br': 7, 'I': 7,
}


def perceive_molecule_from_xyz(
    xyz: dict[str, Any] | str,
    charge: int | None = None,
    multiplicity: int | None = None,
    n_radicals: int | None = None,
    n_fragments: int | None = None,
    single_bond_tolerance: float = 1.20,
) -> Molecule | None:
    """
    Infer a valid localized Lewis‐structure ARC Molecule from Cartesian coordinates.

    Args:
        xyz (dict or str): Either an ARC‐format xyz dict or an XYZ string.
        charge (int): Total formal charge.
        multiplicity (int, optional): Spin multiplicity (2S+1). If None, inferred.
        n_radicals (int, optional): Number of radical centers. If None, defaults to multiplicity-1.
        n_fragments (int, optional): Expected number of fragments. Defaults to 1.
        single_bond_tolerance (float): Tolerance for single bond length perception, default is 1.20.

    Returns:
        Molecule | None: The perceived Molecule, or None if perception failed.
    """
    charge = charge or 0
    n_fragments = n_fragments or 1
    xyz_dict = validate_xyz(xyz)
    if xyz_dict is None:
        return None

    symbols, coords = xyz_dict["symbols"], xyz_dict["coords"]

    # build the zero‐order graph and find its connected components
    dmat = distance_matrix(a=np.array(xyz_to_coords_list(xyz_dict)),
                           b=np.array(xyz_to_coords_list(xyz_dict)))
    raw_bonds = get_bonds_from_dmat(dmat, symbols, charges=[0] * len(symbols), n_fragments=n_fragments)

    bonds: list[tuple[int, int]] = list()
    for i, j in raw_bonds:
        maxd = get_single_bond_length(symbols[i], symbols[j]) * (single_bond_tolerance)
        if dmat[i, j] <= maxd:
            bonds.append((i, j))

    # immediately detach into fragments if requested
    atoms0 = [Atom(element=sym, radical_electrons=0, charge=0, lone_pairs=0, coords=np.array(c))
              for sym, c in zip(symbols, coords)]
    mol0 = Molecule(atoms=atoms0)
    for i, j in bonds:
        mol0.add_bond(Bond(mol0.atoms[i], mol0.atoms[j], order=1))

    # find connected‐component index lists
    visited = set()
    fragments = []
    for idx in range(len(atoms0)):
        if idx in visited:
            continue
        stack, comp = [idx], []
        while stack:
            i = stack.pop()
            if i in visited:
                continue
            visited.add(i)
            comp.append(i)
            for neigh in mol0.atoms[i].edges:
                j = mol0.atoms.index(neigh)
                if j not in visited:
                    stack.append(j)
        fragments.append(comp)

    # drop any all‐H fragments (very common small microwave peaks, etc.)
    non_h = [f for f in fragments if not all(symbols[i] == 'H' for i in f)]
    if non_h:
        fragments = non_h

    # if we expected multiple fragments, hand off to the multi‐frag helper
    if len(fragments) != 1:
        if len(fragments) == n_fragments:
            return _combine_fragments(symbols, coords, fragments, charge)
        logger.warning(f"Expected {n_fragments} fragments, found {len(fragments)}")
        return None

    # otherwise fall back on the existing single‐molecule code
    # (note: symbols, coords, mol0 already built above)
    multiplicity = multiplicity or infer_multiplicity(symbols, charge)
    n_radicals = n_radicals or (multiplicity - 1)

    rep = generate_lewis_structure(mol0, charge, multiplicity, n_radicals)
    if rep is None:
        logger.warning("Falling back to all‐single‐bond Lewis structure")
        rep = mol0.copy(deep=True)
        rep.multiplicity = multiplicity
        adjust_atoms_for_octet(rep, n_radicals)
        assign_formal_charges(rep)

    rep = get_representative_resonance_structure(rep)
    rep = reduce_charge_separation(rep, n_radicals)
    assign_formal_charges(rep)
    enforce_target_charge(rep, charge)
    rep.multiplicity = multiplicity
    if not validate_atom_types(rep, charge, multiplicity, n_radicals):
        rep = _resurrect_molecule(mol=rep, total_charge=charge, multiplicity=multiplicity, n_radicals=n_radicals)
    return rep


def _combine_fragments(
    symbols: tuple[str, ...],
    coords: tuple[tuple[float, float, float], ...],
    fragments: list[list[int]],
    total_charge: int,
) -> Molecule:
    """
    Build each fragment separately (with its own Lewis/resonance pipeline)
    and then stitch their atoms & bonds into one disconnected Molecule,
    choosing the distribution of fragment charges that minimizes
    the sum of absolute charges while keeping each fragment valid.
    """
    # helper to perceive one fragment given a desired fragment charge
    def _perceive_frag(frag_idx: int, frag_charge: int) -> Molecule | None:
        idxs = fragments[frag_idx]
        sub_symbols = tuple(symbols[i] for i in idxs)
        sub_coords  = tuple(coords[i]   for i in idxs)
        frag_mult   = infer_multiplicity(sub_symbols, frag_charge)
        frag_rad    = frag_mult - 1
        sub_xyz = {"symbols": sub_symbols, "coords": sub_coords}
        return perceive_molecule_from_xyz(
            sub_xyz,
            charge=frag_charge,
            multiplicity=frag_mult,
            n_radicals=frag_rad,
            n_fragments=1
        )

    nfr = len(fragments)
    # generate all small integer partitions of total_charge into nfr fragments
    # here we only try splits in [-2..2] per frag; adjust range if you expect bigger
    charge_range = range(-2, 3)
    best_mol = None
    best_sep = float('inf')

    for charges_tuple in product(charge_range, repeat=nfr):
        if sum(charges_tuple) != total_charge:
            continue
        sep = sum(abs(c) for c in charges_tuple)
        if sep >= best_sep:
            continue

        # perceive each fragment under its assigned charge
        submols = []
        ok = True
        for idx_frag, fch in enumerate(charges_tuple):
            submol = _perceive_frag(idx_frag, fch)
            if submol is None or submol.get_net_charge() != fch:
                ok = False
                break
            submols.append(submol)
        if not ok:
            continue

        # glue them
        all_atoms, all_bonds = [], []
        offset = 0
        for sm in submols:
            for a in sm.atoms:
                all_atoms.append(Atom(
                    element=a.element,
                    radical_electrons=a.radical_electrons,
                    charge=a.charge,
                    lone_pairs=a.lone_pairs,
                    coords=a.coords.copy()
                ))
            for e in sm.get_all_edges():
                i = sm.atoms.index(e.vertex1) + offset
                j = sm.atoms.index(e.vertex2) + offset
                all_bonds.append((i, j, e.order))
            offset += len(sm.atoms)

        cand = Molecule(atoms=all_atoms)
        for i, j, o in all_bonds:
            cand.add_bond(Bond(cand.atoms[i], cand.atoms[j], order=o))
        # final bookkeeping
        try:
            cand.update(sort_atoms=False)
        except AtomTypeError:
            continue

        # keep the best (lowest separation) one
        best_mol, best_sep = cand, sep
        if sep == 0:
            break

    if best_mol is None:
        raise ValueError(f"Failed to find any valid charge‐split for fragments {fragments}")

    # set overall multiplicity & (re)assign formal charges to be safe
    best_mol.multiplicity = max(sm.multiplicity for sm in submols)
    assign_formal_charges(best_mol)
    enforce_target_charge(best_mol, total_charge)
    return best_mol


def validate_xyz(xyz: dict[str, Any] | str) -> dict[str, Any] | None:
    """
    Accept either an ARC-format xyz dict or a plain XYZ string.

    For dict input, requires at least 'symbols' and 'coords' keys.
    For string input, build {'symbols': (...), 'isotopes': (...), 'coords': (...)}.
    """
    if isinstance(xyz, dict):
        if "symbols" in xyz and "coords" in xyz:
            return xyz
        logger.warning("Provided dict is missing 'symbols' or 'coords'.")
        return None
    if isinstance(xyz, str):
        symbols, coords = list(), list()
        for line in xyz.strip().splitlines():
            parts = line.split()
            if len(parts) != 4:
                continue
            sym, xs, ys, zs = parts
            try:
                x, y, z = float(xs), float(ys), float(zs)
            except ValueError:
                logger.warning(f"Could not parse coordinates on line: {line}")
                return None
            symbols.append(sym)
            coords.append((x, y, z))
        if not symbols:
            logger.warning("No valid atom lines found in XYZ string.")
            return None
        return {"symbols": tuple(symbols), "coords": tuple(coords)}
    logger.warning(f"Unsupported XYZ input type: {type(xyz)}")
    return None


def xyz_to_coords_list(xyz: dict) -> list[list[float]]:
    """
    Get the coords part of an xyz dict as a (mutable) list of lists (rather than a tuple of tuples).

    Args:
        xyz (dict): The ARC xyz format.

    Returns: Optional[List[List[float]]]
        The coordinates.
    """
    coords_tuple = xyz['coords']
    coords_list = list()
    for coords_tup in coords_tuple:
        coords_list.append([coords_tup[0], coords_tup[1], coords_tup[2]])
    return coords_list


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
    n_radicals: int,
) -> Molecule | None:
    """
    A* search over bond‐order assignments (1→2→3) to satisfy octet, charge & spin;
    aggressively prunes any partial assignment exceeding atomic valence.
    """
    max_bond = list()
    for at in base_mol.atoms:
        sym = getattr(at.element, 'symbol', at.element)
        ge = _VALENCE_ELECTRONS.get(sym, NUMBER_BY_SYMBOL[sym] - 2)
        max_bond.append(ge)

    bond_pairs = [(base_mol.atoms.index(e.vertex1), base_mol.atoms.index(e.vertex2))
                  for e in base_mol.get_all_edges()]
    initial = tuple(1 for _ in bond_pairs)
    seen = {initial}
    queue: list[tuple[int, int, int, tuple[int, ...]]] = list()
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

        if adjust_atoms_for_octet(mol, n_radicals):
            assign_formal_charges(mol)
            if mol.get_net_charge() == total_charge and validate_atom_types(mol, total_charge, multiplicity, n_radicals):
                return mol

        for idx in range(len(orders)):
            if orders[idx] < 3:
                new_orders = list(orders)
                new_orders[idx] += 1
                tpl = tuple(new_orders)
                if tpl not in seen:
                    seen.add(tpl)
                    heapq.heappush(queue, (increments + 1 + heuristic(tpl), increments + 1, counter, tpl))
                    counter += 1

    return None


def adjust_atoms_for_octet(
    mol: Molecule,
    target_rad: int,
) -> bool:
    """
    Place exactly `target_rad` unpaired electrons and fill lone‐pairs to minimize octet deviations.
    But avoid placing radicals on O or S atoms that already have two bonds, unless no other site will work.

    Args:
        mol (Molecule): The molecule to adjust.
        target_rad (int): The number of radical electrons to place.

    Returns:
        bool: True if the adjustment was successful, False otherwise.
    """
    atoms = mol.atoms
    n = len(atoms)
    # trivial single‐atom case
    if n == 1:
        a = atoms[0]
        a.lone_pairs = 0
        a.radical_electrons = target_rad
        return True

    # precompute bond‐order totals
    bond_sums = [sum(e.order for e in a.edges.values()) for a in atoms]

    halogens = {'F', 'Cl', 'Br', 'I'}

    def allowed_cores(a: Atom, has_rad: bool) -> set[int]:
        sym = getattr(a.element, 'symbol', a.element)
        if a.is_hydrogen():
            cores = {2}
        else:
            cores = {8}
            if sym == 'S':
                cores |= {10, 12}
            if sym == 'P':
                cores |= {10}
        if has_rad:
            cores |= {c - 1 for c in cores if c > 1}
        return cores

    def test_sites(sites: tuple[int, ...]) -> Molecule | None:
        cand = mol.copy(deep=True)
        for x in cand.atoms:
            x.lone_pairs = 0
            x.radical_electrons = 0
        for idx in sites:
            cand.atoms[idx].radical_electrons += 1
        for i, x in enumerate(cand.atoms):
            has = x.radical_electrons > 0
            for core in sorted(allowed_cores(x, has)):
                rem = core - x.radical_electrons - 2 * bond_sums[i]
                if rem >= 0 and rem % 2 == 0 and rem // 2 <= 3:
                    x.lone_pairs = rem // 2
                    break
            else:
                return None
        if sum(x.radical_electrons for x in cand.atoms) != target_rad:
            return None
        return cand

    heavy = [i for i, a in enumerate(atoms) if not a.is_hydrogen()]
    hydros = [i for i, a in enumerate(atoms) if     a.is_hydrogen()]

    # carve out “bad” radical sites: O/S with two bonds, **or any halogen** with ≥1 bond
    bad_rad = []
    for i in heavy:
        sym = getattr(atoms[i].element, 'symbol', atoms[i].element)
        if (sym in ('O', 'S') and bond_sums[i] == 2) or (sym in halogens and bond_sums[i] >= 1):
            bad_rad.append(i)
    good_rad = [i for i in heavy if i not in bad_rad]

    candidates: list[tuple[float, Molecule]] = []

    # 1) closed‐shell
    if target_rad == 0:
        c0 = test_sites(())
        if c0:
            candidates.append((get_octet_deviation(c0), c0))

    # 2) single radical
    elif target_rad == 1:
        # a) if there's a true “dangling” H (bond_sums==0), allow it first
        unbonded_h = [i for i in hydros if bond_sums[i] == 0]
        if unbonded_h:
            # score all atoms by fraction of octet missing → pick highest
            fracs: list[tuple[float, int]] = []
            for i, a in enumerate(atoms):
                cores = allowed_cores(a, False)
                if not cores:
                    continue
                core = max(cores)
                missing = core - 2 * bond_sums[i]
                if missing > 0:
                    fracs.append((missing / core, i))
            for _, idx in sorted(fracs, key=lambda x: x[0], reverse=True):
                c1 = test_sites((idx,))
                if c1:
                    candidates.append((get_octet_deviation(c1), c1))
                    break
        else:
            # b) otherwise use our prioritization over “good” sites only
            #    (note: halogens will have been carved out into bad_rad)
            term_het = [i for i in good_rad
                        if getattr(atoms[i].element, 'symbol', atoms[i].element) not in ('C', 'H')
                        and bond_sums[i] == 1]
            carbs    = [i for i in good_rad
                        if getattr(atoms[i].element, 'symbol', atoms[i].element) == 'C']
            nonterm_het = [i for i in good_rad
                           if getattr(atoms[i].element, 'symbol', atoms[i].element) not in ('C', 'H')
                           and bond_sums[i] > 1]
            pools = (term_het, carbs, nonterm_het, hydros, bad_rad)
            for pool in pools:
                for idx in pool:
                    c1 = test_sites((idx,))
                    if c1:
                        candidates.append((get_octet_deviation(c1), c1))
                        break
                if candidates:
                    break

    # 3) multi‐radical (unchanged)
    else:
        if len(good_rad) >= target_rad:
            for combo in combinations(good_rad, target_rad):
                cN = test_sites(combo)
                if cN:
                    candidates.append((get_octet_deviation(cN), cN))
        if not candidates:
            if len(heavy) >= target_rad:
                for combo in combinations(heavy, target_rad):
                    cN = test_sites(combo)
                    if cN:
                        candidates.append((get_octet_deviation(cN), cN))
            if not candidates:
                for i in heavy:
                    ca = test_sites((i,) * target_rad)
                    if ca:
                        candidates.append((get_octet_deviation(ca), ca))

    if not candidates:
        return False

    # pick the minimal‐deviation solution
    _, best = min(candidates, key=lambda x: x[0])
    for orig, new in zip(mol.atoms, best.atoms):
        orig.lone_pairs        = new.lone_pairs
        orig.radical_electrons = new.radical_electrons
    return True


def assign_formal_charges(mol: Molecule) -> None:
    """
    Assign formal charges to atoms in the molecule based on their valence electrons,
    """
    for a in mol.atoms:
        if a.radical_electrons:
            a.charge = 0
        else:
            sym = getattr(a.element, 'symbol', a.element)
            ge = _VALENCE_ELECTRONS.get(sym, NUMBER_BY_SYMBOL[sym] - 2)
            a.charge = ge - (2 * a.lone_pairs + sum(b.order for b in a.edges.values()))


def get_representative_resonance_structure(mol: Molecule) -> Molecule:
    """
    Generate resonance structures and return a representative one.
    """
    variants = generate_resonance_structures_safely(mol)
    if not variants:
        return mol

    charge_seps = [sum(abs(atom.charge) for atom in variant.atoms)
                   for variant in variants]

    for variant, sep in zip(variants, charge_seps):
        if sep == 0:
            return variant

    min_sep = min(charge_seps)
    for variant, sep in zip(variants, charge_seps):
        if sep == min_sep:
            return variant

    return variants[0]


def enforce_target_charge(mol: Molecule, target_charge: int) -> None:
    """
    Adjust the net charge of the molecule to match the target charge
    """
    current = mol.get_net_charge()
    delta = target_charge - current
    if delta == 0:
        return

    # 1) use the biggest radical center
    radicals = sorted([a for a in mol.atoms if a.radical_electrons > 0], key=lambda a: -a.radical_electrons)
    if radicals:
        recipient = radicals[0]
    else:
        het = [a for a in mol.atoms if getattr(a.element, 'symbol', a.element) not in ('C', 'H')]
        recipient = het[0] if het else mol.atoms[0]
    recipient.charge += delta


def validate_atom_types(mol: Molecule, charge: int, multiplicity: int, n_radicals: int | None) -> bool:
    """
    Return True if `mol` has a valid atom‐type assignment, False if
    an AtomTypeError is raised (i.e. any atom violates its valence rules).
    """
    n_radicals = n_radicals or (multiplicity - 1)
    try:
        mol.copy(deep=True).update()
    except AtomTypeError:
        return False
    if mol.get_net_charge() != charge or mol.multiplicity != multiplicity:
        return False
    if get_octet_deviation(mol) > n_radicals:
        return False
    return True


def update_mol(mol: Molecule, multiplicity: int, charge: int) -> Molecule:
    """
    Update the molecule by reassigning atom types and formal charges.
    This is a fallback for when the initial atom type assignment fails.
    """
    mol_copy = mol.copy(deep=True)
    mol_copy.update(sort_atoms=False, raise_atomtype_exception=False)
    if mol_copy.get_net_charge() == charge and mol_copy.multiplicity == multiplicity:
        return mol_copy
    mol.update_charge()
    return mol


def _resurrect_molecule(
    mol: Molecule,
    total_charge: int,
    multiplicity: int,
    n_radicals: int,
) -> Molecule:
    """
    Attempt to fix an invalid Molecule by resetting to all single bonds,
    re‐assigning lone pairs/radicals, and re‐distributing formal charges.
    """
    cand = mol.copy(deep=True)
    # remove all partial charges
    for a in cand.atoms:
        a.charge = 0
    if validate_atom_types(cand, total_charge, multiplicity, n_radicals):
        return cand
    if mol.fingerprint == 'C00H02N02O00S00' and multiplicity == 3 and any(sum(e.order for e in a.edges.values()) == 3 for a in mol.atoms):
        # hard-code for N2H3(T)
        for atom in mol.atoms:
            if atom.is_nitrogen():
                if sum(e.order for e in atom.edges.values()) == 3:
                    atom.lone_pairs, atom.radical_electrons = 1, 0
                elif sum(e.order for e in atom.edges.values()) == 1:
                    atom.lone_pairs, atom.radical_electrons = 1, 2

    logger.error(f"Failed to resurrect the perception of {mol.to_smiles()} with fingerprint {mol.fingerprint}, "
                 f"multiplicity {multiplicity}, and charge {total_charge}.")
    return mol


def reduce_charge_separation(mol: Molecule, target_rad: int) -> Molecule:
    """
    Reduce charge separation by increasing bond orders where possible,
    but skip for radical species to preserve unpaired electron localization.
    """
    best = mol.copy(deep=True)
    best.multiplicity = mol.multiplicity
    assign_formal_charges(best)
    base_sep = sum(abs(a.charge) for a in best.atoms)
    if base_sep <= 1:
        return best

    pairs = [(best.vertices.index(e.vertex1), best.vertices.index(e.vertex2))
             for e in best.get_all_edges()]

    improved = True
    while improved:
        improved = False
        curr = best
        dev0 = get_octet_deviation(curr)
        sep0 = sum(abs(a.charge) for a in curr.atoms)

        for i, j in pairs:
            e = curr.get_edge(curr.atoms[i], curr.atoms[j])
            if e.order >= 3: continue
            cand = curr.copy(deep=True)
            cand.multiplicity = curr.multiplicity
            cand.get_edge(cand.atoms[i], cand.atoms[j]).order += 1
            if not adjust_atoms_for_octet(cand, target_rad):
                continue
            assign_formal_charges(cand)
            if get_octet_deviation(cand) == dev0 and sum(abs(a.charge) for a in cand.atoms) < sep0:
                best = cand
                improved = True
                break

    return best
