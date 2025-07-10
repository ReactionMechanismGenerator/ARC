"""
Perceive 3D Cartesian coordinates and generate a representative resonance structure as a Molecule object.
"""

import heapq
import numpy as np
from itertools import combinations, product
from math import dist
from typing import Any, Iterable

from openbabel import pybel

from arc.common import NUMBER_BY_SYMBOL, distance_matrix, get_bonds_from_dmat, get_logger, get_single_bond_length
from arc.exceptions import AtomTypeError, InputError, SanitizationError
from arc.molecule.filtration import get_octet_deviation
from arc.molecule.molecule import Atom, Bond, Molecule
from arc.molecule.resonance import generate_resonance_structures_safely
from arc.species.xyz_to_smiles import xyz_to_smiles

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
    is_fragment: bool = False,
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
        is_fragment (bool): If True, treat the input as a fragment rather than a full molecule.

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
        return None

    # otherwise fall back on the existing single‐molecule code
    # (note: symbols, coords, mol0 already built above)
    multiplicity = multiplicity or infer_multiplicity(symbols, charge)
    n_radicals = n_radicals or (multiplicity - 1)

    rep = generate_lewis_structure(mol0, charge, multiplicity, n_radicals, is_fragment=is_fragment)
    if rep is None:
        rep = mol0.copy(deep=True)
        rep.multiplicity = multiplicity
        adjust_atoms_for_octet(rep, n_radicals)
        assign_formal_charges(rep)

    rep = get_representative_resonance_structure(rep)
    rep = reduce_charge_separation(rep, n_radicals)
    assign_formal_charges(rep)
    enforce_target_charge(rep, charge)
    rep.multiplicity = multiplicity
    if not is_mol_valid(rep, charge, multiplicity, n_radicals, is_fragment=is_fragment):
        rep = alternative_perception(mol=rep, total_charge=charge, multiplicity=multiplicity, n_radicals=n_radicals, xyz=xyz, is_fragment=is_fragment)
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
    automatically distributing total_charge across fragments to minimize
    the sum of absolute fragment charges. Any atom index not listed in
    fragments will be added to the nearest fragment by centroid distance.
    """
    nfr = len(fragments)

    # 1) ensure every atom is covered by some fragment
    all_idxs = set(range(len(symbols)))
    covered = set().union(*fragments)
    missing = all_idxs - covered
    if missing:
        centroids = list()
        for idxs in fragments:
            pts = [coords[i] for i in idxs]
            centroids.append(tuple(sum(c)/len(c) for c in zip(*pts)))
        # assign missing atoms to nearest fragment
        for i in missing:
            x,y,z = coords[i]
            distances = [(x-cx)**2 + (y-cy)**2 + (z-cz)**2 for cx,cy,cz in centroids]
            j = int(min(range(nfr), key=lambda k: distances[k]))
            fragments[j].append(i)
        for idxs in fragments:
            idxs.sort()

    # helper to perceive one fragment given a charge
    def _perceive_frag(idxs: list[int], charge: int) -> Molecule | None:
        sub_symbols = tuple(symbols[i] for i in idxs)
        sub_coords  = tuple(coords[i]   for i in idxs)
        mult = infer_multiplicity(sub_symbols, charge)
        rad  = mult - 1
        sub_xyz = {"symbols": sub_symbols, "coords": sub_coords}
        return perceive_molecule_from_xyz(sub_xyz,
                                          charge=charge,
                                          multiplicity=mult,
                                          n_radicals=rad,
                                          n_fragments=1,
                                          is_fragment=True,
                                          )

    # 2) search over all splits of total_charge into nfr integers (small range)
    charge_range = range(-2, 3)
    best_mol = None
    best_sep = float('inf')

    for charges in product(charge_range, repeat=nfr):
        if sum(charges) != total_charge:
            continue
        sep = sum(abs(c) for c in charges)
        if sep >= best_sep:
            continue

        # perceive each fragment
        submols = list()
        ok = True
        for idx_frag, frag_charge in enumerate(charges):
            sm = _perceive_frag(fragments[idx_frag], frag_charge)
            if sm is None or sm.get_net_charge() != frag_charge:
                ok = False
                break
            submols.append(sm)
        if not ok:
            continue

        # stitch atoms & bonds
        all_atoms, all_bonds = list(), list()
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
        for i, j, order in all_bonds:
            cand.add_bond(Bond(cand.atoms[i], cand.atoms[j], order=order))
        try:
            cand.update(sort_atoms=False)
        except AtomTypeError:
            continue

        best_mol, best_sep = cand, sep
        if sep == 0:
            break

    if best_mol is None:
        raise ValueError(f"Could not find a valid charge split for {fragments}")

    best_mol.multiplicity = max(sm.multiplicity for sm in submols)
    assign_formal_charges(best_mol)
    enforce_target_charge(best_mol, total_charge)
    _add_interfragment_bonds(best_mol, fragments, coords)

    # restore original atom order
    idx_map: dict[int, int] = dict()
    offset = 0
    for frag in fragments:
        for local_idx, orig_idx in enumerate(frag):
            idx_map[orig_idx] = offset + local_idx
        offset += len(frag)
    n = len(symbols)
    reordered = [None] * n
    for orig_idx, mol_idx in idx_map.items():
        reordered[orig_idx] = best_mol.atoms[mol_idx]
    best_mol.atoms = reordered

    if best_mol.get_net_charge() != total_charge:
        raise RuntimeError(f"Post-combine net charge {best_mol.get_net_charge()} != target {total_charge}")
    return best_mol


def _add_interfragment_bonds(
    mol: Molecule,
    fragments: list[list[int]],
    coords: tuple[tuple[float, float, float], ...],
) -> None:
    """
    If there are multiple fragments, connect each fragment pair by a single bond
    between their closest atom pair—but only if at least one of those atoms is a radical.
    Modifies `mol` in place and then updates its topology.
    """
    if len(fragments) < 2:
        return

    # map original‐atom index → index in mol.atoms
    idx_map: dict[int,int] = dict()
    offset = 0
    for frag in fragments:
        for local_idx, orig_idx in enumerate(frag):
            idx_map[orig_idx] = offset + local_idx
        offset += len(frag)

    for frag_i in range(len(fragments)):
        if frag_i == len(fragments) - 1:
            break
        frag_1, frag_2 = fragments[frag_i], fragments[frag_i + 1]
        i0, j0 = min(((i, j) for i in frag_1 for j in frag_2), key=lambda pair: dist(coords[pair[0]], coords[pair[1]]))
        a1 = mol.atoms[idx_map[i0]]
        a2 = mol.atoms[idx_map[j0]]

        if a1.radical_electrons > 0 or a2.radical_electrons > 0:
            bond_order = 1.0 if n_missing_electrons(a1) or n_missing_electrons(a2) else 0.05
        else:
            bond_order = 0.0
        mol.add_bond(Bond(a1, a2, order=bond_order))


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
    is_fragment: bool = False,
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
            if mol.get_net_charge() == total_charge and is_mol_valid(mol, total_charge, multiplicity, n_radicals, is_fragment):
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
            fracs: list[tuple[float, int]] = list()
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
        orig.lone_pairs = new.lone_pairs
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
    radicals = sorted([a for a in mol.atoms if a.radical_electrons > 0], key=lambda a: -a.radical_electrons)
    if radicals:
        recipient = radicals[0]
    else:
        het = [a for a in mol.atoms if getattr(a.element, 'symbol', a.element) not in ('C', 'H')]
        recipient = het[0] if het else mol.atoms[0]
    recipient.charge += delta


def is_mol_valid(
        mol: Molecule | None,
        charge: int,
        multiplicity: int | None,
        n_radicals: int | None,
        is_fragment: bool = True,
) -> bool:
    """
    Return True if `mol` has a valid atom‐type assignment, False if
    an AtomTypeError is raised (i.e. any atom violates its valence rules).
    """
    if not mol:
        return False
    n_radicals = n_radicals or (multiplicity - 1) if multiplicity is not None else None
    try:
        mol.copy(deep=True).update()
    except AtomTypeError:
        return False
    if mol.get_net_charge() != charge or mol.multiplicity != multiplicity:
        return False
    if n_radicals is not None and get_octet_deviation(mol) > n_radicals:
        return False
    actual_radicals = mol.get_radical_count()
    if mol.multiplicity > actual_radicals + 1:
        return False
    # check if an even number of radicals results in an odd multiplicity, or vice versa
    if divmod(mol.multiplicity, 2)[1] == divmod(actual_radicals, 2)[1] and not charge:
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


def alternative_perception(
    mol: Molecule,
    total_charge: int,
    multiplicity: int,
    n_radicals: int,
    xyz: dict,
    is_fragment: bool = True,
) -> Molecule | None:
    """
    Attempt to perceive invalid Molecule using alternative methods.
    """
    # (1.) remove all partial charges
    mol_1 = mol.copy(deep=True)
    for a in mol_1.atoms:
        a.charge = 0
    mol_1 = _resurrect_molecule(mol_1, n_radicals)
    if is_mol_valid(mol_1, total_charge, multiplicity, n_radicals):
        return mol_1

    # (2.) hard-coding edge-cases
    mol_2 = mol.copy(deep=True)
    if mol_2.fingerprint == 'C00H02N02O00S00' and multiplicity == 3 and any(sum(e.order for e in a.edges.values()) == 3 for a in mol.atoms):
        # hard-code for N2H3(T)
        for atom in mol_2.atoms:
            if atom.is_nitrogen():
                if sum(e.order for e in atom.edges.values()) == 3:
                    atom.lone_pairs, atom.radical_electrons = 1, 0
                elif sum(e.order for e in atom.edges.values()) == 1:
                    atom.lone_pairs, atom.radical_electrons = 1, 2
        mol_2.multiplicity = multiplicity
        if is_mol_valid(mol_2, total_charge, multiplicity, n_radicals):
            return mol_2

    # (3.) use xyz => open babel (pybel) => InChI => RMG Molecule with bond orders
    mol_3 = mol.copy(deep=True)
    try:
        xyz_file_format = str(len(xyz['symbols'])) + '\n\n' + xyz_to_str(xyz) + '\n'
        pybel_mol = pybel.readstring('xyz', xyz_file_format)
    except (IOError, InputError):
        pybel_mol = None
    if pybel_mol is not None:
        if bool(len([atom.is_hydrogen() for atom in mol_3.atoms])):
            inchi = pybel_mol.write('inchi', opt={'F': None}).strip()  # Add a fixed H layer
        else:
            inchi = pybel_mol.write('inchi').strip()
        try:
            mol_3 = Molecule().from_inchi(inchi)
        except (AtomTypeError, ValueError, KeyError, TypeError):
            mol_3 = None
        if mol_3 is not None:
            try:
                order_atoms(ref_mol=mol_2, mol=mol_3)
            except SanitizationError:
                pass
            mol_3.multiplicity = multiplicity
            mol_3 = _resurrect_molecule(mol_3, n_radicals)
            if is_mol_valid(mol_3, total_charge, multiplicity, n_radicals):
                return mol_3

    # (4.) use xyz_to_smiles
    smiles_list = xyz_to_smiles(xyz=xyz, charge=total_charge)
    mol_4 = Molecule(smiles=smiles_list[0]) if smiles_list else None
    mol_4 = _resurrect_molecule(mol_4, n_radicals)
    if is_mol_valid(mol_4, total_charge, multiplicity, n_radicals):
        return mol_4

    return mol


def n_missing_electrons(atom: Atom) -> int:
    """
    Check if an atom is missing electrons to complete its octet.
    """
    if atom.is_hydrogen():
        return 2 - atom.radical_electrons - atom.radical_electrons
    occ_orbitals = atom.lone_pairs + atom.radical_electrons + atom.get_total_bond_order()
    return 4 - occ_orbitals


def _resurrect_molecule(
    mol: Molecule | None,
    n_radicals: int,
) -> Molecule | None:
    """
    Attempt to resurrect/sanitize a Molecule by fixing its perceived electronic structure.
    """
    if mol is None:
        return None
    mol = mol.copy(deep=True)
    for atom in mol.atoms:
        if atom.is_non_hydrogen():
            if (missing := n_missing_electrons(atom)) > 0:
                atom.radical_electrons += missing

    n_radicals = n_radicals or mol.get_radical_count()
    if mol.multiplicity < n_radicals + 1:
        carbenes, nitrenes = 0, 0
        for atom in mol.atoms:
            if atom.is_carbon() and atom.radical_electrons >= 2:
                carbenes += 1
            elif atom.is_nitrogen() and atom.radical_electrons >= 2:
                nitrenes += 1
        if 2 * (carbenes + nitrenes) + mol.multiplicity == n_radicals + 1:
            if carbenes:
                for atom in mol.atoms:
                    if atom.is_carbon() and atom.radical_electrons >= 2:
                        atom.lone_pairs += 1
                        atom.radical_electrons -= 2
            if nitrenes:
                for atom in mol.atoms:
                    if atom.is_nitrogen() and atom.radical_electrons >= 2:
                        for atom2, bond12 in atom.edges.items():
                            if atom2.is_sulfur() and atom2.lone_pairs >= 2 and bond12.is_single():
                                bond12.set_order_num(3)
                                atom2.lone_pairs -= 1
                                break
                            elif atom2.is_sulfur() and atom2.lone_pairs == 1 and bond12.is_single():
                                bond12.set_order_num(2)
                                atom2.lone_pairs -= 1
                                atom2.charge += 1
                                atom.charge -= 1
                                break
                            elif atom2.is_nitrogen() and atom2.lone_pairs == 1 and bond12.is_single():
                                bond12.set_order_num(2)
                                atom2.lone_pairs -= 1
                                atom.lone_pairs += 1
                                atom2.charge += 1
                                atom.charge -= 1
                                break
                        else:
                            atom.lone_pairs += 1
                        atom.radical_electrons -= 2
    if len(mol.atoms) == 1 and mol.multiplicity == 1 and mol.atoms[0].radical_electrons == 4:
        # This is a singlet atomic C or Si, convert all radicals to lone pairs
        mol.atoms[0].radical_electrons = 0
        mol.atoms[0].lone_pairs = 2
    actual_radicals = sum(atom.radical_electrons for atom in mol.atoms)
    neg_atoms = [atom for atom in mol.atoms if atom.charge < 0]
    if neg_atoms and actual_radicals < mol.multiplicity - 1:
        for atom in mol.atoms:
            atom.charge = 0
        center = neg_atoms[0]
        center.radical_electrons += 2
        mol.update(raise_atomtype_exception=False)
    return mol


def xyz_to_str(xyz_dict: dict) -> str | None:
    """
    Convert xyz dictionary to a simple XYZ-format string
    """
    lines = list()
    for sym, (x, y, z) in zip(xyz_dict['symbols'], xyz_dict['coords']):
        lines.append(f"{sym:4}{x:14.8f}{y:14.8f}{z:14.8f}")
    return "\n".join(lines)


def order_atoms(ref_mol, mol):
    """
    Order the atoms in `mol` to match the atom order in `ref_mol`.
    """
    ref_mol_is_iso_copy, mol_is_iso_copy = ref_mol.copy(deep=True), mol.copy(deep=True)
    ref_mol_find_iso_copy, mol_find_iso_copy = ref_mol.copy(deep=True), mol.copy(deep=True)

    ref_mol_is_iso_copy = create_a_single_bond_mol_copy(ref_mol_is_iso_copy)
    mol_is_iso_copy = create_a_single_bond_mol_copy(mol_is_iso_copy)
    ref_mol_find_iso_copy = create_a_single_bond_mol_copy(ref_mol_find_iso_copy)
    mol_find_iso_copy = create_a_single_bond_mol_copy(mol_find_iso_copy)

    if mol_is_iso_copy.is_isomorphic(ref_mol_is_iso_copy, save_order=True, strict=False):
        mapping = mol_find_iso_copy.find_isomorphism(ref_mol_find_iso_copy, save_order=True)
        if len(mapping):
            if isinstance(mapping, list):
                mapping = mapping[0]
            index_map = {ref_mol_find_iso_copy.atoms.index(val): mol_find_iso_copy.atoms.index(key)
                         for key, val in mapping.items()}
            mol.atoms = [mol.atoms[index_map[i]] for i, _ in enumerate(mol.atoms)]
        else:
            raise SanitizationError('Could not map molecules')
    else:
        raise SanitizationError('Could not map non isomorphic molecules')


def create_a_single_bond_mol_copy(mol):
    """
    Create a copy of the molecule with all bonds set to single bonds.
    """
    if not hasattr(mol, 'atoms'):
        return None
    new_mol = Molecule()
    atom_map = {old: new_mol.add_atom(Atom(old.element)) for old in mol.atoms}
    seen = set()
    for old in mol.atoms:
        for nbr, bond in old.bonds.items():
            key = frozenset({old, nbr})
            if key in seen:
                continue
            seen.add(key)
            new_mol.add_bond(Bond(atom_map[old], atom_map[nbr], 1.0))
    try:
        new_mol.update_atomtypes(raise_exception=False)
    except KeyError:
        pass
    new_mol.multiplicity = getattr(mol, 'multiplicity', new_mol.multiplicity)
    return new_mol


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
