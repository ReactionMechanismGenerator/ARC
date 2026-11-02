"""
Enumeration and equivalence-clustering of atom maps.

The standard mapping entry point, ``arc.mapping.driver.map_reaction``, returns a single atom map per
reaction. This module enumerates *many* valid atom maps and groups them into equivalence classes, where
each class corresponds to one chemically distinct reaction channel (one transition state to search for)
and the size of a class is that channel's reaction path degeneracy.

Theory
------
An atom map is a bijection ``sigma: atoms(R) -> atoms(P)`` from the reactant complex to the product
complex, stored as a list where ``atom_map[reactant_index] == product_index``.

If ``sigma`` is valid then so is ``beta . sigma . alpha`` for any ``alpha`` in ``Aut(R)`` and ``beta`` in
``Aut(P)``, since composing with an automorphism only relabels symmetry-equivalent atoms. The equivalence
classes are therefore the double cosets ``Aut(P) . sigma . Aut(R)``.

That definition can be reduced to something much cheaper. Let ``C(sigma)`` be the *changed-bond set*: the
bonds of ``R`` that are broken, formed, or change order under ``sigma``. Write ``R'`` for ``R`` with those
changes applied. Any valid ``sigma`` is a graph isomorphism ``R' -> P``, so:

- If ``C(sigma_1) == C(sigma_2)`` then ``sigma_2 . sigma_1^-1`` maps ``P`` to ``P`` preserving all bonds,
  i.e. it lies in ``Aut(P)``, hence ``sigma_1 ~ sigma_2``.
- If ``C(sigma_2) == alpha(C(sigma_1))`` for some ``alpha`` in ``Aut(R)``, then ``sigma_1 . alpha^-1`` has
  changed-bond set ``C(sigma_2)``, so by the previous point ``sigma_2 = beta . sigma_1 . alpha^-1``.

Therefore::

    sigma_1 ~ sigma_2   <=>   C(sigma_1) and C(sigma_2) are in the same Aut(R) orbit

Only ``Aut(R)`` is ever needed; ``Aut(P)`` does not appear. Clustering reduces to canonicalizing an edge
subset of ``R`` under a single group action.

Hydrogens
---------
Automorphism groups are computed on the *core skeleton* rather than the full molecular graph, because the
full group is dominated by intra-XHn hydrogen permutations that provably cannot change a cluster (the
hydrogens on a methyl group are one orbit of ``Aut(R)`` by construction). For isobutane the full group has
order 1296 while the core skeleton group has order 6.

A "core" atom is any non-hydrogen atom, plus any hydrogen that has no unique non-hydrogen neighbor (a free
H radical, H2, ...). Remaining hydrogens are represented in the changed-bond set by their parent core atom,
which is well defined precisely because hydrogens on a common parent are interchangeable. This keeps
hydrogen-transfer reactions - where the migrating atom *is* a hydrogen - fully representable.

Reaction path degeneracy is not counted from the enumeration, which only ever sees the maps RMG's templates
happened to produce. It is the orbit size of the reaction center under the *full* automorphism group - the
core group extended by permutations of the hydrogens on each core atom - see :func:`center_degeneracy`.
"""

import itertools
import operator
from dataclasses import dataclass, field

from arc.common import logger
from arc.mapping.driver import MAX_PDI, map_rxn_all, prepare_flipped_reaction
from arc.mapping.engine import flip_map
from arc.species import ARCSpecies

# An upper bound on the number of automorphisms enumerated for one complex. Hitting this bound means the
# clustering is computed under a truncated group and may split classes that are genuinely equivalent, so it
# is reported rather than applied silently.
MAX_AUTOMORPHISMS = 100_000

# An upper bound on the number of maps enumerated for one reaction.
MAX_ENUMERATED_MAPS = 5_000

# Endpoint descriptor tags used in a changed-bond set.
CORE = 'A'
HYDROGEN = 'H'


@dataclass
class ComplexGraph:
    """
    A flat graph view of a reactant or product complex, indexed by *running* atom indices across the whole
    complex - the same indexing convention used by ``atom_map``.

    Attributes:
        n_atoms (int): Total number of atoms in the complex.
        symbols (list[str]): Element symbol per atom.
        bonds (dict[frozenset[int], float]): Bond orders keyed by the unordered pair of atom indices.
        core (list[int]): Indices of core (skeleton) atoms.
        parent (dict[int, int]): Maps a non-core hydrogen index to its parent core atom index.
        adj (dict[int, dict[int, float]]): Core-core adjacency with bond orders.
        invariant (dict[int, tuple]): Per-core-atom initial colour used to seed the automorphism search.
    """
    n_atoms: int
    symbols: list[str]
    bonds: dict[frozenset, float]
    core: list[int]
    parent: dict[int, int]
    adj: dict[int, dict[int, float]]
    invariant: dict[int, tuple]


@dataclass
class MapCluster:
    """
    One equivalence class of atom maps, i.e. one distinct reaction channel.

    Attributes:
        representative (list[int]): A representative atom map for this channel.
        members (list[list[int]]): All enumerated atom maps belonging to this channel.
        centers (set): The distinct changed-bond sets seen in this channel, one per distinct reaction path.
        key (tuple): The canonical ``Aut(R)``-invariant form of the changed-bond set.
        signature (tuple): A human-readable orbit-level signature of the reaction center.
        truncated (bool): Whether the automorphism group used to build ``key`` was truncated.
        recipe_validated (bool): Whether this channel's reaction center is one the family recipe actually
                                 predicts. ``False`` means no recipe reference was available, or that it
                                 rejected every enumerated map and the relative minimal-center heuristic
                                 was used instead; the channel is then only as good as the enumeration
                                 behind it. ``False`` alongside an implausibly large center - 13 changed
                                 bonds for a recombination that forms one - marks a scrambled enumeration
                                 rather than a real reaction channel.
    """
    representative: list[int]
    members: list[list[int]] = field(default_factory=list)
    centers: set = field(default_factory=set)
    degeneracy: int = 0
    key: tuple = ()
    signature: tuple = ()
    truncated: bool = False
    recipe_validated: bool = True

    @property
    def enumerated_degeneracy(self) -> int:
        """
        int: How many distinct reaction centers the enumeration actually found.

        This counts distinct changed-bond sets rather than distinct atom maps, since two maps sharing a
        changed-bond set differ only by an element of ``Aut(P)`` and are the same reaction path. It is a
        lower bound on :attr:`degeneracy`, and can be a poor one: RMG generates only two template matches
        for ``C2H5Cl -> C2H4 + HCl``, so the enumeration sees at most two of the three equivalent methyl
        hydrogens even though the true degeneracy is 3. Compare the two to gauge enumeration coverage.
        """
        return len(self.centers)

    def __repr__(self) -> str:
        return f'<MapCluster degeneracy={self.degeneracy} enumerated={self.enumerated_degeneracy} ' \
               f'maps={len(self.members)} signature={self.signature}>'


def build_complex_graph(species_list: list[ARCSpecies]) -> ComplexGraph:
    """
    Build a :class:`ComplexGraph` from a list of species, concatenating their atoms in order so that atom
    indices match the running-index convention of ``atom_map``.

    Args:
        species_list (list[ARCSpecies]): The species forming the complex.

    Returns:
        ComplexGraph: The flat graph view of the complex.
    """
    symbols: list[str] = list()
    bonds: dict[frozenset, float] = dict()
    heavy_neighbors: dict[int, list[int]] = dict()
    attributes: dict[int, tuple] = dict()
    offset = 0
    for spc in species_list:
        atoms = spc.mol.atoms
        local_index = {id(atom): i for i, atom in enumerate(atoms)}
        for i, atom in enumerate(atoms):
            symbols.append(atom.element.symbol)
            attributes[offset + i] = (atom.element.symbol,
                                      getattr(atom, 'charge', 0),
                                      getattr(atom, 'radical_electrons', 0),
                                      getattr(atom, 'lone_pairs', 0))
            heavy_neighbors[offset + i] = list()
        for bond in spc.mol.get_all_edges():
            i = offset + local_index[id(bond.atom1)]
            j = offset + local_index[id(bond.atom2)]
            bonds[frozenset((i, j))] = bond.order
            if bond.atom2.element.symbol != 'H':
                heavy_neighbors[i].append(j)
            if bond.atom1.element.symbol != 'H':
                heavy_neighbors[j].append(i)
        offset += len(atoms)

    # A hydrogen is represented by its parent only when it has exactly one non-hydrogen neighbor.
    # Everything else (a free H radical, H2, a bridging H) is promoted to a core atom in its own right.
    parent: dict[int, int] = dict()
    core: list[int] = list()
    for i, symbol in enumerate(symbols):
        if symbol == 'H' and len(heavy_neighbors[i]) == 1:
            parent[i] = heavy_neighbors[i][0]
        else:
            core.append(i)

    core_set = set(core)
    n_hydrogens = {v: 0 for v in core}
    for h, p in parent.items():
        n_hydrogens[p] = n_hydrogens.get(p, 0) + 1

    adj: dict[int, dict[int, float]] = {v: dict() for v in core}
    for pair, order in bonds.items():
        i, j = tuple(pair)
        if i in core_set and j in core_set:
            adj[i][j] = order
            adj[j][i] = order

    invariant = {v: attributes[v] + (n_hydrogens[v],) for v in core}
    return ComplexGraph(n_atoms=len(symbols), symbols=symbols, bonds=bonds,
                        core=core, parent=parent, adj=adj, invariant=invariant)


def effective_adjacency(graph: ComplexGraph, ignore_bond_orders: bool = True) -> dict[int, dict[int, float]]:
    """
    Return the core-core adjacency used for symmetry detection, optionally with all bond orders normalized
    to 1.

    Normalizing matters because RMG stores an aromatic ring as a Kekule structure with alternating single
    and double bonds. Only 6 of benzene's 12 graph automorphisms preserve that alternation, so an
    order-sensitive group under-counts the symmetry and splits clusters that are genuinely equivalent.
    This must agree with the ``ignore_bond_orders`` setting used by :func:`changed_bonds`.

    Note that this normalizes bond orders only. The matching treatment of the resonance-mobile per-atom
    attributes - radical electrons, lone pairs, formal charge - lives in :func:`effective_invariant`, and
    the two must be used with the same ``ignore_bond_orders`` setting.

    Args:
        graph (ComplexGraph): The complex graph.
        ignore_bond_orders (bool, optional): Whether to normalize all bond orders to 1.

    Returns:
        dict[int, dict[int, float]]: The adjacency to use.
    """
    if not ignore_bond_orders:
        return graph.adj
    return {v: {u: 1 for u in neighbors} for v, neighbors in graph.adj.items()}


def effective_invariant(graph: ComplexGraph, ignore_bond_orders: bool = True) -> dict[int, tuple]:
    """
    The per-core-atom colour seeding the symmetry search.

    ``ignore_bond_orders`` exists because RMG stores one Kekule structure of what is physically a
    delocalized system. Resonance moves more than bond orders, though - it also moves radical electrons,
    lone pairs and formal charges between atoms that are physically equivalent. Seeding the colouring with
    those makes a delocalized species look less symmetric than it is: allyl's two terminal carbons are
    distinguished only by which one the stored Lewis structure happens to carry the radical on, which
    halves every degeneracy computed through it. They are therefore dropped alongside the bond orders,
    leaving the element and the attached-hydrogen count, both of which resonance preserves.

    The trade-off is that two fragments differing *only* in charge or radical count - and not in element,
    connectivity or hydrogen count - are then seen as interchangeable. That is the correct reading for
    resonance forms of one species, and the price paid for it elsewhere.

    Args:
        graph (ComplexGraph): The complex graph.
        ignore_bond_orders (bool, optional): Whether resonance-mobile attributes are dropped.

    Returns:
        dict[int, tuple]: The colour seed per core atom.
    """
    if not ignore_bond_orders:
        return graph.invariant
    return {v: (graph.invariant[v][0], graph.invariant[v][-1]) for v in graph.core}


def refine_colors(graph: ComplexGraph, ignore_bond_orders: bool = True) -> dict[int, int]:
    """
    Compute a stable colour refinement (1-dimensional Weisfeiler-Leman) of the core skeleton, used to prune
    the automorphism search. Two core atoms may only be mapped onto each other if they share a colour.

    Args:
        graph (ComplexGraph): The complex graph.
        ignore_bond_orders (bool, optional): Whether to ignore bond orders, see :func:`effective_adjacency`.

    Returns:
        dict[int, int]: The stable colour per core atom.
    """
    adj = effective_adjacency(graph, ignore_bond_orders=ignore_bond_orders)
    labels = effective_invariant(graph, ignore_bond_orders=ignore_bond_orders)
    colors = _compress(labels)
    while True:
        signatures = {v: (colors[v], tuple(sorted((colors[u], order) for u, order in adj[v].items())))
                      for v in graph.core}
        new_colors = _compress(signatures)
        if len(set(new_colors.values())) == len(set(colors.values())):
            return new_colors
        colors = new_colors


def _compress(labels: dict[int, tuple]) -> dict[int, int]:
    """
    Relabel arbitrary hashable colours to consecutive integers, deterministically.

    Args:
        labels (dict[int, tuple]): Per-node colour.

    Returns:
        dict[int, int]: Per-node integer colour.
    """
    ranking = {label: i for i, label in enumerate(sorted(set(labels.values()), key=repr))}
    return {v: ranking[label] for v, label in labels.items()}


def core_automorphisms(graph: ComplexGraph,
                       max_count: int = MAX_AUTOMORPHISMS,
                       ignore_bond_orders: bool = True,
                       ) -> tuple[list[dict[int, int]], bool]:
    """
    Enumerate the automorphism group of the core skeleton by colour-pruned backtracking.

    Note that the complex is generally disconnected (it holds several molecules), so the group includes
    permutations that exchange identical molecules. That is intended: swapping two identical reactants is a
    genuine symmetry of the complex.

    Args:
        graph (ComplexGraph): The complex graph.
        max_count (int, optional): Stop after this many automorphisms.
        ignore_bond_orders (bool, optional): Whether to ignore bond orders, see :func:`effective_adjacency`.
                                             Must agree with the setting used by :func:`changed_bonds`.

    Returns:
        tuple[list[dict[int, int]], bool]:
            The automorphisms as index->index dicts over core atoms, and whether the search was truncated.
    """
    adj = effective_adjacency(graph, ignore_bond_orders=ignore_bond_orders)
    colors = refine_colors(graph, ignore_bond_orders=ignore_bond_orders)
    buckets: dict[int, list[int]] = dict()
    for v, color in colors.items():
        buckets.setdefault(color, list()).append(v)
    # Place the most constrained atoms first so that contradictions surface early.
    order = sorted(graph.core, key=lambda v: (len(buckets[colors[v]]), v))

    automorphisms: list[dict[int, int]] = list()
    mapping: dict[int, int] = dict()
    used: set[int] = set()
    truncated = False

    def backtrack(idx: int) -> None:
        nonlocal truncated
        if truncated:
            return
        if idx == len(order):
            automorphisms.append(dict(mapping))
            if len(automorphisms) >= max_count:
                truncated = True
            return
        v = order[idx]
        for w in buckets[colors[v]]:
            if w in used:
                continue
            # Verify both the presence and the absence of every edge to an already-placed atom.
            if any(adj[v].get(u) != adj[w].get(image) for u, image in mapping.items()):
                continue
            mapping[v] = w
            used.add(w)
            backtrack(idx + 1)
            del mapping[v]
            used.discard(w)
            if truncated:
                return

    backtrack(0)
    if truncated:
        logger.warning(f'Automorphism enumeration hit the cap of {max_count}; atom map clustering may '
                       f'report more channels than actually exist.')
    return automorphisms, truncated


def core_orbits(graph: ComplexGraph, automorphisms: list[dict[int, int]]) -> dict[int, int]:
    """
    Compute the orbits of the core atoms under the automorphism group, as a canonical orbit id per atom.

    Args:
        graph (ComplexGraph): The complex graph.
        automorphisms (list[dict[int, int]]): The automorphism group.

    Returns:
        dict[int, int]: Orbit id per core atom, the id being the smallest atom index in that orbit.
    """
    representative = {v: v for v in graph.core}

    def find(v: int) -> int:
        while representative[v] != v:
            representative[v] = representative[representative[v]]
            v = representative[v]
        return v

    for alpha in automorphisms:
        for v, w in alpha.items():
            root_v, root_w = find(v), find(w)
            if root_v != root_w:
                if root_w < root_v:
                    root_v, root_w = root_w, root_v
                representative[root_w] = root_v
    return {v: find(v) for v in graph.core}


def changed_bonds(r_graph: ComplexGraph,
                  p_graph: ComplexGraph,
                  atom_map: list[int],
                  ignore_bond_orders: bool = True,
                  ) -> frozenset:
    """
    Compute the changed-bond set ``C(sigma)`` of an atom map, expressed in *reactant-complex* indices.

    Each entry is ``(i, j, order_before, order_after)`` with ``i < j`` reactant atom indices. A broken bond
    has ``order_after == 0``, a formed bond has ``order_before == 0``.

    Because the set is expressed entirely in reactant indices, it is invariant under any relabeling of the
    product atoms, i.e. under ``Aut(P)``. Two maps sharing a changed-bond set therefore describe the same
    reaction path, which is what makes this the right thing to count for reaction path degeneracy. For the
    ``Aut(R)``-canonical cluster key the set must first be passed through :func:`collapse_hydrogens`.

    Args:
        r_graph (ComplexGraph): The reactant complex.
        p_graph (ComplexGraph): The product complex.
        atom_map (list[int]): The atom map, ``atom_map[reactant_index] == product_index``.
        ignore_bond_orders (bool, optional): If ``True``, only bond breaking and formation are recorded and
                                             pure bond-order changes are ignored. This is the default
                                             because the mapping pipeline mutates bond orders when handling
                                             resonance (see ``make_bond_changes``), which would otherwise
                                             inject spurious entries.

    Returns:
        frozenset: The changed-bond set.
    """
    changes = set()
    inverse_map = {product_index: reactant_index for reactant_index, product_index in enumerate(atom_map)}
    pairs = set(r_graph.bonds.keys())
    for pair in p_graph.bonds.keys():
        i, j = tuple(pair)
        pairs.add(frozenset((inverse_map[i], inverse_map[j])))
    for pair in pairs:
        i, j = tuple(pair)
        order_before = r_graph.bonds.get(frozenset((i, j)), 0)
        order_after = p_graph.bonds.get(frozenset((atom_map[i], atom_map[j])), 0)
        if ignore_bond_orders:
            order_before = 1 if order_before else 0
            order_after = 1 if order_after else 0
        if order_before == order_after:
            continue
        changes.add((min(i, j), max(i, j), order_before, order_after))
    return frozenset(changes)


def collapse_hydrogens(center: frozenset, graph: ComplexGraph) -> frozenset:
    """
    Rewrite a changed-bond set in terms of core atoms, replacing each non-core hydrogen by its parent core
    atom. This is what makes the set act-able by the core-skeleton automorphism group, which is only defined
    on core atoms.

    Each entry becomes ``(endpoint_a, endpoint_b, order_before, order_after)`` with the two endpoint
    descriptors sorted, where a descriptor is ``('A', core_index)`` for a core atom and
    ``('H', parent_core_index)`` for a collapsed hydrogen.

    The collapse is lossy by design: abstracting any of methane's four hydrogens yields the same collapsed
    set, which is precisely why all four land in one cluster. Reaction path degeneracy must therefore be
    counted on the uncollapsed sets, see :attr:`MapCluster.degeneracy`.

    Args:
        center (frozenset): The changed-bond set in reactant indices.
        graph (ComplexGraph): The reactant complex graph.

    Returns:
        frozenset: The collapsed changed-bond set.
    """
    collapsed = set()
    for i, j, order_before, order_after in center:
        endpoints = tuple(sorted((_endpoint(graph, i), _endpoint(graph, j))))
        collapsed.add((endpoints[0], endpoints[1], order_before, order_after))
    return frozenset(collapsed)


def _endpoint(graph: ComplexGraph, index: int) -> tuple:
    """
    Describe an atom as a changed-bond endpoint, collapsing a hydrogen onto its parent core atom.

    Args:
        graph (ComplexGraph): The complex graph.
        index (int): The atom index.

    Returns:
        tuple: The endpoint descriptor.
    """
    if index in graph.parent:
        return HYDROGEN, graph.parent[index]
    return CORE, index


def hydrogen_stabilizer(center: frozenset, hydrogens: list[int]) -> int:
    """
    How many permutations of ``hydrogens`` leave the changed-bond set unchanged.

    The changed-bond set is a *set*, so two hydrogens on the same parent that play the same role in it are
    interchangeable: swapping them reproduces the same center rather than describing a second reaction path.
    :func:`center_degeneracy` divides the ordered count by this to recover the orbit of the set.

    Args:
        center (frozenset): The changed-bond set, not hydrogen-collapsed.
        hydrogens (list[int]): The named hydrogens of one parent core atom.

    Returns:
        int: The size of the stabilizer, at least 1.
    """
    hydrogens = sorted(hydrogens)
    stabilizer = 0
    for permutation in itertools.permutations(hydrogens):
        relabel = dict(zip(hydrogens, permutation))
        image = frozenset(tuple(sorted((relabel.get(i, i), relabel.get(j, j)))) + (before, after)
                          for i, j, before, after in center)
        if image == center:
            stabilizer += 1
    return stabilizer or 1


def center_degeneracy(center: frozenset,
                      graph: ComplexGraph,
                      automorphisms: list[dict[int, int]],
                      ) -> int:
    """
    The exact reaction path degeneracy of a reaction center: the size of its orbit under the *full*
    automorphism group of the reactant complex.

    Counting the distinct centers actually enumerated is a lower bound, and often a poor one - RMG generates
    only two template matches for ``C2H5Cl -> C2H4 + HCl``, so enumeration sees at most two of the three
    equivalent methyl hydrogens. The orbit size is exact and does not depend on the enumeration at all.

    The full automorphism group is the core-skeleton group extended by arbitrary permutations of the
    hydrogens attached to each core atom, and the orbit factorises accordingly::

        degeneracy = (number of distinct images of the collapsed center under the core group)
                     * product over core atoms p of  n_p! / (n_p - k_p)!

    where ``n_p`` is how many hydrogens hang off core atom ``p`` and ``k_p`` how many distinct ones the
    center actually names. The falling factorial counts the *ordered* choices of which hydrogens play the
    named roles, so it is divided by :func:`hydrogen_stabilizer` to recover the orbit of the center as a
    set - see there for why.

    The result is the orbit size and is deliberately *not* halved when the two reactants are the same
    species. RMG halves it in that case (``reduce_same_reactant_degeneracy``, after Bishop and Laidler
    1965) so a rate expression does not double count indistinguishable colliding partners:
    ``C2H5 + C2H5 <=> C4H10`` is 1 here and 0.5 in RMG. Which convention is correct depends on how the
    number is consumed, and a caller folding it into a rate coefficient for identical reactants has to
    apply that factor itself. For CH4 + OH the core group is trivial and one of methane's four hydrogens is named, giving
    4; for C2H6 + OH the core group swaps the two carbons and one of three hydrogens is named, giving 6.

    Args:
        center (frozenset): The changed-bond set, in reactant indices and *not* hydrogen-collapsed.
        graph (ComplexGraph): The reactant complex.
        automorphisms (list[dict[int, int]]): The core-skeleton automorphism group.

    Returns:
        int: The reaction path degeneracy.
    """
    if not center:
        # The orbit of the empty set is itself, so the answer is one path, never zero. An empty center
        # means this map changes no bonds *under the current bond-order setting* - families such as
        # 1,2-Birad_to_alkene only change orders, so ignore_bond_orders=True erases their whole center.
        # cluster_atom_maps() detects that case and retries with orders honored.
        return 1
    collapsed = collapse_hydrogens(center, graph)
    images = set()
    for alpha in automorphisms:
        image = list()
        for endpoint_a, endpoint_b, order_before, order_after in collapsed:
            mapped_a = (endpoint_a[0], alpha.get(endpoint_a[1], endpoint_a[1]))
            mapped_b = (endpoint_b[0], alpha.get(endpoint_b[1], endpoint_b[1]))
            if mapped_b < mapped_a:
                mapped_a, mapped_b = mapped_b, mapped_a
            image.append((mapped_a, mapped_b, order_before, order_after))
        images.add(tuple(sorted(image)))
    core_orbit = len(images) or 1

    hydrogen_counts = {v: 0 for v in graph.core}
    for hydrogen, parent in graph.parent.items():
        hydrogen_counts[parent] = hydrogen_counts.get(parent, 0) + 1
    named: dict[int, set] = dict()
    for i, j, _, _ in center:
        for index in (i, j):
            if index in graph.parent:
                named.setdefault(graph.parent[index], set()).add(index)

    hydrogen_factor = 1
    for parent, hydrogens in named.items():
        # named[parent] is a subset of the hydrogens on that parent, so available >= len(hydrogens) and
        # each factor below is >= 1; no clamping is needed.
        available = hydrogen_counts[parent]
        for offset in range(len(hydrogens)):
            hydrogen_factor *= available - offset
        # The falling factorial counts *ordered* assignments of hydrogens to the named roles, but the
        # center is a set. Interchangeable hydrogens on one parent therefore have to be divided out, or a
        # single path is counted once per ordering: both C-H bonds of CH2O breaking is one path to H2, not
        # two. Dividing by the stabilizer turns the ordered count into the orbit of the set.
        hydrogen_factor //= hydrogen_stabilizer(center, list(hydrogens))
    return core_orbit * hydrogen_factor


def canonical_center_key(center: frozenset, automorphisms: list[dict[int, int]]) -> tuple:
    """
    Canonicalize a changed-bond set under the reactant automorphism group: the key is the lexicographic
    minimum over the group orbit, so two maps share a key exactly when they are equivalent.

    Args:
        center (frozenset): The changed-bond set.
        automorphisms (list[dict[int, int]]): The reactant automorphism group.

    Returns:
        tuple: The canonical, hashable form of the changed-bond set.
    """
    if not automorphisms:
        return tuple(sorted(center))
    best = None
    for alpha in automorphisms:
        image = list()
        for endpoint_a, endpoint_b, order_before, order_after in center:
            mapped_a = (endpoint_a[0], alpha.get(endpoint_a[1], endpoint_a[1]))
            mapped_b = (endpoint_b[0], alpha.get(endpoint_b[1], endpoint_b[1]))
            if mapped_b < mapped_a:
                mapped_a, mapped_b = mapped_b, mapped_a
            image.append((mapped_a, mapped_b, order_before, order_after))
        candidate = tuple(sorted(image))
        if best is None or candidate < best:
            best = candidate
    return best


def center_signature(center: frozenset, orbits: dict[int, int], symbols: list[str]) -> tuple:
    """
    Build a readable orbit-level signature of a reaction center. Equivalent maps always share a signature,
    but distinct maps may also collide, so this is a cheap bucketing invariant rather than a decision
    procedure - :func:`canonical_center_key` is the exact test.

    Args:
        center (frozenset): The changed-bond set.
        orbits (dict[int, int]): Orbit id per reactant core atom.
        symbols (list[str]): Element symbol per reactant atom.

    Returns:
        tuple: The signature.
    """
    entries = list()
    for endpoint_a, endpoint_b, order_before, order_after in center:
        described = tuple(sorted(f'{"H@" if tag == HYDROGEN else ""}{symbols[index]}{orbits.get(index, index)}'
                                 for tag, index in (endpoint_a, endpoint_b)))
        entries.append((described[0], described[1], order_before, order_after))
    return tuple(sorted(entries))


def enumerate_atom_maps(rxn,
                        backend: str = 'ARC',
                        include_flipped: bool = True,
                        max_maps: int = MAX_ENUMERATED_MAPS,
                        ) -> list[list[int]]:
    """
    Enumerate valid atom maps for a reaction by sweeping every RMG template ``product_dict`` and, optionally,
    both reaction directions. Duplicate maps are removed while preserving discovery order.

    Multiplicity enters from two independent sources, both of which the single-map path discards:
    the template ``product_dict`` sweep, and - within one product dictionary - the several superimposable
    backbone maps per scissored fragment pair, combined by ``map_rxn_all``.

    Args:
        rxn (ARCReaction): The reaction to map.
        backend (str, optional): Currently only ``ARC``'s method is implemented as the backend.
        include_flipped (bool, optional): Whether to also map the flipped reaction and un-flip the results.
        max_maps (int, optional): Stop after this many distinct maps.

    Returns:
        list[list[int]]: The distinct atom maps found.
    """
    maps: list[list[int]] = list()
    seen: set[tuple] = set()

    def collect(candidates: list[list[int]], flip: bool = False) -> None:
        for atom_map in candidates:
            if len(maps) >= max_maps:
                return
            if flip:
                atom_map = flip_map(atom_map)
            if atom_map is None:
                continue
            key = tuple(atom_map)
            if key not in seen:
                seen.add(key)
                maps.append(list(atom_map))

    def sweep(target, flip: bool) -> None:
        n_product_dicts = len(target.product_dicts) if getattr(target, 'product_dicts', None) else 1
        for pdi in range(min(n_product_dicts, MAX_PDI)):
            try:
                collect(map_rxn_all(target, backend=backend, product_dict_index_to_try=pdi), flip=flip)
            except (ValueError, IndexError, KeyError) as e:
                logger.debug(f'enumerate_atom_maps: product_dict {pdi} of {target} '
                             f'(flip={flip}) failed with {e!r}.')
            if len(maps) >= max_maps:
                return

    sweep(rxn, flip=False)
    if include_flipped and len(maps) < max_maps:
        # Use prepare_flipped_reaction rather than flip_reaction: the latter resets the family, so the
        # flipped copy re-derives its product dictionaries with the default family set and comes back
        # empty. That is the whole enumeration for a reaction whose template was only discovered in
        # reverse, since every forward product dictionary then fails at get_template_product_order.
        sweep(prepare_flipped_reaction(rxn), flip=True)

    if len(maps) >= max_maps:
        logger.warning(f'enumerate_atom_maps hit the cap of {max_maps} maps for {rxn}; '
                       f'reported degeneracies are lower bounds.')
    return maps


def is_permutation_map(atom_map, n_reactant_atoms: int, n_product_atoms: int) -> bool:
    """
    Whether ``atom_map`` is a genuine bijection of reactant indices onto product indices.

    :func:`changed_bonds` inverts the map, so anything short of a bijection - a duplicated or missing
    product index, an out-of-range entry, a non-integer entry - raises a ``KeyError`` there and aborts the
    entire clustering run rather than just discarding the offending map. Callers screen maps with this
    first, in the same spirit as ``driver.check_atom_map_and_return``.

    Args:
        atom_map: The candidate atom map, ``atom_map[reactant_index] == product_index``.
        n_reactant_atoms (int): Number of atoms in the reactant complex.
        n_product_atoms (int): Number of atoms in the product complex.

    Returns:
        bool: ``True`` iff the map is a bijection of ``range(n_reactant_atoms)`` onto
              ``range(n_product_atoms)``.
    """
    if not isinstance(atom_map, (list, tuple)) or len(atom_map) != n_reactant_atoms:
        return False
    try:
        # operator.index() accepts numpy integers, which the mapping pipeline can produce, while still
        # rejecting floats, None and strings.
        indices = [operator.index(value) for value in atom_map]
    except TypeError:
        return False
    return len(set(indices)) == n_product_atoms and all(0 <= index < n_product_atoms for index in indices)


def cluster_atom_maps(atom_maps: list[list[int]],
                      rxn,
                      ignore_bond_orders: bool = True,
                      validate_centers: bool = True,
                      ) -> list[MapCluster]:
    """
    Group atom maps into equivalence classes, one per distinct reaction channel.

    Two maps land in the same cluster exactly when their changed-bond sets lie in the same orbit of the
    reactant complex automorphism group - see the module docstring for why this is equivalent to the double
    coset ``Aut(P) . sigma . Aut(R)``.

    Args:
        atom_maps (list[list[int]]): The atom maps to cluster.
        rxn (ARCReaction): The reaction the maps belong to.
        ignore_bond_orders (bool, optional): Passed through to :func:`changed_bonds`.
        validate_centers (bool, optional): Whether to discard maps whose reaction center is invalid. The
                                           family recipe is used as an absolute reference where available
                                           (:func:`expected_reaction_centers`), otherwise the relative
                                           minimal-center heuristic (:func:`filter_minimal_centers`).

    Returns:
        list[MapCluster]: The clusters, ordered by descending degeneracy.
    """
    if not atom_maps:
        return list()
    reactants, products = rxn.get_reactants_and_products(return_copies=True)
    r_graph, p_graph = build_complex_graph(reactants), build_complex_graph(products)
    # The automorphism group and the changed-bond set must agree on bond orders, otherwise a Kekule
    # structure makes the group too small and splits clusters that are genuinely equivalent.
    automorphisms, truncated = core_automorphisms(r_graph, ignore_bond_orders=ignore_bond_orders)
    orbits = core_orbits(r_graph, automorphisms)

    scored: list[tuple[list[int], frozenset]] = list()
    for atom_map in atom_maps:
        if not is_permutation_map(atom_map, r_graph.n_atoms, p_graph.n_atoms):
            logger.warning(f'Skipping an atom map for {rxn} that is not a bijection of the '
                           f'{r_graph.n_atoms} reactant atoms onto the {p_graph.n_atoms} product '
                           f'atoms: {atom_map}.')
            continue
        scored.append((atom_map, changed_bonds(r_graph, p_graph, atom_map,
                                               ignore_bond_orders=ignore_bond_orders)))
    if ignore_bond_orders and scored and not any(center for _, center in scored):
        # Every center is empty, which means no bond was made or broken and the reaction is a pure
        # bond-order change: 1,2-Birad_to_alkene, Singlet_Val6_to_triplet. Ignoring orders erases the whole
        # reaction center for those, leaving nothing to cluster on, nothing for either validity filter to
        # judge, and a degenerate answer. Retry honoring orders, which is the only setting that can
        # describe them. Note the recipe-based reference is unavailable there (it predicts only breaking
        # and formation), so validation falls back to the relative minimal-center filter.
        logger.debug(f'Every enumerated atom map for {rxn} changes only bond orders; re-clustering with '
                     f'bond orders honored.')
        return cluster_atom_maps(atom_maps, rxn, ignore_bond_orders=False,
                                 validate_centers=validate_centers)

    recipe_validated = False
    if validate_centers and scored:
        expected = expected_reaction_centers(rxn, ignore_bond_orders=ignore_bond_orders)
        validated = filter_expected_centers(scored, expected, rxn) if expected else list()
        if validated:
            scored = validated
            recipe_validated = True
        else:
            # Either no recipe reference was available, or it rejected every enumerated map - which would
            # mean returning no channels at all. Fall back to the relative filter rather than that.
            if expected:
                logger.warning(f'The {rxn.family} recipe rejected all {len(scored)} enumerated atom maps '
                               f'for {rxn}; falling back to the minimal-center filter.')
            scored = filter_minimal_centers(scored, rxn)

    clusters: dict[tuple, MapCluster] = dict()
    for atom_map, center in scored:
        if not center:
            logger.debug(f'An atom map for {rxn} induces no bond changes; clustering it under an empty key.')
        collapsed = collapse_hydrogens(center, r_graph)
        key = canonical_center_key(collapsed, automorphisms)
        if key not in clusters:
            clusters[key] = MapCluster(representative=list(atom_map),
                                       key=key,
                                       degeneracy=center_degeneracy(center, r_graph, automorphisms),
                                       signature=center_signature(collapsed, orbits, r_graph.symbols),
                                       truncated=truncated,
                                       recipe_validated=recipe_validated)
        clusters[key].members.append(list(atom_map))
        clusters[key].centers.add(center)
    return sorted(clusters.values(), key=lambda cluster: (-cluster.degeneracy, cluster.key))


def resolve_duplicate_label(index: int, r_label_map: dict) -> int | None:
    """
    Recover the second site of a recipe action that names one template label twice.

    ``R_Recombination``'s recipe is ``FORM_BOND(*, *)``: one label, two distinct atoms.
    ``ARCReaction.get_expected_changing_bonds`` reads the base label only, so both endpoints resolve to the
    same index and the bond looks degenerate. The label map does carry the second site, under a suffixed
    key (``{'*': 0, '*_2': 4}``), so it can be recovered instead of discarding the reference entirely -
    which previously left every ``R_Recombination`` reaction with no absolute validity reference at all.

    Args:
        index (int): The index both endpoints resolved to.
        r_label_map (dict): The product dictionary's label map.

    Returns:
        int | None: The sibling index, or ``None`` if the map carries no suffixed variant.
    """
    bases = {key.split('_')[0] for key, value in r_label_map.items() if value == index}
    for key, value in sorted(r_label_map.items()):
        if value != index and key.split('_')[0] in bases:
            return value
    return None


def expected_reaction_centers(rxn, ignore_bond_orders: bool = True) -> set[frozenset] | None:
    """
    The reaction centers predicted by the RMG family recipe, one per template product dictionary, expressed
    in reactant-complex indices.

    This is an *absolute* reference for map validity, unlike :func:`filter_minimal_centers` which can only
    compare enumerated maps against each other. Each product dictionary's ``r_label_map`` assigns the
    family's labelled atoms (``*1``, ``*2``, ...) to concrete reactant indices, and
    ``ARCReaction.get_expected_changing_bonds`` turns the family's ``BREAK_BOND`` and ``FORM_BOND`` actions
    into index pairs. For CH4 + OH the four product dictionaries yield exactly the four abstraction centers.

    Note the indices in ``r_label_map`` are 0-indexed, matching ``atom_map``, despite the docstring of
    ``find_all_breaking_bonds`` describing its own return value as 1-indexed.

    A product dictionary is skipped, rather than the whole reference being abandoned, when its recipe cannot
    be read. Two causes are common:

    - **A missing label.** ``get_expected_changing_bonds`` reads a single family-level ``actions`` list, but
      a family such as ``intra_H_migration`` spans several template variants with different label sets, so
      some product dictionaries lack a label the actions reference and raise ``KeyError``. Only 6 of the 32
      product dictionaries of one benchmark reaction are readable this way.
    - **A degenerate self-bond.** An action naming the same label twice (``R_Recombination``'s
      ``* + * -> *-*``) resolves both endpoints to one index. :func:`resolve_duplicate_label` recovers the
      second site from the suffixed label key the map carries (``*_2``); only when no such sibling exists
      is the dictionary skipped, rather than contributing a wrong center.

    Returns ``None``, meaning "no usable reference at all", when the reaction has no family or no product
    dictionaries, when ``ignore_bond_orders`` is ``False`` (the recipe describes only bond breaking and
    formation and cannot predict the pure order changes that :func:`changed_bonds` would then report), or
    when every product dictionary was skipped.

    Args:
        rxn (ARCReaction): The reaction.
        ignore_bond_orders (bool, optional): Must match the setting used by :func:`changed_bonds`.

    Returns:
        set[frozenset] | None: The predicted centers, or ``None`` if no reliable reference is available.
    """
    if not ignore_bond_orders or rxn.family is None or not getattr(rxn, 'product_dicts', None):
        return None
    centers, skipped = set(), 0
    for product_dict in rxn.product_dicts[:MAX_PDI]:
        r_label_map = product_dict.get('r_label_map')
        if not r_label_map or product_dict.get('discovered_in_reverse'):
            # A reverse-discovered dictionary's label maps describe the flipped reaction, so the indices
            # they carry are in the flipped frame and cannot be compared against a changed-bond set built
            # in this reaction's own frame. Using them anyway predicts centers that match nothing.
            skipped += 1
            continue
        try:
            breaking, forming = rxn.get_expected_changing_bonds(r_label_dict=r_label_map)
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if breaking is None and forming is None:
            skipped += 1
            continue
        center = set()
        for pairs, orders in ((breaking or list(), (1, 0)), (forming or list(), (0, 1))):
            for i, j in pairs:
                if i == j:
                    j = resolve_duplicate_label(i, r_label_map)
                    if j is None or j == i:
                        center = None
                        break
                center.add((min(i, j), max(i, j)) + orders)
            if center is None:
                break
        if not center:
            skipped += 1
            continue
        centers.add(frozenset(center))
    if skipped:
        logger.debug(f'expected_reaction_centers: skipped {skipped} unreadable product dictionaries of '
                     f'{rxn} ({rxn.family}), kept {len(centers)} predicted centers.')
    return centers or None


def filter_expected_centers(scored: list[tuple[list[int], frozenset]],
                            expected: set[frozenset],
                            rxn=None,
                            ) -> list[tuple[list[int], frozenset]]:
    """
    Keep only the maps whose reaction center is one the family recipe actually predicts.

    Args:
        scored (list[tuple[list[int], frozenset]]): ``(atom_map, changed_bond_set)`` pairs.
        expected (set[frozenset]): The centers from :func:`expected_reaction_centers`.
        rxn (ARCReaction, optional): The reaction, used only for logging.

    Returns:
        list[tuple[list[int], frozenset]]: The pairs whose center matches the recipe.
    """
    kept = [entry for entry in scored if entry[1] in expected]
    if len(kept) != len(scored):
        logger.debug(f'Discarded {len(scored) - len(kept)} of {len(scored)} atom maps for {rxn} whose '
                     f'reaction center is not predicted by the {getattr(rxn, "family", None)} recipe.')
    return kept


def filter_minimal_centers(scored: list[tuple[list[int], frozenset]],
                           rxn=None,
                           ) -> list[tuple[list[int], frozenset]]:
    """
    Keep only the maps whose reaction center is as small as the smallest one observed.

    This is a validity filter, and it is needed because :func:`changed_bonds` computes a changed-bond set
    for *any* element-preserving bijection - there is nothing in the arithmetic that distinguishes a correct
    atom map from a scrambled one. ``map_rxn_all`` deliberately keeps every graph-superimposable backbone
    candidate per fragment, including those that ``map_two_species`` would have rejected on RMSD, so
    scrambled maps do reach this point. Left unfiltered they register as extra "channels": for the
    Diels-Alder of butadiene with ethene the correct map changes 2 bonds while the scrambled ones change 10
    to 14, and each scrambled variant would otherwise open its own cluster.

    An elementary reaction rearranges a minimal set of bonds, so the smallest observed center is the correct
    one and anything larger is a mapping error rather than a distinct channel.

    Note this is a heuristic over the enumerated set, not an independent check against the family recipe. It
    assumes at least one correct map was enumerated; if every enumerated map is wrong, the least wrong one
    survives. ``ARCReaction.get_expected_changing_bonds`` would give an absolute reference and is the natural
    next step.

    Args:
        scored (list[tuple[list[int], frozenset]]): ``(atom_map, changed_bond_set)`` pairs.
        rxn (ARCReaction, optional): The reaction, used only for logging.

    Returns:
        list[tuple[list[int], frozenset]]: The pairs whose center is of minimal size.
    """
    if not scored:
        return scored
    smallest = min(len(center) for _, center in scored)
    kept = [entry for entry in scored if len(entry[1]) == smallest]
    if len(kept) != len(scored):
        discarded = sorted({len(center) for _, center in scored if len(center) != smallest})
        logger.debug(f'Discarded {len(scored) - len(kept)} of {len(scored)} atom maps for {rxn} whose '
                     f'reaction center ({discarded} changed bonds) exceeds the minimal {smallest}.')
    return kept


def warn_if_atom_map_is_unclustered(rxn, clusters: list, ignore_bond_orders: bool = True) -> bool:
    """
    Warn when the single map ``ARCReaction.atom_map`` returns is in none of the enumerated channels.

    That disagreement is worth surfacing: the clusters are filtered against the family recipe, so a map
    outside all of them usually means the single-map pipeline returned a scrambled result rather than that
    a channel is missing. It shows up as a wildly oversized reaction center - 11 changed bonds against a
    recipe predicting 3, for ``C2H5OO <=> C2H4 + HO2``.

    The atom map is only inspected when it has *already* been computed. Reading ``rxn.atom_map`` here would
    run the whole single-map pipeline as a side effect of clustering, which this module deliberately never
    does.

    Args:
        rxn (ARCReaction): The reaction.
        clusters (list): The clusters just computed for it.
        ignore_bond_orders (bool, optional): Must match the setting the clusters were built with.

    Returns:
        bool: Whether a warning was issued.
    """
    atom_map = getattr(rxn, '_atom_map', None)
    if not atom_map or not clusters:
        return False
    try:
        reactants, products = rxn.get_reactants_and_products(return_copies=True)
        r_graph, p_graph = build_complex_graph(reactants), build_complex_graph(products)
    except (KeyError, ValueError, AttributeError):
        return False
    if not is_permutation_map(atom_map, r_graph.n_atoms, p_graph.n_atoms):
        return False
    center = changed_bonds(r_graph, p_graph, atom_map, ignore_bond_orders=ignore_bond_orders)
    if any(center in cluster.centers for cluster in clusters):
        return False
    sizes = sorted({len(known) for cluster in clusters for known in cluster.centers})
    logger.warning(f'The atom map of {rxn} is in none of its {len(clusters)} enumerated reaction '
                   f'channels: it changes {len(center)} bonds, against {sizes} for the channels. '
                   f'atom_map is most likely scrambled for this reaction.')
    return True


def map_reaction_clusters(rxn,
                          backend: str = 'ARC',
                          include_flipped: bool = True,
                          ignore_bond_orders: bool = True,
                          ) -> list[MapCluster]:
    """
    Convenience wrapper: enumerate every atom map for a reaction and cluster them into distinct channels.

    Args:
        rxn (ARCReaction): The reaction to map.
        backend (str, optional): Currently only ``ARC``'s method is implemented as the backend.
        include_flipped (bool, optional): Whether to also map the flipped reaction.
        ignore_bond_orders (bool, optional): Passed through to :func:`changed_bonds`.

    Returns:
        list[MapCluster]: The clusters, ordered by descending degeneracy.
    """
    clusters = cluster_atom_maps(enumerate_atom_maps(rxn, backend=backend,
                                                     include_flipped=include_flipped),
                                 rxn,
                                 ignore_bond_orders=ignore_bond_orders)
    warn_if_atom_map_is_unclustered(rxn, clusters, ignore_bond_orders=ignore_bond_orders)
    return clusters
