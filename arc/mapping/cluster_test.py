#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.mapping.cluster module
"""

import unittest
import unittest.mock as mock

import arc.mapping.cluster as cluster
from arc.reaction import ARCReaction
from arc.species import ARCSpecies


class TestComplexGraph(unittest.TestCase):
    """
    Contains unit tests for building the flat complex graph.
    """

    def test_build_complex_graph_methane(self):
        """Test that a single species is flattened correctly, with hydrogens attached to their parent."""
        graph = cluster.build_complex_graph([ARCSpecies(label='CH4', smiles='C')])
        self.assertEqual(graph.n_atoms, 5)
        self.assertEqual(graph.symbols, ['C', 'H', 'H', 'H', 'H'])
        self.assertEqual(graph.core, [0])
        self.assertEqual(graph.parent, {1: 0, 2: 0, 3: 0, 4: 0})
        self.assertEqual(len(graph.bonds), 4)
        # The single core atom has no core neighbors, but carries four hydrogens in its invariant.
        self.assertEqual(graph.adj[0], {})
        self.assertEqual(graph.invariant[0][-1], 4)

    def test_build_complex_graph_uses_running_indices(self):
        """Test that atom indices continue across species, matching the atom_map convention."""
        graph = cluster.build_complex_graph([ARCSpecies(label='CH4', smiles='C'),
                                             ARCSpecies(label='OH', smiles='[OH]')])
        self.assertEqual(graph.n_atoms, 7)
        self.assertEqual(graph.symbols, ['C', 'H', 'H', 'H', 'H', 'O', 'H'])
        self.assertEqual(graph.core, [0, 5])
        self.assertEqual(graph.parent, {1: 0, 2: 0, 3: 0, 4: 0, 6: 5})
        # The O-H bond of the second species must be offset onto the running indices.
        self.assertIn(frozenset((5, 6)), graph.bonds)

    def test_build_complex_graph_promotes_parentless_hydrogens(self):
        """Test that a hydrogen without a unique heavy neighbor is promoted to a core atom."""
        h2 = cluster.build_complex_graph([ARCSpecies(label='H2', smiles='[H][H]')])
        self.assertEqual(h2.core, [0, 1])
        self.assertEqual(h2.parent, {})
        h_atom = cluster.build_complex_graph([ARCSpecies(label='H', smiles='[H]')])
        self.assertEqual(h_atom.core, [0])

    def test_build_complex_graph_core_adjacency(self):
        """Test that only core-core bonds enter the adjacency, with their bond orders."""
        graph = cluster.build_complex_graph([ARCSpecies(label='ethene', smiles='C=C')])
        self.assertEqual(graph.core, [0, 1])
        self.assertEqual(graph.adj[0], {1: 2})
        self.assertEqual(graph.adj[1], {0: 2})


class TestAutomorphisms(unittest.TestCase):
    """
    Contains unit tests for the core-skeleton automorphism machinery.
    """

    @staticmethod
    def _automorphism_count(smiles_list, ignore_bond_orders=True):
        """Return |Aut| of the core skeleton of a complex built from a list of SMILES."""
        graph = cluster.build_complex_graph([ARCSpecies(label=f's{i}', smiles=smiles)
                                             for i, smiles in enumerate(smiles_list)])
        automorphisms, truncated = cluster.core_automorphisms(graph, ignore_bond_orders=ignore_bond_orders)
        return len(automorphisms), truncated

    def test_core_automorphism_counts(self):
        """Test |Aut| of the core skeleton against known graph automorphism group orders."""
        for smiles, expected in [('C', 1),               # methane, one core atom
                                 ('CC', 2),              # ethane, swap the two carbons
                                 ('CCC', 2),             # propane, reflect the chain
                                 ('c1ccccc1', 12),       # benzene, the dihedral group D6
                                 ('CC(C)C', 6),          # isobutane, permute three methyls
                                 ('CC(C)(C)C', 24),      # neopentane, permute four methyls
                                 ('c1ccc2ccccc2c1', 4)]:  # naphthalene, Z2 x Z2
            count, truncated = self._automorphism_count([smiles])
            self.assertEqual(count, expected, msg=f'|Aut| of {smiles}')
            self.assertFalse(truncated)

    def test_core_automorphisms_include_molecule_swap(self):
        """Test that exchanging two identical molecules counts as a symmetry of the complex."""
        self.assertEqual(self._automorphism_count(['O[O]', 'O[O]'])[0], 2)
        # Two benzenes: 12 per ring, times the swap.
        self.assertEqual(self._automorphism_count(['c1ccccc1', 'c1ccccc1'])[0], 12 * 12 * 2)

    def test_core_automorphisms_bond_order_sensitivity(self):
        """Test that honoring Kekule bond orders under-counts an aromatic ring's symmetry."""
        # Only the 6 automorphisms preserving the single/double alternation survive.
        self.assertEqual(self._automorphism_count(['c1ccccc1'], ignore_bond_orders=False)[0], 6)
        self.assertEqual(self._automorphism_count(['c1ccccc1'], ignore_bond_orders=True)[0], 12)
        # A saturated species has no multiple bonds, so the setting cannot matter.
        self.assertEqual(self._automorphism_count(['CC(C)(C)C'], ignore_bond_orders=False)[0],
                         self._automorphism_count(['CC(C)(C)C'], ignore_bond_orders=True)[0])

    def test_bond_orders_are_ignored_by_default(self):
        """Test that the default settings ignore bond orders, everywhere the choice is made.

        The explicit-flag tests above would still pass if a default were wired the wrong way round, so this
        pins the defaults themselves. Getting one of them wrong reintroduces the Kekule bug: an aromatic
        ring loses half its automorphisms and its clusters over-split.
        """
        benzene = cluster.build_complex_graph([ARCSpecies(label='benzene', smiles='c1ccccc1')])
        self.assertEqual(len(cluster.core_automorphisms(benzene)[0]), 12)
        ethene = cluster.build_complex_graph([ARCSpecies(label='ethene', smiles='C=C')])
        self.assertEqual(cluster.effective_adjacency(ethene)[0], {1: 1})
        # Colour refinement must see the ring as uniform rather than as alternating bond orders.
        self.assertEqual(len(set(cluster.refine_colors(benzene).values())), 1)

    def test_core_automorphisms_are_genuine(self):
        """Test that every returned permutation really preserves the core adjacency."""
        graph = cluster.build_complex_graph([ARCSpecies(label='isobutane', smiles='CC(C)C')])
        automorphisms, _ = cluster.core_automorphisms(graph)
        adjacency = cluster.effective_adjacency(graph, ignore_bond_orders=True)
        for alpha in automorphisms:
            self.assertEqual(sorted(alpha.keys()), sorted(graph.core))
            self.assertEqual(sorted(alpha.values()), sorted(graph.core))
            for v in graph.core:
                for u in graph.core:
                    self.assertEqual(adjacency[v].get(u), adjacency[alpha[v]].get(alpha[u]))

    def test_core_automorphisms_truncation_is_reported(self):
        """Test that hitting the cap is reported rather than applied silently."""
        graph = cluster.build_complex_graph([ARCSpecies(label='neopentane', smiles='CC(C)(C)C')])
        automorphisms, truncated = cluster.core_automorphisms(graph, max_count=5)
        self.assertTrue(truncated)
        self.assertLessEqual(len(automorphisms), 5)

    def test_core_orbits(self):
        """Test that orbits identify the symmetry-equivalent atoms of propane."""
        graph = cluster.build_complex_graph([ARCSpecies(label='propane', smiles='CCC')])
        automorphisms, _ = cluster.core_automorphisms(graph)
        orbits = cluster.core_orbits(graph, automorphisms)
        # The two terminal carbons share an orbit, the central one is on its own.
        self.assertEqual(orbits[0], orbits[2])
        self.assertNotEqual(orbits[0], orbits[1])
        self.assertEqual(len(set(orbits.values())), 2)

    def test_effective_adjacency_normalizes_orders(self):
        """Test that ignoring bond orders replaces every order by 1 without changing connectivity."""
        graph = cluster.build_complex_graph([ARCSpecies(label='ethene', smiles='C=C')])
        honored = cluster.effective_adjacency(graph, ignore_bond_orders=False)
        ignored = cluster.effective_adjacency(graph, ignore_bond_orders=True)
        self.assertEqual(honored[0], {1: 2})
        self.assertEqual(ignored[0], {1: 1})
        self.assertEqual(set(honored.keys()), set(ignored.keys()))

    def test_refine_colors_separates_inequivalent_atoms(self):
        """Test that colour refinement distinguishes propane's primary and secondary carbons."""
        graph = cluster.build_complex_graph([ARCSpecies(label='propane', smiles='CCC')])
        colors = cluster.refine_colors(graph)
        self.assertEqual(colors[0], colors[2])
        self.assertNotEqual(colors[0], colors[1])


class TestReactionCenters(unittest.TestCase):
    """
    Contains unit tests for changed-bond sets, their hydrogen collapse, and their canonical form.
    """

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None
        # Reactants CH4 + OH: 0=C, 1-4=H, 5=O, 6=H.  Products CH3 + H2O: 0=C, 1-3=H, 4=O, 5-6=H.
        cls.r_graph = cluster.build_complex_graph([ARCSpecies(label='CH4', smiles='C'),
                                                   ARCSpecies(label='OH', smiles='[OH]')])
        cls.p_graph = cluster.build_complex_graph([ARCSpecies(label='CH3', smiles='[CH3]'),
                                                   ARCSpecies(label='H2O', smiles='O')])
        # Abstract reactant hydrogen 1; it becomes product hydrogen 6 of water.
        cls.abstraction_map = [0, 6, 1, 2, 3, 4, 5]

    def test_changed_bonds_h_abstraction(self):
        """Test that an abstraction map yields exactly one broken and one formed bond."""
        center = cluster.changed_bonds(self.r_graph, self.p_graph, self.abstraction_map)
        self.assertEqual(len(center), 2)
        # The C-H bond to reactant hydrogen 1 breaks, and that hydrogen bonds to the oxygen.
        self.assertIn((0, 1, 1, 0), center)
        self.assertIn((1, 5, 0, 1), center)

    def test_changed_bonds_are_expressed_in_reactant_indices(self):
        """Test that every changed-bond endpoint is a valid reactant atom index."""
        center = cluster.changed_bonds(self.r_graph, self.p_graph, self.abstraction_map)
        for i, j, _, _ in center:
            self.assertLess(i, j)
            self.assertLess(j, self.r_graph.n_atoms)

    def test_changed_bonds_invariant_to_product_relabeling(self):
        """Test that relabeling product atoms by a product automorphism leaves the center unchanged.

        This is the property that makes the changed-bond set the right unit of degeneracy: swapping the two
        equivalent hydrogens of the product water is an element of Aut(P) and is not a different path.
        """
        swapped = [5 if index == 6 else 6 if index == 5 else index for index in self.abstraction_map]
        self.assertNotEqual(swapped, self.abstraction_map)
        self.assertEqual(cluster.changed_bonds(self.r_graph, self.p_graph, self.abstraction_map),
                         cluster.changed_bonds(self.r_graph, self.p_graph, swapped))

    def test_changed_bonds_identity_is_empty(self):
        """Test that mapping a complex onto an identical complex changes no bonds."""
        identity = list(range(self.r_graph.n_atoms))
        self.assertEqual(cluster.changed_bonds(self.r_graph, self.r_graph, identity), frozenset())

    def test_collapse_hydrogens(self):
        """Test that hydrogens are replaced by their parent core atom, and core atoms are left alone."""
        center = cluster.changed_bonds(self.r_graph, self.p_graph, self.abstraction_map)
        collapsed = cluster.collapse_hydrogens(center, self.r_graph)
        self.assertEqual(len(collapsed), 2)
        # Reactant hydrogen 1 hangs off carbon 0, the oxygen is core index 5.
        self.assertIn(((cluster.CORE, 0), (cluster.HYDROGEN, 0), 1, 0), collapsed)
        self.assertIn(((cluster.CORE, 5), (cluster.HYDROGEN, 0), 0, 1), collapsed)

    def test_collapse_hydrogens_merges_equivalent_hydrogens(self):
        """Test that abstracting any hydrogen of methane gives the same collapsed center.

        This loss is intended - it is what places all four abstractions in one cluster - and it is why
        degeneracy is counted on the uncollapsed centers instead.
        """
        collapsed = set()
        for hydrogen in (1, 2, 3, 4):
            center = frozenset({(0, hydrogen, 1, 0)})
            collapsed.add(cluster.collapse_hydrogens(center, self.r_graph))
        self.assertEqual(len(collapsed), 1)


class TestCanonicalKey(unittest.TestCase):
    """
    Contains unit tests for canonicalizing a reaction center under the reactant automorphism group.
    """

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.graph = cluster.build_complex_graph([ARCSpecies(label='propane', smiles='CCC')])
        cls.automorphisms, _ = cluster.core_automorphisms(cls.graph)
        cls.hydrogens = dict()
        for hydrogen, parent in cls.graph.parent.items():
            cls.hydrogens.setdefault(parent, list()).append(hydrogen)

    def _key_for_abstraction_at(self, carbon):
        """Build a minimal abstraction center at ``carbon`` and canonicalize it."""
        hydrogen = self.hydrogens[carbon][0]
        center = frozenset({(min(carbon, hydrogen), max(carbon, hydrogen), 1, 0)})
        return cluster.canonical_center_key(cluster.collapse_hydrogens(center, self.graph),
                                            self.automorphisms)

    def test_equivalent_sites_share_a_key(self):
        """Test that abstraction at either primary carbon of propane gives the same key."""
        self.assertEqual(self._key_for_abstraction_at(0), self._key_for_abstraction_at(2))

    def test_inequivalent_sites_have_different_keys(self):
        """Test that abstraction at a primary and at the secondary carbon give different keys."""
        self.assertNotEqual(self._key_for_abstraction_at(0), self._key_for_abstraction_at(1))

    def test_key_is_hashable_and_stable(self):
        """Test that the key is hashable and does not depend on which equivalent site produced it."""
        keys = {self._key_for_abstraction_at(0), self._key_for_abstraction_at(2)}
        self.assertEqual(len(keys), 1)

    def test_key_without_automorphisms_is_the_sorted_center(self):
        """Test that an empty group leaves the center sorted but otherwise untouched."""
        center = frozenset({((cluster.CORE, 1), (cluster.HYDROGEN, 0), 1, 0)})
        self.assertEqual(cluster.canonical_center_key(center, list()), tuple(sorted(center)))


class TestMinimalCenterFilter(unittest.TestCase):
    """
    Contains unit tests for the validity filter that discards scrambled atom maps.
    """

    def test_keeps_only_the_smallest_centers(self):
        """Test that maps with a larger reaction center than the minimum are discarded."""
        small_a = ([0, 1], frozenset({(0, 1, 1, 0)}))
        small_b = ([1, 0], frozenset({(0, 2, 1, 0)}))
        large = ([0, 1], frozenset({(0, 1, 1, 0), (1, 2, 0, 1), (2, 3, 1, 0)}))
        kept = cluster.filter_minimal_centers([small_a, large, small_b])
        self.assertEqual(kept, [small_a, small_b])

    def test_keeps_everything_when_all_centers_match(self):
        """Test that the filter is a no-op when every center is already of minimal size."""
        scored = [([0, 1], frozenset({(0, 1, 1, 0)})), ([1, 0], frozenset({(0, 2, 1, 0)}))]
        self.assertEqual(cluster.filter_minimal_centers(scored), scored)

    def test_empty_input(self):
        """Test that an empty input is passed through."""
        self.assertEqual(cluster.filter_minimal_centers(list()), list())


class _StubReaction(object):
    """A minimal stand-in exposing only what expected_reaction_centers touches."""

    def __init__(self, family=None, product_dicts=None, breaking=None, forming=None):
        self.family = family
        self.product_dicts = product_dicts
        self._breaking = breaking
        self._forming = forming

    def get_expected_changing_bonds(self, r_label_dict, family=None):
        """Return the canned recipe bonds."""
        return self._breaking, self._forming


class TestExpectedCenters(unittest.TestCase):
    """
    Contains unit tests for the family-recipe validity reference.
    """

    def test_returns_none_without_a_family(self):
        """Test that a reaction with no family gives no reference."""
        self.assertIsNone(cluster.expected_reaction_centers(_StubReaction(family=None)))

    def test_returns_none_without_product_dicts(self):
        """Test that a reaction with no template product dictionaries gives no reference."""
        self.assertIsNone(cluster.expected_reaction_centers(_StubReaction(family='H_Abstraction',
                                                                         product_dicts=list())))

    def test_returns_none_when_bond_orders_are_honored(self):
        """Test that the recipe is refused when it cannot describe what changed_bonds would report.

        The recipe only names bond breaking and formation, so it cannot predict the pure order changes that
        changed_bonds reports when bond orders are honored.
        """
        rxn = _StubReaction(family='H_Abstraction', product_dicts=[{'r_label_map': {'*1': 0}}],
                            breaking=[(0, 1)], forming=[(1, 5)])
        self.assertIsNone(cluster.expected_reaction_centers(rxn, ignore_bond_orders=False))
        self.assertIsNotNone(cluster.expected_reaction_centers(rxn, ignore_bond_orders=True))

    def test_returns_none_on_a_degenerate_self_bond(self):
        """Test that a same-label recipe action is rejected rather than used as a wrong reference."""
        rxn = _StubReaction(family='R_Recombination', product_dicts=[{'r_label_map': {'*': 3}}],
                            breaking=list(), forming=[(3, 3)])
        self.assertIsNone(cluster.expected_reaction_centers(rxn))

    def test_unreadable_product_dicts_are_skipped_not_fatal(self):
        """Test that one unreadable product dictionary does not discard the whole reference.

        A family such as intra_H_migration spans template variants with different label sets, so some
        product dictionaries raise KeyError while others are perfectly readable. Abandoning the reference on
        the first failure would silently drop the reaction back to the relative filter.
        """

        class _PartialStub(_StubReaction):
            def get_expected_changing_bonds(self, r_label_dict, family=None):
                if 'good' not in r_label_dict:
                    raise KeyError('*6')
                return [(0, 1)], [(1, 5)]

        rxn = _PartialStub(family='intra_H_migration',
                           product_dicts=[{'r_label_map': {'bad': 0}}, {'r_label_map': {'good': 0}}])
        self.assertEqual(cluster.expected_reaction_centers(rxn), {frozenset({(0, 1, 1, 0), (1, 5, 0, 1)})})

    def test_returns_none_when_every_product_dict_is_unreadable(self):
        """Test that the reference is only abandoned once nothing at all could be read."""

        class _BrokenStub(_StubReaction):
            def get_expected_changing_bonds(self, r_label_dict, family=None):
                raise KeyError('*6')

        rxn = _BrokenStub(family='intra_H_migration', product_dicts=[{'r_label_map': {'a': 0}}])
        self.assertIsNone(cluster.expected_reaction_centers(rxn))

    def test_reverse_discovered_product_dicts_are_skipped(self):
        """Test that a reverse-discovered dictionary is not used to predict centers.

        Its label maps describe the flipped reaction, so the indices they carry are in the flipped frame.
        Building a reference from them predicts centers that match nothing, which silently rejects every
        enumerated map and drops the reaction back to the relative filter.
        """
        rxn = _StubReaction(family='XY_Addition_MultipleBond',
                            product_dicts=[{'r_label_map': {'*1': 0}, 'discovered_in_reverse': True}],
                            breaking=[(0, 1)], forming=[(1, 5)])
        self.assertIsNone(cluster.expected_reaction_centers(rxn))
        # The same dictionary discovered forward is usable.
        rxn.product_dicts = [{'r_label_map': {'*1': 0}, 'discovered_in_reverse': False}]
        self.assertEqual(cluster.expected_reaction_centers(rxn),
                         {frozenset({(0, 1, 1, 0), (1, 5, 0, 1)})})

    def test_builds_one_center_per_product_dict(self):
        """Test that each product dictionary contributes its own predicted center."""
        rxn = _StubReaction(family='H_Abstraction',
                            product_dicts=[{'r_label_map': {'*1': 0}}, {'r_label_map': {'*1': 0}}],
                            breaking=[(0, 1)], forming=[(1, 5)])
        centers = cluster.expected_reaction_centers(rxn)
        # Both dictionaries yield the same canned bonds here, so they collapse to one center.
        self.assertEqual(centers, {frozenset({(0, 1, 1, 0), (1, 5, 0, 1)})})

    def test_filter_expected_centers(self):
        """Test that only maps whose center the recipe predicts are kept."""
        good = ([0, 1], frozenset({(0, 1, 1, 0)}))
        bad = ([1, 0], frozenset({(2, 3, 1, 0)}))
        kept = cluster.filter_expected_centers([good, bad], {frozenset({(0, 1, 1, 0)})})
        self.assertEqual(kept, [good])

    def test_filter_expected_centers_can_reject_everything(self):
        """Test that the filter reports an empty result rather than silently keeping invalid maps."""
        entry = ([0, 1], frozenset({(9, 9, 1, 0)}))
        self.assertEqual(cluster.filter_expected_centers([entry], {frozenset({(0, 1, 1, 0)})}), list())


class TestMapCluster(unittest.TestCase):
    """
    Contains unit tests for the MapCluster container.
    """

    def test_enumerated_degeneracy_counts_centers_not_maps(self):
        """Test that the enumerated count tracks reaction paths, so an Aut(P) relabeling does not inflate it."""
        map_cluster = cluster.MapCluster(representative=[0, 1, 2])
        map_cluster.members.extend([[0, 1, 2], [0, 2, 1]])
        # Both maps describe the same path, so they share a center.
        map_cluster.centers.add(frozenset({(0, 1, 1, 0)}))
        self.assertEqual(len(map_cluster.members), 2)
        self.assertEqual(map_cluster.enumerated_degeneracy, 1)
        # A genuinely different path adds a center.
        map_cluster.centers.add(frozenset({(0, 2, 1, 0)}))
        self.assertEqual(map_cluster.enumerated_degeneracy, 2)

    def test_degeneracy_is_stored_not_derived_from_the_enumeration(self):
        """Test that the reported degeneracy is the exact orbit size, independent of what was enumerated."""
        map_cluster = cluster.MapCluster(representative=[0, 1, 2], degeneracy=3)
        map_cluster.centers.add(frozenset({(0, 1, 1, 0)}))
        # Only one path was enumerated, but three exist.
        self.assertEqual(map_cluster.enumerated_degeneracy, 1)
        self.assertEqual(map_cluster.degeneracy, 3)

    def test_empty_cluster_has_zero_degeneracy(self):
        """Test the degenerate case of a cluster with no recorded center."""
        self.assertEqual(cluster.MapCluster(representative=[0]).degeneracy, 0)
        self.assertEqual(cluster.MapCluster(representative=[0]).enumerated_degeneracy, 0)


class TestCenterDegeneracy(unittest.TestCase):
    """
    Contains unit tests for the exact, orbit-based reaction path degeneracy.
    """

    @staticmethod
    def _degeneracy(smiles_list, center):
        """Compute the exact degeneracy of a center on a complex built from a list of SMILES."""
        graph = cluster.build_complex_graph([ARCSpecies(label=f's{i}', smiles=smiles)
                                             for i, smiles in enumerate(smiles_list)])
        automorphisms, _ = cluster.core_automorphisms(graph)
        return cluster.center_degeneracy(center, graph, automorphisms), graph

    def test_methane_abstraction(self):
        """Test that abstracting one of methane's four equivalent hydrogens has degeneracy 4."""
        # CH4 + OH: 0=C, 1-4=H, 5=O, 6=H. Break C-H(1), form H(1)-O.
        degeneracy, _ = self._degeneracy(['C', '[OH]'], frozenset({(0, 1, 1, 0), (1, 5, 0, 1)}))
        self.assertEqual(degeneracy, 4)

    def test_ethane_abstraction_combines_core_symmetry_and_hydrogen_choice(self):
        """Test that ethane's degeneracy of 6 is the 2 equivalent carbons times 3 hydrogens each."""
        # CC: 0,1=C, 2-7=H. Break C(0)-H(2).
        graph = cluster.build_complex_graph([ARCSpecies(label='ethane', smiles='CC')])
        automorphisms, _ = cluster.core_automorphisms(graph)
        hydrogen = min(h for h, parent in graph.parent.items() if parent == graph.core[0])
        center = frozenset({(graph.core[0], hydrogen, 1, 0)})
        self.assertEqual(cluster.center_degeneracy(center, graph, automorphisms), 6)

    def test_propane_primary_and_secondary_differ(self):
        """Test that propane's primary sites give 6 and its secondary site 2."""
        graph = cluster.build_complex_graph([ARCSpecies(label='propane', smiles='CCC')])
        automorphisms, _ = cluster.core_automorphisms(graph)
        hydrogens = dict()
        for hydrogen, parent in graph.parent.items():
            hydrogens.setdefault(parent, list()).append(hydrogen)
        primary = frozenset({(0, min(hydrogens[0]), 1, 0)})
        secondary = frozenset({(1, min(hydrogens[1]), 1, 0)})
        self.assertEqual(cluster.center_degeneracy(primary, graph, automorphisms), 6)
        self.assertEqual(cluster.center_degeneracy(secondary, graph, automorphisms), 2)

    def test_empty_center_is_one_path_not_zero(self):
        """Test that an empty center has degeneracy 1.

        The orbit of the empty set under any group is the empty set itself, so the count is one. A
        degeneracy of zero is not a meaningful answer for a reaction that exists. An empty center means
        only that no bond was made or broken under the current bond-order setting; see
        test_order_only_reaction_is_reclustered_with_orders_honored for the case that produces one.
        """
        degeneracy, _ = self._degeneracy(['C'], frozenset())
        self.assertEqual(degeneracy, 1)

    def test_interchangeable_hydrogens_count_as_one_path(self):
        """Test that two hydrogens leaving the same carbon in the same role is one path, not two.

        The hydrogen factor counts ordered assignments of hydrogens to the roles the center names, but the
        center is a set. For CH2O losing both hydrogens to H2 the two C-H bonds are interchangeable, so the
        ordered count of 2!/0! = 2 must be divided by the stabilizer of size 2. RMG reports 1 for
        1,2_Insertion_CO on CH2O, and a direct orbit count agrees.
        """
        graph = cluster.build_complex_graph([ARCSpecies(label='CH2O', smiles='C=O')])
        automorphisms, _ = cluster.core_automorphisms(graph)
        hydrogens = sorted(graph.parent)
        carbon = graph.parent[hydrogens[0]]
        both = frozenset({(min(carbon, h), max(carbon, h), 1, 0) for h in hydrogens})
        self.assertEqual(cluster.center_degeneracy(both, graph, automorphisms), 1)
        one = frozenset({(min(carbon, hydrogens[0]), max(carbon, hydrogens[0]), 1, 0)})
        self.assertEqual(cluster.center_degeneracy(one, graph, automorphisms), 2)

    def test_distinguishable_hydrogens_on_one_parent_stay_separate_paths(self):
        """Test that two hydrogens on one carbon in *different* roles remain two paths.

        The stabilizer division must not collapse hydrogens the center tells apart: here one C-H breaks and
        the other changes order, so swapping them yields a different center.
        """
        graph = cluster.build_complex_graph([ARCSpecies(label='CH2O', smiles='C=O')])
        automorphisms, _ = cluster.core_automorphisms(graph)
        h1, h2 = sorted(graph.parent)[:2]
        carbon = graph.parent[h1]
        center = frozenset({(min(carbon, h1), max(carbon, h1), 1, 0),
                            (min(carbon, h2), max(carbon, h2), 1, 2)})
        self.assertEqual(cluster.center_degeneracy(center, graph, automorphisms), 2)


class TestHydrogenStabilizer(unittest.TestCase):
    """
    Contains unit tests for the stabilizer that turns an ordered hydrogen count into a set orbit.
    """

    def test_identical_roles_are_interchangeable(self):
        """Test that hydrogens playing the same role give a stabilizer of their factorial."""
        self.assertEqual(cluster.hydrogen_stabilizer(frozenset({(0, 1, 1, 0), (0, 2, 1, 0)}), [1, 2]), 2)

    def test_different_roles_are_not_interchangeable(self):
        """Test that hydrogens the center tells apart give a trivial stabilizer."""
        self.assertEqual(cluster.hydrogen_stabilizer(frozenset({(0, 1, 1, 0), (0, 2, 1, 2)}), [1, 2]), 1)

    def test_three_identical_roles(self):
        """Test that three interchangeable hydrogens give 3! = 6."""
        center = frozenset({(0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0)})
        self.assertEqual(cluster.hydrogen_stabilizer(center, [1, 2, 3]), 6)

    def test_single_hydrogen_is_trivial(self):
        """Test that one hydrogen can only map to itself."""
        self.assertEqual(cluster.hydrogen_stabilizer(frozenset({(0, 1, 1, 0)}), [1]), 1)


class TestResonanceInsensitiveColoring(unittest.TestCase):
    """
    Contains unit tests for symmetry detection across resonance forms.
    """

    @staticmethod
    def _abstraction_degeneracy(radical_smiles):
        """CH4 + R. -> CH3. + RH, abstracting onto the radical site of R."""
        radical = ARCSpecies(label='R', smiles=radical_smiles)
        methane = ARCSpecies(label='CH4', smiles='C')
        graph = cluster.build_complex_graph([radical, methane])
        site = [i for i, atom in enumerate(radical.mol.atoms) if atom.radical_electrons][0]
        carbon = [v for v in graph.core
                  if graph.symbols[v] == 'C' and v >= len(radical.mol.atoms)][0]
        hydrogen = [h for h, parent in graph.parent.items() if parent == carbon][0]
        center = frozenset({(min(carbon, hydrogen), max(carbon, hydrogen), 1, 0),
                            (min(site, hydrogen), max(site, hydrogen), 0, 1)})
        automorphisms, _ = cluster.core_automorphisms(graph)
        return cluster.center_degeneracy(center, graph, automorphisms)

    def test_delocalized_radicals_match_rmg(self):
        """Test that a radical spread over equivalent sites is as symmetric as it physically is.

        The stored Lewis structure puts the radical on one particular atom, which made allyl's two terminal
        carbons look inequivalent and halved every degeneracy computed through it. Expected values are
        RMG's own reaction path degeneracies for these abstractions.
        """
        self.assertEqual(self._abstraction_degeneracy('[CH2]C=C'), 8)
        self.assertEqual(self._abstraction_degeneracy('[O]N=O'), 8)
        self.assertEqual(self._abstraction_degeneracy('[CH]1C=CC=C1'), 20)

    def test_a_genuinely_unique_site_is_not_inflated(self):
        """Test that benzyl, whose radical carbon is genuinely unique, still gives 4 and not more."""
        self.assertEqual(self._abstraction_degeneracy('[CH2]c1ccccc1'), 4)

    def test_effective_invariant_drops_only_resonance_mobile_attributes(self):
        """Test that the colour seed keeps element and hydrogen count but drops radical placement."""
        graph = cluster.build_complex_graph([ARCSpecies(label='allyl', smiles='[CH2]C=C')])
        insensitive = cluster.effective_invariant(graph, ignore_bond_orders=True)
        terminals = [v for v in graph.core
                     if graph.symbols[v] == 'C' and sum(1 for q in graph.parent.values() if q == v) == 2]
        self.assertEqual(len(terminals), 2)
        self.assertEqual(insensitive[terminals[0]], insensitive[terminals[1]])
        self.assertEqual(cluster.effective_invariant(graph, ignore_bond_orders=False), graph.invariant)
        self.assertNotEqual(graph.invariant[terminals[0]], graph.invariant[terminals[1]])


class TestDuplicateTemplateLabels(unittest.TestCase):
    """
    Contains unit tests for recipes naming one template label twice, as R_Recombination does.
    """

    def test_resolve_duplicate_label(self):
        """Test that the suffixed sibling of a duplicated label is recovered."""
        self.assertEqual(cluster.resolve_duplicate_label(0, {'*': 0, '*_2': 4}), 4)
        self.assertEqual(cluster.resolve_duplicate_label(4, {'*': 0, '*_2': 4}), 0)

    def test_resolve_duplicate_label_without_a_sibling(self):
        """Test that a label map carrying no suffixed variant yields nothing to recover."""
        self.assertIsNone(cluster.resolve_duplicate_label(3, {'*': 3}))
        self.assertIsNone(cluster.resolve_duplicate_label(3, {'*1': 3, '*2': 7}))

    def test_recombination_recipe_yields_a_center(self):
        """Test that R_Recombination now has an absolute reference instead of none at all.

        Its recipe is FORM_BOND(*, *): get_expected_changing_bonds resolves both endpoints to the base
        label's index, so the bond looked degenerate and every R_Recombination reaction fell back to the
        relative minimal-center filter, which cannot reject a uniformly scrambled enumeration.
        """
        rxn = _StubReaction(family='R_Recombination',
                            product_dicts=[{'r_label_map': {'*': 0, '*_2': 4}}],
                            breaking=list(), forming=[(0, 0)])
        self.assertEqual(cluster.expected_reaction_centers(rxn), {frozenset({(0, 4, 0, 1)})})


class TestPermutationValidation(unittest.TestCase):
    """
    Contains unit tests for the guard that keeps malformed atom maps out of changed_bonds().
    """

    def test_accepts_a_genuine_permutation(self):
        """Test that a bijection of the atoms is accepted."""
        self.assertTrue(cluster.is_permutation_map([2, 0, 1], 3, 3))
        self.assertTrue(cluster.is_permutation_map([0, 1, 2], 3, 3))
        self.assertTrue(cluster.is_permutation_map((2, 1, 0), 3, 3))

    def test_rejects_a_wrong_length(self):
        """Test that a map that does not cover every reactant atom is rejected."""
        self.assertFalse(cluster.is_permutation_map([0, 1], 3, 3))
        self.assertFalse(cluster.is_permutation_map([0, 1, 2, 3], 3, 3))

    def test_rejects_duplicates(self):
        """Test that a map sending two reactant atoms to one product atom is rejected.

        This is the case that matters: it has the right length, so a length check passes it through, and it
        leaves changed_bonds() with an inverse map that is missing a product index.
        """
        self.assertFalse(cluster.is_permutation_map([0, 0, 2], 3, 3))

    def test_rejects_out_of_range_indices(self):
        """Test that a product index outside the product complex is rejected."""
        self.assertFalse(cluster.is_permutation_map([0, 1, 3], 3, 3))
        self.assertFalse(cluster.is_permutation_map([0, 1, -1], 3, 3))

    def test_rejects_non_integer_entries(self):
        """Test that entries which cannot index a graph are rejected."""
        self.assertFalse(cluster.is_permutation_map([0, 1, None], 3, 3))
        self.assertFalse(cluster.is_permutation_map([0, 1, 2.5], 3, 3))
        self.assertFalse(cluster.is_permutation_map([0, 1, '2'], 3, 3))

    def test_rejects_a_non_sequence(self):
        """Test that something which is not a map at all is rejected rather than raising."""
        self.assertFalse(cluster.is_permutation_map(None, 3, 3))
        self.assertFalse(cluster.is_permutation_map({0: 0, 1: 1, 2: 2}, 3, 3))

    def test_rejects_a_map_onto_a_differently_sized_product(self):
        """Test that a map which cannot be onto the product complex is rejected."""
        self.assertFalse(cluster.is_permutation_map([0, 1, 2], 3, 4))


class TestClusteringIntegration(unittest.TestCase):
    """
    Contains end-to-end unit tests running the full enumerate-and-cluster pipeline on a real reaction.
    """

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None
        ch4_xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'), 'isotopes': (12, 1, 1, 1, 1),
                   'coords': ((-5.45906343962835e-10, 4.233517924761169e-10, 2.9505240956083194e-10),
                              (-0.6505520089868748, -0.7742801979689132, -0.4125187934483119),
                              (-0.34927557824779626, 0.9815958255612931, -0.3276823191685369),
                              (-0.022337921721882443, -0.04887374527620588, 1.0908766524267022),
                              (1.0221655095024578, -0.15844188273952128, -0.350675540104908))}
        oh_xyz = """O       0.48890387    0.00000000    0.00000000
                    H      -0.48890387    0.00000000    0.00000000"""
        ch3_xyz = """C       0.00000000    0.00000001   -0.00000000
                     H       1.06690511   -0.17519582    0.05416493
                     H      -0.68531716   -0.83753536   -0.02808565
                     H      -0.38158795    1.01273118   -0.02607927"""
        h2o_xyz = """O      -0.00032832    0.39781490    0.00000000
                     H      -0.76330345   -0.19953755    0.00000000
                     H       0.76363177   -0.19827735    0.00000000"""
        cls.rxn = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C', xyz=ch4_xyz),
                                         ARCSpecies(label='OH', smiles='[OH]', xyz=oh_xyz)],
                              p_species=[ARCSpecies(label='CH3', smiles='[CH3]', xyz=ch3_xyz),
                                         ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz)])
        cls.atom_maps = cluster.enumerate_atom_maps(cls.rxn)

    def test_enumerate_atom_maps_returns_several_valid_permutations(self):
        """Test that enumeration finds more than one map and that each is a permutation of the atoms."""
        self.assertGreater(len(self.atom_maps), 1)
        for atom_map in self.atom_maps:
            self.assertEqual(sorted(atom_map), list(range(7)))

    def test_enumerate_atom_maps_are_distinct(self):
        """Test that the enumeration does not report the same map twice."""
        self.assertEqual(len({tuple(atom_map) for atom_map in self.atom_maps}), len(self.atom_maps))

    def test_enumerate_atom_maps_honors_the_cap(self):
        """Test that the cap on the number of enumerated maps is respected."""
        self.assertLessEqual(len(cluster.enumerate_atom_maps(self.rxn, max_maps=2)), 2)

    def test_cluster_atom_maps_finds_a_single_channel(self):
        """Test that abstracting any of methane's four hydrogens is recognized as one channel."""
        clusters = cluster.cluster_atom_maps(self.atom_maps, self.rxn)
        self.assertEqual(len(clusters), 1)

    def test_cluster_degeneracy_matches_the_reaction_path_degeneracy(self):
        """Test that the reaction path degeneracy of CH4 + OH abstraction is recovered as 4."""
        clusters = cluster.cluster_atom_maps(self.atom_maps, self.rxn)
        self.assertEqual(clusters[0].degeneracy, 4)
        self.assertFalse(clusters[0].truncated)

    def test_cluster_signature_describes_the_abstraction(self):
        """Test that the signature reports one C-H bond breaking and one O-H bond forming."""
        clusters = cluster.cluster_atom_maps(self.atom_maps, self.rxn)
        signature = clusters[0].signature
        self.assertEqual(len(signature), 2)
        broken = [entry for entry in signature if (entry[2], entry[3]) == (1, 0)]
        formed = [entry for entry in signature if (entry[2], entry[3]) == (0, 1)]
        self.assertEqual(len(broken), 1)
        self.assertEqual(len(formed), 1)
        self.assertIn('C', broken[0][0] + broken[0][1])
        self.assertIn('O', formed[0][0] + formed[0][1])

    def test_cluster_representative_is_a_member(self):
        """Test that a cluster's representative is one of its own members."""
        for map_cluster in cluster.cluster_atom_maps(self.atom_maps, self.rxn):
            self.assertIn(map_cluster.representative, map_cluster.members)

    def test_cluster_atom_maps_with_no_maps(self):
        """Test that clustering an empty list returns no clusters."""
        self.assertEqual(cluster.cluster_atom_maps(list(), self.rxn), list())

    def test_cluster_atom_maps_skips_wrongly_sized_maps(self):
        """Test that a map whose length does not match the reactant complex is skipped."""
        self.assertEqual(cluster.cluster_atom_maps([[0, 1, 2]], self.rxn), list())

    def test_map_reaction_clusters_wrapper(self):
        """Test that the convenience wrapper reproduces enumerate followed by cluster."""
        clusters = cluster.map_reaction_clusters(self.rxn)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].degeneracy, 4)

    def test_expected_reaction_centers_matches_the_product_dicts(self):
        """Test that the family recipe predicts one abstraction center per methane hydrogen.

        Reactant frame: 0=C, 1-4=H of methane, 5=O, 6=H of the hydroxyl. Abstracting hydrogen h breaks
        (0, h) and forms (h, 5).
        """
        centers = cluster.expected_reaction_centers(self.rxn)
        self.assertEqual(len(centers), 4)
        for hydrogen in (1, 2, 3, 4):
            self.assertIn(frozenset({(0, hydrogen, 1, 0), (hydrogen, 5, 0, 1)}), centers)

    def test_clustered_maps_all_have_a_recipe_predicted_center(self):
        """Test that every map surviving validation has a center the family recipe predicts.

        This pins that the absolute reference is what actually admitted the maps, rather than the relative
        minimal-center fallback silently doing the work.
        """
        expected = cluster.expected_reaction_centers(self.rxn)
        self.assertTrue(expected)
        clusters = cluster.cluster_atom_maps(self.atom_maps, self.rxn)
        for map_cluster in clusters:
            for center in map_cluster.centers:
                self.assertIn(center, expected)

    def test_enumerates_a_reverse_discovered_reaction(self):
        """Test that a reaction whose template was only discovered in reverse still yields maps.

        Every forward product dictionary of such a reaction fails at ``get_template_product_order``, so the
        whole enumeration rests on the flipped sweep. That sweep is useless unless the flipped reaction is
        seeded with a family, since ``flip_reaction`` resets it and the default family set finds nothing.
        """
        rxn = ARCReaction(r_species=[ARCSpecies(label='C2H5Cl', smiles='CCCl')],
                          p_species=[ARCSpecies(label='C2H4', smiles='C=C'),
                                     ARCSpecies(label='HCl', smiles='Cl')])
        rxn.product_dicts = rxn.get_product_dicts(rmg_family_set='all')
        rxn.family = rxn.product_dicts[0]['family']
        rxn.family_own_reverse = rxn.product_dicts[0]['own_reverse']
        self.assertTrue(rxn.product_dicts[0]['discovered_in_reverse'])
        atom_maps = cluster.enumerate_atom_maps(rxn, max_maps=20)
        self.assertTrue(atom_maps)
        for atom_map in atom_maps:
            self.assertEqual(sorted(atom_map), list(range(len(atom_map))))
        clusters = cluster.cluster_atom_maps(atom_maps, rxn)
        self.assertEqual(len(clusters), 1)

    def test_sec_butyl_to_n_butyl_has_two_channels(self):
        """Test that sec-butyl -> n-butyl is recognized as two distinct channels, three paths each.

        ``CC[CH]C <=> CCC[CH2]`` is C0-C1-C2(.)-C3 losing a hydrogen from a terminal methyl onto the radical
        carbon C2. Either methyl works, and both give n-butyl, because n-butyl is the same molecule numbered
        from either end:

            C3 -> C2 is a 1,2 shift through a three-membered TS
            C0 -> C2 is a 1,3 shift through a four-membered TS

        The two are not symmetry-equivalent - sec-butyl has no automorphism exchanging its ends, and the two
        transition states are structurally different - so they must not merge into one channel of degeneracy
        6. Each carries the three hydrogens of its methyl.

        Both transition states have been located and optimized separately, with distinct barriers, by
        supplying the two atom maps below to ARC by hand. This test pins that the clustering derives the same
        two channels without them.
        """
        rxn = ARCReaction(r_species=[ARCSpecies(label='sec-butyl', smiles='CC[CH]C')],
                          p_species=[ARCSpecies(label='n-butyl', smiles='CCC[CH2]')])
        self.assertEqual(rxn.family, 'intra_H_migration')

        # The two hand-written maps, one per located transition state.
        one_two_shift = [0, 1, 2, 3, 6, 4, 5, 8, 7, 10, 9, 12, 11]
        one_three_shift = [3, 2, 1, 0, 7, 12, 11, 10, 9, 8, 4, 5, 6]
        manual = cluster.cluster_atom_maps([one_two_shift, one_three_shift], rxn)
        self.assertEqual(len(manual), 2)
        for map_cluster in manual:
            self.assertEqual(map_cluster.degeneracy, 3)

        # The enumeration finds the same two channels unaided.
        enumerated = cluster.cluster_atom_maps(cluster.enumerate_atom_maps(rxn), rxn)
        self.assertEqual(len(enumerated), 2)
        self.assertEqual(sorted(c.degeneracy for c in enumerated), [3, 3])
        self.assertEqual({c.key for c in manual}, {c.key for c in enumerated})

        # One channel moves a hydrogen off C0, the other off C3; both onto the radical carbon C2.
        donors = {tuple(sorted(entry[:2])) for c in enumerated for entry in c.signature
                  if (entry[2], entry[3]) == (1, 0)}
        self.assertEqual(len(donors), 2)

    def test_arc_reaction_atom_map_clusters_property(self):
        """Test that ARCReaction exposes the clusters and caches them."""
        clusters = self.rxn.atom_map_clusters
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].degeneracy, 4)
        self.assertIn(clusters[0].representative, clusters[0].members)
        # A second access must reuse the cached result rather than recomputing.
        self.assertIs(self.rxn.atom_map_clusters, clusters)

    def test_arc_reaction_atom_map_does_not_trigger_clustering(self):
        """Test that the cheap atom_map property never pays for the expensive enumeration."""
        rxn = ARCReaction(r_species=[spc.copy() for spc in self.rxn.r_species],
                          p_species=[spc.copy() for spc in self.rxn.p_species])
        self.assertIsNone(rxn._atom_map_clusters)
        self.assertIsNotNone(rxn.atom_map)
        self.assertIsNone(rxn._atom_map_clusters)

    def test_arc_reaction_atom_map_clusters_is_settable(self):
        """Test that the clusters can be set, and reset to None to force a recomputation."""
        rxn = ARCReaction(r_species=[spc.copy() for spc in self.rxn.r_species],
                          p_species=[spc.copy() for spc in self.rxn.p_species])
        rxn.atom_map_clusters = ['sentinel']
        self.assertEqual(rxn.atom_map_clusters, ['sentinel'])
        rxn.atom_map_clusters = None
        self.assertIsNone(rxn._atom_map_clusters)

    def test_cluster_atom_maps_skips_a_non_permutation_map(self):
        """Test that a malformed map is discarded instead of aborting the whole clustering run.

        changed_bonds() inverts the atom map, so a duplicated product index leaves it with a missing key and
        it raises KeyError. A length check alone lets such a map through, since it has the right length.
        """
        malformed = [0, 0, 1, 2, 3, 4, 5]
        self.assertEqual(len(malformed), len(self.atom_maps[0]))
        self.assertNotEqual(sorted(malformed), sorted(self.atom_maps[0]))
        clusters = cluster.cluster_atom_maps(list(self.atom_maps) + [malformed], self.rxn)
        # The valid maps are still clustered exactly as they are without the malformed one.
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].degeneracy, 4)
        self.assertNotIn(malformed, clusters[0].members)

    def test_cluster_atom_maps_survives_only_malformed_input(self):
        """Test that clustering reports no channels rather than raising when nothing is usable."""
        self.assertEqual(cluster.cluster_atom_maps([[0, 0, 1, 2, 3, 4, 5]], self.rxn), list())

    def test_atom_map_clusters_does_not_log_an_error_without_geometries(self):
        """Test that 'not yet computable' is not reported as a mapping failure.

        Without geometries nothing is attempted, so an error here would be indistinguishable from a real
        failure and would repeat on every access.
        """
        rxn = ARCReaction(r_species=[spc.copy() for spc in self.rxn.r_species],
                          p_species=[spc.copy() for spc in self.rxn.p_species])
        # A species whose geometry is not available yet.
        rxn.r_species[0].get_xyz = lambda *args, **kwargs: None
        with mock.patch('arc.reaction.reaction.map_reaction_clusters') as mocked_map:
            with mock.patch('arc.reaction.reaction.logger') as mocked_logger:
                self.assertIsNone(rxn.atom_map_clusters)
                self.assertIsNone(rxn.atom_map_clusters)
        # Nothing was attempted, so nothing is reported and nothing is cached.
        mocked_map.assert_not_called()
        mocked_logger.error.assert_not_called()
        self.assertIsNone(rxn._atom_map_clusters)

    def test_atom_map_clusters_logs_a_failure_once_and_caches_it(self):
        """Test that an attempt which yields no channels is reported once, not on every access."""
        rxn = ARCReaction(r_species=[spc.copy() for spc in self.rxn.r_species],
                          p_species=[spc.copy() for spc in self.rxn.p_species])
        with mock.patch('arc.reaction.reaction.map_reaction_clusters', return_value=list()) as mocked_map:
            with mock.patch('arc.reaction.reaction.logger') as mocked_logger:
                self.assertEqual(rxn.atom_map_clusters, list())
                self.assertEqual(rxn.atom_map_clusters, list())
                self.assertEqual(rxn.atom_map_clusters, list())
        # Attempted once, reported once, then served from the cache.
        self.assertEqual(mocked_map.call_count, 1)
        self.assertEqual(mocked_logger.error.call_count, 1)

    def test_order_only_reaction_is_reclustered_with_orders_honored(self):
        """Test that a reaction changing only bond orders still yields a real channel.

        1,2-Birad_to_alkene and Singlet_Val6_to_triplet make and break nothing; they only change orders.
        With orders ignored their whole reaction center vanishes, which left nothing to cluster on, nothing
        for either validity filter to judge, and a degenerate answer. Clustering now detects that every
        center came out empty and retries with orders honored.
        """
        reactant = ARCSpecies(label='birad', smiles='[CH2][CH2]')
        product = ARCSpecies(label='ethene', smiles='C=C')
        rxn = ARCReaction(r_species=[reactant], p_species=[product])
        identity = list(range(len(reactant.mol.atoms)))
        r_graph = cluster.build_complex_graph([reactant])
        p_graph = cluster.build_complex_graph([product])
        # Ignoring orders erases the center entirely.
        self.assertEqual(cluster.changed_bonds(r_graph, p_graph, identity, ignore_bond_orders=True),
                         frozenset())
        clusters = cluster.cluster_atom_maps([identity], rxn)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].degeneracy, 1)
        centers = list(clusters[0].centers)
        self.assertEqual(len(centers), 1)
        # The C-C bond going from single to double is the reaction center.
        self.assertEqual(sorted(centers[0]), [(0, 1, 1.0, 2.0)])

    def test_warns_when_the_atom_map_is_in_no_cluster(self):
        """Test that a single-map answer outside every enumerated channel is reported."""
        rxn = ARCReaction(r_species=[spc.copy() for spc in self.rxn.r_species],
                          p_species=[spc.copy() for spc in self.rxn.p_species])
        clusters = cluster.cluster_atom_maps(self.atom_maps, rxn)
        self.assertTrue(clusters)
        scrambled = list(reversed(range(7)))
        self.assertEqual(sorted(scrambled), list(range(7)))
        rxn._atom_map = scrambled
        with mock.patch('arc.mapping.cluster.logger') as mocked_logger:
            self.assertTrue(cluster.warn_if_atom_map_is_unclustered(rxn, clusters))
        self.assertEqual(mocked_logger.warning.call_count, 1)

    def test_does_not_warn_when_the_atom_map_matches_a_cluster(self):
        """Test that agreement between the single map and the channels is silent."""
        rxn = ARCReaction(r_species=[spc.copy() for spc in self.rxn.r_species],
                          p_species=[spc.copy() for spc in self.rxn.p_species])
        clusters = cluster.cluster_atom_maps(self.atom_maps, rxn)
        rxn._atom_map = clusters[0].representative
        with mock.patch('arc.mapping.cluster.logger') as mocked_logger:
            self.assertFalse(cluster.warn_if_atom_map_is_unclustered(rxn, clusters))
        mocked_logger.warning.assert_not_called()

    def test_does_not_compute_the_atom_map_just_to_check_it(self):
        """Test that the check never triggers the single-map pipeline as a side effect of clustering."""
        rxn = ARCReaction(r_species=[spc.copy() for spc in self.rxn.r_species],
                          p_species=[spc.copy() for spc in self.rxn.p_species])
        clusters = cluster.cluster_atom_maps(self.atom_maps, rxn)
        self.assertIsNone(rxn._atom_map)
        self.assertFalse(cluster.warn_if_atom_map_is_unclustered(rxn, clusters))
        self.assertIsNone(rxn._atom_map)

    def test_recipe_validated_marks_channels_backed_by_the_family_recipe(self):
        """Test that a channel the recipe predicts is marked validated."""
        clusters = cluster.cluster_atom_maps(self.atom_maps, self.rxn)
        self.assertTrue(clusters[0].recipe_validated)

    def test_recipe_validated_is_false_when_the_recipe_rejects_everything(self):
        """Test that a channel built only from the relative heuristic is flagged as unvalidated.

        CH3 + CH3 <=> C2H6 enumerates a single scrambled map whose center changes 13 bonds, against the
        recipe's one forming C-C bond. The recipe now rejects it, and the fallback channel must not be
        presented as recipe-backed.
        """
        rxn = ARCReaction(r_species=[ARCSpecies(label='a', smiles='[CH3]'),
                                     ARCSpecies(label='b', smiles='[CH3]')],
                          p_species=[ARCSpecies(label='c', smiles='CC')])
        scrambled = [1, 2, 3, 4, 0, 5, 6, 7]
        clusters = cluster.cluster_atom_maps([scrambled], rxn)
        self.assertEqual(len(clusters), 1)
        self.assertFalse(clusters[0].recipe_validated)
        # The correct map is accepted by the recipe and yields a one-bond center.
        clusters = cluster.cluster_atom_maps([[0, 2, 3, 4, 1, 5, 6, 7]], rxn)
        self.assertTrue(clusters[0].recipe_validated)
        self.assertEqual(sorted(list(clusters[0].centers)[0]), [(0, 4, 0, 1)])
        self.assertEqual(clusters[0].degeneracy, 1)

    def test_arc_reaction_atom_map_clusters_not_persisted(self):
        """Test that as_dict carries the atom map but never the derived clusters."""
        rxn = ARCReaction(r_species=[spc.copy() for spc in self.rxn.r_species],
                          p_species=[spc.copy() for spc in self.rxn.p_species])
        self.assertEqual(len(rxn.atom_map_clusters), 1)
        # Clustering must not populate the atom map as a side effect, so as_dict still omits it.
        self.assertIsNone(rxn._atom_map)
        self.assertNotIn('atom_map', rxn.as_dict())
        # Once the atom map is computed it is persisted, but the clusters never are.
        self.assertIsNotNone(rxn.atom_map)
        reaction_dict = rxn.as_dict()
        self.assertIn('atom_map', reaction_dict)
        self.assertNotIn('atom_map_clusters', reaction_dict)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
