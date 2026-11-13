#!/usr/bin/env python3
# encoding: utf-8


import random
import unittest

from numpy import testing

from arc.molecule.graph import Edge, Graph, Vertex, get_vertex_connectivity_value
from arc.molecule.group import Group
from arc.molecule.molecule import Molecule
from arc.molecule.vf2 import VF2


class TestVF2(unittest.TestCase):
    """
    Contains unit tests of the methods for computing symmetry numbers for a
    given Molecule object.
    """

    def setUp(self):
        self.vf2 = VF2()
        self.mol = Molecule().from_smiles("CC(=O)C[CH2]")
        self.mol2 = self.mol.copy(deep=True)

    def test_import_graph(self):
        """Test that we can add graphs to the object and that they are sorted"""

        self.mol.sort_vertices()

        ordered_original_connectivity_order = [get_vertex_connectivity_value(atom) for atom in self.mol.atoms]

        self.vf2.graphA = self.mol
        self.vf2.graphB = self.mol2

        final_connectivity_order = [get_vertex_connectivity_value(atom) for atom in self.vf2.graphA.atoms]
        final_connectivity_order2 = [get_vertex_connectivity_value(atom) for atom in self.vf2.graphB.atoms]

        testing.assert_array_equal(final_connectivity_order, final_connectivity_order2)
        testing.assert_array_equal(final_connectivity_order, ordered_original_connectivity_order)

    def test_feasible(self):
        """
        Test that feasibility returns correct values on highly functional molecule
        
        `feasible` method isn't perfect in assigning values, but it should do a good
        job on highly functional values
        """

        self.vf2.graphA = self.mol
        self.vf2.graphB = self.mol2

        for atom1 in self.vf2.graphA.atoms:
            for atom2 in self.vf2.graphB.atoms:
                # same connectivity values should result in `feasible` being true
                if get_vertex_connectivity_value(atom1) == get_vertex_connectivity_value(atom2):
                    self.assertTrue(self.vf2.feasible(atom1, atom2))
                else:  # different connectivity values should return false
                    self.assertFalse(self.vf2.feasible(atom1, atom2))

    def test_clear_mapping(self):
        """Test that vertex mapping is cleared after isomorphism."""
        self.vf2.is_isomorphic(self.mol, self.mol2, dict())

        for atom in self.mol.atoms:
            self.assertIsNone(atom.mapping)
            self.assertFalse(atom.terminal)


class RaisingVertex(Vertex):
    """A vertex whose equivalence check raises once a neighbour is mapped, i.e. mid-match."""

    def equivalent(self, other, strict=True):
        """
        Match any vertex until a neighbour of `other` is mapped, then raise.

        Args:
            other (Vertex): The vertex to compare against.
            strict (bool): Unused.

        Returns:
            bool: ``True`` while no neighbour of `other` is mapped.
        """
        if any(neighbour.mapping is not None for neighbour in other.edges):
            raise RuntimeError('equivalent() failed')
        return True


def make_chain(n, vertex_class=Vertex):
    """
    Make a linear chain graph whose terminal vertices come first, so sorting reorders it.

    Args:
        n (int): The number of vertices.
        vertex_class (type): The vertex class to instantiate.

    Returns:
        Graph: The chain graph.
    """
    vertices = [vertex_class() for _ in range(n)]
    graph = Graph(vertices)
    for i in range(n - 1):
        graph.add_edge(Edge(vertices[i], vertices[i + 1]))
    graph.vertices = [vertices[0], vertices[-1]] + vertices[1:-1]
    return graph


class TestVF2SaveOrder(unittest.TestCase):
    """
    Contains unit tests that save_order=True leaves the atom order of both graphs unchanged,
    whatever calls ran on the same graphs before.
    """

    def setUp(self):
        self.mol_a = Molecule(smiles='CC=O')
        self.mol_b = Molecule(smiles='O=CC')
        self.order_a = list(self.mol_a.atoms)
        self.order_b = list(self.mol_b.atoms)

    def assert_orders_kept(self):
        """Assert that both fixture molecules keep the atom order they had in setUp."""
        self.assertEqual(self.mol_a.atoms, self.order_a)
        self.assertEqual(self.mol_b.atoms, self.order_b)

    def test_fixture_is_reordered_by_a_plain_call(self):
        """A plain call reorders the fixture atoms."""
        mol_a, mol_b = self.mol_a.copy(deep=True), self.mol_b.copy(deep=True)
        order_a = list(mol_a.atoms)
        self.assertTrue(mol_a.is_isomorphic(mol_b))
        self.assertNotEqual(mol_a.atoms, order_a)

    def test_plain_then_save_order_same_direction(self):
        """A save_order call right after a plain call on the same pair restores the order."""
        self.assertTrue(self.mol_a.is_isomorphic(self.mol_b))
        self.order_a, self.order_b = list(self.mol_a.atoms), list(self.mol_b.atoms)
        self.order_a = self.order_a[::-1]
        self.mol_a.atoms = list(self.order_a)
        self.assertTrue(self.mol_a.is_isomorphic(self.mol_b, save_order=True))
        self.assert_orders_kept()

    def test_plain_then_save_order_reverse_direction(self):
        """The same, with the operands swapped in both calls."""
        self.assertTrue(self.mol_b.is_isomorphic(self.mol_a))
        self.order_a, self.order_b = list(self.mol_a.atoms), list(self.mol_b.atoms)
        self.assertTrue(self.mol_b.is_isomorphic(self.mol_a, save_order=True))
        self.assert_orders_kept()

    def test_save_order_twice_with_reorder_between(self):
        """A reorder made between two save_order calls on the same pair is kept, not reverted."""
        self.assertTrue(self.mol_a.is_isomorphic(self.mol_b, save_order=True))
        self.assert_orders_kept()
        self.order_a = self.order_a[::-1]
        self.mol_a.atoms = list(self.order_a)
        self.assertTrue(self.mol_a.is_isomorphic(self.mol_b, save_order=True))
        self.assert_orders_kept()

    def test_save_order_ab_then_ba_with_reorder_between(self):
        """A reorder is kept across save_order calls in both directions (a, b) then (b, a)."""
        self.assertTrue(self.mol_a.is_isomorphic(self.mol_b, save_order=True))
        self.assertTrue(self.mol_b.is_isomorphic(self.mol_a, save_order=True))
        self.assert_orders_kept()
        self.order_a = self.order_a[::-1]
        self.mol_a.atoms = list(self.order_a)
        self.assertTrue(self.mol_b.is_isomorphic(self.mol_a, save_order=True))
        self.assert_orders_kept()

    def test_save_order_restores_in_place(self):
        """A reference to the atom list taken before the call sees the original order afterwards."""
        atoms_a, atoms_b = self.mol_a.atoms, self.mol_b.atoms
        self.assertTrue(self.mol_a.is_isomorphic(self.mol_b, save_order=True))
        self.assertIs(self.mol_a.atoms, atoms_a)
        self.assertIs(self.mol_b.atoms, atoms_b)
        self.assert_orders_kept()

    def test_save_order_gives_the_same_result(self):
        """save_order=True gives the same result as save_order=False, for a match and a non-match."""
        for smiles_b, expected in (('O=CC', True), ('C=CO', False)):
            for save_order in (False, True):
                mol_a, mol_b = Molecule(smiles='CC=O'), Molecule(smiles=smiles_b)
                self.assertEqual(mol_a.is_isomorphic(mol_b), expected)
                self.assertEqual(mol_a.is_isomorphic(mol_b, save_order=save_order), expected)
                self.assertEqual(bool(mol_a.find_isomorphism(mol_b, save_order=save_order)), expected,
                                 msg=f'find_isomorphism(CC=O, {smiles_b}, save_order={save_order})')

    def test_find_isomorphism_plain_then_save_order(self):
        """find_isomorphism with save_order after a plain call restores the order and finds the same maps."""
        plain = self.mol_a.find_isomorphism(self.mol_b)
        self.order_a, self.order_b = list(self.mol_a.atoms), list(self.mol_b.atoms)
        self.order_b = self.order_b[::-1]
        self.mol_b.atoms = list(self.order_b)
        saved = self.mol_a.find_isomorphism(self.mol_b, save_order=True)
        self.assert_orders_kept()
        self.assertEqual(len(saved), len(plain))
        self.assertEqual({frozenset(m.items()) for m in saved}, {frozenset(m.items()) for m in plain})

    def test_save_order_self_comparison(self):
        """Comparing a reordered molecule with itself under save_order keeps its order, on a fresh VF2 cache."""
        for reset_labels in (False, True):
            mol = Molecule(smiles='CC=O')
            self.assertTrue(mol.is_isomorphic(mol.copy(deep=True)))
            order = list(mol.atoms)[::-1]
            mol.atoms = list(order)
            if reset_labels:
                for atom in mol.atoms:
                    atom.sorting_label = -1
            self.assertTrue(VF2().is_isomorphic(mol, mol, None, save_order=True))
            self.assertEqual(mol.atoms, order, msg=f'reset_labels={reset_labels}')

    def test_save_order_on_size_mismatch(self):
        """The early return for graphs of different size restores the order."""
        graph1, graph2 = make_chain(4), make_chain(5)
        order1, order2 = list(graph1.vertices), list(graph2.vertices)
        self.assertFalse(graph1.is_isomorphic(graph2, save_order=True))
        self.assertEqual(graph1.vertices, order1)
        self.assertEqual(graph2.vertices, order2)

    def test_save_order_on_subgraph_size_mismatch(self):
        """The early return for a subgraph larger than the graph restores the order."""
        graph1, graph2 = make_chain(4), make_chain(5)
        order1, order2 = list(graph1.vertices), list(graph2.vertices)
        self.assertFalse(graph1.is_subgraph_isomorphic(graph2, save_order=True))
        self.assertEqual(graph1.vertices, order1)
        self.assertEqual(graph2.vertices, order2)

    def test_save_order_on_exception_in_match(self):
        """An exception raised inside the match propagates, and the order and mappings are restored."""
        graph1, graph2 = make_chain(4, RaisingVertex), make_chain(4, RaisingVertex)
        order1, order2 = list(graph1.vertices), list(graph2.vertices)
        with self.assertRaises(RuntimeError):
            graph1.is_isomorphic(graph2, save_order=True)
        self.assertEqual(graph1.vertices, order1)
        self.assertEqual(graph2.vertices, order2)
        for vertex in graph1.vertices + graph2.vertices:
            self.assertIsNone(vertex.mapping)
            self.assertFalse(vertex.terminal)

    def test_subgraph_plain_then_save_order(self):
        """Subgraph isomorphism with save_order after a plain call restores both orders, same result."""
        mol = Molecule(smiles='CCO')
        group = Group().from_adjacency_list("""1 *1 C u0 {2,S} {3,S}
2 *2 O u0 {1,S}
3    C u0 {1,S}""")
        self.assertTrue(mol.is_subgraph_isomorphic(group))
        plain_count = len(mol.find_subgraph_isomorphisms(group))
        mol_order = list(mol.atoms)[::-1]
        mol.atoms = list(mol_order)
        group_order = list(group.atoms)
        self.assertTrue(mol.is_subgraph_isomorphic(group, save_order=True))
        self.assertEqual(len(mol.find_subgraph_isomorphisms(group, save_order=True)), plain_count)
        self.assertEqual(mol.atoms, mol_order)
        self.assertEqual(group.atoms, group_order)


def shuffled_butane(seed):
    """
    Make butane with its atoms shuffled and its sorting labels unset, like a molecule built from xyz.

    Args:
        seed (int): The shuffle seed.

    Returns:
        Molecule: The molecule.
    """
    mol = Molecule(smiles='CCCC')
    random.Random(seed).shuffle(mol.atoms)
    for atom in mol.atoms:
        atom.sorting_label = -1
    return mol


def ethane_as(mol):
    """
    Make ethane carrying the fingerprint of `mol`, so a size-mismatch comparison with it reaches VF2.

    Args:
        mol (Molecule): The molecule whose fingerprint to copy.

    Returns:
        Molecule: The ethane molecule.
    """
    ethane = Molecule(smiles='CC')
    ethane.fingerprint = mol.fingerprint
    return ethane


def map_indices(maps, mol_1, mol_2):
    """
    Express isomorphism mappings as atom index pairs.

    Args:
        maps (list): The mappings returned by find_isomorphism.
        mol_1 (Molecule): The molecule whose atoms are the keys.
        mol_2 (Molecule): The molecule whose atoms are the values.

    Returns:
        list: One sorted list of (index in mol_1, index in mol_2) pairs per mapping, in the order found.
    """
    return [sorted((mol_1.atoms.index(k), mol_2.atoms.index(v)) for k, v in m.items()) for m in maps]


class TestVF2SaveOrderEarlyExit(unittest.TestCase):
    """
    Contains unit tests that a save_order call which exits before a full match of two distinct
    graphs (size mismatch, self-comparison) leaves the sorting labels and the VF2 cache as if it
    never ran, so the next search sorts the graphs.
    """

    def assert_labels_unset(self, graph):
        """Assert that every sorting label of `graph` is unset."""
        self.assertEqual([vertex.sorting_label for vertex in graph.vertices], [-1] * len(graph.vertices))

    def test_size_mismatch_leaves_labels_unset(self):
        """A size-mismatch save_order call keeps the order and leaves unset labels unset."""
        graph1, graph2 = make_chain(4), make_chain(5)
        order1, order2 = list(graph1.vertices), list(graph2.vertices)
        self.assertFalse(graph1.is_isomorphic(graph2, save_order=True))
        self.assertEqual(graph1.vertices, order1)
        self.assertEqual(graph2.vertices, order2)
        self.assert_labels_unset(graph1)
        self.assert_labels_unset(graph2)

    def test_self_comparison_leaves_labels_unset(self):
        """A self-comparison save_order call keeps the order and leaves unset labels unset."""
        graph = make_chain(4)
        order = list(graph.vertices)
        self.assertTrue(graph.is_isomorphic(graph, save_order=True))
        self.assertEqual(graph.vertices, order)
        self.assert_labels_unset(graph)

    def test_next_call_sorts_after_a_size_mismatch(self):
        """A plain call right after a size-mismatch save_order call on the same graph sorts it."""
        graph1, graph2, other = make_chain(4), make_chain(5), make_chain(4)
        self.assertFalse(graph1.is_isomorphic(graph2, save_order=True))
        self.assertTrue(graph1.is_isomorphic(other))
        self.assertEqual([vertex.sorting_label for vertex in graph1.vertices], list(range(4)))

    def test_next_call_sorts_after_a_self_comparison(self):
        """A plain call right after a self-comparison save_order call on the same graph sorts it."""
        graph, other = make_chain(4), make_chain(4)
        self.assertTrue(graph.is_isomorphic(graph, save_order=True))
        self.assertTrue(graph.is_isomorphic(other))
        self.assertEqual([vertex.sorting_label for vertex in graph.vertices], list(range(4)))

    def test_next_search_matches_a_fresh_search(self):
        """After early-exit save_order calls on both operands, find_isomorphism finds the maps in the fresh order."""
        fresh_a, fresh_b = shuffled_butane(1), shuffled_butane(2)
        expected = map_indices(fresh_a.find_isomorphism(fresh_b, save_order=True), fresh_a, fresh_b)
        for name, before in (
                ('size mismatch', lambda mol: mol.is_isomorphic(ethane_as(mol), save_order=True)),
                ('self-comparison', lambda mol: mol.is_isomorphic(mol, save_order=True))):
            mol_a, mol_b = shuffled_butane(1), shuffled_butane(2)
            before(mol_a)
            before(mol_b)
            found = map_indices(mol_a.find_isomorphism(mol_b, save_order=True), mol_a, mol_b)
            self.assertEqual(found, expected, msg=name)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
