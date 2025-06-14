#!/usr/bin/env python3
# encoding: utf-8


import unittest
from scipy.special import comb

from arc.molecule.util import get_element_count, agglomerate, generate_combo, partition, swap


class ElementCountTest(unittest.TestCase):

    def test_inchi_count(self):
        """Test element counting for InChI"""
        inchi = 'InChI=1S/C4H10O/c1-2-3-4-5/h5H,2-4H2,1H3'

        expected = {'C': 4, 'H': 10, 'O': 1}

        count = get_element_count(inchi)

        self.assertEqual(count, expected)

    def test_inchi_count_disconnected(self):
        """Test element counting for InChI with a disconnected molecule"""
        inchi = 'InChI=1S/C4H10O.CH2O/c1-2-3-4-5;1-2/h5H,2-4H2,1H3;1H2'

        expected = {'C': 5, 'H': 12, 'O': 2}

        count = get_element_count(inchi)

        self.assertEqual(count, expected)


class PartitionTest(unittest.TestCase):

    def test_singleton(self):
        """
        Test that a index not part of the parameter list, results in a key-value pair with
        an empty list.
        """
        indices = [7]
        list_of_samples = [[1, 2, 3, 4], [5, 6]]
        expected_partitions, expected_sample_lists = [[7]], [[]]

        partitions, sample_lists = partition(indices, list_of_samples)

        self.assertEqual(partitions, expected_partitions)
        self.assertEqual(sample_lists, expected_sample_lists)

    def test_2_elements_in_1_layer(self):
        indices = [1, 3]
        list_of_samples = [[1, 2, 3, 4], [5, 6]]
        expected_partitions, expected_sample_lists = [[1, 3]], [[1, 2, 3, 4]]

        partitions, sample_lists = partition(indices, list_of_samples)

        self.assertEqual(partitions, expected_partitions)
        self.assertEqual(sample_lists, expected_sample_lists)

    def test_2_elements_in_2_layers(self):
        indices = [1, 5]
        list_of_samples = [[1, 2, 3, 4], [5, 6]]
        expected_partitions, expected_sample_lists = [[1], [5]], [[1, 2, 3, 4], [5, 6]]

        partitions, sample_lists = partition(indices, list_of_samples)

        self.assertEqual(partitions, expected_partitions)
        self.assertEqual(sample_lists, expected_sample_lists)

    def test_3_elements_in_2_layers(self):
        indices = [1, 4, 5]
        list_of_samples = [[1, 2, 3, 4], [5, 6]]
        expected_partitions, expected_sample_lists = [[1, 4], [5]], [[1, 2, 3, 4], [5, 6]]

        partitions, sample_lists = partition(indices, list_of_samples)

        self.assertEqual(partitions, expected_partitions)
        self.assertEqual(sample_lists, expected_sample_lists)

    def test_3_elements_in_2_layers_1_singleton(self):
        indices = [1, 5, 7]
        list_of_samples = [[1, 2, 3, 4], [5, 6]]
        expected_partitions, expected_sample_lists = [[1], [5], [7]], [[1, 2, 3, 4], [5, 6], []]

        partitions, sample_lists = partition(indices, list_of_samples)

        self.assertEqual(partitions, expected_partitions)
        self.assertEqual(sample_lists, expected_sample_lists)


class AgglomerateTest(unittest.TestCase):

    def test_normal(self):
        groups = [[1, 2, 3], [4], [5, 6], [7]]
        agglomerates = agglomerate(groups)
        expected = [[1, 2, 3], [5, 6], [4, 7]]

        self.assertEqual(agglomerates, expected)


class ComboGeneratorTest(unittest.TestCase):
    def test_2_elements(self):
        samples = [[1, 2, 3], [6]]
        sample_spaces = [[1, 2, 3, 4], [5, 6]]

        combos = generate_combo(samples, sample_spaces)

        expected = 1
        for sample, sample_space in zip(samples, sample_spaces):
            expected *= comb(len(sample_space), len(sample), exact=True)

        # we leave out the original combination
        expected -= 1

        self.assertEqual(len(combos), expected)


class SwapTest(unittest.TestCase):

    def test_2_elements_sets(self):
        to_be_swapped = [2, 3]
        sample = [1, 3]

        result = swap(to_be_swapped, sample)
        expected = (1, 3, 2)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
