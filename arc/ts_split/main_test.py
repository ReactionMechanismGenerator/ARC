#!/usr/bin/env python3
# encoding: utf-8


import unittest

from arc.imports import settings
import arc.ts_split.split as split
from arc.species.converter import check_xyz_dict, xyz_to_dmat

servers = settings['servers']


class TestTSSplit(unittest.TestCase):
    """
    Contains unit tests for ARC's split module
    """

    def test_bonded(self):
        """Test bonded"""
        self.assertTrue(split.bonded(1.0, 'C', 'C'))
        self.assertTrue(split.bonded(1.0, 'C', 'H'))
        self.assertTrue(split.bonded(1.0, 'H', 'C'))
        self.assertTrue(split.bonded(1.0, 'H', 'H'))
        self.assertFalse(split.bonded(1.7, 'C', 'C'))
        self.assertFalse(split.bonded(1.5, 'C', 'H'))
        self.assertFalse(split.bonded(1.5, 'H', 'C'))
        self.assertFalse(split.bonded(1.5, 'H', 'H'))

    def test_get_adjlist_from_dmat(self):
        """Test get_adjlist_from_dmat"""
        symbols = ('O', 'N', 'C', 'H', 'H', 'S', 'H')
        d = [[0., 1.44678738, 2.1572649, 3.07926623, 2.69780089, 1.74022888, 1.95867823],
             [1.44678738, 0., 1.47078693, 2.16662322, 2.12283495, 2.34209263, 1.02337844],
             [2.1572649, 1.47078693, 0., 1.09133324, 1.09169397, 1.82651322, 2.02956962],
             [3.07926623, 2.16662322, 1.09133324, 0., 1.80097071, 2.51409166, 2.30585633],
             [2.69780089, 2.12283495, 1.09169397, 1.80097071, 0., 2.45124337, 2.92889793],
             [1.74022888, 2.34209263, 1.82651322, 2.51409166, 2.45124337, 0., 2.68310024],
             [1.95867823, 1.02337844, 2.02956962, 2.30585633, 2.92889793, 2.68310024, 0.]]
        adjlist = split.get_adjlist_from_dmat(dmat=d, symbols=symbols, h=3, a=2, b=5)  # b is incorrect chemically
        self.assertEqual(adjlist, {0: [1, 5], 1: [0, 2, 6], 2: [1, 4, 5, 3], 4: [2], 5: [0, 2, 3], 6: [1], 3: [2, 5]})

        xyz = """ C                 -3.80799396    1.05904061    0.12143410
                  H                 -3.75776386    0.09672979   -0.34366835
                  H                 -3.24934849    1.76454448   -0.45741718
                  H                 -4.82886508    1.37420677    0.17961125
                  C                 -3.21502590    0.97505234    1.54021348
                  H                 -3.26525696    1.93736874    2.00531040
                  H                 -3.77366533    0.26954471    2.11907272
                  C                 -1.74572494    0.52144864    1.45646938
                  H                 -1.18708880    1.22694232    0.87759054
                  H                 -1.69550074   -0.44087971    0.99139265
                  O                 -0.57307243    0.35560699    4.26172088
                  H                 -1.12770789    0.43395779    2.93512192
                  O                  0.45489302    1.17807207    4.35811043
                  H                  1.12427554    0.93029226    3.71613651"""
        xyz_dict = check_xyz_dict(xyz)
        dmat = xyz_to_dmat(xyz_dict)
        adjlist = split.get_adjlist_from_dmat(dmat=dmat, symbols=xyz_dict['symbols'], h=11, a=7, b=10)
        self.assertEqual(adjlist,
                         {0: [1, 2, 3, 4],
                          1: [0],
                          2: [0],
                          3: [0],
                          4: [0, 5, 6, 7],
                          5: [4],
                          6: [4],
                          7: [4, 8, 9, 11],
                          8: [7],
                          9: [7],
                          10: [12, 11],
                          12: [10, 13],
                          13: [12],
                          11: [7, 10]})

    def test_iterative_dfs(self):
        """Test iterative_dfs"""
        adjlist = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0, 5, 6, 7], 5: [4], 6: [4], 7: [4, 8, 9, 11], 8: [7],
                   9: [7], 10: [12, 11], 12: [10, 13], 13: [12], 11: [7, 10]}
        g1 = split.iterative_dfs(adjlist=adjlist, start=10, border=11)
        self.assertEqual(g1, [11, 10, 12, 13])
        g2 = split.iterative_dfs(adjlist=adjlist, start=7, border=11)
        self.assertEqual(g2, [11, 7, 9, 8, 4, 6, 5, 0, 3, 2, 1])


    def test_get_group_xyzs_and_key_indices_from_ts_xyz(self):
        """Test get_group_xyzs_and_key_indices_from_ts_xyz"""
        xyz = """ C                 -3.80799396    1.05904061    0.12143410
                  H                 -3.75776386    0.09672979   -0.34366835
                  H                 -3.24934849    1.76454448   -0.45741718
                  H                 -4.82886508    1.37420677    0.17961125
                  C                 -3.21502590    0.97505234    1.54021348
                  H                 -3.26525696    1.93736874    2.00531040
                  H                 -3.77366533    0.26954471    2.11907272
                  C                 -1.74572494    0.52144864    1.45646938
                  H                 -1.18708880    1.22694232    0.87759054
                  H                 -1.69550074   -0.44087971    0.99139265
                  O                 -0.57307243    0.35560699    4.26172088
                  H                 -1.12770789    0.43395779    2.93512192
                  O                  0.45489302    1.17807207    4.35811043
                  H                  1.12427554    0.93029226    3.71613651"""
        a, b, h = 7, 10, 11
        xyz_dict = check_xyz_dict(xyz)
        g1_xyz, g2_xyz, index_dict = split.get_group_xyzs_and_key_indices_from_ts(xyz=xyz_dict, a=a, b=b, h=h)
        print(f'g1_xyz: {g1_xyz}')
        print(f'g2_xyz: {g2_xyz}')
        print(f'index_dict: {index_dict}')
        raise


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))









