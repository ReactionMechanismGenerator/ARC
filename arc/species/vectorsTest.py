#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.species.vectors module
"""

import math
import unittest

import arc.species.converter as converter
import arc.species.vectors as vectors
from arc.species.species import ARCSpecies


class TestVectors(unittest.TestCase):
    """
    Contains unit tests for the vectors module
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None

    def test_get_normal(self):
        """Test calculating a normal vector"""
        v1 = [6, 0, 0]
        v2 = [0, 3, 0]
        n = vectors.get_normal(v1, v2)
        self.assertEqual(n[0], 0)
        self.assertEqual(n[1], 0)
        self.assertEqual(n[2], 1)

        v1 = [5, 1, 1]
        v2 = [1, 8, 2]
        n = vectors.get_normal(v1, v2)
        expected_n = vectors.unit_vector([-6, -9, 39])
        for ni, expected_ni in zip(n, expected_n):
            self.assertEqual(ni, expected_ni)

    def test_get_theta(self):
        """Test calculating the angle between two vectors"""
        v1 = [-1.45707856 + 0.02416711, -0.94104506 - 0.17703194, -0.20275830 - 0.08644641]
        v2 = [-0.03480906 + 0.02416711, 1.11948179 - 0.17703194, -0.82988874 - 0.08644641]
        theta = vectors.get_theta(v1, v2)
        self.assertAlmostEqual(theta, 1.8962295, 5)
        self.assertAlmostEqual(theta * 180 / math.pi, 108.6459, 3)

    def test_unit_vector(self):
        """Test calculating a unit vector"""
        v1 = [1, 0, 0]
        self.assertEqual(vectors.unit_vector(v1)[0], 1.)  # trivial
        self.assertEqual(vectors.unit_vector(v1)[1], 0.)  # trivial
        self.assertEqual(vectors.unit_vector(v1)[2], 0.)  # trivial
        v2 = [1, 1, 1]
        self.assertAlmostEqual(vectors.unit_vector(v2)[0], (1 / 3) ** 0.5)
        self.assertAlmostEqual(vectors.unit_vector(v2)[1], (1 / 3) ** 0.5)
        self.assertAlmostEqual(vectors.unit_vector(v2)[2], (1 / 3) ** 0.5)

    def test_set_vector_length(self):
        """Test changing a vector's length"""
        v1 = [1, 0, 0]
        self.assertEqual(vectors.get_vector_length(v1), 1)
        v1_transformed = vectors.set_vector_length(v1, 5)
        self.assertAlmostEqual(vectors.get_vector_length(v1_transformed), 5)

        v1 = [1, 1, 1]
        self.assertEqual(vectors.get_vector_length(v1), 3 ** 0.5)
        v1_transformed = vectors.set_vector_length(v1, 5)
        self.assertAlmostEqual(vectors.get_vector_length(v1_transformed), 5)

        label = 'CNCC'
        pivot = 0
        xyz = {'symbols': ('N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
               'isotopes': (14, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1),
               'coords': ((0.7419854952964929, -0.18588322859265055, -0.8060295165375653),
                          (-0.38476668007897186, -0.8774643553523614, -0.1815530887172187),
                          (-1.4977348513273125, 0.05995564693605262, 0.26652181022311233),
                          (1.5633235727172392, 0.5360966415350092, 0.15477859056711452),
                          (-0.04458112063725271, -1.4936027355391557, 0.6589418973690523),
                          (-0.7986335015469359, -1.5715787743431335, -0.9219907626214912),
                          (-2.348455608682208, -0.5210498432021002, 0.6375394558854425),
                          (-1.8523669868240424, 0.6790455638159553, -0.5642494434208211),
                          (-1.170505453235269, 0.7210016856743618, 1.0746899133307615),
                          (2.4283037770451084, 0.9651590522064675, -0.36083882142892065),
                          (1.945994527876002, -0.1322800197070601, 0.9328203647772167),
                          (1.0178974719106297, 1.3595978302624294, 0.6250164549219148),
                          (0.39953935748654607, 0.4610025363062083, -1.5156468543485933))}
        mol = ARCSpecies(label='CNCC', xyz=xyz).mol
        v1 = vectors.get_lp_vector(label, mol, xyz, pivot)
        self.assertAlmostEqual(vectors.get_vector_length(v1), 1)  # should return a unit vector
        v1_transformed = vectors.set_vector_length(v1, 5)
        self.assertAlmostEqual(vectors.get_vector_length(v1_transformed), 5)

    def test_rotate_vector(self):
        """Test rotating a vector"""
        point_a, point_b, normal, theta = [0, 0, 0], [0, 0, 1], [0, 0, 1], 90.0 * math.pi / 180  # trivial, no rotation
        new_vector = vectors.rotate_vector(point_a, point_b, normal, theta)
        self.assertEqual(new_vector, [0, 0, 1])

        point_a, point_b, normal, theta = [0, 0, 0], [1, 0, 0], [0, 0, 1], 90.0 * math.pi / 180  # rot x to y around z
        new_vector = vectors.rotate_vector(point_a, point_b, normal, theta)
        self.assertAlmostEqual(new_vector[0], 0, 5)
        self.assertAlmostEqual(new_vector[1], 1, 5)
        self.assertAlmostEqual(new_vector[2], 0, 5)

        point_a, point_b, normal, theta = [0, 0, 0], [3, 5, 0], [4, 4, 1], 1.2
        new_vector = vectors.rotate_vector(point_a, point_b, normal, theta)
        self.assertAlmostEqual(new_vector[0], 2.749116, 5)
        self.assertAlmostEqual(new_vector[1], 4.771809, 5)
        self.assertAlmostEqual(new_vector[2], 1.916297, 5)

    def test_get_vector(self):
        """Test getting a vector between two atoms in the molecule"""
        xyz1 = converter.str_to_xyz("""O      0.0   0.0    0.0
N      1.0    0.0   0.0""")  # trivial
        vector = vectors.get_vector(pivot=0, anchor=1, xyz=xyz1)
        self.assertAlmostEqual(vector[0], 1.0, 5)
        self.assertAlmostEqual(vector[1], 0.0, 5)
        self.assertAlmostEqual(vector[2], 0.0, 5)

        xyz2 = converter.str_to_xyz("""O      -0.39141517   -1.49218505    0.23537907
N      -1.29594218    0.36660772   -0.33360920
C      -0.24369399   -0.21522785    0.47237314
C       1.11876670    0.24246665   -0.06138419
H      -0.34055624    0.19728442    1.48423848
H       1.27917500   -0.02124533   -1.11576163
H       1.93896021   -0.20110894    0.51754953
H       1.21599040    1.33219465    0.01900272
H      -2.12405283   -0.11420423    0.01492411
H      -1.15723190   -0.09458204   -1.23271202""")  # smiles='NC([O])(C)'
        vector = vectors.get_vector(pivot=1, anchor=2, xyz=xyz2)
        self.assertAlmostEqual(vector[0], 1.052248, 5)
        self.assertAlmostEqual(vector[1], -0.581836, 5)
        self.assertAlmostEqual(vector[2], 0.805982, 5)

    def test_get_lp_vector(self):
        """Test the lone pair vector"""
        xyz1 = converter.str_to_xyz("""O       1.13971727   -0.35763357   -0.91809799
N      -0.16022228   -0.63832421   -0.32863338
C      -0.42909096    0.49864538    0.54457751
H      -1.36471297    0.33135829    1.08632108
H       0.37059419    0.63632068    1.27966893
H      -0.53867601    1.41749835   -0.03987146
H       0.03832076   -1.45968957    0.24914206
H       0.94407000   -0.42817536   -1.87310674""")
        spc1 = ARCSpecies(label='tst1', smiles='CN(O)', xyz=xyz1)
        vector = vectors.get_lp_vector(label='tst1', mol=spc1.mol, xyz=xyz1, pivot=1)
        self.assertAlmostEqual(vector[0], -0.7582151013592212, 5)
        self.assertAlmostEqual(vector[1], -0.14276808320949216, 5)
        self.assertAlmostEqual(vector[2], -0.6361816835523585, 5)
        self.assertAlmostEqual((sum([vi ** 2 for vi in vector])) ** 0.5, 1)
        # puts the following dummy atom in xyz1: 'Cl     -0.91844 -0.78109 -0.96482'

        xyz2 = converter.str_to_xyz("""N      -0.70735114    0.81971647    0.24999886
C       0.58016992    0.65919122   -0.42405305
C       1.44721132   -0.43727777    0.17945348
C      -1.63900905   -0.25796649   -0.04936095
H       1.11974047    1.60931343   -0.33768790
H       0.43764604    0.48458543   -1.49689220
H       1.00255021   -1.42757899    0.04242741
H       2.42947502   -0.44523307   -0.30432399
H       1.60341053   -0.27376799    1.25093890
H      -1.81252045   -0.34624671   -1.12667881
H      -2.60396918   -0.04100469    0.41960198
H      -1.29274859   -1.22036999    0.33877281
H      -0.56460509    0.87663914    1.25780346""")
        spc2 = ARCSpecies(label='tst2', smiles='CNCC', xyz=xyz2)
        vector = vectors.get_lp_vector(label='tst2', mol=spc2.mol, xyz=xyz2, pivot=0)
        self.assertAlmostEqual(vector[0], -0.40585301456248446, 5)
        self.assertAlmostEqual(vector[1], 0.8470158636326891, 5)
        self.assertAlmostEqual(vector[2], -0.34328917449449764, 5)
        self.assertAlmostEqual((sum([vi ** 2 for vi in vector])) ** 0.5, 1)
        # puts the following dummy atom in xyz1: 'Cl     -1.1132 1.666732 -0.09329'

    def test_get_vector_length(self):
        """Test getting a vector's length"""
        
        # unit vector
        v1 = [1, 0, 0]
        self.assertEqual(vectors.get_vector_length(v1), 1)
        
        # a vector with 0 entries
        v1 = [0, 0, 0]
        self.assertEqual(vectors.get_vector_length(v1), 0)
        
        # 10D vector
        v1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertAlmostEqual(vectors.get_vector_length(v1), 19.6214169)   # default: places=7
        
        # 1D vector
        v1 = [2]
        self.assertEqual(vectors.get_vector_length(v1), 2)
        
        # a vector with small entries
        v1 = [0.0000001, 0.0000002, 0.0000003]
        self.assertAlmostEqual(vectors.get_vector_length(v1), 0.000000374165739, places=15)
        
        # a vector with large entries
        v1 = [100, 200, 300]
        self.assertAlmostEqual(vectors.get_vector_length(v1), 374.165738677394000)  # default: places=7
        
        # a real example borrowed from test_set_vector_length
        label = 'CNCC'
        pivot = 0
        xyz = {'symbols': ('N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
               'isotopes': (14, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1),
               'coords': ((0.7419854952964929, -0.18588322859265055, -0.8060295165375653),
                          (-0.38476668007897186, -0.8774643553523614, -0.1815530887172187),
                          (-1.4977348513273125, 0.05995564693605262, 0.26652181022311233),
                          (1.5633235727172392, 0.5360966415350092, 0.15477859056711452),
                          (-0.04458112063725271, -1.4936027355391557, 0.6589418973690523),
                          (-0.7986335015469359, -1.5715787743431335, -0.9219907626214912),
                          (-2.348455608682208, -0.5210498432021002, 0.6375394558854425),
                          (-1.8523669868240424, 0.6790455638159553, -0.5642494434208211),
                          (-1.170505453235269, 0.7210016856743618, 1.0746899133307615),
                          (2.4283037770451084, 0.9651590522064675, -0.36083882142892065),
                          (1.945994527876002, -0.1322800197070601, 0.9328203647772167),
                          (1.0178974719106297, 1.3595978302624294, 0.6250164549219148),
                          (0.39953935748654607, 0.4610025363062083, -1.5156468543485933))}
        mol = ARCSpecies(label='CNCC', xyz=xyz).mol
        v1 = vectors.get_lp_vector(label, mol, xyz, pivot)
        # --> Returns a unit vector pointing from the pivotal (nitrogen) atom towards its lone electron pairs orbital.
        self.assertAlmostEqual(vectors.get_vector_length(v1), 1)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
