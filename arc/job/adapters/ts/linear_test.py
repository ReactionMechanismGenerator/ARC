#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.heuristics module
"""

import unittest
import shutil

from arc.common import ARC_PATH, _check_r_n_p_symbols_between_rmg_and_arc_rxns, almost_equal_coords
from arc.job.adapters.ts.linear import (LinearAdapter,
                                        average_zmat_params,
                                        get_rxn_weight,
                                        get_weight,
                                        interpolate_isomerization,
                                        )
from arc.reaction import ARCReaction
from arc.rmgdb import load_families_only, make_rmg_database_object
from arc.species.converter import str_to_xyz, zmat_to_xyz, xyz_to_str
from arc.species.species import ARCSpecies
from arc.species.zmat import _compare_zmats


class TestHeuristicsAdapter(unittest.TestCase):
    """
    Contains unit tests for the HeuristicsAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        # cls.rmgdb = make_rmg_database_object()
        # load_families_only(cls.rmgdb)

    def test_average_zmat_params(self):
        """Test the average_zmat_params() function."""
        zmat_1 = {'symbols': ('H', 'H'),
                  'coords': ((None, None, None),
                             ('R_1_0', None, None)),
                  'vars': {'R_1_0': 0.7},
                  'map': {0: 0, 1: 1}}
        zmat_2 = {'symbols': ('H', 'H'),
                  'coords': ((None, None, None),
                             ('R_1_0', None, None)),
                  'vars': {'R_1_0': 1.3},
                  'map': {0: 0, 1: 1}}
        expected_zmat = {'symbols': ('H', 'H'),
                         'coords': ((None, None, None),
                                    ('R_1_0', None, None)),
                         'vars': {'R_1_0': 1.0},
                         'map': {0: 0, 1: 1}}
        zmat = average_zmat_params(zmat_1, zmat_2)
        self.assertTrue(_compare_zmats(zmat, expected_zmat))

        expected_zmat = {'symbols': ('H', 'H'),
                         'coords': ((None, None, None),
                                    ('R_1_0', None, None)),
                         'vars': {'R_1_0': 0.85},
                         'map': {0: 0, 1: 1}}
        zmat = average_zmat_params(zmat_1, zmat_2, weight=0.25)
        self.assertTrue(_compare_zmats(zmat, expected_zmat))
        zmat = average_zmat_params(zmat_2, zmat_1, weight=0.75)
        self.assertTrue(_compare_zmats(zmat, expected_zmat))

        zmat_1 = {'symbols': ('C', 'N', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2|4_0|0', 'A_2|4_0|0_1|1', None),
                             ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_3_1_0_2'),
                             ('R_2|4_0|0', 'A_2|4_0|0_1|1', 'D_4_0_1_3'), ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'),
                             ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_6_1_0_5')),
                  'vars': {'R_1_0': 1.451965854148702, 'D_3_1_0_2': 60.83821034525936,
                           'D_4_0_1_3': 301.30263742432356, 'R_5_0': 1.0936965384360282,
                           'A_5_0_1': 110.59878027260544, 'D_5_0_1_4': 239.76779188408136,
                           'D_6_1_0_5': 65.17113681053117, 'R_2|4_0|0': 1.0935188594180785,
                           'R_3|6_1|1': 1.019169330302324, 'A_2|4_0|0_1|1': 110.20495980110817,
                           'A_3|6_1|1_0|0': 109.41187648524644},
                  'map': {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4, 6: 6}}
        zmat_2 = {'symbols': ('C', 'N', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2|4_0|0', 'A_2|4_0|0_1|1', None),
                             ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_3_1_0_2'),
                             ('R_2|4_0|0', 'A_2|4_0|0_1|1', 'D_4_0_1_3'), ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'),
                             ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_6_1_0_5')),
                  'vars': {'R_1_0': 1.2, 'D_3_1_0_2': 50,
                           'D_4_0_1_3': 250, 'R_5_0': 1.0936965384360282,
                           'A_5_0_1': 110.59878027260544, 'D_5_0_1_4': 239.76779188408136,
                           'D_6_1_0_5': 120, 'R_2|4_0|0': 1.0935188594180785,
                           'R_3|6_1|1': 1.6, 'A_2|4_0|0_1|1': 110.20495980110817,
                           'A_3|6_1|1_0|0': 109.41187648524644},
                  'map': {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4, 6: 6}}
        expected_zmat = {'symbols': ('C', 'N', 'H', 'H', 'H', 'H', 'H'),
                         'coords': ((None, None, None), ('R_1_0', None, None), ('R_2|4_0|0', 'A_2|4_0|0_1|1', None),
                                    ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_3_1_0_2'), ('R_2|4_0|0', 'A_2|4_0|0_1|1', 'D_4_0_1_3'),
                                    ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_6_1_0_5')),
                         'vars': {'R_1_0': 1.3259829270743508, 'D_3_1_0_2': 55.419105172629685,
                                  'D_4_0_1_3': 275.6513187121618, 'R_5_0': 1.0936965384360282,
                                  'A_5_0_1': 110.59878027260544, 'D_5_0_1_4': 239.76779188408136,
                                  'D_6_1_0_5': 92.58556840526558, 'R_2|4_0|0': 1.0935188594180785,
                                  'R_3|6_1|1': 1.309584665151162, 'A_2|4_0|0_1|1': 110.20495980110817,
                                  'A_3|6_1|1_0|0': 109.41187648524644},
                         'map': {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4, 6: 6}}
        zmat = average_zmat_params(zmat_1, zmat_2)
        self.assertTrue(_compare_zmats(zmat, expected_zmat))

    def test_get_weight(self):
        """Test the get_weight() function."""
        self.assertEqual(get_weight([0], [0], 4), 0.5)  # 4 / 8
        self.assertEqual(get_weight([0], [8], 12), 0.75)  # 12 / 20
        self.assertEqual(get_weight([0], [2], 6), 0.6)  # 6 / 10
        self.assertEqual(get_weight([10], [0], 30), 0.4)  # 20 / 50
        self.assertEqual(get_weight([20], [10], 40), 0.4)  # 20 / 50
        self.assertIsNone(get_weight([20], [None], 40), 0.4)  # 20 / 50
        self.assertEqual(get_weight([8, 2], [0], 30), 0.4)  # 20 / 50
        self.assertEqual(get_weight([4, 1], [5.5, 1.5], 11), 0.6)  # 6 / 10

    def test_get_rxn_weight(self):
        """Test the get_rxn_weight() function."""
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='HO2', smiles='[O]O'),
                                       ARCSpecies(label='NH', smiles='[NH]')],
                            p_species=[ARCSpecies(label='N', smiles='[N]'),
                                       ARCSpecies(label='H2O2', smiles='OO')])
        rxn_1.r_species[0].e0 = 252.0
        rxn_1.r_species[1].e0 = 100.5
        rxn_1.p_species[0].e0 = 116.0
        rxn_1.p_species[1].e0 = 200.3
        rxn_1.ts_species = ARCSpecies(label='TS', is_ts=True)
        rxn_1.ts_species.e0 = 391.6
        self.assertAlmostEquals(get_rxn_weight(rxn_1), 0.3417832)

    def test_interpolate_isomerization(self):
        """Test the interpolate_isomerization() function."""
        nc3h7_xyz = """C                  0.00375165   -0.48895802   -1.20586379
                       C                  0.00375165   -0.48895802    0.28487510
                       C                  0.00375165    0.91997987    0.85403684
                       H                  0.41748586   -1.33492098   -1.74315104
                       H                 -0.57506729    0.24145491   -1.76006154
                       H                 -0.87717095   -1.03203740    0.64280162
                       H                  0.88948616   -1.02465371    0.64296621
                       H                  0.88512433    1.48038223    0.52412379
                       H                  0.01450405    0.88584135    1.94817394
                       H                 -0.88837301    1.47376959    0.54233121"""
        ic3h7_xyz = """C                 -0.40735690   -0.74240205   -0.34312948
                       C                  0.38155377   -0.25604705    0.82450968
                       C                  0.54634593    1.25448345    0.81064511
                       H                  0.00637731   -1.58836501   -0.88041673
                       H                 -0.98617584   -0.01198912   -0.89732723
                       H                 -1.29710684   -1.29092340    0.08598983
                       H                  1.36955428   -0.72869684    0.81102246
                       H                  1.06044877    1.58846788   -0.09702437
                       H                  1.13774084    1.57830484    1.67308862
                       H                 -0.42424546    1.75989927    0.85794283"""
        nc3h7 = ARCSpecies(label='nC3H7', smiles='[CH2]CC', xyz=nc3h7_xyz)
        ic3h7 = ARCSpecies(label='iC3H7', smiles='C[CH]C', xyz=ic3h7_xyz)
        rxn = ARCReaction(r_species=[nc3h7], p_species=[ic3h7])
        expected_ts_xyz = str_to_xyz("""C       0.01099731   -0.46789926   -1.15958911
                                        C       0.01099731   -0.46789926    0.33114978
                                        C       0.01099731    0.94103865    0.90031155
                                        H       0.57795661   -1.24174248   -1.65467180
                                        H      -0.39690222    0.34527841   -1.69240298
                                        H      -1.19440431   -1.28933062   -0.47327539
                                        H       0.89689057   -1.16420498    0.45967951
                                        H       0.76979130    1.33747945    0.33815513
                                        H      -0.04544494    0.70455273    1.77835334
                                        H      -1.00071642    1.24557408    0.38839197""")
        ts_xyz = interpolate_isomerization(rxn, use_weights=False)
        self.assertTrue(almost_equal_coords(ts_xyz, expected_ts_xyz))

        nc3h7.e0 = 101.55
        ic3h7.e0 = 88.91
        ts = ARCSpecies(label='TS', is_ts=True, multiplicity=2, xyz=expected_ts_xyz)
        ts.e0 = 105
        rxn.ts_species = ts
        expected_ts_xyz = str_to_xyz("""C       0.01224420   -0.47400672   -1.18787451
                                        C       0.01224420   -0.47400672    0.30286438
                                        C       0.01224420    0.93493122    0.87202615
                                        H       0.47981756   -1.29923732   -1.70742021
                                        H      -0.50470551    0.28201158   -1.73526026
                                        H      -1.06475721   -1.18141451    0.26785378
                                        H       0.86736552   -1.12118386    0.54383845
                                        H       0.79813573    1.38347069    0.43772483
                                        H      -0.03897336    0.76031233    1.86961141
                                        H      -0.97425159    1.33180895    0.47825005""")
        ts_xyz = interpolate_isomerization(rxn, use_weights=True)
        self.assertTrue(almost_equal_coords(ts_xyz, expected_ts_xyz))

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        pass


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
