#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.heuristics module
"""

import os
import shutil
import unittest

from rmgpy.data.kinetics import KineticsFamily

from arc.common import ARC_PATH, almost_equal_coords
from arc.job.adapters.ts.linear import (LinearAdapter,
                                        average_zmat_params,
                                        get_r_constraints,
                                        get_rxn_weight,
                                        get_weight,
                                        interpolate_isomerization,
                                        )
from arc.reaction import ARCReaction
from arc.rmgdb import make_rmg_database_object, load_families_only
from arc.species.converter import str_to_xyz, xyz_to_str
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
        cls.rmgdb = make_rmg_database_object()
        load_families_only(cls.rmgdb)

        cls.rxn_1 = ARCReaction(r_species=[ARCSpecies(label='CPD', smiles='C1C=CC=C1',
                                                      xyz="""C      -1.11689933   -0.16076292   -0.17157587
                                                             C      -0.34122713    1.12302797   -0.12498608
                                                             C       0.95393962    0.86179733    0.10168911
                                                             C       1.14045506   -0.56033684    0.22004768
                                                             C      -0.03946631   -1.17782376    0.06650470
                                                             H      -1.58827673   -0.30386166   -1.14815401
                                                             H      -1.87502410   -0.19463481    0.61612857
                                                             H      -0.77193310    2.10401684   -0.25572143
                                                             H       1.74801386    1.58807889    0.18578522
                                                             H       2.09208098   -1.03534789    0.40412258
                                                             H      -0.20166282   -2.24415315    0.10615953""")],
                                p_species=[ARCSpecies(label='C5_carbene', adjlist="""1  C u0 p1 c0 {2,S} {6,S}
                                                                                     2  C u0 p0 c0 {1,S} {3,D} {7,S}
                                                                                     3  C u0 p0 c0 {2,D} {4,S} {8,S}
                                                                                     4  C u0 p0 c0 {3,S} {5,D} {9,S}
                                                                                     5  C u0 p0 c0 {4,D} {10,S} {11,S}
                                                                                     6  H u0 p0 c0 {1,S}
                                                                                     7  H u0 p0 c0 {2,S}
                                                                                     8  H u0 p0 c0 {3,S}
                                                                                     9  H u0 p0 c0 {4,S}
                                                                                     10 H u0 p0 c0 {5,S}
                                                                                     11 H u0 p0 c0 {5,S}""",
                                                      xyz="""C       2.62023459    0.49362130   -0.23013873
                                                             C       1.48006570   -0.33866786   -0.38699247
                                                             C       1.53457595   -1.45115429   -1.13132450
                                                             C       0.40179762   -2.32741928   -1.31937443
                                                             C       0.45595744   -3.43865596   -2.06277224
                                                             H       3.47507694    1.11901971   -0.11163109
                                                             H       0.56454036   -0.04212124    0.11659958
                                                             H       2.46516705   -1.72493574   -1.62516589
                                                             H      -0.53390611   -2.06386676   -0.83047533
                                                             H      -0.42088759   -4.06846526   -2.17670487
                                                             H       1.36205133   -3.75009763   -2.57288841""")])
        cls.rxn_1.determine_family(rmg_database=cls.rmgdb)

        cls.rxn_2 = ARCReaction(r_species=[ARCSpecies(label='CCONO', smiles='CCON=O',
                                                      xyz="""C      -1.36894499    0.07118059   -0.24801399
                                                             C      -0.01369535    0.17184136    0.42591278
                                                             O      -0.03967083   -0.62462610    1.60609048
                                                             N       1.23538512   -0.53558048    2.24863846
                                                             O       1.25629155   -1.21389295    3.27993827
                                                             H      -2.16063255    0.41812452    0.42429392
                                                             H      -1.39509985    0.66980796   -1.16284741
                                                             H      -1.59800183   -0.96960842   -0.49986392
                                                             H       0.19191326    1.21800574    0.68271847
                                                             H       0.76371340   -0.19234475   -0.25650067""")],
                                p_species=[ARCSpecies(label='CCNO2', smiles='CC[N+](=O)[O-]',
                                                      xyz="""C      -1.12362739   -0.04664655   -0.08575959
                                                             C       0.24488022   -0.51587553    0.36119196
                                                             N       0.57726975   -1.77875156   -0.37104243
                                                             O       1.16476543   -1.66382529   -1.45384186
                                                             O       0.24561669   -2.84385320    0.16410116
                                                             H      -1.87655344   -0.80826847    0.13962125
                                                             H      -1.14729169    0.14493421   -1.16405294
                                                             H      -1.41423043    0.87863077    0.42354512
                                                             H       1.02430791    0.21530309    0.12674144
                                                             H       0.27058353   -0.73979548    1.43184405""")])
        cls.rxn_2.determine_family(rmg_database=cls.rmgdb)

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

    def test_interpolate_isomerization_intra_h_migration(self):
        """Test the interpolate_isomerization() function for intra H migration reactions."""
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
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        self.assertEqual(len(ts_xyzs), 2)
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))

        nc3h7.e0 = 101.55
        ic3h7.e0 = 88.91
        ts = ARCSpecies(label='TS', is_ts=True, multiplicity=2, xyz=expected_ts_xyz)
        ts.e0 = 105
        rxn.ts_species = ts
        expected_ts_xyz = str_to_xyz("""C       0.00591772   -0.48764618   -1.20069282
                                        C       0.00591772   -0.48764618    0.29004607
                                        C       0.00591772    0.92129176    0.85920784
                                        H       0.47693974   -1.30982443   -1.72763512
                                        H      -0.52424330    0.28530048   -1.74580953
                                        H      -1.07348606   -1.15308709    0.40763997
                                        H       0.89165221   -1.02334186    0.64813718
                                        H       0.88729039    1.48169408    0.52929476
                                        H       0.01667012    0.88715326    1.95334482
                                        H      -0.88620694    1.47508143    0.54750218""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=True)
        self.assertEqual(len(ts_xyzs), 2)
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))

        # r_xyz = """C      -1.05582103   -0.03329574   -0.10080257
        #            C       0.41792695    0.17831205    0.21035514
        #            O       1.19234020   -0.65389683   -0.61111443
        #            O       2.44749684   -0.41401220   -0.28381363
        #            H      -1.33614002   -1.09151783    0.08714882
        #            H      -1.25953618    0.21489046   -1.16411897
        #            H      -1.67410396    0.62341419    0.54699514
        #            H       0.59566350   -0.06437686    1.28256640
        #            H       0.67254676    1.24676329    0.02676370"""
        # p_xyz = """C      -1.40886397    0.22567351   -0.37379668
        #            C       0.06280787    0.04097694   -0.38515682
        #            O       0.44130326   -0.57668419    0.84260864
        #            O       1.89519755   -0.66754203    0.80966180
        #            H      -1.87218376    0.90693511   -1.07582340
        #            H      -2.03646287   -0.44342165    0.20255768
        #            H       0.35571681   -0.60165457   -1.22096147
        #            H       0.56095122    1.01161503   -0.47393734
        #            H       2.05354047   -0.10415729    1.58865243"""
        # r = ARCSpecies(label='R', smiles='CCO[O]', xyz=r_xyz)
        # p = ARCSpecies(label='P', smiles='[CH2]COO', xyz=p_xyz)
        # rxn = ARCReaction(r_species=[r], p_species=[p])
        # expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
        #                                 C       0.00598652   -0.48762088    0.30473835
        #                                 C       0.00598652    0.92131703    0.87390011
        #                                 H       0.57807817   -1.25594905   -1.69382911
        #                                 H      -0.42698663    0.35443110   -1.71434916
        #                                 H      -1.27461406   -1.27709743   -0.24121083
        #                                 H       0.89172104   -1.02331658    0.66282944
        #                                 H       0.88735917    1.48171935    0.54398704
        #                                 H       0.01673891    0.88717852    1.96803717
        #                                 H      -0.88613815    1.47510670    0.56219446""")
        # ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        # for ts_xyz in ts_xyzs:
        #     print(f'\nTS xyz:\n\n')
        #     print(xyz_to_str(ts_xyz))
        # self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))

    # def test_interpolate_isomerization_1_2_shift_C(self):
    #     """Test the interpolate_isomerization() function for 1,2_shift_C reactions."""
    #     r_xyz = """C      -2.08011725   -0.87098529   -0.24102896
    #                C      -1.38616808    0.31243567    0.41701874
    #                C       0.09289885    0.19281646    0.35695343
    #                C       0.92864438    0.68782411   -0.70819340
    #                C       2.18636908    0.35957487   -0.37255721
    #                C       2.18107732   -0.34427638    0.89885251
    #                C       0.92008966   -0.45002583    1.34718072
    #                H      -1.81540032   -1.81207279    0.25290661
    #                H      -1.80896484   -0.95285941   -1.29907735
    #                H      -3.16674193   -0.75248342   -0.17993048
    #                H      -1.70601347    1.23815887   -0.07595913
    #                H      -1.71241303    0.38717620    1.46117876
    #                H       0.59841756    1.21177196   -1.58944757
    #                H       3.07727387    0.57336525   -0.94230522
    #                H       3.06757417   -0.71677188    1.38813917
    #                H       0.58240767   -0.91755259    2.25688920"""
    #     p_xyz = """C      -0.91419261   -0.92211886    1.28775915
    #                C      -0.38593444   -0.06230282    0.18302891
    #                C       0.67135826    0.91653743    0.70043366
    #                C      -1.50477869    0.67123329   -0.52120219
    #                C      -1.56600546    0.29578439   -1.80779070
    #                C      -0.54194075   -0.68042899   -2.06986329
    #                C       0.15393453   -0.90959193   -0.94583904
    #                H      -1.87479029   -1.41555103    1.18169570
    #                H      -0.24773685   -1.28139411    2.06376191
    #                H       1.52757979    0.38651768    1.13544157
    #                H       1.05855879    1.55868444   -0.10049234
    #                H       0.25983545    1.57373963    1.47630068
    #                H      -2.15472843    1.39365373   -0.04968993
    #                H      -2.26617185    0.65598910   -2.54596855
    #                H      -0.37671673   -1.14563680   -3.02965012
    #                H       0.98207416   -1.59686236   -0.85449771"""
    #     r = ARCSpecies(label='R', smiles='CC[C]1C=CC=C1', xyz=r_xyz)
    #     p = ARCSpecies(label='P', smiles='[CH2]C1(C)C=CC=C1', xyz=p_xyz)
    #     rxn = ARCReaction(r_species=[r], p_species=[p])
    #     expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
    #                                     C       0.00598652   -0.48762088    0.30473835
    #                                     C       0.00598652    0.92131703    0.87390011
    #                                     H       0.57807817   -1.25594905   -1.69382911
    #                                     H      -0.42698663    0.35443110   -1.71434916
    #                                     H      -1.27461406   -1.27709743   -0.24121083
    #                                     H       0.89172104   -1.02331658    0.66282944
    #                                     H       0.88735917    1.48171935    0.54398704
    #                                     H       0.01673891    0.88717852    1.96803717
    #                                     H      -0.88613815    1.47510670    0.56219446""")
    #     ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
    #     # self.assertEqual(len(ts_xyzs), 3)
    #     for ts_xyz in ts_xyzs:
    #         print(f'\nTS xyz:\n\n')
    #         print(xyz_to_str(ts_xyz))
    #     self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # got no TS

    # def test_interpolate_isomerization_1_3_sigmatropic_rearrangement(self):
    #     """Test the interpolate_isomerization() function for 1,3_sigmatropic_rearrangement reactions."""
    #     r_xyz = """C      -0.96405208   -0.58870010   -0.35675666
    #                N       0.09948347   -1.35699528   -0.30406608
    #                C       1.08781769   -0.57088551    0.22943180
    #                C       0.61245126    0.68985747    0.50218591
    #                N      -0.70083129    0.66320502    0.12207481
    #                H      -1.93870511   -0.87854432   -0.72608823
    #                H       2.08729155   -0.95482079    0.38815067
    #                H       1.07812779    1.57128662    0.91862266
    #                H      -1.36158329    1.42559689    0.18141711"""
    #     p_xyz = """N       0.76582385   -0.14849540   -1.32485588
    #                C       0.78208226    0.49284271   -0.20399502
    #                N      -0.04861443    0.34490826    0.88039960
    #                C      -0.56227958   -0.84609375    1.31645778
    #                C      -1.38522743    0.06039446    0.80970400
    #                H       1.52092135    0.20130809   -1.92536405
    #                H       1.53681129    1.27833147   -0.02452505
    #                H      -0.33519514   -1.78256934    0.82247210
    #                H      -1.89445111   -0.06503499   -0.13767862"""
    #     r = ARCSpecies(label='R', smiles='c1ncc[nH]1', xyz=r_xyz)
    #     p = ARCSpecies(label='P', smiles='N=CN1C=C1', xyz=p_xyz)
    #     rxn = ARCReaction(r_species=[r], p_species=[p])
    #     expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
    #                                     C       0.00598652   -0.48762088    0.30473835
    #                                     C       0.00598652    0.92131703    0.87390011
    #                                     H       0.57807817   -1.25594905   -1.69382911
    #                                     H      -0.42698663    0.35443110   -1.71434916
    #                                     H      -1.27461406   -1.27709743   -0.24121083
    #                                     H       0.89172104   -1.02331658    0.66282944
    #                                     H       0.88735917    1.48171935    0.54398704
    #                                     H       0.01673891    0.88717852    1.96803717
    #                                     H      -0.88613815    1.47510670    0.56219446""")
    #     ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
    #     # self.assertEqual(len(ts_xyzs), 3)
    #     for ts_xyz in ts_xyzs:
    #         print(f'\nTS xyz:\n\n')
    #         print(xyz_to_str(ts_xyz))
    #     self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # got a wrong TS

    def test_interpolate_isomerization_6_membered_central_C_C_shift(self):
        """Test the interpolate_isomerization() function for 6_membered_central_C-C_shift reactions."""
        r_xyz = """C       3.03272979   -0.11060195   -0.24229461
                   C       1.85599055   -0.34675713   -0.20247149
                   C       0.41485966   -0.64142590   -0.15352412
                   C      -0.41485965    0.64142578   -0.17240633
                   C      -1.85599061    0.34675702   -0.12346178
                   C      -3.03272995    0.11060190   -0.08364096
                   H       4.07762286    0.09693448   -0.27758589
                   H       0.19106566   -1.21954180    0.75163518
                   H       0.14301783   -1.27648597   -1.00582442
                   H      -0.19106412    1.21954271   -1.07756459
                   H      -0.14301928    1.27648492    0.67989514
                   H      -4.07762310   -0.09693448   -0.04835177"""
        p_xyz = """C      -3.03124363    0.21595810   -0.01068883
                   C      -1.77136356   -0.00875193   -0.22839960
                   C      -0.51035344   -0.23538255   -0.44913569
                   C       0.51035356    0.23538291    0.44913621
                   C       1.77136365    0.00875234    0.22839985
                   C       3.03124358   -0.21595777    0.01068824
                   H      -3.50880107    1.10742857   -0.40051872
                   H      -3.62554573   -0.48341738    0.56587595
                   H      -0.21235801   -0.79338469   -1.33170668
                   H       0.21235823    0.79338484    1.33170737
                   H       3.50880076   -1.10742925    0.40051615
                   H       3.62554580    0.48341866   -0.56587535"""
        r = ARCSpecies(label='R', smiles='C#CCCC#C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C=C=CC=C=C', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # vectors.py:91: VectorsError

    def test_interpolate_isomerization_Concered_Intra_diels_alder_mono_cyclic_1_2_shiftH(self):
        """Test the interpolate_isomerization() function for Concered_Intra_diels_alder_mono-cyclic_1,2_shiftH reactions."""
        r_xyz = """C       3.25931086   -0.62469725   -0.85172041
                   C       2.21431341   -0.47587355   -0.28012458
                   C       0.97982054   -0.30103766    0.40159285
                   C      -0.13573006    0.04859177   -0.25399636
                   C      -1.40841692    0.23576534    0.40283655
                   C      -2.52147798    0.58481406   -0.25261333
                   H       4.18607967   -0.75650475   -1.35982959
                   H       0.97405553   -0.46156800    1.47539715
                   H      -0.10113962    0.20314181   -1.33112279
                   H      -1.45501699    0.08400757    1.47935301
                   H      -3.45759474    0.71479393    0.28157503
                   H      -2.53403295    0.74932608   -1.32550473"""
        p_xyz = """C       0.32377429   -1.34449694   -0.14122370
                   C       1.47572737   -0.75313342   -0.07853288
                   C       1.30166202    0.68051955    0.07213267
                   C      -0.03314415    1.28528351    0.13512766
                   C      -1.14328725    0.51323467    0.05345670
                   C      -1.10099711   -1.00983052   -0.10666836
                   H       2.43679364   -1.24237771   -0.12954979
                   H       2.17524509    1.31823771    0.13957240
                   H      -0.09926704    2.36242259    0.24835509
                   H      -2.12527706    0.97366262    0.10143239
                   H      -1.60600902   -1.48495894    0.73931727
                   H      -1.60522077   -1.29856310   -1.03341945"""
        r = ARCSpecies(label='R', smiles='C#CC=CC=C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[C]1C=CC=CC1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # got no TS

    def test_interpolate_isomerization_Cyclopentadiene_scission(self):
        """Test the interpolate_isomerization() function for Cyclopentadiene_scission reactions."""
        r_xyz = """C       1.46526410    0.35550989    0.15268159
                   C       0.45835467    1.13529910   -0.26555553
                   C      -0.75500438    0.35970165   -0.56989350
                   C      -1.48532781   -0.35660657    0.46119178
                   C      -0.34144780   -1.06077923   -0.11686057
                   C       0.98794173   -1.00683992    0.12489717
                   H       2.46308379    0.66299943    0.41975788
                   H       0.51108826    2.21009512   -0.37348204
                   H      -1.11928864    0.38428608   -1.58978132
                   H      -2.45322496   -0.77587088    0.21588387
                   H      -1.38590137   -0.05438209    1.49711542
                   H       1.65446241   -1.85341259    0.04404524"""
        p_xyz = """C       0.32377429   -1.34449694   -0.14122370
                   C       1.47572737   -0.75313342   -0.07853288
                   C       1.30166202    0.68051955    0.07213267
                   C      -0.03314415    1.28528351    0.13512766
                   C      -1.14328725    0.51323467    0.05345670
                   C      -1.10099711   -1.00983052   -0.10666836
                   H       2.43679364   -1.24237771   -0.12954979
                   H       2.17524509    1.31823771    0.13957240
                   H      -0.09926704    2.36242259    0.24835509
                   H      -2.12527706    0.97366262    0.10143239
                   H      -1.60600902   -1.48495894    0.73931727
                   H      -1.60522077   -1.29856310   -1.03341945"""
        r = ARCSpecies(label='R', smiles='C1=CC2CC2=C1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[C]1C=CC=CC1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # got no TS

    def test_interpolate_isomerization_intra_2_2_cycloaddition_Cd(self):
        """Test the interpolate_isomerization() function for intra_2+2_cycloaddition_Cd reactions."""
        r_xyz = """C       1.82756305    0.09282277   -0.10513133
                   C       0.58818449   -0.41022877   -0.07453536
                   C      -0.58817345    0.41021329    0.07471351
                   C      -1.82755191   -0.09284018    0.10530914
                   H       2.02221248    1.15701349   -0.01869217
                   H       2.68495777   -0.56301880   -0.21927041
                   H       0.45128207   -1.48580231   -0.16562548
                   H      -0.45127003    1.48578749    0.16580454
                   H      -2.02220285   -1.15703226    0.01886889
                   H      -2.68494614    0.56300277    0.21944957"""
        p_xyz = """C      -0.06902948    0.84060039    0.77804358
                   C      -1.09821853    0.32180183    0.08417575
                   C      -0.22613156   -0.65945413   -0.66625740
                   C       0.95345020   -0.06484505    0.12900378
                   H       0.01489202    1.62040295    1.52931163
                   H      -2.16551579    0.52129254    0.05930486
                   H      -0.41639366   -1.72015227   -0.46931593
                   H      -0.16106002   -0.51376156   -1.75004960
                   H       1.45633649   -0.77613764    0.79325853
                   H       1.71167034    0.43025295   -0.48747520"""
        r = ARCSpecies(label='R', smiles='C=CC=C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C1=CCC1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # got no TS

    def test_interpolate_isomerization_Intra_5_membered_conjugated_C_C_C_C_addition(self):
        """Test the interpolate_isomerization() function for Intra_5_membered_conjugated_C=C_C=C_addition reactions."""
        r_xyz = """C      -3.03124363    0.21595810   -0.01068883
C      -1.77136356   -0.00875193   -0.22839960
C      -0.51035344   -0.23538255   -0.44913569
C       0.51035356    0.23538291    0.44913621
C       1.77136365    0.00875234    0.22839985
C       3.03124358   -0.21595777    0.01068824
H      -3.50880107    1.10742857   -0.40051872
H      -3.62554573   -0.48341738    0.56587595
H      -0.21235801   -0.79338469   -1.33170668
H       0.21235823    0.79338484    1.33170737
H       3.50880076   -1.10742925    0.40051615
H       3.62554580    0.48341866   -0.56587535"""
        p_xyz = """C      -1.75380171    0.48873088   -0.19068706
C      -0.47932309    0.10898312   -0.05277466
C       0.65826648    1.02120016    0.10389800
C       1.80731799    0.33759624    0.21908285
C       1.46131594   -1.02335073    0.14235481
C       0.04527758   -1.32931253   -0.03040690
H      -2.03784850    1.53610489   -0.19562618
H      -2.54598297   -0.24449127   -0.30238247
H       0.56818218    2.09730281    0.12230795
H       2.80053789    0.73996491    0.34529891
H      -0.15810977   -1.84238058   -0.97429551
H      -0.36583394   -1.89034834    0.81324667"""
        r = ARCSpecies(label='R', smiles='C=C=CC=C=C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C=C1C=C[C]C1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # got no TS

    def test_interpolate_isomerization_Intra_Diels_alder_monocyclic(self):
        """Test the interpolate_isomerization() function for Intra_5_membered_conjugated_C=C_C=C_addition reactions."""
        r_xyz = """C       1.98835311   -0.06142285   -0.00200142
C       0.65433874   -0.02021339   -0.00065871
C      -0.15871220    1.17598961   -0.05259916
C      -1.43846254    0.76969575   -0.03122551
C      -1.48316252   -0.67944287    0.03416677
C      -0.23088987   -1.16395429    0.05299117
H       2.52365896   -1.00407776    0.03918308
H       2.58073849    0.84639620   -0.04432075
H       0.20218412    2.19006475   -0.09915008
H      -2.31139440    1.40309362   -0.05766688
H      -2.39347065   -1.25775412    0.06240301
H       0.06681877   -2.19837465    0.09887848"""
        p_xyz = """C       1.46526410    0.35550989    0.15268159
C       0.45835467    1.13529910   -0.26555553
C      -0.75500438    0.35970165   -0.56989350
C      -1.48532781   -0.35660657    0.46119178
C      -0.34144780   -1.06077923   -0.11686057
C       0.98794173   -1.00683992    0.12489717
H       2.46308379    0.66299943    0.41975788
H       0.51108826    2.21009512   -0.37348204
H      -1.11928864    0.38428608   -1.58978132
H      -2.45322496   -0.77587088    0.21588387
H      -1.38590137   -0.05438209    1.49711542
H       1.65446241   -1.85341259    0.04404524"""
        r = ARCSpecies(label='R', smiles='C=C1C=CC=C1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C1=CC2CC2=C1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # got no TS

    def test_interpolate_isomerization_Intra_RH_Add_Endocyclic(self):
        """Test the interpolate_isomerization() function for Intra_RH_Add_Endocyclic reactions."""
        r_xyz = """O      -2.82721787   -0.50660445    0.17006246
C      -1.42423100   -0.54684140   -0.06325185
C      -0.87209394    0.86948372    0.03307759
C       0.61720459    0.95856824   -0.29789433
C       1.48253511    0.23212057    0.69200350
C       2.29552591   -0.78361542    0.37581471
H      -3.14871804   -1.42308221    0.12085976
H      -0.97202705   -1.20419596    0.68543030
H      -1.25175899   -0.96687073   -1.05925336
H      -1.43076345    1.51736264   -0.65450031
H      -1.06765501    1.27028264    1.03611030
H       0.78939439    0.58580466   -1.31508343
H       0.91578668    2.01385902   -0.29802098
H       1.44262464    0.57498713    1.72449826
H       2.37872656   -1.15563129   -0.64063021
H       2.90404934   -1.26413200    1.13615558"""
        p_xyz = """C       1.49877622    0.35148592    0.11170549
O       0.76070649    1.27450052   -0.68915109
C      -0.59292278    1.40225747   -0.25390393
C      -1.32229318    0.06922018   -0.35968935
C      -0.58250938   -1.01416260    0.41788069
C       0.89132432   -1.04279750    0.02723025
H       1.52973169    0.71083537    1.14705425
H       2.52718630    0.33292459   -0.26270713
H      -1.08027028    2.14514168   -0.89325682
H      -0.61092541    1.78620120    0.77288739
H      -2.34982872    0.16634301    0.00594495
H      -1.37187159   -0.22970950   -1.41399977
H      -0.67028618   -0.81482402    1.49296781
H      -1.04113828   -1.99160041    0.23257309
H       1.44210014   -1.73854431    0.66873858
H       0.97222065   -1.40727159   -1.00427440"""
        r = ARCSpecies(label='R', smiles='OCCCC=C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C1OCCCC1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # ZMatError: Could not determine a starting atom for the zmat

    def test_interpolate_isomerization_Intra_R_Add_Exo_scission(self):
        """Test the interpolate_isomerization() function for Intra_R_Add_Exo_scission reactions."""
        r_xyz = """C       4.23346824   -0.94993099    0.35203386
C       3.27701964   -0.24080855    0.20374022
C       2.14744760    0.59958012    0.02913587
C       1.05580292    0.28887172   -0.94496833
C       0.81218753    1.42065783   -1.91405738
C       1.64631371    1.59084870   -3.02865931
C       1.43302606    2.64455795   -3.91865828
C       0.38802147    3.54057215   -3.70164600
C      -0.44293222    3.38571167   -2.59350729
C      -0.23130881    2.33270646   -1.70172217
H       5.08357894   -1.57799142    0.48427170
H       2.06445726    1.49172813    0.64082561
H       1.28121839   -0.62374363   -1.51126960
H       0.13557426    0.07471597   -0.38754578
H       2.47003331    0.90233884   -3.20563155
H       2.08451055    2.76729073   -4.77982138
H       0.22310907    4.36126939   -4.39461454
H      -1.25568049    4.08672428   -2.42251918
H      -0.88637790    2.22866854   -0.83978334"""
        p_xyz = """C       2.36461930    2.47614099   -0.28244424
C       1.99604231    1.33229290   -0.27285987
C       1.54413882   -0.07391351   -0.26910014
C       2.23688538   -0.81857550    0.83009369
C       0.03108541   -0.18363640   -0.17163035
C      -0.69688127   -0.82392311   -1.18801718
C      -2.08662674   -0.93685605   -1.10708839
C      -2.76791137   -0.41299498   -0.01050932
C      -2.06176492    0.22460524    1.00687539
C      -0.67253825    0.33861613    0.92823056
H       2.68644091    3.49178442   -0.29058723
H       1.87049420   -0.52966109   -1.21308108
H       3.24746658   -0.55865176    1.12244838
H       1.78706966   -1.70886735    1.25422569
H      -0.18547358   -1.23966395   -2.05345409
H      -2.63770469   -1.43352626   -1.90142994
H      -3.84933084   -0.50063791    0.05076577
H      -2.59152597    0.63498118    1.86245464
H      -0.13614794    0.84056704    1.73128084"""
        r = ARCSpecies(label='R', smiles='C#C[CH]Cc1ccccc1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C#CC([CH2])c1ccccc1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # got no TS

    def test_interpolate_isomerization_Intra_R_Add_Exocyclic(self):
        """Test the interpolate_isomerization() function for Intra_R_Add_Exocyclic reactions."""
        r_xyz = """C      -0.10963719   -1.94278328    0.14292941
C      -0.01615129   -0.62011917   -0.55371008
C       1.43738698   -0.10430460   -0.63116148
C       1.69330445    0.06562185   -2.09623303
C       0.63003145   -0.26150822   -2.83431380
C      -0.46489166   -0.68557609   -2.00067251
C      -1.66629105   -1.05892346   -2.45632044
H      -0.17711562   -1.97633031    1.22368923
H       0.10151773   -2.86109453   -0.39408440
H      -0.63261635    0.10871952   -0.01235661
H       2.16203120   -0.79979630   -0.19330736
H       1.53472101    0.85907739   -0.11813416
H       2.63393891    0.41423020   -2.49874047
H       0.58604951   -0.21568141   -3.91331394
H      -2.45097997   -1.36443019   -1.77155996
H      -1.89603539   -1.07075910   -3.51660759"""
        p_xyz = """C       1.78867186    0.18808600    1.08254456
C       0.72594276    0.24149539    0.09992962
C       0.63194514   -0.86489224   -0.86783199
C      -0.61469912   -1.34274785   -1.02500739
C      -1.56795638   -0.57091392   -0.14698479
C      -0.70277748    0.44742414    0.57012130
C       0.02163384    1.54157581   -0.17172013
H       2.67773199    0.79478683    0.95855116
H       1.71925168   -0.48078299    1.93253868
H       1.49635915   -1.24478006   -1.39801088
H      -0.90323661   -2.14800147   -1.68296229
H      -2.33896337   -0.07556696   -0.74496608
H      -2.04619474   -1.24318736    0.57234486
H      -0.92514177    0.57122850    1.61936714
H      -0.27905859    1.76601624   -1.18928709
H       0.31648739    2.42027069    0.39137850"""
        r = ARCSpecies(label='R', smiles='[CH2]C1CC=CC1=C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[CH2]C12C=CCC1C2', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # wrong TS, scattered Hs

    def test_interpolate_isomerization_Intra_ene_reaction(self):
        """Test the interpolate_isomerization() function for Intra_ene_reaction reactions."""
        r_xyz = """C      -1.36262485   -1.33614811   -0.08702211
C      -2.70350854   -0.92818151   -0.37050725
C      -2.82526762    0.39244641   -0.22866599
C      -1.53162451    1.04129650    0.15534834
C      -0.55045629   -0.13880493    0.34325202
C       0.75806531    0.04144663   -0.45229836
C       1.56883117    1.23494814    0.00700816
C       2.78039338    0.83865071    0.42147165
C       2.89382351   -0.58708003    0.27767673
C       1.75271705   -1.07741548   -0.22616482
H      -1.05051909   -2.36554120    0.00006046
H      -3.49586637   -1.60940735   -0.64527347
H      -3.74292673    0.94272427   -0.38363872
H      -1.21830081    1.73449687   -0.63264320
H      -1.64384351    1.60511830    1.08761833
H      -0.32513128   -0.24664896    1.41393869
H       0.55020854    0.13069214   -1.52629302
H       1.22315660    2.25787247   -0.01144886
H       3.56816493    1.47549428    0.79467916
H       3.77674406   -1.15486684    0.52950354
H       1.56743535   -2.11665067   -0.45350367"""
        p_xyz = """C       1.55366820    1.03635804   -0.37383560
C       2.63079531    0.53077850    0.42091537
C       2.55443245   -0.79857991    0.51237427
C       1.37944943   -1.35137258   -0.23811366
C       0.68735565   -0.10328814   -0.83089129
C      -0.73645682    0.05951324   -0.40031363
C      -1.82811265    0.06502298   -1.18415319
C      -3.00360769    0.24633528   -0.37442144
C      -2.64580020    0.35570522    0.91219748
C      -1.15271513    0.25068735    1.03706722
H       1.42462012    2.07140794   -0.64873786
H       3.39167286    1.15259470    0.86987740
H       3.25529926   -1.41683639    1.05573837
H       0.73217040   -1.93146639    0.42772748
H       1.72555758   -2.01696275   -1.03709643
H       0.73432541   -0.15827843   -1.92650695
H      -1.84899398   -0.04738546   -2.25782761
H      -4.00957440    0.28602957   -0.76372185
H      -3.31941194    0.50012284    1.74306462
H      -0.73715017    1.16894496    1.46262037
H      -0.87738606   -0.60600160    1.65882609"""
        r = ARCSpecies(label='R', smiles='[CH]1C=CCC1C1C=CC=C1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[CH]1C=CCC1C1=CC=CC1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))

    def test_interpolate_isomerization_Singlet_Carbene_Intra_Disproportionation(self):
        """Test the interpolate_isomerization() function for Singlet_Carbene_Intra_Disproportionation reactions."""
        r_xyz = """C      -1.75380171    0.48873088   -0.19068706
C      -0.47932309    0.10898312   -0.05277466
C       0.65826648    1.02120016    0.10389800
C       1.80731799    0.33759624    0.21908285
C       1.46131594   -1.02335073    0.14235481
C       0.04527758   -1.32931253   -0.03040690
H      -2.03784850    1.53610489   -0.19562618
H      -2.54598297   -0.24449127   -0.30238247
H       0.56818218    2.09730281    0.12230795
H       2.80053789    0.73996491    0.34529891
H      -0.15810977   -1.84238058   -0.97429551
H      -0.36583394   -1.89034834    0.81324667"""
        p_xyz = """C       1.98835311   -0.06142285   -0.00200142
C       0.65433874   -0.02021339   -0.00065871
C      -0.15871220    1.17598961   -0.05259916
C      -1.43846254    0.76969575   -0.03122551
C      -1.48316252   -0.67944287    0.03416677
C      -0.23088987   -1.16395429    0.05299117
H       2.52365896   -1.00407776    0.03918308
H       2.58073849    0.84639620   -0.04432075
H       0.20218412    2.19006475   -0.09915008
H      -2.31139440    1.40309362   -0.05766688
H      -2.39347065   -1.25775412    0.06240301
H       0.06681877   -2.19837465    0.09887848"""
        r = ARCSpecies(label='R', smiles='C=C1C=C[C]C1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C=C1C=CC=C1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # got no TS

    def test_interpolate_isomerization_Intra_NO2_ONO_conversion(self):
        r_xyz = """O       1.77136558   -0.91790626    0.88650594
N       1.34754589   -0.18857388   -0.01862669
O       1.86645005   -0.03906737   -1.13182045
C       0.08946605    0.57559465    0.25484606
C       0.46072863    1.91146690    0.86342166
H      -0.52075344   -0.02737899    0.93392769
H      -0.43797095    0.69242674   -0.69660400
H       1.09014915    2.48001164    0.17179384
H      -0.42932512    2.51112436    1.08295532
H       1.01533324    1.78326517    1.79934783"""
        p_xyz = """C      -1.36894499    0.07118059   -0.24801399
C      -0.01369535    0.17184136    0.42591278
O      -0.03967083   -0.62462610    1.60609048
N       1.23538512   -0.53558048    2.24863846
O       1.25629155   -1.21389295    3.27993827
H      -2.16063255    0.41812452    0.42429392
H      -1.39509985    0.66980796   -1.16284741
H      -1.59800183   -0.96960842   -0.49986392
H       0.19191326    1.21800574    0.68271847
H       0.76371340   -0.19234475   -0.25650067"""
        r = ARCSpecies(label='R', smiles='[O-][N+](=O)CC', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='CCON=O', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # got no TS

    def test_interpolate_isomerization_Intra_OH_migration(self):
        """Test the interpolate_isomerization() function for Intra_OH_migration reactions."""
        r_xyz = """C      -1.40886397    0.22567351   -0.37379668
C       0.06280787    0.04097694   -0.38515682
O       0.44130326   -0.57668419    0.84260864
O       1.89519755   -0.66754203    0.80966180
H      -1.87218376    0.90693511   -1.07582340
H      -2.03646287   -0.44342165    0.20255768
H       0.35571681   -0.60165457   -1.22096147
H       0.56095122    1.01161503   -0.47393734
H       2.05354047   -0.10415729    1.58865243"""
        p_xyz = """O       0.97298522    1.16961708    0.68631092
C       0.83017736    0.23002128   -0.24518707
C      -0.46505265   -0.55857538    0.09146589
O      -1.54540067    0.36524471    0.24441655
H       1.61381747   -0.53531530   -0.35348282
H       0.69744639    0.56361493   -1.28695526
H      -0.71560487   -1.25802813   -0.71249310
H      -0.36288272   -1.12613201    1.02419042
H      -1.03086141    1.13813060    0.58426610"""
        r = ARCSpecies(label='R', smiles='[CH2]COO', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[O]CCO', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # maybe!

    def test_interpolate_isomerization_Intra_substitutionCS_isomerization(self):
        """Test the interpolate_isomerization() function for Intra_substitutionCS_isomerization reactions."""
        r_xyz = """C       1.49359756    0.26065949    0.12401718
S       0.26300557    0.39883151   -1.09862837
C      -1.16266168   -0.18440138   -0.15528675
H       2.53018726    0.54294783   -0.11178158
H       1.26337431   -0.10426874    1.13589597
H      -1.33584607    0.46046027    0.71053683
H      -0.99959059   -1.21037607    0.18570227
H      -2.05206195   -0.16387486   -0.79046245"""
        p_xyz = """C       0.77758633   -0.01229993    0.03615697
C      -0.74224664   -0.12075638   -0.00763792
S      -1.45914811    1.52171047   -0.37183152
H       1.11694877    0.67695693    0.81818542
H       1.19225093    0.32412348   -0.92127862
H       1.21257173   -0.99402300    0.25412065
H      -1.08603665   -0.52423142    0.95210784
H      -1.01192637   -0.87148015   -0.75982286"""
        r = ARCSpecies(label='R', smiles='[CH2]SC', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='CC[S]', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # got no TS

    def test_interpolate_isomerization_Intra_substitutionS_isomerization(self):
        """Test the interpolate_isomerization() function for Intra_substitutionS_isomerization reactions."""
        r_xyz = """C       2.02473594    0.05810114    0.12967514
S       0.94173618    1.38848441   -0.00439602
S       1.99155683    2.55179194   -1.33352089
C       3.05975458    3.50692441   -0.22777177
H       1.79171393   -0.74186961    0.82204853
H       2.90913559   -0.02956306   -0.49048675
H       3.72773084    2.84617735    0.33119562
H       3.67272000    4.18684912   -0.82584520
H       2.46084746    4.10465096    0.46458235"""
        p_xyz = """C       1.39454780   -0.02562661   -0.01611442
S       0.02609945   -0.59423158   -1.05475409
C       0.59788634    0.03226201   -2.67186639
S       0.44786605    1.84643452   -2.74780581
H       1.38239969    1.06036220    0.10207113
H       1.28884668   -0.47108212    0.97718956
H       2.35414865   -0.34501856   -0.43126943
H      -0.01997740   -0.46631648   -3.42654341
H       1.61593633   -0.33730052   -2.83543977"""
        r = ARCSpecies(label='R', smiles='[CH2]SSC', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='CSC[S]', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        for ts_xyz in ts_xyzs:
            print(f'\nTS xyz:\n\n')
            print(xyz_to_str(ts_xyz))
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))  # maybe!














    def test_linear_adapter(self):
        """Test the LinearAdapter class."""
        self.assertEqual(self.rxn_1.family.label, 'Cyclopentadiene_scission')
        linear_1 = LinearAdapter(job_type='tsg',
                                 reactions=[self.rxn_1],
                                 testing=True,
                                 project='test',
                                 project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_linear', 'rxn_1'),
                                 )
        self.assertIsNone(self.rxn_1.ts_species)
        linear_1.execute()
        self.assertEqual(len(self.rxn_1.ts_species.ts_guesses), 1)
        self.assertEqual(self.rxn_1.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'))

    def test_linear_adapter_2(self):
        self.rxn_2.family = KineticsFamily(label='intra_NO2_ONO_conversion')
        self.rxn_2.atom_map = [0, 1, 3, 2, 4, 5, 7, 6, 9, 8]
        self.assertEqual(self.rxn_2.family.label, 'intra_NO2_ONO_conversion')
        linear_2 = LinearAdapter(job_type='tsg',
                                 reactions=[self.rxn_2],
                                 testing=True,
                                 project='test',
                                 project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_linear', 'rxn_2'),
                                 )
        self.assertIsNone(self.rxn_2.ts_species)
        linear_2.execute()
        self.assertEqual(len(self.rxn_2.ts_species.ts_guesses), 1)
        print(xyz_to_str(self.rxn_2.ts_species.ts_guesses[0].initial_xyz))
        self.assertEqual(self.rxn_2.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'))

    def test_get_r_constraints(self):
        """Test the get_r_constraints() function."""
        self.assertEqual(get_r_constraints([(1, 5)], [(0, 5)]), {'R_atom': [(5, 1)]})
        self.assertEqual(get_r_constraints([(1, 5), (7, 2), (8, 2)], [(0, 5), (7, 4), (8, 1)]), {'R_atom': [(1, 5), (5, 0), (7, 2), (2, 8)]})

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_linear'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
