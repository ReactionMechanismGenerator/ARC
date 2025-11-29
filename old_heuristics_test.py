#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.heuristics module
"""

import copy
import itertools
import os
import unittest
import shutil

from arc.common import ARC_PATH, almost_equal_coords
from arc.family import get_reaction_family_products
from arc.job.adapters.ts.heuristics import (HeuristicsAdapter,
                                            are_h_abs_wells_reversed,
                                            combine_coordinates_with_redundant_atoms,
                                            determine_glue_params,
                                            find_distant_neighbor,
                                            generate_the_two_constrained_zmats,
                                            get_modified_params_from_zmat_2,
                                            get_new_map_based_on_zmat_1,
                                            get_new_zmat_2_map,
                                            stretch_zmat_bond,
                                            )
from arc.reaction import ARCReaction
from arc.species.converter import str_to_xyz, zmat_to_xyz
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
        cls.oh_xyz = """O 0.0000000 0.0000000 0.1078170
                        H 0.0000000 0.0000000 -0.8625320"""
        cls.h2o_xyz = """O      -0.00032832    0.39781490    0.00000000
                         H      -0.76330345   -0.19953755    0.00000000
                         H       0.76363177   -0.19827735    0.00000000"""
        cls.h2_xyz = {'coords': ((0, 0, 0.3736550), (0, 0, -0.3736550)), 'isotopes': (1, 1), 'symbols': ('H', 'H')}
        cls.h2 = ARCSpecies(label='H2', smiles='[H][H]', xyz=cls.h2_xyz)
        cls.o = ARCSpecies(label='O', smiles='[O]')
        cls.h = ARCSpecies(label='H', smiles='[H]')
        cls.oh = ARCSpecies(label='OH', smiles='[OH]', xyz=cls.oh_xyz)
        cls.h2o = ARCSpecies(label='H2O', smiles='O', xyz=cls.h2o_xyz)
        cls.ccooh_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                         'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1, 1),
                         'coords': ((-1.34047, -0.03188, 0.16703), (0.07658, -0.19298, -0.34334),
                                    (0.27374, 0.70670, -1.43275), (1.64704, 0.49781, -1.86879),
                                    (-2.06314, -0.24194, -0.62839), (-1.53242, -0.70687, 1.00574),
                                    (-1.51781, 0.99794, 0.49424), (0.24018, -1.21958, -0.68782),
                                    (0.79344, 0.03863, 0.45152), (1.95991, 1.39912, -1.67215))}
        cls.ccooj_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'),
                         'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1),
                         'coords': ((-1.10653, -0.06552, 0.042602), (0.385508, 0.205048, 0.049674),
                                    (0.759622, 1.114927, -1.032928), (0.675395, 0.525342, -2.208593),
                                    (-1.671503, 0.860958, 0.166273), (-1.396764, -0.534277, -0.898851),
                                    (-1.36544, -0.740942, 0.862152), (0.97386, -0.704577, -0.082293),
                                    (0.712813, 0.732272, 0.947293))}
        cls.c2h6_xyz = {'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'),
                        'isotopes': (12, 12, 1, 1, 1, 1, 1, 1),
                        'coords': ((0.75560, 0.02482, 0.00505), (-0.75560, -0.02482, -0.00505),
                                   (1.17380, -0.93109, -0.32410), (1.11891, 0.80903, -0.66579),
                                   (1.12656, 0.23440, 1.01276), (-1.17380, 0.93109, 0.32410),
                                   (-1.11891, -0.809039, 0.66579), (-1.12656, -0.23440, -1.01276))}
        cls.c2h5_xyz = """C      -0.62870399    0.02330636   -0.00849448
                          C       0.85160870   -0.05497517    0.04976674
                          H      -1.06100002   -0.98045393   -0.04651515
                          H      -0.94436731    0.57609393   -0.89725353
                          H      -1.01744999    0.53143528    0.87837737
                          H       1.37701694   -0.84406878   -0.47511337
                          H       1.42289567    0.74866241    0.49923247"""
        cls.h2_mol = ARCSpecies(label='H2', smiles='[H][H]', xyz=cls.h2_xyz).mol
        cls.ch3_xyz = """C       0.00000000    0.00000001   -0.00000000
                         H       1.06690511   -0.17519582    0.05416493
                         H      -0.68531716   -0.83753536   -0.02808565
                         H      -0.38158795    1.01273118   -0.02607927"""
        cls.ch4_xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'), 'isotopes': (12, 1, 1, 1, 1),
                       'coords': ((-5.45906343962835e-10, 4.233517924761169e-10, 2.9505240956083194e-10),
                                  (-0.6505520089868748, -0.7742801979689132, -0.4125187934483119),
                                  (-0.34927557824779626, 0.9815958255612931, -0.3276823191685369),
                                  (-0.022337921721882443, -0.04887374527620588, 1.0908766524267022),
                                  (1.0221655095024578, -0.15844188273952128, -0.350675540104908))}
        cls.ch4_mol = ARCSpecies(label='CH4', smiles='C', xyz=cls.ch4_xyz).mol
        cls.n2h4_xyz = str_to_xyz("""N      -0.67026921   -0.02117571   -0.25636419
                          N       0.64966276    0.05515705    0.30069593
                          H      -1.27787600    0.74907557    0.03694453
                          H      -1.14684483   -0.88535632    0.02014513
                          H       0.65472168    0.28979031    1.29740292
                          H       1.21533718    0.77074524   -0.16656810""")
        cls.n2h4_mol = ARCSpecies(label='N2H4', smiles='NN', xyz=cls.n2h4_xyz).mol
        cls.nh3_xyz = str_to_xyz("""N       0.00064924   -0.00099698    0.29559292
                         H      -0.41786606    0.84210396   -0.09477452
                         H      -0.52039228   -0.78225292   -0.10002797
                         H       0.93760911   -0.05885406   -0.10079043""")
        cls.nh3_mol = ARCSpecies(label='NH3', smiles='N', xyz=cls.nh3_xyz).mol
        cls.nh2_xyz = str_to_xyz("""N       0.00022972    0.40059496    0.00000000
                                    H      -0.83174214   -0.19982058    0.00000000
                                    H       0.83151242   -0.20077438    0.00000000""")
        cls.h2o_xyz = str_to_xyz("""O      -0.00032832    0.39781490    0.00000000
                                    H      -0.76330345   -0.19953755    0.00000000
                                    H       0.76363177   -0.19827735    0.00000000""")
        cls.ch3ch2oh = ARCSpecies(label='CH3CH2OH', smiles='CCO', xyz="""C      -0.97459464    0.29181710    0.10303882
                                                                         C       0.39565894   -0.35143697    0.10221676
                                                                         O       0.30253309   -1.63748710   -0.49196889
                                                                         H      -1.68942501   -0.32359616    0.65926091
                                                                         H      -0.93861751    1.28685508    0.55523033
                                                                         H      -1.35943743    0.38135479   -0.91822428
                                                                         H       0.76858330   -0.46187184    1.12485643
                                                                         H       1.10301149    0.25256708   -0.47388355
                                                                         H       1.19485981   -2.02360458   -0.47786539""")
        cls.ch3ooh = ARCSpecies(label='CH3OOH', smiles='COO', xyz="""C      -0.76039072    0.01483858   -0.00903344
                                                                     O       0.44475333    0.76952102    0.02291303
                                                                     O       0.16024511    1.92327904    0.86381800
                                                                     H      -1.56632337    0.61401630   -0.44251282
                                                                     H      -1.02943316   -0.30449156    1.00193709
                                                                     H      -0.60052507   -0.86954495   -0.63086438
                                                                     H       0.30391344    2.59629139    0.17435159""")
        cls.zmat_1 = {'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'O', 'O', 'H'),
                      'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                 ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                 ('R_5_1', 'A_5_1_0', 'D_5_1_0_4'), ('R_6_1', 'A_6_1_0', 'D_6_1_0_5'),
                                 ('R_7_1', 'A_7_1_0', 'D_7_1_0_6'), ('R_8_7', 'A_8_7_1', 'D_8_7_1_0'),
                                 ('R_9_8', 'A_9_8_7', 'D_9_8_7_1')),
                      'vars': {'R_1_0': 1.514747977256775, 'R_2_0': 1.0950205326080322, 'A_2_0_1': 110.62464141845703,
                               'R_3_0': 1.093567967414856, 'A_3_0_1': 110.91426086425781,
                               'D_3_0_1_2': 120.87266977773987, 'R_4_0': 1.0950090885162354,
                               'A_4_0_1': 110.62271118164062, 'D_4_0_1_3': 120.87301274044218,
                               'R_5_1': 1.0951433181762695, 'A_5_1_0': 110.20822143554688,
                               'D_5_1_0_4': 181.16392677464202, 'R_6_1': 1.095141053199768,
                               'A_6_1_0': 110.2014389038086, 'D_6_1_0_5': 239.4199964284852,
                               'R_7_1': 1.4265729188919067, 'A_7_1_0': 108.63387298583984,
                               'D_7_1_0_6': 240.2886512602055, 'R_8_7': 1.455925464630127,
                               'A_8_7_1': 105.58023071289062, 'D_8_7_1_0': 179.9922243050821,
                               'R_9_8': 0.9741224646568298, 'A_9_8_7': 96.3006591796875,
                               'D_9_8_7_1': 242.3527063196313},
                      'map': {0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 2, 8: 3, 9: 9}}
        cls.zmat_2 = {'symbols': ('H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                      'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                 ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                                 ('R_5_2', 'A_5_2_1', 'D_5_2_1_4'), ('R_6_2', 'A_6_2_1', 'D_6_2_1_5'),
                                 ('R_7_2', 'A_7_2_1', 'D_7_2_1_6')),
                      'vars': {'R_1_0': 1.0940775871276855, 'R_2_1': 1.5120487213134766, 'A_2_1_0': 110.5680160522461,
                               'R_3_1': 1.0940725803375244, 'A_3_1_2': 110.56890869140625,
                               'D_3_1_2_0': 239.99938309284218, 'R_4_1': 1.0940817594528198,
                               'A_4_1_2': 110.56755065917969, 'D_4_1_2_3': 239.9997190582892,
                               'R_5_2': 1.0940725803375244, 'A_5_2_1': 110.56890869140625,
                               'D_5_2_1_4': 59.99971758419434, 'R_6_2': 1.0940840244293213,
                               'A_6_2_1': 110.56790924072266, 'D_6_2_1_5': 239.99905123159166,
                               'R_7_2': 1.0940817594528198, 'A_7_2_1': 110.56755065917969,
                               'D_7_2_1_6': 240.00122783407815},
                      'map': {0: 3, 1: 0, 2: 1, 3: 2, 4: 4, 5: 5, 6: 6, 7: 7}}
        cls.zmat_3 = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'X'),
                      'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                 ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                 ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_5'),
                                 ('R_7_1', 'A_7_1_0', 'D_7_1_0_6'), ('R_8_1', 'A_8_1_0', 'D_8_1_0_7'),
                                 ('R_9_3', 'A_9_3_2', 'D_9_3_2_1'), ('RX_10_9', 'AX_10_9_3', 'DX_10_9_3_2')),
                      'vars': {'R_1_0': 1.5147479951212197, 'R_2_1': 1.4265728986680748, 'A_2_1_0': 108.63387152978416,
                               'R_3_2': 1.4559254886404387, 'A_3_2_1': 105.58023544826183, 'D_3_2_1_0': 179.9922243050821,
                               'R_4_0': 1.0950205915944824, 'A_4_0_1': 110.62463321031589, 'D_4_0_1_3': 59.13545080998071,
                               'R_5_0': 1.093567969297245, 'A_5_0_1': 110.91425998596507, 'D_5_0_1_4': 120.87266977773987,
                               'R_6_0': 1.0950091062890002, 'A_6_0_1': 110.62270362433773, 'D_6_0_1_5': 120.87301274044218,
                               'R_7_1': 1.0951433842986755, 'A_7_1_0': 110.20822115119915, 'D_7_1_0_6': 181.16392677464265,
                               'R_8_1': 1.0951410439636102, 'A_8_1_0': 110.20143800025897, 'D_8_1_0_7': 239.4199964284852,
                               'R_9_3': 1.1689469645782498, 'A_9_3_2': 96.30065819269021, 'D_9_3_2_1': 242.3527063196313,
                               'RX_10_9': 1.0, 'AX_10_9_3': 90.0, 'DX_10_9_3_2': 0},
                      'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'X10'}}
        cls.zmat_4 = {'symbols': ('H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                      'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                 ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                                 ('R_5_2', 'A_5_2_1', 'D_5_2_1_4'), ('R_6_2', 'A_6_2_1', 'D_6_2_1_5'),
                                 ('R_7_2', 'A_7_2_1', 'D_7_2_1_6')),
                      'vars': {'R_1_0': 1.3128870801982788, 'R_2_1': 1.5120487296562577, 'A_2_1_0': 110.56890700195424,
                               'R_3_1': 1.0940775789443724, 'A_3_1_2': 110.56801921096591, 'D_3_1_2_0': 120.00061587714492,
                               'R_4_1': 1.0940817193677925, 'A_4_1_2': 110.56754686774481, 'D_4_1_2_3': 119.99910067703652,
                               'R_5_2': 1.0940725668318991, 'A_5_2_1': 110.56890700195424, 'D_5_2_1_4': 59.99971758419434,
                               'R_6_2': 1.0940840619688397, 'A_6_2_1': 110.56790845138725, 'D_6_2_1_5': 239.99905123159166,
                               'R_7_2': 1.0940817193677925, 'A_7_2_1': 110.56754686774481, 'D_7_2_1_6': 240.00122783407815},
                      'map': {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}}
        cls.zmat_5 = {'symbols': ('C', 'C', 'X', 'C', 'C', 'C', 'X', 'C', 'X', 'H', 'H', 'H', 'H', 'H', 'X', 'H', 'X'),
                      'coords': ((None, None, None), ('R_1_0', None, None), ('RX_2_1', 'AX_2_1_0', None),
                                 ('R_3_1', 'AX_3_1_2', 'DX_3_1_2_0'), ('R_4_3', 'A_4_3_1', 'DX_4_3_1_2'),
                                 ('R_5_3', 'A_5_3_1', 'D_5_3_1_4'), ('RX_6_5', 'AX_6_5_3', 'DX_6_5_3_1'),
                                 ('R_7_5', 'AX_7_5_6', 'DX_7_5_6_3'), ('RX_8_0', 'AX_8_0_1', 'DX_8_0_1_7'),
                                 ('R_9_0', 'AX_9_0_8', 'DX_9_0_8_1'), ('R_10_3', 'A_10_3_1', 'D_10_3_1_7'),
                                 ('R_11_4', 'A_11_4_3', 'D_11_4_3_1'), ('R_12_4', 'A_12_4_3', 'D_12_4_3_1'),
                                 ('R_13_4', 'A_13_4_3', 'D_13_4_3_1'), ('RX_14_7', 'AX_14_7_5', 'DX_14_7_5_1'),
                                 ('R_15_7', 'AX_15_7_14', 'DX_15_7_14_5'), ('RX_16_10', 'AX_16_10_3', 'DX_16_10_3_1')),
                      'vars': {'R_1_0': 1.2014696201892123, 'RX_2_1': 1.0, 'AX_2_1_0': 90.0, 'R_3_1': 1.4764335616394748,
                               'AX_3_1_2': 90.0, 'DX_3_1_2_0': 180.0, 'R_4_3': 1.528403848430877,
                               'A_4_3_1': 110.10474745663315, 'DX_4_3_1_2': 250.79698398730164, 'R_5_3': 1.4764334808980883,
                               'A_5_3_1': 112.26383853893992, 'D_5_3_1_4': 123.05092674196338, 'RX_6_5': 1.0,
                               'AX_6_5_3': 90.0, 'DX_6_5_3_1': 180, 'R_7_5': 1.201469520969646, 'AX_7_5_6': 90.0,
                               'DX_7_5_6_3': 180.0, 'RX_8_0': 1.0, 'AX_8_0_1': 90.0, 'DX_8_0_1_7': 180,
                               'R_9_0': 1.065642981503376, 'AX_9_0_8': 90.0, 'DX_9_0_8_1': 180.0,
                               'R_10_3': 1.3169771399805865, 'A_10_3_1': 108.17388195099538,
                               'D_10_3_1_7': 119.20794850242746, 'R_11_4': 1.0969184758191393,
                               'A_11_4_3': 111.59730790975621, 'D_11_4_3_1': 62.15337627950438, 'R_12_4': 1.096052090430251,
                               'A_12_4_3': 111.0304823817703, 'D_12_4_3_1': 302.14665453695886,
                               'R_13_4': 1.0960521991926764, 'A_13_4_3': 111.03046862714851,
                               'D_13_4_3_1': 182.16006499876246, 'RX_14_7': 1.0, 'AX_14_7_5': 90.0, 'DX_14_7_5_1': 180,
                               'R_15_7': 1.0656433171015254, 'AX_15_7_14': 90.0, 'DX_15_7_14_5': 180.0, 'RX_16_10': 1.0,
                               'AX_16_10_3': 90.0, 'DX_16_10_3_1': 0},
                      'map': {0: 0, 1: 1, 2: 'X12', 3: 2, 4: 3, 5: 4, 6: 'X13', 7: 5, 8: 'X14', 9: 6, 10: 7, 11: 8, 12: 9,
                              13: 10, 14: 'X15', 15: 11, 16: 'X16'}}
        cls.zmat_6 = {'symbols': ('H', 'C', 'C', 'C', 'X', 'C', 'X', 'C', 'X', 'H', 'H', 'X', 'H'),
                      'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                 ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('RX_4_2', 'AX_4_2_1', 'DX_4_2_1_3'),
                                 ('R_5_2', 'AX_5_2_4', 'DX_5_2_4_1'), ('RX_6_3', 'AX_6_3_1', 'DX_6_3_1_2'),
                                 ('R_7_3', 'AX_7_3_6', 'DX_7_3_6_1'), ('RX_8_5', 'AX_8_5_2', 'DX_8_5_2_7'),
                                 ('R_9_5', 'AX_9_5_8', 'DX_9_5_8_2'), ('R_10_1', 'A_10_1_2', 'D_10_1_2_7'),
                                 ('RX_11_7', 'AX_11_7_3', 'DX_11_7_3_2'), ('R_12_7', 'AX_12_7_11', 'DX_12_7_11_3')),
                      'vars': {'R_1_0': 1.3155122903491134, 'R_2_1': 1.4707410587114869, 'A_2_1_0': 109.22799244788278,
                               'R_3_1': 1.4707410587114869, 'A_3_1_2': 113.21235050581743, 'D_3_1_2_0': 121.94357782706227,
                               'RX_4_2': 1.0, 'AX_4_2_1': 90.0, 'DX_4_2_1_3': 180, 'R_5_2': 1.2013089233618282,
                               'AX_5_2_4': 90.0, 'DX_5_2_4_1': 180.0, 'RX_6_3': 1.0, 'AX_6_3_1': 90.0, 'DX_6_3_1_2': 180,
                               'R_7_3': 1.2013088241289895, 'AX_7_3_6': 90.0, 'DX_7_3_6_1': 180.0, 'RX_8_5': 1.0,
                               'AX_8_5_2': 90.0, 'DX_8_5_2_7': 180, 'R_9_5': 1.06567033240585, 'AX_9_5_8': 90.0,
                               'DX_9_5_8_2': 180.0, 'R_10_1': 1.0962601875867035, 'A_10_1_2': 109.22799322222649,
                               'D_10_1_2_7': 121.94358050468233, 'RX_11_7': 1.0, 'AX_11_7_3': 90.0, 'DX_11_7_3_2': 180,
                               'R_12_7': 1.0656705002006313, 'AX_12_7_11': 90.0, 'DX_12_7_11_3': 180.0},
                      'map': {0: 7, 1: 2, 2: 1, 3: 3, 4: 'X9', 5: 0, 6: 'X10', 7: 4, 8: 'X11', 9: 5, 10: 6, 11: 'X12', 12: 8}}

    def test_heuristics_for_h_abstraction_1(self):
        """
        Test that ARC can generate TS guesses based on heuristics for H Abstraction reactions.
        """
        # H2 + O <=> H + OH
        rxn1 = ARCReaction(r_species=[self.h2, self.o], p_species=[self.h, self.oh])
        self.assertEqual(rxn1.family, 'H_Abstraction')
        heuristics_1 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn1],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=10,
                                         )
        heuristics_1.execute_incore()
        self.assertTrue(rxn1.ts_species.is_ts)
        self.assertEqual(rxn1.ts_species.charge, 0)
        self.assertEqual(rxn1.ts_species.multiplicity, 3)
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 2)
        expected_xyz = {'symbols': ('H', 'H', 'O'), 'isotopes': (1, 1, 16),
                        'coords': ((0.0, 0.0, -1.8806939503689344),
                                   (0.0, 0.0, -0.9839219710369642), (0.0, 0.0, 0.1804968455295033))}
        self.assertTrue(almost_equal_coords(rxn1.ts_species.ts_guesses[0].initial_xyz, expected_xyz))

        # H + OH <=> H2 + O
        rxn2 = ARCReaction(r_species=[self.h, self.oh], p_species=[self.h2, self.o])
        heuristics_2 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn2],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=10,
                                         )
        heuristics_2.execute_incore()
        self.assertEqual(len(rxn2.ts_species.ts_guesses), 1)
        self.assertEqual(rxn2.ts_species.ts_guesses[0].initial_xyz['symbols'], ('H', 'O', 'H'))
        expected_xyz = {'symbols': ('H', 'O', 'H'), 'isotopes': (1, 16, 1),
                        'coords': ((0.0, 0.0, 1.8806939503689346),
                                   (0.0, 0.0, -0.1804968455295031),
                                   (0.0, 0.0, 0.9839219710369642))}
        self.assertTrue(almost_equal_coords(rxn2.ts_species.ts_guesses[0].initial_xyz, expected_xyz))

        # OH + H <=> H2 + O
        rxn3 = ARCReaction(r_species=[self.oh, self.h], p_species=[self.h2, self.o])
        heuristics_3 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn3],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=10,
                                         )
        heuristics_3.execute_incore()
        self.assertEqual(len(rxn3.ts_species.ts_guesses), 1)
        self.assertEqual(rxn3.ts_species.ts_guesses[0].initial_xyz['symbols'], ('O', 'H', 'H'))
        expected_xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                        'coords': ((0.0, 0.0, -0.1804968455295031),
                                   (0.0, 0.0, 0.9839219710369642),
                                   (0.0, 0.0, 1.8806939503689346))}
        self.assertTrue(almost_equal_coords(rxn3.ts_species.ts_guesses[0].initial_xyz, expected_xyz))

        # CH4 + H <=> CH3 + H2
        ch4 = ARCSpecies(label='CH4', smiles='C', xyz=self.ch4_xyz)
        ch3 = ARCSpecies(label='CH3', smiles='[CH3]', xyz=self.ch3_xyz)
        rxn4 = ARCReaction(reactants=['CH4', 'H'], products=['CH3', 'H2'],
                           r_species=[ch4, self.h], p_species=[ch3, self.h2])
        self.assertEqual(rxn4.family, 'H_Abstraction')
        self.assertEqual(rxn4.atom_map[0], 0)
        for index in [1, 2, 3, 4]:
            self.assertIn(rxn4.atom_map[index], [1, 2, 3, 4, 5])
        self.assertIn(rxn4.atom_map[5], [4, 5])
        heuristics_4 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn4],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=120,
                                         )
        heuristics_4.execute_incore()
        self.assertTrue(rxn4.ts_species.is_ts)
        self.assertEqual(rxn4.ts_species.charge, 0)
        self.assertEqual(rxn4.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn4.ts_species.ts_guesses), 4)  # No dihedral scans for H attacking at 180 degrees.
        self.assertTrue(rxn4.ts_species.ts_guesses[0].success)
        expected_xyz = {'symbols': ('C', 'H', 'H', 'H', 'H', 'H'), 'isotopes': (12, 1, 1, 1, 1, 1),
                        'coords': ((-0.14348351563387568, 3.0646463033967564e-08, 1.4446001062040636e-09),
                                   (1.1671558180910826, -1.5372222794685086e-07, -1.0128514993379412e-07),
                                   (-0.5075499920716767, -0.5148677050021889, -0.8917769786794892),
                                   (-0.5075499920716767, -0.5148677050021889, 0.8917772176782378),
                                   (-0.5075499920716767, 1.0297354786962867, 1.666119475718375e-08),
                                   (2.063927797423041, -2.798718649055232e-07, -1.7157539600187732e-07))}
        self.assertTrue(almost_equal_coords(rxn4.ts_species.ts_guesses[0].initial_xyz, expected_xyz))

    def test_heuristics_for_h_abstraction_2(self):
        # C3H8 + HO2 <=> C3H7 + H2O2
        c3h8_xyz = """C	0.0000000 0.0000000 0.5949240
                      C 0.0000000 1.2772010 -0.2630030
                      C 0.0000000 -1.2772010 -0.2630030
                      H 0.8870000 0.0000000 1.2568980
                      H -0.8870000 0.0000000 1.2568980
                      H 0.0000000 2.1863910 0.3643870
                      H 0.0000000 -2.1863910 0.3643870
                      H 0.8933090 1.3136260 -0.9140200
                      H -0.8933090 1.3136260 -0.9140200
                      H -0.8933090 -1.3136260 -0.9140200
                      H 0.8933090 -1.3136260 -0.9140200"""
        ho2_xyz = """O 0.0553530 -0.6124600 0.0000000
                     O 0.0553530 0.7190720 0.0000000
                     H -0.8856540 -0.8528960 0.0000000"""
        c3h7_xyz = """C 1.3077700 -0.2977690 0.0298660
                      C 0.0770610 0.5654390 -0.0483740
                      C -1.2288150 -0.2480100 0.0351080
                      H -2.1137100 0.4097560 -0.0247200
                      H -1.2879330 -0.9774520 -0.7931500
                      H -1.2803210 -0.8079420 0.9859990
                      H 0.1031750 1.3227340 0.7594170
                      H 0.0813910 1.1445730 -0.9987260
                      H 2.2848940 0.1325040 0.2723890
                      H 1.2764100 -1.3421290 -0.3008110"""
        h2o2_xyz = """O 0.0000000 0.7275150 -0.0586880
                      O 0.0000000 -0.7275150 -0.0586880
                      H 0.7886440 0.8942950 0.4695060
                      H -0.7886440 -0.8942950 0.4695060"""
        c3h8 = ARCSpecies(label='C3H8', smiles='CCC', xyz=c3h8_xyz)
        ho2 = ARCSpecies(label='HO2', smiles='O[O]', xyz=ho2_xyz)
        c3h7 = ARCSpecies(label='C3H7', smiles='[CH2]CC', xyz=c3h7_xyz)
        h2o2 = ARCSpecies(label='H2O2', smiles='OO', xyz=h2o2_xyz)
        rxn5 = ARCReaction(reactants=['C3H8', 'HO2'], products=['C3H7', 'H2O2'],
                           r_species=[c3h8, ho2], p_species=[c3h7, h2o2])
        self.assertEqual(rxn5.family, 'H_Abstraction')
        heuristics_5 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn5],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=120,
                                         )
        heuristics_5.execute_incore()
        self.assertTrue(rxn5.ts_species.is_ts)
        self.assertEqual(rxn5.ts_species.charge, 0)
        self.assertEqual(rxn5.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn5.ts_species.ts_guesses), 18)
        self.assertEqual(rxn5.ts_species.ts_guesses[0].initial_xyz['symbols'], ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'O', 'H'))
        self.assertEqual(rxn5.ts_species.ts_guesses[1].initial_xyz['symbols'], ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'O', 'H'))
        self.assertEqual(rxn5.ts_species.ts_guesses[2].initial_xyz['symbols'], ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'O', 'H'))
        self.assertTrue(rxn5.ts_species.ts_guesses[0].success)
        self.assertTrue(rxn5.ts_species.ts_guesses[1].success)
        self.assertTrue(rxn5.ts_species.ts_guesses[2].success)

    def test_heuristics_for_h_abstraction_3(self):
        # CCCOH + OH <=> CCCO + H2O
        cccoh_xyz = """C -1.4562640 1.2257490 0.0000000
                       C 0.0000000 0.7433860 0.0000000
                       C 0.1008890 -0.7771710 0.0000000
                       O 1.4826600 -1.1256940 0.0000000
                       H -1.5081640 2.3212940 0.0000000
                       H -1.9909330 0.8624630 0.8882620
                       H -1.9909330 0.8624630 -0.8882620
                       H 0.5289290 1.1236530 0.8845120
                       H 0.5289290 1.1236530 -0.8845120
                       H -0.4109400 -1.1777970 0.8923550
                       H -0.4109400 -1.1777970 -0.8923550
                       H 1.5250230 -2.0841670 0.0000000"""
        ccco_xyz = """C      -1.22579665    0.34157501   -0.08330600
                      C      -0.04626439   -0.57243496    0.22897599
                      C      -0.11084721   -1.88672335   -0.59040103
                      O       0.94874959   -2.60335587   -0.24842497
                      H      -2.17537216   -0.14662734    0.15781317
                      H      -1.15774972    1.26116047    0.50644174
                      H      -1.23871523    0.61790236   -1.14238547
                      H       0.88193016   -0.02561912    0.01201028
                      H      -0.05081615   -0.78696747    1.30674288
                      H      -1.10865982   -2.31155703   -0.39617740
                      H      -0.21011639   -1.57815495   -1.64338139"""
        cccoh = ARCSpecies(label='CCCOH', smiles='CCCO', xyz=cccoh_xyz)
        ccco = ARCSpecies(label='CCCO', smiles='CCC[O]', xyz=ccco_xyz)
        rxn6 = ARCReaction(reactants=['CCCOH', 'OH'], products=['CCCO', 'H2O'],
                           r_species=[cccoh, self.oh], p_species=[ccco, self.h2o])
        self.assertEqual(rxn6.family, 'H_Abstraction')
        heuristics_6 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn6],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=20,
                                         )
        heuristics_6.execute_incore()
        self.assertTrue(rxn6.ts_species.is_ts)
        self.assertEqual(rxn6.ts_species.charge, 0)
        self.assertEqual(rxn6.ts_species.multiplicity, 2)
        self.assertEqual(rxn6.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'H'))
        self.assertEqual(len(rxn6.ts_species.ts_guesses[1].initial_xyz['coords']), 14)

    def test_heuristics_for_h_abstraction_4(self):
        # C=COH + H <=> C=CO + H2
        cdcoh_xyz = """C      -0.80601307   -0.11773769    0.32792128
                       C       0.23096883    0.47536513   -0.26437348
                       O       1.44620485   -0.11266560   -0.46339257
                       H      -1.74308628    0.41660480    0.45016601
                       H      -0.75733964   -1.13345488    0.70278513
                       H       0.21145717    1.48838416   -0.64841675
                       H       1.41780836   -1.01649567   -0.10468897"""
        cdco_xyz = """C      -0.68324480   -0.04685539   -0.10883672
                      C       0.63642204    0.05717653    0.10011041
                      O       1.50082619   -0.82476680    0.32598015
                      H      -1.27691852    0.84199331   -0.29048852
                      H      -1.17606821   -1.00974165   -0.10030145
                      H       0.99232452    1.08896899    0.06242974"""
        cdcoh = ARCSpecies(label='C=COH', smiles='C=CO', xyz=cdcoh_xyz)
        cdco = ARCSpecies(label='C=CO', smiles='C=C[O]', xyz=cdco_xyz)
        rxn7 = ARCReaction(reactants=['C=COH', 'H'], products=['C=CO', 'H2'],
                           r_species=[cdcoh, self.h], p_species=[cdco, self.h2])
        self.assertEqual(rxn7.family, 'H_Abstraction')
        heuristics_7 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn7],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=120,
                                         )
        heuristics_7.execute_incore()
        self.assertTrue(rxn7.ts_species.is_ts)
        self.assertEqual(rxn7.ts_species.charge, 0)
        self.assertEqual(rxn7.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn7.ts_species.ts_guesses), 1)  # No dihedral scans for H attacking at 180 degrees.
        self.assertEqual(rxn7.ts_species.ts_guesses[0].initial_xyz['symbols'], ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'))

    def test_heuristics_for_h_abstraction_5(self):
        # NCO + NH2 <=> HNCO + NH
        nco_xyz = """N       1.36620399    0.00000000    0.00000000
                     C      -0.09510200    0.00000000    0.00000000
                     O      -1.27110200    0.00000000    0.00000000"""
        nh2_xyz = """N       0.00022972    0.40059496    0.00000000
                     H      -0.83174214   -0.19982058    0.00000000
                     H       0.83151242   -0.20077438    0.00000000"""
        hnco_xyz = """N      -0.70061553    0.28289128   -0.18856549
                      C       0.42761869    0.11537693    0.07336374
                      O       1.55063087   -0.07323229    0.35677630
                      H      -1.27763403   -0.32503592    0.39725197"""
        nh_xyz = """N       0.50949998    0.00000000    0.00000000
                    H      -0.50949998    0.00000000    0.00000000"""
        nco = ARCSpecies(label='NCO', smiles='[N]=C=O', xyz=nco_xyz)
        nh2 = ARCSpecies(label='NH2', smiles='[NH2]', xyz=nh2_xyz)
        hnco = ARCSpecies(label='HNCO', smiles='N=C=O', xyz=hnco_xyz)
        nh = ARCSpecies(label='NH', smiles='[NH]', xyz=nh_xyz)
        rxn8 = ARCReaction(r_species=[nco, nh2], p_species=[hnco, nh])
        self.assertEqual(rxn8.family, 'H_Abstraction')
        heuristics_8 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn8],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=120,
                                         )
        heuristics_8.execute_incore()
        self.assertTrue(rxn8.ts_species.is_ts)
        self.assertEqual(rxn8.ts_species.charge, 0)
        self.assertEqual(rxn8.ts_species.multiplicity, 3)
        self.assertEqual(len(rxn8.ts_species.ts_guesses), 6)
        self.assertTrue(almost_equal_coords(rxn8.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('N', 'C', 'O', 'N', 'H', 'H'), 'isotopes': (14, 12, 16, 14, 1, 1),
                                             'coords': ((-3.657596721635545e-09, 0.08876698337705413, 0.9329034620293603),
                                                        (-3.657596721635545e-09, 0.8029731964901674, 0.005816759497990542),
                                                        (9.609228590831403e-09, 1.4947776654181317, -0.9420542025729876),
                                                        (-3.657596721635545e-09, -2.2452231277287726, 0.16101001965294726),
                                                        (-3.657596721635545e-09, -1.0762904363668742, 0.547597552628467),
                                                        (-3.657596721635545e-09, -2.2452231277287726, -0.8649899620469162))}))
        self.assertTrue(almost_equal_coords(rxn8.ts_species.ts_guesses[1].initial_xyz,
                                            {'symbols': ('N', 'C', 'O', 'N', 'H', 'H'), 'isotopes': (14, 12, 16, 14, 1, 1),
                                             'coords': ((0.7304309896785263, 0.4813753237349452, -0.26044855417424406),
                                                        (-0.22605518455485318, 0.6753956743209286, 0.3853614572987496),
                                                        (-1.2013936879956266, 0.8535814918821567, 1.0130695121236126),
                                                        (0.7304309896785263, -1.8526147873708816, -1.032341996550657),
                                                        (0.7304309896785263, -0.6836820960089831, -0.6457544635751373),
                                                        (0.7304309896785263, -1.8526147873708816, -2.0583419782505206))}))
        self.assertTrue(almost_equal_coords(rxn8.ts_species.ts_guesses[2].initial_xyz,
                                            {'symbols': ('N', 'C', 'O', 'N', 'H', 'H'), 'isotopes': (14, 12, 16, 14, 1, 1),
                                             'coords': ((-0.7304309882994099, 0.48137532979234665, -0.26044855803662603),
                                                        (0.22605518593396912, 0.6753956803783296, 0.3853614534363681),
                                                        (1.201393684372416, 0.8535814759681686, 1.0130695222708512),
                                                        (-0.7304309882994099, -1.8526147813134801, -1.032342000413039),
                                                        (-0.7304309882994099, -0.6836820899515816, -0.6457544674375193),
                                                        (-0.7304309882994099, -1.8526147813134801, -2.0583419821129025))}))

    def test_heuristics_for_h_abstraction_6(self):
        # butenylnebzene + CCOO <=> butenylnebzene_rad + CCOOH
        butenylnebzene_xyz = """C      -1.71226453   -0.84554485    0.38526063
                                C      -3.04422965   -0.42914264    0.41012166
                                C      -3.42029899    0.74723413   -0.23559352
                                C      -2.46544277    1.50753989   -0.90818256
                                C      -1.13299223    1.09288376   -0.93424603
                                C      -0.74341165   -0.08513522   -0.28306031
                                C       0.69357354   -0.54411184   -0.32741768
                                C       0.96111341   -1.45397252   -1.53003141
                                C       2.38975018   -1.92155599   -1.55932472
                                C       3.25117244   -1.64080060   -2.54512753
                                H      -1.43447830   -1.76767058    0.89062734
                                H      -3.78886508   -1.02369999    0.93270709
                                H      -4.45754439    1.07063686   -0.21594061
                                H      -2.75843025    2.42373310   -1.41396154
                                H      -0.39940594    1.69445181   -1.46624302
                                H       0.93550160   -1.06555454    0.60815996
                                H       1.35380726    0.33269483   -0.35901826
                                H       0.70851184   -0.92832434   -2.45979346
                                H       0.31045732   -2.33637041   -1.48650631
                                H       2.72442421   -2.53507765   -0.72507466
                                H       2.96318630   -1.03906198   -3.40168902
                                H       4.27083810   -2.01270921   -2.51458029"""
        peroxyl_xyz = """C      -1.05582103   -0.03329574   -0.10080257
                         C       0.41792695    0.17831205    0.21035514
                         O       1.19234020   -0.65389683   -0.61111443
                         O       2.44749684   -0.41401220   -0.28381363
                         H      -1.33614002   -1.09151783    0.08714882
                         H      -1.25953618    0.21489046   -1.16411897
                         H      -1.67410396    0.62341419    0.54699514
                         H       0.59566350   -0.06437686    1.28256640
                         H       0.67254676    1.24676329    0.02676370"""
        peroxide_xyz = """C      -1.34047532   -0.03188958    0.16703197
                          C       0.07658214   -0.19298564   -0.34334978
                          O       0.27374087    0.70670928   -1.43275058
                          O       1.64704173    0.49781461   -1.86879814
                          H      -2.06314673   -0.24194344   -0.62839800
                          H      -1.53242454   -0.70687225    1.00574309
                          H      -1.51781587    0.99794893    0.49424821
                          H       0.24018222   -1.21958121   -0.68782344
                          H       0.79344780    0.03863434    0.45152272
                          H       1.95991273    1.39912383   -1.67215155"""
        butenylnebzene_rad1_xyz = """C      -1.88266976   -0.87687529   -0.63607576
                                     C      -3.06025073   -0.20236914   -0.30870889
                                     C      -3.07096712    0.71611053    0.73950868
                                     C      -1.90403911    0.96368054    1.45998739
                                     C      -0.72563713    0.29023608    1.13460173
                                     C      -0.70473521   -0.64221365    0.08717649
                                     C       0.57567571   -1.35823396   -0.27232070
                                     C       1.47982006   -0.51744899   -1.11962456
                                     C       2.90883262   -0.59134257   -0.96873657
                                     C       3.75183420    0.10706027   -1.73634793
                                     H      -1.88917338   -1.58645450   -1.46031384
                                     H      -3.96913141   -0.39219681   -0.87352986
                                     H      -3.98783434    1.24160615    0.99265871
                                     H      -1.91005379    1.68377589    2.27388853
                                     H       0.18057426    0.49821548    1.69940227
                                     H       0.34051301   -2.28273463   -0.81410484
                                     H       1.07516828   -1.66856944    0.65418851
                                     H       1.05110536    0.09965361   -1.90323723
                                     H       3.30722691   -1.24504321   -0.19595001
                                     H       4.82473956    0.02594832   -1.59289339
                                     H       3.40309973    0.77336779   -2.51917155"""
        butenylnebzene_rad2_xyz = """C       0.86557186    0.88201467    0.63421602
                                     C       2.10949019    1.51449365    0.71246655
                                     C       3.25224601    0.86186581    0.25912210
                                     C       3.14957012   -0.42059281   -0.27342877
                                     C       1.90421947   -1.05251000   -0.35083861
                                     C       0.74699841   -0.41118094    0.10593634
                                     C      -0.58479314   -1.01936532    0.04460380
                                     C      -0.85980476   -2.42967089   -0.36144757
                                     C      -1.19954275   -2.52225943   -1.82034160
                                     C      -2.37408626   -2.95334594   -2.29914371
                                     H      -0.01246578    1.41307286    0.99522772
                                     H       2.18292853    2.51633055    1.12713435
                                     H       4.22023612    1.35204102    0.31838735
                                     H       4.03880532   -0.93210312   -0.63266934
                                     H       1.86256917   -2.04849879   -0.78187681
                                     H      -1.44058451   -0.38162928    0.24463988
                                     H      -0.01456714   -3.08941132   -0.14056348
                                     H      -1.69219098   -2.80180034    0.24897564
                                     H      -0.42576150   -2.21925087   -2.52323755
                                     H      -2.55420789   -2.99614384   -3.36897081
                                     H      -3.17872709   -3.26780622   -1.64194211"""
        butenylnebzene = ARCSpecies(label='butenylnebzene', smiles='c1ccccc1CCC=C', xyz=butenylnebzene_xyz)
        peroxyl = ARCSpecies(label='CCOO', smiles='CCO[O]', xyz=peroxyl_xyz)
        peroxide = ARCSpecies(label='CCOOH', smiles='CCOO', xyz=peroxide_xyz)
        butenylnebzene_rad1 = ARCSpecies(label='butenylnebzene_rad1', smiles='c1ccccc1C[CH]C=C', xyz=butenylnebzene_rad1_xyz)
        butenylnebzene_rad2 = ARCSpecies(label='butenylnebzene_rad2', smiles='c1ccccc1[CH]CC=C', xyz=butenylnebzene_rad2_xyz)
        rxn9 = ARCReaction(r_species=[butenylnebzene, peroxyl], p_species=[peroxide, butenylnebzene_rad1])
        rxn10 = ARCReaction(r_species=[butenylnebzene, peroxyl], p_species=[peroxide, butenylnebzene_rad2])
        self.assertEqual(rxn9.family, 'H_Abstraction')
        self.assertEqual(rxn10.family, 'H_Abstraction')
        heuristics_9 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn9],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=120,
                                         )
        heuristics_9.execute_incore()
        self.assertTrue(rxn9.ts_species.is_ts)
        self.assertEqual(rxn9.ts_species.charge, 0)
        self.assertEqual(rxn9.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn9.ts_species.ts_guesses), 6)
        self.assertEqual(rxn9.ts_species.ts_guesses[2].initial_xyz['symbols'],
                         ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H',
                          'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C',
                          'O', 'O', 'H', 'H', 'H', 'H', 'H'))
        heuristics_10 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn10],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=120,
                                          )
        heuristics_10.execute_incore()
        self.assertTrue(rxn10.ts_species.is_ts)
        self.assertEqual(rxn10.ts_species.charge, 0)
        self.assertEqual(rxn10.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn10.ts_species.ts_guesses), 6)
        self.assertEqual(rxn10.ts_species.ts_guesses[1].initial_xyz['symbols'],
                         ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H',
                          'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C',
                          'O', 'O', 'H', 'H', 'H', 'H', 'H'))

    def test_heuristics_for_h_abstraction_7(self):
        # C2H5O + CH3OH <=> C2H5OH + CH3O
        c2h5o_xyz = """C      -0.74046271    0.02568566   -0.00568694
                       C       0.79799272   -0.01511040    0.00517437
                       O       1.17260343   -0.72227959   -1.04851579
                       H      -1.13881231   -0.99286049    0.06963185
                       H      -1.14162013    0.59700303    0.84092854
                       H      -1.13266865    0.46233725   -0.93283228
                       H       1.11374677    1.03794239    0.06905096
                       H       1.06944350   -0.38306117    1.00698657"""
        ch3oh_xyz = """C      -0.36862686   -0.00871354    0.04292587
                       O       0.98182901   -0.04902010    0.46594709
                       H      -0.57257378    0.95163086   -0.43693396
                       H      -0.55632373   -0.82564527   -0.65815446
                       H      -1.01755588   -0.12311763    0.91437513
                       H       1.10435907    0.67758465    1.10037299"""
        c2h5oh_xyz = """C      -0.97459464    0.29181710    0.10303882
                        C       0.39565894   -0.35143697    0.10221676
                        O       0.30253309   -1.63748710   -0.49196889
                        H      -1.68942501   -0.32359616    0.65926091
                        H      -0.93861751    1.28685508    0.55523033
                        H      -1.35943743    0.38135479   -0.91822428
                        H       0.76858330   -0.46187184    1.12485643
                        H       1.10301149    0.25256708   -0.47388355
                        H       1.19485981   -2.02360458   -0.47786539"""
        ch3o_xyz = """C       0.03807240    0.00035621   -0.00484242
                      O       1.35198769    0.01264937   -0.17195885
                      H      -0.33965241   -0.14992727    1.02079480
                      H      -0.51702680    0.90828035   -0.29592912
                      H      -0.53338088   -0.77135867   -0.54806440"""
        c2h5o = ARCSpecies(label='C2H5O', smiles='CC[O]', xyz=c2h5o_xyz)
        ch3oh = ARCSpecies(label='CH3OH', smiles='CO', xyz=ch3oh_xyz)
        c2h5oh = ARCSpecies(label='C2H5OH', smiles='CCO', xyz=c2h5oh_xyz)
        ch3o = ARCSpecies(label='CH3O', smiles='C[O]', xyz=ch3o_xyz)
        rxn11 = ARCReaction(r_species=[c2h5o, ch3oh], p_species=[c2h5oh, ch3o])
        self.assertEqual(rxn11.family, 'H_Abstraction')
        heuristics_11 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn11],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=30,
                                          )
        heuristics_11.execute_incore()
        self.assertTrue(rxn11.ts_species.is_ts)
        self.assertEqual(rxn11.ts_species.charge, 0)
        self.assertEqual(rxn11.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn11.ts_species.ts_guesses), 12)
        self.assertEqual(rxn11.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'C', 'O', 'H', 'H', 'H', 'H'))

    def test_heuristics_for_h_abstraction_8(self):
        # NH3 + OH <=> NH2 + H2O
        nh3 = ARCSpecies(label='NH3', smiles='N', xyz=self.nh3_xyz)
        oh = ARCSpecies(label='OH', smiles='[OH]', xyz=self.oh_xyz)
        nh2 = ARCSpecies(label='NH2', smiles='[NH2]', xyz=self.nh2_xyz)
        h2o = ARCSpecies(label='H2O', smiles='O', xyz=self.h2o_xyz)
        rxn12 = ARCReaction(r_species=[nh3, oh], p_species=[nh2, h2o])
        self.assertEqual(rxn12.family, 'H_Abstraction')
        heuristics_12 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn12],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=60,
                                          )
        heuristics_12.execute_incore()
        self.assertTrue(rxn12.ts_species.is_ts)
        self.assertEqual(rxn12.ts_species.charge, 0)
        self.assertEqual(rxn12.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn12.ts_species.ts_guesses), 18)
        for i in range(6):
            self.assertTrue(rxn12.ts_species.ts_guesses[i].success)
        self.assertEqual(rxn12.ts_species.ts_guesses[0].initial_xyz['symbols'], ('N', 'H', 'H', 'H', 'O', 'H'))

        # Reverse order
        # NH2 + H2O <=> NH3 + OH
        rxn13 = ARCReaction(r_species=[nh2, h2o], p_species=[nh3, oh])
        self.assertEqual(rxn13.family, 'H_Abstraction')
        heuristics_13 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn13],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=180,
                                          )
        heuristics_13.execute_incore()
        self.assertTrue(rxn13.ts_species.is_ts)
        self.assertEqual(rxn13.ts_species.charge, 0)
        self.assertEqual(rxn13.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn13.ts_species.ts_guesses), 4)
        self.assertEqual(rxn13.ts_species.ts_guesses[0].initial_xyz['symbols'], ('N', 'H', 'H', 'O', 'H', 'H'))

        # different reactant order
        # H2O + NH2 <=> NH3 + OH
        rxn14 = ARCReaction(r_species=[h2o, nh2], p_species=[nh3, oh])
        self.assertEqual(rxn14.family, 'H_Abstraction')
        heuristics_14 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn14],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=180,
                                          )
        heuristics_14.execute_incore()
        self.assertTrue(rxn14.ts_species.is_ts)
        self.assertEqual(rxn14.ts_species.charge, 0)
        self.assertEqual(rxn14.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn14.ts_species.ts_guesses), 4)
        self.assertEqual(rxn14.ts_species.ts_guesses[0].initial_xyz['symbols'], ('O', 'H', 'H', 'N', 'H', 'H'))

        # different product order
        # NH2 + H2O <=> OH + NH3
        rxn15 = ARCReaction(r_species=[h2o, nh2], p_species=[nh3, oh])
        self.assertEqual(rxn15.family, 'H_Abstraction')
        heuristics_15 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn15],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=180,
                                          )
        heuristics_15.execute_incore()
        self.assertTrue(rxn15.ts_species.is_ts)
        self.assertEqual(rxn15.ts_species.charge, 0)
        self.assertEqual(rxn15.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn15.ts_species.ts_guesses), 4)
        self.assertEqual(rxn15.ts_species.ts_guesses[0].initial_xyz['symbols'], ('O', 'H', 'H', 'N', 'H', 'H'))

    def test_heuristics_for_h_abstraction_9(self):
        # NFCl + H2O <=> NHFCl + OH
        hnfcl = ARCSpecies(label='NHFCl', smiles='N(F)Cl', xyz="""N      -0.14626256    0.12816405    0.30745256
                                                                  F      -0.94719775   -0.91910939   -0.09669786
                                                                  Cl      1.53982436   -0.20497454   -0.07627978
                                                                  H      -0.44636405    0.99591988   -0.13447493""")
        nfcl = ARCSpecies(label='NFCl', smiles='[N](F)Cl', xyz="""N      -0.17697493    0.58788903    0.00000000
                                                                  F      -1.17300047   -0.36581404    0.00000000
                                                                  Cl      1.34997541   -0.22207499    0.00000000""")
        rxn16 = ARCReaction(r_species=[nfcl, self.h2o], p_species=[hnfcl, self.oh])
        self.assertEqual(rxn16.family, 'H_Abstraction')
        heuristics_16 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn16],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=360,
                                          )
        heuristics_16.execute_incore()
        self.assertEqual(len(rxn16.ts_species.ts_guesses), 2)
        self.assertTrue(rxn16.ts_species.ts_guesses[0].success)
        self.assertEqual(rxn16.ts_species.ts_guesses[0].initial_xyz['symbols'], ('N', 'F', 'Cl', 'O', 'H', 'H'))

    def test_heuristics_for_h_abstraction_10(self):
        # HO2 + H2NN(T) <=> O2 + N2H3
        ho2_xyz = """O 0.0553530 -0.6124600 0.0000000
                     O 0.0553530 0.7190720 0.0000000
                     H -0.8856540 -0.8528960 0.0000000"""
        ho2 = ARCSpecies(label='HO2', smiles='O[O]', xyz=ho2_xyz)
        h2nnt = ARCSpecies(label='H2NN(T)', smiles='[N]N', xyz="""N       1.25464159   -0.04494405   -0.06271952
                                                                  N      -0.11832785   -0.00810069    0.29783210
                                                                  H      -0.59897890   -0.78596704   -0.15190060
                                                                  H      -0.53733484    0.83901179   -0.08321197""")
        o2 = ARCSpecies(label='O2', smiles='[O][O]', xyz="""O	0.0000000	0.0000000	0.6029240
                                                            O	0.0000000	0.0000000	-0.6029240""")
        n2h3 = ARCSpecies(label='N2H3', smiles='N[NH]')
        rxn17 = ARCReaction(r_species=[ho2, h2nnt], p_species=[o2, n2h3])
        self.assertEqual(rxn17.family, 'H_Abstraction')
        heuristics_16 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn17],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=60,
                                          )
        heuristics_16.execute_incore()
        self.assertEqual(len(rxn17.ts_species.ts_guesses), 6)
        self.assertTrue(rxn17.ts_species.ts_guesses[0].success)
        self.assertEqual(rxn17.ts_species.ts_guesses[0].initial_xyz['symbols'], ('O', 'O', 'H', 'N', 'N', 'H', 'H'))

    def test_heuristics_for_h_abstraction_11(self):
        # HONO + HNOH <=> NO2 + NH2OH
        hono = ARCSpecies(label='HONO', smiles='ON=O')
        hnoh = ARCSpecies(label='HNOH', smiles='[NH]O')
        no2 = ARCSpecies(label='NO2', smiles='[O-][N+]=O')
        nh2oh = ARCSpecies(label='NH2OH', smiles='NO')
        rxn1 = ARCReaction(r_species=[hono, hnoh], p_species=[no2, nh2oh])
        self.assertEqual(rxn1.family, 'H_Abstraction')
        heuristics_1 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn1],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=60,
                                         )
        heuristics_1.execute_incore()
        self.assertTrue(rxn1.ts_species.is_ts)
        self.assertEqual(rxn1.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 6)

    def test_heuristics_for_h_abstraction_12(self):
        # H2NN(T) + N2H4 <=> N2H3 + N2H3
        h2nn = ARCSpecies(label='H2NN(T)', smiles='[N]N')
        n2h4 = ARCSpecies(label='N2H4', smiles='NN')
        n2h3 = ARCSpecies(label='N2H3', smiles='[NH]N')
        rxn1 = ARCReaction(r_species=[h2nn, n2h4], p_species=[n2h3, n2h3])
        self.assertEqual(rxn1.family, 'H_Abstraction')
        heuristics_1 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn1],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=60,
                                         )
        heuristics_1.execute_incore()
        self.assertTrue(rxn1.ts_species.is_ts)
        self.assertEqual(rxn1.ts_species.charge, 0)
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 24)

    def test_heuristics_for_h_abstraction_13(self):
        # Molecules with linear motifs (and many dummy atoms in both R1H and P2H):
        rxn1 = ARCReaction(r_species=[ARCSpecies(label='CtC[CH]CtC', smiles='C#C[CH]C#C'),
                                      ARCSpecies(label='CtCC[C]CtC', smiles='C#CC(C)C#C')],
                           p_species=[ARCSpecies(label='CtCCCtC', smiles='C#CCC#C'),
                                      ARCSpecies(label='CtC[C][C]CtC', smiles='C#C[C](C)C#C')])
        self.assertEqual(rxn1.family, 'H_Abstraction')
        heuristics_1 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn1],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=60,
                                         )
        heuristics_1.execute_incore()
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 6)

        ch3 = ARCSpecies(label='CH3', smiles='[CH3]')
        cdcdc = ARCSpecies(label='C=C=C', smiles='C=C=C')
        ch4 = ARCSpecies(label='CH4', smiles='C')
        chdcdc = ARCSpecies(label='CH=C=C', smiles='[CH]=C=C')
        rxn1 = ARCReaction(r_species=[ch3, cdcdc], p_species=[ch4, chdcdc])
        self.assertEqual(rxn1.family, 'H_Abstraction')
        heuristics_1 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn1],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=120,
                                         )
        heuristics_1.execute_incore()
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 12)

    def test_keeping_atom_order_in_ts(self):
        """Test that the generated TS has the same atom order as in the reactants"""
        # reactant_reversed, products_reversed = False, False
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz),
                                       ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=self.ccooj_xyz)],
                            p_species=[ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
                                       ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz)])
        self.assertIn(rxn_1.atom_map[0], [0, 1])
        self.assertIn(rxn_1.atom_map[1], [0, 1])
        for index in [2, 3, 4, 5, 6, 7]:
            self.assertIn(rxn_1.atom_map[index], [2, 3, 4, 5, 6, 16])
        self.assertEqual(rxn_1.atom_map[8:12], [7, 8, 9, 10])
        self.assertIn(tuple(rxn_1.atom_map[12:15]), itertools.permutations([13, 11, 12]))
        self.assertIn(rxn_1.atom_map[15:], [[14, 15], [15, 14]])
        heuristics_1 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn_1],
                                         testing=True,
                                         project='test_1',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics_1'),
                                         dihedral_increment=120,
                                         )
        heuristics_1.execute_incore()
        for tsg in rxn_1.ts_species.ts_guesses:
            self.assertEqual(tsg.initial_xyz['symbols'],
                             ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'))

        # reactant_reversed, products_reversed = False, True
        rxn_2 = ARCReaction(r_species=[ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz),
                                       ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=self.ccooj_xyz)],
                            p_species=[ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz),
                                       ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz)])
        self.assertEqual(rxn_2.family, 'H_Abstraction')
        self.assertEqual(rxn_2.atom_map[:2], [11, 10])
        self.assertIn(tuple(rxn_2.atom_map[2:5]), itertools.permutations([9, 16, 15]))
        self.assertIn(tuple(rxn_2.atom_map[5:8]), itertools.permutations([12, 13, 14]))
        self.assertEqual(rxn_2.atom_map[8:12], [0, 1, 2, 3])
        self.assertIn(tuple(rxn_2.atom_map[12:15]), itertools.permutations([4, 5, 6]))
        self.assertIn(tuple(rxn_2.atom_map[15:]), itertools.permutations([7, 8]))
        heuristics_2 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn_2],
                                         testing=True,
                                         project='test_1',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics_1'),
                                         dihedral_increment=120,
                                         )
        heuristics_2.execute_incore()
        for tsg in rxn_2.ts_species.ts_guesses:
            self.assertEqual(tsg.initial_xyz['symbols'],
                             ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'))

        # reactant_reversed, products_reversed = True, False
        rxn_3 = ARCReaction(r_species=[ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=self.ccooj_xyz),
                                       ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz)],
                            p_species=[ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
                                       ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz)])
        self.assertEqual(rxn_3.atom_map[:4], [7, 8, 9, 10])
        self.assertIn(tuple(rxn_3.atom_map[4:7]), itertools.permutations([11, 12, 13]))
        self.assertIn(tuple(rxn_3.atom_map[7:9]), itertools.permutations([14, 15]))
        self.assertEqual(rxn_3.atom_map[9:11], [1, 0])
        self.assertIn(tuple(rxn_3.atom_map[11:14]), itertools.permutations([16, 5, 6]))
        self.assertIn(tuple(rxn_3.atom_map[14:]), itertools.permutations([3, 4, 2]))

        heuristics_3 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn_3],
                                         testing=True,
                                         project='test_1',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics_1'),
                                         dihedral_increment=120,
                                         )
        heuristics_3.execute_incore()
        for tsg in rxn_3.ts_species.ts_guesses:
            self.assertEqual(tsg.initial_xyz['symbols'],
                             ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'))

        # reactant_reversed, products_reversed = True, True
        rxn_4 = ARCReaction(r_species=[ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=self.ccooj_xyz),
                                       ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz)],
                            p_species=[ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz),
                                       ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz)])
        self.assertEqual(rxn_4.atom_map[:4], [0, 1, 2, 3])
        self.assertIn(tuple(rxn_4.atom_map[4:7]), itertools.permutations([4, 5, 6]))
        self.assertIn(tuple(rxn_4.atom_map[7:9]), itertools.permutations([7, 8]))
        self.assertEqual(rxn_4.atom_map[9:11], [11, 10])
        self.assertIn(tuple(rxn_4.atom_map[11:14]), itertools.permutations([9, 15, 16]))
        self.assertIn(tuple(rxn_4.atom_map[14:]), itertools.permutations([12, 13, 14 ]))
        heuristics_4 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn_4],
                                         testing=True,
                                         project='test_1',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics_1'),
                                         dihedral_increment=120,
                                         )
        heuristics_4.execute_incore()
        for tsg in rxn_4.ts_species.ts_guesses:
            self.assertEqual(tsg.initial_xyz['symbols'],
                             ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'))

    def test_combine_coordinates_with_redundant_atoms(self):
        """Test the combine_coordinates_with_redundant_atoms() function."""
        ts_xyz = combine_coordinates_with_redundant_atoms(
            xyz_1=self.ccooh_xyz,
            xyz_2=self.c2h6_xyz,
            mol_1=ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz).mol,
            mol_2=ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz).mol,
            reactant_2=ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
            h1=9,
            h2=5,
            c=2,
            d=0,
            a2=180,
            d2=None,
            d3=0,
            reactants_reversed=False,
        )
        expected_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                        'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1, 1, 12, 12, 1, 1, 1, 1, 1),
                        'coords': ((1.5410341894056017, 0.03255278579527917, -2.1958803093604273),
                                   (1.5410341894056017, 0.03255278579527917, -0.6811323321036524),
                                   (0.3807836408624652, 0.7262089976674171, -0.22531436706462338),
                                   (0.44725483370102004, 0.6862474252614342, 1.2285438144001),
                                   (1.5410341894056017, 1.0573914061715723, -2.581594930574602),
                                   (2.417813888620787, -0.491621171880301, -2.5862518253483486),
                                   (0.6382978847026384, -0.45259259592322265, -2.5815563738259377),
                                   (2.436246144172599, 0.5373544263141657, -0.30283384265673163),
                                   (1.5201959575389625, -0.995008683824121, -0.3029562875823373),
                                   (-0.5371462088963908, 0.07564215831498397, 1.3852370974886976),
                                   (-2.2867840076236203, -0.9301984769015986, 0.23115065682441438),
                                   (-1.6427630361819756, -0.6101509572468562, 1.5612250566727561),
                                   (-3.208131363694941, -1.5016927398697992, 0.3778072894777962),
                                   (-1.609416630920172, -1.5224707666144137, -0.3912654967124345),
                                   (-2.5339216518555596, -0.010500763584261032, -0.3074446156924231),
                                   (-1.3956300152903744, -1.5298551170593468, 2.099824519093628),
                                   (-2.320130400140207, -0.017878639343882265, 2.183641197255159))}
        self.assertTrue(almost_equal_coords(ts_xyz, expected_xyz))

        ts_xyz = combine_coordinates_with_redundant_atoms(xyz_1=self.ccooh_xyz,
                                                          xyz_2=self.c2h6_xyz,
                                                          mol_1=ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz).mol,
                                                          mol_2=ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz).mol,
                                                          reactant_2=ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
                                                          h1=9,
                                                          h2=5,
                                                          c=2,
                                                          d=0,
                                                          a2=150,
                                                          d2=30,
                                                          d3=120,
                                                          reactants_reversed=False,
                                                          )
        expected_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                        'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1, 1, 12, 12, 1, 1, 1, 1, 1),
                        'coords': ((1.3718025969171979, 0.23742954398890095, -2.378138339962993),
                                   (1.3718025969171979, 0.23742954398890095, -0.8633903627062183),
                                   (0.21155204837406139, 0.9310857558610388, -0.4075723976671892),
                                   (0.27802324121261623, 0.891124183455056, 1.0462857837975341),
                                   (1.3718025969171979, 1.262268164365194, -2.763852961177168),
                                   (2.2485822961323834, -0.28674441368667924, -2.7685098559509145),
                                   (0.46906629221423457, -0.24771583772960085, -2.7638144044285036),
                                   (2.2670145516841953, 0.7422311845077875, -0.4850918732592975),
                                   (1.3509643650505587, -0.7901319256304993, -0.4852143181849031),
                                   (-0.7063778013847946, 0.2805189165086057, 1.2029790668861318),
                                   (-1.6675899857195289, -1.7647048216534997, 1.748867627178241),
                                   (-1.5718122041529685, -0.6101944928182649, 0.7771558114650299),
                                   (-2.3887853213596744, -2.5069659960925583, 1.3940149143273226),
                                   (-1.9912618716664985, -1.4129403631715014, 2.7329982955164334),
                                   (-0.6957522841048858, -2.2546439158856977, 1.860657306122405),
                                   (-2.5436577650309165, -0.1202574787360858, 0.6653623387328302),
                                   (-1.2481402862001663, -0.961958946530624, -0.20697484805157007))}
        self.assertTrue(almost_equal_coords(ts_xyz, expected_xyz))

    def test_get_new_zmat2_map(self):
        """Test the get_new_zmat_2_map() function."""
        new_map = get_new_zmat_2_map(zmat_1=self.zmat_3,
                                     zmat_2=self.zmat_4,
                                     reactant_2=ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
                                     reactants_reversed=False,
                                     )
        expected_new_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'X10',
                            11: 12, 12: 11, 13: 16, 14: 17, 15: 14, 16: 15, 17: 13}
        self.assertEqual(new_map, expected_new_map)

        new_map = get_new_zmat_2_map(zmat_1=self.zmat_3,
                                     zmat_2=self.zmat_4,
                                     reactant_2=ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
                                     reactants_reversed=True,
                                     )
        expected_new_map = {0: 7, 1: 8, 2: 9, 3: 10, 4: 11, 5: 12, 6: 13, 7: 14, 8: 15, 9: 16, 10: 'X17',
                            11: 1, 12: 0, 13: 5, 14: 6, 15: 3, 16: 4, 17: 2}
        self.assertEqual(new_map, expected_new_map)

        reactant_2 = ARCSpecies(label='CtC[CH]CtC', smiles='C#C[CH]C#C',
                                xyz={'symbols': ('C', 'C', 'C', 'C', 'C', 'H', 'H', 'H'),
                                     'isotopes': (12, 12, 12, 12, 12, 1, 1, 1),
                                     'coords': ((-2.291605883571667, -0.42331822283552906, -0.5890813676244919),
                                                (-1.2473364930399486, 0.10981539119094576, -0.33203134389873684),
                                                (-0.014210927595876454, 0.7422328036476595, -0.027581014832650772),
                                                (1.2434491739534939, 0.09321854939774785, -0.12912903785018146),
                                                (2.308402589136876, -0.453969497537813, -0.2143594496463511),
                                                (-3.216005202211226, -0.9003163297195633, -0.8182367178025766),
                                                (-0.03399884322236979, 1.7757469592667339, 0.2998700214665439),
                                                (3.251305586550735, -0.9434096534101759, -0.2914025307027816))})
        new_map = get_new_zmat_2_map(zmat_1=self.zmat_5,
                                     zmat_2=self.zmat_6,
                                     reactant_2=reactant_2,
                                     reactants_reversed=True,
                                     )
        # To determine if this test fails for atom-mapping related reasons, use the following xyz:
        # xyz_7 = {'coords': ((-0.11052302098955041, -0.5106945989206113, -2.3628726319919022),
        #     (-0.11052302098955041, -0.5106945989206113, -1.16140301180269),
        #     (-0.11052302098955023, -0.5106945989206112, 0.3150305498367847),
        #     (1.2448888490560643, -0.9827789526552368, 0.8404002762169092),
        #     (-0.4375559903969747, 0.8159552435098156, 0.8744100775429131),
        #     (-0.7036838926552011, 1.8955361195204183, 1.3296134184916002),
        #     (-0.11052302098955026, -0.5106945989206114, -3.4285156134952786),
        #     (-1.0248180325342278, -1.3649565013173555, 0.7257981498364177),
        #     (1.4854985838822663, -1.9838179319127962, 0.46442407690321375),
        #     (1.2491645770965545, -1.0250999599192192, 1.9356267705316639),
        #     (-0.939726019056252, 2.853070310535801, 1.733355993511537)),
        # 'isotopes': (12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1),
        # 'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H')}
        # To generate a reaction, and check it's atom mapping!
        # Another mapping option to try is:
        # expected_new_map = {0: 12, 1: 13, 2: 'X24', 3: 14, 4: 15, 5: 16, 6: 'X25', 7: 17, 8: 'X26', 9: 18, 10: 19,
        #                     11: 20, 12: 21, 13: 22, 14: 'X27', 15: 23, 16: 'X28', 17: 2, 18: 3, 19: 1, 21: 4, 23: 0,
        #                     25: 7, 26: 6, 28: 5, 20: 'X8', 22: 'X9', 24: 'X10', 27: 'X11'}
        expected_new_map =  {0: 12, 1: 13, 2: 'X24', 3: 14, 4: 15, 5: 16, 6: 'X25', 7: 17, 8: 'X26', 9: 18, 10: 19,
                             11: 20, 12: 21, 13: 22, 14: 'X27', 15: 23, 16: 'X28', 17: 2, 18: 1, 19: 3, 21: 0, 23: 4,
                             25: 5, 26: 6, 28: 7, 20: 'X8', 22: 'X9', 24: 'X10', 27: 'X11'}

        self.assertEqual(new_map, expected_new_map)

    def test_get_new_map_based_on_zmat_1(self):
        """Test the get_new_map_based_on_zmat_1() function."""
        new_map = get_new_map_based_on_zmat_1(zmat_1=self.zmat_1, zmat_2=self.zmat_2, reactants_reversed=False)
        self.assertEqual(new_map, {0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 2, 8: 3, 9: 9})

        new_map = get_new_map_based_on_zmat_1(zmat_1=self.zmat_1, zmat_2=self.zmat_2, reactants_reversed=True)
        self.assertEqual(new_map, {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15, 7: 9, 8: 10, 9: 16})  # +7

        new_map = get_new_map_based_on_zmat_1(zmat_1=self.zmat_2, zmat_2=self.zmat_1, reactants_reversed=False)
        self.assertEqual(new_map, {0: 3, 1: 0, 2: 1, 3: 2, 4: 4, 5: 5, 6: 6, 7: 7})

        new_map = get_new_map_based_on_zmat_1(zmat_1=self.zmat_2, zmat_2=self.zmat_1, reactants_reversed=True)
        self.assertEqual(new_map, {0: 12, 1: 9, 2: 10, 3: 11, 4: 13, 5: 14, 6: 15, 7: 16})  # +9

        new_map = get_new_map_based_on_zmat_1(zmat_1=self.zmat_3, zmat_2=self.zmat_4, reactants_reversed=False)
        self.assertEqual(new_map, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'X10'})

        new_map = get_new_map_based_on_zmat_1(zmat_1=self.zmat_3, zmat_2=self.zmat_4, reactants_reversed=True)
        self.assertEqual(new_map, {0: 7, 1: 8, 2: 9, 3: 10, 4: 11, 5: 12, 6: 13, 7: 14, 8: 15, 9: 16, 10: 'X17'})  # +7

        new_map = get_new_map_based_on_zmat_1(zmat_1=self.zmat_4, zmat_2=self.zmat_3, reactants_reversed=False)
        self.assertEqual(new_map, {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7})

        new_map = get_new_map_based_on_zmat_1(zmat_1=self.zmat_4, zmat_2=self.zmat_3, reactants_reversed=True)
        self.assertEqual(new_map, {0: 12, 1: 10, 2: 11, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17})  # +10

        new_map = get_new_map_based_on_zmat_1(zmat_1=self.zmat_5, zmat_2=self.zmat_6, reactants_reversed=False)
        self.assertEqual(new_map, {0: 0, 1: 1, 2: 'X12', 3: 2, 4: 3, 5: 4, 6: 'X13', 7: 5, 8: 'X14', 9: 6, 10: 7, 11: 8,
                                   12: 9, 13: 10, 14: 'X15', 15: 11, 16: 'X16'})

        new_map = get_new_map_based_on_zmat_1(zmat_1=self.zmat_5, zmat_2=self.zmat_6, reactants_reversed=True)
        self.assertEqual(new_map, {0: 12, 1: 13, 2: 'X24', 3: 14, 4: 15, 5: 16, 6: 'X25', 7: 17, 8: 'X26', 9: 18,
                                   10: 19, 11: 20, 12: 21, 13: 22, 14: 'X27', 15: 23, 16: 'X28'})  # +12

        new_map = get_new_map_based_on_zmat_1(zmat_1=self.zmat_6, zmat_2=self.zmat_5, reactants_reversed=False)
        self.assertEqual(new_map, {0: 7, 1: 2, 2: 1, 3: 3, 4: 'X9', 5: 0, 6: 'X10', 7: 4, 8: 'X11', 9: 5, 10: 6,
                                   11: 'X12', 12: 8})

        new_map = get_new_map_based_on_zmat_1(zmat_1=self.zmat_6, zmat_2=self.zmat_5, reactants_reversed=True)
        self.assertEqual(new_map, {0: 23, 1: 18, 2: 17, 3: 19, 4: 'X25', 5: 16, 6: 'X26', 7: 20, 8: 'X27', 9: 21,
                                   10: 22, 11: 'X28', 12: 24})  # +16

    def test_generate_the_two_constrained_zmats_1(self):
        """Test the generate_the_two_constrained_zmats() function."""
        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.ccooh_xyz,
                                                            xyz_2=self.c2h6_xyz,
                                                            mol_1=ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz).mol,
                                                            mol_2=ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz).mol,
                                                            h1=9,
                                                            h2=3,
                                                            a=3,
                                                            b=0,
                                                            c=2,
                                                            d=1,
                                                            )
        self.assertTrue(_compare_zmats(zmat_1, self.zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, self.zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.ch4_xyz,
                                                            xyz_2=self.h2_xyz,
                                                            mol_1=self.ch4_mol,
                                                            mol_2=self.h2_mol,
                                                            h1=2,
                                                            h2=0,
                                                            a=0,
                                                            b=1,
                                                            c=None,
                                                            d=None,
                                                            )
        expected_zmat_1 = {'symbols': ('H', 'H', 'H', 'C', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_3', 'A_4_3_0', 'D_4_3_0_2')),
                           'vars': {'R_1_0': 1.783554196357727, 'R_2_1': 1.783554196357727,
                                    'A_2_1_0': 59.999996185302734, 'R_3_0': 1.092199444770813,
                                    'A_3_0_1': 35.26438522338867, 'D_3_0_1_2': 324.7356118469746,
                                    'R_4_3': 1.0921993255615234, 'A_4_3_0': 109.47122192382812,
                                    'D_4_3_0_2': 120.00000068116007},
                           'map': {0: 1, 1: 3, 2: 4, 3: 0, 4: 2}}
        expected_zmat_2 = {'symbols': ('H', 'H'), 'coords': ((None, None, None), ('R_1_0', None, None)),  # R_1_0
                           'vars': {'R_1_0': 0.7473099866382779}, 'map': {0: 0, 1: 1}}
        self.assertTrue(_compare_zmats(zmat_1, expected_zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, expected_zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

    def test_generate_the_two_constrained_zmats_2(self):
        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.n2h4_xyz,
                                                            xyz_2=self.nh3_xyz,
                                                            mol_1=self.n2h4_mol,
                                                            mol_2=self.nh3_mol,
                                                            h1=2,
                                                            h2=1,
                                                            a=0,
                                                            b=0,
                                                            c=1,
                                                            d=None,
                                                            )
        expected_zmat_1 = {'symbols': ('H', 'H', 'H', 'N', 'N', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # A_2_1_0
                                      ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_3', 'A_4_3_1', 'D_4_3_1_0'),
                                      ('R_5_4', 'A_5_4_3', 'D_5_4_3_2')),
                           'vars': {'R_1_0': 2.5015993118286133, 'R_2_1': 1.6397618055343628,
                                    'A_2_1_0': 85.88031768798828, 'R_3_1': 1.023964524269104,
                                    'A_3_1_2': 36.857383728027344, 'D_3_1_2_0': 23.003053425503172,
                                    'R_4_3': 1.4346996545791626, 'A_4_3_1': 113.24586486816406,
                                    'D_4_3_1_0': 27.701374618632435, 'R_5_4': 1.0239644050598145,
                                    'A_5_4_3': 113.24588012695312, 'D_5_4_3_2': 284.3887426902492},
                           'map': {0: 3, 1: 4, 2: 5, 3: 1, 4: 0, 5: 2}}
        expected_zmat_2 = {'symbols': ('H', 'N', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # R_1_0
                                      ('R_3_1', 'A_3_1_0', 'D_3_1_0_2')),
                           'vars': {'R_1_0': 1.0190000355938578, 'R_2_1': 1.0189999771005855,
                                    'A_2_1_0': 105.99799962283844, 'R_3_1': 1.0190000940871264,
                                    'A_3_1_0': 105.99799852287603, 'D_3_1_0_2': 112.36218461898632},
                           'map': {0: 1, 1: 0, 2: 2, 3: 3}}
        self.assertTrue(_compare_zmats(zmat_1, expected_zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, expected_zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

    def test_generate_the_two_constrained_zmats_3(self):
        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_2=self.n2h4_xyz,
                                                            xyz_1=self.nh3_xyz,
                                                            mol_2=self.n2h4_mol,
                                                            mol_1=self.nh3_mol,
                                                            h2=2,
                                                            h1=1,
                                                            b=0,
                                                            a=0,
                                                            d=1,
                                                            c=None,
                                                            )
        expected_zmat_1 = {'symbols': ('H', 'H', 'N', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_2', 'A_3_2_0', 'D_3_2_0_1')),
                           'vars': {'R_1_0': 1.627597689628601, 'R_2_0': 1.0189999341964722,
                                    'A_2_0_1': 37.0009880065918, 'R_3_2': 1.0190000534057617,  # R_3_2
                                    'A_3_2_0': 105.99800872802734, 'D_3_2_0_1': 112.36217876566015},
                           'map': {0: 2, 1: 3, 2: 0, 3: 1}}
        expected_zmat_2 = {'symbols': ('H', 'N', 'N', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # R_1_0, A_2_1_0
                                      ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_2', 'A_4_2_1', 'D_4_2_1_3'),
                                      ('R_5_2', 'A_5_2_1', 'D_5_2_1_4')),
                           'vars': {'R_1_0': 1.0239644050598145, 'R_2_1': 1.4346996545791626,
                                    'A_2_1_0': 113.24588012695312, 'R_3_1': 1.0248854160308838,
                                    'A_3_1_2': 111.58697509765625, 'D_3_1_2_0': 240.07077704046904,
                                    'R_4_2': 1.023964524269104, 'A_4_2_1': 113.24586486816406,
                                    'D_4_2_1_3': 284.3887260014507, 'R_5_2': 1.0248851776123047,
                                    'A_5_2_1': 111.58697509765625, 'D_5_2_1_4': 240.0707951638342},
                           'map': {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 5}}
        self.assertTrue(_compare_zmats(zmat_1, expected_zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, expected_zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

    def test_generate_the_two_constrained_zmats_4(self):
        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.ch3ch2oh.get_xyz(),
                                                            xyz_2=self.ch3ooh.get_xyz(),
                                                            mol_1=self.ch3ch2oh.mol,
                                                            mol_2=self.ch3ooh.mol,
                                                            h1=8,
                                                            a=2,
                                                            c=1,
                                                            h2=6,
                                                            b=2,
                                                            d=1,
                                                            )
        expected_zmat_1 = {'symbols': ('C', 'H', 'H', 'H', 'H', 'H', 'C', 'O', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_3', 'A_4_3_0', 'D_4_3_0_1'),
                                      ('R_5_4', 'A_5_4_3', 'D_5_4_3_0'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_4'),
                                      ('R_7_6', 'A_7_6_0', 'D_7_6_0_5'), ('R_8_7', 'A_8_7_6', 'D_8_7_6_0')),  # A_8_7_6
                           'vars': {'R_1_0': 1.0950337648391724, 'R_2_0': 1.0935593843460083,
                                    'A_2_0_1': 108.8328628540039, 'R_3_0': 1.0950340032577515,
                                    'A_3_0_1': 106.87371826171875, 'D_3_0_1_2': 242.6215154409431,
                                    'R_4_3': 3.0681724548339844, 'A_4_3_0': 27.468975067138672,
                                    'D_4_3_0_1': 240.4633372341249, 'R_5_4': 1.7827603816986084,
                                    'A_5_4_3': 54.746910095214844, 'D_5_4_3_0': 219.39614758137319,
                                    'R_6_0': 1.5137276649475098, 'A_6_0_1': 110.6325912475586,
                                    'D_6_0_1_4': 24.509123650965563, 'R_7_6': 1.4197372198104858,
                                    'A_7_6_0': 109.0130386352539, 'D_7_6_0_5': 240.46405277770435,
                                    'R_8_7': 0.972385048866272, 'A_8_7_6': 107.0633544921875,
                                    'D_8_7_6_0': 179.99951584936258},
                           'map': {0: 0, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 1, 7: 2, 8: 8}}
        expected_zmat_2 = {'symbols': ('H', 'O', 'C', 'O', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                      ('R_3_1', 'A_3_1_0', 'D_3_1_0_2'), ('R_4_2', 'A_4_2_3', 'D_4_2_3_1'),
                                      ('R_5_2', 'A_5_2_3', 'D_5_2_3_4'), ('R_6_2', 'A_6_2_3', 'D_6_2_3_5')),
                           'vars': {'R_1_0': 0.9741406440734863, 'R_2_1': 2.2916336059570312,
                                    'A_2_1_0': 111.40901184082031, 'R_3_1': 1.4557554721832275,
                                    'A_3_1_0': 96.30924987792969, 'D_3_1_0_2': 34.63115564274292,
                                    'R_4_2': 1.09382164478302, 'A_4_2_3': 110.0324935913086,
                                    'D_4_2_3_1': 60.84271231265853, 'R_5_2': 1.0938084125518799,
                                    'A_5_2_3': 110.03299713134766, 'D_5_2_3_4': 238.4134975536592,
                                    'R_6_2': 1.0928699970245361, 'A_6_2_3': 108.55512237548828,
                                    'D_6_2_3_5': 240.7911024479184}, 'map': {0: 6, 1: 2, 2: 0, 3: 1, 4: 3, 5: 4, 6: 5}}
        self.assertTrue(_compare_zmats(zmat_1, expected_zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, expected_zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

    def test_generate_the_two_constrained_zmats_5(self):
        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_2=self.ch3ch2oh.get_xyz(),
                                                            xyz_1=self.ch3ooh.get_xyz(),
                                                            mol_2=self.ch3ch2oh.mol,
                                                            mol_1=self.ch3ooh.mol,
                                                            h2=8,
                                                            b=2,
                                                            d=1,
                                                            h1=6,
                                                            a=2,
                                                            c=1,
                                                            )
        expected_zmat_1 = {'symbols': ('C', 'H', 'H', 'H', 'O', 'O', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                      ('R_5_4', 'A_5_4_0', 'D_5_4_0_3'), ('R_6_5', 'A_6_5_4', 'D_6_5_4_0')),  # A_6_5_4
                           'vars': {'R_1_0': 1.09382164478302, 'R_2_0': 1.0938084125518799,
                                    'A_2_0_1': 110.18033599853516, 'R_3_0': 1.0928699970245361,
                                    'A_3_0_1': 109.0003662109375, 'D_3_0_1_2': 119.56910439183824,
                                    'R_4_0': 1.4223003387451172, 'A_4_0_1': 110.0324935913086,
                                    'D_4_0_1_3': 118.9322728845415, 'R_5_4': 1.4557554721832275,
                                    'A_5_4_0': 105.53988647460938, 'D_5_4_0_3': 180.04731322811497,
                                    'R_6_5': 0.9741406440734863, 'A_6_5_4': 96.30924987792969,
                                    'D_6_5_4_0': 242.22744613021646}, 'map': {0: 0, 1: 3, 2: 4, 3: 5, 4: 1, 5: 2, 6: 6}}
        expected_zmat_2 = {'symbols': ('H', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                      ('R_3_1', 'A_3_1_0', 'D_3_1_0_2'), ('R_4_2', 'A_4_2_3', 'D_4_2_3_1'),
                                      ('R_5_2', 'A_5_2_3', 'D_5_2_3_4'), ('R_6_2', 'A_6_2_3', 'D_6_2_3_5'),
                                      ('R_7_3', 'A_7_3_2', 'D_7_3_2_6'), ('R_8_3', 'A_8_3_2', 'D_8_3_2_7')),
                           'vars': {'R_1_0': 0.972385048866272, 'R_2_1': 2.3889963626861572,
                                    'A_2_1_0': 143.86575317382812, 'R_3_1': 1.4197372198104858,
                                    'A_3_1_0': 107.0633544921875, 'D_3_1_0_2': 359.99950824713164,
                                    'R_4_2': 1.0950337648391724, 'A_4_2_3': 110.6325912475586,
                                    'D_4_2_3_1': 59.12026293645304, 'R_5_2': 1.0935593843460083,
                                    'A_5_2_3': 110.9225845336914, 'D_5_2_3_4': 120.87939658650379,
                                    'R_6_2': 1.0950340032577515, 'A_6_2_3': 110.63255310058594,
                                    'D_6_2_3_5': 120.87939551565508, 'R_7_3': 1.0941026210784912,
                                    'A_7_3_2': 110.54409790039062, 'D_7_3_2_6': 181.34310124526942,
                                    'R_8_3': 1.094102382659912, 'A_8_3_2': 110.54409790039062,
                                    'D_8_3_2_7': 239.07189759901027},
                           'map': {0: 8, 1: 2, 2: 0, 3: 1, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}}
        self.assertTrue(_compare_zmats(zmat_1, expected_zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, expected_zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

    def test_generate_the_two_constrained_zmats_6(self):
        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.ch3ch2oh.get_xyz(),
                                                            xyz_2=self.ch3ooh.get_xyz(),
                                                            mol_1=self.ch3ch2oh.mol,
                                                            mol_2=self.ch3ooh.mol,
                                                            h1=7,
                                                            a=4,
                                                            c=2,
                                                            h2=3,
                                                            b=0,
                                                            d=1,
                                                            )
        expected_zmat_1 = {'symbols': ('O', 'H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                      ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_2', 'A_4_2_3', 'D_4_2_3_1'),
                                      ('R_5_2', 'A_5_2_3', 'D_5_2_3_4'), ('R_6_3', 'A_6_3_2', 'D_6_3_2_5'),
                                      ('R_7_0', 'A_7_0_3', 'D_7_0_3_2'), ('R_8_1', 'A_8_1_0', 'D_8_1_0_3')),
                           'vars': {'R_1_0': 3.344974994659424, 'R_2_1': 1.0935593843460083,
                                    'A_2_1_0': 24.11415672302246, 'R_3_2': 1.5137276649475098,
                                    'A_3_2_1': 110.9225845336914, 'D_3_2_1_0': 359.99966761000894,
                                    'R_4_2': 1.0950337648391724, 'A_4_2_3': 110.6325912475586,
                                    'D_4_2_3_1': 239.12060339004628, 'R_5_2': 1.0950340032577515,
                                    'A_5_2_3': 110.63255310058594, 'D_5_2_3_4': 241.75879172737805,
                                    'R_6_3': 1.0941026210784912, 'A_6_3_2': 110.54409790039062,
                                    'D_6_3_2_5': 181.34310124526942, 'R_7_0': 0.972385048866272,
                                    'A_7_0_3': 107.0633544921875, 'D_7_0_3_2': 179.99951584936258,
                                    'R_8_1': 2.509397506713867, 'A_8_1_0': 37.757564544677734, 'D_8_1_0_3': 35.458506109251914},
                           'map': {0: 2, 1: 4, 2: 0, 3: 1, 4: 3, 5: 5, 6: 6, 7: 8, 8: 7}}
        expected_zmat_2 = {'symbols': ('H', 'C', 'O', 'O', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                      ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                                      ('R_5_1', 'A_5_1_2', 'D_5_1_2_4'), ('R_6_3', 'A_6_3_2', 'D_6_3_2_1')),
                           'vars': {'R_1_0': 1.09382164478302, 'R_2_1': 1.4223003387451172,
                                    'A_2_1_0': 110.0324935913086, 'R_3_2': 1.4557554721832275,
                                    'A_3_2_1': 105.53988647460938, 'D_3_2_1_0': 60.84271231265853,
                                    'R_4_1': 1.0938084125518799, 'A_4_1_2': 110.03299713134766,
                                    'D_4_1_2_3': 299.2562112207745, 'R_5_1': 1.0928699970245361,
                                    'A_5_1_2': 108.55512237548828, 'D_5_1_2_4': 240.7911024479184,
                                    'R_6_3': 0.9741406440734863, 'A_6_3_2': 96.30924987792969,
                                    'D_6_3_2_1': 242.22744613021646}, 'map': {0: 3, 1: 0, 2: 1, 3: 2, 4: 4, 5: 5, 6: 6}}
        self.assertTrue(_compare_zmats(zmat_1, expected_zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, expected_zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

    def test_stretch_zmat_bond(self):
        """Test the stretch_zmat_bond function."""
        zmat2_copy = copy.deepcopy(self.zmat_2)
        stretch_zmat_bond(zmat=zmat2_copy, indices=(1, 0), stretch=1.5)
        self.assertEqual(zmat2_copy['vars']['R_2_1'], self.zmat_2['vars']['R_2_1'] * 1.5)

    def test_determine_glue_params(self):
        """Test the determine_glue_params() function."""
        zmat_0 = {'symbols': ('O', 'H'), 'coords': ((None, None, None), ('R_1_0', None, None)),
                  'vars': {'R_1_0': 1.1644188088546794}, 'map': {0: 0, 1: 1}}
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_0,
                                                             add_dummy=False,
                                                             h1=0,
                                                             a=1,
                                                             c=None,
                                                             d=None,
                                                             )
        self.assertEqual(param_a2, 'A_2_0_1')  # B-H-A
        self.assertEqual(param_d2, None)  # B-H-A-C
        self.assertEqual(param_d3, None)  # D-B-H-A

        # None linear
        zmat_1 = {'symbols': ('H', 'N', 'C', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_2', 'A_4_2_1', 'D_4_2_1_3'),
                             ('R_5_2', 'A_5_2_1', 'D_5_2_1_4'), ('R_6_1', 'A_6_1_2', 'D_6_1_2_5')),
                  'vars': {'R_1_0': 1.019169359544101, 'R_2_1': 1.451965854148702, 'A_2_1_0': 109.41187480598941,
                           'R_3_2': 1.0935188594180785, 'A_3_2_1': 110.20496026605478,
                           'D_3_2_1_0': 60.83821034525936, 'R_4_2': 1.0935188594180785,
                           'A_4_2_1': 110.20495933616156, 'D_4_2_1_3': 240.4644263689792,
                           'R_5_2': 1.0936965384360282, 'A_5_2_1': 110.59878027260544,
                           'D_5_2_1_4': 239.76779188408136, 'R_6_1': 1.0191693010605467,
                           'A_6_1_2': 109.41187816450345, 'D_6_1_2_5': 65.17113681053117},
                  'map': {0: 5, 1: 1, 2: 0, 3: 2, 4: 3, 5: 4, 6: 6}}
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_1,
                                                             add_dummy=False,
                                                             h1=5,
                                                             a=1,
                                                             c=0,
                                                             d=1,
                                                             )
        self.assertEqual(param_a2, 'A_7_0_1')  # B-H-A
        self.assertEqual(param_d2, 'D_7_0_1_2')  # B-H-A-C
        self.assertEqual(param_d3, 'D_8_7_0_1')  # D-B-H-A

        # Linear
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_1,
                                                             add_dummy=True,
                                                             h1=5,
                                                             a=1,
                                                             c=0,
                                                             d=1,
                                                             )
        self.assertEqual(param_a2, 'A_8_0_1')  # B-H-A
        self.assertEqual(param_d2, 'D_8_0_7_1')  # B-H-X-A
        self.assertEqual(param_d3, 'D_9_8_1_2')  # D-B-A-C/X

        zmat_1_copy = copy.deepcopy(self.zmat_1)
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_1_copy,
                                                             add_dummy=True,
                                                             h1=9,
                                                             a=3,
                                                             c=2,
                                                             d=0,
                                                             )
        self.assertEqual(param_a2, 'A_11_9_8')
        self.assertEqual(param_d2, 'D_11_9_10_8')
        self.assertEqual(param_d3, 'D_12_11_8_7')
        self.assertEqual(zmat_1_copy['symbols'][-1], 'X')
        self.assertEqual(zmat_1_copy['coords'][-1], ('RX_10_9', 'AX_10_9_8', 'DX_10_9_8_7'))
        self.assertEqual(zmat_1_copy['vars']['RX_10_9'], 1.0)
        self.assertEqual(zmat_1_copy['vars']['AX_10_9_8'], 90)
        self.assertEqual(zmat_1_copy['vars']['DX_10_9_8_7'], 0)
        self.assertEqual(zmat_1_copy['map'][10], 'X10')
        expected_xyz_1 = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'X'),
                          'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1, 1, None),
                          'coords': ((0.5984683885502095, -0.3414997156337019, -1.784376056988566),
                                     (0.5984683885502095, -0.3414997156337019, -0.26962807973179115),
                                     (-0.561782159992927, 0.352156496238436, 0.1861898853072379),
                                     (-0.4953109671543722, 0.3121949238324531, 1.6400480667719612),
                                     (0.5984683885502095, 0.6833389047425913, -2.170090678202741),
                                     (1.4752480877653946, -0.865673673309282, -2.174747572976487),
                                     (-0.30426791615275384, -0.8266450973522037, -2.170052121454076),
                                     (1.493680343317207, 0.16330192488518464, 0.10867040971512965),
                                     (0.5776301566835702, -1.369061185253102, 0.108547964789524),
                                     (-1.315645169318881, -0.19664279862292228, 1.770625802679126),
                                     (-1.4545596701784862, -0.22670306931712683, 0.7807777630403547))}
        self.assertTrue(almost_equal_coords(zmat_to_xyz(zmat_1_copy, keep_dummy=True), expected_xyz_1))

        zmat_2 = {'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                             ('R_5_2', 'A_5_2_1', 'D_5_2_1_4'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_2'),
                             ('R_7_2', 'A_7_2_1', 'D_7_2_1_6'), ('R_8_2', 'A_8_2_1', 'D_8_2_1_7'),
                             ('R_9_0', 'A_9_0_1', 'D_9_0_1_2'), ('R_10_0', 'A_10_0_1', 'D_10_0_1_2')),
                  'vars': {'R_1_0': 1.538597152766632, 'R_2_1': 1.538597152766632, 'A_2_1_0': 112.21956716174662,
                           'R_3_1': 1.1067875520527994, 'A_3_1_2': 109.48164208764113, 'D_3_1_2_0': 121.77882792718154,
                           'R_4_1': 1.1067875520527994, 'A_4_1_2': 109.48164208764113, 'D_4_1_2_3': 116.44234414563692,
                           'R_5_2': 1.325576275438579, 'A_5_2_1': 111.50202139070583, 'D_5_2_1_4': 58.22117207281846,
                           'R_6_0': 1.1046468961988158, 'A_6_0_1': 111.50202139070583, 'D_6_0_1_2': 180.0,
                           'R_7_2': 1.1059615583516615, 'A_7_2_1': 110.8283376252455, 'D_7_2_1_6': 300.2088700889345,
                           'R_8_2': 1.1059615583516615, 'A_8_2_1': 110.8283376252455, 'D_8_2_1_7': 119.58225745206313,
                           'R_9_0': 1.1059615583516615, 'A_9_0_1': 110.8283376252455, 'D_9_0_1_2': 300.2088700889345,
                           'R_10_0': 1.1059615583516615, 'A_10_0_1': 110.8283376252455, 'D_10_0_1_2': 59.79112991106552},
                  'map': {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}}
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_2,
                                                             add_dummy=False,
                                                             h1=1,
                                                             a=5,
                                                             c=0,
                                                             d=1,
                                                             )
        self.assertEqual(param_a2, 'A_11_2_5')
        self.assertEqual(param_d2, 'D_11_2_5_1')
        self.assertEqual(param_d3, 'D_12_11_2_5')

    def test_get_modified_params_from_zmat_2(self):
        """Test the get_modified_params_from_zmat_2() function."""
        zmat_1 = {'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'X'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                             ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_1', 'A_4_1_0', 'D_4_1_0_3'),
                             ('R_5_1', 'A_5_1_0', 'D_5_1_0_4'), ('R_6_1', 'A_6_1_0', 'D_6_1_0_5'),
                             ('R_7_0', 'A_7_0_1', 'D_7_0_1_6'), ('RX_8_7', 'AX_8_7_0', 'DX_8_7_0_1')),
                  'vars': {'R_1_0': 1.5120487296562577, 'R_2_0': 1.0940775789443724, 'A_2_0_1': 110.56801921096591,
                           'R_3_0': 1.0940817193677925, 'A_3_0_1': 110.56754686774481, 'D_3_0_1_2': 119.99910067703652,
                           'R_4_1': 1.0940725668318991, 'A_4_1_0': 110.56890700195424, 'D_4_1_0_3': 59.99971758419434,
                           'R_5_1': 1.0940840619688397, 'A_5_1_0': 110.56790845138725, 'D_5_1_0_4': 239.99905123159166,
                           'R_6_1': 1.0940817193677925, 'A_6_1_0': 110.56754686774481, 'D_6_1_0_5': 240.00122783407815,
                           'R_7_0': 1.3128870801982788, 'A_7_0_1': 110.56890700195424, 'D_7_0_1_6': 300.00028241580566,
                           'RX_8_7': 1.0, 'AX_8_7_0': 90.0, 'DX_8_7_0_1': 0},
                  'map': {0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 2, 8: 'X8'}}
        zmat_2 = {'symbols': ('H', 'O', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1'),
                             ('R_5_4', 'A_5_4_3', 'D_5_4_3_2'), ('R_6_4', 'A_6_4_3', 'D_6_4_3_5'),
                             ('R_7_4', 'A_7_4_3', 'D_7_4_3_6'), ('R_8_3', 'A_8_3_4', 'D_8_3_4_7'),
                             ('R_9_3', 'A_9_3_4', 'D_9_3_4_8')),
                  'vars': {'R_1_0': 1.1689469645782498, 'R_2_1': 1.4559254886404387, 'A_2_1_0': 96.30065819269021,
                           'R_3_2': 1.4265728986680748, 'A_3_2_1': 105.58023544826183, 'D_3_2_1_0': 242.3527063196313,
                           'R_4_3': 1.5147479951212197, 'A_4_3_2': 108.63387152978416, 'D_4_3_2_1': 179.9922243050821,
                           'R_5_4': 1.0950205915944824, 'A_5_4_3': 110.62463321031589, 'D_5_4_3_2': 59.1268942923763,
                           'R_6_4': 1.093567969297245, 'A_6_4_3': 110.91425998596507, 'D_6_4_3_5': 120.87266977773987,
                           'R_7_4': 1.0950091062890002, 'A_7_4_3': 110.62270362433773, 'D_7_4_3_6': 120.87301274044218,
                           'R_8_3': 1.0951433842986755, 'A_8_3_4': 110.20822115119915, 'D_8_3_4_7': 181.16392677464265,
                           'R_9_3': 1.0951410439636102, 'A_9_3_4': 110.20143800025897, 'D_9_3_4_8': 239.4199964284852},
                  'map': {0: 9, 1: 3, 2: 2, 3: 1, 4: 0, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}}
        new_symbols, new_coords, new_vars, new_map = \
            get_modified_params_from_zmat_2(zmat_1=zmat_1,
                                            zmat_2=zmat_2,
                                            reactant_2=ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=self.ccooj_xyz),
                                            add_dummy=True,
                                            glue_params=('A_9_7_8', 'D_9_7_8_0', 'D_10_9_7_0'),
                                            h1=2,
                                            a=0,
                                            c=1,
                                            a2=150,
                                            d2=0,
                                            d3=120,
                                            )
        expected_new_symbols = ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'X', 'O', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H')
        expected_new_coords = (
            (None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None), ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'),
            ('R_4_1', 'A_4_1_0', 'D_4_1_0_3'), ('R_5_1', 'A_5_1_0', 'D_5_1_0_4'), ('R_6_1', 'A_6_1_0', 'D_6_1_0_5'),
            ('R_7_0', 'A_7_0_1', 'D_7_0_1_6'), ('RX_8_7', 'AX_8_7_0', 'DX_8_7_0_1'), ('R_9_2', 'A_9_7_8', 'D_9_7_8_0'),
            ('R_10_9', 'A_10_9_0', 'D_10_9_7_0'), ('R_11_10', 'A_11_10_9', 'D_11_10_9_8'),
            ('R_12_11', 'A_12_11_10', 'D_12_11_10_9'), ('R_13_12', 'A_13_12_11', 'D_13_12_11_10'),
            ('R_14_12', 'A_14_12_11', 'D_14_12_11_13'), ('R_15_12', 'A_15_12_11', 'D_15_12_11_14'),
            ('R_16_11', 'A_16_11_12', 'D_16_11_12_15'), ('R_17_11', 'A_17_11_12', 'D_17_11_12_16'))
        expected_new_vars = {
            'R_1_0': 1.5120487296562577, 'R_2_0': 1.0940775789443724, 'A_2_0_1': 110.56801921096591,
            'R_3_0': 1.0940817193677925, 'A_3_0_1': 110.56754686774481, 'D_3_0_1_2': 119.99910067703652,
            'R_4_1': 1.0940725668318991, 'A_4_1_0': 110.56890700195424, 'D_4_1_0_3': 59.99971758419434,
            'R_5_1': 1.0940840619688397, 'A_5_1_0': 110.56790845138725, 'D_5_1_0_4': 239.99905123159166,
            'R_6_1': 1.0940817193677925, 'A_6_1_0': 110.56754686774481, 'D_6_1_0_5': 240.00122783407815,
            'R_7_0': 1.3128870801982788, 'A_7_0_1': 110.56890700195424, 'D_7_0_1_6': 300.00028241580566, 'RX_8_7': 1.0,
            'AX_8_7_0': 90.0, 'DX_8_7_0_1': 0, 'R_9_2': 1.1689469645782498, 'A_9_7_8': 240, 'D_9_7_8_0': 0,
            'R_10_9': 1.4559254886404387, 'A_10_9_0': 96.30065819269021, 'D_10_9_7_0': 120,
            'R_11_10': 1.4265728986680748, 'A_11_10_9': 105.58023544826183, 'D_11_10_9_8': 242.3527063196313,
            'R_12_11': 1.5147479951212197, 'A_12_11_10': 108.63387152978416, 'D_12_11_10_9': 179.9922243050821,
            'R_13_12': 1.0950205915944824, 'A_13_12_11': 110.62463321031589, 'D_13_12_11_10': 59.1268942923763,
            'R_14_12': 1.093567969297245, 'A_14_12_11': 110.91425998596507, 'D_14_12_11_13': 120.87266977773987,
            'R_15_12': 1.0950091062890002, 'A_15_12_11': 110.62270362433773, 'D_15_12_11_14': 120.87301274044218,
            'R_16_11': 1.0951433842986755, 'A_16_11_12': 110.20822115119915, 'D_16_11_12_15': 181.16392677464265,
            'R_17_11': 1.0951410439636102, 'A_17_11_12': 110.20143800025897, 'D_17_11_12_16': 239.4199964284852}
        expected_new_map = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 2, 8: 'X8', 9: 12,
                            10: 11, 11: 10, 12: 9, 13: 14, 14: 15, 15: 13, 16: 16, 17: 17}
        self.assertTrue(_compare_zmats({'symbols': new_symbols, 'coords': new_coords, 'vars': new_vars, 'map': new_map},
                                       {'symbols': expected_new_symbols, 'coords': expected_new_coords,
                                        'vars': expected_new_vars, 'map': expected_new_map},
                                       r_tol=0.01, a_tol=0.01, d_tol=0.01))

    def test_find_distant_neighbor(self):
        """Test the find_distant_neighbor() function."""
        xyz_1 = {'symbols': ('S', 'O', 'O', 'N', 'C', 'H', 'H', 'H', 'H', 'H'),
                 'isotopes': (32, 16, 16, 14, 12, 1, 1, 1, 1, 1),
                 'coords': ((-0.06618943, -0.12360663, -0.07631983),
                            (-0.79539707, 0.86755487, 1.02675668),
                            (-0.68919931, 0.25421823, -1.34830853),
                            (0.01546439, -1.54297548, 0.44580391),
                            (1.59721519, 0.47861334, 0.00711),
                            (1.94428095, 0.40772394, 1.03719428),
                            (2.20318015, -0.14715186, -0.64755729),
                            (1.59252246, 1.5117895, -0.33908352),
                            (-0.8785689, -2.02453514, 0.38494433),
                            (-1.34135876, 1.49608206, 0.53295071))}
        mol_1 = ARCSpecies(label='CS(=O)(O)[NH]', xyz=xyz_1).mol
        self.assertEqual(find_distant_neighbor(mol=mol_1, start=8), 0)
        self.assertEqual(find_distant_neighbor(mol=mol_1, start=9), 0)
        self.assertEqual(find_distant_neighbor(mol=mol_1, start=5), 0)
        self.assertIn(find_distant_neighbor(mol=mol_1, start=2), [1, 3, 4])
        self.assertIn(find_distant_neighbor(mol=mol_1, start=4), [1, 2, 3])
        self.assertIn(find_distant_neighbor(mol=mol_1, start=3), [1, 2, 4])
        self.assertIn(find_distant_neighbor(mol=mol_1, start=0), [5, 6, 7, 8, 9])

        xyz_2 = {'symbols': ('S', 'H', 'O', 'N', 'C', 'H', 'H', 'H', 'H'),
                 'isotopes': (32, 1, 16, 14, 12, 1, 1, 1, 1),
                 'coords': ((-0.06618943, -0.12360663, -0.07631983),
                            (-0.79539707, 0.86755487, 1.02675668),
                            (-0.68919931, 0.25421823, -1.34830853),
                            (0.01546439, -1.54297548, 0.44580391),
                            (1.59721519, 0.47861334, 0.00711),
                            (1.94428095, 0.40772394, 1.03719428),
                            (2.20318015, -0.14715186, -0.64755729),
                            (1.59252246, 1.5117895, -0.33908352),
                            (-0.8785689, -2.02453514, 0.38494433))}
        mol_2 = ARCSpecies(label='C[SH](=O)[NH]', xyz=xyz_2).mol
        self.assertEqual(find_distant_neighbor(mol=mol_2, start=8), 0)
        self.assertEqual(find_distant_neighbor(mol=mol_2, start=5), 0)
        self.assertIn(find_distant_neighbor(mol=mol_2, start=2), [3, 4])
        self.assertIn(find_distant_neighbor(mol=mol_2, start=4), [2, 3])
        self.assertIn(find_distant_neighbor(mol=mol_2, start=3), [2, 4])
        self.assertIn(find_distant_neighbor(mol=mol_2, start=0), [5, 6, 7, 8])

        xyz_3 = """C                 -2.27234259   -0.78101274   -0.00989219
                   H                 -1.94047502   -1.78971513   -0.14135849
                   H                 -1.87026865   -0.16873290   -0.78986011
                   H                 -3.34123881   -0.74952096   -0.04689398
                   C                 -1.79025824   -0.25578524    1.35514655
                   H                 -2.12212581    0.75291715    1.48661285
                   H                 -2.19233218   -0.86806508    2.13511447
                   N                 -0.32177464   -0.29904964    1.40598078
                   H                  0.05399540    0.27317450    0.67703880
                   H                 -0.00873286    0.04200717    2.29236958"""
        mol_3 = ARCSpecies(label='EA', smiles='NCC', xyz=xyz_3).mol
        self.assertEqual(find_distant_neighbor(mol=mol_3, start=9), 4)
        self.assertEqual(find_distant_neighbor(mol=mol_3, start=8), 4)
        self.assertIn(find_distant_neighbor(mol=mol_3, start=5), [0, 7])
        self.assertIn(find_distant_neighbor(mol=mol_3, start=6), [0, 7])
        self.assertEqual(find_distant_neighbor(mol=mol_3, start=1), 4)
        self.assertEqual(find_distant_neighbor(mol=mol_3, start=2), 4)
        self.assertEqual(find_distant_neighbor(mol=mol_3, start=2), 4)

    def test_are_h_abs_wells_reversed(self):
        """
        Test the are_h_abs_wells_reversed() function.
        The expected order is: R(*1)-H(*2) + R(*3)j <=> R(*1)j + R(*3)-H(*2)
        """
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='C2H6', smiles='CC'), ARCSpecies(label='OH', smiles='[OH]')],  # none are reversed
                            p_species=[ARCSpecies(label='C2H5', smiles='[CH2]C'), ARCSpecies(label='H2O', smiles='O')])
        rxn_2 = ARCReaction(r_species=[ARCSpecies(label='OH', smiles='[OH]'), ARCSpecies(label='C2H6', smiles='CC')],  # r reversed
                            p_species=[ARCSpecies(label='C2H5', smiles='[CH2]C'), ARCSpecies(label='H2O', smiles='O')])
        rxn_3 = ARCReaction(r_species=[ARCSpecies(label='C2H6', smiles='CC'), ARCSpecies(label='OH', smiles='[OH]')],  # p reversed
                            p_species=[ARCSpecies(label='H2O', smiles='O'), ARCSpecies(label='C2H5', smiles='[CH2]C')])
        rxn_4 = ARCReaction(r_species=[ARCSpecies(label='OH', smiles='[OH]'), ARCSpecies(label='C2H6', smiles='CC')],  # r and p reversed
                            p_species=[ARCSpecies(label='H2O', smiles='O'), ARCSpecies(label='C2H5', smiles='[CH2]C')])

        product_dicts = get_reaction_family_products(rxn=rxn_1,
                                                     rmg_family_set=[rxn_1.family],
                                                     consider_rmg_families=True,
                                                     consider_arc_families=False,
                                                     discover_own_reverse_rxns_in_reverse=False,
                                                     )
        r_reversed, p_reversed = are_h_abs_wells_reversed(rxn_1, product_dict=product_dicts[0])
        self.assertFalse(r_reversed)
        self.assertFalse(p_reversed)

        product_dicts = get_reaction_family_products(rxn=rxn_2,
                                                     rmg_family_set=[rxn_2.family],
                                                     consider_rmg_families=True,
                                                     consider_arc_families=False,
                                                     discover_own_reverse_rxns_in_reverse=False,
                                                     )
        r_reversed, p_reversed = are_h_abs_wells_reversed(rxn_2, product_dict=product_dicts[0])
        self.assertTrue(r_reversed)
        self.assertFalse(p_reversed)

        product_dicts = get_reaction_family_products(rxn=rxn_3,
                                                     rmg_family_set=[rxn_3.family],
                                                     consider_rmg_families=True,
                                                     consider_arc_families=False,
                                                     discover_own_reverse_rxns_in_reverse=False,
                                                     )
        r_reversed, p_reversed = are_h_abs_wells_reversed(rxn_3, product_dict=product_dicts[0])
        self.assertFalse(r_reversed)
        self.assertTrue(p_reversed)

        product_dicts = get_reaction_family_products(rxn=rxn_4,
                                                     rmg_family_set=[rxn_4.family],
                                                     consider_rmg_families=True,
                                                     consider_arc_families=False,
                                                     discover_own_reverse_rxns_in_reverse=False,
                                                     )
        r_reversed, p_reversed = are_h_abs_wells_reversed(rxn_4, product_dict=product_dicts[0])
        self.assertTrue(r_reversed)
        self.assertTrue(p_reversed)

        rxn_5 = ARCReaction(r_species=[ARCSpecies(label='H', smiles='[H]'), ARCSpecies(label='H2O', smiles='O')],  # r and p reversed
                            p_species=[ARCSpecies(label='H2', smiles='[H][H]'), ARCSpecies(label='OH', smiles='[OH]')])
        product_dicts = get_reaction_family_products(rxn=rxn_5,
                                                     rmg_family_set=[rxn_5.family],
                                                     consider_rmg_families=True,
                                                     consider_arc_families=False,
                                                     discover_own_reverse_rxns_in_reverse=False,
                                                     )
        r_reversed, p_reversed = are_h_abs_wells_reversed(rxn_5, product_dict=product_dicts[0])
        self.assertTrue(r_reversed)
        self.assertTrue(p_reversed)

        rxn_6 = ARCReaction(r_species=[ARCSpecies(label='CCCC(O)=O', smiles='CCCC(O)=O'), ARCSpecies(label='OH', smiles='[OH]')],  # none are reversed
                            p_species=[ARCSpecies(label='CCCC([O])=O', smiles='CCCC([O])=O'), ARCSpecies(label='H2O', smiles='O')])
        product_dicts = get_reaction_family_products(rxn=rxn_6,
                                                     rmg_family_set=[rxn_6.family],
                                                     consider_rmg_families=True,
                                                     consider_arc_families=False,
                                                     discover_own_reverse_rxns_in_reverse=False,
                                                     )
        r_reversed, p_reversed = are_h_abs_wells_reversed(rxn_6, product_dict=product_dicts[0])
        self.assertFalse(r_reversed)
        self.assertFalse(p_reversed)



    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'), ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics_1'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
