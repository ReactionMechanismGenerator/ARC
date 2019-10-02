#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.species.converter module
"""

import unittest

from rdkit.Chem import rdMolTransforms as rdMT, rdchem

from rmgpy.molecule.molecule import Molecule
from rmgpy.species import Species

import arc.species.converter as converter
from arc.common import almost_equal_coords_lists
from arc.species.species import ARCSpecies


class TestConverter(unittest.TestCase):
    """
    Contains unit tests for the converter module
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None

        cls.xyz1 = {'str': """C       0.00000000    0.00000000    0.00000000
H       0.63003260    0.63003260    0.63003260
H      -0.63003260   -0.63003260    0.63003260
H      -0.63003260    0.63003260   -0.63003260
H       0.63003260   -0.63003260   -0.63003260""",  # no line break at the end on purpose
                    'file': """5
test methane xyz conversion
C       0.00000000    0.00000000    0.00000000
H       0.63003260    0.63003260    0.63003260
H      -0.63003260   -0.63003260    0.63003260
H      -0.63003260    0.63003260   -0.63003260
H       0.63003260   -0.63003260   -0.63003260
""",
                    'dict': {'symbols': ('C', 'H', 'H', 'H', 'H'),
                             'isotopes': (12, 1, 1, 1, 1),
                             'coords': ((0.0, 0.0, 0.0),
                                        (0.6300326, 0.6300326, 0.6300326),
                                        (-0.6300326, -0.6300326, 0.6300326),
                                        (-0.6300326, 0.6300326, -0.6300326),
                                        (0.6300326, -0.6300326, -0.6300326))}
                    }

        cls.xyz2 = {'str': """S       1.02558264   -0.04344404   -0.07343859
O      -0.25448248    1.10710477    0.18359696
N      -1.30762173    0.15796567   -0.10489290
C      -0.49011438   -1.03704380    0.15365747
H      -0.64869950   -1.85796321   -0.54773423
H      -0.60359153   -1.37304859    1.18613964
H      -1.43009127    0.23517346   -1.11797908


""",  # extra line breaks added on purpose
                    'file': """7
test xyz2
S       1.02558264   -0.04344404   -0.07343859
O      -0.25448248    1.10710477    0.18359696
N      -1.30762173    0.15796567   -0.10489290
C      -0.49011438   -1.03704380    0.15365747
H      -0.64869950   -1.85796321   -0.54773423
H      -0.60359153   -1.37304859    1.18613964
H      -1.43009127    0.23517346   -1.11797908
""",
                    'dict': {'symbols': ('S', 'O', 'N', 'C', 'H', 'H', 'H'),
                             'isotopes': (32, 16, 14, 12, 1, 1, 1),
                             'coords': ((1.02558264, -0.04344404, -0.07343859),
                                        (-0.25448248, 1.10710477, 0.18359696),
                                        (-1.30762173, 0.15796567, -0.1048929),
                                        (-0.49011438, -1.0370438, 0.15365747),
                                        (-0.6486995, -1.85796321, -0.54773423),
                                        (-0.60359153, -1.37304859, 1.18613964),
                                        (-1.43009127, 0.23517346, -1.11797908))},
                    }

        cls.xyz3 = {'str': """O          -0.25448248    1.10710477    0.18359696
N          -1.30762173    0.15796567   -0.10489290
C(Iso=13)  -0.49011438   -1.03704380    0.15365747
H(Iso=2)   -0.64869950   -1.85796321   -0.54773423
H          -0.60359153   -1.37304859    1.18613964
S           1.02558264   -0.04344404   -0.07343859
H          -1.43009127    0.23517346   -1.11797908
""",  # one line break at the end on purpose
                    'dict': {'symbols': ('O', 'N', 'C', 'H', 'H', 'S', 'H'),
                             'isotopes': (16, 14, 13, 2, 1, 32, 1),
                             'coords': ((-0.25448248, 1.10710477, 0.18359696),
                                        (-1.30762173, 0.15796567, -0.1048929),
                                        (-0.49011438, -1.0370438, 0.15365747),
                                        (-0.6486995, -1.85796321, -0.54773423),
                                        (-0.60359153, -1.37304859, 1.18613964),
                                        (1.02558264, -0.04344404, -0.07343859),
                                        (-1.43009127, 0.23517346, -1.11797908))},
                    }

        cls.xyz4 = {'str': """B 0.0000000 0.0000000 0.0000000
Br 0.0000000 1.9155570 0.0000000
Br 1.6589210 -0.9577780 0.0000000
Br -1.6589210	-0.9577780	0.0000000
""",  # last line contains tabs
                    'dict': {'symbols': ('B', 'Br', 'Br', 'Br'),
                             'isotopes': (11, 79, 79, 79),
                             'coords': ((0.0, 0.0, 0.0),
                                        (0.0, 1.915557, 0.0),
                                        (1.658921, -0.957778, 0.0),
                                        (-1.658921, -0.957778, 0.0))},
                    }

        cls.xyz5 = {'str': """O       1.17464110   -0.15309781    0.00000000
N       0.06304988    0.35149648    0.00000000
C      -1.12708952   -0.11333971    0.00000000
H      -1.93800144    0.60171738    0.00000000
H      -1.29769464   -1.18742971    0.00000000""",
                    'dict': {'symbols': ('O', 'N', 'C', 'H', 'H'),
                             'isotopes': (16, 14, 12, 1, 1),
                             'coords': ((1.1746411, -0.15309781, 0.0),
                                        (0.06304988, 0.35149648, 0.0),
                                        (-1.12708952, -0.11333971, 0.0),
                                        (-1.93800144, 0.60171738, 0.0),
                                        (-1.29769464, -1.18742971, 0.0))},
                    }

        cls.xyz6 = {'str': """S      -0.06618943   -0.12360663   -0.07631983
O      -0.79539707    0.86755487    1.02675668
O      -0.68919931    0.25421823   -1.34830853
N       0.01546439   -1.54297548    0.44580391
C       1.59721519    0.47861334    0.00711000
H       1.94428095    0.40772394    1.03719428
H       2.20318015   -0.14715186   -0.64755729
H       1.59252246    1.51178950   -0.33908352
H      -0.87856890   -2.02453514    0.38494433
H      -1.34135876    1.49608206    0.53295071""",
                    'file': """10
test xyz6
S      -0.06618943   -0.12360663   -0.07631983
O      -0.79539707    0.86755487    1.02675668
O      -0.68919931    0.25421823   -1.34830853
N       0.01546439   -1.54297548    0.44580391
C       1.59721519    0.47861334    0.00711000
H       1.94428095    0.40772394    1.03719428
H       2.20318015   -0.14715186   -0.64755729
H       1.59252246    1.51178950   -0.33908352
H      -0.87856890   -2.02453514    0.38494433
H      -1.34135876    1.49608206    0.53295071
""",
                    'dict': {'symbols': ('S', 'O', 'O', 'N', 'C', 'H', 'H', 'H', 'H', 'H'),
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
                                        (-1.34135876, 1.49608206, 0.53295071))},
                    }

        cls.xyz7 = {'str': """O       2.64631000   -0.59546000    0.29327900
O       2.64275300    2.05718500   -0.72942300
C       1.71639100    1.97990400    0.33793200
C      -3.48200000    1.50082200    0.03091100
C      -3.85550400   -1.05695100   -0.03598300
C       3.23017500   -1.88003900    0.34527100
C      -2.91846400    0.11144600    0.02829400
C       0.76935400    0.80820200    0.23396500
C      -1.51123800   -0.09830700    0.09199100
C       1.28495500   -0.50051800    0.22531700
C      -0.59550400    0.98573400    0.16444900
C      -0.94480400   -1.39242500    0.08331900
C       0.42608700   -1.59172200    0.14650400
H       2.24536500    1.93452800    1.29979800
H       1.14735500    2.91082400    0.31665700
H      -3.24115200    2.03800800    0.95768700
H      -3.08546100    2.10616100   -0.79369800
H      -4.56858900    1.48636200   -0.06630800
H      -4.89652000   -0.73067200   -0.04282300
H      -3.69325500   -1.65970000   -0.93924100
H      -3.72742500   -1.73294900    0.81894100
H       3.02442400   -2.44854700   -0.56812500
H       4.30341500   -1.72127600    0.43646000
H       2.87318600   -2.44236600    1.21464900
H      -0.97434200    2.00182800    0.16800300
H      -1.58581300   -2.26344700    0.02264400
H       0.81122400   -2.60336100    0.13267800
H       3.16280800    1.25020800   -0.70346900
""",
                    'dict': {'symbols': ('O', 'O', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H',
                                         'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                             'isotopes': (16, 16, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 1,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                             'coords': ((2.64631, -0.59546, 0.293279), (2.642753, 2.057185, -0.729423),
                                        (1.716391, 1.979904, 0.337932), (-3.482, 1.500822, 0.030911),
                                        (-3.855504, -1.056951, -0.035983), (3.230175, -1.880039, 0.345271),
                                        (-2.918464, 0.111446, 0.028294), (0.769354, 0.808202, 0.233965),
                                        (-1.511238, -0.098307, 0.091991), (1.284955, -0.500518, 0.225317),
                                        (-0.595504, 0.985734, 0.164449), (-0.944804, -1.392425, 0.083319),
                                        (0.426087, -1.591722, 0.146504), (2.245365, 1.934528, 1.299798),
                                        (1.147355, 2.910824, 0.316657), (-3.241152, 2.038008, 0.957687),
                                        (-3.085461, 2.106161, -0.793698), (-4.568589, 1.486362, -0.066308),
                                        (-4.89652, -0.730672, -0.042823), (-3.693255, -1.6597, -0.939241),
                                        (-3.727425, -1.732949, 0.818941), (3.024424, -2.448547, -0.568125),
                                        (4.303415, -1.721276, 0.43646), (2.873186, -2.442366, 1.214649),
                                        (-0.974342, 2.001828, 0.168003), (-1.585813, -2.263447, 0.022644),
                                        (0.811224, -2.603361, 0.132678), (3.162808, 1.250208, -0.703469))}
                    }

        cls.xyz8 = {'str': """N      -1.1997440839    -0.1610052059     0.0274738287
H      -1.4016624407    -0.6229695533    -0.8487034080
H      -0.0000018759     1.2861082773     0.5926077870
N       0.0000008520     0.5651072858    -0.1124621525
H      -1.1294692206    -0.8709078271     0.7537518889
N       1.1997613019    -0.1609980472     0.0274604887
H       1.1294795781    -0.8708998550     0.7537444446
H       1.4015274689    -0.6230592706    -0.8487058662""",
                    'dict': {'symbols': ('N', 'H', 'H', 'N', 'H', 'N', 'H', 'H'),
                             'isotopes': (14, 1, 1, 14, 1, 14, 1, 1),
                             'coords': ((-1.1997440839, -0.1610052059, 0.0274738287),
                                        (-1.4016624407, -0.6229695533, -0.848703408),
                                        (-1.8759e-06, 1.2861082773, 0.592607787),
                                        (8.52e-07, 0.5651072858, -0.1124621525),
                                        (-1.1294692206, -0.8709078271, 0.7537518889),
                                        (1.1997613019, -0.1609980472, 0.0274604887),
                                        (1.1294795781, -0.870899855, 0.7537444446),
                                        (1.4015274689, -0.6230592706, -0.8487058662))}}

        cls.xyz9 = {'str': """O       3.13231900    0.76911100   -0.08086900
O       3.38743600   -2.11675900   -0.03858500
C      -2.36919300   -0.54695600    0.56682700
C      -3.15360600    0.17105900    1.66307400
C      -2.72802700   -2.02644500    0.45926800
C       2.33156000   -1.73423500   -0.92148100
C       3.65011300    2.04916900    0.27583500
C      -0.93121600   -0.18690000    0.42819300
C       1.35285800   -0.75515100   -0.30846400
C       1.79433800    0.52230200    0.09841000
C       0.01159300   -1.07956000   -0.13549700
C      -0.44828900    1.08210200    0.80429800
C       0.89316900    1.43644300    0.64990400
H      -2.89113500   -0.05394500   -0.49913900
H       2.74879900   -1.31147200   -1.84752800
H       1.80991500   -2.65831900   -1.18214800
H      -3.11220800    1.25882600    1.56763000
H      -4.20773200   -0.11655100    1.61916700
H      -2.76884700   -0.09784700    2.65693400
H      -2.29498600   -2.59841700    1.29217500
H      -3.81389700   -2.15150400    0.49848800
H      -2.38217200   -2.47865600   -0.47430600
H       3.52516600    2.24195700    1.34780100
H       4.71260700    2.01840000    0.03253700
H       3.16623600    2.84537400   -0.30166300
H      -0.30596000   -2.07000300   -0.44289400
H      -1.12238100    1.81600000    1.22939200
H       1.21751200    2.42129300    0.96452300
H       3.88922100   -1.31541600    0.16697100
O      -3.43304800    0.46172100   -1.53075600
O      -2.89487900    1.76177800   -1.59155700
H      -2.12457300    1.65249500   -2.17600500""",
                    'dict': {'symbols': ('O', 'O', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                                         'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                                         'H', 'H', 'H', 'O', 'O', 'H'),
                             'isotopes': (16, 16, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1, 16, 16, 1),
                             'coords': ((3.132319, 0.769111, -0.080869),
                                        (3.387436, -2.116759, -0.038585),
                                        (-2.369193, -0.546956, 0.566827),
                                        (-3.153606, 0.171059, 1.663074),
                                        (-2.728027, -2.026445, 0.459268),
                                        (2.33156, -1.734235, -0.921481),
                                        (3.650113, 2.049169, 0.275835),
                                        (-0.931216, -0.1869, 0.428193),
                                        (1.352858, -0.755151, -0.308464),
                                        (1.794338, 0.522302, 0.09841),
                                        (0.011593, -1.07956, -0.135497),
                                        (-0.448289, 1.082102, 0.804298),
                                        (0.893169, 1.436443, 0.649904),
                                        (-2.891135, -0.053945, -0.499139),
                                        (2.748799, -1.311472, -1.847528),
                                        (1.809915, -2.658319, -1.182148),
                                        (-3.112208, 1.258826, 1.56763),
                                        (-4.207732, -0.116551, 1.619167),
                                        (-2.768847, -0.097847, 2.656934),
                                        (-2.294986, -2.598417, 1.292175),
                                        (-3.813897, -2.151504, 0.498488),
                                        (-2.382172, -2.478656, -0.474306),
                                        (3.525166, 2.241957, 1.347801),
                                        (4.712607, 2.0184, 0.032537),
                                        (3.166236, 2.845374, -0.301663),
                                        (-0.30596, -2.070003, -0.442894),
                                        (-1.122381, 1.816, 1.229392),
                                        (1.217512, 2.421293, 0.964523),
                                        (3.889221, -1.315416, 0.166971),
                                        (-3.433048, 0.461721, -1.530756),
                                        (-2.894879, 1.761778, -1.591557),
                                        (-2.124573, 1.652495, -2.176005))},
                    'gaussian': """
              1          8           0        3.132319    0.769111   -0.080869
              2          8           0        3.387436   -2.116759   -0.038585
              3          6           0       -2.369193   -0.546956    0.566827
              4          6           0       -3.153606    0.171059    1.663074
              5          6           0       -2.728027   -2.026445    0.459268
              6          6           0        2.331560   -1.734235   -0.921481
              7          6           0        3.650113    2.049169    0.275835
              8          6           0       -0.931216   -0.186900    0.428193
              9          6           0        1.352858   -0.755151   -0.308464
             10          6           0        1.794338    0.522302    0.098410
             11          6           0        0.011593   -1.079560   -0.135497
             12          6           0       -0.448289    1.082102    0.804298
             13          6           0        0.893169    1.436443    0.649904
             14          1           0       -2.891135   -0.053945   -0.499139
             15          1           0        2.748799   -1.311472   -1.847528
             16          1           0        1.809915   -2.658319   -1.182148
             17          1           0       -3.112208    1.258826    1.567630
             18          1           0       -4.207732   -0.116551    1.619167
             19          1           0       -2.768847   -0.097847    2.656934
             20          1           0       -2.294986   -2.598417    1.292175
             21          1           0       -3.813897   -2.151504    0.498488
             22          1           0       -2.382172   -2.478656   -0.474306
             23          1           0        3.525166    2.241957    1.347801
             24          1           0        4.712607    2.018400    0.032537
             25          1           0        3.166236    2.845374   -0.301663
             26          1           0       -0.305960   -2.070003   -0.442894
             27          1           0       -1.122381    1.816000    1.229392
             28          1           0        1.217512    2.421293    0.964523
             29          1           0        3.889221   -1.315416    0.166971
             30          8           0       -3.433048    0.461721   -1.530756
             31          8           0       -2.894879    1.761778   -1.591557
             32          1           0       -2.124573    1.652495   -2.176005

        """,
                    }

        nh_s_adj = """1 N u0 p2 c0 {2,S}
                          2 H u0 p0 c0 {1,S}"""
        nh_s_xyz = """N       0.50949998    0.00000000    0.00000000
                          H      -0.50949998    0.00000000    0.00000000"""
        cls.spc1 = ARCSpecies(label='NH2(S)', adjlist=nh_s_adj, xyz=nh_s_xyz, multiplicity=1, charge=0)
        spc = Species().from_adjacency_list(nh_s_adj)
        cls.spc2 = ARCSpecies(label='NH2(S)', rmg_species=spc, xyz=nh_s_xyz)

        cls.spc3 = ARCSpecies(label='NCN(S)', smiles='[N]=C=[N]', multiplicity=1, charge=0)

        cls.spc4 = ARCSpecies(label='NCN(T)', smiles='[N]=C=[N]', multiplicity=3, charge=0)

    def test_str_to_xyz(self):
        """Test converting a string xyz format to the ARC xyz format"""
        xyz1 = converter.str_to_xyz(xyz_str=self.xyz1['str'])
        xyz2 = converter.str_to_xyz(xyz_str=self.xyz2['str'])
        xyz3 = converter.str_to_xyz(xyz_str=self.xyz3['str'])
        xyz4 = converter.str_to_xyz(xyz_str=self.xyz4['str'])
        xyz9a = converter.str_to_xyz(xyz_str=self.xyz9['str'])
        xyz9b = converter.str_to_xyz(xyz_str=self.xyz9['gaussian'])  # check parsing a Gaussian output format

        self.assertEqual(xyz1, self.xyz1['dict'])
        self.assertEqual(xyz2, self.xyz2['dict'])
        self.assertEqual(xyz3, self.xyz3['dict'])
        self.assertEqual(xyz4, self.xyz4['dict'])
        self.assertEqual(xyz9a, self.xyz9['dict'])
        self.assertEqual(xyz9b, self.xyz9['dict'])

    def test_xyz_to_str(self):
        """Test converting an ARC xyz format to a string xyz format"""
        xyz_str1 = converter.xyz_to_str(xyz_dict=self.xyz1['dict'])
        xyz_str2 = converter.xyz_to_str(xyz_dict=self.xyz2['dict'])
        xyz_str3 = converter.xyz_to_str(xyz_dict=self.xyz3['dict'])
        xyz_str4 = converter.xyz_to_str(xyz_dict=self.xyz4['dict'])
        self.assertEqual(xyz_str1, converter.standardize_xyz_string(self.xyz1['str']))
        self.assertEqual(xyz_str2, converter.standardize_xyz_string(self.xyz2['str']))
        self.assertEqual(xyz_str3, converter.standardize_xyz_string(self.xyz3['str']))
        self.assertEqual(xyz_str4, converter.standardize_xyz_string(self.xyz4['str']))

    def test_xyz_to_x_y_z(self):
        """Test the xyz_to_x_y_z function"""
        x, y, z = converter.xyz_to_x_y_z(self.xyz1['dict'])
        self.assertEqual(x, (0.0, 0.6300326, -0.6300326, -0.6300326, 0.6300326))
        self.assertEqual(y, (0.0, 0.6300326, -0.6300326, 0.6300326, -0.6300326))
        self.assertEqual(z, (0.0, 0.6300326, 0.6300326, -0.6300326, -0.6300326))

    def test_xyz_to_xyz_file_format(self):
        """Test generating the XYZ file format from the xyz dictionary"""
        xyzf1 = converter.xyz_to_xyz_file_format(xyz_dict=self.xyz1['dict'], comment='test methane xyz conversion')
        xyzf2 = converter.xyz_to_xyz_file_format(xyz_dict=self.xyz2['dict'], comment='test xyz2')
        xyzf6 = converter.xyz_to_xyz_file_format(xyz_dict=self.xyz6['dict'], comment='test xyz6')
        self.assertEqual(xyzf1, self.xyz1['file'])
        self.assertEqual(xyzf2, self.xyz2['file'])
        self.assertEqual(xyzf6, self.xyz6['file'])

    def test_xyz_file_format_to_xyz(self):
        """Test getting the ARC xyz dictionary from an xyz file format"""
        xyz1 = converter.xyz_file_format_to_xyz(xyz_file=self.xyz1['file'])
        xyz2 = converter.xyz_file_format_to_xyz(xyz_file=self.xyz2['file'])
        xyz6 = converter.xyz_file_format_to_xyz(xyz_file=self.xyz6['file'])
        self.assertEqual(xyz1, self.xyz1['dict'])
        self.assertEqual(xyz2, self.xyz2['dict'])
        self.assertEqual(xyz6, self.xyz6['dict'])

    def test_xyz_from_data(self):
        """Test getting the ARC xyz dictionary from data"""
        symbols = ('C', 'H', 'H', 'H', 'H')
        isotopes = (12, 1, 1, 1, 1)
        coords = ((0.0, 0.0, 0.0),
                  (0.6300326, 0.6300326, 0.6300326),
                  (-0.6300326, -0.6300326, 0.6300326),
                  (-0.6300326, 0.6300326, -0.6300326),
                  (0.6300326, -0.6300326, -0.6300326))
        xyz_dict0 = converter.xyz_from_data(coords=coords, symbols=symbols, isotopes=isotopes)
        self.assertEqual(xyz_dict0, self.xyz1['dict'])
        xyz_dict1 = converter.xyz_from_data(coords=coords, symbols=symbols)  # no specifying isotopes
        self.assertEqual(xyz_dict1, self.xyz1['dict'])

        numbers = [6, 1, 1, 1, 1]
        coords = [[0.0, 0.0, 0.0],
                  [0.6300326, 0.6300326, 0.6300326],
                  [-0.6300326, -0.6300326, 0.6300326],
                  [-0.6300326, 0.6300326, -0.6300326],
                  [0.6300326, -0.6300326, -0.6300326]]
        xyz_dict2 = converter.xyz_from_data(coords=coords, numbers=numbers)
        self.assertEqual(xyz_dict2, self.xyz1['dict'])

    def test_get_most_common_isotope_for_element(self):
        """Test the get_most_common_isotope_for_element function"""
        common_isotopes = list()
        common_isotopes.append(converter.get_most_common_isotope_for_element('H'))
        common_isotopes.append(converter.get_most_common_isotope_for_element('B'))
        common_isotopes.append(converter.get_most_common_isotope_for_element('C'))
        common_isotopes.append(converter.get_most_common_isotope_for_element('Zn'))
        common_isotopes.append(converter.get_most_common_isotope_for_element('U'))
        common_isotopes.append(converter.get_most_common_isotope_for_element('Og'))
        self.assertEqual(common_isotopes, [1, 11, 12, 64, 238, 294])

    def test_standardize_xyz_string(self):
        """Test the standardize_xyz_string() function"""
        xyz = """
        
        
 C                 -0.67567701    1.18507660    0.04672449
 H                 -0.25592948    1.62415961    0.92757746
 H                 -2.26870864    1.38030564    0.05865317
 O                 -0.36671999   -0.21081064    0.01630374
 H                 -0.73553821   -0.63718986    0.79332805
 C                 -0.08400571    1.86907236   -1.19973252
 
 H                 -0.50375517    1.42998100   -2.08057962
 H                 -0.31518819    2.91354759   -1.17697025
 H                  0.97802159    1.73893214   -1.20769117
 O                 -3.69788377    1.55609096    0.07050345
 O                 -4.28667752    0.37487691    0.04916102
 H                 -4.01978712   -0.12970163    0.82103635
 
 """
        expected_xyz = """C                 -0.67567701    1.18507660    0.04672449
H                 -0.25592948    1.62415961    0.92757746
H                 -2.26870864    1.38030564    0.05865317
O                 -0.36671999   -0.21081064    0.01630374
H                 -0.73553821   -0.63718986    0.79332805
C                 -0.08400571    1.86907236   -1.19973252
H                 -0.50375517    1.42998100   -2.08057962
H                 -0.31518819    2.91354759   -1.17697025
H                  0.97802159    1.73893214   -1.20769117
O                 -3.69788377    1.55609096    0.07050345
O                 -4.28667752    0.37487691    0.04916102
H                 -4.01978712   -0.12970163    0.82103635"""
        new_xyz = converter.standardize_xyz_string(xyz)
        self.assertEqual(new_xyz, converter.standardize_xyz_string(expected_xyz))

    def test_xyz_to_pybel_mol(self):
        """Test xyz conversion into Open Babel"""
        pbmol1 = converter.xyz_to_pybel_mol(self.xyz1['dict'])
        pbmol2 = converter.xyz_to_pybel_mol(self.xyz5['dict'])
        pbmol3 = converter.xyz_to_pybel_mol(self.xyz2['dict'])
        pbmol4 = converter.xyz_to_pybel_mol(self.xyz6['dict'])

        # These tests check that the atoms we expect appear in the correct order:

        self.assertEqual(pbmol1.atoms[0].idx, 1)  # C
        self.assertAlmostEqual(pbmol1.atoms[0].atomicmass, 12.0107, 2)
        self.assertEqual(pbmol1.atoms[0].coords, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(pbmol1.atoms[1].atomicmass, 1.00794, 2)  # H
        self.assertEqual(pbmol1.atoms[1].coords, (0.6300326, 0.6300326, 0.6300326))

        self.assertAlmostEqual(pbmol2.atoms[0].atomicmass, 15.9994, 2)  # O
        self.assertEqual(pbmol2.atoms[0].coords, (1.1746411, -0.15309781, 0.0))
        self.assertAlmostEqual(pbmol2.atoms[1].atomicmass, 14.0067, 2)  # N
        self.assertEqual(pbmol2.atoms[1].coords, (0.06304988, 0.35149648, 0.0))
        self.assertAlmostEqual(pbmol2.atoms[2].atomicmass, 12.0107, 2)  # C
        self.assertEqual(pbmol2.atoms[2].coords, (-1.12708952, -0.11333971, 0.0))
        self.assertAlmostEqual(pbmol2.atoms[3].atomicmass, 1.00794, 2)  # H
        self.assertEqual(pbmol2.atoms[3].coords, (-1.93800144, 0.60171738, 0.0))

        self.assertAlmostEqual(pbmol3.atoms[0].atomicmass, 32.065, 2)  # S
        self.assertEqual(pbmol3.atoms[0].coords, (1.02558264, -0.04344404, -0.07343859))
        self.assertAlmostEqual(pbmol3.atoms[1].atomicmass, 15.9994, 2)  # O
        self.assertEqual(pbmol3.atoms[1].coords, (-0.25448248, 1.10710477, 0.18359696))
        self.assertAlmostEqual(pbmol3.atoms[2].atomicmass, 14.0067, 2)  # N
        self.assertEqual(pbmol3.atoms[2].coords, (-1.30762173, 0.15796567, -0.1048929))
        self.assertAlmostEqual(pbmol3.atoms[3].atomicmass, 12.0107, 2)  # C
        self.assertEqual(pbmol3.atoms[3].coords, (-0.49011438, -1.0370438, 0.15365747))
        self.assertAlmostEqual(pbmol3.atoms[-1].atomicmass, 1.00794, 2)  # H
        self.assertEqual(pbmol3.atoms[-1].coords, (-1.43009127, 0.23517346, -1.11797908))

        self.assertAlmostEqual(pbmol4.atoms[0].atomicmass, 32.065, 2)  # S
        self.assertEqual(pbmol4.atoms[0].coords, (-0.06618943, -0.12360663, -0.07631983))
        self.assertAlmostEqual(pbmol4.atoms[3].atomicmass, 14.0067, 2)  # N
        self.assertEqual(pbmol4.atoms[3].coords, (0.01546439, -1.54297548, 0.44580391))

    def test_pybel_to_inchi(self):
        """Tests the conversion of Open Babel molecules to InChI"""
        pbmol1 = converter.xyz_to_pybel_mol(self.xyz1['dict'])
        pbmol2 = converter.xyz_to_pybel_mol(self.xyz5['dict'])
        pbmol3 = converter.xyz_to_pybel_mol(self.xyz2['dict'])
        pbmol4 = converter.xyz_to_pybel_mol(self.xyz6['dict'])

        inchi1 = converter.pybel_to_inchi(pbmol1)
        inchi2 = converter.pybel_to_inchi(pbmol2)
        inchi3 = converter.pybel_to_inchi(pbmol3)
        inchi4 = converter.pybel_to_inchi(pbmol4)

        self.assertEqual(inchi1, 'InChI=1/CH4/h1H4')
        self.assertEqual(inchi2, 'InChI=1/CH2NO/c1-2-3/h1H2')
        self.assertEqual(inchi3, 'InChI=1/CH3NOS/c1-2-3-4-1/h2H,1H2')
        self.assertEqual(inchi4, 'InChI=1/CH5NO2S/c1-5(2,3)4/h2-3H,1H3')

    def test_rmg_mol_from_inchi(self):
        """Test generating RMG Molecule objects from InChI's"""
        mol1 = converter.rmg_mol_from_inchi('InChI=1S/NO2/c2-1-3')
        mol2 = converter.rmg_mol_from_inchi('InChI=1/CH3NOS/c1-2-3-4-1/h2H,1H2')

        self.assertTrue(isinstance(mol1, Molecule))
        self.assertTrue(isinstance(mol2, Molecule))

        smi1 = mol1.to_smiles()
        smi2 = mol2.to_smiles()

        self.assertEqual(smi1, '[O]N=O')
        self.assertEqual(smi2, 'C1NOS1')

    def test_elementize(self):
        """Test converting an RMG:Atom's atom type to its elemental atom type"""
        mol = Molecule(smiles='O=C=O')
        atom1 = mol.atoms[0]
        atom2 = mol.atoms[1]
        self.assertEqual(atom1.atomtype.label, 'O2d')
        self.assertEqual(atom2.atomtype.label, 'Cdd')
        converter.elementize(atom1)
        converter.elementize(atom2)
        self.assertEqual(atom1.atomtype.label, 'O')
        self.assertEqual(atom2.atomtype.label, 'C')

    def test_molecules_from_xyz(self):
        """Tests that atom orders are preserved when converting xyz's into RMG Molecules"""
        s_mol, b_mol = converter.molecules_from_xyz(self.xyz6['dict'])

        # check that the atom order is the same
        self.assertTrue(s_mol.atoms[0].is_sulfur())
        self.assertTrue(b_mol.atoms[0].is_sulfur())
        self.assertTrue(s_mol.atoms[1].is_oxygen())
        self.assertTrue(b_mol.atoms[1].is_oxygen())
        self.assertTrue(s_mol.atoms[2].is_oxygen())
        self.assertTrue(b_mol.atoms[2].is_oxygen())
        self.assertTrue(s_mol.atoms[3].is_nitrogen())
        self.assertTrue(b_mol.atoms[3].is_nitrogen())
        self.assertTrue(s_mol.atoms[4].is_carbon())
        self.assertTrue(b_mol.atoms[4].is_carbon())
        self.assertTrue(s_mol.atoms[5].is_hydrogen())
        self.assertTrue(b_mol.atoms[5].is_hydrogen())
        self.assertTrue(s_mol.atoms[6].is_hydrogen())
        self.assertTrue(b_mol.atoms[6].is_hydrogen())
        self.assertTrue(s_mol.atoms[7].is_hydrogen())
        self.assertTrue(b_mol.atoms[7].is_hydrogen())
        self.assertTrue(s_mol.atoms[8].is_hydrogen())
        self.assertTrue(b_mol.atoms[8].is_hydrogen())
        self.assertTrue(s_mol.atoms[9].is_hydrogen())
        self.assertTrue(b_mol.atoms[9].is_hydrogen())

        s_mol, b_mol = converter.molecules_from_xyz(self.xyz7['dict'])
        self.assertTrue(s_mol.atoms[0].is_oxygen())
        self.assertTrue(b_mol.atoms[0].is_oxygen())
        self.assertTrue(s_mol.atoms[2].is_carbon())
        self.assertTrue(b_mol.atoms[2].is_carbon())

        expected_bonded_adjlist = """multiplicity 2
1  O u0 p2 c0 {6,S} {10,S}
2  O u0 p2 c0 {3,S} {28,S}
3  C u0 p0 c0 {2,S} {8,S} {14,S} {15,S}
4  C u0 p0 c0 {7,S} {16,S} {17,S} {18,S}
5  C u0 p0 c0 {7,S} {19,S} {20,S} {21,S}
6  C u0 p0 c0 {1,S} {22,S} {23,S} {24,S}
7  C u1 p0 c0 {4,S} {5,S} {9,S}
8  C u0 p0 c0 {3,S} {10,D} {11,S}
9  C u0 p0 c0 {7,S} {11,D} {12,S}
10 C u0 p0 c0 {1,S} {8,D} {13,S}
11 C u0 p0 c0 {8,S} {9,D} {25,S}
12 C u0 p0 c0 {9,S} {13,D} {26,S}
13 C u0 p0 c0 {10,S} {12,D} {27,S}
14 H u0 p0 c0 {3,S}
15 H u0 p0 c0 {3,S}
16 H u0 p0 c0 {4,S}
17 H u0 p0 c0 {4,S}
18 H u0 p0 c0 {4,S}
19 H u0 p0 c0 {5,S}
20 H u0 p0 c0 {5,S}
21 H u0 p0 c0 {5,S}
22 H u0 p0 c0 {6,S}
23 H u0 p0 c0 {6,S}
24 H u0 p0 c0 {6,S}
25 H u0 p0 c0 {11,S}
26 H u0 p0 c0 {12,S}
27 H u0 p0 c0 {13,S}
28 H u0 p0 c0 {2,S}
"""
        expected_mol = Molecule().from_adjacency_list(expected_bonded_adjlist)
        self.assertEqual(b_mol.to_adjacency_list(), expected_bonded_adjlist)
        # the isIsomorphic test must come after the adjlist test since it changes the atom order
        self.assertTrue(b_mol.is_isomorphic(expected_mol))

    def test_unsorted_xyz_mol_from_xyz(self):
        """Test atom order conservation when xyz isn't sorted with heavy atoms first"""
        n3h5 = ARCSpecies(label='N3H5', xyz=self.xyz8['str'], smiles='NNN')
        expected_adjlist = """1 N u0 p1 c0 {2,S} {4,S} {5,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {4,S}
4 N u0 p1 c0 {1,S} {3,S} {6,S}
5 H u0 p0 c0 {1,S}
6 N u0 p1 c0 {4,S} {7,S} {8,S}
7 H u0 p0 c0 {6,S}
8 H u0 p0 c0 {6,S}
"""
        self.assertEqual(n3h5.mol.to_adjacency_list(), expected_adjlist)
        self.assertEqual(n3h5.conformers[0], self.xyz8['dict'])

    def test_xyz_to_smiles(self):
        """Test xyz to SMILES conversion and inferring correct bond orders"""
        xyz1 = """S      -0.06618943   -0.12360663   -0.07631983
O      -0.79539707    0.86755487    1.02675668
O      -0.68919931    0.25421823   -1.34830853
N       0.01546439   -1.54297548    0.44580391
C       1.59721519    0.47861334    0.00711000
H       1.94428095    0.40772394    1.03719428
H       2.20318015   -0.14715186   -0.64755729
H       1.59252246    1.51178950   -0.33908352
H      -0.87856890   -2.02453514    0.38494433
H      -1.34135876    1.49608206    0.53295071"""

        xyz2 = """O       2.64631000   -0.59546000    0.29327900
O       2.64275300    2.05718500   -0.72942300
C       1.71639100    1.97990400    0.33793200
C      -3.48200000    1.50082200    0.03091100
C      -3.85550400   -1.05695100   -0.03598300
C       3.23017500   -1.88003900    0.34527100
C      -2.91846400    0.11144600    0.02829400
C       0.76935400    0.80820200    0.23396500
C      -1.51123800   -0.09830700    0.09199100
C       1.28495500   -0.50051800    0.22531700
C      -0.59550400    0.98573400    0.16444900
C      -0.94480400   -1.39242500    0.08331900
C       0.42608700   -1.59172200    0.14650400
H       2.24536500    1.93452800    1.29979800
H       1.14735500    2.91082400    0.31665700
H      -3.24115200    2.03800800    0.95768700
H      -3.08546100    2.10616100   -0.79369800
H      -4.56858900    1.48636200   -0.06630800
H      -4.89652000   -0.73067200   -0.04282300
H      -3.69325500   -1.65970000   -0.93924100
H      -3.72742500   -1.73294900    0.81894100
H       3.02442400   -2.44854700   -0.56812500
H       4.30341500   -1.72127600    0.43646000
H       2.87318600   -2.44236600    1.21464900
H      -0.97434200    2.00182800    0.16800300
H      -1.58581300   -2.26344700    0.02264400
H       0.81122400   -2.60336100    0.13267800
H       3.16280800    1.25020800   -0.70346900"""

        xyz3 = """N       2.24690600   -0.00006500    0.11597700
C      -1.05654800    1.29155000   -0.02642500
C      -1.05661400   -1.29150400   -0.02650600
C      -0.30514100    0.00000200    0.00533200
C       1.08358900   -0.00003400    0.06558000
H      -0.39168300    2.15448600   -0.00132500
H      -1.67242600    1.35091400   -0.93175000
H      -1.74185400    1.35367700    0.82742800
H      -0.39187100   -2.15447800    0.00045500
H      -1.74341400   -1.35278100    0.82619100
H      -1.67091600   -1.35164600   -0.93286400"""

        xyz4 = """C      -0.86594600    0.19886100    2.37159000
C       0.48486900   -0.16232000    1.75422500
C       1.58322700    0.83707500    2.14923200
C       0.88213600   -1.51753600    2.17861400
N       1.17852900   -2.57013900    2.53313600
N       0.51051200   -0.21074800    0.26080100
N      -0.51042000    0.21074000   -0.26079600
C      -0.48479200    0.16232300   -1.75422300
C       0.86590400   -0.19926100   -2.37161200
C      -1.58344900   -0.83674100   -2.14921800
C      -0.88166600    1.51765700   -2.17859800
N      -1.17777100    2.57034900   -2.53309500
H      -1.16019200    1.20098300    2.05838400
H      -1.64220300   -0.50052400    2.05954500
H      -0.78054100    0.17214100    3.45935000
H       1.70120000    0.85267300    3.23368300
H       2.53492600    0.56708700    1.69019900
H       1.29214500    1.83331400    1.80886700
H       1.15987300   -1.20145600   -2.05838100
H       0.78046800   -0.17257000   -3.45937100
H       1.64236100    0.49992400   -2.05962300
H      -2.53504500   -0.56650600   -1.69011500
H      -1.70149200   -0.85224500   -3.23366300
H      -1.29263300   -1.83308300   -1.80892900"""

        xyz5 = """O       0.90973400   -0.03064000   -0.09605500
O       0.31656600   -0.00477100   -1.21127600
O       2.17315400   -0.03069900   -0.09349100"""

#         xyz6 = """S       0.38431300    0.05370100    0.00000000
# N      -1.13260000    0.07859900    0.00000000
# H       0.85151800   -1.28998600    0.00000000"""
#
#         xyz7 = """N       0.00000000    0.00000000    0.44654700
# N       0.00000000    0.00000000   -0.77510900
# H       0.86709400    0.00000000    1.02859700
# H      -0.86709400    0.00000000    1.02859700"""

        xyz8 = """N       0.00000000    0.00000000    0.65631400
C       0.00000000    0.00000000   -0.50136500
H       0.00000000    0.00000000   -1.57173600"""

#         xyz9 = """S      -0.00866000   -0.60254900    0.00000000
# N      -0.96878800    0.63275900    0.00000000
# N       1.01229100    0.58298500    0.00000000"""
#
#         xyz10 = """O      -0.79494500   -0.93969200    0.00000000
# O      -0.32753500    1.24003800    0.00000000
# O       1.28811400   -0.24729000    0.00000000
# N       0.14143500    0.11571500    0.00000000
# H      -1.65602000   -0.48026800    0.00000000"""
#
#         xyz11 = """O       1.64973000   -0.57433600    0.02610800
# O       0.49836300    1.28744800   -0.18806200
# N      -0.57621600   -0.65116600    0.24595200
# N      -1.78357200   -0.10211200   -0.14953800
# N       0.61460400    0.08152700   -0.00952700
# H      -0.42001200   -1.61494900   -0.03311600
# H      -1.72480300    0.33507600   -1.06884500
# H      -2.07362100    0.59363400    0.53038600"""

        xyz12 = """O       1.10621000    0.00000000   -0.13455300
O      -1.10621000    0.00000000   -0.13455300
N       0.00000000    0.00000000    0.33490500"""

#         xyz13 = """O      -0.37723000   -1.27051900    0.00000000
# N      -0.12115000   -0.04252600    0.00000000
# N      -0.95339100    0.91468300    0.00000000
# C       1.31648000    0.33217600    0.00000000
# H       1.76422500   -0.11051900   -0.89038300
# H       1.76422500   -0.11051900    0.89038300
# H       1.40045900    1.41618100    0.00000000
# H      -1.88127600    0.47189500    0.00000000"""

        xyz14 = """S      -0.12942800    0.11104800    0.22427200
O       0.98591500   -1.00752300   -0.31179100
O      -1.43956200   -0.44459900   -0.15048900
O       0.32982400    1.44755400   -0.21682700
H       1.85512700   -0.56879900   -0.36563700"""

        xyz15 = """N       1.11543700    0.11100500    0.00000000
N      -0.11982300   -0.03150800    0.00000000
N      -1.25716400    0.01530300    0.00000000
H       1.57747800   -0.80026300    0.00000000"""

        xyz16 = """O       1.21678000   -0.01490600    0.00000000
N       0.04560300    0.35628400    0.00000000
C      -1.08941100   -0.23907800    0.00000000
H      -1.97763400    0.37807800    0.00000000
H      -1.14592100   -1.32640500    0.00000000"""

        xyz17 = """S       0.00000000    0.00000000    0.18275300
O      -0.94981300   -0.83167500   -0.84628900
O       0.94981300    0.83167500   -0.84628900
O       0.80426500   -0.99804200    0.85548500
O      -0.80426500    0.99804200    0.85548500
H      -1.67833300   -0.25442300   -1.13658700
H       1.67833300    0.25442300   -1.13658700"""

        xyz18 = """S       0.00000000    0.00000000    0.12264300
O       1.45413200    0.00000000    0.12264300
O      -0.72706600    1.25931500    0.12264300
O      -0.72706600   -1.25931500    0.12264300"""

        xyz19 = """N       1.16672400    0.35870400   -0.00000400
N      -1.16670800    0.35879500   -0.00000400
C      -0.73775600   -0.89086600   -0.00000100
C       0.73767000   -0.89093000   -0.00000100
C       0.00005200    1.08477000   -0.00000500
H      -1.40657400   -1.74401100    0.00000000
H       1.40645000   -1.74411900    0.00000000
H       0.00009400    2.16788100   -0.00000700"""

        xyz20 = """C       3.09980400   -0.16068000    0.00000600
C       1.73521600    0.45534600   -0.00002200
C       0.55924400   -0.24765400   -0.00000300
C      -0.73300200    0.32890400   -0.00001600
C      -1.93406200   -0.42115800    0.00001300
C      -3.19432700    0.11090700    0.00000900
H       3.67991400    0.15199400   -0.87914100
H       3.67984100    0.15191400    0.87923000
H       3.04908000   -1.25419800   -0.00004300
H       1.68713300    1.54476700   -0.00005100
H      -0.81003200    1.41627100   -0.00004600
H      -1.83479400   -1.50747300    0.00004100
H       0.61489300   -1.33808300    0.00002500
H      -3.35410300    1.18597200   -0.00001700
H      -4.07566100   -0.52115800    0.00003300"""

        mol1 = converter.molecules_from_xyz(converter.str_to_xyz(xyz1))[1]
        mol2 = converter.molecules_from_xyz(converter.str_to_xyz(xyz2))[1]
        mol3 = converter.molecules_from_xyz(converter.str_to_xyz(xyz3))[1]
        mol4 = converter.molecules_from_xyz(converter.str_to_xyz(xyz4))[1]
        mol5 = converter.molecules_from_xyz(converter.str_to_xyz(xyz5))[1]
        # mol6 = converter.molecules_from_xyz(xyz6)[1]
        # mol7 = converter.molecules_from_xyz(xyz7)[1]
        mol8 = converter.molecules_from_xyz(converter.str_to_xyz(xyz8))[1]
        # mol9 = converter.molecules_from_xyz(xyz9)[1]
        # mol10 = converter.molecules_from_xyz(xyz10)[1]
        # mol11 = converter.molecules_from_xyz(xyz11)[1]
        mol12 = converter.molecules_from_xyz(converter.str_to_xyz(xyz12))[1]
        # mol13 = converter.molecules_from_xyz(xyz13)[1]
        mol14 = converter.molecules_from_xyz(converter.str_to_xyz(xyz14))[1]
        mol15 = converter.molecules_from_xyz(converter.str_to_xyz(xyz15))[1]
        mol16 = converter.molecules_from_xyz(converter.str_to_xyz(xyz16))[1]
        mol17 = converter.molecules_from_xyz(converter.str_to_xyz(xyz17))[1]
        mol18 = converter.molecules_from_xyz(converter.str_to_xyz(xyz18))[1]
        mol19 = converter.molecules_from_xyz(converter.str_to_xyz(xyz19))[1]
        mol20 = converter.molecules_from_xyz(converter.str_to_xyz(xyz20))[1]

        self.assertEqual(mol1.to_smiles(), '[NH-][S+](=O)(O)C')
        self.assertIn(mol2.to_smiles(), ['COC1=C(CO)C=C([C](C)C)C=C1', 'COC1C=CC(=CC=1CO)[C](C)C'])
        self.assertEqual(mol3.to_smiles(), '[N]=C=C(C)C')
        self.assertEqual(mol4.to_smiles(), 'N#CC(N=NC(C#N)(C)C)(C)C')
        self.assertEqual(mol5.to_smiles(), '[O-][O+]=O')
        # self.assertEqual(mol6.to_smiles(), 'N#S')  # gives '[N]S', multiplicity 3
        # self.assertEqual(mol7.to_smiles(), '[NH2+]=[N-]')  # gives '[N]N', multiplicity 3
        self.assertEqual(mol8.to_smiles(), 'C#N')
        # self.assertEqual(mol9.to_smiles(), '[N-]=[S+]#N')  # gives [N]S#N, multiplicity 3
        # self.assertEqual(mol10.to_smiles(), '[N+](=O)(O)[O-]')  # gives None
        # self.assertEqual(mol11.to_smiles(), 'N(N)[N+](=O)[O-]')  # gives None
        self.assertEqual(mol12.to_smiles(), '[O]N=O')
        # self.assertEqual(mol13.to_smiles(), 'C[N+]([NH-])=O')  # gives None
        self.assertEqual(mol14.to_smiles(), 'OS(=O)[O]')
        self.assertEqual(mol15.to_smiles(), '[N-]=[N+]=N')
        self.assertEqual(mol16.to_smiles(), '[O]N=C')
        self.assertEqual(mol17.to_smiles(), 'OS(=O)(=O)O')
        self.assertEqual(mol18.to_smiles(), 'O=S(=O)=O')
        self.assertEqual(mol19.to_adjacency_list(), """multiplicity 2
1 N u1 p1 c0 {4,S} {5,S}
2 N u0 p1 c0 {3,S} {5,D}
3 C u0 p0 c0 {2,S} {4,D} {6,S}
4 C u0 p0 c0 {1,S} {3,D} {7,S}
5 C u0 p0 c0 {1,S} {2,D} {8,S}
6 H u0 p0 c0 {3,S}
7 H u0 p0 c0 {4,S}
8 H u0 p0 c0 {5,S}
""")  # cannot read SMILES 'c1ncc[n]1' (but can generate them)
        self.assertEqual(mol20.to_smiles(), 'C=C[CH]C=CC')

    def test_to_rdkit_mol(self):
        """Test converting an RMG Molecule object to an RDKit Molecule object"""
        n3_xyz = """N      -1.1997440839    -0.1610052059     0.0274738287
        H      -1.4016624407    -0.6229695533    -0.8487034080
        H      -0.0000018759     1.2861082773     0.5926077870
        N       0.0000008520     0.5651072858    -0.1124621525
        H      -1.1294692206    -0.8709078271     0.7537518889
        N       1.1997613019    -0.1609980472     0.0274604887
        H       1.1294795781    -0.8708998550     0.7537444446
        H       1.4015274689    -0.6230592706    -0.8487058662"""
        spc1 = ARCSpecies(label='N3', xyz=n3_xyz, smiles='NNN')
        rdkitmol, rd_atom_indices = converter.to_rdkit_mol(spc1.mol)
        for atom, index in rd_atom_indices.items():
            if atom.symbol == 'N':
                self.assertIn(index, [0, 1, 2])
            else:
                self.assertIn(index, [3, 4, 5, 6, 7])
        self.assertIsInstance(rdkitmol, rdchem.Mol)

    def test_rdkit_conf_from_mol(self):
        """Test rdkit_conf_from_mol"""
        _, b_mol = converter.molecules_from_xyz(self.xyz5['dict'])
        conf, rd_mol, index_map = converter.rdkit_conf_from_mol(mol=b_mol, xyz=self.xyz5['dict'])
        self.assertTrue(conf.Is3D())
        self.assertEqual(rd_mol.GetNumAtoms(), 5)
        self.assertEqual(index_map, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})

    def test_s_bonds_mol_from_xyz(self):
        """Test creating a molecule with only single bonds from xyz"""
        xyz1 = converter.str_to_xyz("""S      -0.06618943   -0.12360663   -0.07631983
O      -0.79539707    0.86755487    1.02675668
O      -0.68919931    0.25421823   -1.34830853
N       0.01546439   -1.54297548    0.44580391
C       1.59721519    0.47861334    0.00711000
H       1.94428095    0.40772394    1.03719428
H       2.20318015   -0.14715186   -0.64755729
H       1.59252246    1.51178950   -0.33908352
H      -0.87856890   -2.02453514    0.38494433
H      -1.34135876    1.49608206    0.53295071""")

        xyz2 = converter.str_to_xyz("""O       2.64631000   -0.59546000    0.29327900
O       2.64275300    2.05718500   -0.72942300
C       1.71639100    1.97990400    0.33793200
C      -3.48200000    1.50082200    0.03091100
C      -3.85550400   -1.05695100   -0.03598300
C       3.23017500   -1.88003900    0.34527100
C      -2.91846400    0.11144600    0.02829400
C       0.76935400    0.80820200    0.23396500
C      -1.51123800   -0.09830700    0.09199100
C       1.28495500   -0.50051800    0.22531700
C      -0.59550400    0.98573400    0.16444900
C      -0.94480400   -1.39242500    0.08331900
C       0.42608700   -1.59172200    0.14650400
H       2.24536500    1.93452800    1.29979800
H       1.14735500    2.91082400    0.31665700
H      -3.24115200    2.03800800    0.95768700
H      -3.08546100    2.10616100   -0.79369800
H      -4.56858900    1.48636200   -0.06630800
H      -4.89652000   -0.73067200   -0.04282300
H      -3.69325500   -1.65970000   -0.93924100
H      -3.72742500   -1.73294900    0.81894100
H       3.02442400   -2.44854700   -0.56812500
H       4.30341500   -1.72127600    0.43646000
H       2.87318600   -2.44236600    1.21464900
H      -0.97434200    2.00182800    0.16800300
H      -1.58581300   -2.26344700    0.02264400
H       0.81122400   -2.60336100    0.13267800
H       3.16280800    1.25020800   -0.70346900""")

        xyz3 = converter.str_to_xyz("""N       2.24690600   -0.00006500    0.11597700
C      -1.05654800    1.29155000   -0.02642500
C      -1.05661400   -1.29150400   -0.02650600
C      -0.30514100    0.00000200    0.00533200
C       1.08358900   -0.00003400    0.06558000
H      -0.39168300    2.15448600   -0.00132500
H      -1.67242600    1.35091400   -0.93175000
H      -1.74185400    1.35367700    0.82742800
H      -0.39187100   -2.15447800    0.00045500
H      -1.74341400   -1.35278100    0.82619100
H      -1.67091600   -1.35164600   -0.93286400""")

        xyz4 = converter.str_to_xyz("""C      -0.86594600    0.19886100    2.37159000
C       0.48486900   -0.16232000    1.75422500
C       1.58322700    0.83707500    2.14923200
C       0.88213600   -1.51753600    2.17861400
N       1.17852900   -2.57013900    2.53313600
N       0.51051200   -0.21074800    0.26080100
N      -0.51042000    0.21074000   -0.26079600
C      -0.48479200    0.16232300   -1.75422300
C       0.86590400   -0.19926100   -2.37161200
C      -1.58344900   -0.83674100   -2.14921800
C      -0.88166600    1.51765700   -2.17859800
N      -1.17777100    2.57034900   -2.53309500
H      -1.16019200    1.20098300    2.05838400
H      -1.64220300   -0.50052400    2.05954500
H      -0.78054100    0.17214100    3.45935000
H       1.70120000    0.85267300    3.23368300
H       2.53492600    0.56708700    1.69019900
H       1.29214500    1.83331400    1.80886700
H       1.15987300   -1.20145600   -2.05838100
H       0.78046800   -0.17257000   -3.45937100
H       1.64236100    0.49992400   -2.05962300
H      -2.53504500   -0.56650600   -1.69011500
H      -1.70149200   -0.85224500   -3.23366300
H      -1.29263300   -1.83308300   -1.80892900""")

        xyz5 = converter.str_to_xyz("""O       0.90973400   -0.03064000   -0.09605500
O       0.31656600   -0.00477100   -1.21127600
O       2.17315400   -0.03069900   -0.09349100""")

        mol1 = converter.s_bonds_mol_from_xyz(xyz1)
        mol2 = converter.s_bonds_mol_from_xyz(xyz2)
        mol3 = converter.s_bonds_mol_from_xyz(xyz3)
        mol4 = converter.s_bonds_mol_from_xyz(xyz4)
        mol5 = converter.s_bonds_mol_from_xyz(xyz5)

        self.assertEqual(len(mol1.atoms), 10)
        self.assertEqual(len(mol2.atoms), 28)
        self.assertEqual(len(mol3.atoms), 11)
        self.assertEqual(len(mol4.atoms), 24)
        self.assertEqual(len(mol5.atoms), 3)

    def test_set_rdkit_dihedrals(self):
        """Test setting the dihedral angle of an RDKit molecule"""
        xyz0 = converter.str_to_xyz("""O       1.17961475   -0.92725986    0.15472373
C       0.45858928    0.27919340   -0.04589251
C      -1.02470597   -0.01894626    0.00226686
H       0.73480842    0.69726202   -1.01850832
H       0.73330833    0.98882191    0.74024781
H      -1.29861662   -0.45953441    0.96660817
H      -1.29713649   -0.74721756   -0.76877222
H      -1.61116041    0.89155300   -0.14917209
H       2.12529871   -0.70387223    0.11849858""")
        spc0 = ARCSpecies(label='CCO', smiles='CCO', xyz=xyz0)  # define with xyz for consistent atom order
        mol0 = spc0.mol

        torsion0 = (3, 2, 1, 9)  # the OH rotor
        new_dihedral = -60
        deg_increment = 240  # -180 + 240 = +60

        conf, rd_mol, index_map = converter.rdkit_conf_from_mol(mol0, xyz0)
        rd_tor_map = [index_map[i - 1] for i in torsion0]  # convert the atom indices in the torsion to RDKit indices
        new_xyz1 = converter.set_rdkit_dihedrals(conf, rd_mol, index_map, rd_tor_map, deg_abs=new_dihedral)

        conf, rd_mol, index_map = converter.rdkit_conf_from_mol(mol0, xyz0)  # convert again to init the conf object
        rd_tor_map = [index_map[i - 1] for i in torsion0]  # convert the atom indices in the torsion to RDKit indices
        new_xyz2 = converter.set_rdkit_dihedrals(conf, rd_mol, index_map, rd_tor_map, deg_increment=deg_increment)

        expected_xyz1 = """O       1.17961475   -0.92725986    0.15472373
C       0.45858928    0.27919340   -0.04589251
C      -1.02470597   -0.01894626    0.00226686
H       0.73480842    0.69726202   -1.01850832
H       0.73330833    0.98882191    0.74024781
H      -1.29861662   -0.45953441    0.96660817
H      -1.29713649   -0.74721756   -0.76877222
H      -1.61116041    0.89155300   -0.14917209
H       0.92345327   -1.27098714    1.02751540
"""
        expected_xyz2 = """O       1.17961475   -0.92725986    0.15472373
C       0.45858928    0.27919340   -0.04589251
C      -1.02470597   -0.01894626    0.00226686
H       0.73480842    0.69726202   -1.01850832
H       0.73330833    0.98882191    0.74024781
H      -1.29861662   -0.45953441    0.96660817
H      -1.29713649   -0.74721756   -0.76877222
H      -1.61116041    0.89155300   -0.14917209
H       0.92480849   -1.53430645   -0.56088835
"""

        self.assertTrue(almost_equal_coords_lists(new_xyz1, converter.str_to_xyz(expected_xyz1)))
        self.assertTrue(almost_equal_coords_lists(new_xyz2, converter.str_to_xyz(expected_xyz2)))

        xyz1 = converter.str_to_xyz("""N      -0.29070308    0.26322835    0.48770927
N       0.29070351   -0.26323281   -0.48771096
N      -2.61741263    1.38275080    2.63428181
N       2.61742270   -1.38276006   -2.63427425
C      -1.77086206    0.18100754    0.43957605
C       1.77086254   -0.18101028   -0.43957552
C      -2.22486176   -1.28143567    0.45202312
C      -2.30707039    0.92407663   -0.78734681
C       2.30707074   -0.92407071    0.78735246
C       2.22485929    1.28143406   -0.45203080
C      -2.23868798    0.85547218    1.67084736
C       2.23869247   -0.85548109   -1.67084185
H      -1.90398693   -1.81060764   -0.45229645
H      -3.31681639   -1.35858536    0.51240600
H      -1.80714051   -1.81980551    1.31137107
H      -3.40300863    0.95379538   -0.78701415
H      -1.98806037    0.44494681   -1.71978670
H      -1.94802915    1.96005927   -0.81269573
H       1.98805486   -0.44493850    1.71978893
H       1.94803425   -1.96005464    0.81270509
H       3.40300902   -0.95378386    0.78702431
H       1.90398036    1.81061002    0.45228426
H       3.31681405    1.35858667   -0.51241516
H       1.80713611    1.81979843   -1.31138136""")
        spc1 = ARCSpecies(label='AIBN', smiles='CC(C)(C#N)/N=N/C(C)(C)C#N', xyz=xyz1)
        mol1 = spc1.mol

        torsion1 = (1, 2, 6, 9)
        new_dihedral = 118.2

        conf, rd_mol, index_map = converter.rdkit_conf_from_mol(mol1, xyz1)
        rd_tor_map = [index_map[i - 1] for i in torsion1]  # convert the atom indices in the torsion to RDKit indices
        new_xyz3 = converter.set_rdkit_dihedrals(conf, rd_mol, index_map, rd_tor_map, deg_abs=new_dihedral)

        expected_xyz3 = """N      -0.29070308    0.26322835    0.48770927
N       0.29070351   -0.26323281   -0.48771096
N      -2.61741263    1.38275080    2.63428181
N       2.48573367    1.01638899   -2.68295766
C      -1.77086206    0.18100754    0.43957605
C       1.77086254   -0.18101028   -0.43957552
C      -2.22486176   -1.28143567    0.45202312
C      -2.30707039    0.92407663   -0.78734681
C       2.38216062   -1.58430507   -0.39387342
C       2.21983062    0.66527087    0.75509913
C      -2.23868798    0.85547218    1.67084736
C       2.16482620    0.49023713   -1.69815092
H      -1.90398693   -1.81060764   -0.45229645
H      -3.31681639   -1.35858536    0.51240600
H      -1.80714051   -1.81980551    1.31137107
H      -3.40300863    0.95379538   -0.78701415
H      -1.98806037    0.44494681   -1.71978670
H      -1.94802915    1.96005927   -0.81269573
H       2.11909310   -2.10839740    0.53181512
H       2.02775663   -2.19945525   -1.22981644
H       3.47613291   -1.54390687   -0.45350823
H       1.95308217    0.19222185    1.70685860
H       3.30593713    0.81467275    0.75113509
H       1.74954927    1.65592664    0.73932447
"""

        self.assertTrue(almost_equal_coords_lists(new_xyz3, converter.str_to_xyz(expected_xyz3)))

        rd_conf, rd_mol, index_map = converter.rdkit_conf_from_mol(mol1, converter.str_to_xyz(expected_xyz3))
        rd_scan = [index_map[i - 1] for i in torsion1]  # convert the atom indices to RDKit indices
        angle = rdMT.GetDihedralDeg(rd_conf, rd_scan[0], rd_scan[1], rd_scan[2], rd_scan[3])

        self.assertAlmostEqual(angle, 118.2, 5)

        xyz4 = """O       1.28706525    0.52121353    0.04219198
C       0.39745682   -0.35265044   -0.63649234
C      -0.98541845    0.26289370   -0.64801959
H       0.76016885   -0.50111637   -1.65799025
H       0.38478504   -1.31559717   -0.11722981
H      -0.96971239    1.23774091   -1.14654347
H      -1.69760597   -0.38642828   -1.16478035
H      -1.34010718    0.43408610    0.37373771
H       2.16336803    0.09985803    0.03295192"""
        spc4 = ARCSpecies(label='ethanol', smiles='CCO', xyz=xyz4)
        rd_conf, rd_mol, index_map = converter.rdkit_conf_from_mol(mol=spc4.mol, xyz=converter.str_to_xyz(xyz4))
        torsion4 = [9, 1, 2, 3]
        rd_tor_map = [index_map[i - 1] for i in torsion4]  # convert the atom indices to RDKit indices
        new_xyz4 = converter.set_rdkit_dihedrals(rd_conf, rd_mol, index_map, rd_tor_map, deg_abs=60)
        expected_xyz4 = """O       1.28706525    0.52121353    0.04219198
C       0.39745682   -0.35265044   -0.63649234
C       0.36441173   -1.68197093    0.08682400
H      -0.59818222    0.10068325   -0.65235399
H       0.74799641   -0.48357798   -1.66461710
H       0.03647269   -1.54932006    1.12314420
H      -0.31340646   -2.38081353   -0.41122551
H       1.36475837   -2.12581592    0.12433596
H       2.16336803    0.09985803    0.03295192"""
        self.assertEqual(converter.xyz_to_str(new_xyz4), expected_xyz4)

    def test_get_center_of_mass(self):
        """Test calculating the center of mass for coordinates"""
        xyz = """O       1.28706525    0.52121353    0.04219198
C       0.39745682   -0.35265044   -0.63649234
C       0.36441173   -1.68197093    0.08682400
H      -0.59818222    0.10068325   -0.65235399
H       0.74799641   -0.48357798   -1.66461710
H       0.03647269   -1.54932006    1.12314420
H      -0.31340646   -2.38081353   -0.41122551
H       1.36475837   -2.12581592    0.12433596
H       2.16336803    0.09985803    0.03295192
"""
        cm_x, cm_y, cm_z = converter.get_center_of_mass(xyz=converter.str_to_xyz(xyz))
        self.assertAlmostEqual(cm_x, 0.7201, 3)
        self.assertAlmostEqual(cm_y, -0.4880, 3)
        self.assertAlmostEqual(cm_z, -0.1603, 3)

        xyz = """C	1.1714680	-0.4048940	0.0000000
C	0.0000000	0.5602500	0.0000000
O	-1.1945070	-0.2236470	0.0000000
H	-1.9428910	0.3834580	0.0000000
H	2.1179810	0.1394450	0.0000000
H	1.1311780	-1.0413680	0.8846660
H	1.1311780	-1.0413680	-0.8846660
H	0.0448990	1.2084390	0.8852880
H	0.0448990	1.2084390	-0.8852880"""
        cm_x, cm_y, cm_z = converter.get_center_of_mass(xyz=converter.str_to_xyz(xyz))
        self.assertAlmostEqual(cm_x, -0.0540, 3)
        self.assertAlmostEqual(cm_y, -0.0184, 3)
        self.assertAlmostEqual(cm_z, -0.0000, 3)

        xyz = {'coords': ((0.0, 0.0, 0.113488),
                          (0.0, 0.93867, -0.264806),
                          (0.812912, -0.469335, -0.264806),
                          (-0.812912, -0.469335, -0.264806)),
               'symbols': ('N', 'H', 'H', 'H')}
        cm_x, cm_y, cm_z = converter.get_center_of_mass(converter.check_xyz_dict(xyz))
        self.assertAlmostEqual(cm_x, 0.0000, 3)
        self.assertAlmostEqual(cm_y, 0.0000, 3)
        self.assertAlmostEqual(cm_z, 0.0463, 3)

    def test_translate_to_center_of_mass(self):
        """Test calculating the center of mass for coordinates"""
        xyz = """O       1.28706525    0.52121353    0.04219198
C       0.39745682   -0.35265044   -0.63649234
C       0.36441173   -1.68197093    0.08682400
H      -0.59818222    0.10068325   -0.65235399
H       0.74799641   -0.48357798   -1.66461710
H       0.03647269   -1.54932006    1.12314420
H      -0.31340646   -2.38081353   -0.41122551
H       1.36475837   -2.12581592    0.12433596
H       2.16336803    0.09985803    0.03295192
"""
        translated_xyz = converter.translate_to_center_of_mass(converter.str_to_xyz(xyz))
        cm_x, cm_y, cm_z = converter.get_center_of_mass(xyz=translated_xyz)
        self.assertAlmostEqual(cm_x, 0.0000, 3)
        self.assertAlmostEqual(cm_y, 0.0000, 3)
        self.assertAlmostEqual(cm_z, 0.0000, 3)

        xyz = {'coords': ((0.0, 0.0, 0.113488),
                          (0.0, 0.93867, -0.264806),
                          (0.812912, -0.469335, -0.264806),
                          (-0.812912, -0.469335, -0.264806)),
               'symbols': ('N', 'H', 'H', 'H')}
        translated_xyz = converter.translate_to_center_of_mass(converter.check_xyz_dict(xyz))
        expected_xyz = """N       0.00000000    0.00000000    0.06717524
H       0.00000000    0.93867000   -0.31111876
H       0.81291200   -0.46933500   -0.31111876
H      -0.81291200   -0.46933500   -0.31111876"""
        self.assertEqual(converter.xyz_to_str(translated_xyz), expected_xyz)
        cm_x, cm_y, cm_z = converter.get_center_of_mass(translated_xyz)
        self.assertAlmostEqual(cm_x, 0.0000, 3)
        self.assertAlmostEqual(cm_y, 0.0000, 3)
        self.assertAlmostEqual(cm_z, 0.0000, 3)

    def set_radicals_correctly_from_xyz(self):
        """Test that we determine the number of radicals correctly from given xyz and multiplicity"""
        self.assertEqual(self.spc1.multiplicity, 1)  # NH(S), a nitrene
        self.assertTrue(all([atom.radical_electrons == 0 for atom in self.spc1.mol.atoms]))
        self.assertEqual(self.spc2.multiplicity, 1)  # NH(S), a nitrene
        self.assertTrue(all([atom.radical_electrons == 0 for atom in self.spc2.mol.atoms]))
        self.assertEqual(self.spc3.multiplicity, 1)  # NCN(S), a singlet birad
        self.assertTrue(all([atom.radical_electrons == 1 for atom in self.spc3.mol.atoms if atom.is_nitrogen()]))
        self.assertEqual(self.spc3.multiplicity, 3)  # NCN(T)
        self.assertTrue(all([atom.radical_electrons == 1 for atom in self.spc3.mol.atoms if atom.is_nitrogen()]))


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
