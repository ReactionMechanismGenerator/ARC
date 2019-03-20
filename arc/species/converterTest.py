#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests of the arc.species.converter module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest

from rmgpy.molecule.molecule import Molecule
from rmgpy.species import Species

import arc.species.converter as converter
from arc.species.species import ARCSpecies

################################################################################


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
        cls.xyz1 = """C       0.00000000    0.00000000    0.00000000
H       0.63003260    0.63003260    0.63003260
H      -0.63003260   -0.63003260    0.63003260
H      -0.63003260    0.63003260   -0.63003260
H       0.63003260   -0.63003260   -0.63003260"""

        cls.xyz2 = """O       1.17464110   -0.15309781    0.00000000
N       0.06304988    0.35149648    0.00000000
C      -1.12708952   -0.11333971    0.00000000
H      -1.93800144    0.60171738    0.00000000
H      -1.29769464   -1.18742971    0.00000000"""

        cls.xyz3 = """S       1.02558264   -0.04344404   -0.07343859
O      -0.25448248    1.10710477    0.18359696
N      -1.30762173    0.15796567   -0.10489290
C      -0.49011438   -1.03704380    0.15365747
H      -0.64869950   -1.85796321   -0.54773423
H      -0.60359153   -1.37304859    1.18613964
H      -1.43009127    0.23517346   -1.11797908
"""

        cls.xyz4 = """S      -0.06618943   -0.12360663   -0.07631983
O      -0.79539707    0.86755487    1.02675668
O      -0.68919931    0.25421823   -1.34830853
N       0.01546439   -1.54297548    0.44580391
C       1.59721519    0.47861334    0.00711000
H       1.94428095    0.40772394    1.03719428
H       2.20318015   -0.14715186   -0.64755729
H       1.59252246    1.51178950   -0.33908352
H      -0.87856890   -2.02453514    0.38494433
H      -1.34135876    1.49608206    0.53295071"""

        cls.xyz5 = """O       2.64631000   -0.59546000    0.29327900
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
"""

        cls.xyz6 = """N      -1.1997440839    -0.1610052059     0.0274738287
        H      -1.4016624407    -0.6229695533    -0.8487034080
        H      -0.0000018759     1.2861082773     0.5926077870
        N       0.0000008520     0.5651072858    -0.1124621525
        H      -1.1294692206    -0.8709078271     0.7537518889
        N       1.1997613019    -0.1609980472     0.0274604887
        H       1.1294795781    -0.8708998550     0.7537444446
        H       1.4015274689    -0.6230592706    -0.8487058662"""

        nh_s_adj = str("""1 N u0 p2 c0 {2,S}
                          2 H u0 p0 c0 {1,S}""")
        nh_s_xyz = str("""N       0.50949998    0.00000000    0.00000000
                          H      -0.50949998    0.00000000    0.00000000""")
        cls.spc1 = ARCSpecies(label=str('NH2(S)'), adjlist=nh_s_adj, xyz=nh_s_xyz, multiplicity=1, charge=0)
        spc = Species().fromAdjacencyList(nh_s_adj)
        cls.spc2 = ARCSpecies(label=str('NH2(S)'), rmg_species=spc, xyz=nh_s_xyz)

        cls.spc3 = ARCSpecies(label=str('NCN(S)'), smiles=str('[N]=C=[N]'), multiplicity=1, charge=0)

        cls.spc4 = ARCSpecies(label=str('NCN(T)'), smiles=str('[N]=C=[N]'), multiplicity=3, charge=0)

    def test_get_xyz_string(self):
        """Test conversion of xyz array to string format"""
        xyz_array = [[-0.06618943, -0.12360663, -0.07631983],
                     [-0.79539707, 0.86755487, 1.02675668],
                     [-0.68919931, 0.25421823, -1.34830853],
                     [0.01546439, -1.54297548, 0.44580391],
                     [1.59721519, 0.47861334, 0.00711],
                     [1.94428095, 0.40772394, 1.03719428],
                     [2.20318015, -0.14715186, -0.64755729],
                     [1.59252246, 1.5117895, -0.33908352],
                     [-0.8785689, -2.02453514, 0.38494433],
                     [-1.34135876, 1.49608206, 0.53295071]]
        symbols = ['S', 'O', 'O', 'N', 'C', 'H', 'H', 'H', 'H', 'H']
        xyz_expected = """S      -0.06618943   -0.12360663   -0.07631983
O      -0.79539707    0.86755487    1.02675668
O      -0.68919931    0.25421823   -1.34830853
N       0.01546439   -1.54297548    0.44580391
C       1.59721519    0.47861334    0.00711000
H       1.94428095    0.40772394    1.03719428
H       2.20318015   -0.14715186   -0.64755729
H       1.59252246    1.51178950   -0.33908352
H      -0.87856890   -2.02453514    0.38494433
H      -1.34135876    1.49608206    0.53295071
"""
        xyz1 = converter.get_xyz_string(xyz=xyz_array, symbol=symbols)
        self.assertEqual(xyz1, xyz_expected)
        number = [16, 8, 8, 7, 6, 1, 1, 1, 1, 1]
        xyz2 = converter.get_xyz_string(xyz=xyz_array, number=number)
        self.assertEqual(xyz2, xyz_expected)
        mol = Molecule().fromAdjacencyList(str("""1  S u0 p0 c0 {2,D} {3,S} {4,D} {5,S}
2  O u0 p2 c0 {1,D}
3  O u0 p2 c0 {1,S} {6,S}
4  N u0 p1 c0 {1,D} {7,S}
5  C u0 p0 c0 {1,S} {8,S} {9,S} {10,S}
6  H u0 p0 c0 {3,S}
7  H u0 p0 c0 {4,S}
8  H u0 p0 c0 {5,S}
9  H u0 p0 c0 {5,S}
10 H u0 p0 c0 {5,S}"""))
        xyz3 = converter.get_xyz_string(xyz=xyz_array, mol=mol)
        self.assertEqual(xyz3, xyz_expected)

    def test_get_xyz_matrix(self):
        """Test conversion of xyz string to array format"""
        xyz, symbols, x, y, z = converter.get_xyz_matrix(self.xyz2)
        self.assertEqual(xyz, [[1.1746411, -0.15309781, 0.0],
                               [0.06304988, 0.35149648, 0.0],
                               [-1.12708952, -0.11333971, 0.0],
                               [-1.93800144, 0.60171738, 0.0],
                               [-1.29769464, -1.18742971, 0.0]])
        self.assertEqual(symbols, ['O', 'N', 'C', 'H', 'H'])
        self.assertEqual(x, [1.1746411, 0.06304988, -1.12708952, -1.93800144, -1.29769464])
        self.assertEqual(y, [-0.15309781, 0.35149648, -0.11333971, 0.60171738, -1.18742971])
        self.assertEqual(z, [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_xyz_string_to_xyz_file_format(self):
        """Test generating the XYZ file format from xyz string format"""
        xyzf = converter.xyz_string_to_xyz_file_format(xyz=self.xyz1, comment='test methane xyz conversion')
        expected_xyzf = """5
test methane xyz conversion
C       0.00000000    0.00000000    0.00000000
H       0.63003260    0.63003260    0.63003260
H      -0.63003260   -0.63003260    0.63003260
H      -0.63003260    0.63003260   -0.63003260
H       0.63003260   -0.63003260   -0.63003260
"""
        self.assertEqual(xyzf, expected_xyzf)
        xyzf = converter.xyz_string_to_xyz_file_format(xyz=self.xyz3, comment='test xyz3')
        expected_xyzf = """7
test xyz3
S       1.02558264   -0.04344404   -0.07343859
O      -0.25448248    1.10710477    0.18359696
N      -1.30762173    0.15796567   -0.10489290
C      -0.49011438   -1.03704380    0.15365747
H      -0.64869950   -1.85796321   -0.54773423
H      -0.60359153   -1.37304859    1.18613964
H      -1.43009127    0.23517346   -1.11797908
"""
        self.assertEqual(xyzf, expected_xyzf)
        xyzf = converter.xyz_string_to_xyz_file_format(xyz=self.xyz4, comment='test xyz4')
        expected_xyzf = """10
test xyz4
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
"""
        self.assertEqual(xyzf, expected_xyzf)

    def test_standardize_xyz_string(self):
        """Test the standardize_xyz_string function"""
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
        expected_xyz = """ C                 -0.67567701    1.18507660    0.04672449
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
        self.assertEqual(new_xyz, expected_xyz)

        gaussian_format = """
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
     
"""

        expected_xyz = """O 3.132319 0.769111 -0.080869
O 3.387436 -2.116759 -0.038585
C -2.369193 -0.546956 0.566827
C -3.153606 0.171059 1.663074
C -2.728027 -2.026445 0.459268
C 2.331560 -1.734235 -0.921481
C 3.650113 2.049169 0.275835
C -0.931216 -0.186900 0.428193
C 1.352858 -0.755151 -0.308464
C 1.794338 0.522302 0.098410
C 0.011593 -1.079560 -0.135497
C -0.448289 1.082102 0.804298
C 0.893169 1.436443 0.649904
H -2.891135 -0.053945 -0.499139
H 2.748799 -1.311472 -1.847528
H 1.809915 -2.658319 -1.182148
H -3.112208 1.258826 1.567630
H -4.207732 -0.116551 1.619167
H -2.768847 -0.097847 2.656934
H -2.294986 -2.598417 1.292175
H -3.813897 -2.151504 0.498488
H -2.382172 -2.478656 -0.474306
H 3.525166 2.241957 1.347801
H 4.712607 2.018400 0.032537
H 3.166236 2.845374 -0.301663
H -0.305960 -2.070003 -0.442894
H -1.122381 1.816000 1.229392
H 1.217512 2.421293 0.964523
H 3.889221 -1.315416 0.166971
O -3.433048 0.461721 -1.530756
O -2.894879 1.761778 -1.591557
H -2.124573 1.652495 -2.176005"""
        new_xyz = converter.standardize_xyz_string(gaussian_format)
        self.assertEqual(new_xyz, expected_xyz)

    def test_xyz_to_pybel_mol(self):
        """Test xyz conversion into Open Babel"""
        pbmol1 = converter.xyz_to_pybel_mol(self.xyz1)
        pbmol2 = converter.xyz_to_pybel_mol(self.xyz2)
        pbmol3 = converter.xyz_to_pybel_mol(self.xyz3)
        pbmol4 = converter.xyz_to_pybel_mol(self.xyz4)

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
        pbmol1 = converter.xyz_to_pybel_mol(self.xyz1)
        pbmol2 = converter.xyz_to_pybel_mol(self.xyz2)
        pbmol3 = converter.xyz_to_pybel_mol(self.xyz3)
        pbmol4 = converter.xyz_to_pybel_mol(self.xyz4)

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

        smi1 = mol1.toSMILES()
        smi2 = mol2.toSMILES()

        self.assertEqual(smi1, '[O]N=O')
        self.assertEqual(smi2, 'C1NOS1')

    def test_elementize(self):
        """Test converting an RMG:Atom's atomType to its elemental atomType"""
        mol = Molecule(SMILES=str('O=C=O'))
        atom1 = mol.atoms[0]
        atom2 = mol.atoms[1]
        self.assertEqual(atom1.atomType.label, 'O2d')
        self.assertEqual(atom2.atomType.label, 'Cdd')
        converter.elementize(atom1)
        converter.elementize(atom2)
        self.assertEqual(atom1.atomType.label, 'O')
        self.assertEqual(atom2.atomType.label, 'C')

    def test_molecules_from_xyz(self):
        """Tests that atom orders are preserved when converting xyz's into RMG Molecules"""
        s_mol, b_mol = converter.molecules_from_xyz(self.xyz4)

        # check that the atom order is the same
        self.assertTrue(s_mol.atoms[0].isSulfur())
        self.assertTrue(b_mol.atoms[0].isSulfur())
        self.assertTrue(s_mol.atoms[1].isOxygen())
        self.assertTrue(b_mol.atoms[1].isOxygen())
        self.assertTrue(s_mol.atoms[2].isOxygen())
        self.assertTrue(b_mol.atoms[2].isOxygen())
        self.assertTrue(s_mol.atoms[3].isNitrogen())
        self.assertTrue(b_mol.atoms[3].isNitrogen())
        self.assertTrue(s_mol.atoms[4].isCarbon())
        self.assertTrue(b_mol.atoms[4].isCarbon())
        self.assertTrue(s_mol.atoms[5].isHydrogen())
        self.assertTrue(b_mol.atoms[5].isHydrogen())
        self.assertTrue(s_mol.atoms[6].isHydrogen())
        self.assertTrue(b_mol.atoms[6].isHydrogen())
        self.assertTrue(s_mol.atoms[7].isHydrogen())
        self.assertTrue(b_mol.atoms[7].isHydrogen())
        self.assertTrue(s_mol.atoms[8].isHydrogen())
        self.assertTrue(b_mol.atoms[8].isHydrogen())
        self.assertTrue(s_mol.atoms[9].isHydrogen())
        self.assertTrue(b_mol.atoms[9].isHydrogen())

        s_mol, b_mol = converter.molecules_from_xyz(self.xyz5)
        self.assertTrue(s_mol.atoms[0].isOxygen())
        self.assertTrue(b_mol.atoms[0].isOxygen())
        self.assertTrue(s_mol.atoms[2].isCarbon())
        self.assertTrue(b_mol.atoms[2].isCarbon())

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
        expected_mol = Molecule().fromAdjacencyList(str(expected_bonded_adjlist))
        self.assertEqual(b_mol.toAdjacencyList(), expected_bonded_adjlist)
        # the isIsomorphic test must come after the adjlist test since it changes the atom order
        self.assertTrue(b_mol.isIsomorphic(expected_mol))

    def test_unsorted_xyz_mol_from_xyz(self):
        """Test atom order conservation when xyz isn't sorted with heavy atoms first"""
        n3h5 = ARCSpecies(label=str('N3H5'), xyz=self.xyz6, smiles=str('NNN'))
        expected_adjlist = """1 N u0 p1 c0 {2,S} {4,S} {5,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {4,S}
4 N u0 p1 c0 {1,S} {3,S} {6,S}
5 H u0 p0 c0 {1,S}
6 N u0 p1 c0 {4,S} {7,S} {8,S}
7 H u0 p0 c0 {6,S}
8 H u0 p0 c0 {6,S}
"""
        self.assertEqual(n3h5.mol.toAdjacencyList(), expected_adjlist)
        self.assertEqual(n3h5.initial_xyz, self.xyz6)

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

        _, mol1 = converter.molecules_from_xyz(xyz1)
        _, mol2 = converter.molecules_from_xyz(xyz2)
        _, mol3 = converter.molecules_from_xyz(xyz3)
        _, mol4 = converter.molecules_from_xyz(xyz4)
        _, mol5 = converter.molecules_from_xyz(xyz5)
        # _, mol6 = converter.molecules_from_xyz(xyz6)
        # _, mol7 = converter.molecules_from_xyz(xyz7)
        _, mol8 = converter.molecules_from_xyz(xyz8)
        # _, mol9 = converter.molecules_from_xyz(xyz9)
        # _, mol10 = converter.molecules_from_xyz(xyz10)
        # _, mol11 = converter.molecules_from_xyz(xyz11)
        _, mol12 = converter.molecules_from_xyz(xyz12)
        # _, mol13 = converter.molecules_from_xyz(xyz13)
        _, mol14 = converter.molecules_from_xyz(xyz14)
        _, mol15 = converter.molecules_from_xyz(xyz15)
        _, mol16 = converter.molecules_from_xyz(xyz16)
        _, mol17 = converter.molecules_from_xyz(xyz17)
        _, mol18 = converter.molecules_from_xyz(xyz18)
        _, mol19 = converter.molecules_from_xyz(xyz19)
        _, mol20 = converter.molecules_from_xyz(xyz20)

        self.assertEqual(mol1.toSMILES(), '[NH-][S+](=O)(O)C')
        self.assertEqual(mol2.toSMILES(), 'COC1C=CC(=CC=1CO)[C](C)C')
        self.assertEqual(mol3.toSMILES(), '[N]=C=C(C)C')
        self.assertEqual(mol4.toSMILES(), 'N#CC(N=NC(C#N)(C)C)(C)C')
        self.assertEqual(mol5.toSMILES(), '[O-][O+]=O')
        # self.assertEqual(mol6.toSMILES(), 'N#S')  # gives '[N]S', multiplicity 3
        # self.assertEqual(mol7.toSMILES(), '[NH2+]=[N-]')  # gives '[N]N', multiplicity 3
        self.assertEqual(mol8.toSMILES(), 'C#N')
        # self.assertEqual(mol9.toSMILES(), '[N-]=[S+]#N')  # gives [N]S#N, multiplicity 3
        # self.assertEqual(mol10.toSMILES(), '[N+](=O)(O)[O-]')  # gives None
        # self.assertEqual(mol11.toSMILES(), 'N(N)[N+](=O)[O-]')  # gives None
        self.assertEqual(mol12.toSMILES(), '[O]N=O')
        # self.assertEqual(mol13.toSMILES(), 'C[N+]([NH-])=O')  # gives None
        self.assertEqual(mol14.toSMILES(), 'OS(=O)[O]')
        self.assertEqual(mol15.toSMILES(), '[N-]=[N+]=N')
        self.assertEqual(mol16.toSMILES(), '[O]N=C')
        self.assertEqual(mol17.toSMILES(), 'OS(=O)(=O)O')
        self.assertEqual(mol18.toSMILES(), 'O=S(=O)=O')
        self.assertEqual(mol19.toAdjacencyList(), """multiplicity 2
1 N u1 p1 c0 {4,S} {5,S}
2 N u0 p1 c0 {3,S} {5,D}
3 C u0 p0 c0 {2,S} {4,D} {6,S}
4 C u0 p0 c0 {1,S} {3,D} {7,S}
5 C u0 p0 c0 {1,S} {2,D} {8,S}
6 H u0 p0 c0 {3,S}
7 H u0 p0 c0 {4,S}
8 H u0 p0 c0 {5,S}
""")  # cannot read SMILES 'c1ncc[n]1' (but can generate them)
        self.assertEqual(mol20.toSMILES(), 'C=C[CH]C=CC')

    def test_rdkit_conf_from_mol(self):
        """Test rdkit_conf_from_mol"""
        _, b_mol = converter.molecules_from_xyz(self.xyz2)
        xyz, _, _, _, _ = converter.get_xyz_matrix(self.xyz2)
        conf, rd_mol, indx_map = converter.rdkit_conf_from_mol(mol=b_mol, coordinates=xyz)
        self.assertTrue(conf.Is3D())
        self.assertEqual(rd_mol.GetNumAtoms(), 5)
        self.assertEqual(indx_map, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})

    def test_s_bonds_mol_from_xyz(self):
        """Test creating a molecule with only single bonds from xyz"""
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

        mol1, _ = converter.s_bonds_mol_from_xyz(xyz1)
        mol2, _ = converter.s_bonds_mol_from_xyz(xyz2)
        mol3, _ = converter.s_bonds_mol_from_xyz(xyz3)
        mol4, _ = converter.s_bonds_mol_from_xyz(xyz4)
        mol5, _ = converter.s_bonds_mol_from_xyz(xyz5)

        self.assertEqual(len(mol1.atoms), 10)
        self.assertEqual(len(mol2.atoms), 28)
        self.assertEqual(len(mol3.atoms), 11)
        self.assertEqual(len(mol4.atoms), 24)
        self.assertEqual(len(mol5.atoms), 3)

    def set_radicals_correctly_from_xyz(self):
        """Test that we determine the number of radicals correctly from given xyz and multiplicity"""
        self.assertEqual(self.spc1.multiplicity, 1)  # NH(S), a nitrene
        self.assertTrue(all([atom.radicalElectrons == 0 for atom in self.spc1.mol.atoms]))
        self.assertEqual(self.spc2.multiplicity, 1)  # NH(S), a nitrene
        self.assertTrue(all([atom.radicalElectrons == 0 for atom in self.spc2.mol.atoms]))
        self.assertEqual(self.spc3.multiplicity, 1)  # NCN(S), a singlet birad
        self.assertTrue(all([atom.radicalElectrons == 1 for atom in self.spc3.mol.atoms if atom.isNitrogen()]))
        self.assertEqual(self.spc3.multiplicity, 3)  # NCN(T)
        self.assertTrue(all([atom.radicalElectrons == 1 for atom in self.spc3.mol.atoms if atom.isNitrogen()]))


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
