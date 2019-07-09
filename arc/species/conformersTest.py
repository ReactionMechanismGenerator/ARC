#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains unit tests of the arc.species.conformers module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import math

from rdkit.Chem import rdMolTransforms as rdMT

from rmgpy.molecule.molecule import Molecule
from rmgpy.molecule.atomtype import atomTypes
from rmgpy.molecule.group import GroupAtom, GroupBond, Group

import arc.species.converter as converter
import arc.species.conformers as conformers
from arc.species.species import ARCSpecies
from arc.arc_exceptions import ConformerError

################################################################################


class TestConformers(unittest.TestCase):
    """
    Contains unit tests for the conformers module
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        xyz1 = """O      -1.48027320    0.36597456    0.41386552
C      -0.49770656   -0.40253648   -0.26500019
C       0.86215119    0.24734211   -0.11510338
H      -0.77970114   -0.46128090   -1.32025907
H      -0.49643724   -1.41548311    0.14879346
H       0.84619526    1.26924854   -0.50799415
H       1.14377239    0.31659216    0.94076336
H       1.62810781   -0.32407050   -0.64676910
H      -1.22610851    0.40421362    1.35170355"""
        cls.spc0 = ARCSpecies(label='CCO', smiles='CCO', xyz=xyz1)  # define from xyz for consistent atom order
        cls.mol0 = cls.spc0.mol

        adj1 = """multiplicity 2
1 C u0 p0 c0 {2,S} {5,S} {6,S} {7,S}
2 C u0 p0 c0 {1,S} {3,D} {8,S}
3 C u0 p0 c0 {2,D} {4,S} {9,S}
4 O u1 p2 c0 {3,S}
5 H u0 p0 c0 {1,S}
6 H u0 p0 c0 {1,S}
7 H u0 p0 c0 {1,S}
8 H u0 p0 c0 {2,S}
9 H u0 p0 c0 {3,S}"""
        cls.mol1 = Molecule().fromAdjacencyList(str(adj1))

    def test_CONFS_VS_HEAVY_ATOMS(self):
        """Test that the CONFS_VS_HEAVY_ATOMS dictionary has 0 and 'inf' in its keys"""
        found_zero = False
        for key in conformers.CONFS_VS_HEAVY_ATOMS.keys():
            if key[0] == 0:
                found_zero = True
                break
        self.assertTrue(found_zero, 'The CONFS_VS_HEAVY_ATOMS dictionary has to have a key that srarts at zero. '
                                    'got:\n{0}'.format(conformers.CONFS_VS_HEAVY_ATOMS))
        found_inf = False
        for key in conformers.CONFS_VS_HEAVY_ATOMS.keys():
            if key[1] == 'inf':
                found_inf = True
                break
        self.assertTrue(found_inf, "The CONFS_VS_HEAVY_ATOMS dictionary has to have a key that ends with 'inf'. "
                                   "got:\n{0}".format(conformers.CONFS_VS_HEAVY_ATOMS))

    def test_CONFS_VS_TORSIONS(self):
        """Test that the CONFS_VS_TORSIONS dictionary has 0 and 'inf' in its keys"""
        found_zero = False
        for key in conformers.CONFS_VS_TORSIONS.keys():
            if key[0] == 0:
                found_zero = True
                break
        self.assertTrue(found_zero, 'The CONFS_VS_TORSIONS dictionary has to have a key that srarts at zero. '
                                    'got:\n{0}'.format(conformers.CONFS_VS_TORSIONS))
        found_inf = False
        for key in conformers.CONFS_VS_TORSIONS.keys():
            if key[1] == 'inf':
                found_inf = True
                break
        self.assertTrue(found_inf, "The CONFS_VS_TORSIONS dictionary has to have a key that ends with 'inf'. "
                                   "got:\n{0}".format(conformers.CONFS_VS_TORSIONS))

    def test_deduce_new_conformers(self):
        """Test deducing new conformers"""
        confs = [{'index': 0, 'xyz': 'O       1.49632323    0.74450682    0.93344565\nC      -0.27411342   '
                                     '-0.73369536    0.23273076\nC       1.00918754   -0.01585250   -0.16288919\n'
                                     'C      -1.40106678    0.23458846    0.56318695\nH      -0.08785698   -1.36178955'
                                     '    1.11163236\nH      -0.59069375   -1.38959579   -0.58564668\nH       '
                                     '0.83747022    0.65606977   -1.00993468\nH       1.77546073   -0.74556140   '
                                     '-0.44289977\nH      -1.14379759    0.86897309    1.41716858\nH      '
                                     '-2.31108163   -0.31832139    0.81688419\nH      -1.62427975    0.88145943   '
                                     '-0.29124479\nH       2.31444818    1.17921842    0.63815656\n',
                  'torsion_dihedrals': {(9, 4, 2, 3): -61.78942, (4, 2, 3, 1): 63.79634, (2, 3, 1, 12): 179.70585},
                  'source': 'MMFF94', 'FF energy': -1.5241875152610689},
                 {'index': 1, 'xyz': 'O       2.09496537   -0.68203123    0.41738811\nC      -0.17540789    0.11818414'
                                     '    0.51180976\nC       0.92511172   -0.46810337   -0.36086829\nC      '
                                     '-1.45486974    0.34772573   -0.27221056\nH      -0.37104415   -0.55290987    '
                                     '1.35664654\nH       0.16538089    1.06351120    0.95069762\nH       0.61854668   '
                                     '-1.43140600   -0.78023161\nH       1.17698645    0.20191002   -1.18924062\n'
                                     'H      -2.22790196    0.76917222    0.37791757\nH      -1.83476390   -0.59150616'
                                     '   -0.68674647\nH      -1.28838516    1.04591267   -1.09873240\nH       '
                                     '2.37138170    0.17954064    0.77357037\n',
                  'torsion_dihedrals': {(9, 4, 2, 3): -179.98511, (4, 2, 3, 1): -179.22951, (2, 3, 1, 12): -60.04260},
                  'source': 'MMFF94', 'FF energy': -1.641467160999287},
                 {'index': 2, 'xyz': 'O       1.68732977   -0.20570878    0.86043980\nC      -0.41954613    0.64790835'
                                     '    0.00757451\nC       1.08358067    0.51540154   -0.20395210\nC      '
                                     '-1.14589170   -0.68756266   -0.06383850\nH      -0.83158455    1.31816915   '
                                     '-0.75500120\nH      -0.61484441    1.09768535    0.98791309\nH       1.31520902'
                                     '    0.01348062   -1.14907531\nH       1.54143425    1.50913982   -0.22977912\n'
                                     'H      -0.95895886   -1.18493330   -1.02082479\nH      -0.83081898   -1.35799497'
                                     '    0.74145742\nH      -2.22543313   -0.53415060    0.03344438\nH       '
                                     '1.39952405   -1.13143453    0.79164182\n',
                  'torsion_dihedrals': {(9, 4, 2, 3): -57.02407, (4, 2, 3, 1): -66.21040, (2, 3, 1, 12): 69.65707},
                  'source': 'MMFF94', 'FF energy': -1.0563449757315282}]
        torsions = [[9, 4, 2, 3], [4, 2, 3, 1], [2, 3, 1, 12]]
        tops = [[4, 9, 10, 11], [3, 7, 8, 1, 12], [1, 12]]

        spc1 = ARCSpecies(label='propanol', smiles='CCCO', xyz=confs[0]['xyz'])
        base_xyz, multiple_tors, multiple_sampling_points, confs, torsion_angles, multiple_sampling_points_dict,\
            wells_dict, hypothetical_num_comb = conformers.deduce_new_conformers(
                label='', conformers=confs, torsions=torsions, tops=tops, mol_list=[spc1.mol], plot_path=None)

        new_conformers = conformers.generate_conformer_combinations(
            label='', mol=spc1.mol, base_xyz=base_xyz, hypothetical_num_comb=hypothetical_num_comb,
            multiple_tors=multiple_tors, multiple_sampling_points=multiple_sampling_points, len_conformers=len(confs),
            plot_path=None, torsion_angles=torsion_angles, multiple_sampling_points_dict=multiple_sampling_points_dict,
            wells_dict=wells_dict, force_field='MMFF94', max_combination_iterations=25, combination_threshold=10)

        self.assertEqual(len(new_conformers), 9)
        self.assertEqual(hypothetical_num_comb, 40)

        expected_new_conformers = [{'FF energy': -1.641,
  'dihedral': -179.99,
  'index': 3,
  'source': 'Changing dihedrals on most stable conformer, iteration 0',
  'torsion': (9, 4, 2, 3),
  'xyz': 'O       2.09496535   -0.68203129    0.41738813\nC      -0.17540789    0.11818414    0.51180976\nC       '
          '0.92511170   -0.46810343   -0.36086827\nC      -1.45486974    0.34772573   -0.27221056\nH      -0.37104416'
          '   -0.55290983    1.35664657\nH       0.16538092    1.06351121    0.95069758\nH       0.61854663   '
          '-1.43140607   -0.78023155\nH       1.17698644    0.20190992   -1.18924063\nH      -2.22790196    0.76917222'
          '    0.37791757\nH      -1.83476390   -0.59150616   -0.68674647\nH      -1.28838516    1.04591267   '
          '-1.09873240\nH       2.37138170    0.17954059    0.77357035\n'},
 {'FF energy': -1.641,
  'dihedral': -59.41,
  'index': 4,
  'source': 'Changing dihedrals on most stable conformer, iteration 0',
  'torsion': (9, 4, 2, 3),
  'xyz': 'O       1.56291282    1.15027309    1.82211062\nC      -0.17540789    0.11818414    0.51180976\nC       '
         '0.35814705    1.41001900    1.11424253\nC      -1.45486974    0.34772573   -0.27221056\nH       0.58674745   '
         '-0.32552850   -0.13982529\nH      -0.35452520   -0.61428924    1.30792692\nH       0.57826474    '
         '2.14318556    0.33205670\nH      -0.36298458    1.85454978    1.80772729\nH      -2.22790196    '
         '0.76917222    0.37791757\nH      -1.83476390   -0.59150616   -0.68674647\nH      -1.28838516    '
         '1.04591267   -1.09873240\nH       1.35734564    0.51103145    2.52545274\n'},
 {'FF energy': -1.405,
  'dihedral': -46.21,
  'index': 5,
  'source': 'Changing dihedrals on most stable conformer, iteration 0',
  'torsion': (4, 2, 3, 1),
  'xyz': 'O       0.40448892   -1.53634698   -1.14049206\nC      -0.17540789    0.11818414    0.51180976\nC       '
         '0.92511172   -0.46810337   -0.36086829\nC      -1.45486974    0.34772573   -0.27221056\nH      -0.37104415   '
         '-0.55290987    1.35664654\nH       0.16538089    1.06351120    0.95069762\nH       1.32280723    '
         '0.28669989   -1.04636811\nH       1.75277033   -0.85148848    0.24457090\nH      -2.22790196    '
         '0.76917222    0.37791757\nH      -1.83476390   -0.59150616   -0.68674647\nH      -1.28838516    1.04591267   '
         '-1.09873240\nH       0.06773060   -2.21044741   -0.52586997\n'},
 {'FF energy': -1.405,
  'dihedral': -16.21,
  'index': 6,
  'source': 'Changing dihedrals on most stable conformer, iteration 0',
  'torsion': (4, 2, 3, 1),
  'xyz': 'O       0.36484434   -0.98698526   -1.55956248\nC      -0.17540789    0.11818414    0.51180976\nC       '
         '0.92511172   -0.46810337   -0.36086829\nC      -1.45486974    0.34772573   -0.27221056\nH      -0.37104415   '
         '-0.55290987    1.35664654\nH       0.16538089    1.06351120    0.95069762\nH       1.65571980    '
         '0.29927577   -0.63498646\nH       1.45401410   -1.27688913    0.15360951\nH      -2.22790196    '
         '0.76917222    0.37791757\nH      -1.83476390   -0.59150616   -0.68674647\nH      -1.28838516    1.04591267   '
         '-1.09873240\nH      -0.27100848   -1.67908563   -1.31003062\n'},
 {'FF energy': -1.056,
  'dihedral': 13.79,
  'index': 7,
  'source': 'Changing dihedrals on most stable conformer, iteration 0',
  'torsion': (4, 2, 3, 1),
  'xyz': 'O       0.56872334   -0.34834741   -1.73150739\nC      -0.17540789    0.11818414    0.51180976\nC       '
         '0.92511172   -0.46810337   -0.36086829\nC      -1.45486974    0.34772573   -0.27221056\nH      -0.37104415   '
         '-0.55290987    1.35664654\nH       0.16538089    1.06351120    0.95069762\nH       1.86840135    '
         '0.06599283   -0.21005186\nH       1.09084503   -1.52676011   -0.13650716\nH      -2.22790196    '
         '0.76917222    0.37791757\nH      -1.83476390   -0.59150616   -0.68674647\nH      -1.28838516    1.04591267   '
         '-1.09873240\nH      -0.26099475   -0.83829374   -1.86226804\n'},
 {'FF energy': -1.056,
  'dihedral': 43.79,
  'index': 8,
  'source': 'Changing dihedrals on most stable conformer, iteration 0',
  'torsion': (4, 2, 3, 1),
  'xyz': 'O       0.96149673    0.20844408   -1.61025430\nC      -0.17540789    0.11818414    0.51180976\nC       '
         '0.92511172   -0.46810337   -0.36086829\nC      -1.45486974    0.34772573   -0.27221056\nH      -0.37104415   '
         '-0.55290987    1.35664654\nH       0.16538089    1.06351120    0.95069762\nH       1.90386405   '
         '-0.35064096    0.11457481\nH       0.76057398   -1.53414871   -0.54804258\nH      -2.22790196    '
         '0.76917222    0.37791757\nH      -1.83476390   -0.59150616   -0.68674647\nH      -1.28838516    1.04591267   '
         '-1.09873240\nH       0.09508862    0.08663874   -2.03461066\n'},
 {'FF energy': -1.641,
  'dihedral': 94.68,
  'index': 9,
  'source': 'Changing dihedrals on most stable conformer, iteration 0',
  'torsion': (2, 3, 1, 12),
  'xyz': 'O       2.09496537   -0.68203123    0.41738811\nC      -0.17540789    0.11818414    0.51180976\nC       '
         '0.92511172   -0.46810337   -0.36086829\nC      -1.45486974    0.34772573   -0.27221056\nH      -0.37104415   '
         '-0.55290987    1.35664654\nH       0.16538089    1.06351120    0.95069762\nH       0.61854668   '
         '-1.43140600   -0.78023161\nH       1.17698645    0.20191002   -1.18924062\nH      -2.22790196    '
         '0.76917222    0.37791757\nH      -1.83476390   -0.59150616   -0.68674647\nH      -1.28838516    1.04591267   '
         '-1.09873240\nH       2.07973614   -1.60561598    0.72125842\n'},
 {'FF energy': -1.81,
  'dihedral': 124.68,
  'index': 10,
  'source': 'Changing dihedrals on most stable conformer, iteration 0',
  'torsion': (2, 3, 1, 12),
  'xyz': 'O       2.09496537   -0.68203123    0.41738811\nC      -0.17540789    0.11818414    0.51180976\nC       '
         '0.92511172   -0.46810337   -0.36086829\nC      -1.45486974    0.34772573   -0.27221056\nH      -0.37104415   '
         '-0.55290987    1.35664654\nH       0.16538089    1.06351120    0.95069762\nH       0.61854668   '
         '-1.43140600   -0.78023161\nH       1.17698645    0.20191002   -1.18924062\nH      -2.22790196    '
         '0.76917222    0.37791757\nH      -1.83476390   -0.59150616   -0.68674647\nH      -1.28838516    1.04591267   '
         '-1.09873240\nH       2.34407173   -1.61701329    0.32078352\n'},
 {'FF energy': -1.81,
  'dihedral': 154.68,
  'index': 11,
  'source': 'Changing dihedrals on most stable conformer, iteration 0',
  'torsion': (2, 3, 1, 12),
  'xyz': 'O       2.09496537   -0.68203123    0.41738811\nC      -0.17540789    0.11818414    0.51180976\nC       '
         '0.92511172   -0.46810337   -0.36086829\nC      -1.45486974    0.34772573   -0.27221056\nH      -0.37104415   '
         '-0.55290987    1.35664654\nH       0.16538089    1.06351120    0.95069762\nH       0.61854668   '
         '-1.43140600   -0.78023161\nH       1.17698645    0.20191002   -1.18924062\nH      -2.22790196    '
         '0.76917222    0.37791757\nH      -1.83476390   -0.59150616   -0.68674647\nH      -1.28838516    1.04591267   '
         '-1.09873240\nH       2.60625305   -1.38969496   -0.01083476\n'}]

        self.assertEqual(new_conformers, expected_new_conformers)

    def test_get_force_field_energies(self):
        """Test attaining force field conformer energies"""
        xyzs, energies = conformers.get_force_field_energies(label='', mol=self.mol0, num_confs=10)
        self.assertEqual(len(xyzs), 10)
        self.assertEqual(len(energies), 10)
        xyz0 = converter.get_xyz_string(coords=xyzs[0], mol=self.mol0)
        mol0 = converter.molecules_from_xyz(xyz0)[1]
        self.assertTrue(self.mol0.isIsomorphic(mol0), 'Could not complete a round trip from Molecule to xyz and back '
                                                      'via RDKit')

        ch2oh_xyz = """O       0.83632835   -0.29575461    0.40459411
C      -0.43411393   -0.07778692   -0.05635829
H      -1.16221394   -0.80894238    0.24815765
H      -0.64965442    0.77699377   -0.67782845
H       1.40965394    0.40549015    0.08143497"""
        ch2oh_mol = Molecule(SMILES=str('[CH2]O'))
        energies = conformers.get_force_field_energies(label='', mol=ch2oh_mol, xyz=ch2oh_xyz, optimize=True)[1]
        self.assertAlmostEqual(energies[0], 13.466911, 5)

    def test_generate_force_field_conformers(self):
        """Test generating conformers from RDKit """
        mol_list = [self.mol0]
        label = 'ethanol'
        xyzs = ["""O       1.22700646   -0.74306134   -0.46642912
C       0.44275447    0.24386237    0.18670577
C      -1.02171998   -0.04112700   -0.06915927
H       0.65837353    0.21260997    1.25889575
H       0.71661793    1.22791280   -0.20534525
H      -1.29179957   -1.03551511    0.30144722
H      -1.23432965   -0.03370364   -1.14332936
H      -1.65581027    0.70264687    0.42135230
H       2.15890709   -0.53362491   -0.28413803""",
                """O       0.97434661    0.08786861   -0.03552502
C       2.39398906    0.09669513   -0.02681997
C       2.90028285   -0.91910243   -1.02843131
H       2.74173418   -0.15134085    0.98044759
H       2.74173583    1.10037467   -0.28899552
H       2.53391442   -1.92093297   -0.78104074
H       2.53391750   -0.68582304   -2.03364248
H       3.99353906   -0.93757558   -1.04664384
H       0.68104300    0.74807180    0.61546062"""]
        torsion_num = 2
        charge = 0
        multiplicity = 1

        confs = conformers.generate_force_field_conformers(mol_list=mol_list, label=label, torsion_num=torsion_num,
                                                           charge=charge, multiplicity=multiplicity, xyzs=xyzs,
                                                           num_confs=50)

        self.assertEqual(len(confs), 52)
        self.assertEqual(confs[0]['source'], 'MMFF94')
        self.assertEqual(confs[0]['index'], 0)
        self.assertEqual(confs[1]['index'], 1)
        self.assertEqual(confs[-1]['index'], 51)
        self.assertEqual(confs[-1]['source'], 'User Guess')
        self.assertFalse(any([confs[i]['xyz'] == confs[0]['xyz'] for i in range(1, 52)]))

    def test_determine_number_of_conformers_to_generate(self):
        """Test that the correct number of conformers to generate is determined"""
        self.assertEqual(conformers.determine_number_of_conformers_to_generate(heavy_atoms=0, torsion_num=0, label=''),
                         10)
        self.assertEqual(conformers.determine_number_of_conformers_to_generate(heavy_atoms=15, torsion_num=0, label=''),
                         500)
        self.assertEqual(conformers.determine_number_of_conformers_to_generate(heavy_atoms=5, torsion_num=31, label=''),
                         5000)
        self.assertEqual(conformers.determine_number_of_conformers_to_generate(heavy_atoms=150, torsion_num=0,
                                                                               label=''), 10000)

    def test_openbabel_force_field(self):
        """Test Open Babel force field"""
        xyz = """S      -0.19093478    0.57933906    0.00000000
O      -1.21746139   -0.72237602    0.00000000
O       1.40839617    0.14303696    0.00000000"""
        spc = ARCSpecies(label='SO2', smiles='O=S=O', xyz=xyz)
        xyzs, energies = conformers.openbabel_force_field(label='', mol=spc.mol, num_confs=1, force_field='GAFF',
                                                          return_xyz_strings=True, method='diverse')
        self.assertEqual(len(xyzs), 1)
        self.assertAlmostEqual(energies[0], 2.9310163, 3)

    def test_read_rdkit_embedded_conformers(self):
        """Test reading coordinates from embedded RDKit conformers"""
        xyz = """S      -0.19093478    0.57933906    0.00000000
O      -1.21746139   -0.72237602    0.00000000
O       1.40839617    0.14303696    0.00000000"""
        spc = ARCSpecies(label='SO2', smiles='O=S=O', xyz=xyz)
        rd_mol, rd_index_map = conformers.embed_rdkit(label='', mol=spc.mol, num_confs=3, xyz=xyz)
        xyzs = conformers.read_rdkit_embedded_conformers(label='', rd_mol=rd_mol, rd_index_map=rd_index_map,
                                                         mol=spc.mol, return_xyz_strings=False)
        expected_xyzs = [[[-0.0007230118849883151, 0.4313717594780365, -0.0],
                          [-1.1690509082499037, -0.21589003940054327, -0.0],
                          [1.169773920134892, -0.2154817200774931, 0.0]],
                         [[-0.0014867645051002793, 0.45116968803294244, -0.0],
                          [-1.1527224078813536, -0.2260261622412862, -0.0],
                          [1.154209172386454, -0.2251435257916561, 0.0]],
                         [[-0.0021414150293084812, 0.42384518006634525, -0.0],
                          [-1.176450577057819, -0.21250387455074374, -0.0],
                          [1.1785919920871275, -0.21134130551560104, 0.0]]]
        self.assertEqual(xyzs, expected_xyzs)

    def test_rdkit_force_field(self):
        """Test embedding molecule and applying force field using RDKit"""
        xyz = """S      -0.19093478    0.57933906    0.00000000
O      -1.21746139   -0.72237602    0.00000000
O       1.40839617    0.14303696    0.00000000"""
        spc = ARCSpecies(label='SO2', smiles='O=S=O', xyz=xyz)
        rd_mol, rd_index_map = conformers.embed_rdkit(label='', mol=spc.mol, num_confs=3, xyz=xyz)
        self.assertEqual(rd_index_map, {0: 0, 1: 1, 2: 2})
        xyzs, energies = conformers.rdkit_force_field(label='', rd_mol=rd_mol, rd_index_map=rd_index_map, mol=spc.mol,
                                                      force_field='MMFF94', return_xyz_strings=True, optimize=True)
        self.assertEqual(len(energies), 3)
        self.assertAlmostEqual(energies[0], 2.8820960262158292e-11, 5)
        self.assertAlmostEqual(energies[1], 4.496464369416183e-14, 5)
        self.assertAlmostEqual(energies[2], 1.8168786624672814e-12, 5)
        expected_xyzs1 = ['S      -0.04869777    0.60804465    0.00000000\nO      -1.35363403   -0.41438355    '
                          '0.00000000\nO       1.40233180   -0.19366111    0.00000000\n',
                          'S      -0.05319781    0.60766754    0.00000000\nO      -1.35052935   -0.42439346    '
                          '0.00000000\nO       1.40372716   -0.18327408    0.00000000\n',
                          'S       0.07722440    0.60508372    0.00000000\nO      -1.40988473   -0.12753188    '
                          '0.00000000\nO       1.33266033   -0.47755184    0.00000000\n']
        self.assertEqual(xyzs, expected_xyzs1)
        xyzs, energies = conformers.rdkit_force_field(label='', rd_mol=rd_mol, rd_index_map=rd_index_map, mol=spc.mol,
                                                      force_field='MMFF94', return_xyz_strings=False, optimize=False)
        self.assertEqual(len(energies), 0)
        expected_xyzs2 = [[[-0.048697770520464284, 0.6080446547953217, 0.0],
                           [-1.353634030198068, -0.41438354841733305, 0.0],
                           [1.402331800718534, -0.19366110637798933, 0.0]],
                          [[-0.05319780617176572, 0.607667542238931, 0.0],
                           [-1.3505293496269846, -0.42439346057066346, 0.0],
                           [1.4037271557987503, -0.18327408166827253, 0.0]],
                          [[0.07722439890127433, 0.6050837187112847, 0.0],
                           [-1.4098847335098195, -0.12753188106391833, 0.0],
                           [1.3326603346085464, -0.4775518376473646, 0.0]]]
        self.assertEqual(xyzs, expected_xyzs2)

    def test_determine_rotors(self):
        """Test determining the rotors"""
        mol = Molecule(SMILES=str('C=[C]C(=O)O[O]'))
        mol_list = mol.generate_resonance_structures()
        torsions, tops = conformers.determine_rotors(mol_list)
        self.assertEqual(torsions, [[3, 1, 4, 2], [1, 4, 6, 5]])
        self.assertEqual(sum(tops[0]), 4)
        self.assertEqual(sum(tops[1]), 10)

        mol_list = [Molecule(SMILES=str('CCCO'))]
        torsions, tops = conformers.determine_rotors(mol_list)
        self.assertEqual(torsions, [[5, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 12]])
        self.assertEqual(sum(tops[0]), 19)
        self.assertEqual(sum(tops[1]), 40)

    def test_get_wells(self):
        """Test determining wells characteristics from a list of angles"""
        scan = [-179, -178, -175, -170, -61, -59, -58, -50, -40, -30, -20, -10, 0, 10, 150, 160]
        wells = conformers.get_wells(label='', angles=scan)
        self.assertEqual(wells, [{'angles': [-179, -178, -175, -170],
                                  'end_angle': -170,
                                  'end_idx': 3,
                                  'start_angle': -179,
                                  'start_idx': 0},
                                 {'angles': [-61, -59, -58, -50, -40, -30, -20, -10, 0, 10],
                                  'end_angle': 10,
                                  'end_idx': 13,
                                  'start_angle': -61,
                                  'start_idx': 4},
                                 {'angles': [150, 160],
                                  'end_angle': 160,
                                  'end_idx': 15,
                                  'start_angle': 150,
                                  'start_idx': 14}])

        scan = [-73.68336628556762, -73.6698270329844, -73.23989527267882, -73.21461046998135, -72.98404533147658,
                -72.98164225603858, -69.39937902123044, -69.3988528569368, -67.77744864514945, -67.77148282968275,
                -65.17239618718334, -65.15907633527517, -64.26652813162181, -64.24439959181805, -63.240801195360554,
                -63.23822372040009, -63.1200724631762, -63.11971243825017, -60.63742589032739, -60.63139420286131,
                -60.173003247364996, -60.16784651044193, -59.468854511900325, -59.46786228068227, -58.65720743567386,
                -58.65683432963624, -57.98873394285648, -57.982941836860924, -57.40064081386206, -57.40044768735372,
                -53.74282998560906, -53.73596914102977, -52.68824260862714, -52.68597330918877, -46.27234815136188,
                -46.25545908648955, -45.31808116497048, -45.31451574906208, -37.35933646586181, -37.35852002741229,
                -36.88328657195036, -36.87053094294823, -36.01343709893113, -36.012150772093335, -34.86948467964051,
                -34.8688938780246, -34.72494851508614, -34.72482090531513, 37.92144833495903, 37.998896394653386,
                41.96215337416899, 42.02448094052937, 45.810576693278705, 45.817331523047116, 51.29257628301334,
                51.30311799717333, 51.61623061568205, 51.617786414116125, 53.34477784221306, 53.34580850447212,
                53.48942661659711, 53.4901617185321, 56.47511675765871, 56.47592191783421, 56.89219524846863,
                56.89667826685056, 58.21791320569343, 58.22084771674736, 58.56402491310407, 58.56610158979941,
                58.575585167060105, 58.61170917818044, 58.72432829799178, 58.73704018872237, 58.86716586121682,
                58.86911130502126, 58.88121731495366, 58.88577729197807, 59.080599884436346, 59.08922104350318,
                59.11353250453865, 59.12147007319143, 59.47503346965784, 59.49304004152549, 60.01935141189763,
                60.02668302599606, 60.251501427706465, 60.25315632266764, 60.46989855245081, 60.47493321918853,
                60.682093573955186, 60.70801046425914, 60.720951804151774, 60.7245195946674, 61.25184661219729,
                61.25714897361994, 61.70601073036617, 61.707906267456195, 61.83301219526896, 61.86382363861279,
                62.13779819494127, 62.13972013334631, 62.220744529176095, 62.223284838706526, 62.36544009963831,
                62.3720484546934, 62.81168956759717, 62.83423018985191, 64.9521798738434, 64.9629609319081,
                66.16746748827471, 66.17893910758922, 66.7668740766453, 66.76788523167403, 67.60590251354962,
                67.62187329547093, 68.06080902669052, 68.0842956220337, 68.09623011492323, 68.10199767805355,
                68.48522980414184, 68.48583421208666, 73.26005419282097, 73.26808846126718, 73.45281733280729,
                73.45469759141737, 74.51490380770962, 74.53803845428752, 74.58370293245521, 74.60616218954709,
                83.33528261426564, 83.35937288850262, 83.43016173886006, 83.45479594676438, 158.0577370327542,
                158.0618058607164, 158.9044370338183, 158.9221003782838, 162.31749728503348, 162.3209967086902,
                164.52195355333234, 164.53582069333316, 164.74621421776186, 164.7467225861628, 165.2164723949667,
                165.2265767573193, 168.07838462664859, 168.09950407181762, 168.88464375222418, 168.88580356066808,
                171.689937353885, 171.72548498246235, 178.33869822654015, 178.3814143461112, 178.63816074390414,
                178.69372759467555, 179.27900877194176, 179.28377950998077, 179.5472101554281, 179.5534956989596,
                179.61109850679009, 179.61481948231528, 180.72497299223755, 180.74053436010533, 180.9941404710718,
                181.02077030292295, 181.11974593201793, 181.1243463457858, 181.37177080545533, 181.37611255671746,
                181.4051354501779, 181.41711108890976, 181.48880148090316, 181.49758562851264, 182.26201874998708,
                182.26780045263806, 182.28202738170322, 182.2825980166329, 182.4882342397495, 182.5114677002835,
                183.34205829177768, 183.35856051636569, 187.10490626605906, 187.10934663583262, 191.6284272944037,
                191.6332491563272, 191.65467120190945, 191.65980785427425, 194.12584582734019, 194.1354751568166,
                194.25775316213299, 194.26909817380425, 203.2455591397814, 203.25703331666415, 203.36649309444803,
                203.37095080136055, 205.1767218350291, 205.17998346889013, 205.78470608775663, 205.78904336991963]
        wells = conformers.get_wells(label='', angles=scan)
        self.assertEqual(wells, [{'angles': [-73.68336628556762, -73.6698270329844, -73.23989527267882,
                                             -73.21461046998135, -72.98404533147658, -72.98164225603858,
                                             -69.39937902123044, -69.3988528569368, -67.77744864514945,
                                             -67.77148282968275, -65.17239618718334, -65.15907633527517,
                                             -64.26652813162181, -64.24439959181805, -63.240801195360554,
                                             -63.23822372040009, -63.1200724631762, -63.11971243825017,
                                             -60.63742589032739, -60.63139420286131, -60.173003247364996,
                                             -60.16784651044193, -59.468854511900325, -59.46786228068227,
                                             -58.65720743567386, -58.65683432963624, -57.98873394285648,
                                             -57.982941836860924, -57.40064081386206, -57.40044768735372,
                                             -53.74282998560906, -53.73596914102977, -52.68824260862714,
                                             -52.68597330918877, -46.27234815136188, -46.25545908648955,
                                             -45.31808116497048, -45.31451574906208, -37.35933646586181,
                                             -37.35852002741229, -36.88328657195036, -36.87053094294823,
                                             -36.01343709893113, -36.012150772093335, -34.86948467964051,
                                             -34.8688938780246, -34.72494851508614, -34.72482090531513],
                                  'end_angle': -34.72482090531513,
                                  'end_idx': 47,
                                  'start_angle': -73.68336628556762,
                                  'start_idx': 0},
                                 {'angles': [37.92144833495903, 37.998896394653386, 41.96215337416899,
                                             42.02448094052937, 45.810576693278705, 45.817331523047116,
                                             51.29257628301334, 51.30311799717333, 51.61623061568205,
                                             51.617786414116125, 53.34477784221306, 53.34580850447212,
                                             53.48942661659711, 53.4901617185321, 56.47511675765871,
                                             56.47592191783421, 56.89219524846863, 56.89667826685056,
                                             58.21791320569343, 58.22084771674736, 58.56402491310407,
                                             58.56610158979941, 58.575585167060105, 58.61170917818044,
                                             58.72432829799178, 58.73704018872237, 58.86716586121682,
                                             58.86911130502126, 58.88121731495366, 58.88577729197807,
                                             59.080599884436346, 59.08922104350318, 59.11353250453865,
                                             59.12147007319143, 59.47503346965784, 59.49304004152549,
                                             60.01935141189763, 60.02668302599606, 60.251501427706465,
                                             60.25315632266764, 60.46989855245081, 60.47493321918853,
                                             60.682093573955186, 60.70801046425914, 60.720951804151774,
                                             60.7245195946674, 61.25184661219729, 61.25714897361994,
                                             61.70601073036617, 61.707906267456195, 61.83301219526896,
                                             61.86382363861279, 62.13779819494127, 62.13972013334631,
                                             62.220744529176095, 62.223284838706526, 62.36544009963831,
                                             62.3720484546934, 62.81168956759717, 62.83423018985191,
                                             64.9521798738434, 64.9629609319081, 66.16746748827471,
                                             66.17893910758922, 66.7668740766453, 66.76788523167403,
                                             67.60590251354962, 67.62187329547093, 68.06080902669052,
                                             68.0842956220337, 68.09623011492323, 68.10199767805355,
                                             68.48522980414184, 68.48583421208666, 73.26005419282097,
                                             73.26808846126718, 73.45281733280729, 73.45469759141737,
                                             74.51490380770962, 74.53803845428752, 74.58370293245521,
                                             74.60616218954709, 83.33528261426564, 83.35937288850262,
                                             83.43016173886006, 83.45479594676438],
                                  'end_angle': 83.45479594676438,
                                  'end_idx': 133,
                                  'start_angle': 37.92144833495903,
                                  'start_idx': 48},
                                 {'angles': [158.0577370327542, 158.0618058607164, 158.9044370338183,
                                             158.9221003782838, 162.31749728503348, 162.3209967086902,
                                             164.52195355333234, 164.53582069333316, 164.74621421776186,
                                             164.7467225861628, 165.2164723949667, 165.2265767573193,
                                             168.07838462664859, 168.09950407181762, 168.88464375222418,
                                             168.88580356066808, 171.689937353885, 171.72548498246235,
                                             178.33869822654015, 178.3814143461112, 178.63816074390414,
                                             178.69372759467555, 179.27900877194176, 179.28377950998077,
                                             179.5472101554281, 179.5534956989596, 179.61109850679009,
                                             179.61481948231528, 180.72497299223755, 180.74053436010533,
                                             180.9941404710718, 181.02077030292295, 181.11974593201793,
                                             181.1243463457858, 181.37177080545533, 181.37611255671746,
                                             181.4051354501779, 181.41711108890976, 181.48880148090316,
                                             181.49758562851264, 182.26201874998708, 182.26780045263806,
                                             182.28202738170322, 182.2825980166329, 182.4882342397495,
                                             182.5114677002835, 183.34205829177768, 183.35856051636569,
                                             187.10490626605906, 187.10934663583262, 191.6284272944037,
                                             191.6332491563272, 191.65467120190945, 191.65980785427425,
                                             194.12584582734019, 194.1354751568166, 194.25775316213299,
                                             194.26909817380425, 203.2455591397814, 203.25703331666415,
                                             203.36649309444803, 203.37095080136055, 205.1767218350291,
                                             205.17998346889013, 205.78470608775663, 205.78904336991963],
                                  'end_angle': 205.78904336991963,
                                  'end_idx': 199,
                                  'start_angle': 158.0577370327542,
                                  'start_idx': 134}])

    def test_determine_dihedrals(self):
        """Test determining the dihedrals in a molecule"""
        confs = list()
        rd_xyzs, rd_energies = conformers.get_force_field_energies(label='', mol=self.mol0, num_confs=10)
        for xyz, energy in zip(rd_xyzs, rd_energies):
            confs.append({'xyz': converter.get_xyz_string(coords=xyz, mol=self.mol0),
                          'FF energy': energy,
                          'molecule': self.mol0,
                          'source': 'RDKit'})
        self.spc0.determine_rotors()
        torsions = [rotor_dict['scan'] for rotor_dict in self.spc0.rotors_dict.values()]
        confs = conformers.determine_dihedrals(confs, torsions)
        self.assertAlmostEqual(confs[0]['torsion_dihedrals'][(9, 1, 2, 3)], 300.14463, 3)
        self.assertAlmostEqual(confs[0]['torsion_dihedrals'][(1, 2, 3, 6)], 301.81907, 3)

    def test_check_special_non_rotor_cases(self):
        """Test that special cases (cyano and azide groups) are not considered as rotors"""
        adj0 = """1 N u0 p1 c0 {3,T}
2 C u0 p0 c0 {3,S} {4,S} {5,S} {6,S}
3 C u0 p0 c0 {2,S} {1,T}
4 H u0 p0 c0 {2,S}
5 H u0 p0 c0 {2,S}
6 H u0 p0 c0 {2,S}"""
        adj1 = """1 C u0 p0 c0 {2,S} {5,S} {6,S} {7,S}
2 N u0 p2 c-1 {1,S} {3,S}
3 N u0 p0 c+1 {2,S} {4,T}
4 N u0 p1 c0 {3,T}
5 H u0 p0 c0 {1,S}
6 H u0 p0 c0 {1,S}
7 H u0 p0 c0 {1,S}"""
        adj2 = """1 C u0 p0 c0 {2,S} {4,S} {5,S} {6,S}
2 C u0 p0 c0 {1,S} {3,T}
3 C u0 p0 c0 {2,T} {7,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
6 H u0 p0 c0 {1,S}
7 H u0 p0 c0 {3,S}"""
        mol0 = Molecule().fromAdjacencyList(str(adj0))
        mol1 = Molecule().fromAdjacencyList(str(adj1))
        mol2 = Molecule().fromAdjacencyList(str(adj2))
        non_rotor0 = conformers.check_special_non_rotor_cases(mol=mol0, top1=[2, 1], top2=[3, 4, 5, 6])
        non_rotor1 = conformers.check_special_non_rotor_cases(mol=mol1, top1=[2, 1, 5, 6, 7], top2=[3, 4])
        non_rotor2 = conformers.check_special_non_rotor_cases(mol=mol2, top1=[1, 4, 5, 6], top2=[2, 3, 7])
        self.assertTrue(non_rotor0)
        self.assertTrue(non_rotor1)
        self.assertFalse(non_rotor2)

        spc0 = ARCSpecies(label='spc0', mol=mol0)
        spc1 = ARCSpecies(label='spc1', mol=mol1)
        spc2 = ARCSpecies(label='spc2', mol=mol2)

        spc0.determine_rotors()
        spc1.determine_rotors()
        spc2.determine_rotors()

        torsions0 = [rotor_dict['scan'] for rotor_dict in spc0.rotors_dict.values()]
        self.assertFalse(torsions0)

        torsions1 = [rotor_dict['scan'] for rotor_dict in spc1.rotors_dict.values()]
        self.assertTrue(len(torsions1) == 1)  # expecting only the CH3 rotor

        torsions2 = [rotor_dict['scan'] for rotor_dict in spc2.rotors_dict.values()]
        self.assertTrue(len(torsions2) == 1)  # expecting only the CH3 rotor

    def test_determine_top_group_indices(self):
        """Test determining the top group in a molecule"""
        mol = Molecule(SMILES=str('c1cc(OC)ccc1OC(CC)SF'))
        atom1 = mol.atoms[9]  # this is the C atom at the S, O, H, and C junction
        atom2a = mol.atoms[10]  # C
        atom2b = mol.atoms[8]  # O
        atom2c = mol.atoms[12]  # S
        atom2d = mol.atoms[21]  # H

        top, top_has_heavy_atoms = conformers.determine_top_group_indices(mol, atom1, atom2a)
        self.assertEqual(len(top), 7)
        self.assertTrue(top_has_heavy_atoms)

        top, top_has_heavy_atoms = conformers.determine_top_group_indices(mol, atom1, atom2b)
        self.assertEqual(len(top), 16)
        self.assertTrue(top_has_heavy_atoms)

        top, top_has_heavy_atoms = conformers.determine_top_group_indices(mol, atom1, atom2c)
        self.assertEqual(len(top), 2)
        self.assertTrue(top_has_heavy_atoms)

        top, top_has_heavy_atoms = conformers.determine_top_group_indices(mol, atom1, atom2d)
        self.assertEqual(top, [22])
        self.assertFalse(top_has_heavy_atoms)  # H

    def test_find_internal_rotors(self):
        """Test finding internal rotors in a molecule list"""
        mol_list = self.mol1.generate_resonance_structures()
        rotors = []
        for mol in mol_list:
            rotors.append(conformers.find_internal_rotors(mol))
        self.assertEqual(len(rotors[0]), 1)
        self.assertEqual(len(rotors[1]), 2)
        self.assertEqual(rotors[0][0]['pivots'], [2, 3])
        self.assertEqual(rotors[1][0]['pivots'], [2, 3])
        self.assertEqual(rotors[1][1]['pivots'], [3, 4])
        self.assertTrue('invalidation_reason' in rotors[0][0])
        self.assertTrue('pivots' in rotors[0][0])
        self.assertTrue('scan' in rotors[0][0])
        self.assertTrue('scan_path' in rotors[0][0])
        self.assertTrue('success' in rotors[0][0])
        self.assertTrue('times_dihedral_set' in rotors[0][0])
        self.assertTrue('top' in rotors[0][0])

        cj_adj = """1  C u0 p0 c0 {2,S} {37,S} {38,S} {39,S}
2  O u0 p2 c0 {1,S} {3,S}
3  C u0 p0 c0 {2,S} {4,D} {11,S}
4  C u0 p0 c0 {3,D} {5,S} {12,S}
5  C u0 p0 c0 {4,S} {6,D} {40,S}
6  C u0 p0 c0 {5,D} {7,S} {10,S}
7  C u0 p0 c0 {6,S} {8,S} {9,S} {41,S}
8  C u0 p0 c0 {7,S} {42,S} {43,S} {44,S}
9  C u0 p0 c0 {7,S} {45,S} {46,S} {47,S}
10 C u0 p0 c0 {6,S} {11,D} {48,S}
11 C u0 p0 c0 {3,S} {10,D} {49,S}
12 C u0 p0 c0 {4,S} {13,S} {50,S} {51,S}
13 N u0 p1 c0 {12,S} {14,S} {52,S}
14 C u0 p0 c0 {13,S} {15,S} {16,S} {20,S}
15 H u0 p0 c0 {14,S}
16 C u0 p0 c0 {14,S} {17,S} {36,S} {53,S}
17 C u0 p0 c0 {16,S} {18,S} {54,S} {55,S}
18 C u0 p0 c0 {17,S} {19,S} {56,S} {57,S}
19 N u0 p1 c0 {18,S} {20,S} {35,S}
20 C u0 p0 c0 {14,S} {19,S} {21,S} {22,S}
21 H u0 p0 c0 {20,S}
22 C u0 p0 c0 {20,S} {23,S} {29,S} {58,S}
23 C u0 p0 c0 {22,S} {24,D} {28,S}
24 C u0 p0 c0 {23,D} {25,S} {59,S}
25 C u0 p0 c0 {24,S} {26,D} {60,S}
26 C u0 p0 c0 {25,D} {27,S} {61,S}
27 C u0 p0 c0 {26,S} {28,D} {62,S}
28 C u0 p0 c0 {23,S} {27,D} {63,S}
29 C u0 p0 c0 {22,S} {30,D} {34,S}
30 C u0 p0 c0 {29,D} {31,S} {64,S}
31 C u0 p0 c0 {30,S} {32,D} {65,S}
32 C u0 p0 c0 {31,D} {33,S} {66,S}
33 C u0 p0 c0 {32,S} {34,D} {67,S}
34 C u0 p0 c0 {29,S} {33,D} {68,S}
35 C u0 p0 c0 {19,S} {36,S} {69,S} {70,S}
36 C u0 p0 c0 {16,S} {35,S} {71,S} {72,S}
37 H u0 p0 c0 {1,S}
38 H u0 p0 c0 {1,S}
39 H u0 p0 c0 {1,S}
40 H u0 p0 c0 {5,S}
41 H u0 p0 c0 {7,S}
42 H u0 p0 c0 {8,S}
43 H u0 p0 c0 {8,S}
44 H u0 p0 c0 {8,S}
45 H u0 p0 c0 {9,S}
46 H u0 p0 c0 {9,S}
47 H u0 p0 c0 {9,S}
48 H u0 p0 c0 {10,S}
49 H u0 p0 c0 {11,S}
50 H u0 p0 c0 {12,S}
51 H u0 p0 c0 {12,S}
52 H u0 p0 c0 {13,S}
53 H u0 p0 c0 {16,S}
54 H u0 p0 c0 {17,S}
55 H u0 p0 c0 {17,S}
56 H u0 p0 c0 {18,S}
57 H u0 p0 c0 {18,S}
58 H u0 p0 c0 {22,S}
59 H u0 p0 c0 {24,S}
60 H u0 p0 c0 {25,S}
61 H u0 p0 c0 {26,S}
62 H u0 p0 c0 {27,S}
63 H u0 p0 c0 {28,S}
64 H u0 p0 c0 {30,S}
65 H u0 p0 c0 {31,S}
66 H u0 p0 c0 {32,S}
67 H u0 p0 c0 {33,S}
68 H u0 p0 c0 {34,S}
69 H u0 p0 c0 {35,S}
70 H u0 p0 c0 {35,S}
71 H u0 p0 c0 {36,S}
72 H u0 p0 c0 {36,S}
"""
        cj_mol = Molecule().fromAdjacencyList(adjlist=str(cj_adj))
        rotors = conformers.find_internal_rotors(cj_mol)
        for rotor in rotors:
            if rotor['pivots'] == [6, 7]:
                self.assertEqual(rotor['scan'], [5, 6, 7, 8])
                self.assertEqual(sum(rotor['top']), 332)  # [7,41,8,44,42,43,9,46,47,45] in non-deterministic order

        mol = Molecule(SMILES=str('CCCO'))
        rotors = conformers.find_internal_rotors(mol)
        for rotor in rotors:
            self.assertIn(rotor['scan'], [[5, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 12]])

    def test_to_group(self):
        """Test converting a part of a molecule into a group"""
        atom_indices = [0, 3, 8]
        group0 = conformers.to_group(mol=self.mol1, atom_indices=atom_indices)

        atom0 = GroupAtom(atomType=[atomTypes[str('Cd')]], radicalElectrons=[0], charge=[0],
                          label=str(''), lonePairs=[0])
        atom1 = GroupAtom(atomType=[atomTypes[str('O2s')]], radicalElectrons=[1], charge=[0],
                          label=str(''), lonePairs=[2])
        atom2 = GroupAtom(atomType=[atomTypes[str('H')]], radicalElectrons=[0], charge=[0],
                          label=str(''), lonePairs=[0])
        group1 = Group(atoms=[atom0, atom1, atom2], multiplicity=[2])
        bond01 = GroupBond(atom0, atom1, order=[1.0])
        bond02 = GroupBond(atom0, atom2, order=[1.0])
        group1.addBond(bond01)
        group1.addBond(bond02)

        self.assertTrue(group0.isIsomorphic(group1))

    def test_get_torsion_angles(self):
        """Test determining the torsion angles from all conformers"""
        torsions = conformers.determine_rotors(self.spc0.mol_list)[0]
        confs = [{'FF energy': -1.5170975770157071,
                  'molecule': self.mol0,
                  'source': 'RDKit',
                  'xyz': 'O       1.22700646   -0.74306134   -0.46642912\nC       0.44275447    0.24386237    '
                         '0.18670577\nC      -1.02171998   -0.04112700   -0.06915927\nH       0.65837353    '
                         '0.21260997    1.25889575\nH       0.71661793    1.22791280   -0.20534525\nH      '
                         '-1.29179957   -1.03551511    0.30144722\nH      -1.23432965   -0.03370364   -1.14332936\n'
                         'H      -1.65581027    0.70264687    0.42135230\nH       2.15890709   -0.53362491   '
                         '-0.28413803\n'},
                 {'FF energy': -1.336857063940798,
                  'molecule': self.mol0,
                  'source': 'RDKit',
                  'xyz': 'O       1.45741739    0.61020598   -0.00386149\nC       0.49432210   -0.39620513   '
                         '-0.28042773\nC      -0.89596057    0.11891813    0.02906668\nH       0.57628556   '
                         '-0.66552902   -1.33744779\nH       0.72801279   -1.28050031    0.32015417\n'
                         'H      -1.11282223    1.01829132   -0.55660310\nH      -0.97906607    0.39858005    '
                         '1.08445491\nH      -1.65147628   -0.63942945   -0.19448738\nH       1.38328732    '
                         '0.83566844    0.93915173\n'},
                 {'FF energy': -1.3368570629495138,
                  'molecule': self.mol0,
                  'source': 'RDKit',
                  'xyz': 'O      -1.48027320    0.36597456    0.41386552\nC      -0.49770656   -0.40253648   '
                         '-0.26500019\nC       0.86215119    0.24734211   -0.11510338\nH      -0.77970114   '
                         '-0.46128090   -1.32025907\nH      -0.49643724   -1.41548311    0.14879346\nH       '
                         '0.84619526    1.26924854   -0.50799415\nH       1.14377239    0.31659216    0.94076336\n'
                         'H       1.62810781   -0.32407050   -0.64676910\nH      -1.22610851    0.40421362    '
                         '1.35170355\n'}]
        confs = conformers.determine_dihedrals(confs, torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(torsions[0], [9, 1, 2, 3])
        self.assertEqual(torsions[1], [1, 2, 3, 6])
        self.assertTrue(all([int(round(angle / 5.0) * 5.0) in [60, 180, 300]
                             for angle in torsion_angles[tuple(torsions[0])]]))  # batch check almost equal
        self.assertTrue(all([int(round(angle / 5.0) * 5.0) in [60, 300]
                             for angle in torsion_angles[tuple(torsions[1])]]))  # batch check almost equal

    def test_check_atom_collisions(self):
        """Check that we correctly determine when atoms collide in xyz"""
        xyz0 = """C	0.0000000	0.0000000	0.6505570
C	0.0000000	0.0000000	-0.6505570
C	0.0000000	0.0000000	1.9736270
C	0.0000000	0.0000000	-1.9736270"""  # no colliding atoms
        xyz1 = """ C                 -2.09159210   -1.05731582   -0.12426100
 H                 -2.49607181   -1.61463460   -0.94322229
 H                 -1.03488175   -1.21696026   -0.07152162
 H                 -2.54731584   -1.38200055    0.78777873
 N                 -2.36156239    0.37373413   -0.32458326
 N                 -2.10618543   -0.96914562   -0.62731730
 N                 -1.23618386   -0.31836000   -0.51825841"""  # colliding atoms
        self.assertFalse(conformers.check_atom_collisions(xyz0))
        self.assertTrue(conformers.check_atom_collisions(xyz1))

    def test_determine_torsion_symmetry(self):
        """Test that we correctly determine the torsion symmetry"""
        adj0 = """1 O u0 p2 c0 {2,S} {9,S}
2 C u0 p0 c0 {1,S} {3,S} {4,S} {5,S}
3 C u0 p0 c0 {2,S} {6,S} {7,S} {8,S}
4 H u0 p0 c0 {2,S}
5 H u0 p0 c0 {2,S}
6 H u0 p0 c0 {3,S}
7 H u0 p0 c0 {3,S}
8 H u0 p0 c0 {3,S}
9 H u0 p0 c0 {1,S}"""
        mol0 = Molecule().fromAdjacencyList(str(adj0))
        torsion_scan0 = [-179.5316509266246, -179.5316509266246, -179.5316509266246, -179.5316509266246,
                         -179.5316509266246, -179.5316509266246, -179.5316509266246, -179.5316509266246,
                         -60.11816797679059, -60.11810624433042, -60.11804469757614, -58.18097368133174,
                         -58.18045845481007, 58.18063730000221, 58.18087223853245, 58.18090794745689, 60.1180249902725,
                         60.11822550406657, 179.06251373730467, 179.0631045514683, 179.06310597052072,
                         179.0633554023178, 179.99994796571877, 179.99999648225656]
        top0 = [3, 6, 7, 8]
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=top0, mol_list=[mol0],
                                                               torsion_scan=torsion_scan0), 3)

        # both rotors are symmetric with symmetry numbers of 2 and 3
        mol1 = Molecule(SMILES=str('CC[N+](=O)[O-]'))
        mol1.update()
        torsions, tops = conformers.determine_rotors([mol1])
        confs = conformers.generate_force_field_conformers(mol_list=[mol1], label='mol1', num_confs=50,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        with self.assertRaises(ConformerError):
            conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[0], mol_list=[mol1],
                                                               torsion_scan=torsion_angles[tuple(torsions[0])]), 2)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[1], mol_list=[mol1],
                                                               torsion_scan=torsion_angles[tuple(torsions[1])]), 3)

        # only one rotor is symmetric
        mol2 = Molecule(SMILES=str('CC[N+](=S)[O-]'))
        mol2.update()
        torsions, tops = conformers.determine_rotors([mol2])
        confs = conformers.generate_force_field_conformers(mol_list=[mol2], label='mol2', num_confs=50,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        # NSO rotor:
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[0], mol_list=[mol2],
                                                               torsion_scan=torsion_angles[tuple(torsions[0])]), 1)
        # CH3 rotor:
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[1], mol_list=[mol2],
                                                               torsion_scan=torsion_angles[tuple(torsions[1])]), 3)

        # The COH rotor is symmetric because of the bottom of the molecule
        mol3 = Molecule(SMILES=str('c1ccccc1C(c1ccccc1)(c1ccccc1)O'))
        mol3.update()
        torsions, tops = conformers.determine_rotors([mol3])
        confs = conformers.generate_force_field_conformers(mol_list=[mol3], label='mol3', num_confs=100,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[1], mol_list=[mol3],
                                                               torsion_scan=torsion_angles[tuple(torsions[1])]), 2)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[2], mol_list=[mol3],
                                                               torsion_scan=torsion_angles[tuple(torsions[2])]), 2)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[3], mol_list=[mol3],
                                                               torsion_scan=torsion_angles[tuple(torsions[3])]), 2)

        mol4 = Molecule(SMILES=str('c1ccccc1CO'))
        mol4.update()
        torsions, tops = conformers.determine_rotors([mol4])
        confs = conformers.generate_force_field_conformers(mol_list=[mol4], label='mol4', num_confs=100,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertTrue(any([conformers.determine_torsion_symmetry(label='', top1=tops[i], mol_list=[mol4],
                                                                   torsion_scan=torsion_angles[tuple(torsions[i])])
                             == 2 for i in range(2)]))

        mol5 = Molecule(SMILES=str('OCC(C)C'))
        mol5.update()
        torsions, tops = conformers.determine_rotors([mol5])
        confs = conformers.generate_force_field_conformers(mol_list=[mol5], label='mol5', num_confs=100,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(len(torsions), 4)
        self.assertEqual(sum([conformers.determine_torsion_symmetry(label='', top1=tops[i], mol_list=[mol5],
                                                                    torsion_scan=torsion_angles[tuple(torsions[i])])
                              for i in range(len(torsions))]), 8)

        mol7 = Molecule(SMILES=str('CC'))
        mol7.update()
        torsions, tops = conformers.determine_rotors([mol7])
        confs = conformers.generate_force_field_conformers(mol_list=[mol7], label='mol7', num_confs=50,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(len(torsions), 1)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[0], mol_list=[mol7],
                                                               torsion_scan=torsion_angles[tuple(torsions[0])]), 9)

        mol8 = Molecule(SMILES=str('C[N+](=O)[O-]'))
        mol8.update()
        torsions, tops = conformers.determine_rotors([mol8])
        confs = conformers.generate_force_field_conformers(mol_list=[mol8], label='mol8', num_confs=200,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(len(torsions), 1)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[0], mol_list=[mol8],
                                                               torsion_scan=torsion_angles[tuple(torsions[0])]), 6)

        mol9 = Molecule(SMILES=str('Cc1ccccc1'))
        mol9.update()
        torsions, tops = conformers.determine_rotors([mol9])
        confs = conformers.generate_force_field_conformers(mol_list=[mol9], label='mol9', num_confs=50,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(len(torsions), 1)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[0], mol_list=[mol9],
                                                               torsion_scan=torsion_angles[tuple(torsions[0])]), 6)

    def test_determine_torsion_sampling_points(self):
        torsion_angles = [-179, -178, -175, -170, -61, -59, -58, -50, -40, -30, -20, -10, 0, 10, 150, 160]
        sampling_points, wells = conformers.determine_torsion_sampling_points(label='', torsion_angles=torsion_angles)
        expected_sampling_points = [-175.5, -46.8, -16.8, 155.0]
        for i, entry in enumerate(sampling_points):
            self.assertAlmostEqual(entry, expected_sampling_points[i])
        self.assertEqual(wells, [{'angles': [-179, -178, -175, -170],
                                  'end_angle': -170,
                                  'end_idx': 3,
                                  'start_angle': -179,
                                  'start_idx': 0},
                                 {'angles': [-61, -59, -58, -50, -40, -30, -20, -10, 0, 10],
                                  'end_angle': 10,
                                  'end_idx': 13,
                                  'start_angle': -61,
                                  'start_idx': 4},
                                 {'angles': [150, 160],
                                  'end_angle': 160,
                                  'end_idx': 15,
                                  'start_angle': 150,
                                  'start_idx': 14}])

        torsion_angles = [-179, -178, -175, -170, -61, -59, -58, -50, -40, -30, -20, -10, 0, 10, 150, 160]
        sampling_points = conformers.determine_torsion_sampling_points(label='', torsion_angles=torsion_angles, symmetry=3)[0]
        self.assertEqual(sampling_points, [-175.5])

        torsion_angles = [-179.59209512848707, -179.59039374817334, -178.5381860689443, -178.5381860689443,
                          -100.0213361656215, -100.01202781166394, -97.57008398091493, -97.5656354570548,
                          -87.22958350400677, -87.22873051318311, -82.76374562675298, -82.76281808490587,
                          -77.32052021810122, -77.31679209349836, -76.31480385405374, -76.31479880568214,
                          -75.74086282984688, -75.74065682695682, -74.44997435821709, -74.44504133182852,
                          -71.7211598083571, -71.6855764028349, -69.20957455738775, -69.20955093487336,
                          76.44226785146085, 76.44226796206006, 78.8000414071327, 78.80248169737817,
                          97.86638633387129, 97.86690688613884, 101.5620159123216, 101.56211964495982,
                          101.60273917268496, 101.62220693019022, 167.9999696197525, 168.015496963019,
                          173.50361410674796, 173.50361428302764, 179.87419886059158, 179.87466884415952]
        sampling_points = conformers.determine_torsion_sampling_points(label='', torsion_angles=torsion_angles)[0]
        self.assertEqual(sampling_points, [-81.23116365828709, 91.25694337981986, 176.65127016627497])

        torsion_angles = [-140.93700443614793, -123.36532069364058, -123.36193452991425, -111.43276833900227,
                          -108.85003802492857, -108.80628841146448, -106.7480881860873, -106.74806981156777,
                          -100.91787288940608, -100.89098870382315, -100.87197726936017, -100.8670100492721,
                          -100.07926884972234, -100.07340029718088, -98.8552582913082, -98.85092067879054,
                          -97.06677422431518, -97.0599902038826, -94.91878866854296, -93.99047972275888,
                          -93.9871448266935, -80.83953665192314, -80.7985814752682, -73.3974041445711,
                          -73.3967278065724, -66.02347934836105, -62.94758463904744, -62.94601240731786,
                          88.10464521993997, 95.68946323823499, 95.69459080513944, 97.33055063592927,
                          97.33445482833133, 97.49952914172731, 97.49958226697981, 108.85221511940138,
                          112.13184073171584, 112.1329020043724, 113.43204366643745, 113.43508423709228]
        sampling_points = conformers.determine_torsion_sampling_points(label='', torsion_angles=torsion_angles)[0]
        self.assertEqual(sampling_points, [-111.75102548503112, -81.75102548503112, 102.4280751579418])

        torsion_angles = [-176.32367786496187, -176.32366756512965, -172.41839534439572, -172.41775488821958,
                          -170.91972028737058, -170.91684662144468, -168.67365935090615, -168.6494823319953,
                          -157.3378655697005, -157.3303178609214, -130.88111889588166, -130.8801033220279,
                          -123.03442492591823, -123.03254926057159, -106.62037325134804, -106.61491298368965,
                          -106.49370355433265, -106.49062503974048, -106.44815309277438, -106.44661556514869,
                          -106.1184053949384, -106.02356734374266, -105.45571395093299, -105.45419525625267,
                          -102.87853709840053, -102.87470223350834, -101.65258358714979, -101.65147600570826,
                          -99.06728895084679, -99.0659742624123, -97.09248330283366, -97.09226379424082,
                          -97.05006981831355, -97.04745525527609, -94.85960670363507, -94.83688753416565,
                          -94.15546343837148, -94.13935516875053, -91.37023462319588, -91.36840071425124,
                          -91.1281919778042, -91.12767582757697, -89.25224100918345, -89.21821902496742,
                          -84.22570292955713, -84.2257024797515, -81.92657059114696, -81.91700530674886,
                          -81.5393507005209, -81.51907146787487, -80.76940047484456, -80.76830046874234,
                          -80.76692295509889, -80.76585014807056, -80.7635271472528, -80.76003515803873,
                          -80.52672200275894, -80.39218945327103, -79.45139539728271, -79.44802596060535,
                          -78.82939626622594, -78.8238874815072, -78.82171404634528, -78.82122816661064,
                          -78.82116824043291, -78.81999626732221, -78.81873300109925, -78.80988501047182,
                          -78.79109480712224, -78.79104797380982, -78.02486723775623, -78.0207943229613,
                          -77.76478276489519, -77.75105651990408, -76.76274679518627, -76.75994306044464,
                          -76.75720516627095, -76.75655929365495, -76.40065136049024, -76.40035264169505,
                          -76.4002489893333, -76.39857658106183, -76.18148182071192, -76.17758687843958,
                          -75.55883250235938, -75.55792284240913, -75.27220635873151, -74.43470979615228,
                          -74.43128717991019, -74.42862438989405, -74.42254895368205, -74.42121375406396,
                          -74.42061328648637, -74.4185494461102, -74.41698375776336, -74.41242116374644,
                          -74.41134549205456, -74.03194338082253, -74.03193538166298, -72.98778492414132,
                          -72.56513389061807, -72.56420170854751, -72.55606770902523, -72.55131847164766,
                          -72.1556610545832, -72.15404015444578, -72.1525939681839, -72.14881534733867,
                          -72.13536838650838, -72.13117581461707, -72.11109870935833, -71.76338037325804,
                          -71.68635507869195, -71.4965594065109, -71.49416372425631, -71.01365957663336,
                          -70.99580090673685, -70.79600382818957, -70.79136095072394, -68.99327656761112,
                          -68.98628389958294, -68.801431426, -68.64071819515623, -68.56798026211527,
                          -68.55142880531791, -68.38995743379408, -68.34725708951349, -67.63376975355544,
                          -67.63247950673237, -67.24224371676223, -67.20016187584905, -66.86440439544594,
                          -66.38265734217568, -66.38137004090761, -66.0420035720752, -66.02008093273373,
                          -65.10329969062232, -64.95585077635508, -64.94802760197513, -64.94566407229142,
                          -64.94550631058783, -64.94113135846264, -64.31992398454976, -64.30549484837044,
                          -64.28164114664871, -63.93255270295563, -63.93176633496354, -63.83853501942231,
                          -63.83756926081165, -63.67223694847794, -63.67099787070451, -63.313172532743806,
                          -63.31272974694704, -62.52978432495333, -62.51748053301454, -62.435478952991026,
                          -62.42795694545291, -62.30628986511537, -62.299942625126135, -61.97778732375277,
                          -61.96265686315364, -61.02813902852594, -60.98157761499027, -60.766682952179764,
                          -60.76465244915844, -60.48690489577148, -60.471611562285965, -60.01598089789667,
                          -60.00496318172631, -59.92086886768599, -59.913771310936156, -59.1354858606218,
                          -59.13530340212875, -59.10794689078138, -59.09337905184425, -58.984663695611985,
                          -58.9843124168589, -58.72762414372221, -58.696480641562154, -58.695654875465095,
                          -58.62435983047518, -58.03715200185537, -58.03427221628028, -58.03219999603017,
                          -58.031876379135774, -58.02487684530359, -58.02068962330961, -57.59371977772342,
                          -57.56078584525814, -57.55561183399493, -57.50072840208864, -57.032757001725024,
                          -57.010274386456054, -56.35293875133191, -56.33199560974967, -56.074985773345816,
                          -56.07490421929733, -47.275322578671656, -47.26657758073491, 25.809606429720823,
                          48.25078667982481, 48.27271314406701, 50.17085636345509, 50.171666700725766,
                          54.825786971684956, 54.8270959675416, 54.82801972678648, 54.83998709758166,
                          55.217356366890705, 55.222640628360544, 55.63284261357208, 55.63470843229376,
                          56.06991437742509, 56.072010852678616, 56.16268683800219, 56.17532397584247,
                          57.563521896714526, 57.56379561118491, 57.58220162436901, 57.58238054823384,
                          57.773083571358136, 57.77759654244658, 58.02807739899398, 58.0366638254267,
                          58.6602605445575, 58.8186647391465, 59.078472206071325, 59.08343206068942,
                          59.13088372662295, 59.13833825100024, 61.16787507452899, 61.19150600898746, 61.46146030387509,
                          61.48207696062333, 62.00611886241341, 62.007061944422155, 62.302417977239266,
                          62.30258831738516, 62.304595670178955, 62.30752051958369, 62.3928572433257,
                          62.394690027161005, 62.39507804066775, 62.40761950553415, 62.43300411487009,
                          62.438690527218625, 62.84124026732686, 62.8506486485056, 63.19285387679225, 63.19421010254325,
                          63.75017305004417, 63.7647047937204, 63.821893406046755, 63.84121559550037,
                          63.848486286450935, 63.87463519698978, 64.09430722365335, 64.09441331439972,
                          64.2455888373737, 64.24757421619192, 64.30553145396728, 64.30574858897346, 64.91706498924965,
                          64.93292069662333, 64.94411171945082, 64.94469126990732, 65.10560181105447, 65.14520620184406,
                          65.15039226448064, 65.15873721160678, 65.17031707742622, 65.20502692437536, 65.75025128293476,
                          65.7518510416996, 67.48435748370368, 67.48993483575366, 67.78223180118508, 67.79865159086725,
                          68.57163156828182, 68.58278969476429, 68.9835342681565, 68.98502489257125, 69.02485893048764,
                          69.02531831229979, 69.30083134611509, 69.32639761608996, 69.78727996013161, 69.79003948478928,
                          69.81646328793155, 69.81650466417702, 70.75457219440229, 70.7711916067069, 70.79267671910102,
                          70.7944938028614, 70.79540510667567, 70.94549462058072, 70.9531816227119, 71.07947581077214,
                          72.10598479387446, 72.106971801984, 72.14765977159695, 72.15631579934205, 72.15809553596962,
                          72.16477284932475, 72.55513044781213, 72.55524795156039, 75.28412012313748, 75.55541153577926,
                          75.56300919137861, 76.39916569514754, 76.40012661444992, 77.50634180413392, 77.6575588981426,
                          78.0301542193255, 78.03303193448812, 78.78787044406549, 78.79082915408495, 78.80116145781852,
                          78.82627274476296, 79.10247828055287, 79.12271367210117, 79.4377244591139, 79.45142875982222,
                          79.4782352817987, 79.47896782421039, 79.47949360294194, 79.47952864923629, 79.80643678517221,
                          79.81306411762233, 81.98172874001276, 81.98685806299791, 89.73711444222226, 89.74110812758445,
                          91.04881077542787, 91.04924826404962, 91.60403875417842, 91.60672185467155, 92.13554278700374,
                          92.15206547263318, 93.5653324027846, 93.56591614818197, 93.56718452183638, 93.56880449913933,
                          94.17783361803986, 94.19514712640881, 95.54189421759462, 95.54196730280678, 99.45620288193062,
                          99.46318922816309, 99.4783063650388, 99.4847190781842, 99.48634425879055, 101.65124311361308,
                          101.65178162868837, 102.85679501819145, 102.85872102229916, 102.8597074345882,
                          102.85995894555535, 104.86907002547052, 104.90241841732139, 105.44836556399554,
                          105.45349950525973, 105.45748075964613, 105.46428509851808, 105.8141771701505,
                          105.81833520271985, 106.61321552233747, 106.61497971551212, 108.55192687444472,
                          108.55315365741819, 108.55393853803722, 108.5547410527443, 115.08686295289823,
                          115.0967600690075, 130.84084515283902, 130.84104663344476, 130.878423658668,
                          130.88412149967814, 132.52441710224534, 132.5302238969977, 132.5338586978291,
                          132.5357908447517, 138.3415593476458, 138.34462568197915, 155.72658039999197,
                          155.8138698964472, 156.1174332413083, 157.60010974730972, 158.46355864481313,
                          158.4843791142827, 164.91378968522747, 164.91381098080547, 168.67215306371415,
                          168.67217353280967, 168.8914569801022, 168.90350649651364, 168.98439201781522,
                          169.04580022622054, 173.35440894486018, 173.3630011494851]
        sampling_points = conformers.determine_torsion_sampling_points(label='', torsion_angles=torsion_angles)[0]
        self.assertEqual(sampling_points, [-89.8944923069914, -59.894492306991395, 25.809606429720823,
                                           66.64007136763342, 96.64007136763342, 126.64007136763342,
                                           156.64007136763342, 186.64007136763342])

        torsion_angles = [-152.26161397481087, -152.2587259229901, -149.85232239217063, -149.85131396187322,
                          -146.77192479199664, -146.767884027469, -145.57851706076082, -145.57795571301875,
                          -145.29868530905043, -145.2917116719423, -145.29051311436885, -145.27897926714755,
                          -144.9353292713405, -144.90031439769575, -144.01906055400786, -144.01368503493777,
                          -141.83362732032572, -141.83073689807793, -141.31689674406957, -141.3155452010979,
                          -126.52583775531349, -126.51374906104809, -125.48279255327547, -125.48052065515803,
                          -116.76032183509822, -116.68669731942401, -114.42412582417622, -114.42397929900876,
                          -114.3219200223148, -114.31689064578866, -112.92857969038049, -112.90289334814572,
                          -111.64516286895822, -111.64174408828625, -107.783758893093, -107.78362668770382,
                          -107.78272377509681, -107.78139720397787, -106.10499301818908, -106.07920805912944,
                          -101.30973306676242, -101.30770571605902, -97.96232148286222, -97.95803268012227,
                          -94.68109066403396, -94.68001291333789, -94.45954873115072, -94.4585310442054,
                          -91.10881237653345, -91.10498380222138, -89.79721536455558, -89.79540897901609,
                          -89.24508995784996, -89.24171285149438, -87.15903006851333, -87.14098455539963,
                          -75.07189277945282, -75.0700724822563, -73.2370916471699, -71.28789425243174,
                          -71.26642188810762, -69.98087189130806, -69.98067956102912, -67.08119083228857,
                          -66.07470925160028, -66.07266698089337, -66.01691326227456, -66.01307797096364,
                          -62.39897676703242, -60.569973114326544, -59.45659371612436, -59.44747737440295,
                          -57.407645476308815, -57.39323383017332, -56.424915199923326, -56.379627906445805,
                          -43.40112923452506, -43.382936778423975, -41.20778045506249, -41.19119600629522,
                          -40.65626740259298, -40.633128098813266, -39.57723608008444, -39.573836290145245,
                          -35.88881046562531, -35.88858196712795, -35.11143569652265, -35.11110464719864,
                          -34.61204504109422, -34.59246788758265, -34.000752449057806, -34.000105454686725,
                          -33.98396562146134, -33.98149000952066, -32.849645686709955, -32.84888126347425,
                          -30.887186617425712, -30.884043293767018, -28.35154271492328, -28.343390034828456,
                          -28.31636358820504, -28.31498925871636, -28.312921830770883, -28.307604531049005,
                          -28.294819929386918, -28.243308163923558, 21.029952658998006, 21.03683763790061,
                          22.318426332229294, 22.336215183961727, 23.333588670057164, 23.53377212293256,
                          25.49301296124286, 25.557679425383313, 27.37408014443026, 27.375068021347385,
                          28.302628635236076, 28.30666188407542, 35.11676917687218, 35.122528645010405,
                          35.38600242978858, 35.423586629995526, 36.49418243038614, 36.49461194898948,
                          38.08080106534097, 38.08143077323737, 39.63415129543948, 39.63522549410671, 41.15932732141922,
                          50.9477443315772, 50.95423555170428, 56.17775368411345, 56.17902819213001, 59.46352938407191,
                          59.47772321411631, 59.53170081489957, 60.70182746213326, 60.709505081165965,
                          64.79089192650625, 64.80853497686292, 64.9224755210989, 64.92656059591395, 68.68587366805725,
                          68.821036584528, 69.78204869965228, 69.81931017893343, 72.73119662519044, 72.74064499939671,
                          76.13914875143365, 76.14034180826447, 80.51285821060877, 80.51360853678266, 82.62855279474225,
                          82.62962281642685, 82.66153389416449, 82.67279209613228, 85.01626903974508, 85.02283268800834,
                          85.38800082360545, 85.38986517017983, 86.10988589746262, 86.11012064044289, 86.26592905483845,
                          86.2679012539459, 96.20740209593461, 96.21360170732405, 100.16258231764904,
                          100.16297697301955, 104.69626153683052, 106.92608118016551, 106.93146790288502,
                          116.61863123769633, 116.61864592924242, 118.24417546451886, 118.2493793312905,
                          120.1668712499721, 120.182922789032, 121.53479077752151, 121.53654524750456,
                          123.38491194765737, 123.4205961326216, 141.79830426741128, 141.84598851738662,
                          142.15429059974863, 142.16774851816908, 145.57607725441045, 145.57876371361897,
                          146.14196953860653, 146.1481645699347, 146.29244880461678, 146.3017402883685,
                          146.8840668702791, 146.89163317627043, 146.99548693580584, 147.0234330114963,
                          152.95573242995786, 152.95868183843845, 154.80682233127425, 154.80812898533412,
                          154.84602670053238]
        sampling_points = conformers.determine_torsion_sampling_points(label='', torsion_angles=torsion_angles,
                                                                       symmetry=2)[0]
        self.assertEqual(sampling_points, [-130.19252237896558, -100.19252237896558, -70.19252237896558,
                                           -40.19252237896558])

    def test_change_dihedrals_and_force_field_it(self):
        """Test changing a dihedral and retrieving the new FF energy"""
        ncc_xyz = """N       0.92795000   -0.06591600   -0.03643200
C       2.38932500   -0.06185100   -0.06491100
C       2.91383400    1.35741700   -0.22361700
H       2.74111100   -0.47429900    0.88565600
H       2.81050800   -0.69503700   -0.86161200
H       2.54377900    1.99297300    0.58410700
H       4.00671000    1.37386200   -0.21263700
H       2.58394500    1.79116300   -1.17337000
H       0.55243400    0.27426600   -0.91441800
H       0.56679600   -1.00155900    0.10247100"""
        ncc_spc = ARCSpecies(label='NCC', smiles='NCC', xyz=ncc_xyz)
        ncc_mol = ncc_spc.mol
        energies = conformers.get_force_field_energies(label='NCC', mol=ncc_mol, xyz=ncc_xyz, optimize=True)[1]
        self.assertAlmostEqual(energies[0], -6.15026868, 5)
        idx0 = 10
        for i, atom in enumerate(ncc_mol.atoms):
            if atom.isNitrogen():
                idx1 = i + 1
            elif atom.isCarbon():
                for atom2 in atom.edges.keys():
                    if atom2.isNitrogen():
                        idx2 = i + 1
                        break
                else:
                    idx3 = i + 1
            elif atom.isHydrogen():
                for atom2 in atom.edges.keys():
                    if atom2.isNitrogen():
                        if i + 1 < idx0:
                            idx0 = i + 1
        torsion = (idx0, idx1, idx2, idx3)

        rd_conf, _, index_map = converter.rdkit_conf_from_mol(ncc_mol, converter.get_xyz_matrix(ncc_xyz)[0])
        rd_scan = [index_map[i - 1] for i in torsion]  # convert the atom indices to RDKit indices
        angle = rdMT.GetDihedralDeg(rd_conf, rd_scan[0], rd_scan[1], rd_scan[2], rd_scan[3])

        self.assertAlmostEqual(angle, 62.9431377, 5)

        xyzs, energies = conformers.change_dihedrals_and_force_field_it(label='NCC', mol=ncc_mol, xyz=ncc_xyz,
                                                                        torsions=[torsion], new_dihedrals=[180])

        expected_xyz = """N       0.93418641   -0.03285604   -0.03001207
C       2.38993503   -0.07484791   -0.06911647
C       2.94140390   -0.57952557    1.25275023
H       2.71784999   -0.72977465   -0.88308842
H       2.77585269    0.92949655   -0.27090234
H       2.59029218   -1.59438178    1.46820445
H       4.03547080   -0.60133377    1.22136235
H       2.64077179    0.07038090    2.08153013
H       0.57958892    0.25714689   -0.94038297
H       0.56736059   -0.96947011    0.13439310
"""

        self.assertAlmostEqual(energies[0], -6.1502687, 5)
        self.assertEqual(xyzs[0], expected_xyz)
        self.assertEqual(len(energies), 1)

        xyzs, energies = conformers.change_dihedrals_and_force_field_it(label='NCC', mol=ncc_mol, xyz=ncc_xyz,
                                                                        torsions=[torsion, torsion],
                                                                        new_dihedrals=[0, 180])
        self.assertEqual(len(energies), 2)

        xyzs, energies = conformers.change_dihedrals_and_force_field_it(label='NCC', mol=ncc_mol, xyz=ncc_xyz,
                                                                        torsions=[torsion, torsion],
                                                                        new_dihedrals=[[0, 180], [90, -120]])
        self.assertEqual(len(energies), 4)

    def test_determine_well_width_tolerance(self):
        """Test determining well width tolerance"""
        tols = list()
        tols.append(conformers.determine_well_width_tolerance(mean_width=101))
        tols.append(conformers.determine_well_width_tolerance(mean_width=90))
        tols.append(conformers.determine_well_width_tolerance(mean_width=50))
        tols.append(conformers.determine_well_width_tolerance(mean_width=45))
        tols.append(conformers.determine_well_width_tolerance(mean_width=23))
        tols.append(conformers.determine_well_width_tolerance(mean_width=7))
        tols.append(conformers.determine_well_width_tolerance(mean_width=1.5))
        self.assertEqual(tols, [0.1, 0.10530934999999897, 0.15021874999999973, 0.16273341406249986, 0.2647389825514999,
                                0.46149436430350005, 0.5777707774184844])

    def test_get_lowest_confs(self):
        """Test getting the n lowest conformers"""
        # test a case where confs is a list of dicts:
        confs = [{'index': 0,
                  'FF energy': 20},
                 {'index': 1,
                  'FF energy': 30},
                 {'index': 1,
                  'FF energy': 40},
                 {'index': 1,
                  'some other energy': 10}]
        lowest_confs = conformers.get_lowest_confs(label='', confs=confs, n=2, energy='FF energy')
        self.assertEqual(len(lowest_confs), 2)
        for conf in lowest_confs:
            self.assertTrue(conf['FF energy'] in [20, 30])
            self.assertTrue(conf['index'] in [0, 1])

        # test a case where confs is a list of lists:
        confs = [['C 1 0 0', 8],
                 ['C 2 0 0', 7],
                 ['C 3 0 0', 6],
                 ['C 4 0 0', 5]]
        lowest_confs = conformers.get_lowest_confs(label='', confs=confs)
        self.assertEqual(len(lowest_confs), 1)
        self.assertEqual(lowest_confs[0][0], 'C 4 0 0')
        self.assertEqual(lowest_confs[0][1], 5)

        # test a case where the number of confs is also the number to return:
        confs = [{'index': 0,
                  'FF energy': 20},
                 {'index': 1,
                  'FF energy': 10}]
        lowest_confs = conformers.get_lowest_confs(label='', confs=confs, n=2, energy='FF energy')
        self.assertEqual(len(lowest_confs), 2)

        # test a case where the number of confs is lower than the number to return:
        confs = [{'index': 0,
                  'FF energy': 20},
                 {'index': 1,
                  'some other energy': 10}]
        lowest_confs = conformers.get_lowest_confs(label='', confs=confs, n=2, energy='FF energy')
        self.assertEqual(len(lowest_confs), 1)  # only 1, not 2

    def test_compare_xyz(self):
        """Test determining whether two conformers have the same xyz to a certain precision"""
        xyz1 = """C       0.05984800   -0.62319600    0.00000000
H      -0.46898100   -1.02444400    0.87886100
H      -0.46898100   -1.02444400   -0.87886100
H       1.08093800   -1.00826200    0.00000000
N       0.05980600    0.81236000    0.00000000
H      -0.92102100    1.10943400    0.00000000"""
        xyz2 = """C       0.05984800   -0.62319600    0.00000000
H      -0.46898100   -1.02444400    0.87886100
H      -0.46898100   -1.02444400   -0.87886100
H       1.08093800   -1.00826200    0.00000000
N       0.05980600    0.81236000    0.00000000
H      -0.92102100    1.10943400    0.00000000"""  # identical to xyz1
        xyz3 = """C       0.05984800   -0.62319600    0.03
H      -0.46898100   -1.02444400    0.87886100
H      -0.46898100   -1.02444400   -0.87886100
H       1.08093800   -1.00826200    0.00000000
N       0.05980600    0.81236000    0.00000000
H      -0.92102100    1.10943400    0.004"""  # different within the default tolerance
        self.assertTrue(conformers.compare_xyz(xyz1, xyz2))
        self.assertTrue(conformers.compare_xyz(xyz1, xyz3))
        self.assertFalse(conformers.compare_xyz(xyz1, xyz3, precision=0.001))

    def test_generate_all_enantiomers(self):
        """Test generating all enantiomers"""
        xyz = """N       0.62388335   -0.46225337   -0.85033449
C       0.10472552    0.23488155    0.33281869
C      -1.43643918    0.23724156    0.28512538
C       0.61272054    1.67839609    0.41138541
C      -2.04365851   -1.15982532    0.33358533
C       2.03573696   -0.79863202   -0.74695173
H       0.41164123   -0.29851174    1.24164770
H      -1.82243200    0.80654659    1.14008874
H      -1.78964905    0.75063900   -0.61836546
H       0.17005107    2.19641387    1.26934951
H       0.35338297    2.24432808   -0.49038279
H       1.69920448    1.71770118    0.53711142
H      -3.13588073   -1.09287652    0.36674874
H      -1.71084751   -1.70051643    1.22528047
H      -1.77327921   -1.74785215   -0.54870906
H       2.32788522   -1.39534053   -1.61687715
H       2.67183227    0.09094319   -0.73043211
H       2.23743061   -1.40201550    0.14391187
H       0.46369194    0.10073248   -1.68500050"""
        spc = ARCSpecies(label='tst', smiles='CC(CC)(N(C))', xyz=xyz)  # Has two chiral centers (C, N)
        xyzs = conformers.generate_all_enantiomers(label='', mol=spc.mol, xyz=xyz)
        expected_xyzs = ['N       0.62388335   -0.46225337   -0.85033449\nC       0.10472552    0.23488155    0.3328186'
                         '9\nC      -1.43643918    0.23724156    0.28512538\nC       0.53320489   -0.50977899    '
                         '1.60161812\nC      -2.04365851   -1.15982532    0.33358533\nC       2.03573696   '
                         '-0.79863202   -0.74695173\nH       0.46859754    1.26885733    0.38909529\nH      '
                         '-1.82243200    0.80654659    1.14008874\nH      -1.78964905    0.75063900   '
                         '-0.61836546\nH      -0.34377313   -0.85107702    2.16271196\nH       1.12364860    '
                         '0.13330636    2.26384039\nH       1.13040483   -1.39642383    1.36707081\nH      '
                         '-3.13588073   -1.09287652    0.36674874\nH      -1.71084751   -1.70051643    1.22528047\n'
                         'H      -1.77327921   -1.74785215   -0.54870906\nH       2.32788522   -1.39534053   -1.6168771'
                         '5\nH       2.67183227    0.09094319   -0.73043211\nH       2.23743061   -1.40201550    '
                         '0.14391187\nH       0.46369194    0.10073248   -1.68500050\n',

                         'N       0.62388335   -0.46225337   -0.85033449\nC       0.10472552    0.23488155    '
                         '0.33281869\nC      -1.43643918    0.23724156    0.28512538\nC       0.53320489   '
                         '-0.50977899    1.60161812\nC      -2.04365851   -1.15982532    0.33358533\nC       '
                         '2.03573696   -0.79863202   -0.74695173\nH       0.46859754    1.26885733    0.38909529\n'
                         'H      -1.82243200    0.80654659    1.14008874\nH      -1.78964905    0.75063900   -0.6183654'
                         '6\nH      -0.34377313   -0.85107702    2.16271196\nH       1.12364860    0.13330636    '
                         '2.26384039\nH       1.13040483   -1.39642383    1.36707081\nH      -3.13588073   '
                         '-1.09287652    0.36674874\nH      -1.71084751   -1.70051643    1.22528047\nH      '
                         '-1.77327921   -1.74785215   -0.54870906\nH       2.32788522   -1.39534053   -1.61687715\n'
                         'H       2.67183227    0.09094319   -0.73043211\nH       2.23743061   -1.40201550    '
                         '0.14391187\nH       0.11514627   -1.34334139   -0.91477190\n',

                         'N       0.62388335   -0.46225337   -0.85033449\nC       0.10472552    0.23488155    '
                         '0.33281869\nC      -1.43643918    0.23724156    0.28512538\nC       0.61272054    '
                         '1.67839609    0.41138541\nC      -2.04365851   -1.15982532    0.33358533\nC       '
                         '2.03573696   -0.79863202   -0.74695173\nH       0.41164123   -0.29851174    1.24164770\n'
                         'H      -1.82243200    0.80654659    1.14008874\nH      -1.78964905    0.75063900   '
                         '-0.61836546\nH       0.17005107    2.19641387    1.26934951\nH       0.35338297    '
                         '2.24432808   -0.49038279\nH       1.69920448    1.71770118    0.53711142\nH      '
                         '-3.13588073   -1.09287652    0.36674874\nH      -1.71084751   -1.70051643    1.22528047\n'
                         'H      -1.77327921   -1.74785215   -0.54870906\nH       2.32788522   -1.39534053   '
                         '-1.61687715\nH       2.67183227    0.09094319   -0.73043211\nH       2.23743061   '
                         '-1.40201550    0.14391187\nH       0.11514627   -1.34334139   -0.91477190\n']
        self.assertEqual(xyzs, expected_xyzs)

    def test_identify_chiral_centers(self):
        mol1 = Molecule(SMILES=str('OC(N)N'))
        chiral_centers = conformers.identify_chiral_centers(mol1)
        self.assertEqual(chiral_centers, [])

        mol2 = Molecule(SMILES=str('OC(S)N'))
        chiral_centers = conformers.identify_chiral_centers(mol2)
        self.assertEqual(chiral_centers, [1])
        self.assertEqual(mol2.atoms[1].symbol, 'C')

        mol3 = Molecule(SMILES=str('COC1=C(C=C(C(C)C)C=C1)CN[C@H]2C3CCN([C@H]2C(C4=CC=CC=C4)C5=CC=CC=C5)CC3'))
        chiral_centers = conformers.identify_chiral_centers(mol3)
        self.assertEqual(chiral_centers, [19, 13, 12])
        self.assertEqual(mol3.atoms[19].symbol, 'C')
        self.assertEqual(mol3.atoms[12].symbol, 'N')
        self.assertEqual(mol3.atoms[13].symbol, 'C')

        mol4 = Molecule(SMILES=str('C1C(F)CC(F)C1'))
        # 1,3-Difluoro-cyclopentane has two chiral centers, but the molecule is NOT chiral due to symmetry.
        # This is a false positive!
        chiral_centers = conformers.identify_chiral_centers(mol4)
        self.assertEqual(chiral_centers, [1, 4])
        self.assertEqual(mol4.atoms[1].symbol, 'C')
        self.assertEqual(mol4.atoms[4].symbol, 'C')

        mol5 = Molecule(SMILES=str('C1C(F)CCC1'))  # not chiral due to symmetry
        chiral_centers = conformers.identify_chiral_centers(mol5)
        self.assertEqual(chiral_centers, [])

        mol6 = Molecule(SMILES=str('C1C(F)C=CC1'))  # chiral, symmetry broke by the double bond
        chiral_centers = conformers.identify_chiral_centers(mol6)
        self.assertEqual(chiral_centers, [1])
        self.assertEqual(mol6.atoms[1].symbol, 'C')

        mol7 = Molecule(SMILES=str('CC(C)CCc1ccc(C(C)C(=O)O)cc1'))  # ibuprofen, one chiral cunter
        chiral_centers = conformers.identify_chiral_centers(mol7)
        self.assertEqual(chiral_centers, [9])
        self.assertEqual(mol7.atoms[9].symbol, 'C')

        mol8 = Molecule(SMILES=str('CNCC'))  # simple umbrella mode
        chiral_centers = conformers.identify_chiral_centers(mol8)
        self.assertEqual(chiral_centers, [1])
        self.assertEqual(mol8.atoms[1].symbol, 'N')

        mol9 = Molecule(SMILES=str('NNNN'))  # two umbrella modes
        chiral_centers = conformers.identify_chiral_centers(mol9)
        self.assertEqual(chiral_centers, [1, 2])
        self.assertEqual(mol9.atoms[1].symbol, 'N')
        self.assertEqual(mol9.atoms[2].symbol, 'N')

        mol10 = Molecule(SMILES=str('NCC'))  # no umbrella modes
        chiral_centers = conformers.identify_chiral_centers(mol10)
        self.assertEqual(chiral_centers, [])

    def test_get_lp_vector(self):
        """Test the lone pair vector"""
        xyz1 = """O       1.13971727   -0.35763357   -0.91809799
N      -0.16022228   -0.63832421   -0.32863338
C      -0.42909096    0.49864538    0.54457751
H      -1.36471297    0.33135829    1.08632108
H       0.37059419    0.63632068    1.27966893
H      -0.53867601    1.41749835   -0.03987146
H       0.03832076   -1.45968957    0.24914206
H       0.94407000   -0.42817536   -1.87310674"""
        spc1 = ARCSpecies(label='tst1', smiles='CN(O)', xyz=xyz1)
        vector = conformers.get_lp_vector(label='', mol=spc1.mol, xyz=xyz1, pivot=1)
        self.assertAlmostEqual(vector[0], -0.301081, 5)
        self.assertAlmostEqual(vector[1], -0.056692, 5)
        self.assertAlmostEqual(vector[2], -0.252623, 5)
        # puts the following dummy atom in xyz1: 'Cl     -0.4613 -0.6950 -0.5813'

    def test_get_vector(self):
        """Test getting a vector between two atoms in the molecule"""
        xyz1 = """O      0.0   0.0    0.0
N      1.0    0.0   0.0"""  # trivial
        vector = conformers.get_vector(pivot=0, anchor=1, xyz=xyz1)
        self.assertAlmostEqual(vector[0], 1.0, 5)
        self.assertAlmostEqual(vector[1], 0.0, 5)
        self.assertAlmostEqual(vector[2], 0.0, 5)

        xyz2 = """O      -0.39141517   -1.49218505    0.23537907
N      -1.29594218    0.36660772   -0.33360920
C      -0.24369399   -0.21522785    0.47237314
C       1.11876670    0.24246665   -0.06138419
H      -0.34055624    0.19728442    1.48423848
H       1.27917500   -0.02124533   -1.11576163
H       1.93896021   -0.20110894    0.51754953
H       1.21599040    1.33219465    0.01900272
H      -2.12405283   -0.11420423    0.01492411
H      -1.15723190   -0.09458204   -1.23271202"""  # smiles='NC([O])(C)'
        vector = conformers.get_vector(pivot=1, anchor=2, xyz=xyz2)
        self.assertAlmostEqual(vector[0], 1.052248, 5)
        self.assertAlmostEqual(vector[1], -0.581836, 5)
        self.assertAlmostEqual(vector[2], 0.805982, 5)

    def test_rotate_vector(self):
        """Test rotating a vector"""
        point_a, point_b, normal, theta = [0, 0, 0], [0, 0, 1], [0, 0, 1], 90 * math.pi / 180  # trivial, no rotation
        new_vector = conformers.rotate_vector(point_a, point_b, normal, theta)
        self.assertEqual(new_vector, [0, 0, 1])

        point_a, point_b, normal, theta = [0, 0, 0], [1, 0, 0], [0, 0, 1], 90 * math.pi / 180  # rotate x to y around z
        new_vector = conformers.rotate_vector(point_a, point_b, normal, theta)
        self.assertAlmostEqual(new_vector[0], 0, 5)
        self.assertAlmostEqual(new_vector[1], 1, 5)
        self.assertAlmostEqual(new_vector[2], 0, 5)

        point_a, point_b, normal, theta = [0, 0, 0], [3, 5, 0], [4, 4, 1], 1.2
        new_vector = conformers.rotate_vector(point_a, point_b, normal, theta)
        self.assertAlmostEqual(new_vector[0], 2.749116, 5)
        self.assertAlmostEqual(new_vector[1], 4.771809, 5)
        self.assertAlmostEqual(new_vector[2], 1.916297, 5)

    def test_unit_vector(self):
        """Test calculating a unit vector"""
        v1 = [1, 0, 0]
        self.assertEqual(conformers.unit_vector(v1)[0], 1.)  # trivial
        self.assertEqual(conformers.unit_vector(v1)[1], 0.)  # trivial
        self.assertEqual(conformers.unit_vector(v1)[2], 0.)  # trivial
        v2 = [1, 1, 1]
        self.assertAlmostEqual(conformers.unit_vector(v2)[0], (1 / 3) ** 0.5)
        self.assertAlmostEqual(conformers.unit_vector(v2)[1], (1 / 3) ** 0.5)
        self.assertAlmostEqual(conformers.unit_vector(v2)[2], (1 / 3) ** 0.5)

    def test_get_normal(self):
        """Test calculating a normal vector"""
        v1 = [6, 0, 0]
        v2 = [0, 3, 0]
        n = conformers.get_normal(v1, v2)
        self.assertEqual(n[0], 0)
        self.assertEqual(n[1], 0)
        self.assertEqual(n[2], 1)

        v1 = [5, 1, 1]
        v2 = [1, 8, 2]
        n = conformers.get_normal(v1, v2)
        expected_n = conformers.unit_vector([-6, -9, 39])
        for ni, expected_ni in zip(n, expected_n):
            self.assertEqual(ni, expected_ni)

    def test_get_theta(self):
        """Test calculating the angle between two vectors"""
        v1 = [-1.45707856 + 0.02416711, -0.94104506 - 0.17703194, -0.20275830 - 0.08644641]
        v2 = [-0.03480906 + 0.02416711, 1.11948179 - 0.17703194, -0.82988874 - 0.08644641]
        theta = conformers.get_theta(v1, v2)
        self.assertAlmostEqual(theta, 1.8962295, 5)
        self.assertAlmostEqual(theta * 180 / math.pi, 108.6459, 3)

    def test_translate_group(self):
        """Test translating groups within a molecule"""
        xyz1 = """O       1.40486421    0.01953338    0.00000000
O      -1.40486421   -0.01953339    0.00000000
C       0.00000000    0.00000000    0.00000000"""
        spc1 = ARCSpecies(label='CO2', smiles='O=C=O', xyz=xyz1)
        # translate the first O by 90 degrees:
        new_xyz1 = conformers.translate_group(mol=spc1.mol, xyz=xyz1, pivot=2, anchor=0, vector=[0, 0, 1])
        expected_xyz1 = """O       0.00000000    0.00000000    1.40500000
O      -1.40486421   -0.01953339    0.00000000
C       0.00000000    0.00000000    0.00000000
"""
        self.assertEqual(new_xyz1, expected_xyz1)

        xyz2 = """Cl      1.47512188   -0.78746253   -0.20393322
S      -1.45707856   -0.94104506   -0.20275830
O      -0.03480906    1.11948179   -0.82988874
C      -0.02416711    0.17703194    0.08644641
H       0.04093286    0.43199386    1.15013385"""
        spc2 = ARCSpecies(label='chiral', smiles='[S]C([O])Cl', xyz=xyz2)
        vector1 = conformers.unit_vector(conformers.get_vector(pivot=3, anchor=1, xyz=xyz2))
        vector2 = conformers.unit_vector(conformers.get_vector(pivot=3, anchor=2, xyz=xyz2))
        new_xyz2 = conformers.translate_group(mol=spc2.mol, xyz=xyz2, pivot=3, anchor=2, vector=vector1)
        new_xyz2 = conformers.translate_group(mol=spc2.mol, xyz=new_xyz2, pivot=3, anchor=1, vector=vector2)
        expected_xyz2 = """Cl      1.47512188   -0.78746253   -0.20393322
S      -0.03906606    1.49648129   -1.19644182
O      -1.04766012   -0.62158265   -0.12012532
C      -0.02416711    0.17703194    0.08644641
H       0.04093286    0.43199386    1.15013385
"""
        self.assertEqual(new_xyz2, expected_xyz2)

        xyz3 = """ C                  1.50048866   -0.50848248   -0.64006761
 F                  0.27568368    0.01156702   -0.41224910
 Cl                 3.09727149   -1.18647281   -0.93707550
 O                  1.93561914    0.74741610   -1.16759033
 O                  2.33727805    1.90670710   -1.65453438
 H                  3.25580294    1.83954553   -1.92546105"""
        spc3 = ARCSpecies(label='chiralOOH', smiles='OO[C](F)Cl', xyz=xyz3)
        vector1 = conformers.unit_vector(conformers.get_vector(pivot=0, anchor=1, xyz=xyz3))
        vector2 = conformers.unit_vector(conformers.get_vector(pivot=0, anchor=3, xyz=xyz3))
        new_xyz3 = conformers.translate_group(mol=spc3.mol, xyz=xyz3, pivot=0, anchor=3, vector=vector1)
        new_xyz3 = conformers.translate_group(mol=spc3.mol, xyz=new_xyz3, pivot=0, anchor=1, vector=vector2)
        expected_xyz3 = """C       1.50048866   -0.50848248   -0.64006761
F       1.91127618    0.67715604   -1.13807857
Cl      3.09727149   -1.18647281   -0.93707550
O       0.20310264    0.04238477   -0.39874874
O      -0.99448445    0.55087762   -0.17599287
H      -1.00986076    1.46908426   -0.45574283
"""
        self.assertEqual(new_xyz3, expected_xyz3)

        xyz4 = """ C                  1.18149528   -0.70041459   -0.31471741
 C                  0.93123100    0.49135224   -1.25740484
 C                  0.68096672    1.68311908   -2.20009227
 C                  0.90913879    1.54885051   -3.56990280
 C                  0.22609012    2.89672219   -1.68453149
 C                  0.68310320    2.62819874   -4.42379802
 H                  1.26848882    0.59217080   -3.97594868
 C                 -0.00100499    3.97619810   -2.53867988
 H                  0.04603318    3.00266663   -0.60495551
 C                  0.22754321    3.84217921   -3.90811103
 H                  0.86352778    2.52263257   -5.50342726
 H                 -0.36004784    4.93282504   -2.13195230
 H                  0.04933714    4.69304911   -4.58157919
 F                  0.25569319   -1.42216987   -0.98139560
 Cl                 2.38846685    0.24054066    0.55443324"""
        spc4 = ARCSpecies(label='chiral_benzene', smiles='ClC(F)=[C]c1ccccc1', xyz=xyz4)
        vector1 = conformers.unit_vector(conformers.get_vector(pivot=0, anchor=1, xyz=xyz4))
        vector2 = conformers.unit_vector(conformers.get_vector(pivot=0, anchor=13, xyz=xyz4))
        new_xyz4 = conformers.translate_group(mol=spc4.mol, xyz=xyz4, pivot=0, anchor=13, vector=vector1)
        new_xyz4 = conformers.translate_group(mol=spc4.mol, xyz=new_xyz4, pivot=0, anchor=1, vector=vector2)
        expected_xyz4 = """C       1.18149528   -0.70041459   -0.31471741
C       0.12539512   -1.52375024   -1.07522438
C      -0.93070504   -2.34708589   -1.83573135
C      -0.56553338   -3.53564257   -2.46854578
C      -2.25231991   -1.90466577   -1.89182711
C      -1.52168662   -4.28119970   -3.15784780
H       0.47664391   -3.88377085   -2.42478536
C      -3.20909105   -2.65084001   -2.58046554
H      -2.54034188   -0.96801190   -1.39297653
C      -2.84398183   -3.83887752   -3.21355255
H      -1.23379910   -5.21772042   -3.65719364
H      -4.25113565   -2.30197553   -2.62417846
H      -3.59771174   -4.42692116   -3.75705255
F       0.96210776    0.34431608   -1.14109925
Cl      2.38846685    0.24054066    0.55443324
"""
        self.assertEqual(new_xyz4, expected_xyz4)

    def test_translate_groups(self):
        """Test converting a chiral center"""
        xyz1 = """Cl      1.38346248    1.33352376    0.05890374
O       0.42405110   -0.73855006    1.08316776
N      -0.85054134    0.17787474   -0.37480771
C       0.50839421   -0.20402577   -0.10066066
H       0.92397006   -0.71586300   -0.97353222
H      -1.03560134    0.80870122    0.40254095
H      -1.35373518   -0.66166089   -0.09561187"""
        spc1 = ARCSpecies(label='chiral1', smiles='[O]C(N)Cl', xyz=xyz1)
        new_xyz1 = conformers.translate_groups(label='', mol=spc1.mol, xyz=xyz1, pivot=3)
        expected_xyz1 = """Cl      1.38346248    1.33352376    0.05890374
O       1.00289957   -0.81307525   -1.13931502
N      -0.85054134    0.17787474   -0.37480771
C       0.50839421   -0.20402577   -0.10066066
H       0.43751336   -0.65323300    0.89421328
H      -1.03560134    0.80870122    0.40254095
H      -1.35373518   -0.66166089   -0.09561187
"""
        self.assertEqual(new_xyz1, expected_xyz1)

        xyz2 = """O      -1.74064909    0.21906484    1.02158943
O      -3.01958653    0.57362697    0.39911927
N      -0.78688194    1.38451169   -0.85929405
C      -0.73454332    0.17093577   -0.02005703
C       0.62187688   -0.00862010    0.71286218
C      -1.04891412   -1.03276103   -0.95404861
C       1.84944766   -0.12402968   -0.19586144
C      -1.10146025   -2.38752650   -0.25446980
C       3.13114267   -0.21889544    0.61917595
C      -0.46924454    2.61449953   -0.14264770
H      -0.32123827   -1.08504290   -1.77318159
H      -2.02817065   -0.88512565   -1.42900304
H       0.56892958   -0.89247123    1.36248079
H       0.76679334    0.81765871    1.42197172
H       1.91889368    0.74255605   -0.86175510
H       1.77338036   -1.01540850   -0.82711171
H      -1.42131670   -3.15882414   -0.96300156
H      -0.11988614   -2.68251157    0.12681712
H      -1.81257708   -2.38273188    0.57692021
H       3.11494498   -1.09512766    1.27514598
H       3.26896265    0.67297161    1.23878643
H       3.99659234   -0.30770128   -0.04496554
H       0.59165280    2.67290981    0.11637148
H      -1.07160900    2.73729811    0.76280673
H      -0.67945328    3.46885484   -0.79455921
H      -1.76710371    1.45567982   -1.14227021
H      -3.47998233   -0.25379020    0.62817931"""
        spc2 = ARCSpecies(label='chiral', smiles='CCC(CCC)(NC)OO', xyz=xyz2)
        new_xyz2 = conformers.translate_groups(label='', mol=spc2.mol, xyz=xyz2, pivot=3)
        expected_xyz2 = ["""O      -0.78590971    1.36196848   -0.84370450
O      -0.80029118    0.94072052   -2.24770155
N      -1.75969211    0.21997580    1.04130514
C      -0.73454332    0.17093577   -0.02005703
C       0.62187688   -0.00862010    0.71286218
C      -1.04891412   -1.03276103   -0.95404861
C       1.84944766   -0.12402968   -0.19586144
C      -1.10146025   -2.38752650   -0.25446980
C       3.13114267   -0.21889544    0.61917595
C      -1.21205506    0.25580387    2.39265839
H      -0.32123827   -1.08504290   -1.77318159
H      -2.02817065   -0.88512565   -1.42900304
H       0.56892958   -0.89247123    1.36248079
H       0.76679334    0.81765871    1.42197172
H       1.91889368    0.74255605   -0.86175510
H       1.77338036   -1.01540850   -0.82711171
H      -1.42131670   -3.15882414   -0.96300156
H      -0.11988614   -2.68251157    0.12681712
H      -1.81257708   -2.38273188    0.57692021
H       3.11494498   -1.09512766    1.27514598
H       3.26896265    0.67297161    1.23878643
H       3.99659234   -0.30770128   -0.04496554
H      -0.73861859    1.21618224    2.61522781
H      -0.49710699   -0.55257308    2.57471917
H      -2.03077149    0.13706745    3.11008741
H      -2.26046932   -0.66728712    0.95207613
H      -1.67555316    1.31253869   -2.45917185
""",
                         """O      -1.74064909    0.21906484    1.02158943
O      -3.01958653    0.57362697    0.39911927
N      -1.03290314   -0.97145646   -0.90648019
C      -0.73454332    0.17093577   -0.02005703
C       0.62187688   -0.00862010    0.71286218
C      -0.78969060    1.44963621   -0.90433030
C       1.84944766   -0.12402968   -0.19586144
C      -1.17776916    1.20403119   -2.35919603
C       3.13114267   -0.21889544    0.61917595
C      -1.23934688   -0.60042638   -2.30185228
H       0.17141663    1.97767302   -0.87836134
H      -1.53008279    2.14978675   -0.49453779
H       0.56892958   -0.89247123    1.36248079
H       0.76679334    0.81765871    1.42197172
H       1.91889368    0.74255605   -0.86175510
H       1.77338036   -1.01540850   -0.82711171
H      -1.28563773    2.16074307   -2.88101228
H      -0.41314728    0.62875010   -2.88861901
H      -2.13101472    0.67270327   -2.43622219
H       3.11494498   -1.09512766    1.27514598
H       3.26896265    0.67297161    1.23878643
H       3.99659234   -0.30770128   -0.04496554
H      -0.30716524   -0.29291833   -2.78401621
H      -1.98629051    0.19114299   -2.41701870
H      -1.60173671   -1.47553562   -2.85135820
H      -1.92677503   -1.33188001   -0.56433542
H      -3.47998233   -0.25379020    0.62817931
""",
                         """O      -1.02736085   -0.95023555   -0.89001412
O      -1.42581639   -0.41726180   -2.19616746
N      -0.78688194    1.38451169   -0.85929405
C      -0.73454332    0.17093577   -0.02005703
C       0.62187688   -0.00862010    0.71286218
C      -1.81470500    0.22260745    1.09826137
C       1.84944766   -0.12402968   -0.19586144
C      -1.26707632    0.13898856    2.51976472
C       3.13114267   -0.21889544    0.61917595
C      -0.46924454    2.61449953   -0.14264770
H      -2.55465373   -0.57459215    0.95622951
H      -2.37135748    1.16695968    1.03001061
H       0.56892958   -0.89247123    1.36248079
H       0.76679334    0.81765871    1.42197172
H       1.91889368    0.74255605   -0.86175510
H       1.77338036   -1.01540850   -0.82711171
H      -2.08041564    0.27903125    3.23955126
H      -0.81795913   -0.83741122    2.72180645
H      -0.51814948    0.91403927    2.70771275
H       3.11494498   -1.09512766    1.27514598
H       3.26896265    0.67297161    1.23878643
H       3.99659234   -0.30770128   -0.04496554
H       0.59165280    2.67290981    0.11637148
H      -1.07160900    2.73729811    0.76280673
H      -0.67945328    3.46885484   -0.79455921
H      -1.76710371    1.45567982   -1.14227021
H      -0.68937317   -0.81042410   -2.69828972
"""]
        self.assertIn(new_xyz2, expected_xyz2)

        xyz3 = """S       1.54673527   -0.35827061    1.02876774
O       0.53564085    1.18674431   -0.72188033
N       0.18926060    0.37718561    0.43275531
C      -0.95531219   -0.42315058   -0.00040366
H      -1.32965925   -1.03967804    0.82436632
H      -1.77871076    0.22815029   -0.31356151
H      -0.69921453   -1.08376479   -0.83670429
H       0.53243278    2.07205105   -0.31434209
H       1.95882724   -0.95926723   -0.09899749"""
        spc3 = ARCSpecies(label='N_lp', smiles='CN(O)S', xyz=xyz3)
        new_xyz3 = conformers.translate_groups(label='', mol=spc3.mol, xyz=xyz3, pivot=2)
        expected_xyz3 = ["""S       1.54673527   -0.35827061    1.02876774
O      -0.25905177    1.08235459    1.62031616
N       0.18926060    0.37718561    0.43275531
C      -0.95531219   -0.42315058   -0.00040366
H      -1.32965925   -1.03967804    0.82436632
H      -1.77871076    0.22815029   -0.31356151
H      -0.69921453   -1.08376479   -0.83670429
H      -0.24668428    0.35169809    2.26518347
H       1.95882724   -0.95926723   -0.09899749
""",
                         """S       1.54673527   -0.35827061    1.02876774
O       0.53564085    1.18674431   -0.72188033
N       0.18926060    0.37718561    0.43275531
C      -0.26219423    1.08729751    1.62864042
H       0.42961485    0.91955940    2.46155786
H      -0.28639979    2.16692009    1.44404620
H      -1.26281273    0.76700514    1.94100449
H       0.53243278    2.07205105   -0.31434209
H       1.95882724   -0.95926723   -0.09899749
"""]
        self.assertIn(new_xyz3, expected_xyz3)

    def test_calculate_dihedral_angle(self):
        """Test calculating a dihedral angle"""
        propene = converter.get_xyz_matrix("""C       1.22905000   -0.16449200    0.00000000
C      -0.13529200    0.45314000    0.00000000
C      -1.27957200   -0.21983000    0.00000000
H       1.17363000   -1.25551200    0.00000000
H       1.79909600    0.15138400    0.87934300
H       1.79909600    0.15138400   -0.87934300
H      -0.16831500    1.54137600    0.00000000
H      -2.23664600    0.28960500    0.00000000
H      -1.29848800   -1.30626200    0.00000000""")[0]
        hydrazine = converter.get_xyz_matrix("""N       0.70683700   -0.07371000   -0.21400700
N      -0.70683700    0.07371000   -0.21400700
H       1.11984200    0.81113900   -0.47587600
H       1.07456200   -0.35127300    0.68988300
H      -1.11984200   -0.81113900   -0.47587600
H      -1.07456200    0.35127300    0.68988300""")[0]
        cj_11974 = converter.get_xyz_matrix("""C 	5.675	2.182	1.81
O 	4.408	1.923	1.256
C 	4.269	0.813	0.479
C 	5.303	-0.068	0.178
C 	5.056	-1.172	-0.639
C 	3.794	-1.414	-1.169
C 	2.77	-0.511	-0.851
C 	2.977	0.59	-0.032
C 	1.872	1.556	0.318
N 	0.557	1.029	-0.009
C 	-0.537	1.879	0.448
C 	-0.535	3.231	-0.298
C 	-1.831	3.983	0.033
C 	-3.003	3.199	-0.61
N 	-2.577	1.854	-0.99
C 	-1.64	1.962	-2.111
C 	-0.501	2.962	-1.805
C 	-1.939	1.236	0.178
C 	-1.971	-0.305	0.069
C 	-3.385	-0.794	-0.209
C 	-4.336	-0.893	0.81
C 	-5.631	-1.324	0.539
C 	-5.997	-1.673	-0.759
C 	-5.056	-1.584	-1.781
C 	-3.764	-1.147	-1.505
C 	-1.375	-1.024	1.269
C 	-1.405	-0.508	2.569
C 	-0.871	-1.226	3.638
C 	-0.296	-2.475	3.429
C 	-0.259	-3.003	2.14
C 	-0.794	-2.285	1.078
C 	3.533	-2.614	-2.056
C 	2.521	-3.574	-1.424
C 	3.087	-2.199	-3.461
H 	5.569	3.097	2.395
H 	6.433	2.338	1.031
H 	6.003	1.368	2.47
H 	6.302	0.091	0.57
H 	5.874	-1.854	-0.864
H 	1.772	-0.654	-1.257
H 	1.963	1.832	1.384
H 	2.033	2.489	-0.239
H 	0.469	0.13	0.461
H 	-0.445	2.089	1.532
H 	0.328	3.83	0.012
H 	-1.953	4.059	1.122
H 	-1.779	5.008	-0.352
H 	-3.365	3.702	-1.515
H 	-3.856	3.118	0.074
H 	-1.226	0.969	-2.31
H 	-2.211	2.259	-2.999
H 	-0.639	3.906	-2.348
H 	0.466	2.546	-2.105
H 	-2.586	1.501	1.025
H 	-1.36	-0.582	-0.799
H 	-4.057	-0.647	1.831
H 	-6.355	-1.396	1.347
H 	-7.006	-2.015	-0.97
H 	-5.329	-1.854	-2.798
H 	-3.038	-1.07	-2.311
H 	-1.843	0.468	2.759
H 	-0.904	-0.802	4.638
H 	0.125	-3.032	4.262
H 	0.189	-3.977	1.961
H 	-0.772	-2.708	0.075
H 	4.484	-3.155	-2.156
H 	1.543	-3.093	-1.308
H 	2.383	-4.464	-2.049
H 	2.851	-3.899	-0.431
H 	3.826	-1.542	-3.932
H 	2.134	-1.659	-3.429
H 	2.951	-3.078	-4.102""")[0]

        dihedral0 = conformers.calculate_dihedral_angle(coord=propene, torsion=[9, 3, 2, 7])
        dihedral1 = conformers.calculate_dihedral_angle(coord=propene, torsion=[5, 1, 2, 7])
        self.assertAlmostEqual(dihedral0, 180, 2)
        self.assertAlmostEqual(dihedral1, 59.26447, 2)

        dihedral2 = conformers.calculate_dihedral_angle(coord=hydrazine, torsion=[3, 1, 2, 5])
        self.assertAlmostEqual(dihedral2, 148.31829, 2)

        dihedral3 = conformers.calculate_dihedral_angle(coord=cj_11974, torsion=[15, 18, 19, 20])
        self.assertAlmostEqual(dihedral3, 308.04758, 2)

################################################################################


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
