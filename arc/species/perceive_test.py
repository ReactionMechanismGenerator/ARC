#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.species.perceive module
"""

import os
import unittest
import numpy as np

from arc.common import ARC_PATH
from arc.parser.parser import parse_geometry
from arc.species import ARCSpecies
from arc.species.converter import str_to_xyz
from arc.species.perceive import (
    get_representative_resonance_structure,
    infer_multiplicity,
    perceive_molecule_from_xyz,
    validate_xyz,
    xyz_to_str,
)
from arc.molecule.molecule import Molecule


class TestPerceive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A perfectly tetrahedral methane
        cls.xyz_methane = {'symbols': ('C', 'H', 'H', 'H', 'H'), 'isotopes': (12, 1, 1, 1, 1),
                           'coords': ((0.49999999, 0.25, 0.0), (0.85665442, -0.75881001, 0.0),
                                      (0.85667283, 0.75439819, 0.8736515), (0.85667283, 0.75439819, -0.8736515),
                                      (-0.57000001, 0.25001318, 0.0))}
        # A bent water molecule
        cls.xyz_water = {
            'symbols': ('O', 'H', 'H'),
            'isotopes': (16, 1, 1),
            'coords': (
                (0.0, 0.0, 0.0),
                (0.96, 0.0, 0.0),
                (-0.24, 0.93, 0.0),
            )}
        # A single hydrogen atom (radical)
        cls.xyz_h = {
            'symbols': ('H',),
            'isotopes': (1,),
            'coords': ((0.0, 0.0, 0.0),)}
        # An invalid xyz
        cls.xyz_invalid = {
            'invalid': ('C', 5),
            'coords': (0.0, 0.0, 0.0)}
        cls.xyz_crazy_oxy_s_with_n = """S      -0.06618943   -0.12360663   -0.07631983
                                        O      -0.79539707    0.86755487    1.02675668
                                        O      -0.68919931    0.25421823   -1.34830853
                                        N       0.01546439   -1.54297548    0.44580391
                                        C       1.59721519    0.47861334    0.00711000
                                        H       1.94428095    0.40772394    1.03719428
                                        H       2.20318015   -0.14715186   -0.64755729
                                        H       1.59252246    1.51178950   -0.33908352
                                        H      -0.87856890   -2.02453514    0.38494433
                                        H      -1.34135876    1.49608206    0.53295071"""
        cls.xyz_oh_anion = """O    0.0000000    0.0000000    0.1075630
                              H    0.0000000    0.0000000   -0.8605010"""
        cls.xyz_co3_anion = """C    0.0000000    0.0000000    0.0000000
                               O    0.0000000    1.3065800    0.0000000
                               O    1.1315310   -0.6532900    0.0000000
                               O   -1.1315310   -0.6532900    0.0000000"""
        cls.xyz_nh4_cation = """N    0.0000000    0.0000000    0.0000000
                                H    0.5888990    0.5888990    0.5888990
                                H   -0.5888990   -0.5888990    0.5888990
                                H   -0.5888990    0.5888990   -0.5888990
                                H    0.5888990   -0.5888990   -0.5888990"""
        cls.xyz_h3o_cation = """O    0.0000000    0.0000000    0.0776780
                                H    0.0000000    0.9325950   -0.2071400
                                H    0.8076510   -0.4662980   -0.2071400
                                H   -0.8076510   -0.4662980   -0.2071400"""
        cls.xyz_conjugated_oxy_benzene = """O       2.64631000   -0.59546000    0.29327900
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
        cls.xyz_ozone = """O       0.90973400   -0.03064000   -0.09605500
                           O       0.31656600   -0.00477100   -1.21127600
                           O       2.17315400   -0.03069900   -0.09349100"""
        cls.xyz_hno3 = """O      -0.79494500   -0.93969200    0.00000000
                          O      -0.32753500    1.24003800    0.00000000
                          O       1.28811400   -0.24729000    0.00000000
                          N       0.14143500    0.11571500    0.00000000
                          H      -1.65602000   -0.48026800    0.00000000"""
        cls.xyz_n3o2 = """O       1.64973000   -0.57433600    0.02610800
                          O       0.49836300    1.28744800   -0.18806200
                          N      -0.57621600   -0.65116600    0.24595200
                          N      -1.78357200   -0.10211200   -0.14953800
                          N       0.61460400    0.08152700   -0.00952700
                          H      -0.42001200   -1.61494900   -0.03311600
                          H      -1.72480300    0.33507600   -1.06884500
                          H      -2.07362100    0.59363400    0.53038600"""
        cls.xyz_no2 = """O       1.10621000    0.00000000   -0.13455300
                         O      -1.10621000    0.00000000   -0.13455300
                         N       0.00000000    0.00000000    0.33490500"""
        cls.xyz_ch3n2oh = """O      -0.37723000   -1.27051900    0.00000000
                             N      -0.12115000   -0.04252600    0.00000000
                             N      -0.95339100    0.91468300    0.00000000
                             C       1.31648000    0.33217600    0.00000000
                             H       1.76422500   -0.11051900   -0.89038300
                             H       1.76422500   -0.11051900    0.89038300
                             H       1.40045900    1.41618100    0.00000000
                             H      -1.88127600    0.47189500    0.00000000"""
        cls.xyz_ch2no = """O       1.21678000   -0.01490600    0.00000000
                           N       0.04560300    0.35628400    0.00000000
                           C      -1.08941100   -0.23907800    0.00000000
                           H      -1.97763400    0.37807800    0.00000000
                           H      -1.14592100   -1.32640500    0.00000000"""
        cls.xyz_h2so4 = """S       0.00000000    0.00000000    0.18275300
                           O      -0.94981300   -0.83167500   -0.84628900
                           O       0.94981300    0.83167500   -0.84628900
                           O       0.80426500   -0.99804200    0.85548500
                           O      -0.80426500    0.99804200    0.85548500
                           H      -1.67833300   -0.25442300   -1.13658700
                           H       1.67833300    0.25442300   -1.13658700"""
        cls.xyz_so3 = """S       0.00000000    0.00000000    0.12264300
                         O       1.45413200    0.00000000    0.12264300
                         O      -0.72706600    1.25931500    0.12264300
                         O      -0.72706600   -1.25931500    0.12264300"""
        cls.xyz_cyc_c3n2 = """N       1.16672400    0.35870400   -0.00000400
                              N      -1.16670800    0.35879500   -0.00000400
                              C      -0.73775600   -0.89086600   -0.00000100
                              C       0.73767000   -0.89093000   -0.00000100
                              C       0.00005200    1.08477000   -0.00000500
                              H      -1.40657400   -1.74401100    0.00000000
                              H       1.40645000   -1.74411900    0.00000000
                              H       0.00009400    2.16788100   -0.00000700"""
        cls.xyz_c6_dd_yl = """C       3.09980400   -0.16068000    0.00000600
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
        cls.xyz_c3_linear_cn = """N       2.24690600   -0.00006500    0.11597700
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
        cls.aibn = """C      -0.86594600    0.19886100    2.37159000
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
        cls.xyz_hsn = """S       0.38431300    0.05370100    0.00000000
                         N      -1.13260000    0.07859900    0.00000000
                         H       0.85151800   -1.28998600    0.00000000"""
        cls.xyz_h2nn_singlet = """N       0.00000000    0.00000000    0.44654700
                                  N       0.00000000    0.00000000   -0.77510900
                                  H       0.86709400    0.00000000    1.02859700
                                  H      -0.86709400    0.00000000    1.02859700"""
        cls.xyz_hcn = """N       0.00000000    0.00000000    0.65631400
                         C       0.00000000    0.00000000   -0.50136500
                         H       0.00000000    0.00000000   -1.57173600"""
        cls.xyz_sn2 = """S      -0.00866000   -0.60254900    0.00000000
                         N      -0.96878800    0.63275900    0.00000000
                         N       1.01229100    0.58298500    0.00000000"""
        cls.xyz_n2h2_triplet = {'coords': ((0.5974274138372041, -0.41113104979405946, 0.08609839663782763),
                                           (-0.5974274348582206, 0.41113108883884353, -0.08609846602622732),
                                           (1.421955422639823, 0.19737093442024492, 0.02508578507394823),
                                           (-1.4219554016188147, -0.19737097346502322, -0.02508571568554942)),
                                'isotopes': (14, 1, 14, 1),
                                'symbols': ('N', 'N', 'H', 'H')}
        cls.xyz_ccooj = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'),
                         'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1),
                         'coords': ((-1.10653, -0.06552, 0.042602),
                                    (0.385508, 0.205048, 0.049674),
                                    (0.759622, 1.114927, -1.032928),
                                    (0.675395, 0.525342, -2.208593),
                                    (-1.671503, 0.860958, 0.166273),
                                    (-1.396764, -0.534277, -0.898851),
                                    (-1.36544, -0.740942, 0.862152),
                                    (0.97386, -0.704577, -0.082293),
                                    (0.712813, 0.732272, 0.947293),
                                    )}
        cls.xyz_change_chirality = {'coords': ((1.38346248, 1.33352376, 0.05890374),
                                               (0.4240511, -0.73855006, 1.08316776),
                                               (-0.85054134, 0.17787474, -0.37480771),
                                               (0.50839421, -0.20402577, -0.10066066),
                                               (0.92397006, -0.715863, -0.97353222),
                                               (-1.03560134, 0.80870122, 0.40254095),
                                               (-1.35373518, -0.66166089, -0.09561187)),
                                    'isotopes': (35, 16, 14, 12, 1, 1, 1),
                                    'symbols': ('Cl', 'O', 'N', 'C', 'H', 'H', 'H')}

    def test_validate_xyz(self):
        """
        Test that the validate_xyz function correctly identifies valid and invalid XYZ formats.
        """
        self.assertTrue(validate_xyz(self.xyz_methane))
        self.assertTrue(validate_xyz(self.xyz_water))
        self.assertTrue(validate_xyz(self.xyz_h))
        self.assertIsNone(validate_xyz(self.xyz_invalid))

    def test_infer_multiplicity(self):
        """
        Test that the multiplicity inference works correctly based on the number of electrons.
        """
        # CH4 has 6 + 4 = 10 electrons → singlet
        self.assertEqual(infer_multiplicity(['C', 'H', 'H', 'H', 'H'], total_charge=0), 1)
        # single H has 1 electron → doublet
        self.assertEqual(infer_multiplicity(['H'], total_charge=0), 2)
        # OH has 8 + 1 = 8 electrons → doublet
        self.assertEqual(infer_multiplicity(['O', 'H'], total_charge=0), 2)
        # OH– has 8 + 1 – 1 = 8 electrons → singlet
        self.assertEqual(infer_multiplicity(['O', 'H'], total_charge=-1), 1)

    def test_xyz_to_str(self):
        """Test that the XYZ dictionary can be converted to a string format."""
        xyz_str = xyz_to_str(self.xyz_methane)
        expected_str = """C       0.49999999    0.25000000    0.00000000
H       0.85665442   -0.75881001    0.00000000
H       0.85667283    0.75439819    0.87365150
H       0.85667283    0.75439819   -0.87365150
H      -0.57000001    0.25001318    0.00000000"""
        self.assertEqual(xyz_str, expected_str)


    def test_perceive_methane(self):
        """
        Test that perceiving a methane molecule (CH4) works correctly.
        """
        mol = perceive_molecule_from_xyz(self.xyz_methane, charge=0, multiplicity=None)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 5)
        edges = mol.get_all_edges()
        # There should be exactly 4 C–H bonds
        self.assertEqual(len(edges), 4)
        for edge in edges:
            self.assertEqual(edge.order, 1)
            elems = {edge.vertex1.element, edge.vertex2.element}
            self.assertIn([str(e) for e in elems], [['C', 'H'], ['H', 'C']])
        # No radicals, no formal charges
        for atom in mol.atoms:
            self.assertEqual(atom.radical_electrons, 0)
            self.assertEqual(atom.charge, 0)
        # Multiplicity is singlet
        self.assertEqual(mol.multiplicity, 1)

    def test_perceive_water(self):
        """
        Test that perceiving a water molecule (H2O) works correctly.
        """
        mol = perceive_molecule_from_xyz(self.xyz_water)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 3)
        edges = mol.get_all_edges()
        # Exactly two O–H bonds
        self.assertEqual(len(edges), 2)
        for edge in edges:
            self.assertEqual(edge.order, 1)
            elems = {str(edge.vertex1.element), str(edge.vertex2.element)}
            self.assertEqual(elems, {'O', 'H'})
        # O should carry two lone pairs
        oxy = next(a for a in mol.atoms if a.element.symbol == 'O')
        self.assertEqual(oxy.lone_pairs, 2)
        self.assertEqual(oxy.radical_electrons, 0)
        self.assertEqual(oxy.charge, 0)
        # Water is a singlet
        self.assertEqual(mol.multiplicity, 1)

    def test_perceive_atomic_radicals(self):
        """
        Test that perceiving atomic radicals works correctly.
        """
        # H
        mol = perceive_molecule_from_xyz(self.xyz_h)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 1)
        h = mol.atoms[0]
        self.assertEqual(h.element.symbol, 'H')
        self.assertEqual(h.radical_electrons, 1)
        self.assertEqual(len(mol.get_all_edges()), 0)
        self.assertEqual(mol.multiplicity, 2)
        # C
        mol = perceive_molecule_from_xyz(ARCSpecies(label='C', smiles='[C]').get_xyz(), multiplicity=3)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 1)
        c = mol.atoms[0]
        self.assertEqual(c.element.symbol, 'C')
        self.assertEqual(c.radical_electrons, 2)
        self.assertEqual(len(mol.get_all_edges()), 0)
        self.assertEqual(mol.multiplicity, 3)
        # O
        mol = perceive_molecule_from_xyz(ARCSpecies(label='O', smiles='[O]').get_xyz(), multiplicity=3)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 1)
        o = mol.atoms[0]
        self.assertEqual(o.element.symbol, 'O')
        self.assertEqual(o.radical_electrons, 2)
        self.assertEqual(len(mol.get_all_edges()), 0)
        self.assertEqual(mol.multiplicity, 3)
        # N
        mol = perceive_molecule_from_xyz(ARCSpecies(label='N', smiles='[N]').get_xyz(), multiplicity=4)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 1)
        n = mol.atoms[0]
        self.assertEqual(n.element.symbol, 'N')
        self.assertEqual(n.radical_electrons, 3)
        self.assertEqual(len(mol.get_all_edges()), 0)
        self.assertEqual(mol.multiplicity, 4)


    def test_perceive_invalid(self):
        """
        Test that perceiving an invalid XYZ does not raise an error, but returns None
        """
        mol = perceive_molecule_from_xyz(self.xyz_invalid)
        self.assertIsNone(mol)

    def test_get_representative_resonance_structure_passthrough(self):
        """
        Test that the representative resonance structure is the same as the
        """
        # CH4 has no resonance, the representative form is itself
        mol = perceive_molecule_from_xyz(self.xyz_methane)
        rep = get_representative_resonance_structure(mol)
        # isomorphism should hold
        self.assertTrue(mol.is_isomorphic(rep))

    def test_atom_order_preserved(self):
        """
        Test that the atom order is preserved when perceiving a molecule
        """
        # permute the methane input, the perceived graph preserves
        # the original ordering of atoms in the returned Molecule.atoms list
        permuted = dict(self.xyz_methane)
        permuted['symbols'] = tuple(reversed(permuted['symbols']))
        permuted['isotopes'] = tuple(reversed(permuted['isotopes']))
        permuted['coords'] = tuple(reversed(permuted['coords']))
        mol = perceive_molecule_from_xyz(permuted)
        # first atom in mol.atoms should match the first entry of the permuted input
        first = mol.atoms[0]
        self.assertEqual(first.element.symbol, permuted['symbols'][0])
        self.assertTrue(np.allclose(first.coords, permuted['coords'][0]))

        xyz_1 = """C       0.02571153    1.50024692   -0.01880972
                   H      -0.25012379    2.28327632    0.67957788
                   H       0.21710650    1.77015012   -1.05186079
                   C      -0.12961272    0.05931627    0.38298020
                   C      -1.52159692   -0.43413728   -0.00244580
                   C       0.95427547   -0.82618224   -0.25128786
                   O       2.23864587   -0.52290772    0.28688439
                   H      -0.02271951    0.01229964    1.47391586
                   H      -1.67349890   -1.46562132    0.33336150
                   H      -1.67080846   -0.40804497   -1.08793835
                   H      -2.30052614    0.18308086    0.45923715
                   H       0.75830763   -1.88272043   -0.04089782
                   H       0.99720067   -0.70255870   -1.33919508
                   H       2.37763877    0.43380254    0.17647842"""
        spc_1a = ARCSpecies(label='S', smiles='[CH2]C(C)CO', xyz=xyz_1)
        for atom, symbol in zip(spc_1a.mol.atoms, str_to_xyz(xyz_1)['symbols']):
            self.assertEqual(atom.element.symbol, symbol)
        spc_1b = ARCSpecies(label='S', smiles='CC([CH2])CO', xyz=xyz_1)
        for atom, symbol in zip(spc_1b.mol.atoms, str_to_xyz(xyz_1)['symbols']):
            self.assertEqual(atom.element.symbol, symbol)
        spc_1c = ARCSpecies(label='S', smiles='OCC([CH2])C', xyz=xyz_1)
        for atom, symbol in zip(spc_1c.mol.atoms, str_to_xyz(xyz_1)['symbols']):
            self.assertEqual(atom.element.symbol, symbol)

        spc_2 = ARCSpecies(label='chiral1', smiles='[O]C(N)Cl', xyz=self.xyz_change_chirality)
        for atom, symbol in zip(spc_2.mol.atoms, self.xyz_change_chirality['symbols']):
            self.assertEqual(atom.element.symbol, symbol)

        xyz_3 = """N                 -3.40822517   -1.32794120   -0.00037479
                   H                 -3.07822669   -0.86764860    0.82378037
                   C                 -2.93415574   -2.71938134   -0.00762172
                   H                 -1.86428654   -2.73173507    0.00365988
                   O                 -3.43091353   -3.39823062    1.14879315
                   S                 -2.89234486   -4.97898234    1.14056024
                   H                 -3.34741538   -5.60086481    2.19993412
                   F                 -2.96357484   -0.72029988   -1.03548740
                   Cl                -3.51495306   -3.52949631   -1.45813480"""
        spc_3 = ARCSpecies(label='S', smiles='C(NF)(Cl)OS', xyz=xyz_3)
        for atom, symbol in zip(spc_3.mol.atoms, str_to_xyz(xyz_3)['symbols']):
            self.assertEqual(atom.element.symbol, symbol)
        mol_3 = perceive_molecule_from_xyz(xyz_3)
        for atom, symbol in zip(mol_3.atoms, str_to_xyz(xyz_3)['symbols']):
            self.assertEqual(atom.element.symbol, symbol)

        xyz_4 = """N                  4.80913900   -0.18142200   -0.31894800
                   C                  3.70376300   -1.17433400   -0.29502100
                   C                  2.45778600   -0.74850100    0.50058600
                   H                  4.10978900   -2.11991000    0.13095600
                   C                  1.62116900    0.29476100   -0.25698800
                   H                  1.84074900   -1.64060500    0.72043500
                   H                  2.74746400   -0.34556500    1.49184900
                   N                  0.56572800    0.83928800    0.66053500
                   H                  2.27298300    1.13086100   -0.58875700
                   C                 -0.58051700   -0.06632200    0.80490800
                   C                  0.13746100    2.21077400    0.25626800
                   H                 -1.35936800    0.37327500    1.45247800
                   H                 -0.25811600   -1.06042800    1.17657000
                   H                 -0.49831100    2.63509900    1.05018600
                   H                 -0.43367100    2.23564600   -0.68857500
                   H                  1.01817400    2.86272700    0.15917800
                   H                  5.11146200    0.04022000    0.62542100
                   H                  4.49938400    0.69076900   -0.73712200
                   H                  3.43073500   -1.40966400   -1.34806700
                   H                  1.18593500   -0.15003100   -1.17952600
                   O                 -1.18790200   -0.45549200   -0.48823800
                   O                 -2.01122000    0.48874300   -0.93507100
                   H                 -3.16340600    0.19846300   -0.51358000
                   C                 -4.45612600   -0.13163000   -0.38252500
                   H                 -4.89014100    0.77163700    0.03528500
                   H                 -4.72782500   -0.41388400   -1.39028500
                   O                 -4.49534800   -1.12219800    0.58015100
                   H                 -4.25535300   -2.00057000    0.19434000"""
        spc_4 = ARCSpecies(label='imipramine_3_peroxy', is_ts=True, xyz=xyz_4)
        for atom, symbol in zip(spc_4.mol.atoms, str_to_xyz(xyz_4)['symbols']):
            self.assertEqual(atom.element.symbol, symbol)

        xyz_5 = {'symbols': ('O', 'C', 'N', 'C', 'H', 'H', 'H', 'H'),
                   'isotopes': (16, 12, 14, 12, 1, 1, 1, 1),
                   'coords': ((-1.46891188, 0.4782021, -0.74907357), (-0.77981513, -0.5067346, 0.0024359),
                              (0.86369081, 0.1484285, 0.8912832), (1.78225246, 0.27014716, 0.17691),
                              (2.61878546, 0.38607062, -0.47459418), (-1.62732717, 1.19177937, -0.10791543),
                              (-1.40237804, -0.74595759, 0.87143836), (-0.39285462, -1.26299471, -0.69270021))}
        spc_5 = ARCSpecies(label='TS', is_ts=True, xyz=xyz_5)
        for atom, symbol in zip(spc_5.mol.atoms, xyz_5['symbols']):
            self.assertEqual(atom.element.symbol, symbol)


    def test_crazy_oxy_s_with_n(self):
        """
        Test that perceiving a crazy heteroatom molecule works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_crazy_oxy_s_with_n)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 10)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'CS(=O)(=N)O')

    def test_anions(self):
        """
        Test that perceiving anions works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_oh_anion, charge=-1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 2)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), -1)
        self.assertEqual(mol.to_smiles(), '[OH-]')

        mol = perceive_molecule_from_xyz(self.xyz_co3_anion, charge=-2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 4)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), -2)
        self.assertEqual(mol.to_smiles(), 'O=C([O-])[O-]')

    def test_cations(self):
        """
        Test that perceiving cations works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_nh4_cation, charge=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 5)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 1)
        self.assertEqual(mol.to_smiles(), '[NH4+]')

        mol = perceive_molecule_from_xyz(self.xyz_h3o_cation, charge=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 4)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 1)
        self.assertEqual(mol.to_smiles(), '[OH3+]')

    def test_ozone(self):
        """
        Test that perceiving ozone works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_ozone, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 3)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O-][O+]=O')

    def test_hno3(self):
        """
        Test that perceiving HNO3 works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_hno3, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 5)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O-][N+](=O)O')

    def test_n3o2(self):
        """
        Test that perceiving N3O2H3 works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_n3o2, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 8)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O-][N+](=O)NN')

    def test_no2(self):
        """
        Test that perceiving NO2 works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_no2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 3)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['[O-][N+]=O', '[O]N=O'])

    def test_ch2no(self):
        """
        Test that perceiving CH2NO works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_ch2no)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 5)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O]N=C')

    def test_h2so4(self):
        """
        Test that perceiving H2SO4 works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_h2so4, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 7)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'OS(=O)(=O)O')

    def test_so3(self):
        """
        Test that perceiving SO3 works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_so3, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 4)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'O=S(=O)=O')

    def test_c6_dd_yl(self):
        """
        Test that perceiving C6 with double bond and ylide works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_c6_dd_yl, charge=0, multiplicity=2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 15)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['[CH2]C=CC=CC', 'C=C[CH]C=CC', 'C=CC=C[CH]C'])

    def test_c3_linear_cn(self):
        """
        Test that perceiving linear C3N works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_c3_linear_cn)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 11)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[N]=C=C(C)C')

    def test_hsn(self):
        """
        Test that perceiving HSN works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_hsn, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 3)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'N#S')

    def test_h2nn_singlet(self):
        """
        Test that perceiving H2NN singlet works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_h2nn_singlet, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 4)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[N-]=[NH2+]')

    def test_hcn(self):
        """
        Test that perceiving HCN works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_hcn, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 3)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'C#N')

    def test_sn2(self):
        """
        Test that perceiving SN2 works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_sn2, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 3)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'N#S#N')

    def test_n2h2_triplet(self):
        """
        Test that perceiving N2H2 triplet works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_n2h2_triplet, charge=0, multiplicity=3)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 4)
        self.assertEqual(mol.multiplicity, 3)
        self.assertEqual(mol.to_smiles(), '[NH][NH]')
        self.assertEqual(mol.get_net_charge(), 0)

    def test_aibn(self):
        """
        Test that perceiving AIBN works.
        """
        mol = perceive_molecule_from_xyz(self.aibn, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 24)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.to_smiles(), 'N#CC(N=NC(C#N)(C)C)(C)C')
        self.assertEqual(mol.get_net_charge(), 0)

    def test_ch3n2oh(self):
        """
        Test that perceiving CH3N2OH works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_ch3n2oh, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 8)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['[O-][N+](=N)C', '[NH-][N+](=O)C'])

    def test_conjugated_oxy_benzene(self):
        """
        Test that perceiving a conjugated oxybenzene works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_conjugated_oxy_benzene, multiplicity=2, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 28)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        xyz_dict = validate_xyz(self.xyz_conjugated_oxy_benzene)
        for mol_atom, xyz_symbol in zip(mol.atoms, xyz_dict['symbols']):
            self.assertEqual(mol_atom.element.symbol, xyz_symbol)
        self.assertIn(mol.to_smiles(), ['COC1=C(CO)[CH]C(=C(C)C)C=C1',
                                        'COc1ccc([C](C)C)cc1CO',
                                        'COC1=C(CO)C=C([C](C)C)C=C1',
                                        'CO[C]1C=CC(=C(C)C)C=C1CO',
                                        'COC1=C[CH]C(=C(C)C)C=C1CO',
                                        'COC1=CC=C([C](C)C)C=C1CO',
                                        ])

    def test_cyc_c3n2(self):
        """
        Test that perceiving cyclic C3N2 works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_cyc_c3n2, charge=0, multiplicity=2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 8)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'C1=C[N]C=N1')

    def test_carbenes_and_nitrenes(self):
        """
        Test that perceiving carbenes and nitrenes works.
        """
        # triplet carbene
        mol = perceive_molecule_from_xyz(ARCSpecies(label='carbene', smiles='CC[C]O').get_xyz(), multiplicity=3, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 3)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['CC[C]O', 'CC[C][OH]'])
        # singlet carbene
        mol = perceive_molecule_from_xyz(ARCSpecies(label='carbene', smiles='CC[C]O').get_xyz(), multiplicity=1, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['CC[C-]=[OH+]'])
        # triplet nitrene
        mol = perceive_molecule_from_xyz(ARCSpecies(label='nitrene', smiles='CC[N]').get_xyz(), multiplicity=3, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 3)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['CC[N]'])
        # singlet nitrene
        mol = perceive_molecule_from_xyz(ARCSpecies(label='nitrene', smiles='CC[N]').get_xyz(), multiplicity=1, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['CC[N]'])
        # H2NN triplet
        xyz_h2nn_t = """N       1.25464159   -0.04494405   -0.06271952
                        N      -0.11832785   -0.00810069    0.29783210
                        H      -0.59897890   -0.78596704   -0.15190060
                        H      -0.53733484    0.83901179   -0.08321197"""
        mol = perceive_molecule_from_xyz(xyz_h2nn_t, charge=0, multiplicity=3, n_fragments=1)
        self.assertIn(mol.to_smiles(), ['[N]N', '[N-][NH2+]'])

    def test_aromatics(self):
        """
        Test that perceiving aromatic compounds works.
        """
        # benzene
        mol = perceive_molecule_from_xyz(ARCSpecies(label='benzene', smiles='c1ccccc1').get_xyz(), multiplicity=1, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'c1ccccc1')

        # phenyl radical
        mol = perceive_molecule_from_xyz(ARCSpecies(label='phenyl', smiles='c1cccc[c]1').get_xyz(), multiplicity=2, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['[c]1ccccc1'])

        # naphthalene
        mol = perceive_molecule_from_xyz(ARCSpecies(label='naphthalene', smiles='c1ccc2ccccc2c1').get_xyz(), multiplicity=1, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'c1ccc2ccccc2c1')

    def test_resonating_radicals(self):
        """
        Test that perceiving resonating radicals works.
        """
        # allyl radical
        mol = perceive_molecule_from_xyz(ARCSpecies(label='allyl', smiles='C=C[CH]C').get_xyz(), multiplicity=2, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['C=C[CH]C', '[CH2]C=CC'])

        mol = perceive_molecule_from_xyz(ARCSpecies(label='allyl', smiles='C=C[C]C').get_xyz(), multiplicity=3, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 3)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['C=C[C]C', '[CH2]C=[C]C'])

        mol = perceive_molecule_from_xyz(ARCSpecies(label='allyl', smiles='C=C[O]').get_xyz(), multiplicity=2, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['C=C[O]', '[CH2]C=O'])

    def test_birad_singlet(self):
        """
        Test that perceiving biradical singlet works.
        """
        mol = perceive_molecule_from_xyz(ARCSpecies(label='birad_singlet', smiles='[N]=C=[N]').get_xyz(),
                                         multiplicity=1, charge=0, n_radicals=2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['[N]=C=[N]', '[N]C#N'])

        c6h6_b_xyz = {'coords': ((-1.474267041853848, 0.27665693719971857, -0.31815898666696507),
                                 (-0.25527025747758825, 1.1936776717612125, -0.2432148642540069),
                                 (0.9917471212521393, 0.7578589393970138, 0.059037260524552534),
                                 (1.2911962562420976, -0.6524103892231805, 0.34598643264742923),
                                 (0.321535921890914, -1.5867102018006056, 0.32000545365633654),
                                 (-0.9417407846918554, -1.043897260224426, -0.002820356559266387),
                                 (-2.2262364004658077, 0.5956762298613206, 0.40890113659975075),
                                 (-1.90597332290244, 0.31143075666839354, -1.3222845692785703),
                                 (-0.4221153027089989, 2.2469871640348815, -0.4470234892644997),
                                 (1.824518548011024, 1.4543788790156666, 0.0987362566117616),
                                 (2.3174577767359237, -0.9162726684959432, 0.5791638390925197),
                                 (0.4791474859684761, -2.637376058194065, 0.5216718868909702)),
                      'isotopes': (12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1),
                      'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H')}
        mol = perceive_molecule_from_xyz(c6h6_b_xyz, charge=0, multiplicity=1, n_radicals=2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 12)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['C1C=CC=C[C]1', '[C]1C=CC=CC1'])

    def test_o2_s2_os(self):
        """
        Test that perceiving O2, S2, and SO works.
        """
        # O2 triplet
        mol = perceive_molecule_from_xyz(ARCSpecies(label='O2', smiles='[O][O]').get_xyz(), multiplicity=3, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 3)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O][O]')
        # O2 singlet
        mol = perceive_molecule_from_xyz(ARCSpecies(label='O2', smiles='O=O').get_xyz(), multiplicity=1, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'O=O')
        # S2 triplet
        mol = perceive_molecule_from_xyz(ARCSpecies(label='S2', smiles='[S][S]').get_xyz(), multiplicity=3, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 3)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[S][S]')
        # S2 singlet
        mol = perceive_molecule_from_xyz(ARCSpecies(label='S2', smiles='S=S').get_xyz(), multiplicity=1, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'S=S')
        # SO triplet
        mol = perceive_molecule_from_xyz(ARCSpecies(label='SO', smiles='[S][O]').get_xyz(), multiplicity=3, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 3)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['[S][O]', '[O][S]'])
        # SO singlet
        mol = perceive_molecule_from_xyz(ARCSpecies(label='SO', smiles='S=O').get_xyz(), multiplicity=1, charge=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['S=O', 'O=S'])

    def test_ccooj(self):
        """
        Test that perceiving CCOOJ works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_ccooj, charge=0, multiplicity=2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 9)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['CCO[O]'])

    def test_more_radicals(self):
        """
        Test that perceiving more radicals works.
        """
        # C6H5-1
        mol = perceive_molecule_from_xyz(ARCSpecies(label='C6H5', smiles='[CH2]CCCCC').get_xyz())
        self.assertEqual(mol.to_smiles(), '[CH2]CCCCC')
        # C6H5-2
        mol = perceive_molecule_from_xyz(ARCSpecies(label='C6H5', smiles='C[CH]CCCC').get_xyz())
        self.assertIn(mol.to_smiles(), ['C[CH]CCCC', 'CCCC[CH]C'])
        # C6H5-3
        mol = perceive_molecule_from_xyz(ARCSpecies(label='C6H5', smiles='CC[CH]CCC').get_xyz())
        self.assertIn(mol.to_smiles(), ['CC[CH]CCC', 'CCC[CH]CC'])
        # CC(=O)O[O]
        mol = perceive_molecule_from_xyz(ARCSpecies(label='CC(=O)O[O]', smiles='CC(=O)O[O]').get_xyz())
        self.assertEqual(mol.to_smiles(), 'CC(=O)O[O]')
        # NH(T)
        mol = perceive_molecule_from_xyz(ARCSpecies(label='NH', smiles='[NH]').get_xyz(), multiplicity=3)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 3)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[NH]')
        # NH(S)
        mol = perceive_molecule_from_xyz(ARCSpecies(label='NH', smiles='[NH]').get_xyz(), multiplicity=1, n_radicals=0)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[NH]')
        # NH2
        mol = perceive_molecule_from_xyz(ARCSpecies(label='NH2', smiles='[NH2]').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[NH2]')
        # CCO[NH]
        mol = perceive_molecule_from_xyz(ARCSpecies(label='CCO[NH]', smiles='CCO[NH]').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'CCO[NH]')
        # [O]OC(=O)O
        mol = perceive_molecule_from_xyz(ARCSpecies(label='[O]OC(=O)O', smiles='[O]OC(=O)O').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O]OC(=O)O')
        # [O]C1(O)OO1
        mol = perceive_molecule_from_xyz(ARCSpecies(label='[O]C1(O)OO1', smiles='[O]C1(O)OO1').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O]C1(O)OO1')
        # [O]C(=O)OO
        mol = perceive_molecule_from_xyz(ARCSpecies(label='[O]C(=O)OO', smiles='[O]C(=O)OO').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O]C(=O)OO')
        # O=[C]O
        mol = perceive_molecule_from_xyz(ARCSpecies(label='O=[C]O', smiles='O=[C]O').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'O=[C]O')
        # [O]C(=O)O
        mol = perceive_molecule_from_xyz(ARCSpecies(label='[O]C(=O)O', smiles='[O]C(=O)O').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O]C(=O)O')
        # HO2
        mol = perceive_molecule_from_xyz(ARCSpecies(label='[O]O', smiles='[O]O').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O]O')
        # O[C]1OO1
        mol = perceive_molecule_from_xyz(ARCSpecies(label='O[C]1OO1', smiles='O[C]1OO1').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'O[C]1OO1')
        # [O][C]1OO1
        mol = perceive_molecule_from_xyz(ARCSpecies(label='[O][C]1OO1', smiles='[O][C]1OO1').get_xyz(), multiplicity=3, n_radicals=2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 3)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O][C]1OO1')
        # [O]C([O])=O
        mol = perceive_molecule_from_xyz(ARCSpecies(label='[O]C([O])=O', smiles='[O]C([O])=O').get_xyz(), multiplicity=3)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 3)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O]C([O])=O')
        # O[C]1OOO1
        mol = perceive_molecule_from_xyz(ARCSpecies(label='O[C]1OOO1', smiles='O[C]1OOO1').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'O[C]1OOO1')
        # [O]O[C]=O

    def test_more_radicals_2(self):
        mol = perceive_molecule_from_xyz(ARCSpecies(label='[O]O[C]=O', smiles='[O]O[C]=O').get_xyz(), n_radicals=2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O]O[C]=O')
        # [O]OC([O])=O
        mol = perceive_molecule_from_xyz(ARCSpecies(label='[O]OC([O])=O', smiles='[O]OC([O])=O').get_xyz(), multiplicity=3)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 3)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O]OC([O])=O')
        # [O]C1([O])OO1
        mol = perceive_molecule_from_xyz(ARCSpecies(label='[O]C1([O])OO1', smiles='[O]C1([O])OO1').get_xyz(), n_radicals=2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O]C1([O])OO1')
        # [O][C]=O
        mol = perceive_molecule_from_xyz(ARCSpecies(label='[O][C]=O', smiles='[O][C]=O').get_xyz(), multiplicity=3)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 3)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O][C]=O')

    def test_more_species(self):
        """
        Test perceiving more species.
        """
        # O=C1OO1
        mol = perceive_molecule_from_xyz(ARCSpecies(label='O=C1OO1', smiles='O=C1OO1').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'O=C1OO1')
        # CO2
        mol = perceive_molecule_from_xyz(ARCSpecies(label='CO2', smiles='O=C=O').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'O=C=O')
        # N2
        mol = perceive_molecule_from_xyz(ARCSpecies(label='N2', smiles='N#N').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'N#N')
        # CO
        mol = perceive_molecule_from_xyz(ARCSpecies(label='CO', smiles='[C-]#[O+]').get_xyz())
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[C-]#[O+]')

    def test_perceiving_fragments(self):
        """
        Test that perceiving fragments works.
        """
        xyz_frag_1 = """ C                 -5.05813958    0.07751938    0.00000000
                         H                 -4.70148515   -0.93129062    0.00000000
                         H                 -4.70146674    0.58191757    0.87365150
                         H                 -4.70146674    0.58191757   -0.87365150
                         H                 -6.12813958    0.07753256    0.00000000
                         C                  4.82558143   -0.11627907    0.00000000
                         H                  5.18223586   -1.12508907    0.00000000
                         H                  5.18225428    0.38811912    0.87365150
                         H                  5.18225428    0.38811912   -0.87365150
                         H                  3.75558144   -0.11626588    0.00000000"""
        mol = perceive_molecule_from_xyz(xyz_frag_1, n_fragments=2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'C.C')

        xyz_frag_2 = """ C                 -3.60491073    2.25446425    0.00000000
                         H                 -3.24823789    2.75886244    0.87365150
                         H                 -3.24823789    2.75886244   -0.87365150
                         H                 -4.67491073    2.25447744    0.00000000
                         C                 -3.09159501    0.80253210    0.00000000
                         H                 -3.44826974    0.29813271   -0.87365004
                         C                 -3.60493461    0.07657675    1.25740657
                         H                 -3.24988030   -0.93279705    1.25642781
                         H                 -3.24666166    0.57984439    2.13105534
                         H                 -4.67493280    0.07828559    1.25838880
                         C                 -1.55159501    0.80251390   -0.00000377
                         H                 -1.19494059   -0.20629608    0.00021202
                         H                 -1.19492484    1.30672515   -0.87376426
                         H                 -1.19492006    1.30710000    0.87353836
                         C                  2.43852385    0.51857434   -0.08033201
                         H                  2.79517828   -0.49023566   -0.08033201
                         H                  2.79519669    1.02297253    0.79331950
                         O                  2.91519877    1.19267659   -1.24792233
                         H                  2.59399397    2.09734547   -1.24922502
                         O                  1.00852385    0.51859196   -0.08033201
                         H                  0.68804415    0.02460982    0.67787272
                         O                  1.30890714   -2.91480640    2.21044049
                         H                  2.26890714   -2.91480640    2.21044049
                         H                  0.98845256   -2.00987057    2.21044049"""
        mol = perceive_molecule_from_xyz(xyz_frag_2, n_fragments=3)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'CC(C)C.O.OCO')

        xyz_ts_1 = """O      -0.63023600    0.92494700    0.43958200
                      C       0.14513500   -0.07880000   -0.04196400
                      C      -0.97050300   -1.02992900   -1.65916600
                      N      -0.75664700   -2.16458700   -1.81286400
                      H      -1.25079800    0.57954500    1.08412300
                      H       0.98208300    0.28882200   -0.62114100
                      H       0.30969500   -0.94370100    0.59100600
                      H      -1.47626400   -0.10694600   -1.88883800"""
        mol = perceive_molecule_from_xyz(xyz_ts_1, n_fragments=2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'C#N.[CH2]O')

        xyz_ts_2 = """C       0.52123900   -0.93806900   -0.55301700
                      C       0.15387500    0.18173100    0.37122900
                      C      -0.89554000    1.16840700   -0.01362800
                      H       0.33997700    0.06424800    1.44287100
                      H       1.49602200   -1.37860200   -0.29763200
                      H       0.57221700   -0.59290500   -1.59850500
                      H       0.39006800    1.39857900   -0.01389600
                      H      -0.23302200   -1.74751100   -0.52205400
                      H      -1.43670700    1.71248300    0.76258900
                      H      -1.32791000    1.11410600   -1.01554900"""
        mol = perceive_molecule_from_xyz(xyz_ts_2, n_fragments=2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[CH2]CC')

    def test_chiral_species(self):
        xyz_chiral_1 = """C                 -0.81825240   -0.04911020   -0.14065159
                          H                 -1.34163466   -0.39900096   -1.00583797
                          C                  0.51892324    0.16369053   -0.19760928
                          H                  1.05130979   -0.01818286   -1.10776676
                          O                 -1.52975971    0.19395459    1.07572699
                          H                 -2.43039815    0.45695722    0.87255216
                          C                  1.27220245    0.66727126    1.04761235
                          H                  1.28275235    1.73721734    1.04963240
                          N                  0.59593728    0.18162740    2.25910542
                          H                  0.58607755   -0.81832221    2.25721751
                          H                  1.08507962    0.50862787    3.06769089
                          S                  2.94420440    0.05748035    1.01655601
                          H                  3.58498087    0.48585096    2.07580298"""
        mol = perceive_molecule_from_xyz(xyz_chiral_1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['OC=CC(N)(S)', 'NC(C=CO)S'])

    def test_a_collection_of_molecules_from_arc_tests_that_gave_warnings(self):
        """
        Test that perceiving a collection of molecules from ARC tests that gave warnings works.
        """
        xyz_ts_5_nmd = parse_geometry(os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'TS0_composite_2044.out'))
        mol = perceive_molecule_from_xyz(xyz_ts_5_nmd, charge=0, multiplicity=2, n_fragments=2)
        self.assertEqual(mol.to_smiles(), 'CCN')
        self.assertEqual(mol.to_adjacency_list(), """multiplicity 2
1  N u0 p1 c0 {2,S} {4,S} {5,S}
2  C u0 p0 c0 {1,S} {3,S} {6,S} {7,S}
3  C u0 p0 c0 {2,S} {8,S} {9,S} {10,S}
4  H u0 p0 c0 {1,S}
5  H u0 p0 c0 {1,S}
6  H u0 p0 c0 {2,S}
7  H u0 p0 c0 {2,S}
8  H u0 p0 c0 {3,S}
9  H u0 p0 c0 {3,S}
10 H u0 p0 c0 {3,S}
11 H u1 p0 c0
""")

        xyz_h2o_h2O = """O                 -2.25790521    1.74901183    0.00000000
                         H                 -1.29790521    1.74901183    0.00000000
                         H                 -2.57835980    2.65394766    0.00000000
                         O                  2.23814237   -1.56126480    0.00000000
                         H                  3.19814237   -1.56126480    0.00000000
                         H                  1.91768778   -0.65632897    0.00000000"""
        mol = perceive_molecule_from_xyz(xyz_h2o_h2O, charge=0, multiplicity=1, n_fragments=2)
        self.assertEqual(mol.to_smiles(), 'O.O')

        xyz_ts_n3h5_a = {'coords': ((0.9177905887, 0.5194617797, 0.0),
                                  (1.8140204898, 1.0381941417, 0.0),
                                  (-0.4763167868, 0.7509348722, 0.0),
                                  (0.999235086, -0.7048575683, 0.0),
                                  (-1.4430010939, 0.0274543367, 0.0),
                                  (-0.6371484821, -0.7497769134, 0.0),
                                  (-2.0093636431, 0.0331190314, -0.8327683174),
                                  (-2.0093636431, 0.0331190314, 0.8327683174)),
                       'isotopes': (14, 1, 1, 14, 14, 1, 1, 1),
                       'symbols': ('N', 'H', 'H', 'N', 'N', 'H', 'H', 'H')}
        mol = perceive_molecule_from_xyz(xyz_ts_n3h5_a, charge=0, multiplicity=1, n_fragments=2)
        self.assertEqual(mol.to_smiles(), '[N-]=N.[NH4+]')

        xyz_ts_n3h5_b = {'symbols': ('N', 'H', 'H', 'N', 'H', 'H', 'N', 'H'), 'isotopes': (14, 1, 1, 14, 1, 1, 14, 1),
                         'coords': ((-0.424886, 0.694403, -0.092695), (-0.473966, 1.196384, 0.779312),
                                    (0.050017, 0.609187, -0.289804), (-1.230175, -0.49518, -0.005872),
                                    (-1.816882, -0.464956, 0.807105), (-1.846181, -0.503768, -0.816157),
                                    (1.943069, -0.169246, -0.067924), (1.737758, -0.876687, 0.671382))}
        mol = perceive_molecule_from_xyz(xyz_ts_n3h5_b, charge=0, multiplicity=1, n_fragments=2)
        self.assertEqual(mol.to_smiles(), 'NN.[NH]')

        xyz_chiral_chlorine = {'symbols': ('C', 'C', 'Cl', 'C', 'C', 'C', 'C', 'O', 'H', 'H',
                                           'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                               'isotopes': (12, 12, 35, 12, 12, 12, 12, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                               'coords': ((-2.578509712530318, 0.38121240547111007, -0.31545198147258363),
                                          (-1.5941349895084616, -0.635494075258274, 0.2526267766110877),
                                          (-2.446130204434876, -2.199793383445274, 0.4739879765478844),
                                          (-0.3649125812333054, -0.856711907825194, -0.6396701659159703),
                                          (0.4922297868446982, 0.37185241828403975, -0.7809642373311291),
                                          (1.6231568004320585, 0.6568089242362904, -0.12145237874417657),
                                          (2.267486108970528, -0.1753954473511403, 0.9466738166587931),
                                          (2.265480730568699, 1.8439777575730865, -0.4285684239770236),
                                          (-3.4594352741921353, 0.47581928028773773, 0.32913922777496435),
                                          (-2.123704792289639, 1.3742045671010559, -0.3889170186287234),
                                          (-2.920625447943592, 0.09209027422932005, -1.3152490550970735),
                                          (-1.2790241139583118, -0.31597683146205213, 1.2521170706149654),
                                          (-0.6802362877491801, -1.1661589508802492, -1.6449366809148853),
                                          (0.231721951825261, -1.6911255777813388, -0.2530777406072024),
                                          (0.15205839141215702, 1.0933904704061583, -1.5237538914581972),
                                          (2.4559163544367437, 0.43653975141236306, 1.8352257595756216),
                                          (1.6504484708523275, -1.0197745053207323, 1.2634073789063887),
                                          (3.225226315786351, -0.5670700918169522, 0.58983306593164),
                                          (3.082988492711173, 1.9016049221399955, 0.0959260108384249))}
        mol = perceive_molecule_from_xyz(xyz_chiral_chlorine, charge=0, multiplicity=1, n_fragments=1)
        self.assertEqual(mol.to_smiles(), 'CC(O)=CCC(C)Cl')

        xyz_crazy = {'symbols': ('C', 'S', 'C', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'C', 'N', 'O', 'C', 'C', 'C', 'O',
                                 'C', 'H', 'H', 'C', 'O', 'O', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                     'isotopes': (12, 32, 12, 12, 12, 16, 12, 12, 12, 12, 12, 14, 16, 12, 12, 12, 16, 12, 1, 1, 12, 16,
                                  16, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                     'coords': ((0.3274026, -2.26751349, 1.63049586), (-0.45583669, -0.69403507, 2.07966011),
                                (-0.74998553, 0.1674721, 0.40294571), (-1.00035074, -0.91486831, -0.69963969),
                                (-1.36275248, -0.4211373, -2.10180028), (-1.2751424, -1.51271695, -3.01524455),
                                (0.52137932, 1.03702992, 0.05827308), (1.87071211, 0.36242919, -0.0942956),
                                (2.27157054, -0.19914345, -1.31148631), (3.52325995, -0.81277227, -1.43688167),
                                (4.40522429, -0.88047028, -0.3550893), (5.67765022, -1.44973252, -0.47213753),
                                (6.14232632, -2.01780812, -1.64786894), (4.01787295, -0.28926955, 0.85080026),
                                (2.76727967, 0.32378873, 0.98226698), (-2.03762393, 1.15227615, 0.57400664),
                                (-2.08528422, 1.91614517, -0.66817021), (-3.07850461, 2.79943864, -0.89330921),
                                (-3.01406475, 3.29649809, -1.84791173), (-3.82602922, 2.99925098, -0.14930564),
                                (-3.34048249, 0.30576028, 0.76061402), (-4.43606057, 1.09150052, 1.25094611),
                                (-5.5667201, 0.17784882, 1.34169192), (-1.87366601, 2.15813122, 1.74584475),
                                (-0.38980918, -2.92765906, 1.13662409), (0.64993945, -2.7631856, 2.55083355),
                                (1.20742139, -2.132265, 1.00223888), (-1.7923623, -1.60504973, -0.3796867),
                                (-0.1173514, -1.55329785, -0.81227132), (-2.38336335, -0.03585664, -2.15575528),
                                (-0.67883648, 0.35169141, -2.45761307), (-2.10902199, -1.54645788, -3.51490053),
                                (0.34538583, 1.60264434, -0.86560486), (0.65096478, 1.81353001, 0.82157824),
                                (1.62059278, -0.16895436, -2.18187181), (3.79727304, -1.2511092, -2.39297007),
                                (6.19600818, -1.66693258, 0.37451698), (6.3095366, -1.24260293, -2.21706522),
                                (4.68937879, -0.29721399, 1.70527737), (2.49834754, 0.7695989, 1.93797335),
                                (-3.65415206, -0.1279457, -0.19458068), (-3.18902405, -0.502595, 1.48133547),
                                (-5.67928065, 0.23893607, 2.30792673), (-1.02113438, 2.82848721, 1.61108522),
                                (-2.77629877, 2.75649408, 1.89754068), (-1.71528969, 1.64716863, 2.70183805))}
        mol = perceive_molecule_from_xyz(xyz_crazy, charge=0, multiplicity=1, n_fragments=1)
        self.assertIn(mol.to_smiles(), ['OCCC(C(COO)(O[CH2])C)(Cc1ccc(cc1)NO)SC', 'OCCC(C(COO)(O[CH2])C)(CC1=C[CH]C(=C[CH]1)NO)SC'])

        xyz_nhfcl = """N      -0.14626256    0.12816405    0.30745256
                       F      -0.94719775   -0.91910939   -0.09669786
                       Cl      1.53982436   -0.20497454   -0.07627978
                       H      -0.44636405    0.99591988   -0.13447493"""
        mol = perceive_molecule_from_xyz(xyz_nhfcl, charge=0, multiplicity=1, n_fragments=1)
        self.assertIn(mol.to_smiles(), ['FNCl'])


        xyz_nfcl = """N      -0.17697493    0.58788903    0.00000000
                      F      -1.17300047   -0.36581404    0.00000000
                      Cl      1.34997541   -0.22207499    0.00000000"""
        mol = perceive_molecule_from_xyz(xyz_nfcl, charge=0, multiplicity=2, n_fragments=1)
        self.assertIn(mol.to_smiles(), ['F[N]Cl'])

        mol = perceive_molecule_from_xyz(self.xyz_change_chirality, charge=0, multiplicity=2, n_fragments=1)
        self.assertIn(mol.to_smiles(), ['[O]C(N)Cl', 'NC(Cl)[O]'])


if __name__ == '__main__':
    unittest.main()
