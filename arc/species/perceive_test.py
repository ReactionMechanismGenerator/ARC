#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.species.perceive module
"""


import unittest
import numpy as np

from arc.species.perceive import (
    validate_xyz,
    infer_multiplicity,
    perceive_molecule_from_xyz,
    get_representative_resonance,
)
from arc.molecule.molecule import Molecule


class TestPerceive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A perfectly tetrahedral methane
        cls.xyz_methane = {
            'symbols': ('C', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 1, 1, 1, 1),
            'coords': (
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                (-1.0, -1.0, 1.0),
                (-1.0, 1.0, -1.0),
                (1.0, -1.0, -1.0),
            )}
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
            'symbols': ('C', 'wrong_symbol'),
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
        # hypervalance, carbenes, nitrenes, aromatics, resonating radicals like cregee and allele, birad singlet, triplets like H2NN and N2H2(T)


    def test_validate_xyz(self):
        self.assertTrue(validate_xyz(self.xyz_methane))
        self.assertTrue(validate_xyz(self.xyz_water))
        self.assertTrue(validate_xyz(self.xyz_h))
        self.assertIsNone(validate_xyz(self.xyz_invalid))

    def test_infer_multiplicity(self):
        # CH4 has 6 + 4 = 10 electrons → singlet
        self.assertEqual(infer_multiplicity(['C', 'H', 'H', 'H', 'H'], total_charge=0), 1)
        # single H has 1 electron → doublet
        self.assertEqual(infer_multiplicity(['H'], total_charge=0), 2)
        # OH has 8 + 1 = 8 electrons → doublet
        self.assertEqual(infer_multiplicity(['O', 'H'], total_charge=0), 2)
        # OH– has 8 + 1 – 1 = 8 electrons → singlet
        self.assertEqual(infer_multiplicity(['O', 'H'], total_charge=-1), 1)

    def test_perceive_methane(self):
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

    def test_perceive_h_radical(self):
        mol = perceive_molecule_from_xyz(self.xyz_h)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 1)
        h = mol.atoms[0]
        self.assertEqual(h.element.symbol, 'H')
        # H• should have exactly one radical electron
        self.assertEqual(h.radical_electrons, 1)
        # no bonds
        self.assertEqual(len(mol.get_all_edges()), 0)
        # multiplicity is doublet
        self.assertEqual(mol.multiplicity, 2)

    def test_perceive_invalid(self):
        # malformed XYZ should return None, not raise
        mol = perceive_molecule_from_xyz(self.xyz_invalid)
        self.assertIsNone(mol)

    def test_get_representative_resonance_passthrough(self):
        # CH4 has no resonance, the representative form is itself
        mol = perceive_molecule_from_xyz(self.xyz_methane)
        rep = get_representative_resonance(mol)
        # isomorphism should hold
        self.assertTrue(mol.is_isomorphic(rep))

    def test_atom_order_preserved(self):
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
        self.assertIn(mol.to_smiles(), ['COC1C=CC([C](C)C)C=C1CO', 'COC1C=CC([C](C)C)C=C1CO'])

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
        mol = perceive_molecule_from_xyz(self.xyz_no2, multiplicity=1)
        self.assertIsNone(mol)
        mol = perceive_molecule_from_xyz(self.xyz_no2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 3)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertIn(mol.to_smiles(), ['[O-][N+]=O', '[O]N=O'])

    def test_ch3n2oh(self):
        """
        Test that perceiving CH3N2OH works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_ch3n2oh, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 8)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[O-][N+](=N)C')

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

    def test_c6_dd_yl(self):
        """
        Test that perceiving C6 with double bond and ylide works.
        """
        mol = perceive_molecule_from_xyz(self.xyz_c6_dd_yl, charge=0, multiplicity=2)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 15)
        self.assertEqual(mol.multiplicity, 2)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), '[CH2]C=CC=CC')

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

    def test_aibn(self):
        """
        Test that perceiving AIBN works.
        """
        mol = perceive_molecule_from_xyz(self.aibn, charge=0, multiplicity=1)
        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.atoms), 24)
        self.assertEqual(mol.multiplicity, 1)
        self.assertEqual(mol.get_net_charge(), 0)
        self.assertEqual(mol.to_smiles(), 'N#CC(N=NC(C#N)(C)C)(C)C')

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
        self.assertEqual(mol.to_smiles(), '[N-1]=[S+1]#N')





if __name__ == '__main__':
    unittest.main()
