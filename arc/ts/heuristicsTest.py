#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for TS guess generation methods using heuristics
"""

import unittest

from rmgpy.molecule import Molecule
from rmgpy.reaction import Reaction
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.common import almost_equal_coords_lists
from arc.species import ARCSpecies
from arc.ts import heuristics


class TestHeuristics(unittest.TestCase):
    """
    Contains unit tests for TS Heuristics
    """

    @classmethod
    def setUpClass(cls):
        """
        A function run ONCE before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.cooh_xyz = {'symbols': ('C', 'O', 'O', 'H', 'H', 'H', 'H'), 'isotopes': (12, 16, 16, 1, 1, 1, 1),
                        'coords': ((-0.7603907249238114, 0.014838579473692097, -0.009033436538206244),
                                   (0.44475332654100314, 0.7695210236307556, 0.022913026113559408),
                                   (0.16024510925528873, 1.9232790439491338, 0.8638179999945598),
                                   (-1.5663233709772881, 0.6140163013935507, -0.44251281684009025),
                                   (-1.0294331629372442, -0.3044915636223426, 1.0019370868108535),
                                   (-0.6005250700057971, -0.8695449463297907, -0.6308643778066679),
                                   (0.303913438301588, 2.5962913882693015, 0.17435158799241526))}
        cls.ch3cho_xyz = {'symbols': ('O', 'C', 'C', 'H', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1, 1),
                          'coords': ((1.5235984715331445, 1.0194091983360514, 0.011481512100019435),
                                     (0.8340762468248668, 0.013950694248468754, 0.1456681399913643),
                                     (-0.6410882644515854, -0.015555832637685734, -0.11072549877575433),
                                     (1.2805838938746499, -0.9410608625715902, 0.4702026149555385),
                                     (-0.8706477844102058, -0.7390662067201391, -0.8968521676392714),
                                     (-0.9694596003573355, 0.9781441274537922, -0.42403322388984643),
                                     (-1.1580665961750232, -0.3010330887916075, 0.8083311916826158))}
        cls.oh_xyz = {'symbols': ('O', 'H'), 'isotopes': (16, 1),
                      'coords': ((0.48890386738601, 0.0, 0.0), (-0.48890386738601, 0.0, 0.0))}
        cls.h2o2_xyz = {'symbols': ('O', 'O', 'H', 'H'), 'isotopes': (16, 16, 1, 1),
                        'coords': ((0.5811911885918241, -0.43123225989526554, 0.2185585670937377),
                                   (-0.5786509543923134, 0.4454116101690339, 0.19566259711221187),
                                   (1.198868443733836, 0.18357273527858012, -0.2209970731247096),
                                   (-1.2014086871626892, -0.197752096887578, -0.19322405755462888))}

    def test_combine_coordinates_with_redundant_atoms(self):
        """Test combining two coordinates with a redundant atom"""
        combined_xyz = heuristics.combine_coordinates_with_redundant_atoms(
            xyz1=self.cooh_xyz, xyz2=self.ch3cho_xyz,
            mol1=ARCSpecies(label='cooh', xyz=self.cooh_xyz).mol,
            mol2=ARCSpecies(label='ch3coh', xyz=self.ch3cho_xyz).mol,
            h1=6, h2=3, c=1, d=2, r1_stretch=1.2, r2_stretch=1.2, a2=160, d2=30, d3=50)
        expected_xyz = {'symbols': ('C', 'O', 'O', 'H', 'H', 'H', 'H', 'C', 'C', 'O', 'H', 'H', 'H'),
                        'isotopes': (12, 16, 16, 1, 1, 1, 1, 12, 12, 16, 1, 1, 1),
                        'coords': ((-1.2978075206170019, -0.675204324277114, -1.8563699411608872),
                                   (-1.2978075206170019, -0.675204324277114, -0.4340695193392734),
                                   (-1.2978075206170019, 0.7273347146483065, -0.04405946952557316),
                                   (-0.4003810124818391, -0.17452720698474777, -2.231061693714762),
                                   (-2.1943545032130363, -0.17298624265683804, -2.2310663000302973),
                                   (-1.2986630471212148, -1.7112643913562684, -2.204139564382495),
                                   (-0.2697634711098247, 0.7060569039243947, 0.5119627982839361),
                                   (0.9677274470415582, 0.3562472201883693, 0.8255758431423998),
                                   (1.7801438331639898, -0.22044312383012477, -0.2925146365824114),
                                   (1.3959118980533247, 0.5054868920333042, 1.9652158271041513),
                                   (1.2996299857576912, -1.1228365473211788, -0.6784583323199047),
                                   (2.7759199971403596, -0.4701759174474849, 0.08097227815887287),
                                   (1.8621107702820516, 0.5184885833708908, -1.0929460613446231))}
        self.assertTrue(almost_equal_coords_lists(combined_xyz, expected_xyz))

        # test different indices for the same reactants
        combined_xyz = heuristics.combine_coordinates_with_redundant_atoms(
            xyz1=self.cooh_xyz, xyz2=self.ch3cho_xyz,
            mol1=ARCSpecies(label='cooh', xyz=self.cooh_xyz).mol,
            mol2=ARCSpecies(label='ch3coh', xyz=self.ch3cho_xyz).mol,
            h1=6, h2=4, c=1, d=1, r1_stretch=1.2, r2_stretch=1.5, a2=170, d2=40, d3=170)
        expected_xyz = {'symbols': ('C', 'O', 'O', 'H', 'H', 'H', 'H', 'C', 'C', 'O', 'H', 'H', 'H'),
                        'isotopes': (12, 16, 16, 1, 1, 1, 1, 12, 12, 16, 1, 1, 1),
                        'coords': ((-1.4688105492673553, -1.2757783307117707, -2.3036015013684574),
                                   (-1.4688105492673553, -1.2757783307117707, -0.8813010795468437),
                                   (-1.4688105492673553, 0.1267607082136497, -0.4912910297331434),
                                   (-0.5713840411321925, -0.7751012134194045, -2.678293253922332),
                                   (-2.36535753186339, -0.7735602490914948, -2.6782978602378678),
                                   (-1.4696660757715683, -2.3118383977909254, -2.6513711245900655),
                                   (-0.4407664997601781, 0.10548289748973794, 0.06473123807636583),
                                   (1.0853434249034903, -0.0923230998840463, 0.6292852661112529),
                                   (1.6432578460873113, 1.2242291678150101, 1.0744145543267702),
                                   (2.0147220395854317, 1.4478912810117133, 2.221752041230318),
                                   (1.7045400480136113, 2.0034977328879737, 0.2961320245967962),
                                   (1.0727945927223235, -0.7805065859732105, 1.4776219496982836),
                                   (1.716805003860345, -0.4982536462092362, -0.16438077089413383))}
        self.assertTrue(almost_equal_coords_lists(combined_xyz, expected_xyz))

        # test a 180 degree angle
        combined_xyz = heuristics.combine_coordinates_with_redundant_atoms(
            xyz1=self.cooh_xyz, xyz2=self.ch3cho_xyz,
            mol1=ARCSpecies(label='cooh', xyz=self.cooh_xyz).mol,
            mol2=ARCSpecies(label='ch3coh', xyz=self.ch3cho_xyz).mol,
            h1=6, h2=4, c=1, d=1, r1_stretch=1.2, r2_stretch=1.2, a2=180, d3=-20, keep_dummy=True)
        expected_xyz = {'symbols': ('C', 'O', 'O', 'H', 'H', 'H', 'H', 'C', 'C', 'O', 'H', 'H', 'H', 'X'),
                        'isotopes': (12, 16, 16, 1, 1, 1, 1, 12, 12, 16, 1, 1, 1, None),
                        'coords': ((-1.3470612426624362, -0.3724892487516244, -2.08731143239368),
                                   (-1.3470612426624362, -0.3724892487516244, -0.6650110105720664),
                                   (-1.3470612426624362, 1.030049790173796, -0.2750009607583661),
                                   (-0.4496347345272734, 0.12818786854074177, -2.4620031849475548),
                                   (-2.2436082252584706, 0.1297288328686515, -2.4620077912630904),
                                   (-1.3479167691666492, -1.408549315830779, -2.435081055615288),
                                   (-0.319017193155259, 1.0087719794498842, 0.28102130705114314),
                                   (0.8342270722727674, 0.9849028532096444, 0.9047586771657747),
                                   (1.1904986164590234, -0.4224109920132912, 1.2725314140871453),
                                   (2.294113146222996, -0.9162994215755049, 1.0663691342603303),
                                   (0.3874481465430699, -1.0077177227937755, 1.7513473229054797),
                                   (1.7001350436779552, 1.463511393651899, 0.4415455917427975),
                                   (0.5521977589731284, 1.5301121981837396, 1.8084374635866136),
                                   (-0.22178193307831817, 0.03744435296902887, 0.06406982345898715))}
        self.assertTrue(almost_equal_coords_lists(combined_xyz, expected_xyz))

        # test specifying d2 needlessly
        combined_xyz = heuristics.combine_coordinates_with_redundant_atoms(
            xyz1=self.cooh_xyz, xyz2=self.ch3cho_xyz,
            mol1=ARCSpecies(label='cooh', xyz=self.cooh_xyz).mol,
            mol2=ARCSpecies(label='ch3coh', xyz=self.ch3cho_xyz).mol,
            h1=6, h2=4, c=1, d=1, r1_stretch=1.2, r2_stretch=1.2, a2=180, d2=35, d3=-20, keep_dummy=True)
        self.assertTrue(almost_equal_coords_lists(combined_xyz, expected_xyz))

        # test not keeping the dummy atom
        combined_xyz = heuristics.combine_coordinates_with_redundant_atoms(
            xyz1=self.cooh_xyz, xyz2=self.ch3cho_xyz,
            mol1=ARCSpecies(label='cooh', xyz=self.cooh_xyz).mol,
            mol2=ARCSpecies(label='ch3coh', xyz=self.ch3cho_xyz).mol,
            h1=6, h2=4, c=1, d=1, r1_stretch=1.2, r2_stretch=1.2, a2=180, d3=-20, keep_dummy=False)
        expected_xyz = {'symbols': ('C', 'O', 'O', 'H', 'H', 'H', 'H', 'C', 'C', 'O', 'H', 'H', 'H'),
                        'isotopes': (12, 16, 16, 1, 1, 1, 1, 12, 12, 16, 1, 1, 1),
                        'coords': ((-1.3470612426624362, -0.3724892487516244, -2.08731143239368),
                                   (-1.3470612426624362, -0.3724892487516244, -0.6650110105720664),
                                   (-1.3470612426624362, 1.030049790173796, -0.2750009607583661),
                                   (-0.4496347345272734, 0.12818786854074177, -2.4620031849475548),
                                   (-2.2436082252584706, 0.1297288328686515, -2.4620077912630904),
                                   (-1.3479167691666492, -1.408549315830779, -2.435081055615288),
                                   (-0.319017193155259, 1.0087719794498842, 0.28102130705114314),
                                   (0.8342270722727674, 0.9849028532096444, 0.9047586771657747),
                                   (1.1904986164590234, -0.4224109920132912, 1.2725314140871453),
                                   (2.294113146222996, -0.9162994215755049, 1.0663691342603303),
                                   (0.3874481465430699, -1.0077177227937755, 1.7513473229054797),
                                   (1.7001350436779552, 1.463511393651899, 0.4415455917427975),
                                   (0.5521977589731284, 1.5301121981837396, 1.8084374635866136))}
        self.assertTrue(almost_equal_coords_lists(combined_xyz, expected_xyz))

        # test a small abstractor
        combined_xyz = heuristics.combine_coordinates_with_redundant_atoms(
            xyz1=self.cooh_xyz, xyz2=self.oh_xyz,
            mol1=ARCSpecies(label='cooh', xyz=self.cooh_xyz).mol,
            mol2=ARCSpecies(label='OH', xyz=self.oh_xyz).mol,
            h1=6, h2=1, c=1, d=None, r1_stretch=1.2, r2_stretch=1.2, a2=170, d2=-10, d3=None, keep_dummy=True)
        expected_xyz = {'symbols': ('C', 'O', 'O', 'H', 'H', 'H', 'H', 'O'),
                        'isotopes': (12, 16, 16, 1, 1, 1, 1, 16),
                        'coords': ((-0.5277206969373229, -0.6607286071182872, -1.553942203583733),
                                   (-0.5277206969373229, -0.6607286071182872, -0.13164178176211916),
                                   (-0.5277206969373229, 0.7418104318071332, 0.2583682680515811),
                                   (0.36970581119783985, -0.160051489825921, -1.9286339561376078),
                                   (-1.4242676795333575, -0.15851052549801126, -1.928638562453143),
                                   (-0.5285762234415358, -1.6967886741974416, -1.901711826805341),
                                   (0.5003233525698543, 0.7205326210832215, 0.8143905358610903),
                                   (1.519584925695443, 0.4962074929853171, 1.350655070321085))}
        self.assertTrue(almost_equal_coords_lists(combined_xyz, expected_xyz))

        # test a HO2 as the abstractor
        combined_xyz = heuristics.combine_coordinates_with_redundant_atoms(
            xyz1=self.cooh_xyz, xyz2=self.h2o2_xyz,
            mol1=ARCSpecies(label='cooh', xyz=self.cooh_xyz).mol,
            mol2=ARCSpecies(label='OO', xyz=self.h2o2_xyz).mol,
            h1=6, h2=2, c=1, d=1, r1_stretch=1.2, r2_stretch=1.2, a2=170, d2=90, d3=180, keep_dummy=True)
        expected_xyz = {'symbols': ('C', 'O', 'O', 'H', 'H', 'H', 'H', 'O', 'O', 'H'),
                        'isotopes': (12, 16, 16, 1, 1, 1, 1, 16, 16, 1),
                        'coords': ((-0.8782298093057832, -1.1445711335632867, -1.9031271960832787),
                                   (-0.8782298093057832, -1.1445711335632867, -0.480826774261665),
                                   (-0.8782298093057832, 0.25796790536213376, -0.09081672444796474),
                                   (0.019196698829379577, -0.6438940162709205, -2.2778189486371536),
                                   (-1.7747767919018178, -0.6423530519430107, -2.277823554952689),
                                   (-0.879085335809996, -2.1806312006424413, -2.2508968193048866),
                                   (0.149814240201394, 0.236690094638222, 0.4652055433615445),
                                   (1.2590219090721202, 0.2639074604339753, 0.8404835340739159),
                                   (1.2131424490862195, 1.5990841477239615, 1.4144703542538937),
                                   (1.583029482404546, 1.3624084406608703, 2.2861858357526907))}
        self.assertTrue(almost_equal_coords_lists(combined_xyz, expected_xyz))

    def test_label_molecules(self):
        """Test labeling atoms in a reaction"""
        # test OH + O=CCC <=> H2O + O=CC[CH2]
        reactants = [Molecule().from_smiles('[OH]'), Molecule().from_smiles('O=CCC')]
        products = [Molecule().from_smiles('O'), Molecule().from_smiles('O=CC[CH2]')]
        db = rmgdb.make_rmg_database_object()
        rmgdb.load_families_only(db)
        family = rmgdb.determine_reaction_family(rmgdb=db,
                                                 reaction=Reaction(label='R1',
                                                                   reactants=[Species(molecule=[reactants[0]]),
                                                                              Species(molecule=[reactants[1]])],
                                                                   products=[Species(molecule=[products[0]]),
                                                                             Species(molecule=[products[1]])]))[0]
        reaction = heuristics.label_molecules(reactants=reactants, products=products, family=family)
        self.assertEqual(family.label, 'H_Abstraction')
        self.assertEqual(reaction.reactants[0].molecule[0].get_labeled_atoms('*3')[0].atomtype.label, 'O2s')
        self.assertEqual(reaction.reactants[1].molecule[0].get_labeled_atoms('*1')[0].atomtype.label, 'Cs')
        self.assertEqual(reaction.reactants[1].molecule[0].get_labeled_atoms('*2')[0].atomtype.label, 'H')
        self.assertEqual(reaction.products[0].molecule[0].get_labeled_atoms('*1')[0].atomtype.label, 'O2s')
        self.assertEqual(reaction.products[0].molecule[0].get_labeled_atoms('*2')[0].atomtype.label, 'H')
        self.assertEqual(reaction.products[1].molecule[0].get_labeled_atoms('*3')[0].atomtype.label, 'Cs')

        # test the above reaction with a different reactant order: O=CCC + OH <=> H2O + O=CC[CH2]
        reactants = [Molecule().from_smiles('O=CCC'), Molecule().from_smiles('[OH]')]
        products = [Molecule().from_smiles('O'), Molecule().from_smiles('O=CC[CH2]')]
        db = rmgdb.make_rmg_database_object()
        rmgdb.load_families_only(db)
        family = rmgdb.determine_reaction_family(rmgdb=db,
                                                 reaction=Reaction(label='R1',
                                                                   reactants=[Species(molecule=[reactants[0]]),
                                                                              Species(molecule=[reactants[1]])],
                                                                   products=[Species(molecule=[products[0]]),
                                                                             Species(molecule=[products[1]])]))[0]
        reaction = heuristics.label_molecules(reactants=reactants, products=products, family=family)
        self.assertEqual(family.label, 'H_Abstraction')
        self.assertEqual(reaction.reactants[0].molecule[0].get_labeled_atoms('*1')[0].atomtype.label, 'Cs')
        self.assertEqual(reaction.reactants[0].molecule[0].get_labeled_atoms('*2')[0].atomtype.label, 'H')
        self.assertEqual(reaction.reactants[1].molecule[0].get_labeled_atoms('*3')[0].atomtype.label, 'O2s')
        self.assertEqual(reaction.products[0].molecule[0].get_labeled_atoms('*1')[0].atomtype.label, 'O2s')
        self.assertEqual(reaction.products[0].molecule[0].get_labeled_atoms('*2')[0].atomtype.label, 'H')
        self.assertEqual(reaction.products[1].molecule[0].get_labeled_atoms('*3')[0].atomtype.label, 'Cs')

        reactants = [Molecule().from_smiles('CO[O]'), Molecule().from_smiles('c1cc3c(cc1)CCc2c(cccc2)N3CCCN(C)C')]
        products = [Molecule().from_smiles('COO'), Molecule().from_smiles('c1cc3c(cc1)CCc2c(cccc2)N3[CH]CCN(C)C')]
        db = rmgdb.make_rmg_database_object()
        rmgdb.load_families_only(db)
        family = rmgdb.determine_reaction_family(rmgdb=db,
                                                 reaction=Reaction(label='R1',
                                                                   reactants=[Species(molecule=[reactants[0]]),
                                                                              Species(molecule=[reactants[1]])],
                                                                   products=[Species(molecule=[products[0]]),
                                                                             Species(molecule=[products[1]])]))[0]
        reaction = heuristics.label_molecules(reactants=reactants, products=products, family=family)
        self.assertEqual(family.label, 'H_Abstraction')
        self.assertEqual(reaction.reactants[0].molecule[0].get_labeled_atoms('*3')[0].atomtype.label, 'O2s')
        self.assertEqual(reaction.reactants[1].molecule[0].get_labeled_atoms('*1')[0].atomtype.label, 'Cs')
        self.assertEqual(reaction.reactants[1].molecule[0].get_labeled_atoms('*2')[0].atomtype.label, 'H')
        self.assertEqual(reaction.products[0].molecule[0].get_labeled_atoms('*1')[0].atomtype.label, 'O2s')
        self.assertEqual(reaction.products[0].molecule[0].get_labeled_atoms('*2')[0].atomtype.label, 'H')
        self.assertEqual(reaction.products[1].molecule[0].get_labeled_atoms('*3')[0].atomtype.label, 'Cs')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
