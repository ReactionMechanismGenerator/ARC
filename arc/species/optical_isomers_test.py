#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.species.optical_isomers module (ARC issue #413)
"""

import os
import unittest
from unittest.mock import patch

import numpy as np

import arc.species.converter as converter
import arc.species.optical_isomers as oi
from arc.common import ARC_TESTING_PATH
from arc.parser.parser import parse_geometry
from arc.reaction import ARCReaction
from arc.species.conformers_test import (C2H5_XYZ, C2H6_XYZ, CH3_XYZ, CH4_XYZ, NC3H7OO_XYZ, QOOH_XYZ, TS0_XYZ, TS1_XYZ,
                                         TS_ETHANOL_OH_XYZ, ETHANOL_XYZ)
from arc.species.species import ARCSpecies


def rotor(pivots: list[int], torsion: list[int], success: bool | None, symmetry: int | None = 1, reason: str = '') -> dict:
    """
    Make a rotors_dict entry the way ARC stores it.

    Args:
        pivots (list[int]): The 1-indexed pivots.
        torsion (list[int]): The 0-indexed torsion atoms.
        success (bool | None): Whether the scan converged.
        symmetry (int | None, optional): The rotor symmetry number.
        reason (str, optional): The invalidation reason.

    Returns:
        dict: The rotor entry.
    """
    return {'pivots': pivots, 'torsion': torsion, 'scan': [i + 1 for i in torsion], 'success': success,
            'symmetry': symmetry, 'invalidation_reason': reason, 'dimensions': 1}


def treated(species: ARCSpecies) -> list[dict]:
    """
    Get the rotors that an Arkane species file would carry: the converged ones.

    Args:
        species (ARCSpecies): The species.

    Returns:
        list[dict]: The treated rotors.
    """
    return [r for r in species.rotors_dict.values() if r['success']]


# A twisted (C2-like, chiral) tetrahydrofuran ring, MMFF94.
TWISTED_THF_XYZ = """C -1.043394 -0.103182 0.157224
C -0.161580 0.879032 -0.570291
C 1.188480 0.610675 0.058383
O 1.209714 -0.778902 0.414688
C -0.124183 -1.295946 0.306460
H -1.304186 0.294068 1.145346
H -1.969475 -0.340769 -0.372514
H -0.481774 1.919062 -0.467777
H -0.132944 0.631262 -1.637997
H 2.019048 0.831256 -0.617917
H 1.320893 1.192720 0.976666
H -0.358562 -1.900396 1.187135
H -0.162038 -1.938880 -0.579405"""

# 2,2,4,4-tetramethylpentan-1-ol (achiral graph, C1 geometry), MMFF94. The two methyl groups on C6 are enantiotopic:
# the mirror image swaps them, so a rotor axis maps onto the other, equivalent axis.
GEM_DIMETHYL_XYZ = """C -2.504093 1.193404 0.213341
C -1.417933 0.098735 0.367440
C -1.801059 -1.064831 -0.553339
C -1.464389 -0.371137 1.833256
C -0.051111 0.762109 0.006870
C 1.265388 -0.047473 -0.233702
C 1.253092 -0.742628 -1.610584
C 1.548164 -1.095493 0.850641
C 2.454643 0.959066 -0.210574
O 2.302881 2.000586 -1.167225
H -2.558743 1.555177 -0.819846
H -2.293418 2.053398 0.859382
H -3.496759 0.813160 0.480954
H -1.171287 -1.942278 -0.384328
H -1.727544 -0.777771 -1.607707
H -2.834486 -1.384691 -0.373660
H -2.485669 -0.644658 2.124672
H -0.847081 -1.253685 2.010924
H -1.130427 0.419029 2.514761
H 0.155850 1.495945 0.800461
H -0.214523 1.370215 -0.895335
H 0.920797 -0.064946 -2.404145
H 0.601652 -1.619124 -1.628173
H 2.255634 -1.097402 -1.877811
H 2.548059 -1.528569 0.726786
H 1.506551 -0.652349 1.851354
H 0.842654 -1.929443 0.806831
H 2.535397 1.431797 0.774939
H 3.409260 0.456361 -0.405425
H 2.398498 1.607497 -2.050757"""

# Allenes with perpendicular terminal planes: 1,3-dichloroallene (C2, axially chiral) and propadiene (D2d).
DICHLOROALLENE_XYZ = """C 0.0 0.0 1.31
C 0.0 0.0 0.0
C 0.0 0.0 -1.31
Cl 1.50 0.0 2.21
H -0.93 0.0 1.85
Cl 0.0 1.50 -2.21
H 0.0 -0.93 -1.85"""
PROPADIENE_XYZ = """C 0.0 0.0 1.31
C 0.0 0.0 0.0
C 0.0 0.0 -1.31
H 0.93 0.0 1.85
H -0.93 0.0 1.85
H 0.0 0.93 -1.85
H 0.0 -0.93 -1.85"""

# trans (Cs) ethanol, the C-C-O-H dihedral of ETHANOL_XYZ set to 180 deg; atom 4 is the in-plane methyl H.
TRANS_ETHANOL_XYZ = """C 0.88812300 0.17014800 -0.00416600
C -0.46198200 -0.51592700 0.01912000
O -1.45981100 0.35927600 -0.48611300
H 0.87368300 1.07401600 0.61371400
H 1.67103100 -0.49715400 0.36682800
H 1.14323200 0.48412500 -1.02163900
H -0.73537900 -0.80995100 1.03700500
H -0.44640700 -1.41295900 -0.60668800
H -2.30834695 -0.11468574 -0.45531386"""

# H2 and F2 6 A apart: a bond graph that is not connected.
H2_F2_XYZ = """H 0.0 0.0 0.0
H 0.0 0.0 0.74
F 6.0 0.0 0.0
F 6.0 0.0 1.42"""

# Tub (D2d) cyclooctatetraene, MMFF94: C=C 1.341 A (atoms 0-1, 2-3, 4-5, 6-7), C-C 1.445 A.
COT_XYZ = """C -1.497387 -0.729237 0.447367
C -0.508351 -1.621859 0.291966
C 0.657149 -1.495448 -0.553062
C 1.556058 -0.499924 -0.550317
C 1.560917 0.669817 0.298284
C 0.571882 1.562438 0.453685
C -0.720681 1.554870 -0.192587
C -1.619589 0.559345 -0.195332
H -2.342864 -1.003840 1.075424
H -0.596632 -2.579844 0.801048
H 0.831611 -2.342946 -1.213367
H 2.418715 -0.585257 -1.208522
H 2.503550 0.853549 0.810514
H 0.757317 2.429551 1.084891
H -0.992295 2.493236 -0.672573
H -2.579400 0.735547 -0.677418"""

H2O2_THERMO_DIR = os.path.join(ARC_TESTING_PATH, 'restart', '1_restart_thermo', 'calcs')


class TestOpticalIsomers(unittest.TestCase):
    """
    Contains unit tests for the optical-isomer rule: 2 for a chiral geometry, unless the treated rotors reach the mirror.
    The R1 (nC3H7OO <=> QOOH_1_5) and R2 (CH3 + C2H6 <=> CH4 + C2H5) species, TSs and rotors are those of a
    wB97X-D/def2-TZVP ARC run.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.tol = oi.MIRROR_TORSION_TOLERANCE

    def make_nc3h7oo(self, c2_c3_success: bool | None = True) -> ARCSpecies:
        """nC3H7OO with the rotors of the R1 run; the C2-C3 (gauche) scan converged unless told otherwise"""
        spc = ARCSpecies(label='nC3H7OO', smiles='CCCO[O]', xyz=NC3H7OO_XYZ)
        spc.rotors_dict = {0: rotor([1, 2], [5, 0, 1, 2], True, 3),
                           1: rotor([2, 3], [0, 1, 2, 3], c2_c3_success,
                                    reason='' if c2_c3_success else 'scan has a barrier larger than 40 kJ/mol'),
                           2: rotor([3, 4], [1, 2, 3, 4], None, None, 'Another conformer exists which is 1.27 kJ/mol lower.')}
        return spc

    def make_qooh(self) -> ARCSpecies:
        """QOOH_1_5 with the rotors of the R1 run: its C2-C3 (gauche) scan failed"""
        spc = ARCSpecies(label='QOOH_1_5', smiles='[CH2]CCOO', xyz=QOOH_XYZ)
        spc.rotors_dict = {0: rotor([1, 2], [5, 0, 1, 2], True, 2),
                           1: rotor([2, 3], [0, 1, 2, 3], False, None,
                                    'Significant difference observed between consecutive conformers'),
                           2: rotor([3, 4], [1, 2, 3, 4], None, None, 'Another conformer exists which is 3.20 kJ/mol lower.'),
                           3: rotor([4, 5], [2, 3, 4, 11], True, 1)}
        return spc

    def make_r1(self, ring_rotors_success: bool | None = False) -> ARCReaction:
        """The R1 reaction, with the three TS rotors of the run, all of them torsions of the reactive ring"""
        rxn = ARCReaction(r_species=[self.make_nc3h7oo()], p_species=[self.make_qooh()])
        rxn.ts_species = ARCSpecies(label='TS0', is_ts=True, xyz=TS0_XYZ)
        rxn.ts_species.rotors_dict = {0: rotor([1, 2], [6, 0, 1, 2], ring_rotors_success),
                                      1: rotor([2, 3], [0, 1, 2, 3], ring_rotors_success),
                                      2: rotor([3, 4], [1, 2, 3, 4], ring_rotors_success)}
        return rxn

    def make_r2(self) -> ARCReaction:
        """The R2 reaction, with the TS CH3 rotor of the run"""
        species = [ARCSpecies(label=label, smiles=smiles, xyz=xyz) for label, smiles, xyz in
                   [('CH3', '[CH3]', CH3_XYZ), ('C2H6', 'CC', C2H6_XYZ), ('CH4', 'C', CH4_XYZ), ('C2H5', 'C[CH2]', C2H5_XYZ)]]
        rxn = ARCReaction(r_species=species[:2], p_species=species[2:])
        rxn.ts_species = ARCSpecies(label='TS1', is_ts=True, xyz=TS1_XYZ)
        rxn.ts_species.rotors_dict = {0: rotor([5, 6], [6, 4, 5, 9], True, 3),
                                      1: rotor([1, 6], [1, 0, 5, 9], None, None)}
        return rxn

    @staticmethod
    def h2o2(dihedral: float | None = None) -> ARCSpecies:
        """H2O2 at its wB97X-D/def2-TZVP geometry (C2), optionally with the H-O-O-H dihedral set, and its O-O rotor"""
        spc = ARCSpecies(label='H2O2', smiles='OO', xyz=parse_geometry(os.path.join(H2O2_THERMO_DIR, 'freq_a19031.out')))
        spc.rotors_dict = {0: rotor([1, 2], [2, 0, 1, 3], True, 1)}
        if dihedral is not None:
            spc.initial_xyz = oi.set_dihedral(spc.get_xyz(), bonds={(0, 1), (0, 2), (1, 3)},
                                              torsion=[2, 0, 1, 3], value=dihedral)
            spc.final_xyz = None
        return spc

    def test_r1_l_equals_6(self):
        """
        Test that R1's effective reaction path degeneracy, L = degeneracy * n_TS / n_R, is 6:
        three equivalent methyl H atoms, each transferred through either of two mirror-image cyclic TSs.
        nC3H7OO reaches its mirror through its converged C2-C3 rotor (1), the TS ring twist cannot (2).
        """
        rxn = self.make_r1()
        degeneracy = rxn.atom_map_clusters[0].degeneracy
        self.assertEqual(degeneracy, 3)
        n_r = oi.determine_optical_isomers(rxn.r_species[0], treated(rxn.r_species[0])).value
        n_ts = oi.determine_optical_isomers(rxn.ts_species, treated(rxn.ts_species), reaction=rxn).value
        self.assertEqual((n_r, n_ts), (1, 2))
        self.assertEqual(degeneracy * n_ts / n_r, 6)

    def test_r1_and_r2_values(self):
        """Test n and its cause for every well and TS of R1 and R2"""
        r1, r2 = self.make_r1(), self.make_r2()
        results = {spc.label: oi.determine_optical_isomers(spc, treated(spc)) for spc in r1.r_species + r1.p_species
                   + r2.r_species + r2.p_species}
        for rxn in (r1, r2):
            results[rxn.ts_species.label] = oi.determine_optical_isomers(rxn.ts_species, treated(rxn.ts_species),
                                                                         reaction=rxn)
        values = {label: result.value for label, result in results.items()}
        self.assertEqual(values, {'nC3H7OO': 1, 'QOOH_1_5': 2, 'TS0': 2,
                                  'CH3': 1, 'C2H6': 1, 'CH4': 1, 'C2H5': 1, 'TS1': 1})
        self.assertIn('mirror well reached by the C2-C3 rotor', results['nC3H7OO'].cause)
        self.assertIn('C2-C3 torsion harmonic (scan failed: Significant difference', results['QOOH_1_5'].cause)
        self.assertIn('ring twist', results['TS0'].cause)
        for label in ['CH3', 'C2H6', 'CH4', 'C2H5', 'TS1']:
            self.assertEqual(results[label].kind, 'achiral', label)

    def test_failed_scan_flips_n(self):
        """Test that the same nC3H7OO gives 1 with its gauche C2-C3 rotor treated and 2 when that scan failed"""
        converged = oi.determine_optical_isomers(self.make_nc3h7oo(True), treated(self.make_nc3h7oo(True)))
        self.assertEqual(converged.value, 1)
        for success in (False, None):
            spc = self.make_nc3h7oo(success)
            failed = oi.determine_optical_isomers(spc, treated(spc))
            self.assertEqual(failed.value, 2)
            self.assertEqual(failed.kind, 'not_reached')
            self.assertIn('C2-C3 torsion harmonic', failed.cause)
        skipped = oi.determine_optical_isomers(self.make_nc3h7oo(True), list())
        self.assertEqual(skipped.value, 2)
        self.assertIn('C2-C3 torsion harmonic (rotor not used in the Arkane file)', skipped.cause)

    def test_stereocentre_gives_2(self):
        """Test that 2-butanol gives 2 whatever its rotors do"""
        spc = ARCSpecies(label='sBuOH', smiles='CCC(C)O',
                         xyz=parse_geometry(os.path.join(ARC_TESTING_PATH, 'rotor_scans', 'sBuOH.out')))
        spc.determine_rotors()
        for r in spc.rotors_dict.values():
            r['success'] = True
        self.assertGreaterEqual(len(spc.rotors_dict), 3)
        for rotors in (treated(spc), list()):
            result = oi.determine_optical_isomers(spc, rotors)
            self.assertEqual(result.value, 2)
            self.assertEqual(result.kind, 'stereocentre')
            self.assertRegex(result.cause, r'^stereocentre at C\d+$')

    def test_ts_stereocentre_gives_2(self):
        """Test that the TS of H abstraction from the CH2 of ethanol (enantiotopic H atoms) gives 2 with no rotor"""
        ethanol = ARCSpecies(label='ethanol', smiles='CCO', xyz=ETHANOL_XYZ)
        oh = ARCSpecies(label='OH', smiles='[OH]', xyz="""O 0.0 0.0 0.0\nH 0.0 0.0 0.97""")
        rxn = ARCReaction(r_species=[ethanol, oh], p_species=[ARCSpecies(label='CH3CHOH', smiles='C[CH]O'),
                                                              ARCSpecies(label='H2O', smiles='O')])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=TS_ETHANOL_OH_XYZ)
        result = oi.determine_optical_isomers(rxn.ts_species, list(), reaction=rxn)
        self.assertEqual(result.value, 2)

    def test_ring_twist_is_never_a_rotor(self):
        """
        Test that the R1 TS gives 2 even if its reactive-ring torsions report converged scans: a ring twist is never
        a rotor that reaches the mirror. A twisted ring in a well gives 2 as well.
        """
        rxn = self.make_r1(ring_rotors_success=True)
        self.assertEqual(len(treated(rxn.ts_species)), 3)
        result = oi.determine_optical_isomers(rxn.ts_species, treated(rxn.ts_species), reaction=rxn)
        self.assertEqual(result.value, 2)
        self.assertIn('ring twist', result.cause)
        thf = ARCSpecies(label='THF', smiles='C1CCOC1', xyz=TWISTED_THF_XYZ)
        thf.rotors_dict = {0: rotor([1, 2], [4, 0, 1, 2], True), 1: rotor([2, 3], [0, 1, 2, 3], True)}
        self.assertGreater(oi.determine_optical_isomers(thf, list()).deviation, self.tol)
        result = oi.determine_optical_isomers(thf, treated(thf))
        self.assertEqual(result.value, 2)
        self.assertIn('ring twist about C1-C2', result.cause)

    def test_achiral_geometry_gives_1(self):
        """Test that achiral geometries give 1 with no rotor: C2H5 (Cs), trans-H2O2 (C2h), and the R2 TS (Cs)"""
        c2h5 = ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=C2H5_XYZ)
        rxn = self.make_r2()
        for spc, reaction in [(c2h5, None), (self.h2o2(180.0), None), (rxn.ts_species, rxn)]:
            result = oi.determine_optical_isomers(spc, list(), reaction=reaction)
            self.assertEqual((result.value, result.kind), (1, 'achiral'), spc.label)
            self.assertRegex(result.cause, r'^achiral optimised geometry \(D = \d+\.\d deg\)$')

    def test_symmetric_tops(self):
        """
        Test symmetric tops. The CH3 top of nC3H7OO needs no rotor to reach the mirror (a permutation of its H atoms
        does it), so nC3H7OO gives 1 with only its C2-C3 rotor. The pyramidal NH2 top of N2H3 reaches its mirror
        through the N-N rotor only by mapping one H atom onto the mirror position of the other: 1 with the rotor, 2
        without. H2O2 (C2, external symmetry 2) reaches its mirror through its O-O rotor: 1.
        """
        spc = self.make_nc3h7oo()
        result = oi.determine_optical_isomers(spc, [spc.rotors_dict[1]])
        self.assertEqual(result.value, 1)
        self.assertIn('C2-C3', result.cause)
        self.assertNotIn('C1-C2', result.cause)
        n2h3 = ARCSpecies(label='N2H3', smiles='N[NH]', xyz="""N      -0.42467300    0.01074000    0.16766900
N       0.75507100   -0.58161100   -0.06802100
H      -0.56364100    0.98902900   -0.04228800
H      -1.22068800   -0.57930900   -0.00745700
H       1.45393100    0.16115000   -0.04990200""")
        n2h3.rotors_dict = {0: rotor([1, 2], [2, 0, 1, 4], True)}
        self.assertEqual(oi.determine_optical_isomers(n2h3, treated(n2h3)).value, 1)
        self.assertEqual(oi.determine_optical_isomers(n2h3, list()).value, 2)
        h2o2 = self.h2o2()
        result = oi.determine_optical_isomers(h2o2, treated(h2o2))
        self.assertEqual((result.value, result.kind), (1, 'reached'))
        self.assertEqual(oi.determine_optical_isomers(h2o2, list()).value, 2)

    def test_rotor_axis_maps_onto_an_equivalent_axis(self):
        """
        Test that the mirror image is reached when it swaps two enantiotopic methyl groups, so that a rotor's own axis
        maps onto the other, equivalent axis (the rotor targets come from each graph-symmetry map).
        """
        spc = ARCSpecies(label='tmp', smiles='CC(C)(C)CC(C)(C)CO', xyz=GEM_DIMETHYL_XYZ)
        spc.determine_rotors()
        for r in spc.rotors_dict.values():
            r['success'] = True
        result = oi.determine_optical_isomers(spc, treated(spc))
        self.assertEqual((result.value, result.kind), (1, 'reached'))
        self.assertLess(result.deviation, 10.0)
        self.assertEqual(oi.determine_optical_isomers(spc, list()).value, 2)

    def test_isotopes_are_not_permuted(self):
        """
        Test that D and H are never swapped when matching the mirror image: trans-ethanol (Cs) with one out-of-plane
        methyl H replaced by D is chiral (2), unless its CH3 rotor is treated (1); with the in-plane H replaced it stays
        achiral (1).
        """
        for index, expected_none, expected_rotor in [(3, 2, 1), (4, 1, 1)]:
            xyz = converter.str_to_xyz(TRANS_ETHANOL_XYZ)
            isotopes = list(xyz['isotopes'])
            isotopes[index] = 2
            spc = ARCSpecies(label='ethanol-d1', smiles='CCO', xyz=TRANS_ETHANOL_XYZ)
            spc.initial_xyz = converter.xyz_from_data(coords=xyz['coords'], symbols=xyz['symbols'], isotopes=isotopes)
            spc.rotors_dict = {0: rotor([1, 2], [index, 0, 1, 2], True, 3)}
            self.assertEqual(oi.determine_optical_isomers(spc, list()).value, expected_none, index)
            self.assertEqual(oi.determine_optical_isomers(spc, treated(spc)).value, expected_rotor, index)

    def test_axial_chirality(self):
        """Test that torsions across a linear C=C=C chain are compared: 1,3-dichloroallene is chiral, propadiene not"""
        for label, smiles, xyz, expected in [('ClCH=C=CHCl', 'ClC=C=CCl', DICHLOROALLENE_XYZ, 2),
                                             ('propadiene', 'C=C=C', PROPADIENE_XYZ, 1)]:
            spc = ARCSpecies(label=label, smiles=smiles, xyz=xyz)
            self.assertEqual(oi.determine_optical_isomers(spc, list()).value, expected, label)
        reach = oi.MirrorReach(converter.str_to_xyz(DICHLOROALLENE_XYZ), {(0, 1), (1, 2), (0, 3), (0, 4), (2, 5), (2, 6)},
                               oi.MAX_SYMMETRY_MATCHES)
        self.assertGreater(reach.best(list())[0], self.tol)

    def test_search_budget(self):
        """Test that a mirror search over its budget gives an undetermined result, not a hang"""
        with patch('arc.species.optical_isomers.MAX_SEARCH_EVALUATIONS', 3):
            result = oi.determine_optical_isomers(self.make_nc3h7oo(), treated(self.make_nc3h7oo()))
        self.assertIsNone(result.value)
        self.assertIn('exceeded 3 comparisons', result.cause)

    def test_needed_rotors_cannot_erase_a_reached_mirror(self):
        """Test that running out of budget while naming the needed rotors keeps n = 1 once the mirror is reached"""
        spc = self.make_nc3h7oo()
        with patch.object(oi.MirrorReach, 'needed', side_effect=oi.MirrorSearchError('budget')):
            result = oi.determine_optical_isomers(spc, treated(spc))
        self.assertEqual((result.value, result.kind), (1, 'reached'))
        self.assertIn('C1-C2', result.cause)
        self.assertIn('C2-C3', result.cause)

    def test_symmetry_maps_keep_bond_lengths(self):
        """Test that no symmetry map swaps the single and double bonds of cyclooctatetraene (its bond graph does not)"""
        xyz = converter.str_to_xyz(COT_XYZ)
        ring = {(i, (i + 1) % 8) for i in range(8)}
        reach = oi.MirrorReach(xyz, ring | {(i, i + 8) for i in range(8)}, oi.MAX_SYMMETRY_MATCHES)
        self.assertEqual(len(reach.maps), 8)
        coords = reach.coords
        for symmetry_map in reach.maps:
            for i, j in ring:
                self.assertAlmostEqual(float(np.linalg.norm(coords[i] - coords[j])),
                                       float(np.linalg.norm(coords[symmetry_map[i]] - coords[symmetry_map[j]])),
                                       delta=oi.BOND_LENGTH_TOLERANCE)
        spc = ARCSpecies(label='COT', smiles='C1=CC=CC=CC=C1', xyz=COT_XYZ)
        self.assertEqual(oi.determine_optical_isomers(spc, list()).value, 1)

    def test_threshold_flips_n(self):
        """
        Test that D just under and just over MIRROR_TORSION_TOLERANCE flips n: H2O2 with its H-O-O-H dihedral at
        180 - delta and no rotor deviates by D = 2 * delta from its mirror image.
        """
        under = oi.determine_optical_isomers(self.h2o2(180.0 - (self.tol - 1.0) / 2), list())
        over = oi.determine_optical_isomers(self.h2o2(180.0 - (self.tol + 1.0) / 2), list())
        self.assertEqual((under.value, over.value), (1, 2))
        self.assertAlmostEqual(under.deviation, self.tol - 1.0, places=1)
        self.assertAlmostEqual(over.deviation, self.tol + 1.0, places=1)
        self.assertIn(f'D = {self.tol + 1.0:.1f} deg', over.cause)

    def test_mirror_torsion_deviation(self):
        """Test the metric: the largest signed-torsion deviation between a geometry and an aligned mirror image"""
        xyz = self.h2o2(100.0).get_xyz()
        mirror = converter.xyz_from_data(coords=[(-x, y, z) for x, y, z in xyz['coords']], symbols=xyz['symbols'])
        identity = {i: i for i in range(4)}
        deviation, worst = oi.get_mirror_torsion_deviation(xyz['coords'], mirror['coords'], identity,
                                                           torsions=[(2, 0, 1, 3)])
        self.assertAlmostEqual(deviation, 160.0, places=3)
        self.assertEqual(worst, {(2, 0, 1, 3): deviation})
        deviation, _ = oi.get_mirror_torsion_deviation(xyz['coords'], xyz['coords'], identity, torsions=[(2, 0, 1, 3)])
        self.assertAlmostEqual(deviation, 0.0, places=6)

    def test_user_value_wins(self):
        """Test that a valid user value is kept, and its cause says so"""
        spc = self.make_nc3h7oo()
        spc.optical_isomers = 2
        result = oi.determine_optical_isomers(spc, treated(spc))
        self.assertEqual((result.value, result.kind, result.cause), (2, 'user', 'given by the user'))

    def test_undetermined(self):
        """Test that ARC returns None with a reason when it cannot decide: no geometry, a wrong atom order, a graph
        that is not connected, or too many symmetry matches"""
        spc = ARCSpecies(label='CH4', smiles='C')
        spc.initial_xyz = spc.final_xyz = None
        spc.conformers = list()
        cases = {'no optimised geometry': spc}
        wrong = ARCSpecies(label='C2H6', smiles='CC', xyz=C2H6_XYZ)
        wrong.initial_xyz = converter.xyz_from_data(coords=wrong.get_xyz()['coords'][::-1],
                                                    symbols=wrong.get_xyz()['symbols'][::-1])
        wrong.final_xyz = None
        cases['atom order'] = wrong
        cases['not connected'] = ARCSpecies(label='TS', is_ts=True, multiplicity=1, xyz=H2_F2_XYZ)
        for reason, species in cases.items():
            result = oi.determine_optical_isomers(species, list())
            self.assertIsNone(result.value, reason)
            self.assertEqual(result.kind, 'undetermined', reason)
            self.assertIn(reason, result.cause)
        result = oi.determine_optical_isomers(ARCSpecies(label='C2H6', smiles='CC', xyz=C2H6_XYZ), list(),
                                              max_matches=2)
        self.assertIsNone(result.value)
        self.assertIn('symmetry matches', result.cause)

    def test_every_cause_has_a_case(self):
        """
        Drift test: every cause template is produced by a named case of this module, and each produced cause text is
        its template formatted, so a template cannot change or be added without a case that produces it.
        """
        spc = self.make_nc3h7oo()
        user = self.make_nc3h7oo()
        user.optical_isomers = 1
        r1 = self.make_r1()
        n2h3_like = self.make_nc3h7oo(False)
        cases = {'user': (user, treated(user), None),
                 'stereocentre': (ARCSpecies(label='sBuOH', smiles='CCC(C)O', xyz=parse_geometry(
                     os.path.join(ARC_TESTING_PATH, 'rotor_scans', 'sBuOH.out'))), list(), None),
                 'achiral': (ARCSpecies(label='CH4', smiles='C', xyz=CH4_XYZ), list(), None),
                 'reached': (spc, treated(spc), None),
                 'not_reached': (n2h3_like, treated(n2h3_like), None),
                 'undetermined': (ARCSpecies(label='TS', is_ts=True, multiplicity=1, xyz=H2_F2_XYZ), list(), None),
                 }
        self.assertEqual(set(cases), set(oi.OPTICAL_ISOMER_CAUSES))
        for kind, (species, rotors, reaction) in cases.items():
            result = oi.determine_optical_isomers(species, rotors, reaction=reaction)
            self.assertEqual(result.kind, kind)
            self.assertEqual(result.cause, oi.format_cause(kind, **result.fields))
        parts = {oi.determine_optical_isomers(*args[:2], reaction=args[2]).fields.get('parts', '')
                 for args in [(n2h3_like, treated(n2h3_like), None), (r1.ts_species, list(), r1)]}
        self.assertTrue(any('torsion harmonic' in p for p in parts))
        self.assertTrue(any('ring twist' in p for p in parts))
        self.assertEqual(set(oi.OPTICAL_ISOMER_CAUSE_PARTS), {'harmonic', 'ring_twist', 'configuration', 'unnamed'})
        with patch('arc.species.optical_isomers.explain_unreached', side_effect=ValueError('boom')):
            unnamed = oi.determine_optical_isomers(n2h3_like, treated(n2h3_like))
        self.assertEqual(unnamed.value, 2)
        self.assertIn('the cause could not be named: ValueError: boom', unnamed.cause)
        ts_result = oi.determine_optical_isomers(*self._ts_stereo_case())
        self.assertIn('configuration at C', ts_result.cause)

    def _ts_stereo_case(self) -> tuple:
        """The ethanol + OH CH2-abstraction TS, with no rotor"""
        ethanol = ARCSpecies(label='ethanol', smiles='CCO', xyz=ETHANOL_XYZ)
        oh = ARCSpecies(label='OH', smiles='[OH]', xyz="""O 0.0 0.0 0.0\nH 0.0 0.0 0.97""")
        rxn = ARCReaction(r_species=[ethanol, oh], p_species=[ARCSpecies(label='CH3CHOH', smiles='C[CH]O'),
                                                              ARCSpecies(label='H2O', smiles='O')])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=TS_ETHANOL_OH_XYZ)
        return rxn.ts_species, list(), rxn


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
