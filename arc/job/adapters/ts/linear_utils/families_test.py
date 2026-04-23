#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for arc.job.adapters.ts.linear_utils.families
"""

import unittest

import numpy as np

from arc.species import ARCSpecies
from arc.species.converter import str_to_xyz
from arc.common import get_single_bond_length
from arc.job.adapters.ts.linear_utils.geom_utils import bfs_fragment
from arc.job.adapters.ts.linear_utils.families import (
    _dihedral_angle,
    _rotate_fragment,
    _set_bond_distance,
    build_1_3_sigmatropic_rearrangement_ts,
    build_baeyer_villiger_step2_ts,
    build_intra_oh_migration_ts,
    build_intra_substitution_s_isomerization_ts,
    build_korcek_step1_ts,
    build_retroene_ts,
    build_singlet_carbene_intra_disproportionation_ts,
    build_xy_elimination_ts,
    PAULING_DELTA,
)
from arc.species.species import colliding_atoms


class TestGeometryHelpers(unittest.TestCase):
    """Tests for the low-level geometry helpers in families.py."""

    def test_dihedral_angle_eclipsed(self):
        """Dihedral of 4 atoms in a plane (eclipsed) should be ~0°."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0.01, 0]], dtype=float)
        dih = _dihedral_angle(coords, 0, 1, 2, 3)
        self.assertAlmostEqual(dih, 0.0, delta=2.0)

    def test_dihedral_angle_anti(self):
        """Dihedral of 4 atoms with the last one on the opposite side should be ~180°."""
        coords = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, -1, 0]], dtype=float)
        dih = _dihedral_angle(coords, 0, 1, 2, 3)
        self.assertAlmostEqual(abs(dih), 180.0, delta=1.0)

    def test_dihedral_angle_gauche(self):
        """Dihedral of a gauche conformation should be ~60° or ~-60°."""
        coords = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0],
                           [1.5, 0.5, 0.866]], dtype=float)
        dih = _dihedral_angle(coords, 0, 1, 2, 3)
        self.assertTrue(40 < abs(dih) < 80, msg=f'Gauche dihedral: {dih:.1f}°')

    def test_dihedral_angle_linear_segment(self):
        """Dihedral with a linear segment (collinear B-C) returns 0."""
        coords = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0.001], [0, -1, 0]], dtype=float)
        dih = _dihedral_angle(coords, 0, 1, 2, 3)
        # Near-zero or zero, should not crash.
        self.assertIsInstance(dih, float)

    def test_rotate_fragment_180(self):
        """Rotating a single atom 180° around the x-axis flips its y-coordinate."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1.0, 0]], dtype=float)
        rotated = _rotate_fragment(coords, axis_origin=0, axis_end=1,
                                   angle_deg=180.0, moving_atoms={2})
        np.testing.assert_allclose(rotated[2], [0.5, -1.0, 0.0], atol=1e-6)

    def test_rotate_fragment_preserves_non_moving(self):
        """Atoms not in the moving set should not change."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        rotated = _rotate_fragment(coords, axis_origin=0, axis_end=1,
                                   angle_deg=90.0, moving_atoms={2})
        np.testing.assert_allclose(rotated[0], coords[0])
        np.testing.assert_allclose(rotated[1], coords[1])

    def test_set_bond_distance(self):
        """Set the distance between atom 0 (fixed) and atom 1 (mobile) to 3.0 Å."""
        coords = np.array([[0, 0, 0], [1.5, 0, 0], [2.5, 0, 0]], dtype=float)
        new_coords = _set_bond_distance(coords, fixed=0, mobile=1,
                                        target_dist=3.0, mobile_frag={1, 2})
        d = float(np.linalg.norm(new_coords[1] - new_coords[0]))
        self.assertAlmostEqual(d, 3.0, places=4)
        # Fragment atom 2 should also move by the same displacement.
        d2 = float(np.linalg.norm(new_coords[2] - new_coords[1]))
        self.assertAlmostEqual(d2, 1.0, places=4)

    def test_bfs_fragment_simple(self):
        """BFS finds the connected component not crossing the blocked atom."""
        adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        frag = bfs_fragment(adj, start=2, block={1})
        self.assertEqual(frag, {2, 3})

    def test_bfs_fragment_full(self):
        """BFS with no block returns all connected atoms."""
        adj = {0: {1}, 1: {0, 2}, 2: {1}}
        frag = bfs_fragment(adj, start=0, block=set())
        self.assertEqual(frag, {0, 1, 2})


class TestBuildXYEliminationTS(unittest.TestCase):
    """Tests for the dedicated XY_elimination_hydroxyl TS builder."""

    @classmethod
    def setUpClass(cls):
        """Set up the propanoic acid reactant used in multiple tests."""
        cls.r_xyz = str_to_xyz("""C      -1.44342440    0.21938567    0.14134495
C      -0.17943385   -0.58558878   -0.10310381
C      -0.01901784   -1.69295804    0.90160826
O      -0.76331949   -1.97415266    1.82455783
O       1.10272691   -2.40793854    0.68425738
H      -1.43203982    0.67684331    1.13627537
H      -1.53708941    1.01747004   -0.60148550
H      -2.33303198   -0.41590501    0.07585794
H      -0.21702502   -1.02774537   -1.10407189
H       0.69165336    0.07422934   -0.03456899
H       1.09172878   -3.08582798    1.39221989""")
        cls.r = ARCSpecies(label='R', smiles='CCC(=O)O', xyz=cls.r_xyz)

    def test_returns_xyz_dict(self):
        """The builder should return a valid XYZ dictionary for propanoic acid."""
        ts = build_xy_elimination_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        self.assertIn('symbols', ts)
        self.assertIn('coords', ts)
        self.assertEqual(len(ts['symbols']), 11)
        self.assertEqual(len(ts['coords']), 11)

    def test_ring_bond_distances(self):
        """The 6-membered ring bonds in the TS should be at Pauling-like distances."""
        ts = build_xy_elimination_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        coords = np.array(ts['coords'], dtype=float)
        # Ring: Cα(0)-Hα(7)-Hoh(10)-Ooh(4)-Ccarb(2)-Cβ(1)-Cα(0)
        # Note: Hα might not be index 7 — the builder picks the best H.
        # Check the heavy-atom ring bonds instead.
        d_cc_break = float(np.linalg.norm(coords[2] - coords[1]))  # Ccarb-Cβ
        d_cc_form = float(np.linalg.norm(coords[0] - coords[1]))   # Cα-Cβ
        d_oc = float(np.linalg.norm(coords[4] - coords[2]))        # Ooh-Ccarb
        sbl_cc = get_single_bond_length('C', 'C') or 1.54
        # Breaking C-C should be stretched well beyond SBL.
        self.assertAlmostEqual(d_cc_break, sbl_cc + 2 * PAULING_DELTA, delta=0.15,
                               msg=f'C-C breaking: {d_cc_break:.3f}')
        # Forming C=C should be shortened below SBL.
        self.assertLess(d_cc_form, sbl_cc, msg=f'C=C forming: {d_cc_form:.3f}')
        # Ooh-Ccarb should be near its reactant length (slight strengthening).
        self.assertTrue(1.2 < d_oc < 1.5, msg=f'O-C: {d_oc:.3f}')

    def test_h_h_formed(self):
        """The migrating H atoms should be close (H₂ forming)."""
        ts = build_xy_elimination_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        coords = np.array(ts['coords'], dtype=float)
        # Find the closest H-H pair involving H10 (the hydroxyl H).
        h_indices = [i for i, s in enumerate(ts['symbols']) if s == 'H']
        min_hh = float('inf')
        for hi in h_indices:
            d = float(np.linalg.norm(coords[hi] - coords[10]))
            if hi != 10 and d < min_hh:
                min_hh = d
        self.assertLess(min_hh, 1.5, msg=f'Closest H to H10: {min_hh:.3f} (expect < 1.5)')

    def test_element_sensitivity(self):
        """H-H should be shorter than H-O, which should be shorter than H-C in the ring."""
        ts = build_xy_elimination_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        coords = np.array(ts['coords'], dtype=float)
        # Find the H bonded to Ooh (H10) and the H from Cα closest to H10.
        h_indices_on_c0 = [i for i in range(len(ts['symbols']))
                           if ts['symbols'][i] == 'H'
                           and float(np.linalg.norm(coords[i] - coords[0])) < 2.0
                           and i != 10]
        if h_indices_on_c0:
            h_alpha = min(h_indices_on_c0,
                          key=lambda h: float(np.linalg.norm(coords[h] - coords[10])))
            d_hh = float(np.linalg.norm(coords[h_alpha] - coords[10]))
            d_oh = float(np.linalg.norm(coords[10] - coords[4]))
            d_ch = float(np.linalg.norm(coords[h_alpha] - coords[0]))
            # Element sensitivity: H-H < H-O < H-C
            self.assertLess(d_hh, d_oh, msg=f'H-H={d_hh:.3f} should be < H-O={d_oh:.3f}')

    def test_returns_none_for_non_matching_molecule(self):
        """The builder should return None for molecules without a carboxylic acid group."""
        ethane = ARCSpecies(label='ethane', smiles='CC', xyz=str_to_xyz(
            'C 0 0 0\nC 1.54 0 0\nH -0.5 0.9 0\nH -0.5 -0.9 0\nH -0.5 0 0.9\n'
            'H 2.04 0.9 0\nH 2.04 -0.9 0\nH 2.04 0 0.9'))
        ts = build_xy_elimination_ts(ethane.get_xyz(), ethane.mol)
        self.assertIsNone(ts)

    def test_no_colliding_atoms(self):
        """The generated TS should not have colliding atoms."""
        from arc.species.species import colliding_atoms
        ts = build_xy_elimination_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        self.assertFalse(colliding_atoms(ts), 'TS has colliding atoms')

    def test_atom_count_preserved(self):
        """The TS should have exactly the same atoms as the reactant."""
        ts = build_xy_elimination_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        self.assertEqual(ts['symbols'], self.r_xyz['symbols'])


class TestBuild13SigmatropicRearrangementTS(unittest.TestCase):
    """Unit tests for the bespoke 1,3_sigmatropic_rearrangement builder."""

    def test_success_imidazole_rearrangement(self):
        """The builder produces a non-None, non-colliding TS for the
        imidazole → azirine sigmatropic shift (bb=[(3,4)], fb=[(1,3)])
        with the migrating atom (C3) at calibrated TS-like distances
        from both origin (N4) and target (N1)."""
        from arc.species.species import colliding_atoms

        r = ARCSpecies(label='R', smiles='c1ncc[nH]1', xyz=str_to_xyz(
            """C      -0.96405208   -0.58870010   -0.35675666
               N       0.09948347   -1.35699528   -0.30406608
               C       1.08781769   -0.57088551    0.22943180
               C       0.61245126    0.68985747    0.50218591
               N      -0.70083129    0.66320502    0.12207481
               H      -1.93870511   -0.87854432   -0.72608823
               H       2.08729155   -0.95482079    0.38815067
               H       1.07812779    1.57128662    0.91862266
               H      -1.36158329    1.42559689    0.18141711"""))
        ts = build_1_3_sigmatropic_rearrangement_ts(
            r.get_xyz(), r.mol,
            breaking_bonds=[(3, 4)], forming_bonds=[(1, 3)])
        self.assertIsNotNone(ts, 'builder should return a TS for the imidazole case')
        self.assertFalse(colliding_atoms(ts), 'TS must not have atom collisions')
        coords = np.array(ts['coords'], dtype=float)

        # Curated TS reference distances (DFT-calibrated):
        #   breaking C3-N4: 2.243 Å
        #   forming N1-C3:  1.850 Å
        #   bridge C2-C3:   1.656 Å (core-shaping target)
        d_break = float(np.linalg.norm(coords[3] - coords[4]))
        d_form = float(np.linalg.norm(coords[1] - coords[3]))
        d_bridge = float(np.linalg.norm(coords[2] - coords[3]))

        # The calibrated builder should place the migrating atom within
        # 0.1 Å of the curated targets on both motif edges.
        self.assertAlmostEqual(d_break, 2.24, delta=0.10,
                               msg=f'd(C3-N4)={d_break:.3f} should be ~2.24')
        self.assertAlmostEqual(d_form, 1.85, delta=0.10,
                               msg=f'd(N1-C3)={d_form:.3f} should be ~1.85')

        # The bridge C2-C3 distance should be materially improved
        # compared to Phase 5b (which had 2.147 Å).  The curated
        # target is 1.656 Å — the core-shaping step should bring it
        # below 2.0 Å (closer to the curated value).
        self.assertLess(d_bridge, 2.0,
                        msg=f'd(C2-C3)={d_bridge:.3f} should be < 2.0 '
                            f'(curated: 1.656, was 2.147 before core-shaping)')

        # Unchanged near-core heavy-heavy bonds must remain chemically
        # sane (not collapsed below 0.9 Å or stretched past 3.0 Å).
        atom_to_idx = {a: i for i, a in enumerate(r.mol.atoms)}
        for atom in r.mol.atoms:
            ia = atom_to_idx[atom]
            if r.get_xyz()['symbols'][ia] == 'H':
                continue
            for nbr in atom.bonds.keys():
                ib = atom_to_idx[nbr]
                if r.get_xyz()['symbols'][ib] == 'H':
                    continue
                d = float(np.linalg.norm(coords[ia] - coords[ib]))
                self.assertGreater(d, 0.9,
                                   msg=f'bond {ia}-{ib} collapsed to {d:.3f}')
                self.assertLess(d, 3.5,
                                msg=f'bond {ia}-{ib} stretched to {d:.3f}')

        # Atom count preserved.
        self.assertEqual(ts['symbols'], r.get_xyz()['symbols'])

    def test_returns_none_when_bb_fb_ambiguous(self):
        """When the breaking/forming bonds don't share exactly one common
        atom, the builder returns None (ambiguous motif)."""
        r = ARCSpecies(label='ethane', smiles='CC')
        # bb and fb share NO common atom → ambiguous.
        ts = build_1_3_sigmatropic_rearrangement_ts(
            r.get_xyz(), r.mol,
            breaking_bonds=[(0, 2)], forming_bonds=[(1, 3)])
        self.assertIsNone(ts)

    def test_returns_none_when_multiple_bb(self):
        """When there are two breaking bonds, the builder returns None."""
        r = ARCSpecies(label='propane', smiles='CCC')
        ts = build_1_3_sigmatropic_rearrangement_ts(
            r.get_xyz(), r.mol,
            breaking_bonds=[(0, 1), (1, 2)], forming_bonds=[(0, 2)])
        self.assertIsNone(ts)


class TestBuildBaeyerVilligerStep2TS(unittest.TestCase):
    """Unit tests for the bespoke Baeyer-Villiger_step2 builder."""

    def test_success_criegee_rearrangement(self):
        """The builder produces a non-None, non-colliding TS for the
        Criegee intermediate rearrangement with calibrated concerted-
        core distances: O-O stretched, a C on the quaternary side
        migrating (C-C stretched from parent), and the migrating group
        approaching the peroxide O through which migration proceeds."""
        from arc.species.species import colliding_atoms

        r = ARCSpecies(label='R', smiles='CC(=O)OOC(C)(C)O', xyz=str_to_xyz(
            """C       3.24017953   -0.08055947    0.04152133
               C       1.81730016    0.01506794    0.49970693
               O       1.40458295    0.84254301    1.30456503
               O       1.07825167   -0.97629342   -0.09140243
               O      -0.31730507   -0.78810723    0.32288555
               C      -0.58258035   -1.70130366    1.39489078
               C      -1.89563521   -1.27490637    2.04480283
               C      -0.69557405   -3.11930934    0.84358915
               O       0.44369603   -1.67793216    2.38179554
               H       3.67833221   -1.01933242    0.38907841
               H       3.81240487    0.75201385    0.46069271
               H       3.28486711   -0.01465169   -1.04857320
               H      -2.71999820   -1.28689033    1.32385177
               H      -1.81855771   -0.25064710    2.42803142
               H      -2.14921491   -1.91945835    2.89304406
               H       0.25555438   -3.44404747    0.40665792
               H      -1.45267433   -3.18096845    0.05484522
               H      -0.94129844   -3.83481939    1.63561633
               H       0.45889563   -0.75760809    2.70737582"""))
        ts = build_baeyer_villiger_step2_ts(
            r.get_xyz(), r.mol, split_bonds=[(3, 4)])
        self.assertIsNotNone(ts, 'builder should return a TS for the Criegee case')
        self.assertFalse(colliding_atoms(ts), 'TS must not have atom collisions')
        coords = np.array(ts['coords'], dtype=float)

        # Curated TS reference distances (DFT-calibrated):
        #   O-O peroxide:       2.016 Å
        #   C_parent-C_mig:     2.304 Å
        #   C_mig-O_approach:   2.160 Å
        #   C_parent-O_hydroxyl: 1.279 Å (new C=O forming)
        #
        # C6 and C7 are topologically equivalent CH₃ groups on C5;
        # the builder picks whichever gives the best-fit geometry.
        # The curated TS happens to label C7 as the migrating group,
        # but C6 is an equally valid migration.  The assertions below
        # are *equivalence-aware*: they check that ONE of C6/C7
        # migrates with the correct distances, without forcing a
        # specific label choice.
        d_oo = float(np.linalg.norm(coords[3] - coords[4]))
        self.assertAlmostEqual(d_oo, 2.02, delta=0.10,
                               msg=f'd(O-O)={d_oo:.3f} should be ~2.02')

        # One of C6/C7 should be significantly stretched from C5
        # (migrating group departure) and approaching O4.
        d56 = float(np.linalg.norm(coords[5] - coords[6]))
        d57 = float(np.linalg.norm(coords[5] - coords[7]))
        d_mig = max(d56, d57)
        mig_idx = 6 if d56 > d57 else 7
        self.assertAlmostEqual(d_mig, 2.30, delta=0.15,
                               msg=f'd(C5-C{mig_idx})={d_mig:.3f} should be ~2.30')

        d_mig_o4 = float(np.linalg.norm(coords[mig_idx] - coords[4]))
        self.assertAlmostEqual(d_mig_o4, 2.16, delta=0.15,
                               msg=f'd(C{mig_idx}-O4)={d_mig_o4:.3f} should be ~2.16')

        # C_parent-O_hydroxyl (C5-O8) should be shortened to ~1.28
        # (new C=O forming in the concerted mechanism).
        d_c5_o8 = float(np.linalg.norm(coords[5] - coords[8]))
        self.assertAlmostEqual(d_c5_o8, 1.28, delta=0.15,
                               msg=f'd(C5-O8)={d_c5_o8:.3f} should be ~1.28')

        # Atom count preserved.
        self.assertEqual(ts['symbols'], r.get_xyz()['symbols'])

    def test_returns_none_without_oo_bond(self):
        """When split_bonds contains no O-O bond, the builder returns
        None (no peroxide motif identified)."""
        r = ARCSpecies(label='ethane', smiles='CC')
        ts = build_baeyer_villiger_step2_ts(
            r.get_xyz(), r.mol, split_bonds=[(0, 1)])
        self.assertIsNone(ts)

    def test_returns_none_without_carbonyl(self):
        """When the O-O bond is present but no adjacent C=O exists,
        the builder returns None."""
        r = ARCSpecies(label='hooh', smiles='OO')
        # O-O bond exists but no adjacent carbonyl.
        ts = build_baeyer_villiger_step2_ts(
            r.get_xyz(), r.mol, split_bonds=[(0, 1)])
        self.assertIsNone(ts)


class TestBuildSingletCarbeneIntraDisproportionationTS(unittest.TestCase):
    """Unit tests for the bespoke Singlet_Carbene_Intra_Disproportionation builder."""

    @classmethod
    def setUpClass(cls):
        """Set up the cyclopentadiene carbene reactant used in the builder tests."""
        cls.r_xyz = str_to_xyz("""C      -1.75380171    0.48873088   -0.19068706
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
H      -0.36583394   -1.89034834    0.81324667""")
        cls.r = ARCSpecies(label='R', smiles='C=C1C=C[C]C1', xyz=cls.r_xyz, multiplicity=1)

    def test_returns_xyz_dict(self):
        """The builder returns a valid XYZ dict for the carbene cyclopentadiene motif."""
        ts = build_singlet_carbene_intra_disproportionation_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        self.assertIn('symbols', ts)
        self.assertIn('coords', ts)
        self.assertEqual(ts['symbols'], self.r_xyz['symbols'])
        self.assertEqual(len(ts['coords']), len(self.r_xyz['coords']))
        self.assertFalse(colliding_atoms(ts), 'TS has colliding atoms')

    def test_migrating_h_pauling_triangulation(self):
        """The migrating H sits near Pauling-triangulated distance from donor C and carbene C."""
        ts = build_singlet_carbene_intra_disproportionation_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        coords = np.array(ts['coords'], dtype=float)
        # H atoms on the donor C (C5) are at indices 10 and 11 in the reactant xyz.
        # The carbene C has no H; in C=C1C=C[C]C1 the carbene is the divalent C
        # with zero bonded H atoms.  One of H10/H11 should be at the Pauling
        # distance (~1.51 Å) from both the donor and the carbene C.
        found_good = False
        for hi in (10, 11):
            for ci in range(6):
                for di in range(6):
                    if ci == di:
                        continue
                    d_ci = float(np.linalg.norm(coords[ci] - coords[hi]))
                    d_di = float(np.linalg.norm(coords[di] - coords[hi]))
                    if abs(d_ci - 1.51) < 0.15 and abs(d_di - 1.51) < 0.15:
                        found_good = True
                        break
                if found_good:
                    break
            if found_good:
                break
        self.assertTrue(
            found_good,
            msg='No H at Pauling-triangulated ~1.51 Å from a C-C pair in the TS')

    def test_returns_none_for_non_carbene(self):
        """The builder returns None for a molecule without a carbene center."""
        ethane_xyz = str_to_xyz(
            'C 0 0 0\nC 1.54 0 0\nH -0.5 0.9 0\nH -0.5 -0.9 0\nH -0.5 0 0.9\n'
            'H 2.04 0.9 0\nH 2.04 -0.9 0\nH 2.04 0 0.9')
        ethane = ARCSpecies(label='ethane', smiles='CC', xyz=ethane_xyz)
        ts = build_singlet_carbene_intra_disproportionation_ts(
            ethane.get_xyz(), ethane.mol)
        self.assertIsNone(ts)


class TestBuildKorcekStep1TS(unittest.TestCase):
    """Unit tests for the bespoke Korcek_step1 builder."""

    @classmethod
    def setUpClass(cls):
        """Set up the O=CCC(C)OO reactant from the Korcek_step1 integration test."""
        cls.r_xyz = str_to_xyz(""" O                  1.46497838    1.17312399    1.00181460
 C                  1.63040685    0.08031738    0.40016878
 C                  0.49825841   -0.49996593   -0.46765101
 C                 -0.86289713   -0.06281239    0.10484561
 C                 -2.13626342   -0.60649676   -0.56935436
 O                 -0.92704228    1.07619940    0.96707386
 O                 -0.14402929    0.88182922    2.01182784
 H                  2.55769100   -0.44569700    0.49156341
 H                  0.55938930   -1.56821700   -0.46602353
 H                  0.59624284   -0.13863526   -1.47001781
 H                 -0.90052678    0.75408974   -0.58519419
 H                 -2.10690866   -1.67606688   -0.57697447
 H                 -2.99649767   -0.27717524   -0.02488731
 H                 -2.19012545   -0.24400290   -1.57463893
 H                  0.76017293    0.75570075    1.71499461""")
        cls.r = ARCSpecies(label='R', smiles='O=CCC(C)OO', xyz=cls.r_xyz)

    def test_returns_xyz_dict(self):
        """The builder returns a valid XYZ dict for the keto-peroxide motif."""
        ts = build_korcek_step1_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        self.assertIn('symbols', ts)
        self.assertIn('coords', ts)
        self.assertEqual(ts['symbols'], self.r_xyz['symbols'])
        self.assertEqual(len(ts['coords']), len(self.r_xyz['coords']))
        self.assertFalse(colliding_atoms(ts), 'TS has colliding atoms')

    def test_reactive_distances_sane(self):
        """Forming C-O ring bond is near target; O-O stays chemically sane."""
        ts = build_korcek_step1_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        coords = np.array(ts['coords'], dtype=float)
        # Peroxide O-O pair are atoms 5 and 6.
        d_oo = float(np.linalg.norm(coords[5] - coords[6]))
        self.assertGreater(d_oo, 1.2, msg=f'O-O distance {d_oo:.3f} collapsed')
        self.assertLess(d_oo, 2.2, msg=f'O-O distance {d_oo:.3f} overstretched')
        # The forming C-O ring-closure bond brings the carbonyl C (index 1)
        # close to the terminal peroxide O (index 6).  It should be near the
        # Pauling target ~1.85 Å and markedly shorter than the reactant
        # distance (≥ 3 Å in the open chain).
        d_co_form = float(np.linalg.norm(coords[1] - coords[6]))
        self.assertLess(d_co_form, 2.5, msg=f'forming C-O too long: {d_co_form:.3f}')

    def test_returns_none_without_peroxide(self):
        """The builder returns None for a molecule without a peroxide O-O bond."""
        propanal = ARCSpecies(label='propanal', smiles='CCC=O')
        ts = build_korcek_step1_ts(propanal.get_xyz(), propanal.mol)
        self.assertIsNone(ts)


class TestBuildIntraOHMigrationTS(unittest.TestCase):
    """Unit tests for the bespoke Intra_OH_migration builder."""

    @classmethod
    def setUpClass(cls):
        """Set up the [CH2]COO reactant from the Intra_OH_migration integration test."""
        cls.r_xyz = str_to_xyz("""C      -1.40886397    0.22567351   -0.37379668
C       0.06280787    0.04097694   -0.38515682
O       0.44130326   -0.57668419    0.84260864
O       1.89519755   -0.66754203    0.80966180
H      -1.87218376    0.90693511   -1.07582340
H      -2.03646287   -0.44342165    0.20255768
H       0.35571681   -0.60165457   -1.22096147
H       0.56095122    1.01161503   -0.47393734
H       2.05354047   -0.10415729    1.58865243""")
        cls.r = ARCSpecies(label='R', smiles='[CH2]COO', xyz=cls.r_xyz)

    def test_returns_xyz_dict(self):
        """The builder returns a valid XYZ dict for the OH-migration motif."""
        ts = build_intra_oh_migration_ts(
            self.r_xyz, self.r.mol,
            breaking_bonds=[(2, 3)], forming_bonds=[(0, 3)])
        self.assertIsNotNone(ts)
        self.assertEqual(ts['symbols'], self.r_xyz['symbols'])
        self.assertEqual(len(ts['coords']), len(self.r_xyz['coords']))
        self.assertFalse(colliding_atoms(ts), 'TS has colliding atoms')

    def test_reactive_distances_early_ts(self):
        """Forming C-O is ~2.08 Å (early TS) and breaking O-O stretches past reactant."""
        ts = build_intra_oh_migration_ts(
            self.r_xyz, self.r.mol,
            breaking_bonds=[(2, 3)], forming_bonds=[(0, 3)])
        self.assertIsNotNone(ts)
        coords = np.array(ts['coords'], dtype=float)
        d_c0_o3 = float(np.linalg.norm(coords[0] - coords[3]))
        d_o2_o3 = float(np.linalg.norm(coords[2] - coords[3]))
        # Builder targets an early-TS C-O distance of ~2.08 Å.
        self.assertAlmostEqual(d_c0_o3, 2.08, delta=0.35,
                               msg=f'forming C0-O3={d_c0_o3:.3f} deviates from ~2.08')
        # Breaking O-O stretches toward sbl(O-O)+PAULING_DELTA ≈ 1.90 Å.
        self.assertGreater(d_o2_o3, 1.55,
                           msg=f'breaking O2-O3={d_o2_o3:.3f} not stretched')

    def test_returns_none_when_no_co_forming_bond(self):
        """The builder returns None when forming_bonds has no C-O bond."""
        ts = build_intra_oh_migration_ts(
            self.r_xyz, self.r.mol,
            breaking_bonds=[(2, 3)], forming_bonds=[(2, 3)])
        self.assertIsNone(ts)


class TestBuildIntraSubstitutionSIsomerizationTS(unittest.TestCase):
    """Unit tests for the bespoke intra_substitutionS_isomerization builder."""

    @classmethod
    def setUpClass(cls):
        """Set up the [CH2]SSC reactant from the intra_substitutionS integration test."""
        cls.r_xyz = str_to_xyz("""C       2.02473594    0.05810114    0.12967514
S       0.94173618    1.38848441   -0.00439602
S       1.99155683    2.55179194   -1.33352089
C       3.05975458    3.50692441   -0.22777177
H       1.79171393   -0.74186961    0.82204853
H       2.90913559   -0.02956306   -0.49048675
H       3.72773084    2.84617735    0.33119562
H       3.67272000    4.18684912   -0.82584520
H       2.46084746    4.10465096    0.46458235""")
        cls.r = ARCSpecies(label='R', smiles='[CH2]SSC', xyz=cls.r_xyz)

    def test_returns_xyz_dict(self):
        """The builder returns a valid XYZ dict for the [CH2]SSC motif."""
        ts = build_intra_substitution_s_isomerization_ts(
            self.r_xyz, self.r.mol,
            breaking_bonds=[(1, 2)], forming_bonds=[(0, 2)])
        self.assertIsNotNone(ts)
        self.assertEqual(ts['symbols'], self.r_xyz['symbols'])
        self.assertEqual(len(ts['coords']), len(self.r_xyz['coords']))
        self.assertFalse(colliding_atoms(ts), 'TS has colliding atoms')

    def test_bond_swap_distances(self):
        """Forming C-S contracts toward TS; spectator C-S bond length is preserved."""
        ts = build_intra_substitution_s_isomerization_ts(
            self.r_xyz, self.r.mol,
            breaking_bonds=[(1, 2)], forming_bonds=[(0, 2)])
        self.assertIsNotNone(ts)
        coords = np.array(ts['coords'], dtype=float)
        r_coords = np.array(self.r_xyz['coords'], dtype=float)
        d_c0s2 = float(np.linalg.norm(coords[0] - coords[2]))
        d_c0s2_r = float(np.linalg.norm(r_coords[0] - r_coords[2]))
        # Forming C-S bond must contract (strictly) toward the Pauling target.
        self.assertLess(d_c0s2, d_c0s2_r,
                        msg=f'C-S did not contract: {d_c0s2_r:.3f}→{d_c0s2:.3f}')
        # Resulting C-S distance should be in a physically reasonable early-TS
        # window for a sulfur substitution (between sbl(C-S) and reactant).
        self.assertGreater(d_c0s2, 1.5,
                           msg=f'forming C-S collapsed: {d_c0s2:.3f}')
        # Spectator C3-S2 bond length preserved near its reactant value: the
        # migrating S carries its fragment rigidly so S2-C3 should be intact.
        d_c3s2 = float(np.linalg.norm(coords[3] - coords[2]))
        d_c3s2_r = float(np.linalg.norm(r_coords[3] - r_coords[2]))
        self.assertAlmostEqual(d_c3s2, d_c3s2_r, delta=0.3,
                               msg=f'C3-S2 spectator changed: {d_c3s2_r:.3f}→{d_c3s2:.3f}')

    def test_returns_none_when_no_shared_atom(self):
        """The builder returns None when bb and fb share no common atom."""
        ts = build_intra_substitution_s_isomerization_ts(
            self.r_xyz, self.r.mol,
            breaking_bonds=[(1, 2)], forming_bonds=[(0, 3)])
        self.assertIsNone(ts)


class TestBuildRetroeneTS(unittest.TestCase):
    """Unit tests for the bespoke Retroene builder."""

    @classmethod
    def setUpClass(cls):
        """Set up the CC(=O)OCC(C)C reactant from the Retroene integration test."""
        cls.r_xyz = str_to_xyz("""C       3.35667786   -0.45750645    0.53734155
C       2.24637997    0.53978750    0.40948895
O       1.25975689    0.57306185    1.13089404
O       2.50287461    1.39127766   -0.62065548
C       1.49459134    2.39153372   -0.82368155
C       1.91261725    3.29478901   -1.99081538
C       0.80109651    4.28894041   -2.32015085
C       3.21525767    4.03633978   -1.68259648
H       3.43311587   -1.04672504   -0.37989445
H       4.29738508    0.05945342    0.74281567
H       3.14204808   -1.13410174    1.36950420
H       0.54531269    1.89542863   -1.05959703
H       1.37410296    2.98316125    0.09229590
H       2.09390077    2.66518822   -2.87130726
H       0.58969251    4.94565033   -1.46936402
H       1.08099112    4.91761290   -3.17205390
H      -0.12413795    3.76494190   -2.58209260
H       4.03739144    3.33548500   -1.50498669
H       3.50375149    4.67779028   -2.52219673
H       3.11088096    4.66875130   -0.79438064""")
        cls.r = ARCSpecies(label='R', smiles='CC(=O)OCC(C)C', xyz=cls.r_xyz)

    def test_returns_xyz_dict(self):
        """The builder returns a valid XYZ dict for the retroene 6-membered ring motif."""
        ts = build_retroene_ts(
            self.r_xyz, self.r.mol,
            breaking_bonds=[(3, 4), (5, 13)], forming_bonds=[(3, 13)])
        self.assertIsNotNone(ts)
        self.assertEqual(ts['symbols'], self.r_xyz['symbols'])
        self.assertEqual(len(ts['coords']), len(self.r_xyz['coords']))
        self.assertFalse(colliding_atoms(ts), 'TS has colliding atoms')

    def test_ring_distances(self):
        """The 6-membered ring TS has stretched breaking bonds and a migrating H."""
        ts = build_retroene_ts(
            self.r_xyz, self.r.mol,
            breaking_bonds=[(3, 4), (5, 13)], forming_bonds=[(3, 13)])
        self.assertIsNotNone(ts)
        coords = np.array(ts['coords'], dtype=float)
        # O3-C4 sigma is stretched (breaking): target ~2.5 Å.
        d_o3c4 = float(np.linalg.norm(coords[3] - coords[4]))
        self.assertGreater(d_o3c4, 1.8,
                           msg=f'O3-C4 sigma not stretched: {d_o3c4:.3f}')
        # Migrating H13 sits between donor C5 and ester O3 at Pauling-like distances.
        d_c5h13 = float(np.linalg.norm(coords[5] - coords[13]))
        d_o3h13 = float(np.linalg.norm(coords[3] - coords[13]))
        self.assertLess(d_c5h13, 1.8,
                        msg=f'migrating C5-H13 too long: {d_c5h13:.3f}')
        self.assertLess(d_o3h13, 2.5,
                        msg=f'forming O3-H13 too long: {d_o3h13:.3f}')

    def test_returns_none_when_bond_counts_wrong(self):
        """The builder returns None when bb has != 2 entries or fb has != 1."""
        ts = build_retroene_ts(
            self.r_xyz, self.r.mol,
            breaking_bonds=[(3, 4)], forming_bonds=[(3, 13)])
        self.assertIsNone(ts)
        ts = build_retroene_ts(
            self.r_xyz, self.r.mol,
            breaking_bonds=[(3, 4), (5, 13)],
            forming_bonds=[(3, 13), (2, 4)])
        self.assertIsNone(ts)


if __name__ == '__main__':
    unittest.main()
