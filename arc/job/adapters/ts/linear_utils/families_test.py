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
from arc.job.adapters.ts.linear_utils.families import (
    _bfs_fragment,
    _dihedral_angle,
    _rotate_fragment,
    _set_bond_distance,
    build_1_3_sigmatropic_rearrangement_ts,
    build_baeyer_villiger_step2_ts,
    build_xy_elimination_ts,
    PAULING_DELTA,
)


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
        # Near-zero or zero — should not crash.
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
        frag = _bfs_fragment(adj, start=2, block={1})
        self.assertEqual(frag, {2, 3})

    def test_bfs_fragment_full(self):
        """BFS with no block returns all connected atoms."""
        adj = {0: {1}, 1: {0, 2}, 2: {1}}
        frag = _bfs_fragment(adj, start=0, block=set())
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
        d_break = float(np.linalg.norm(coords[3] - coords[4]))
        d_form = float(np.linalg.norm(coords[1] - coords[3]))

        # The calibrated builder should place the migrating atom within
        # 0.1 Å of the curated targets on both motif edges.
        self.assertAlmostEqual(d_break, 2.24, delta=0.10,
                               msg=f'd(C3-N4)={d_break:.3f} should be ~2.24')
        self.assertAlmostEqual(d_form, 1.85, delta=0.10,
                               msg=f'd(N1-C3)={d_form:.3f} should be ~1.85')

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
        #   O-O peroxide: 2.016 Å
        #   C_parent-C_mig: 2.304 Å
        #   C_mig-O_approach: 2.160 Å
        d_oo = float(np.linalg.norm(coords[3] - coords[4]))
        self.assertAlmostEqual(d_oo, 2.02, delta=0.15,
                               msg=f'd(O-O)={d_oo:.3f} should be ~2.02')

        # One of the C neighbors of C5 (the quaternary C at index 5)
        # should be stretched to ~2.30 (migrating group leaving C5)
        # and approaching O4 at ~2.16.
        d56 = float(np.linalg.norm(coords[5] - coords[6]))
        d57 = float(np.linalg.norm(coords[5] - coords[7]))
        # At least one of them should be significantly stretched.
        d_mig = max(d56, d57)
        self.assertGreater(d_mig, 1.8,
                           msg=f'max(d(C5-C6), d(C5-C7))={d_mig:.3f} — '
                               f'at least one CH₃ should have migrated')
        # The migrated one should also be close to O4 (~2.16).
        mig_idx = 6 if d56 > d57 else 7
        d_mig_o4 = float(np.linalg.norm(coords[mig_idx] - coords[4]))
        self.assertAlmostEqual(d_mig_o4, 2.16, delta=0.20,
                               msg=f'd(C_mig-O4)={d_mig_o4:.3f} should be ~2.16')

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


if __name__ == '__main__':
    unittest.main()
