#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for the arc.job.adapters.ts.xy_addition module (XY_Addition_MultipleBond TS seed builder).
"""

import math
import unittest
from unittest.mock import patch

import numpy as np

from arc.common import get_single_bond_length
from arc.job.adapters.ts.xy_addition import (BREAKING_XY_FACTOR,
                                             FORMING_X_FACTOR,
                                             FORMING_Y_FACTOR,
                                             LOCAL_PLANE_RADIUS,
                                             _build_4_center_geometry,
                                             _rotation_matrix_between,
                                             _solve_ring_positions,
                                             xy_addition,
                                             )
from arc.job.adapters.ts.seed_hub import get_ts_seeds, get_wrapper_constraints
from arc.reaction import ARCReaction
from arc.species import ARCSpecies


ETHYLENE_XYZ = ('C 0 0 0.667\nC 0 0 -0.667\nH 0 0.921 1.232\n'
                'H 0 -0.921 1.232\nH 0 0.921 -1.232\nH 0 -0.921 -1.232')
HCL_XYZ = 'Cl 0 0 0.071\nH 0 0 -1.211'
CHLOROETHANE_XYZ = ('C 1.61 -0.36 0\nC 0.49 0.66 0\nCl -1.14 -0.15 0\nH 1.57 -0.99 -0.88\n'
                    'H 2.57 0.16 0\nH 0.53 1.30 0.88\nH 0.53 1.30 -0.88\nH -0.34 -0.5 0')
HCL_COORDS = np.array([(0.0, 0.0, 0.071), (0.0, 0.0, -1.211)], dtype=float)
HCL_SYMBOLS = ('Cl', 'H')


def distance(coords, i: int, j: int) -> float:
    """
    Return the distance between two atoms of a coordinates sequence.

    Args:
        coords: A sequence of Cartesian coordinate triples.
        i (int): The index of the first atom.
        j (int): The index of the second atom.

    Returns:
        float: The interatomic distance.
    """
    return math.sqrt(sum((coords[i][k] - coords[j][k]) ** 2 for k in range(3)))


class TestXYAdditionSeed(unittest.TestCase):
    """Tests for the XY_Addition_MultipleBond 4-center TS seed builder."""

    @classmethod
    def setUpClass(cls):
        """Set up the shared C2H4 + HCl <=> CH3CH2Cl fixture."""
        cls.maxDiff = None
        cls.label_maps = [
            {'*2': 0, '*1': 1, '*3': 7, '*4': 6},
            {'*2': 1, '*1': 0, '*3': 7, '*4': 6},
        ]

    def make_ethylene_hcl_reaction(self) -> ARCReaction:
        """
        Build the C2H4 + HCl <=> CH3CH2Cl reaction with the family labels RMG assigns to it.

        The ``r_label_map`` values match the ones ARC's family module generates for this
        reaction: the multiple-bond carbons are ``*1``/``*2`` (in both orderings), the hydrogen
        of HCl is ``*3`` and its chlorine is ``*4``.

        Returns:
            ARCReaction: The reaction, with ``product_dicts`` and ``family`` populated.
        """
        ethylene = ARCSpecies(label='R1', smiles='C=C', multiplicity=1, xyz=ETHYLENE_XYZ)
        hcl = ARCSpecies(label='R2', smiles='Cl', multiplicity=1, xyz=HCL_XYZ)
        chloroethane = ARCSpecies(label='P1', smiles='CCCl', multiplicity=1, xyz=CHLOROETHANE_XYZ)
        rxn = ARCReaction(r_species=[ethylene, hcl], p_species=[chloroethane])
        rxn.product_dicts = [{'r_label_map': label_map} for label_map in self.label_maps]
        rxn.family = 'XY_Addition_MultipleBond'
        return rxn

    def test_xy_addition_seed_is_four_center(self):
        """The seed for C=C + HCl -> CH3CH2Cl must be a 4-center arrangement:
        X (H) and Y (Cl) both approaching the (former) double bond, with the H-Cl bond STRETCHED.

        A rigid, unstretched H-Cl makes the seed a van der Waals complex rather than a saddle point:
        bond order is created at the two carbons without any being released at H-Cl, so the
        imaginary mode is a hindered rotation of HCl over the pi cloud instead of the reaction
        coordinate. The breaking bond must therefore be well beyond its equilibrium length."""
        rxn = self.make_ethylene_hcl_reaction()
        seeds = xy_addition(reaction=rxn)
        self.assertEqual(len(seeds), 2)
        self.assertEqual(seeds[0]['method'], 'Heuristics-XY')
        self.assertEqual(
            [seed['metadata']['reactive_atoms'] for seed in seeds],
            self.label_maps,
        )
        hub_seeds = get_ts_seeds(reaction=rxn)
        self.assertEqual([seed['metadata']['reactive_atoms'] for seed in hub_seeds], self.label_maps)
        self.assertEqual(
            [get_wrapper_constraints('crest', reaction=rxn, seed=seed) for seed in hub_seeds],
            [
                {
                    'atoms': tuple(label_map[label] for label in ('*1', '*2', '*3', '*4')),
                    'distance_pairs': (
                        (label_map['*1'], label_map['*3']),
                        (label_map['*2'], label_map['*4']),
                        (label_map['*3'], label_map['*4']),
                    ),
                }
                for label_map in self.label_maps
            ],
        )
        xyz = seeds[0]['xyz']
        self.assertEqual(len(xyz['symbols']), 8)

        coords = [tuple(c) for c in xyz['coords']]

        def dist(i, j):
            return distance(coords, i, j)

        carbons = [i for i, s in enumerate(xyz['symbols']) if s == 'C']
        cl = [i for i, s in enumerate(xyz['symbols']) if s == 'Cl'][0]
        hydrogens = [i for i, s in enumerate(xyz['symbols']) if s == 'H']
        transferring_h = min(hydrogens, key=lambda h: dist(h, cl))

        min_cl_c = min(dist(cl, c) for c in carbons)
        min_h_c = min(dist(transferring_h, c) for c in carbons)
        # 4-center TS region: both Cl and the transferring H engage the carbons (forming bonds),
        # while the H-Cl bond is still present (breaking). Contrast with an H-transfer saddle where
        # Cl would be a spectator (> 2.9 A from every carbon).
        # A reference concerted C2H4 + HCl four-centre TS has r(C-Cl) ~ 2.5-2.8 A: the forming bond
        # is far from complete, but the Cl is not a spectator either.
        self.assertLess(min_cl_c, 2.9, 'Cl should be forming a bond to a carbon (not a spectator)')
        self.assertGreater(min_cl_c, 2.0, 'the C-Cl bond must not be near-formed in the seed')
        self.assertLess(min_h_c, 1.9, 'the transferring H should be forming a bond to a carbon')
        h_cl = dist(transferring_h, cl)
        self.assertLess(h_cl, 2.2, 'the breaking H-Cl bond should still be present')
        # 1.275 A is the equilibrium H-Cl length; an unstretched bond means a reactant complex.
        self.assertGreater(h_cl, 1.40, 'the breaking H-Cl bond must be stretched well past equilibrium')

    def test_seed_orientation_follows_the_label_map(self):
        """*3 must approach *1 and *4 must approach *2, at the scaled seed distances.

        Resolving the atoms through the label map (rather than by element) is what makes this
        directional: a seed built with *3 and *4 - or *1 and *2 - interchanged contradicts the
        CREST constraint spec that ``get_wrapper_constraints`` derives from the same label map.
        """
        rxn = self.make_ethylene_hcl_reaction()
        seeds = xy_addition(reaction=rxn)
        self.assertEqual(len(seeds), 2)
        expected_d_13 = 1.31 * get_single_bond_length('C', 'H')
        expected_d_24 = 1.50 * get_single_bond_length('C', 'Cl')
        expected_d_34 = 1.42 * get_single_bond_length('H', 'Cl')
        for seed, label_map in zip(seeds, self.label_maps):
            with self.subTest(label_map=label_map):
                coords = [tuple(c) for c in seed['xyz']['coords']]
                i1, i2 = label_map['*1'], label_map['*2']
                i3, i4 = label_map['*3'], label_map['*4']
                self.assertEqual(seed['xyz']['symbols'][i3], 'H')
                self.assertEqual(seed['xyz']['symbols'][i4], 'Cl')
                self.assertLess(distance(coords, i1, i3), distance(coords, i2, i3),
                                'X (*3) must approach *1, not *2')
                self.assertLess(distance(coords, i2, i4), distance(coords, i1, i4),
                                'Y (*4) must approach *2, not *1')
                self.assertAlmostEqual(distance(coords, i1, i3), expected_d_13, delta=0.05)
                self.assertAlmostEqual(distance(coords, i2, i4), expected_d_24, delta=0.05)
                self.assertAlmostEqual(distance(coords, i3, i4), expected_d_34, delta=0.05)

    def test_seed_scaling_factors(self):
        """The seed scaling factors are pinned; the geometry tests are written against these values."""
        self.assertAlmostEqual(FORMING_X_FACTOR, 1.31, delta=1e-9)
        self.assertAlmostEqual(FORMING_Y_FACTOR, 1.50, delta=1e-9)
        self.assertAlmostEqual(BREAKING_XY_FACTOR, 1.42, delta=1e-9)
        self.assertAlmostEqual(LOCAL_PLANE_RADIUS, 2.6, delta=1e-9)

    def test_approach_is_perpendicular_to_the_local_plane(self):
        """X and Y must approach the pi face, i.e. along the normal of the local plane.

        The fixture is an ethylene rotated by 45 degrees about its C=C axis, so the plane normal
        is not aligned with any Cartesian axis and a fallback direction cannot pass by accident.
        """
        cos45 = math.sqrt(0.5)
        mb_coords = np.array([(0.0, 0.0, 0.667),
                              (0.0, 0.0, -0.667),
                              (-0.921 * cos45, 0.921 * cos45, 1.232),
                              (0.921 * cos45, -0.921 * cos45, 1.232),
                              (-0.921 * cos45, 0.921 * cos45, -1.232),
                              (0.921 * cos45, -0.921 * cos45, -1.232)], dtype=float)
        placed = _build_4_center_geometry(mb_coords, 0, 1, HCL_COORDS, 1, 0,
                                          mb_symbols=('C', 'C', 'H', 'H', 'H', 'H'),
                                          xy_symbols=HCL_SYMBOLS)
        self.assertIsNotNone(placed)
        approach = placed[1] - mb_coords[0]
        approach = approach / np.linalg.norm(approach)
        plane_normal = np.array([cos45, cos45, 0.0])
        self.assertGreater(abs(float(np.dot(approach, plane_normal))), 0.95,
                           'the approach direction must be the local plane normal')

    def test_approach_is_orthogonal_to_the_bond_axis(self):
        """The *1...*3 approach vector must be perpendicular to the multiple-bond axis.

        The fixture is a synthetic non-planar fragment whose least-variance direction coincides
        with the multiple-bond axis, so a normal that is not orthogonalized against that axis
        would place X along the bond instead of over it.
        """
        mb_coords = np.array([(0.0, 0.0, 0.667),
                              (0.0, 0.0, -0.667),
                              (1.5, 0.0, 0.0),
                              (-1.5, 0.0, 0.0),
                              (0.0, 1.5, 0.0),
                              (0.0, -1.5, 0.0)], dtype=float)
        placed = _build_4_center_geometry(mb_coords, 0, 1, HCL_COORDS, 1, 0,
                                          mb_symbols=('C', 'C', 'H', 'H', 'H', 'H'),
                                          xy_symbols=HCL_SYMBOLS)
        self.assertIsNotNone(placed)
        bond_axis = mb_coords[1] - mb_coords[0]
        bond_axis = bond_axis / np.linalg.norm(bond_axis)
        self.assertAlmostEqual(float(np.dot(placed[1] - mb_coords[0], bond_axis)), 0.0, delta=0.05)

    def test_the_less_hindered_face_is_selected(self):
        """The X-Y fragment must approach the face that is not blocked by a substituent.

        The same fragment is placed twice with the blocking atom moved from one face to the
        other; X must land on the opposite side each time.
        """
        placed_x = list()
        for blocker_sign in (1.0, -1.0):
            mb_coords = np.array([(0.0, 0.0, 0.667),
                                  (0.0, 0.0, -0.667),
                                  (0.0, 1.6, 1.6),
                                  (0.0, -1.6, -1.6),
                                  (blocker_sign * 1.4, 0.0, 2.6)], dtype=float)
            placed = _build_4_center_geometry(mb_coords, 0, 1, HCL_COORDS, 1, 0,
                                              mb_symbols=('C', 'C', 'H', 'H', 'H'),
                                              xy_symbols=HCL_SYMBOLS)
            self.assertIsNotNone(placed)
            placed_x.append(float(placed[1][0]))
        self.assertLess(placed_x[0], 0.0, 'X must avoid a blocker on the +x face')
        self.assertGreater(placed_x[1], 0.0, 'X must avoid a blocker on the -x face')

    def test_rotation_matrix_between_is_a_proper_rotation(self):
        """``_rotation_matrix_between`` must never return an improper (mirroring) transform.

        The antiparallel case is the one at risk: -I maps the vector correctly but has a
        determinant of -1, which inverts the chirality of the fragment being placed.
        """
        antiparallel = _rotation_matrix_between(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]))
        self.assertAlmostEqual(float(np.linalg.det(antiparallel)), 1.0, delta=1e-8)
        np.testing.assert_allclose(antiparallel @ np.array([0.0, 0.0, 1.0]),
                                   np.array([0.0, 0.0, -1.0]), atol=1e-8)
        np.testing.assert_allclose(antiparallel.T @ antiparallel, np.eye(3), atol=1e-8)
        antiparallel_x = _rotation_matrix_between(np.array([1.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(np.linalg.det(antiparallel_x)), 1.0, delta=1e-8)
        np.testing.assert_allclose(antiparallel_x @ np.array([1.0, 0.0, 0.0]),
                                   np.array([-1.0, 0.0, 0.0]), atol=1e-8)
        parallel = _rotation_matrix_between(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 3.0]))
        np.testing.assert_allclose(parallel, np.eye(3), atol=1e-8)
        general = _rotation_matrix_between(np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0]))
        self.assertAlmostEqual(float(np.linalg.det(general)), 1.0, delta=1e-8)
        np.testing.assert_allclose(general @ np.array([1.0, 0.0, 0.0]),
                                   np.array([0.0, 1.0, 0.0]), atol=1e-8)

    def test_no_seeds_when_all_labels_are_on_one_reactant(self):
        """A unimolecular arrangement of the four labels yields no seed."""
        rxn = self.make_ethylene_hcl_reaction()
        rxn.product_dicts = [{'r_label_map': {'*1': 0, '*2': 1, '*3': 2, '*4': 3}}]
        self.assertEqual(xy_addition(reaction=rxn), list())

    def test_no_seeds_when_labels_are_incomplete(self):
        """A product dict missing one of the four family labels yields no seed."""
        rxn = self.make_ethylene_hcl_reaction()
        rxn.product_dicts = [{'r_label_map': {'*1': 1, '*2': 0, '*3': 7}}]
        self.assertEqual(xy_addition(reaction=rxn), list())

    def test_no_seeds_for_more_than_two_reactants(self):
        """Only bimolecular reactions are assembled.

        The reassembly loop resolves every atom that is not on the multiple-bond reactant from
        the X-Y fragment, so a third reactant would take its symbols and coordinates from the
        wrong fragment.
        """
        ethylene = ARCSpecies(label='R1', smiles='C=C', multiplicity=1, xyz=ETHYLENE_XYZ)
        hcl = ARCSpecies(label='R2', smiles='Cl', multiplicity=1, xyz=HCL_XYZ)
        water = ARCSpecies(label='R3', smiles='O', multiplicity=1,
                           xyz='O 0 0 0.119\nH 0 0.763 -0.477\nH 0 -0.763 -0.477')
        chloroethane = ARCSpecies(label='P1', smiles='CCCl', multiplicity=1, xyz=CHLOROETHANE_XYZ)
        water_p = ARCSpecies(label='P2', smiles='O', multiplicity=1,
                             xyz='O 0 0 0.119\nH 0 0.763 -0.477\nH 0 -0.763 -0.477')
        rxn = ARCReaction(r_species=[ethylene, hcl, water], p_species=[chloroethane, water_p])
        rxn.product_dicts = [{'r_label_map': label_map} for label_map in self.label_maps]
        rxn.family = 'XY_Addition_MultipleBond'
        self.assertEqual(xy_addition(reaction=rxn), list())

    def test_colliding_seeds_are_discarded(self):
        """A rigidly placed X-Y fragment that lands on another atom is not returned."""
        rxn = self.make_ethylene_hcl_reaction()
        with patch('arc.job.adapters.ts.xy_addition.colliding_atoms', return_value=True):
            self.assertEqual(xy_addition(reaction=rxn), list())
        self.assertEqual(len(xy_addition(reaction=rxn)), 2)

    def test_solve_ring_positions_returns_none_when_infeasible(self):
        """Distances that cannot close the four-membered ring give ``None``, not NaN coordinates."""
        p1, p2 = np.zeros(3), np.array([1.33, 0.0, 0.0])
        bond_axis, normal = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        for d_13, d_24, d_34 in ((1.43, 2.65, 0.2), (1.43, 0.2, 0.2), (1.43, 12.0, 1.8)):
            with self.subTest(d_13=d_13, d_24=d_24, d_34=d_34):
                self.assertIsNone(_solve_ring_positions(p1=p1, p2=p2, bond_axis=bond_axis, normal=normal,
                                                        d_13=d_13, d_24=d_24, d_34=d_34))
        feasible = _solve_ring_positions(p1=p1, p2=p2, bond_axis=bond_axis, normal=normal,
                                         d_13=1.43, d_24=2.65, d_34=1.8)
        self.assertIsNotNone(feasible)
        for position in feasible:
            self.assertTrue(np.all(np.isfinite(position)))

    def test_build_4_center_geometry_returns_none_for_degenerate_bonds(self):
        """Coincident multiple-bond atoms or coincident X-Y atoms give ``None``."""
        degenerate_mb = np.array([(0.0, 0.0, 0.0),
                                  (0.0, 0.0, 0.0),
                                  (0.0, 0.921, 1.232),
                                  (0.0, -0.921, 1.232)], dtype=float)
        self.assertIsNone(_build_4_center_geometry(degenerate_mb, 0, 1, HCL_COORDS, 1, 0,
                                                   mb_symbols=('C', 'C', 'H', 'H'),
                                                   xy_symbols=HCL_SYMBOLS))
        mb_coords = np.array([(0.0, 0.0, 0.667),
                              (0.0, 0.0, -0.667),
                              (0.0, 0.921, 1.232),
                              (0.0, -0.921, 1.232),
                              (0.0, 0.921, -1.232),
                              (0.0, -0.921, -1.232)], dtype=float)
        degenerate_xy = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], dtype=float)
        self.assertIsNone(_build_4_center_geometry(mb_coords, 0, 1, degenerate_xy, 1, 0,
                                                   mb_symbols=('C', 'C', 'H', 'H', 'H', 'H'),
                                                   xy_symbols=HCL_SYMBOLS))

    def test_build_4_center_geometry_returns_none_when_the_ring_cannot_close(self):
        """An X-Y fragment that cannot reach both multiple-bond atoms gives ``None``."""
        mb_coords = np.array([(0.0, 0.0, 6.0),
                              (0.0, 0.0, -6.0),
                              (0.0, 0.921, 7.0),
                              (0.0, -0.921, 7.0),
                              (0.0, 0.921, -7.0),
                              (0.0, -0.921, -7.0)], dtype=float)
        self.assertIsNone(_build_4_center_geometry(mb_coords, 0, 1, HCL_COORDS, 1, 0,
                                                   mb_symbols=('C', 'C', 'H', 'H', 'H', 'H'),
                                                   xy_symbols=HCL_SYMBOLS))


if __name__ == '__main__':
    unittest.main()
