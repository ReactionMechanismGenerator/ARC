#!/usr/bin/env python3
# encoding: utf-8

"""Unit tests for local reactive-center geometry helpers."""

import unittest

import numpy as np

from arc.common import get_single_bond_length
from arc.species import ARCSpecies

from arc.job.adapters.ts.linear_utils.local_geometry import (
    _h_neighbors,
    _heavy_neighbors,
    _xyz_with_coords,
    apply_reactive_center_cleanup,
    clean_migrating_h,
    identify_h_migration_pairs,
    is_internal_reactive_ch2_misoriented,
    is_terminal_group_asymmetric,
    orient_h_away_from_axis,
    regularize_terminal_h_geometry,
    repair_internal_reactive_ch2,
    restore_terminal_h_symmetry,
)
from arc.job.adapters.ts.linear_utils.postprocess import PAULING_DELTA


class TestRegularizeTerminalHGeometry(unittest.TestCase):
    """``regularize_terminal_h_geometry`` snaps distorted terminal H
    bond lengths back to ``get_single_bond_length(C, H)`` while
    preserving direction."""

    def setUp(self):
        # Methane CH4 — center is C[0], 4 H atoms.
        self.sp = ARCSpecies(label='CH4', smiles='C')
        self.xyz = self.sp.get_xyz()
        self.mol = self.sp.mol
        # Identify the C atom and the H atoms.
        self.c_idx = next(i for i, s in enumerate(self.xyz['symbols']) if s == 'C')
        self.h_idxs = [i for i, s in enumerate(self.xyz['symbols']) if s == 'H']

    def test_no_op_when_within_range(self):
        """A fresh CH4 should not be touched."""
        out = regularize_terminal_h_geometry(self.xyz, self.mol, self.c_idx)
        self.assertIs(out, self.xyz)

    def test_pulls_overstretched_h_back(self):
        """An H atom stretched far past sbl is snapped back along its bond."""
        coords = list(map(list, self.xyz['coords']))
        h = self.h_idxs[0]
        c_pos = np.array(coords[self.c_idx])
        h_pos = np.array(coords[h])
        # Stretch this H to 2.0 Å along the same direction.
        direction = (h_pos - c_pos) / float(np.linalg.norm(h_pos - c_pos))
        coords[h] = (c_pos + direction * 2.0).tolist()
        bad = {**self.xyz, 'coords': tuple(tuple(row) for row in coords)}
        out = regularize_terminal_h_geometry(bad, self.mol, self.c_idx)
        sbl = get_single_bond_length('C', 'H')
        new_d = float(np.linalg.norm(
            np.array(out['coords'][h]) - np.array(out['coords'][self.c_idx])))
        self.assertAlmostEqual(new_d, sbl, places=4)

    def test_exclude_atoms_skips_listed_hydrogen(self):
        coords = list(map(list, self.xyz['coords']))
        h = self.h_idxs[0]
        c_pos = np.array(coords[self.c_idx])
        h_pos = np.array(coords[h])
        direction = (h_pos - c_pos) / float(np.linalg.norm(h_pos - c_pos))
        coords[h] = (c_pos + direction * 2.5).tolist()
        bad = {**self.xyz, 'coords': tuple(tuple(row) for row in coords)}
        out = regularize_terminal_h_geometry(
            bad, self.mol, self.c_idx, exclude_atoms={h})
        # Excluded H is not touched — its distance is still 2.5.
        self.assertAlmostEqual(
            float(np.linalg.norm(
                np.array(out['coords'][h]) - np.array(out['coords'][self.c_idx]))),
            2.5, places=4)


class TestOrientHAwayFromAxis(unittest.TestCase):
    """The blocking-H reorientation helper reflects an inward H to the
    opposite side of the donor–parent–acceptor axis when it sits in the
    blocking pocket."""

    def test_inward_h_is_reflected(self):
        """An H bonded to donor and sitting in the inward blocking
        pocket (small angle to the donor–acceptor axis AND close to the
        acceptor) is reflected across that axis to the outward side."""
        # Build a hand-crafted donor–acceptor pair with one H sitting
        # IN the blocking pocket: short distance to acceptor (< 1.60 Å)
        # AND small angle to the donor–acceptor axis (< 85°).
        xyz = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0),     # donor
                (2.5, 0.0, 0.0),     # acceptor — closer than before
                (1.4, 0.10, 0.0),    # inward H: ~1.10 Å from acceptor
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                self.bonds = bonds
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
        c0 = _A('C', {})
        c1 = _A('C', {})
        h2 = _A('H', {})
        c0.bonds = {h2: None}
        h2.bonds = {c0: None}
        c1.bonds = {}
        class _M:
            atoms = [c0, c1, h2]

        out = orient_h_away_from_axis(xyz, _M(), donor=0, acceptor=1, exclude_h=set())
        h_after = np.array(out['coords'][2])
        # Reflected x component flips sign of the axis-parallel piece
        # (1.4 → -1.4 after subtracting 2 × 1.4 × axis_hat from h_vec).
        self.assertLess(h_after[0], 0.0)
        # Heavy atoms unchanged.
        self.assertEqual(out['coords'][0], xyz['coords'][0])
        self.assertEqual(out['coords'][1], xyz['coords'][1])

    def test_excluded_h_is_left_alone(self):
        xyz = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.5, 0.05, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                self.bonds = bonds
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
        c0 = _A('C', {})
        c1 = _A('C', {})
        h2 = _A('H', {})
        c0.bonds = {h2: None}
        h2.bonds = {c0: None}
        class _M:
            atoms = [c0, c1, h2]
        out = orient_h_away_from_axis(
            xyz, _M(), donor=0, acceptor=1, exclude_h={2})
        # The migrating H (excluded) is untouched, so the result should be
        # identical to the input.
        self.assertEqual(out['coords'], xyz['coords'])

    def test_outward_h_is_left_alone(self):
        # H sitting OUTSIDE the blocking pocket — should not be moved.
        xyz = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (-0.7, 0.7, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                self.bonds = bonds
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
        c0 = _A('C', {})
        c1 = _A('C', {})
        h2 = _A('H', {})
        c0.bonds = {h2: None}
        h2.bonds = {c0: None}
        class _M:
            atoms = [c0, c1, h2]
        out = orient_h_away_from_axis(
            xyz, _M(), donor=0, acceptor=1, exclude_h=set())
        self.assertEqual(out['coords'], xyz['coords'])


class TestCleanMigratingH(unittest.TestCase):
    """``clean_migrating_h`` re-places the migrating H at the donor–
    acceptor triangulated TS position and is idempotent."""

    def test_idempotent_on_triangulated_position(self):
        # Two heavy atoms 2.5 Å apart, an H roughly between them.
        xyz = {
            'symbols': ('C', 'N', 'H'),
            'isotopes': (12, 14, 1),
            'coords': (
                (0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (1.0, 0.5, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = {}
        c = _A('C'); n = _A('N'); h = _A('H')
        class _M:
            atoms = [c, n, h]
        out = clean_migrating_h(xyz, _M(), donor=0, acceptor=1, h_idx=2)
        h_pos_1 = np.array(out['coords'][2])
        # Run again — should be a no-op (idempotent).
        out2 = clean_migrating_h(out, _M(), donor=0, acceptor=1, h_idx=2)
        h_pos_2 = np.array(out2['coords'][2])
        self.assertTrue(np.allclose(h_pos_1, h_pos_2, atol=1e-10))

    def test_h_distance_to_donor_matches_target(self):
        xyz = {
            'symbols': ('C', 'N', 'H'),
            'isotopes': (12, 14, 1),
            'coords': (
                (0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (1.0, 0.5, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = {}
        c = _A('C'); n = _A('N'); h = _A('H')
        class _M:
            atoms = [c, n, h]
        out = clean_migrating_h(xyz, _M(), donor=0, acceptor=1, h_idx=2)
        h_pos = np.array(out['coords'][2])
        d_dh = float(np.linalg.norm(h_pos - np.array(out['coords'][0])))
        sbl_ch = get_single_bond_length('C', 'H')
        # The triangulation places H at distance d_DH = sbl + PAULING_DELTA from donor.
        self.assertAlmostEqual(d_dh, sbl_ch + PAULING_DELTA, places=4)


class TestRestoreTerminalHSymmetry(unittest.TestCase):
    """``restore_terminal_h_symmetry`` re-seats H atoms around the
    (parent → center) axis at evenly spaced azimuth.  
    rebuild: the helper now preserves each H's *original* center–H
    distance individually instead of averaging across the group, so
    that umbrella-inversion repairs no longer trip the legacy
    bond-length bail-out.
    """

    def test_ch3_distorted_h_preserves_per_h_bond_lengths(self):
        """each H's center–H distance is preserved exactly,
        even when one H is displaced and the others are not."""
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_neighbors = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        # Distort one H by ~0.02 Å in the y direction.  Under the
        # legacy averaging implementation this would have triggered the
        # 0.05 Å bail-out's marginal regime; under , each H's
        # center–C distance is preserved exactly.
        coords = list(map(list, xyz['coords']))
        coords[h_neighbors[0]][1] += 0.02
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        # Capture the per-H bond lengths BEFORE symmetrization.
        c_pos_before = np.asarray(bad['coords'])[c0]
        original_per_h_dists = {
            h: float(np.linalg.norm(np.asarray(bad['coords'])[h] - c_pos_before))
            for h in h_neighbors
        }
        out = restore_terminal_h_symmetry(bad, sp.mol, c0)
        new_coords = np.asarray(out['coords'])
        c_pos = new_coords[c0]
        # Each individual H should still be at exactly its original
        # distance from c0 ( per-H bond-length preservation).
        for h, original_d in original_per_h_dists.items():
            new_d = float(np.linalg.norm(new_coords[h] - c_pos))
            self.assertAlmostEqual(
                new_d, original_d, places=6,
                msg=f'H{h} bond length not preserved '
                    f'(was {original_d:.6f}, now {new_d:.6f})')

    def test_ch3_inverted_h_is_repaired(self):
        """a true umbrella-inversion case (one H angle to
        the outward axis > 100°) must be repaired (the inverted H is
        moved onto the outward cone) without bailing out, AND each
        H's bond length must be preserved exactly."""
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        c1 = next(i for i, s in enumerate(symbols) if s == 'C' and i != c0)
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_neighbors = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        coords = list(map(list, xyz['coords']))
        c0_pos = np.array(coords[c0])
        c1_pos = np.array(coords[c1])
        outward = c0_pos - c1_pos
        outward /= float(np.linalg.norm(outward))
        # Reflect one H of c0 through the perpendicular plane at c0
        # along the outward axis — this inverts its umbrella so its
        # angle to the outward axis becomes > 100° but its bond
        # length to c0 is preserved (reflections preserve length).
        h_target = h_neighbors[0]
        h_vec = np.array(coords[h_target]) - c0_pos
        proj = float(np.dot(h_vec, outward)) * outward
        coords[h_target] = (np.array(coords[h_target]) - 2.0 * proj).tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        # Capture per-H bond lengths BEFORE symmetrization.
        c_pos_before = np.asarray(bad['coords'])[c0]
        original_per_h_dists = {
            h: float(np.linalg.norm(np.asarray(bad['coords'])[h] - c_pos_before))
            for h in h_neighbors
        }
        out = restore_terminal_h_symmetry(bad, sp.mol, c0)
        # IMPORTANT: the helper must NOT bail out; the returned xyz
        # must be a *new* dict with the inverted H repaired.
        self.assertIsNot(out, bad,
                         'symmetrizer bailed out on an inverted CH₃ — '
                         ' expected the per-H bond-length '
                         'reconstruction to repair this case')
        new_coords = np.asarray(out['coords'])
        c_pos = new_coords[c0]
        # Each H's bond length must be preserved exactly.
        for h, original_d in original_per_h_dists.items():
            new_d = float(np.linalg.norm(new_coords[h] - c_pos))
            self.assertAlmostEqual(
                new_d, original_d, places=6,
                msg=f'H{h} bond length not preserved')
        # The previously-inverted H must now sit on the OUTWARD side
        # of the parent → center axis (positive component along
        # ``outward``).
        repaired_vec = new_coords[h_target] - c_pos
        outward_component = float(np.dot(repaired_vec, outward))
        self.assertGreater(
            outward_component, 0.0,
            'inverted H should have been folded back to the outward side')
        # And no two H atoms on c0 should be unphysically close.
        for i in range(len(h_neighbors)):
            for j in range(i + 1, len(h_neighbors)):
                d = float(np.linalg.norm(
                    new_coords[h_neighbors[i]] - new_coords[h_neighbors[j]]))
                self.assertGreater(
                    d, 1.30,
                    msg=f'H{h_neighbors[i]}-H{h_neighbors[j]} too close: {d:.3f}')

    def test_skips_non_terminal_centers(self):
        # The central C of propane has 2 heavy neighbors → not a clear terminal.
        sp = ARCSpecies(label='propane', smiles='CCC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        # The middle C is the one with 2 C neighbors.
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        middle_c = None
        for atom in sp.mol.atoms:
            if atom.element.symbol == 'C':
                heavy_nbrs = sum(1 for n in atom.bonds.keys() if n.element.symbol != 'H')
                if heavy_nbrs == 2:
                    middle_c = atom_to_idx[atom]
                    break
        self.assertIsNotNone(middle_c)
        out = restore_terminal_h_symmetry(xyz, sp.mol, middle_c)
        # No-op: the function returns the original input.
        self.assertIs(out, xyz)

    def test_does_not_introduce_h_crowding(self):
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        out = restore_terminal_h_symmetry(
            xyz, sp.mol,
            next(i for i, s in enumerate(xyz['symbols']) if s == 'C'),
        )
        # Verify pairwise H distances are not closer than 1.5 Å.
        coords = np.asarray(out['coords'])
        h_idxs = [i for i, s in enumerate(out['symbols']) if s == 'H']
        for i in range(len(h_idxs)):
            for j in range(i + 1, len(h_idxs)):
                d = float(np.linalg.norm(coords[h_idxs[i]] - coords[h_idxs[j]]))
                self.assertGreater(d, 1.50)


class TestIsTerminalGroupAsymmetric(unittest.TestCase):
    """pure detector for unphysically distorted terminal CH₂/CH₃.

    Returns ``True`` only when an umbrella inversion (any H angle to
    the parent → center axis exceeds 100°) OR an azimuthal distortion
    (CH₃ cyclic spacings deviate from 120° by more than the threshold,
    or CH₂ smaller cyclic separation drops below 80°) is present.
    Returns ``False`` for already-good geometries so the symmetry
    restorer leaves them alone.
    """

    @staticmethod
    def _build_axial_ch3(theta_radians):
        """Construct an XYZ for a model -CH₃ group with explicit per-H
        azimuth angles around a fixed parent → center axis.

        Atom layout:
            0: parent (X) at (0, 0, 0)
            1: center (C) at (0, 0, 1.54)         — outward axis is +z
            2..2+n_h: H atoms placed at azimuth ``theta_radians`` around +z
        """
        n_h = len(theta_radians)
        # Standard outward C–H direction relative to the +z axis is at
        # 70.5° from +z (the supplement of the tetrahedral 109.5° H–C–C
        # angle).  Place each H on that cone at the chosen azimuth.
        cone_angle = np.deg2rad(70.5)
        sin_c = float(np.sin(cone_angle))
        cos_c = float(np.cos(cone_angle))
        center = np.array([0.0, 0.0, 1.54])
        symbols = ['C', 'C'] + ['H'] * n_h
        coords = [(0.0, 0.0, 0.0), tuple(center)]
        for theta in theta_radians:
            h = center + np.array([sin_c * np.cos(theta),
                                   sin_c * np.sin(theta),
                                   cos_c])
            coords.append(tuple(h))
        return {
            'symbols': tuple(symbols),
            'isotopes': (12, 12) + (1,) * n_h,
            'coords': tuple(coords),
        }, 1, 0, list(range(2, 2 + n_h))

    def test_ideal_ch3_is_not_asymmetric(self):
        """An ideal CH₃ with H atoms at 0°, 120°, 240° passes."""
        xyz, c, p, hs = self._build_axial_ch3(
            [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs))

    def test_inverted_ch3_umbrella_is_asymmetric(self):
        """An H angle to the outward axis exceeding 100° (the inverted
        umbrella case) is rejected."""
        # Build a CH₃ but flip ONE H to point back toward the parent by
        # placing it on the *inverted* cone (z < 0 below the center).
        n_h = 3
        thetas = [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]
        cone_angle = np.deg2rad(70.5)
        center = np.array([0.0, 0.0, 1.54])
        coords = [(0.0, 0.0, 0.0), tuple(center)]
        for k, theta in enumerate(thetas):
            sin_c = float(np.sin(cone_angle))
            if k == 0:
                # Inverted: place this H on the -z side of the center.
                cos_c = -float(np.cos(cone_angle))  # negative z component
            else:
                cos_c = float(np.cos(cone_angle))
            h = center + np.array([sin_c * np.cos(theta),
                                   sin_c * np.sin(theta),
                                   cos_c])
            coords.append(tuple(h))
        xyz = {
            'symbols': ('C', 'C', 'H', 'H', 'H'),
            'isotopes': (12, 12, 1, 1, 1),
            'coords': tuple(coords),
        }
        self.assertTrue(is_terminal_group_asymmetric(
            xyz, center_idx=1, parent_idx=0, h_indices=[2, 3, 4]))

    def test_ch3_azimuthal_distortion_is_asymmetric(self):
        """A CH₃ where one H is bent far enough that the cyclic spacings
        deviate from 120° by more than the default threshold (20°) is
        rejected."""
        # Place H atoms at 0°, 120°, 240° + 50°  = 290°.  Cyclic
        # spacings become 120°, 170°, 70° → max deviation 50° > 20°.
        thetas = [0.0,
                  2.0 * np.pi / 3.0,
                  4.0 * np.pi / 3.0 + np.deg2rad(50.0)]
        xyz, c, p, hs = self._build_axial_ch3(thetas)
        self.assertTrue(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs))

    def test_ch3_small_distortion_is_not_asymmetric(self):
        """A CH₃ with H atoms only ~10° away from ideal must NOT trip
        the detector — the threshold (20°) gives a comfortable margin
        so RDKit/force-field geometries pass through unchanged."""
        thetas = [0.0,
                  2.0 * np.pi / 3.0 + np.deg2rad(10.0),
                  4.0 * np.pi / 3.0 - np.deg2rad(5.0)]
        xyz, c, p, hs = self._build_axial_ch3(thetas)
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs))

    def test_ch2_squeezed_is_asymmetric(self):
        """A CH₂ whose two H projected azimuths are squeezed to ~60°
        (smaller cyclic separation < 80°) is rejected."""
        thetas = [0.0, np.deg2rad(60.0)]
        xyz, c, p, hs = self._build_axial_ch3(thetas)
        # Update symbols / isotopes for the 2-H case.
        new_symbols = ('C', 'C', 'H', 'H')
        new_iso = (12, 12, 1, 1)
        xyz = {
            'symbols': new_symbols,
            'isotopes': new_iso,
            'coords': xyz['coords'],
        }
        self.assertTrue(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs))

    def test_ch2_outward_pair_is_not_asymmetric(self):
        """A CH₂ with the two H projected azimuths roughly opposite
        (~180° apart) is *not* rejected — that's the normal terminal
        =CH₂ geometry."""
        thetas = [0.0, np.pi]
        xyz, c, p, hs = self._build_axial_ch3(thetas)
        xyz = {
            'symbols': ('C', 'C', 'H', 'H'),
            'isotopes': (12, 12, 1, 1),
            'coords': xyz['coords'],
        }
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs))

    def test_non_terminal_h_count_returns_false(self):
        """Passing fewer than 2 or more than 3 H indices is treated as
        'not eligible' and returns False without throwing."""
        xyz, c, p, hs = self._build_axial_ch3(
            [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        # Single H (not a CH₂/CH₃ shape) → False.
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=[hs[0]]))
        # 4 H atoms (CH₄ shape) → False.
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs + [hs[0]]))

    def test_degenerate_axis_returns_false(self):
        """When parent and center share the same coordinates, the axis
        is degenerate; the function returns False (no opinion)."""
        xyz = {
            'symbols': ('C', 'C', 'H', 'H', 'H'),
            'isotopes': (12, 12, 1, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (0.0, 0.0, 0.0),  # parent ≡ center
                       (1.0, 0.0, 0.0),
                       (-0.5, 0.866, 0.0),
                       (-0.5, -0.866, 0.0)),
        }
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=1, parent_idx=0, h_indices=[2, 3, 4]))

    def test_threshold_is_tunable(self):
        """A wider threshold lets larger distortions pass."""
        thetas = [0.0,
                  2.0 * np.pi / 3.0,
                  4.0 * np.pi / 3.0 + np.deg2rad(25.0)]
        xyz, c, p, hs = self._build_axial_ch3(thetas)
        # Default threshold (20°) → asymmetric.
        self.assertTrue(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs))
        # Wider threshold (40°) → no longer asymmetric.
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs,
            threshold_deg=40.0))


class TestApplyReactiveCenterCleanup(unittest.TestCase):
    """Thin orchestrator over the existing local-geometry helpers.

    These tests demonstrate that the orchestrator:
    * is a no-op when nothing is requested,
    * routes a migration triple through ``clean_migrating_h`` +
      ``orient_h_away_from_axis`` + ``regularize_terminal_h_geometry``,
    * runs ``restore_terminal_h_symmetry`` on each named reactive center
      (and respects the existing terminal-CH₂/CH₃ gating),
    * leaves unrelated atoms in the molecule untouched.
    """

    def test_no_op_when_no_targets(self):
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        out = apply_reactive_center_cleanup(xyz, sp.mol)
        self.assertIs(out, xyz)
        out_none = apply_reactive_center_cleanup(
            xyz, sp.mol, migrations=[], reactive_centers=set())
        self.assertIs(out_none, xyz)

    def test_orchestrator_repositions_migrating_h(self):
        """Given a (donor, acceptor, h_idx) record where the H is sitting
        at its donor-side bond length, the orchestrator triangulates it
        to the symmetric Pauling TS position between donor and acceptor."""
        # Use methylamine (CH3-NH2) — atom 0 is C (donor), atom 1 is N
        # (acceptor), pick one of C's H atoms as the migrating H.  Then
        # override the geometry to make the H sit on the C-N axis at
        # a normal C-H bond length (1.10 Å).
        sp = ARCSpecies(label='MA', smiles='CN')
        symbols = sp.get_xyz()['symbols']
        c_idx = next(i for i, s in enumerate(symbols) if s == 'C')
        n_idx = next(i for i, s in enumerate(symbols) if s == 'N')
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_on_c = next(
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c_idx].bonds.keys()
            if nbr.element.symbol == 'H'
        )
        # Build a synthetic geometry that lays C, N, and the chosen H
        # in a straight line: C at origin, N at +x = 3.0, the H at
        # +x = 1.10 (still bonded to C in the graph).
        coords = np.zeros((len(symbols), 3))
        coords[c_idx] = (0.0, 0.0, 0.0)
        coords[n_idx] = (2.50, 0.0, 0.0)
        coords[h_on_c] = (1.10, 0.10, 0.0)
        # Park the other atoms far enough away that they don't matter.
        for k in range(len(symbols)):
            if k in (c_idx, n_idx, h_on_c):
                continue
            coords[k] = (10.0 + k, 10.0, 10.0)
        synthetic = {'symbols': symbols, 'isotopes': sp.get_xyz()['isotopes'],
                     'coords': tuple(tuple(row) for row in coords)}
        out = apply_reactive_center_cleanup(
            synthetic, sp.mol,
            migrations=[{'h_idx': h_on_c, 'donor': c_idx, 'acceptor': n_idx}],
        )
        coords_out = np.asarray(out['coords'], dtype=float)
        d_ch = float(np.linalg.norm(coords_out[h_on_c] - coords_out[c_idx]))
        d_nh = float(np.linalg.norm(coords_out[h_on_c] - coords_out[n_idx]))
        sbl_ch = float(get_single_bond_length('C', 'H'))
        sbl_nh = float(get_single_bond_length('N', 'H'))
        self.assertAlmostEqual(d_ch, sbl_ch + PAULING_DELTA, places=3)
        self.assertAlmostEqual(d_nh, sbl_nh + PAULING_DELTA, places=3)

    def test_orchestrator_does_not_rotate_already_symmetric_ch3(self):
        """Regression guard for already-symmetric terminal CH₃.

        When the named reactive center is an already-symmetric terminal
        CH₃, the orchestrator's symmetry restoration must NOT fire (the
        asymmetry detector returns False) and the H atoms must end up
        byte-for-byte where they started.

        Earlier wiring that always called
        ``restore_terminal_h_symmetry`` at this site churned the
        coordinates of already-good groups; this test guards against
        that regression by exercising the asymmetry-gated
        groups.
        """
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_neighbors = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        before = np.asarray(xyz['coords'], dtype=float).copy()
        out = apply_reactive_center_cleanup(
            xyz, sp.mol, reactive_centers={c0}, restore_symmetry=True)
        after = np.asarray(out['coords'], dtype=float)
        for h in h_neighbors:
            self.assertTrue(
                np.allclose(before[h], after[h], atol=1e-9),
                msg=f'H{h} moved on an already-symmetric CH₃ '
                    f'(asymmetry detector should have returned False)',
            )

    def test_orchestrator_restores_azimuthally_distorted_ch3_when_signaled(self):
        """when the orchestrator is given a CH₃ whose H
        atoms are azimuthally distorted (cyclic spacings around the
        parent → center axis deviate from 120° by more than the
        detector threshold), the asymmetry detector returns True and
        ``restore_terminal_h_symmetry`` is invoked.

        We use azimuthal-only distortion (rotation of one H around the
        outward axis) so that all per-H bond lengths to the parent C
        are preserved — that keeps the symmetrizer's 0.05 Å bond-length
        bail-out from firing.  This isolates the test to the
        "asymmetry signal triggers restoration" pathway specifically.
        """
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        c1 = next(i for i, s in enumerate(symbols) if s == 'C' and i != c0)
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_neighbors = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        coords = list(map(list, xyz['coords']))
        # Build the outward axis (c0 → away from c1).
        c0_pos = np.array(coords[c0])
        c1_pos = np.array(coords[c1])
        axis = c0_pos - c1_pos
        axis /= float(np.linalg.norm(axis))
        # Rotate ONE H of c0 by +50° around the axis (Rodrigues rotation)
        # — this preserves its bond length to c0 but breaks the 120°
        # azimuthal symmetry, which the detector should pick up.
        h_target = h_neighbors[0]
        h_vec = np.array(coords[h_target]) - c0_pos
        theta_rad = np.deg2rad(50.0)
        cos_t, sin_t = float(np.cos(theta_rad)), float(np.sin(theta_rad))
        rotated = (h_vec * cos_t
                   + np.cross(axis, h_vec) * sin_t
                   + axis * float(np.dot(axis, h_vec)) * (1.0 - cos_t))
        coords[h_target] = (c0_pos + rotated).tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        # Sanity: the bond length is preserved.
        self.assertAlmostEqual(
            float(np.linalg.norm(np.array(coords[h_target]) - c0_pos)),
            float(np.linalg.norm(np.array(xyz['coords'][h_target]) - c0_pos)),
            places=6,
        )
        out = apply_reactive_center_cleanup(
            bad, sp.mol, reactive_centers={c0}, restore_symmetry=True)
        new_coords = np.asarray(out['coords'], dtype=float)
        # The orchestrator should have re-symmetrized: at least one H
        # of c0 must have moved relative to the distorted input.
        moved = any(
            not np.allclose(np.array(coords[h]), new_coords[h], atol=1e-6)
            for h in h_neighbors
        )
        self.assertTrue(
            moved,
            'orchestrator should have re-symmetrized the azimuthally distorted CH₃')
        # And the resulting H–C bond lengths should still all be ~equal
        # (symmetry restoration preserves bond length within 0.05 Å).
        c_pos = new_coords[c0]
        dists = [float(np.linalg.norm(new_coords[h] - c_pos))
                 for h in h_neighbors]
        self.assertLess(max(dists) - min(dists), 0.05)

    def test_orchestrator_repairs_inverted_ch3_end_to_end(self):
        """pass an inverted-CH₃ ethane through the live
        orchestrator with ``restore_symmetry=True``.  The 
        asymmetry detector must fire, the per-H bond-length
        symmetrizer must actually repair the inversion (the previously-
        inverted H ends up on the OUTWARD side of the parent → center
        axis), and per-H bond lengths must be preserved exactly."""
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        c1 = next(i for i, s in enumerate(symbols) if s == 'C' and i != c0)
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_neighbors = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        coords = list(map(list, xyz['coords']))
        c0_pos = np.array(coords[c0])
        c1_pos = np.array(coords[c1])
        outward = c0_pos - c1_pos
        outward /= float(np.linalg.norm(outward))
        h_target = h_neighbors[0]
        h_vec = np.array(coords[h_target]) - c0_pos
        proj = float(np.dot(h_vec, outward)) * outward
        # Reflect to invert: bond length is preserved by reflection.
        coords[h_target] = (np.array(coords[h_target]) - 2.0 * proj).tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        # Capture per-H bond lengths BEFORE the orchestrator runs.
        before = np.asarray(bad['coords'], dtype=float)
        c_pos_before = before[c0]
        original_per_h_dists = {
            h: float(np.linalg.norm(before[h] - c_pos_before))
            for h in h_neighbors
        }
        out = apply_reactive_center_cleanup(
            bad, sp.mol, reactive_centers={c0}, restore_symmetry=True)
        new_coords = np.asarray(out['coords'], dtype=float)
        # The inverted H must have moved (the asymmetry signal fired
        # and the symmetrizer ran end-to-end).
        self.assertFalse(
            np.allclose(before[h_target], new_coords[h_target], atol=1e-4),
            'inverted H should have been repaired end-to-end via the orchestrator')
        # The repaired H must be on the OUTWARD side of the parent
        # → center axis (positive component along ``outward``).
        repaired_vec = new_coords[h_target] - new_coords[c0]
        outward_component = float(np.dot(repaired_vec, outward))
        self.assertGreater(
            outward_component, 0.0,
            'repaired H should be on the outward side of the parent → center axis')
        # Per-H bond lengths must be preserved ( contract).
        c_pos_after = new_coords[c0]
        for h, original_d in original_per_h_dists.items():
            new_d = float(np.linalg.norm(new_coords[h] - c_pos_after))
            self.assertAlmostEqual(
                new_d, original_d, places=6,
                msg=f'H{h} bond length not preserved after orchestrator')

    def test_orchestrator_distorted_ch3_does_not_touch_unrelated_atoms(self):
        """even when symmetry restoration *does* fire on a
        distorted center, atoms outside the immediate first shell of
        that center are not moved at all (no whole-molecule churn)."""
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        c1 = next(i for i, s in enumerate(symbols) if s == 'C' and i != c0)
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_on_c1 = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c1].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        h_on_c0 = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        coords = list(map(list, xyz['coords']))
        # Invert one H on c0 so the asymmetry detector fires there.
        c0_pos = np.array(coords[c0])
        c1_pos = np.array(coords[c1])
        outward = c0_pos - c1_pos
        outward /= float(np.linalg.norm(outward))
        h_vec = np.array(coords[h_on_c0[0]]) - c0_pos
        proj = float(np.dot(h_vec, outward)) * outward
        coords[h_on_c0[0]] = (np.array(coords[h_on_c0[0]]) - 2.0 * proj).tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        before = np.asarray(bad['coords'], dtype=float).copy()
        out = apply_reactive_center_cleanup(
            bad, sp.mol, reactive_centers={c0}, restore_symmetry=True)
        after = np.asarray(out['coords'], dtype=float)
        # The other terminal C and its 3 H atoms must be untouched.
        for idx in [c1] + h_on_c1:
            self.assertTrue(
                np.allclose(before[idx], after[idx], atol=1e-9),
                msg=f'atom {idx} ({symbols[idx]}) moved when it should not have',
            )

    def test_orchestrator_does_not_touch_unrelated_atoms(self):
        """When the orchestrator is given one explicit reactive center,
        atoms not in the immediate first shell of that center are not
        moved at all (no whole-molecule churn)."""
        sp = ARCSpecies(label='propane', smiles='CCC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        # The middle C has 2 heavy neighbors → ``restore_terminal_h_symmetry``
        # bails out, and ``regularize_terminal_h_geometry`` is a no-op when
        # the H bond lengths are already in range.  Pick a *terminal* C
        # so the orchestrator IS allowed to act on it; verify that the
        # atoms in the OTHER terminal group are unchanged.
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        terminal_cs = []
        for atom in sp.mol.atoms:
            if atom.element.symbol == 'C':
                heavy_nbrs = sum(1 for n in atom.bonds.keys()
                                 if n.element.symbol != 'H')
                if heavy_nbrs == 1:
                    terminal_cs.append(atom_to_idx[atom])
        self.assertEqual(len(terminal_cs), 2)
        c_target = terminal_cs[0]
        c_other = terminal_cs[1]
        coords_before = np.asarray(xyz['coords']).copy()
        out = apply_reactive_center_cleanup(
            xyz, sp.mol, reactive_centers={c_target})
        coords_after = np.asarray(out['coords'])
        # The OTHER terminal C and its 3 H atoms must be untouched.
        other_h_idxs = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c_other].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        for idx in [c_other] + other_h_idxs:
            self.assertTrue(
                np.allclose(coords_before[idx], coords_after[idx], atol=1e-9),
                msg=f'atom {idx} ({symbols[idx]}) moved when it should not have',
            )

    def test_orchestrator_skips_migrating_h_in_symmetry_pass(self):
        """The migrating H should never be re-symmetrized — it has been
        intentionally placed at the triangulated TS position by
        ``clean_migrating_h`` and the symmetry pass would otherwise drag
        it back to a normal C–H bond length."""
        sp = ARCSpecies(label='MA', smiles='CN')
        symbols = sp.get_xyz()['symbols']
        c_idx = next(i for i, s in enumerate(symbols) if s == 'C')
        n_idx = next(i for i, s in enumerate(symbols) if s == 'N')
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_on_c = next(
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c_idx].bonds.keys()
            if nbr.element.symbol == 'H'
        )
        # Build a synthetic colinear geometry — same as the previous test.
        coords = np.zeros((len(symbols), 3))
        coords[c_idx] = (0.0, 0.0, 0.0)
        coords[n_idx] = (2.50, 0.0, 0.0)
        coords[h_on_c] = (1.10, 0.10, 0.0)
        for k in range(len(symbols)):
            if k in (c_idx, n_idx, h_on_c):
                continue
            coords[k] = (10.0 + k, 10.0, 10.0)
        synthetic = {'symbols': symbols, 'isotopes': sp.get_xyz()['isotopes'],
                     'coords': tuple(tuple(row) for row in coords)}
        out = apply_reactive_center_cleanup(
            synthetic, sp.mol,
            migrations=[{'h_idx': h_on_c, 'donor': c_idx, 'acceptor': n_idx}],
        )
        coords_out = np.asarray(out['coords'], dtype=float)
        d_ch = float(np.linalg.norm(coords_out[h_on_c] - coords_out[c_idx]))
        sbl_ch = float(get_single_bond_length('C', 'H'))
        # The migrating H must have moved away from its original 1.10 Å
        # bond length toward the Pauling TS distance (~1.51 Å).
        self.assertGreater(d_ch, sbl_ch + 0.10)


class TestIsInternalReactiveCh2Misoriented(unittest.TestCase):
    """pure detector for misoriented internal CH₂ shells.

    Returns ``True`` only when the internal CH₂ shell at ``center_idx``
    fails one of the three local rules (squeezed H–C–H, heavy-corridor
    crowding, or sp³ plane violation).  Returns ``False`` for already-
    good internal CH₂ shells and for non-eligible centers (terminal,
    wrong H count, missing/degenerate frame).
    """

    @staticmethod
    def _propane_middle_ch2():
        """Return ``(sp, xyz, middle_c, heavy_nbrs, h_nbrs)`` for the
        middle CH₂ of propane.  This is the canonical *good* internal
        CH₂ shell used as the baseline geometry for these tests."""
        sp = ARCSpecies(label='propane', smiles='CCC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        middle_c = None
        for atom in sp.mol.atoms:
            if atom.element.symbol != 'C':
                continue
            heavy = [n for n in atom.bonds.keys() if n.element.symbol != 'H']
            if len(heavy) == 2:
                middle_c = atom_to_idx[atom]
                break
        assert middle_c is not None
        middle_atom = sp.mol.atoms[middle_c]
        heavy_nbrs = [atom_to_idx[n] for n in middle_atom.bonds.keys()
                      if n.element.symbol != 'H']
        h_nbrs = [atom_to_idx[n] for n in middle_atom.bonds.keys()
                  if n.element.symbol == 'H']
        return sp, xyz, middle_c, heavy_nbrs, h_nbrs

    def test_well_oriented_internal_ch2_returns_false(self):
        """A fresh propane middle CH₂ is a healthy sp³ shell."""
        _, xyz, middle_c, heavy_nbrs, h_nbrs = self._propane_middle_ch2()
        self.assertFalse(
            is_internal_reactive_ch2_misoriented(
                xyz, center_idx=middle_c,
                heavy_nbr_indices=heavy_nbrs,
                h_indices=h_nbrs))

    def test_signature_has_no_threshold_parameter(self):
        """``threshold_deg`` was removed because it was never wired
        into the rule logic.  Guard the cleanup so the misleading knob
        is not silently re-introduced.
        """
        import inspect
        sig = inspect.signature(is_internal_reactive_ch2_misoriented)
        self.assertNotIn(
            'threshold_deg', sig.parameters,
            msg='is_internal_reactive_ch2_misoriented must not expose '
                'a threshold_deg parameter — the rules use hard-coded '
                'thresholds (80°, 0.30, 0.15) by design')

    def test_squeezed_h_arrangement_returns_true(self):
        """Two H atoms squeezed into a small H–C–H angle (Rule A)."""
        _, xyz, middle_c, heavy_nbrs, h_nbrs = self._propane_middle_ch2()
        coords = list(map(list, xyz['coords']))
        c_pos = np.array(coords[middle_c])
        h0_pos = np.array(coords[h_nbrs[0]])
        h1_pos = np.array(coords[h_nbrs[1]])
        # Move both H atoms onto the same direction (the average of the
        # two H directions), preserving each one's bond length.
        d0 = h0_pos - c_pos
        d1 = h1_pos - c_pos
        r0 = float(np.linalg.norm(d0))
        r1 = float(np.linalg.norm(d1))
        avg = (d0 / r0) + (d1 / r1)
        avg_norm = float(np.linalg.norm(avg))
        self.assertGreater(avg_norm, 1e-6)
        avg_hat = avg / avg_norm
        # Tilt them very slightly off the shared direction so they are
        # not literally on top of each other but still squeezed.
        # H0 ← c + r0 · normalize(avg_hat + 0.05 · perp)
        # H1 ← c + r1 · normalize(avg_hat - 0.05 · perp)
        perp = np.cross(avg_hat, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(perp)) < 1e-3:
            perp = np.cross(avg_hat, np.array([0.0, 1.0, 0.0]))
        perp = perp / float(np.linalg.norm(perp))
        new_dir_0 = avg_hat + 0.05 * perp
        new_dir_0 /= float(np.linalg.norm(new_dir_0))
        new_dir_1 = avg_hat - 0.05 * perp
        new_dir_1 /= float(np.linalg.norm(new_dir_1))
        coords[h_nbrs[0]] = (c_pos + r0 * new_dir_0).tolist()
        coords[h_nbrs[1]] = (c_pos + r1 * new_dir_1).tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        self.assertTrue(
            is_internal_reactive_ch2_misoriented(
                bad, center_idx=middle_c,
                heavy_nbr_indices=heavy_nbrs,
                h_indices=h_nbrs))

    def test_corridor_crowding_returns_true(self):
        """Both H atoms pushed into the heavy-corridor side (Rule B).

        Reflect each H through the center along the heavy-corridor
        bisector ŵ.  Reflection preserves bond length but flips the
        sign of the projection onto ŵ — so an originally back-side
        H (proj_w < 0) becomes a forbidden front-side H (proj_w > 0).
        """
        _, xyz, middle_c, heavy_nbrs, h_nbrs = self._propane_middle_ch2()
        coords_arr = np.asarray(xyz['coords'], dtype=float)
        c_pos = coords_arr[middle_c]
        a = coords_arr[heavy_nbrs[0]]
        b = coords_arr[heavy_nbrs[1]]
        v_a = a - c_pos
        v_b = b - c_pos
        va_hat = v_a / float(np.linalg.norm(v_a))
        vb_hat = v_b / float(np.linalg.norm(v_b))
        w_raw = va_hat + vb_hat
        w_hat = w_raw / float(np.linalg.norm(w_raw))
        coords = list(map(list, xyz['coords']))
        for h in h_nbrs:
            h_vec = coords_arr[h] - c_pos
            proj = float(np.dot(h_vec, w_hat))
            # Reflect: subtract twice the projection along w_hat.
            new_h = coords_arr[h] - 2.0 * proj * w_hat
            # Now move that H so its projection on w_hat is *positive*
            # by another step (push deeper into the corridor).
            extra_along_w = 0.4 * w_hat
            new_h = new_h + extra_along_w
            coords[h] = new_h.tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        self.assertTrue(
            is_internal_reactive_ch2_misoriented(
                bad, center_idx=middle_c,
                heavy_nbr_indices=heavy_nbrs,
                h_indices=h_nbrs))

    def test_same_side_plane_violation_returns_true(self):
        """Both H atoms on the same side of the heavy plane (Rule C)."""
        _, xyz, middle_c, heavy_nbrs, h_nbrs = self._propane_middle_ch2()
        coords_arr = np.asarray(xyz['coords'], dtype=float)
        c_pos = coords_arr[middle_c]
        a = coords_arr[heavy_nbrs[0]]
        b = coords_arr[heavy_nbrs[1]]
        v_a = a - c_pos
        v_b = b - c_pos
        n_raw = np.cross(v_a, v_b)
        n_hat = n_raw / float(np.linalg.norm(n_raw))
        # Identify which H is on the negative side and reflect it
        # through the plane.  The reflected H now sits on the same
        # (positive) side as the other H.
        signs = []
        for h in h_nbrs:
            h_vec = coords_arr[h] - c_pos
            signs.append(float(np.dot(h_vec, n_hat)))
        # Find the H with the smaller (more negative) projection.
        flip_h = h_nbrs[0] if signs[0] < signs[1] else h_nbrs[1]
        coords = list(map(list, xyz['coords']))
        flip_vec = coords_arr[flip_h] - c_pos
        proj_n = float(np.dot(flip_vec, n_hat))
        new_h = coords_arr[flip_h] - 2.0 * proj_n * n_hat
        coords[flip_h] = new_h.tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        self.assertTrue(
            is_internal_reactive_ch2_misoriented(
                bad, center_idx=middle_c,
                heavy_nbr_indices=heavy_nbrs,
                h_indices=h_nbrs))

    def test_terminal_center_is_not_eligible(self):
        """Terminal CH₃ centers must NOT be evaluated by this detector
        (only 1 heavy neighbor — terminal centers go through the
        the terminal-group orchestrator pathway)."""
        sp = ARCSpecies(label='propane', smiles='CCC')
        xyz = sp.get_xyz()
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        terminal_c = None
        terminal_atom = None
        for atom in sp.mol.atoms:
            if atom.element.symbol != 'C':
                continue
            heavy = [n for n in atom.bonds.keys() if n.element.symbol != 'H']
            if len(heavy) == 1:
                terminal_c = atom_to_idx[atom]
                terminal_atom = atom
                break
        self.assertIsNotNone(terminal_c)
        heavy_nbrs = [atom_to_idx[n] for n in terminal_atom.bonds.keys()
                      if n.element.symbol != 'H']
        h_nbrs = [atom_to_idx[n] for n in terminal_atom.bonds.keys()
                  if n.element.symbol == 'H']
        # 1 heavy neighbor → ineligible
        self.assertFalse(
            is_internal_reactive_ch2_misoriented(
                xyz, center_idx=terminal_c,
                heavy_nbr_indices=heavy_nbrs,
                h_indices=h_nbrs))

    def test_wrong_h_count_is_not_eligible(self):
        """A center with the wrong number of H atoms is ineligible."""
        _, xyz, middle_c, heavy_nbrs, h_nbrs = self._propane_middle_ch2()
        # Pass only one H — ineligible.
        self.assertFalse(
            is_internal_reactive_ch2_misoriented(
                xyz, center_idx=middle_c,
                heavy_nbr_indices=heavy_nbrs,
                h_indices=h_nbrs[:1]))

    def test_collinear_heavy_neighbors_is_not_eligible(self):
        """If the two heavy neighbors are collinear with the center, the
        heavy plane is undefined and the detector returns False."""
        _, xyz, middle_c, heavy_nbrs, h_nbrs = self._propane_middle_ch2()
        coords = list(map(list, xyz['coords']))
        c_pos = np.array(coords[middle_c])
        # Place both heavy neighbors anti-parallel along x.
        coords[heavy_nbrs[0]] = (c_pos + np.array([1.5, 0.0, 0.0])).tolist()
        coords[heavy_nbrs[1]] = (c_pos + np.array([-1.5, 0.0, 0.0])).tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        self.assertFalse(
            is_internal_reactive_ch2_misoriented(
                bad, center_idx=middle_c,
                heavy_nbr_indices=heavy_nbrs,
                h_indices=h_nbrs))


class TestRepairInternalReactiveCh2(unittest.TestCase):
    """local repair primitive for internal CH₂ shells.

    The repair must (1) act only on the H shell of the named center,
    (2) preserve each H's original bond length to the center exactly,
    and (3) leave heavy neighbors and unrelated atoms untouched.
    """

    @staticmethod
    def _propane_middle_ch2():
        sp = ARCSpecies(label='propane', smiles='CCC')
        xyz = sp.get_xyz()
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        middle_c = None
        for atom in sp.mol.atoms:
            if atom.element.symbol != 'C':
                continue
            heavy = [n for n in atom.bonds.keys() if n.element.symbol != 'H']
            if len(heavy) == 2:
                middle_c = atom_to_idx[atom]
                break
        middle_atom = sp.mol.atoms[middle_c]
        heavy_nbrs = [atom_to_idx[n] for n in middle_atom.bonds.keys()
                      if n.element.symbol != 'H']
        h_nbrs = [atom_to_idx[n] for n in middle_atom.bonds.keys()
                  if n.element.symbol == 'H']
        return sp, xyz, middle_c, heavy_nbrs, h_nbrs

    def _build_corridor_crowded(self):
        """Return a propane XYZ in which the middle CH₂'s two H atoms
        have been pushed into the heavy-corridor side (the canonical
        Rule B failure)."""
        sp, xyz, middle_c, heavy_nbrs, h_nbrs = self._propane_middle_ch2()
        coords_arr = np.asarray(xyz['coords'], dtype=float)
        c_pos = coords_arr[middle_c]
        a = coords_arr[heavy_nbrs[0]]
        b = coords_arr[heavy_nbrs[1]]
        v_a = a - c_pos
        v_b = b - c_pos
        va_hat = v_a / float(np.linalg.norm(v_a))
        vb_hat = v_b / float(np.linalg.norm(v_b))
        w_hat = (va_hat + vb_hat)
        w_hat = w_hat / float(np.linalg.norm(w_hat))
        coords = list(map(list, xyz['coords']))
        # Reflect each H through the (heavy-plane) so it ends up on the
        # forbidden +ŵ side, preserving bond length to c by reflection.
        for h in h_nbrs:
            h_vec = coords_arr[h] - c_pos
            proj_w = float(np.dot(h_vec, w_hat))
            new_h = coords_arr[h] - 2.0 * proj_w * w_hat + 0.4 * w_hat
            coords[h] = new_h.tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        return sp, bad, middle_c, heavy_nbrs, h_nbrs

    def test_repair_preserves_per_h_bond_lengths(self):
        sp, bad, middle_c, heavy_nbrs, h_nbrs = self._build_corridor_crowded()
        before_arr = np.asarray(bad['coords'], dtype=float)
        c_pos_before = before_arr[middle_c]
        original_per_h_dists = {
            h: float(np.linalg.norm(before_arr[h] - c_pos_before))
            for h in h_nbrs
        }
        out = repair_internal_reactive_ch2(
            bad, center_idx=middle_c,
            heavy_nbr_indices=heavy_nbrs,
            h_indices=h_nbrs)
        new_arr = np.asarray(out['coords'], dtype=float)
        c_pos_after = new_arr[middle_c]
        for h, original_d in original_per_h_dists.items():
            new_d = float(np.linalg.norm(new_arr[h] - c_pos_after))
            self.assertAlmostEqual(
                new_d, original_d, places=6,
                msg=f'H{h} bond length not preserved '
                    f'(was {original_d:.6f}, now {new_d:.6f})')

    def test_repair_leaves_unrelated_atoms_unchanged(self):
        sp, bad, middle_c, heavy_nbrs, h_nbrs = self._build_corridor_crowded()
        n_atoms = len(bad['symbols'])
        before_arr = np.asarray(bad['coords'], dtype=float).copy()
        out = repair_internal_reactive_ch2(
            bad, center_idx=middle_c,
            heavy_nbr_indices=heavy_nbrs,
            h_indices=h_nbrs)
        after_arr = np.asarray(out['coords'], dtype=float)
        moved_set = set(h_nbrs)
        for idx in range(n_atoms):
            if idx in moved_set:
                continue
            self.assertTrue(
                np.allclose(before_arr[idx], after_arr[idx], atol=1e-12),
                msg=f'atom {idx} ({bad["symbols"][idx]}) moved when it '
                    f'should have been read-only')

    def test_repair_relieves_corridor_crowding(self):
        """After repair, the detector should no longer fire on the same
        center — the H atoms should be on the back-side of the heavy
        corridor, not crowding the front."""
        sp, bad, middle_c, heavy_nbrs, h_nbrs = self._build_corridor_crowded()
        # Sanity: the input is misoriented.
        self.assertTrue(
            is_internal_reactive_ch2_misoriented(
                bad, center_idx=middle_c,
                heavy_nbr_indices=heavy_nbrs,
                h_indices=h_nbrs))
        out = repair_internal_reactive_ch2(
            bad, center_idx=middle_c,
            heavy_nbr_indices=heavy_nbrs,
            h_indices=h_nbrs)
        # The repaired shell should pass the detector.
        self.assertFalse(
            is_internal_reactive_ch2_misoriented(
                out, center_idx=middle_c,
                heavy_nbr_indices=heavy_nbrs,
                h_indices=h_nbrs),
            msg='repair did not relieve the misorientation that the '
                'detector flagged on the input')
        # And the H–H angle should now be roughly tetrahedral (clearly
        # above 80°), not squeezed.
        new_arr = np.asarray(out['coords'], dtype=float)
        c = new_arr[middle_c]
        d0 = new_arr[h_nbrs[0]] - c
        d1 = new_arr[h_nbrs[1]] - c
        cos_hch = float(np.dot(d0, d1) / (np.linalg.norm(d0) * np.linalg.norm(d1)))
        cos_hch = max(-1.0, min(1.0, cos_hch))
        angle = float(np.degrees(np.arccos(cos_hch)))
        self.assertGreater(angle, 100.0)
        self.assertLess(angle, 120.0)

    def test_repair_skips_ineligible_centers(self):
        """Wrong H count → no-op."""
        sp, xyz, middle_c, heavy_nbrs, h_nbrs = self._propane_middle_ch2()
        out = repair_internal_reactive_ch2(
            xyz, center_idx=middle_c,
            heavy_nbr_indices=heavy_nbrs,
            h_indices=h_nbrs[:1])
        self.assertIs(out, xyz)


class TestApplyReactiveCenterCleanupInternalCh2(unittest.TestCase):
    """orchestrator-level integration of the internal CH₂
    sibling pass.

    The orchestrator should fire the new pass when ``reactive_centers``
    names a center whose internal CH₂ shell is misoriented, and leave
    already-good internal CH₂ shells (and unrelated atoms) byte-for-byte
    unchanged.
    """

    @staticmethod
    def _propane_middle_ch2():
        sp = ARCSpecies(label='propane', smiles='CCC')
        xyz = sp.get_xyz()
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        middle_c = None
        for atom in sp.mol.atoms:
            if atom.element.symbol != 'C':
                continue
            heavy = [n for n in atom.bonds.keys() if n.element.symbol != 'H']
            if len(heavy) == 2:
                middle_c = atom_to_idx[atom]
                break
        middle_atom = sp.mol.atoms[middle_c]
        heavy_nbrs = [atom_to_idx[n] for n in middle_atom.bonds.keys()
                      if n.element.symbol != 'H']
        h_nbrs = [atom_to_idx[n] for n in middle_atom.bonds.keys()
                  if n.element.symbol == 'H']
        return sp, xyz, middle_c, heavy_nbrs, h_nbrs

    def test_orchestrator_does_not_touch_good_internal_ch2(self):
        """A propane middle CH₂ in a fresh geometry must be left
        byte-for-byte unchanged by the orchestrator."""
        sp, xyz, middle_c, heavy_nbrs, h_nbrs = self._propane_middle_ch2()
        before = np.asarray(xyz['coords'], dtype=float).copy()
        out = apply_reactive_center_cleanup(
            xyz, sp.mol, reactive_centers={middle_c}, restore_symmetry=True)
        after = np.asarray(out['coords'], dtype=float)
        for h in h_nbrs:
            self.assertTrue(
                np.allclose(before[h], after[h], atol=1e-9),
                msg=f'H{h} moved on a healthy internal CH₂ —  '
                    f'detector should have returned False')
        # The middle C and the two heavy neighbors must also be untouched.
        self.assertTrue(np.allclose(before[middle_c], after[middle_c], atol=1e-9))
        for hv in heavy_nbrs:
            self.assertTrue(np.allclose(before[hv], after[hv], atol=1e-9))

    def test_orchestrator_repairs_misoriented_internal_ch2(self):
        """A corridor-crowded internal CH₂ should be detected and
        repaired by the orchestrator end-to-end."""
        sp, xyz, middle_c, heavy_nbrs, h_nbrs = self._propane_middle_ch2()
        coords_arr = np.asarray(xyz['coords'], dtype=float)
        c_pos = coords_arr[middle_c]
        a = coords_arr[heavy_nbrs[0]]
        b = coords_arr[heavy_nbrs[1]]
        va_hat = (a - c_pos) / float(np.linalg.norm(a - c_pos))
        vb_hat = (b - c_pos) / float(np.linalg.norm(b - c_pos))
        w_hat = va_hat + vb_hat
        w_hat = w_hat / float(np.linalg.norm(w_hat))
        coords = list(map(list, xyz['coords']))
        for h in h_nbrs:
            h_vec = coords_arr[h] - c_pos
            proj_w = float(np.dot(h_vec, w_hat))
            new_h = coords_arr[h] - 2.0 * proj_w * w_hat + 0.4 * w_hat
            coords[h] = new_h.tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        # Capture per-H bond lengths before the orchestrator runs.
        before_arr = np.asarray(bad['coords'], dtype=float)
        c_pos_before = before_arr[middle_c]
        original_per_h_dists = {
            h: float(np.linalg.norm(before_arr[h] - c_pos_before))
            for h in h_nbrs
        }
        out = apply_reactive_center_cleanup(
            bad, sp.mol, reactive_centers={middle_c}, restore_symmetry=True)
        new_arr = np.asarray(out['coords'], dtype=float)
        # The H atoms should have moved away from the corridor-crowded
        # input geometry.
        moved = any(
            not np.allclose(before_arr[h], new_arr[h], atol=1e-6)
            for h in h_nbrs
        )
        self.assertTrue(
            moved,
            'orchestrator should have repaired the misoriented internal CH₂')
        # The detector should no longer fire on the result.
        self.assertFalse(
            is_internal_reactive_ch2_misoriented(
                out, center_idx=middle_c,
                heavy_nbr_indices=heavy_nbrs,
                h_indices=h_nbrs))
        # Per-H bond lengths preserved.
        c_pos_after = new_arr[middle_c]
        for h, original_d in original_per_h_dists.items():
            new_d = float(np.linalg.norm(new_arr[h] - c_pos_after))
            self.assertAlmostEqual(
                new_d, original_d, places=6,
                msg=f'H{h} bond length not preserved through orchestrator')

    def test_orchestrator_internal_ch2_does_not_touch_unrelated_atoms(self):
        """When the internal CH₂ pass repairs a misoriented center,
        atoms outside the {center, h0, h1} shell must be untouched."""
        sp, xyz, middle_c, heavy_nbrs, h_nbrs = self._propane_middle_ch2()
        coords_arr = np.asarray(xyz['coords'], dtype=float)
        c_pos = coords_arr[middle_c]
        a = coords_arr[heavy_nbrs[0]]
        b = coords_arr[heavy_nbrs[1]]
        va_hat = (a - c_pos) / float(np.linalg.norm(a - c_pos))
        vb_hat = (b - c_pos) / float(np.linalg.norm(b - c_pos))
        w_hat = va_hat + vb_hat
        w_hat = w_hat / float(np.linalg.norm(w_hat))
        coords = list(map(list, xyz['coords']))
        for h in h_nbrs:
            h_vec = coords_arr[h] - c_pos
            proj_w = float(np.dot(h_vec, w_hat))
            new_h = coords_arr[h] - 2.0 * proj_w * w_hat + 0.4 * w_hat
            coords[h] = new_h.tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        before_arr = np.asarray(bad['coords'], dtype=float).copy()
        out = apply_reactive_center_cleanup(
            bad, sp.mol, reactive_centers={middle_c}, restore_symmetry=True)
        after_arr = np.asarray(out['coords'], dtype=float)
        moved_indices = set(h_nbrs)
        n_atoms = len(bad['symbols'])
        for idx in range(n_atoms):
            if idx in moved_indices:
                continue
            self.assertTrue(
                np.allclose(before_arr[idx], after_arr[idx], atol=1e-9),
                msg=f'atom {idx} ({bad["symbols"][idx]}) moved when it '
                    f'should have remained read-only')

    def test_orchestrator_internal_ch2_pass_skipped_when_disabled(self):
        """When ``restore_symmetry=False`` the internal CH₂ pass must
        not run.  The internal-CH₂ pass is gated on the same flag as
        the terminal-group symmetry-restoration pass so the
        orchestrator's non-restoration mode is byte-for-byte
        identical to the original.
        """
        sp, xyz, middle_c, heavy_nbrs, h_nbrs = self._propane_middle_ch2()
        coords_arr = np.asarray(xyz['coords'], dtype=float)
        c_pos = coords_arr[middle_c]
        a = coords_arr[heavy_nbrs[0]]
        b = coords_arr[heavy_nbrs[1]]
        va_hat = (a - c_pos) / float(np.linalg.norm(a - c_pos))
        vb_hat = (b - c_pos) / float(np.linalg.norm(b - c_pos))
        w_hat = va_hat + vb_hat
        w_hat = w_hat / float(np.linalg.norm(w_hat))
        coords = list(map(list, xyz['coords']))
        for h in h_nbrs:
            h_vec = coords_arr[h] - c_pos
            proj_w = float(np.dot(h_vec, w_hat))
            new_h = coords_arr[h] - 2.0 * proj_w * w_hat + 0.4 * w_hat
            coords[h] = new_h.tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        before_arr = np.asarray(bad['coords'], dtype=float).copy()
        out = apply_reactive_center_cleanup(
            bad, sp.mol, reactive_centers={middle_c}, restore_symmetry=False)
        after_arr = np.asarray(out['coords'], dtype=float)
        for h in h_nbrs:
            self.assertTrue(
                np.allclose(before_arr[h], after_arr[h], atol=1e-9),
                msg=f'H{h} moved while restore_symmetry=False — internal '
                    f'CH₂ pass should be gated')


class TestIdentifyHMigrationPairs(unittest.TestCase):
    """``identify_h_migration_pairs`` recovers (h, donor, acceptor)
    triples mirroring ``migrate_verified_atoms``'s internal selection."""

    def test_returns_cross_bond_acceptor_when_available(self):
        # Mock: H4 is the migrating atom, donor is C0, acceptor is N1.
        xyz = {
            'symbols': ('C', 'N', 'C', 'H', 'H'),
            'isotopes': (12, 14, 12, 1, 1),
            'coords': (
                (0.0, 0.0, 0.0),     # C0 (donor, in large)
                (2.5, 0.0, 0.0),     # N1 (acceptor, in core)
                (1.0, 1.5, 0.0),     # C2 (in core)
                (0.5, -1.0, 0.0),    # H3 (non-migrating, on C0)
                (1.0, 0.5, 0.0),     # H4 (migrating)
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds
        c0 = _A('C', {}); n1 = _A('N', {}); c2 = _A('C', {})
        h3 = _A('H', {}); h4 = _A('H', {})
        c0.bonds = {h3: None, h4: None}
        h3.bonds = {c0: None}
        h4.bonds = {c0: None}
        class _M:
            atoms = [c0, n1, c2, h3, h4]
        recs = identify_h_migration_pairs(
            xyz, _M(), migrating_atoms={4},
            core={1, 2}, large_prod_atoms={0, 3},
            cross_bonds=[(4, 1)],
        )
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]['h_idx'], 4)
        self.assertEqual(recs[0]['donor'], 0)
        self.assertEqual(recs[0]['acceptor'], 1)
        self.assertEqual(recs[0]['source'], 'cross_bond')

    def test_falls_back_to_nearest_core_when_no_cross_bond(self):
        xyz = {
            'symbols': ('C', 'N', 'C', 'H'),
            'isotopes': (12, 14, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0), (2.5, 0.0, 0.0),
                (1.0, 1.5, 0.0), (1.0, 0.5, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds
        c0 = _A('C', {}); n1 = _A('N', {}); c2 = _A('C', {}); h3 = _A('H', {})
        c0.bonds = {h3: None}
        h3.bonds = {c0: None}
        class _M:
            atoms = [c0, n1, c2, h3]
        recs = identify_h_migration_pairs(
            xyz, _M(), migrating_atoms={3},
            core={1, 2}, large_prod_atoms={0}, cross_bonds=None,
        )
        self.assertEqual(len(recs), 1)
        # H3 is closest to C2 (1.0 Å away), not N1 (2.0 Å away).
        self.assertEqual(recs[0]['acceptor'], 2)
        self.assertEqual(recs[0]['source'], 'nearest_core')


# ---------------------------------------------------------------------------
# fragmentation-fallback single-H inference
# ---------------------------------------------------------------------------


class TestInferFragFallbackHMigration(unittest.TestCase):
    """Strict, deterministic single-H migration inference for the
    frag-fallback addition branch.

    The helper combines five signals (S1–S5) and returns either ONE
    migration record or ``None``.  Each test isolates exactly one
    signal to prove it gates correctly.
    """

    def _real_carboxylic_acid_setup(self):
        """Build a small carboxylic acid (formic acid) with the
        post-migration H sitting between the OH oxygen (donor) and a
        nearby acceptor C atom on a *separate* fragment.

        Used as the canonical S1+S2+S3+S4+S5 success case.
        """
        # Acetic acid → CH4 + CO2 (1,3_Insertion_CO2-style toy reaction).
        # Atom indices in CC(=O)O:
        #   0 C (methyl)
        #   1 C (carbonyl)
        #   2 O (=O)
        #   3 O (-OH)
        #   4 H (carboxylic-acid H, the migrating H)
        #   5..7 H on the methyl
        sp = ARCSpecies(label='acid', smiles='CC(=O)O')
        return sp

    def test_s1_zero_h_moved_returns_none(self):
        """No H displaced ⇒ S1 fails ⇒ ``None``."""
        from arc.job.adapters.ts.linear_utils.local_geometry import (
            infer_frag_fallback_h_migration,
        )
        sp = self._real_carboxylic_acid_setup()
        xyz = sp.get_xyz()
        out = infer_frag_fallback_h_migration(
            pre_xyz=xyz, post_xyz=xyz,
            uni_mol=sp.mol, split_bonds=[(1, 3)],
            multi_species=None, label='S1-zero',
        )
        self.assertIsNone(out)

    def test_s1_two_h_moved_returns_none(self):
        """Two H atoms displaced ⇒ S1 fails ⇒ ``None`` (no multi-H
        enrichment in )."""
        from arc.job.adapters.ts.linear_utils.local_geometry import (
            infer_frag_fallback_h_migration,
        )
        sp = self._real_carboxylic_acid_setup()
        symbols = sp.get_xyz()['symbols']
        h_idxs = [i for i, s in enumerate(symbols) if s == 'H']
        self.assertGreaterEqual(len(h_idxs), 2)
        pre_coords = list(map(list, sp.get_xyz()['coords']))
        post_coords = [list(c) for c in pre_coords]
        post_coords[h_idxs[0]][0] += 0.5
        post_coords[h_idxs[1]][0] += 0.5
        pre = {**sp.get_xyz(),
               'coords': tuple(tuple(c) for c in pre_coords)}
        post = {**sp.get_xyz(),
                'coords': tuple(tuple(c) for c in post_coords)}
        out = infer_frag_fallback_h_migration(
            pre_xyz=pre, post_xyz=post,
            uni_mol=sp.mol, split_bonds=[(1, 3)],
            multi_species=None, label='S1-two',
        )
        self.assertIsNone(out)

    def test_s2_h_with_two_heavy_neighbors_returns_none(self):
        """An H atom with two heavy neighbors in the reactant graph
        cannot have a unique donor ⇒ S2 fails ⇒ ``None``."""
        from arc.job.adapters.ts.linear_utils.local_geometry import (
            infer_frag_fallback_h_migration,
        )
        # Build a stub mol where one H has two heavy neighbors
        # (something like a hydride bridge — chemically unusual but
        # exactly the S2 failure case).
        class _A:
            def __init__(self, sym, bonds=None):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds or {}
        c0 = _A('C'); c1 = _A('C'); h = _A('H')
        c0.bonds = {h: 1}
        c1.bonds = {h: 1}
        h.bonds = {c0: 1, c1: 1}
        class _M:
            atoms = [c0, c1, h]
        pre = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.5, 0.0)),
        }
        post = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.5, 0.0)),
        }
        out = infer_frag_fallback_h_migration(
            pre_xyz=pre, post_xyz=post,
            uni_mol=_M(), split_bonds=[(0, 1)],
            multi_species=None, label='S2',
        )
        self.assertIsNone(out)

    def test_s3_donor_and_acceptor_in_same_fragment_returns_none(self):
        """When the inferred donor and acceptor lie in the same
        connected component after the cut, the migration cannot be a
        fragment-to-fragment H transfer ⇒ S3 fails ⇒ ``None``."""
        from arc.job.adapters.ts.linear_utils.local_geometry import (
            infer_frag_fallback_h_migration,
        )
        from arc.species import ARCSpecies
        # Ethanol: O bonded to C1 bonded to C0.  Cut C0-C1; the OH and
        # the methyl are now in different fragments.
        # If we move H bonded to C0 (in the methyl fragment) and the
        # nearest "acceptor" candidate is also in the methyl fragment,
        # S3 must fail.
        sp = ARCSpecies(label='ethanol', smiles='CCO')
        symbols = sp.get_xyz()['symbols']
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        # Find an H bonded to a methyl C (atom 0).
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        h_on_c0 = next(
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        )
        coords = list(map(list, sp.get_xyz()['coords']))
        pre_coords = [list(c) for c in coords]
        post_coords = [list(c) for c in coords]
        # Push the H far from C0 toward atom 1 — also in the same
        # fragment (after cutting C2-O).  Both donor and the nearest
        # heavy partner end up in the methyl fragment ⇒ S3 fails.
        post_coords[h_on_c0][0] += 1.0
        pre = {**sp.get_xyz(),
               'coords': tuple(tuple(c) for c in pre_coords)}
        post = {**sp.get_xyz(),
                'coords': tuple(tuple(c) for c in post_coords)}
        # Cut a bond that is NOT touching the methyl side: e.g. the C-O.
        c_idx_one = next(
            i for i, s in enumerate(symbols)
            if s == 'C' and i != c0
        )
        o_idx = next(i for i, s in enumerate(symbols) if s == 'O')
        out = infer_frag_fallback_h_migration(
            pre_xyz=pre, post_xyz=post,
            uni_mol=sp.mol, split_bonds=[(c_idx_one, o_idx)],
            multi_species=None, label='S3',
        )
        # The migrated H lives entirely inside the methyl fragment, so
        # *every* heavy candidate from the "other side" (just the O) is
        # outside the methyl fragment.  But the methyl fragment side
        # is the donor's side, so S3 enforces the acceptor to be in
        # the OH side — and the H may not be near it.  S5 catches the
        # bad geometry and S5 returns None.
        self.assertIsNone(out)

    def test_inference_success_returns_single_record(self):
        """Acetic acid → CO2 + CH4 toy mock: cut the C-OH bond and move
        the carboxylic-acid H toward the methyl C.  This is the canonical
        single-H migration case the helper must promote."""
        from arc.job.adapters.ts.linear_utils.local_geometry import (
            infer_frag_fallback_h_migration,
        )
        from arc.species import ARCSpecies
        sp = ARCSpecies(label='acid', smiles='CC(=O)O')
        symbols = sp.get_xyz()['symbols']
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}

        # Atom indices for acetic acid.  Find the OH H, the OH O, the
        # carbonyl C, and the methyl C.
        c_methyl = None
        c_carbonyl = None
        o_oh = None
        h_oh = None
        for i, atom in enumerate(sp.mol.atoms):
            if atom.element.symbol == 'C':
                heavy_nbrs = [
                    n for n in atom.bonds.keys()
                    if n.element.symbol != 'H'
                ]
                if len(heavy_nbrs) == 1:
                    c_methyl = i
                elif len(heavy_nbrs) >= 2:
                    c_carbonyl = i
            elif atom.element.symbol == 'O':
                # OH oxygen has exactly one H neighbor.
                if any(n.element.symbol == 'H' for n in atom.bonds.keys()):
                    o_oh = i
                    for n in atom.bonds.keys():
                        if n.element.symbol == 'H':
                            h_oh = atom_to_idx[n]
                            break
        self.assertIsNotNone(c_methyl)
        self.assertIsNotNone(c_carbonyl)
        self.assertIsNotNone(o_oh)
        self.assertIsNotNone(h_oh)

        # Build the pre/post geometry: pre = reactant, post = the OH H
        # has moved toward the methyl C at a TS-like distance.
        coords = list(map(list, sp.get_xyz()['coords']))
        # Move the methyl C far enough that the OH H, when relocated to
        # the C-O midpoint of the C_methyl←OH axis, lies between the
        # two heavy atoms at chemically sensible distances.
        coords[c_methyl] = [3.0, 0.0, 0.0]
        coords[o_oh] = [0.0, 0.0, 0.0]
        coords[c_carbonyl] = [-1.5, 0.0, 0.0]
        # Place all other heavy atoms very far away so they don't
        # collide with the H or interfere with S5 rival checks.
        far = 25.0
        for i, sym in enumerate(symbols):
            if i in (c_methyl, c_carbonyl, o_oh, h_oh):
                continue
            if sym == 'H':
                coords[i] = [far + i, far, far]
            else:
                coords[i] = [far + i, -far, -far]
        # Pre: the H is still on the OH side.
        coords[h_oh] = [-0.95, 0.0, 0.0]
        pre = {
            'symbols': symbols,
            'isotopes': sp.get_xyz().get(
                'isotopes', tuple(0 for _ in symbols)),
            'coords': tuple(tuple(float(c) for c in row) for row in coords),
        }
        # Post: the H has moved to the donor–acceptor TS midpoint.
        from arc.common import get_single_bond_length
        from arc.job.adapters.ts.linear_utils.postprocess import PAULING_DELTA
        d_dh = get_single_bond_length('O', 'H') + PAULING_DELTA
        d_ah = get_single_bond_length('C', 'H') + PAULING_DELTA
        d_da = float(np.linalg.norm(
            np.array(coords[c_methyl]) - np.array(coords[o_oh])))
        x = (d_da ** 2 + d_dh ** 2 - d_ah ** 2) / (2.0 * d_da)
        y = float(np.sqrt(max(d_dh ** 2 - x ** 2, 0.0)))
        coords_post = [list(c) for c in coords]
        coords_post[h_oh] = [x, y, 0.0]
        post = {
            'symbols': symbols,
            'isotopes': sp.get_xyz().get(
                'isotopes', tuple(0 for _ in symbols)),
            'coords': tuple(tuple(float(c) for c in row) for row in coords_post),
        }

        # Cut the carbonyl-C—OH bond so the OH H + OH O are on one
        # fragment and the methyl C is on the other.
        out = infer_frag_fallback_h_migration(
            pre_xyz=pre, post_xyz=post,
            uni_mol=sp.mol, split_bonds=[(c_carbonyl, o_oh)],
            multi_species=None, label='success',
        )
        self.assertIsNotNone(out)
        self.assertEqual(out['h_idx'], h_oh)
        self.assertEqual(out['donor'], o_oh)
        self.assertEqual(out['acceptor'], c_methyl)
        self.assertEqual(out['source'], 'frag_inferred')

    def test_s5_competing_rival_returns_none(self):
        """When a non-acceptor heavy atom in the acceptor's fragment
        sits closer to the migrated H than ``0.95 × sbl(rival, H)``,
        S5 must reject the inference."""
        from arc.job.adapters.ts.linear_utils.local_geometry import (
            infer_frag_fallback_h_migration,
        )
        from arc.species import ARCSpecies
        # Build a CCN reactant.  Cut C-N to create two fragments.
        # The N side has only the N (one heavy atom).  The C side has
        # two C atoms — one of them is the donor of the migrating H.
        # Place the migrated H so it ends up *closer to a non-acceptor
        # carbon* than its acceptor.  S5 must catch this.
        sp = ARCSpecies(label='ccn', smiles='CCN')
        symbols = sp.get_xyz()['symbols']
        # We don't actually need a chemically sensible geometry here:
        # just one where the post-migration H lands between two heavy
        # atoms that are *both* in the acceptor fragment, with the
        # rival closer than the chosen acceptor.
        # The key is: S5 considers the acceptor's fragment.  If the
        # acceptor candidate set includes more than one heavy atom and
        # one of them is too close to the H, S5 fails.
        # We construct this directly with a stub mol.
        class _A:
            def __init__(self, sym, bonds=None):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds or {}
        c0 = _A('C'); c1 = _A('C'); n = _A('N'); h = _A('H')
        c0.bonds = {h: 1}; h.bonds = {c0: 1}
        # No bond from the H side to either acceptor candidate; both
        # acceptors are in a separate fragment.
        class _M:
            atoms = [c0, c1, n, h]
        pre = {
            'symbols': ('C', 'C', 'N', 'H'),
            'isotopes': (12, 12, 14, 1),
            'coords': ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0),
                        (4.0, 0.0, 0.0), (1.0, 0.5, 0.0)),
        }
        post = {
            'symbols': ('C', 'C', 'N', 'H'),
            'isotopes': (12, 12, 14, 1),
            # H moved to be very close to N — but C1 is also close.
            'coords': ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0),
                        (3.5, 0.0, 0.0), (3.05, 0.0, 0.0)),
        }
        # Cut nothing related to (0,1) so the C0 fragment ends up with
        # only itself + the H, and the {C1, N} fragment is the
        # acceptor side.  No split bonds means a single fragment, so
        # use a non-existent cut for the test:
        out = infer_frag_fallback_h_migration(
            pre_xyz=pre, post_xyz=post,
            uni_mol=_M(), split_bonds=[],  # zero cut → single fragment
            multi_species=None, label='S5-rival',
        )
        self.assertIsNone(out)


class TestXyzWithCoords(unittest.TestCase):
    """``_xyz_with_coords`` returns a new XYZ dict with replaced coords and
    preserves the original symbols/isotopes tuple."""

    def test_returns_new_dict_with_replaced_coords(self):
        """The returned dict carries the new coords and the original symbols."""
        xyz = {
            'symbols': ('C', 'H'),
            'isotopes': (12, 1),
            'coords': ((0.0, 0.0, 0.0), (1.09, 0.0, 0.0)),
        }
        new_coords = np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
        out = _xyz_with_coords(xyz, new_coords)
        self.assertEqual(out['symbols'], xyz['symbols'])
        self.assertEqual(out['isotopes'], xyz['isotopes'])
        self.assertEqual(out['coords'], ((0.5, 0.0, 0.0), (1.5, 0.0, 0.0)))
        # Original input is not mutated.
        self.assertEqual(xyz['coords'], ((0.0, 0.0, 0.0), (1.09, 0.0, 0.0)))


class TestHeavyNeighbors(unittest.TestCase):
    """``_heavy_neighbors`` returns only non-H graph neighbors."""

    def test_methanol_heavy_neighbors_of_carbon(self):
        """Methanol's C has exactly one heavy (O) neighbor."""
        sp = ARCSpecies(label='methanol', smiles='CO')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        c_idx = symbols.index('C')
        o_idx = symbols.index('O')
        nbrs = _heavy_neighbors(sp.mol, c_idx, symbols)
        self.assertEqual(nbrs, [o_idx])


class TestHNeighbors(unittest.TestCase):
    """``_h_neighbors`` returns only H graph neighbors."""

    def test_methane_h_neighbors_of_carbon(self):
        """Methane's C has exactly four H neighbors."""
        sp = ARCSpecies(label='CH4', smiles='C')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        c_idx = symbols.index('C')
        h_idxs = _h_neighbors(sp.mol, c_idx, symbols)
        expected = [i for i, s in enumerate(symbols) if s == 'H']
        self.assertEqual(sorted(h_idxs), sorted(expected))
        self.assertEqual(len(h_idxs), 4)


class TestArcCostViaRepairInternalReactiveCh2(unittest.TestCase):
    """Validate the nested ``_arc_cost`` sticky pairing by feeding an input
    whose H directions make the *identity* pairing clearly cheaper than the
    swapped pairing, then checking that each input H ends up near its
    already-closer target slot after the repair."""

    def test_sticky_pairing_identity_wins(self):
        """H_0 sits slightly above +n_hat and H_1 slightly below; both
        bond lengths are ~1.09 Å and the two heavy neighbors lie along
        ±x (so ``w_hat = +x̂``).  Target directions are
        ``-x̂/√3 ± √(2/3) n̂`` where ``n̂ = ẑ``.  With H_0 biased to +z,
        the identity pairing (H_0 → target0, H_1 → target1) has lower
        arc cost than the swap, so H_0 should end up in the +z target
        slot and H_1 in the -z slot."""
        # Two heavy neighbors along ±x.  Center C at origin.
        # Two H atoms at ±z with a small backside tilt into -x, so the
        # direction from the H to the center already matches the target
        # directions reasonably well.
        symbols = ('C', 'C', 'C', 'H', 'H')
        coords = [
            (0.0, 0.0, 0.0),        # center (C, index 0)
            (1.5, 0.0, 0.0),        # heavy neighbor (+x)
            (-1.5, 0.0, 0.0),       # heavy neighbor (-x)
            (-0.3, 0.0, 1.05),      # H_0 — toward +z, slightly backside
            (-0.3, 0.0, -1.05),     # H_1 — toward -z, slightly backside
        ]
        xyz = {
            'symbols': symbols,
            'isotopes': (12, 12, 12, 1, 1),
            'coords': tuple(coords),
        }
        # Build a minimal Molecule with the necessary bond graph.
        class _A:
            def __init__(self, sym, bonds=None):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds or {}
        c0 = _A('C'); c1 = _A('C'); c2 = _A('C'); h0 = _A('H'); h1 = _A('H')
        c0.bonds = {c1: 1, c2: 1, h0: 1, h1: 1}
        c1.bonds = {c0: 1}
        c2.bonds = {c0: 1}
        h0.bonds = {c0: 1}; h1.bonds = {c0: 1}
        class _M:
            atoms = [c0, c1, c2, h0, h1]
        out = repair_internal_reactive_ch2(
            xyz, center_idx=0, heavy_nbr_indices=[1, 2], h_indices=[3, 4])
        # H_0 should end up at +z side; H_1 at -z side.
        h0_post = np.array(out['coords'][3])
        h1_post = np.array(out['coords'][4])
        self.assertGreater(h0_post[2], 0.0)
        self.assertLess(h1_post[2], 0.0)
        # Bond lengths preserved (±fp tolerance).
        self.assertAlmostEqual(float(np.linalg.norm(h0_post)), 1.09, places=1)
        self.assertAlmostEqual(float(np.linalg.norm(h1_post)), 1.09, places=1)


if __name__ == '__main__':
    unittest.main()
