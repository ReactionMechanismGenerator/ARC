#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.heuristics module
"""

import os
import shutil
import unittest

import numpy as np

from arc.common import ARC_PATH, almost_equal_coords, get_single_bond_length
from arc.job.adapters.ts.linear import (BASE_WEIGHT_GRID,
                                        HAMMOND_DELTA,
                                        LinearAdapter,
                                        _FAMILY_POSTPROCESSORS,
                                        _FAMILY_VALIDATORS,
                                        _clip01,
                                        _find_split_bonds_by_fragmentation,
                                        _get_all_referenced_atoms,
                                        _get_all_zmat_rows,
                                        _postprocess_generic,
                                        _postprocess_ts_guess,
                                        _stretch_bond,
                                        _validate_h_migration,
                                        _validate_ts_guess,
                                        average_zmat_params,
                                        get_near_attack_xyz,
                                        get_r_constraints,
                                        get_rxn_weight,
                                        get_weight_grid,
                                        has_inward_migrating_group_h,
                                        interp_dihedral_deg,
                                        interpolate,
                                        interpolate_addition,
                                        interpolate_isomerization,
                                        )
from arc.mapping.driver import map_rxn
from arc.reaction import ARCReaction
from arc.species.converter import compare_zmats, order_mol_by_atom_map, order_xyz_by_atom_map, str_to_xyz, xyz_to_str, zmat_from_xyz
from arc.species.species import ARCSpecies, colliding_atoms


def assert_h_migration_quality(test_case: unittest.TestCase,
                               ts_xyz: dict,
                               max_detached_h: int = 1,
                               attach_tol: float = 1.05,
                               min_contact_frac: float = 0.85,
                               ) -> None:
    """
    Assert geometry quality of a TS guess for an H-migration reaction.

    Checks:
      1. At most *max_detached_h* hydrogens are farther than ``sbl * attach_tol``
         from their nearest heavy atom (these are the migrating H's).
      2. No atom pair involving at least one H is closer than ``sbl * min_contact_frac``.

    Args:
        test_case: The ``unittest.TestCase`` instance (for calling ``fail``).
        ts_xyz: XYZ coordinate dictionary of the TS guess.
        max_detached_h: Maximum number of H atoms allowed to be detached.
        attach_tol: Fraction of single-bond length above which an H is "detached".
        min_contact_frac: Fraction of single-bond length below which an H pair is a close contact.
    """
    symbols = ts_xyz['symbols']
    coords = np.array(ts_xyz['coords'], dtype=float)
    n = len(symbols)

    # --- Check 1: detached H count ---
    detached = []
    for i in range(n):
        if symbols[i] != 'H':
            continue
        min_dist = float('inf')
        nearest_heavy = None
        for j in range(n):
            if j == i or symbols[j] == 'H':
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < min_dist:
                min_dist = d
                nearest_heavy = j
        if nearest_heavy is not None:
            sbl = get_single_bond_length('H', symbols[nearest_heavy])
            if min_dist > sbl * attach_tol:
                detached.append((i, nearest_heavy, min_dist, sbl))

    if len(detached) > max_detached_h:
        lines = [f'Expected at most {max_detached_h} detached H, got {len(detached)}:']
        for h_i, heavy_j, d, sbl in detached:
            lines.append(f'  H[{h_i}] -> {symbols[heavy_j]}[{heavy_j}] = {d:.3f} A '
                         f'(sbl={sbl:.3f}, ratio={d / sbl:.3f})')
        lines.append(f'TS guess:\n{xyz_to_str(ts_xyz)}')
        test_case.fail('\n'.join(lines))

    # --- Check 2: H close contacts ---
    for i in range(n):
        for j in range(i + 1, n):
            if symbols[i] != 'H' and symbols[j] != 'H':
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            sbl = get_single_bond_length(symbols[i], symbols[j])
            if d < sbl * min_contact_frac:
                test_case.fail(
                    f'H close contact: {symbols[i]}[{i}]-{symbols[j]}[{j}] = {d:.3f} A '
                    f'(sbl={sbl:.3f}, min={sbl * min_contact_frac:.3f})\n'
                    f'TS guess:\n{xyz_to_str(ts_xyz)}')


def assert_unique_guesses(test_case: unittest.TestCase,
                          ts_xyzs: list,
                          ) -> None:
    """
    Assert that all TS guesses in the list are pairwise distinct.

    Uses ``almost_equal_coords`` (the same function used by the production
    deduplication logic) to compare every pair.  Fails with a descriptive
    message if any two guesses are near-duplicates.
    """
    for i in range(len(ts_xyzs)):
        for j in range(i + 1, len(ts_xyzs)):
            if almost_equal_coords(ts_xyzs[i], ts_xyzs[j]):
                test_case.fail(
                    f'Guesses {i + 1} and {j + 1} are near-duplicates:\n'
                    f'Guess {i + 1}:\n{xyz_to_str(ts_xyzs[i])}\n'
                    f'Guess {j + 1}:\n{xyz_to_str(ts_xyzs[j])}')


class TestHeuristicsAdapter(unittest.TestCase):
    """
    Contains unit tests for the HeuristicsAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None

        cls.rxn_1 = ARCReaction(r_species=[ARCSpecies(label='CPD', smiles='C1C=CC=C1',
                                                      xyz="""C      -1.11689933   -0.16076292   -0.17157587
                                                             C      -0.34122713    1.12302797   -0.12498608
                                                             C       0.95393962    0.86179733    0.10168911
                                                             C       1.14045506   -0.56033684    0.22004768
                                                             C      -0.03946631   -1.17782376    0.06650470
                                                             H      -1.58827673   -0.30386166   -1.14815401
                                                             H      -1.87502410   -0.19463481    0.61612857
                                                             H      -0.77193310    2.10401684   -0.25572143
                                                             H       1.74801386    1.58807889    0.18578522
                                                             H       2.09208098   -1.03534789    0.40412258
                                                             H      -0.20166282   -2.24415315    0.10615953""")],
                                p_species=[ARCSpecies(label='C5_carbene', adjlist="""1  C u0 p1 c0 {2,S} {6,S}
                                                                                     2  C u0 p0 c0 {1,S} {3,D} {7,S}
                                                                                     3  C u0 p0 c0 {2,D} {4,S} {8,S}
                                                                                     4  C u0 p0 c0 {3,S} {5,D} {9,S}
                                                                                     5  C u0 p0 c0 {4,D} {10,S} {11,S}
                                                                                     6  H u0 p0 c0 {1,S}
                                                                                     7  H u0 p0 c0 {2,S}
                                                                                     8  H u0 p0 c0 {3,S}
                                                                                     9  H u0 p0 c0 {4,S}
                                                                                     10 H u0 p0 c0 {5,S}
                                                                                     11 H u0 p0 c0 {5,S}""",
                                                      xyz="""C       2.62023459    0.49362130   -0.23013873
                                                             C       1.48006570   -0.33866786   -0.38699247
                                                             C       1.53457595   -1.45115429   -1.13132450
                                                             C       0.40179762   -2.32741928   -1.31937443
                                                             C       0.45595744   -3.43865596   -2.06277224
                                                             H       3.47507694    1.11901971   -0.11163109
                                                             H       0.56454036   -0.04212124    0.11659958
                                                             H       2.46516705   -1.72493574   -1.62516589
                                                             H      -0.53390611   -2.06386676   -0.83047533
                                                             H      -0.42088759   -4.06846526   -2.17670487
                                                             H       1.36205133   -3.75009763   -2.57288841""")])

        cls.rxn_2 = ARCReaction(r_species=[ARCSpecies(label='CCONO', smiles='CCON=O',
                                                      xyz="""C      -1.36894499    0.07118059   -0.24801399
                                                             C      -0.01369535    0.17184136    0.42591278
                                                             O      -0.03967083   -0.62462610    1.60609048
                                                             N       1.23538512   -0.53558048    2.24863846
                                                             O       1.25629155   -1.21389295    3.27993827
                                                             H      -2.16063255    0.41812452    0.42429392
                                                             H      -1.39509985    0.66980796   -1.16284741
                                                             H      -1.59800183   -0.96960842   -0.49986392
                                                             H       0.19191326    1.21800574    0.68271847
                                                             H       0.76371340   -0.19234475   -0.25650067""")],
                                p_species=[ARCSpecies(label='CCNO2', smiles='CC[N+](=O)[O-]',
                                                      xyz="""C      -1.12362739   -0.04664655   -0.08575959
                                                             C       0.24488022   -0.51587553    0.36119196
                                                             N       0.57726975   -1.77875156   -0.37104243
                                                             O       1.16476543   -1.66382529   -1.45384186
                                                             O       0.24561669   -2.84385320    0.16410116
                                                             H      -1.87655344   -0.80826847    0.13962125
                                                             H      -1.14729169    0.14493421   -1.16405294
                                                             H      -1.41423043    0.87863077    0.42354512
                                                             H       1.02430791    0.21530309    0.12674144
                                                             H       0.27058353   -0.73979548    1.43184405""")])

    def test_average_zmat_params(self):
        """Test the average_zmat_params() function."""
        zmat_1 = {'symbols': ('H', 'H'),
                  'coords': ((None, None, None),
                             ('R_1_0', None, None)),
                  'vars': {'R_1_0': 0.7},
                  'map': {0: 0, 1: 1}}
        zmat_2 = {'symbols': ('H', 'H'),
                  'coords': ((None, None, None),
                             ('R_1_0', None, None)),
                  'vars': {'R_1_0': 1.3},
                  'map': {0: 0, 1: 1}}
        expected_zmat = {'symbols': ('H', 'H'),
                         'coords': ((None, None, None),
                                    ('R_1_0', None, None)),
                         'vars': {'R_1_0': 1.0},
                         'map': {0: 0, 1: 1}}
        zmat = average_zmat_params(zmat_1, zmat_2)
        self.assertTrue(compare_zmats(zmat, expected_zmat))

        expected_zmat = {'symbols': ('H', 'H'),
                         'coords': ((None, None, None),
                                    ('R_1_0', None, None)),
                         'vars': {'R_1_0': 0.85},
                         'map': {0: 0, 1: 1}}
        zmat = average_zmat_params(zmat_1, zmat_2, weight=0.25)
        self.assertTrue(compare_zmats(zmat, expected_zmat))
        zmat = average_zmat_params(zmat_2, zmat_1, weight=0.75)
        self.assertTrue(compare_zmats(zmat, expected_zmat))

        zmat_1 = {'symbols': ('C', 'N', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2|4_0|0', 'A_2|4_0|0_1|1', None),
                             ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_3_1_0_2'),
                             ('R_2|4_0|0', 'A_2|4_0|0_1|1', 'D_4_0_1_3'), ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'),
                             ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_6_1_0_5')),
                  'vars': {'R_1_0': 1.451965854148702, 'D_3_1_0_2': 60.83821034525936,
                           'D_4_0_1_3': 301.30263742432356, 'R_5_0': 1.0936965384360282,
                           'A_5_0_1': 110.59878027260544, 'D_5_0_1_4': 239.76779188408136,
                           'D_6_1_0_5': 65.17113681053117, 'R_2|4_0|0': 1.0935188594180785,
                           'R_3|6_1|1': 1.019169330302324, 'A_2|4_0|0_1|1': 110.20495980110817,
                           'A_3|6_1|1_0|0': 109.41187648524644},
                  'map': {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4, 6: 6}}
        zmat_2 = {'symbols': ('C', 'N', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2|4_0|0', 'A_2|4_0|0_1|1', None),
                             ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_3_1_0_2'),
                             ('R_2|4_0|0', 'A_2|4_0|0_1|1', 'D_4_0_1_3'), ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'),
                             ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_6_1_0_5')),
                  'vars': {'R_1_0': 1.2, 'D_3_1_0_2': 50,
                           'D_4_0_1_3': 250, 'R_5_0': 1.0936965384360282,
                           'A_5_0_1': 110.59878027260544, 'D_5_0_1_4': 239.76779188408136,
                           'D_6_1_0_5': 120, 'R_2|4_0|0': 1.0935188594180785,
                           'R_3|6_1|1': 1.6, 'A_2|4_0|0_1|1': 110.20495980110817,
                           'A_3|6_1|1_0|0': 109.41187648524644},
                  'map': {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4, 6: 6}}
        expected_zmat = {'symbols': ('C', 'N', 'H', 'H', 'H', 'H', 'H'),
                         'coords': ((None, None, None), ('R_1_0', None, None), ('R_2|4_0|0', 'A_2|4_0|0_1|1', None),
                                    ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_3_1_0_2'), ('R_2|4_0|0', 'A_2|4_0|0_1|1', 'D_4_0_1_3'),
                                    ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_6_1_0_5')),
                         'vars': {'R_1_0': 1.3259829270743508,
                                  'D_3_1_0_2': 55.419105172629685,
                                  'D_4_0_1_3': -84.34868128783822,
                                  'R_5_0': 1.0936965384360282,
                                  'A_5_0_1': 110.59878027260544,
                                  'D_5_0_1_4': -120.23220811591864,
                                  'D_6_1_0_5': 92.58556840526558,
                                  'R_2|4_0|0': 1.0935188594180785,
                                  'R_3|6_1|1': 1.309584665151162, 'A_2|4_0|0_1|1': 110.20495980110817,
                                  'A_3|6_1|1_0|0': 109.41187648524644},
                         'map': {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4, 6: 6}}
        zmat = average_zmat_params(zmat_1, zmat_2)
        self.assertTrue(compare_zmats(zmat, expected_zmat))

        reactive_indices = {1}
        zmat_reactive = average_zmat_params(zmat_1, zmat_2, reactive_xyz_indices=reactive_indices)
        self.assertIsNotNone(zmat_reactive)
        self.assertAlmostEqual(zmat_reactive['vars']['R_1_0'], 1.3259829270743508, places=3)
        # R_3|6_1|1 references atoms [3, 6, 1, 1]; atom 1 is reactive → INTERPOLATED.
        self.assertAlmostEqual(zmat_reactive['vars']['R_3|6_1|1'], 1.309584665151162, places=3)
        self.assertAlmostEqual(zmat_reactive['vars']['R_5_0'], 1.0936965384360282, places=3)
        # D_3_1_0_2 references atoms [3, 1, 0, 2]; atom 1 is reactive → INTERPOLATED.
        self.assertAlmostEqual(zmat_reactive['vars']['D_3_1_0_2'], 55.419105172629685, places=3)

    def test_average_zmat_params_dihedral_wraparound(self):
        """
        Test 1: dihedral interpolation crosses the ±180° boundary via the shortest path.

        175° and -175° are only 10° apart across the ±180° seam.  At weight=0.5 the
        midpoint must be ±180°, NOT 0° (which would be the naïve linear average).
        """
        zmat_1 = {'symbols': ('C', 'C', 'C', 'H'),
                  'coords': ((None, None, None),
                             ('R_1_0', None, None),
                             ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0')),
                  'vars': {'R_1_0': 1.54, 'R_2_1': 1.54, 'A_2_1_0': 109.5,
                           'R_3_2': 1.09, 'A_3_2_1': 109.5, 'D_3_2_1_0': 175.0},
                  'map': {0: 0, 1: 1, 2: 2, 3: 3}}
        zmat_2 = {'symbols': ('C', 'C', 'C', 'H'),
                  'coords': ((None, None, None),
                             ('R_1_0', None, None),
                             ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0')),
                  'vars': {'R_1_0': 1.54, 'R_2_1': 1.54, 'A_2_1_0': 109.5,
                           'R_3_2': 1.09, 'A_3_2_1': 109.5, 'D_3_2_1_0': -175.0},
                  'map': {0: 0, 1: 1, 2: 2, 3: 3}}
        result = average_zmat_params(zmat_1, zmat_2, weight=0.5)
        self.assertIsNotNone(result)
        d_interp = result['vars']['D_3_2_1_0']
        self.assertAlmostEqual(abs(d_interp), 180.0, places=2, msg=f'Expected ±180°, got {d_interp}°.')
        self.assertNotAlmostEqual(d_interp, 0.0, places=1, msg='Dihedral wrapped to 0° — shortest-path arithmetic failed.')

    def test_average_zmat_params_angle_singularity_clamping(self):
        """
        Test 2: valence angles are clamped to [1°, 179°] to avoid Z-matrix singularities.

        An input angle of 185° is unphysical but can arise from poor initial geometries.
        The interpolated result must be clamped to 179° rather than landing at 180° or above.
        """
        zmat_1 = {'symbols': ('C', 'C', 'H'),
                  'coords': ((None, None, None),
                             ('R_1_0', None, None),
                             ('R_2_0', 'A_2_0_1', None)),
                  'vars': {'R_1_0': 1.54, 'R_2_0': 1.09, 'A_2_0_1': 175.0},
                  'map': {0: 0, 1: 1, 2: 2}}
        zmat_2 = {'symbols': ('C', 'C', 'H'),
                  'coords': ((None, None, None),
                             ('R_1_0', None, None),
                             ('R_2_0', 'A_2_0_1', None)),
                  'vars': {'R_1_0': 1.54, 'R_2_0': 1.09, 'A_2_0_1': 185.0},
                  'map': {0: 0, 1: 1, 2: 2}}
        result = average_zmat_params(zmat_1, zmat_2, weight=0.5)
        self.assertIsNotNone(result)
        a_interp = result['vars']['A_2_0_1']
        # Raw linear average would be 180°; must be clamped to 179°.
        self.assertLessEqual(a_interp, 179.0,
                             msg=f'Angle {a_interp}° exceeded the 179° singularity cap.')
        self.assertAlmostEqual(a_interp, 179.0, places=6,
                               msg=f'Expected clamp to exactly 179°, got {a_interp}°.')
        # Also verify the lower bound: interpolating toward a near-zero angle.
        zmat_2_low = {'symbols': ('C', 'C', 'H'),
                      'coords': ((None, None, None),
                                 ('R_1_0', None, None),
                                 ('R_2_0', 'A_2_0_1', None)),
                      'vars': {'R_1_0': 1.54, 'R_2_0': 1.09, 'A_2_0_1': -5.0},
                      'map': {0: 0, 1: 1, 2: 2}}
        result_low = average_zmat_params(zmat_1, zmat_2_low, weight=1.0)
        self.assertIsNotNone(result_low)
        a_low = result_low['vars']['A_2_0_1']
        self.assertGreaterEqual(a_low, 1.0, msg=f'Angle {a_low}° fell below the 1° singularity floor.')

    def test_average_zmat_params_schema_mismatch(self):
        """
        Test 3: mismatched variable schemas return None without raising exceptions.

        zmat_1 carries a real dihedral 'D_4_3_2_1'; zmat_2 replaces it with the
        dummy-dihedral 'DX_4_3_2_1'.  The key sets differ, so check_ordered_zmats
        must fail and the function must return None gracefully.
        """
        zmat_1 = {'symbols': ('C', 'C', 'C', 'C', 'H'),
                  'coords': ((None, None, None),
                             ('R_1_0', None, None),
                             ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'),
                             ('R_4_3', 'A_4_3_2', 'D_4_3_2_1')),
                  'vars': {'R_1_0': 1.54, 'R_2_1': 1.54, 'A_2_1_0': 109.5,
                           'R_3_2': 1.54, 'A_3_2_1': 109.5, 'D_3_2_1_0': 60.0,
                           'R_4_3': 1.09, 'A_4_3_2': 109.5, 'D_4_3_2_1': 180.0},
                  'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}}
        zmat_2 = {'symbols': ('C', 'C', 'C', 'C', 'H'),
                  'coords': ((None, None, None),
                             ('R_1_0', None, None),
                             ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'),
                             ('R_4_3', 'A_4_3_2', 'DX_4_3_2_1')),   # renamed key
                  'vars': {'R_1_0': 1.54, 'R_2_1': 1.54, 'A_2_1_0': 109.5,
                           'R_3_2': 1.54, 'A_3_2_1': 109.5, 'D_3_2_1_0': 60.0,
                           'R_4_3': 1.09, 'A_4_3_2': 109.5, 'DX_4_3_2_1': 180.0},
                  'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}}
        # Must not raise — must return None
        result = average_zmat_params(zmat_1, zmat_2, weight=0.5)
        self.assertIsNone(result, msg='Schema mismatch (D_ vs DX_ key rename) should return None, '
                                      'not crash or return a partial result.')

    def test_order_mol_by_atom_map_reordering(self):
        """
        Test 1: order_mol_by_atom_map reorders the product molecule's atom list to match reactant indexing.

        For ethanol (CCO, heavy atoms C-C-O), pass an atom_map that swaps atom 0 (C) and
        atom 2 (O), and verify that the returned molecule's atom 0 is oxygen.
        """
        # Ethanol: atom 0 = C, atom 1 = C, atom 2 = O, atoms 3-8 = H
        p_mol = ARCSpecies(label='ethanol', smiles='CCO').mol
        n = len(p_mol.atoms)
        self.assertEqual(p_mol.atoms[0].symbol, 'C')
        self.assertEqual(p_mol.atoms[2].symbol, 'O')

        # atom_map[reactant_i] = product_i: swap heavy atoms 0 and 2, leave all others.
        # After mapping: reactant position 0 → product atom 2 (O), position 2 → product atom 0 (C).
        atom_map = list(range(n))
        atom_map[0], atom_map[2] = 2, 0   # swap C↔O

        mapped = order_mol_by_atom_map(p_mol, atom_map)

        # Atom count must be unchanged.
        self.assertEqual(len(mapped.atoms), n)
        # Position 0 in the mapped molecule must now be oxygen.
        self.assertEqual(mapped.atoms[0].symbol, 'O', msg='Mapped atom 0 should be O after swapping indices 0 and 2.')
        # Position 2 should now be carbon.
        self.assertEqual(mapped.atoms[2].symbol, 'C', msg='Mapped atom 2 should be C after swapping indices 0 and 2.')
        # The original p_mol must not be mutated.
        self.assertEqual(p_mol.atoms[0].symbol, 'C', msg='Original p_mol must not be mutated by order_mol_by_atom_map.')
        # Molecular formula must be preserved.
        self.assertEqual(mapped.get_formula(), p_mol.get_formula())
        # Bonds must still be valid (ethanol O is bonded to a C).
        o_atom = mapped.atoms[0]
        bonded_symbols = sorted(nbr.symbol for nbr in o_atom.bonds.keys())
        self.assertIn('C', bonded_symbols, msg='Oxygen in mapped molecule should still be bonded to a carbon.')

    def test_order_mol_by_atom_map_length_mismatch_raises(self):
        """Test that a mismatched atom_map length raises ValueError."""
        p_mol = ARCSpecies(label='ethanol', smiles='CCO').mol
        bad_map = list(range(len(p_mol.atoms) - 1))   # one element short
        with self.assertRaises(ValueError):
            order_mol_by_atom_map(p_mol, bad_map)

    def test_type_p_uses_product_topology(self):
        """
        Test 2: the Type-P chimera Z-matrix is built from product bond topology.

        Use the intra-H migration CCO[O] → [CH2]COO reaction.  In this reaction the
        breaking bond is C-H (index pair involving atom 0 and an H) and the forming bond
        involves the terminal oxygen.  After the fix, the Z-matrix built for Type P uses
        mapped_p_mol (product bonds), so its connectivity graph differs from the one built
        by Type R (which uses r_mol, reactant bonds).

        Concretely: build r_zmat and p_zmat for the same op_xyz using r_mol and
        mapped_p_mol respectively.  They should differ in at least one variable because
        the reactant and product have different bond orders (C–OO vs C–O–O single bonds
        shift).
        """
        r_xyz_str = """C      -1.05582103   -0.03329574   -0.10080257
C       0.41792695    0.17831205    0.21035514
O       1.19234020   -0.65389683   -0.61111443
O       2.44749684   -0.41401220   -0.28381363
H      -1.33614002   -1.09151783    0.08714882
H      -1.25953618    0.21489046   -1.16411897
H      -1.67410396    0.62341419    0.54699514
H       0.59566350   -0.06437686    1.28256640
H       0.67254676    1.24676329    0.02676370"""
        p_xyz_str = """C      -1.40886397    0.22567351   -0.37379668
C       0.06280787    0.04097694   -0.38515682
O       0.44130326   -0.57668419    0.84260864
O       1.89519755   -0.66754203    0.80966180
H      -1.87218376    0.90693511   -1.07582340
H      -2.03646287   -0.44342165    0.20255768
H       0.35571681   -0.60165457   -1.22096147
H       0.56095122    1.01161503   -0.47393734
H       2.05354047   -0.10415729    1.58865243"""
        r = ARCSpecies(label='R', smiles='CCO[O]', xyz=r_xyz_str)
        p = ARCSpecies(label='P', smiles='[CH2]COO', xyz=p_xyz_str)
        rxn = ARCReaction(r_species=[r], p_species=[p])

        atom_map = map_rxn(rxn=rxn, product_dict_index_to_try=0)
        self.assertIsNotNone(atom_map, 'map_rxn returned None — cannot run topology test.')

        r_mol = rxn.r_species[0].mol
        p_mol = rxn.p_species[0].mol
        op_xyz = order_xyz_by_atom_map(xyz=rxn.p_species[0].get_xyz(), atom_map=atom_map)
        mapped_p_mol = order_mol_by_atom_map(p_mol, atom_map)

        # The mapped product molecule must have the same atom count.
        self.assertEqual(len(mapped_p_mol.atoms), len(r_mol.atoms))

        # Build Z-matrices from op_xyz using the reactant mol (old behaviour) and the
        # mapped product mol (new behaviour).
        r_zmat_from_op = zmat_from_xyz(xyz=op_xyz, mol=r_mol, consolidate=False)
        p_zmat_from_op = zmat_from_xyz(xyz=op_xyz, mol=mapped_p_mol, consolidate=False)

        self.assertIsNotNone(r_zmat_from_op, 'zmat_from_xyz with r_mol returned None.')
        self.assertIsNotNone(p_zmat_from_op, 'zmat_from_xyz with mapped_p_mol returned None.')

        # The bond topology may differ between reactant and product (one C-H bond breaks,
        # one O-H bond forms).  When mol is provided, zmat_from_xyz uses the mol's graph to
        # choose which atoms serve as R/A/D references.  If the two mols differ in
        # connectivity, the 'coords' template (reference atom choices) will differ for at
        # least one row, producing different variable names or atom-index assignments.
        r_coords = r_zmat_from_op['coords']
        p_coords = p_zmat_from_op['coords']
        self.assertNotEqual(r_coords, p_coords,
                            msg='Type-P Z-matrix built with mapped_p_mol should have different atom-reference '
                                'choices from the Type-R Z-matrix built with r_mol, because the two molecules '
                                'have different bond topologies.')

    def test_get_all_zmat_rows(self):
        """Test the _get_all_zmat_rows() helper returns every packed row index."""
        # Simple (non-consolidated) variables return a single-element list.
        self.assertEqual(_get_all_zmat_rows('R_1_0'), [1])
        self.assertEqual(_get_all_zmat_rows('A_3_1_0'), [3])
        self.assertEqual(_get_all_zmat_rows('D_4_1_0_2'), [4])
        self.assertEqual(_get_all_zmat_rows('DX_5_1_0_2'), [5])
        # Consolidated variables (multiple rows packed with '|') return all row indices.
        self.assertEqual(_get_all_zmat_rows('R_2|4_0|0'), [2, 4])
        self.assertEqual(_get_all_zmat_rows('A_2|4_0|0_1|1'), [2, 4])
        self.assertEqual(_get_all_zmat_rows('R_3|6_1|1'), [3, 6])
        # Three-way consolidation.
        self.assertEqual(_get_all_zmat_rows('R_1|3|5_0|0|0'), [1, 3, 5])
        # Unparseable names return an empty list (not None, not an exception).
        self.assertEqual(_get_all_zmat_rows('R'), [])
        self.assertEqual(_get_all_zmat_rows('notavar'), [])
        self.assertEqual(_get_all_zmat_rows(''), [])

    def test_get_all_referenced_atoms(self):
        """Test that _get_all_referenced_atoms returns every atom index in a Z-matrix variable name."""
        self.assertEqual(_get_all_referenced_atoms('R_1_0'), [1, 0])
        self.assertEqual(_get_all_referenced_atoms('A_3_1_0'), [3, 1, 0])
        self.assertEqual(_get_all_referenced_atoms('D_4_3_0_2'), [4, 3, 0, 2])
        self.assertEqual(_get_all_referenced_atoms('DX_5_1_0_2'), [5, 1, 0, 2])
        self.assertEqual(_get_all_referenced_atoms('R_2|4_0|0'), [2, 4, 0, 0])
        self.assertEqual(_get_all_referenced_atoms('A_2|4_0|0_1|1'), [2, 4, 0, 0, 1, 1])
        self.assertEqual(_get_all_referenced_atoms('R'), [])
        self.assertEqual(_get_all_referenced_atoms(''), [])

    def test_family_dispatch_tables_populated(self):
        """Test that the family dispatch tables are populated for known families."""
        self.assertIn('intra_H_migration', _FAMILY_POSTPROCESSORS)
        self.assertIn('intra_H_migration', _FAMILY_VALIDATORS)
        self.assertIn('1,2_shiftC', _FAMILY_POSTPROCESSORS)
        self.assertIn('1,2_shiftC', _FAMILY_VALIDATORS)
        self.assertIs(_FAMILY_POSTPROCESSORS['Ketoenol'],
                      _FAMILY_POSTPROCESSORS['intra_H_migration'])
        self.assertIs(_FAMILY_VALIDATORS['Ketoenol'],
                      _FAMILY_VALIDATORS['intra_H_migration'])

    def test_postprocess_dispatch_known_family(self):
        """Test that _postprocess_ts_guess dispatches to the correct family handler."""
        xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'),
               'isotopes': (12, 1, 1, 1, 1),
               'coords': ((0.0, 0.0, 0.0),
                           (0.0, 0.0, 1.09),
                           (1.03, 0.0, -0.36),
                           (-0.51, 0.89, -0.36),
                           (-0.51, -0.89, -0.36))}
        from arc.molecule import Molecule
        mol = Molecule(smiles='C')
        result_xyz, mig_hs = _postprocess_ts_guess(
            xyz, mol, forming_bonds=[], breaking_bonds=[], family=None)
        self.assertIsInstance(result_xyz, dict)
        self.assertIsInstance(mig_hs, set)
        result_xyz2, _ = _postprocess_ts_guess(
            xyz, mol, forming_bonds=[], breaking_bonds=[], family='SomeFutureFamily')
        self.assertIsInstance(result_xyz2, dict)

    def test_validate_generic_filters_always_run(self):
        """Test that generic validation filters always run regardless of family."""
        xyz_collision = {'symbols': ('C', 'C'),
                         'isotopes': (12, 12),
                         'coords': ((0.0, 0.0, 0.0), (0.01, 0.0, 0.0))}
        from arc.molecule import Molecule
        mol = Molecule(smiles='[CH2][CH2]')
        is_valid, reason = _validate_ts_guess(
            xyz_collision, set(), [], mol, label='test', family='intra_H_migration')
        self.assertFalse(is_valid)
        self.assertEqual(reason, 'colliding atoms')
        is_valid2, reason2 = _validate_ts_guess(
            xyz_collision, set(), [], mol, label='test', family='SomeFutureFamily')
        self.assertFalse(is_valid2)
        self.assertEqual(reason2, 'colliding atoms')
        is_valid3, reason3 = _validate_ts_guess(
            xyz_collision, set(), [], mol, label='test', family=None)
        self.assertFalse(is_valid3)
        self.assertEqual(reason3, 'colliding atoms')

    def test_validate_h_migration_filters(self):
        """Test that _validate_h_migration is a no-op when migrating_hs is empty."""
        xyz = {'symbols': ('C', 'C', 'H'),
               'isotopes': (12, 12, 1),
               'coords': ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.75, 0.0, 0.0))}
        from arc.molecule import Molecule
        mol = Molecule(smiles='[CH2][CH2]')
        is_valid, reason = _validate_h_migration(xyz, set(), [(0, 2)], mol, 'test')
        self.assertTrue(is_valid)
        self.assertEqual(reason, '')

    def test_postprocess_generic_no_op(self):
        """Test that _postprocess_generic returns empty migrating_hs."""
        xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'),
               'isotopes': (12, 1, 1, 1, 1),
               'coords': ((0.0, 0.0, 0.0),
                           (0.0, 0.0, 1.09),
                           (1.03, 0.0, -0.36),
                           (-0.51, 0.89, -0.36),
                           (-0.51, -0.89, -0.36))}
        from arc.molecule import Molecule
        mol = Molecule(smiles='C')
        result_xyz, mig_hs = _postprocess_generic(xyz, mol, [], [])
        self.assertEqual(mig_hs, set())
        self.assertEqual(len(result_xyz['symbols']), 5)

    def test_validate_unknown_family_skips_family_filters(self):
        """Test that unknown families skip family-specific validation filters."""
        xyz = {'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'),
               'isotopes': (12, 12, 1, 1, 1, 1, 1, 1),
               'coords': ((0.0, 0.0, 0.0),
                           (1.54, 0.0, 0.0),
                           (-0.36, 1.03, 0.0),
                           (-0.36, -0.51, 0.89),
                           (-0.36, -0.51, -0.89),
                           (1.90, 1.03, 0.0),
                           (1.90, -0.51, 0.89),
                           (1.90, -0.51, -0.89))}
        from arc.molecule import Molecule
        mol = Molecule(smiles='CC')
        is_valid, reason = _validate_ts_guess(
            xyz, set(), [], mol, label='test', family='SomeFutureFamily')
        self.assertTrue(is_valid)
        self.assertEqual(reason, '')

    def test_get_near_attack_xyz(self):
        """get_near_attack_xyz rotates along the donor→acceptor path to reduce forming-bond distance."""

        # Ethyl-peroxy radical (CCO[O]).  In the equilibrium geometry the
        # terminal O (atom 3) is ~3.86 Å from the CH3 hydrogens (atoms 4-6)
        # because the C-C-O-O backbone is in the anti conformation.
        # A 1,4-H-migration TS requires O3 to approach one of those H atoms.
        # get_near_attack_xyz should rotate around the C1-O2 bond to close the gap.
        r_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'),
                 'coords': ((-1.05582103, -0.03329574, -0.10080257),
                             (0.41792695,  0.17831205,  0.21035514),
                             (1.19234020, -0.65389683, -0.61111443),
                             (2.44749684, -0.41401220, -0.28381363),
                             (-1.33614002, -1.09151783,  0.08714882),
                             (-1.25953618,  0.21489046, -1.16411897),
                             (-1.67410396,  0.62341419,  0.54699514),
                             (0.59566350, -0.06437686,  1.28256640),
                             (0.67254676,  1.24676329,  0.02676370)),
                 'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1)}
        mol = ARCSpecies(label='R', smiles='CCO[O]', xyz=r_xyz).mol
        forming_bonds = [(4, 3)]  # H4 (on CH3) → O3 (terminal O)

        na_xyz = get_near_attack_xyz(r_xyz, mol, bonds=forming_bonds)

        # The output must be a valid XYZ dict with the same atoms.
        self.assertEqual(na_xyz['symbols'], r_xyz['symbols'])
        self.assertEqual(len(na_xyz['coords']), len(r_xyz['coords']))

        # Original H4···O3 distance is ~3.86 Å; after near-attack rotation it should
        # be substantially shorter (< 2.5 Å is a reasonable near-attack threshold).
        h4_before = np.array(r_xyz['coords'][4])
        o3_before  = np.array(r_xyz['coords'][3])
        h4_after  = np.array(na_xyz['coords'][4])
        o3_after   = np.array(na_xyz['coords'][3])
        dist_before = float(np.linalg.norm(h4_before - o3_before))
        dist_after  = float(np.linalg.norm(h4_after  - o3_after))
        self.assertGreater(dist_before, 3.5,
                           msg='Test setup: H4···O3 in equilibrium should be > 3.5 Å.')
        self.assertLess(dist_after, 2.5,
                        msg=f'Near-attack rotation should bring H4···O3 below 2.5 Å; got {dist_after:.2f} Å.')

        # The original xyz must not be modified (deep copy semantics).
        self.assertEqual(r_xyz['coords'][4], (-1.33614002, -1.09151783, 0.08714882))

        # Bonds that are already short (directly bonded pairs) must not elongate.
        c0_after = np.array(na_xyz['coords'][0])
        c1_after = np.array(na_xyz['coords'][1])
        self.assertAlmostEqual(float(np.linalg.norm(c0_after - c1_after)),
                               float(np.linalg.norm(np.array(r_xyz['coords'][0])
                                                    - np.array(r_xyz['coords'][1]))),
                               places=4,
                               msg='C0-C1 bond length must be unchanged by the rotation.')

        # Identity case: if donor and acceptor are directly bonded (path length 2),
        # no rotation is possible and xyz must be returned unchanged.
        direct_xyz = get_near_attack_xyz(r_xyz, mol, bonds=[(0, 1)])  # C0-C1 already bonded
        self.assertEqual(direct_xyz['coords'], r_xyz['coords'])

    def test_average_zmat_params_consolidated_reactive_classification(self):
        """Consolidated variable is reactive when ANY of its rows maps to a reactive atom.

        Variable 'R_2|4_0|0' spans rows 2 and 4.  If zmat_map[4] maps to reactive xyz
        index 3 but zmat_map[2] maps to non-reactive xyz index 0, the variable must still
        be interpolated (not frozen as spectator).
        """
        # Minimal 5-atom Z-matrix containing a consolidated bond variable R_2|4_0|0.
        # Row 0: anchor (xyz 0)
        # Row 1: (xyz 1)
        # Row 2: (xyz 2) — non-reactive per map
        # Row 3: (xyz 5) — non-reactive per map
        # Row 4: (xyz 3) — REACTIVE
        zmat_1 = {
            'symbols': ('C', 'C', 'H', 'H', 'H'),
            'coords': (
                (None, None, None),
                ('R_1_0', None, None),
                ('R_2|4_0|0', 'A_2|4_0|0_1|1', None),
                ('R_3_1', 'A_3_1_0', 'D_3_1_0_2'),
                ('R_2|4_0|0', 'A_2|4_0|0_1|1', None),
            ),
            'vars': {'R_1_0': 1.5, 'R_2|4_0|0': 1.1, 'A_2|4_0|0_1|1': 109.5,
                     'R_3_1': 1.1, 'A_3_1_0': 109.5, 'D_3_1_0_2': 120.0},
            'map': {0: 0, 1: 1, 2: 2, 3: 5, 4: 3},
        }
        zmat_2 = {
            'symbols': ('C', 'C', 'H', 'H', 'H'),
            'coords': zmat_1['coords'],
            'vars': {'R_1_0': 1.5, 'R_2|4_0|0': 1.5, 'A_2|4_0|0_1|1': 109.5,
                     'R_3_1': 1.1, 'A_3_1_0': 109.5, 'D_3_1_0_2': 120.0},
            'map': {0: 0, 1: 1, 2: 2, 3: 5, 4: 3},
        }
        # zmat_map[2]=2 (non-reactive), zmat_map[4]=3 (reactive).
        # R_2|4_0|0 spans rows 2 and 4 → row 4 is reactive → variable must be interpolated.
        reactive_indices = {3}  # xyz atom 3 is reactive
        result = average_zmat_params(zmat_1, zmat_2, weight=0.5, reactive_xyz_indices=reactive_indices)
        # Interpolated at weight=0.5: (1.1 + 1.5) / 2 = 1.3
        self.assertAlmostEqual(result['vars']['R_2|4_0|0'], 1.3, places=6,
                               msg='Consolidated variable R_2|4_0|0 should be interpolated '
                                   'because row 4 maps to reactive xyz index 3, '
                                   'even though row 2 maps to non-reactive xyz index 2.')
        # Non-reactive variable R_3_1 (row 3 → xyz 5, not in reactive_indices) should be frozen.
        self.assertAlmostEqual(result['vars']['R_3_1'], 1.1, places=6,
                               msg='Spectator variable R_3_1 should retain zmat_1 value.')

    def test_interpolate_isomerization_multi_species_guard(self):
        """interpolate_isomerization returns [] (not IndexError) for multi-species reactions."""
        r1 = ARCSpecies(label='H', smiles='[H]')
        r2 = ARCSpecies(label='CH4', smiles='C')
        p1 = ARCSpecies(label='H2', smiles='[H][H]')
        p2 = ARCSpecies(label='CH3', smiles='[CH3]')
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p1, p2])
        # Must return an empty list, not raise IndexError or any other exception.
        result = interpolate_isomerization(rxn, weight=0.5)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [],
                         msg='interpolate_isomerization should return [] for reactions '
                             'with more than one reactant or product species.')

    def test_get_rxn_weight(self):
        """Test the get_rxn_weight() function."""
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='HO2', smiles='[O]O'),
                                       ARCSpecies(label='NH', smiles='[NH]')],
                            p_species=[ARCSpecies(label='N', smiles='[N]'),
                                       ARCSpecies(label='H2O2', smiles='OO')])
        rxn_1.r_species[0].e0 = 100
        rxn_1.r_species[1].e0 = 100
        rxn_1.p_species[0].e0 = 100
        rxn_1.p_species[1].e0 = 100
        self.assertAlmostEqual(get_rxn_weight(rxn_1), 0.50, 2)
        rxn_1.r_species[0].e0 = 250
        rxn_1.r_species[1].e0 = 100
        rxn_1.p_species[0].e0 = 100
        rxn_1.p_species[1].e0 = 100
        self.assertAlmostEqual(get_rxn_weight(rxn_1), 0.30, 2)
        rxn_1.r_species[0].e0 = 100
        rxn_1.r_species[1].e0 = 100
        rxn_1.p_species[0].e0 = 100
        rxn_1.p_species[1].e0 = 250
        self.assertAlmostEqual(get_rxn_weight(rxn_1), 0.70, 2)
        rxn_1.r_species[0].e0 = 200
        rxn_1.r_species[1].e0 = 100
        rxn_1.p_species[0].e0 = 100
        rxn_1.p_species[1].e0 = 100
        self.assertAlmostEqual(get_rxn_weight(rxn_1), 0.37, 2)
        rxn_1.r_species[0].e0 = 100
        rxn_1.r_species[1].e0 = 100
        rxn_1.p_species[0].e0 = 100
        rxn_1.p_species[1].e0 = 200
        self.assertAlmostEqual(get_rxn_weight(rxn_1), 0.63, 2)
        rxn_1.r_species[0].e0 = 100
        rxn_1.r_species[1].e0 = 100
        rxn_1.p_species[0].e0 = 100
        rxn_1.p_species[1].e0 = 150
        self.assertAlmostEqual(get_rxn_weight(rxn_1), 0.57, 2)
        rxn_1.r_species[0].e0 = 150
        rxn_1.r_species[1].e0 = 100
        rxn_1.p_species[0].e0 = 100
        rxn_1.p_species[1].e0 = 100
        self.assertAlmostEqual(get_rxn_weight(rxn_1), 0.43, 2)
        rxn_1.r_species[0].e0 = 100
        rxn_1.r_species[1].e0 = 100
        rxn_1.p_species[0].e0 = 100
        rxn_1.p_species[1].e0 = 125
        self.assertAlmostEqual(get_rxn_weight(rxn_1), 0.53, 2)
        rxn_1.r_species[0].e0 = 125
        rxn_1.r_species[1].e0 = 100
        rxn_1.p_species[0].e0 = 100
        rxn_1.p_species[1].e0 = 100
        self.assertAlmostEqual(get_rxn_weight(rxn_1), 0.47, 2)
        rxn_1.r_species[0].e0 = 252.0
        rxn_1.r_species[1].e0 = 100.5
        rxn_1.p_species[0].e0 = 116.0
        rxn_1.p_species[1].e0 = 200.3
        self.assertAlmostEqual(get_rxn_weight(rxn_1), 0.45, 2)

        rxn_1.r_species[0].e0 = 100
        rxn_1.r_species[1].e0 = 100
        rxn_1.p_species[0].e0 = 100
        rxn_1.p_species[1].e0 = 150
        # If lambda is smaller, shift is larger: w = 0.5 + 50/(2*125) = 0.70 (hits w_max)
        self.assertAlmostEqual(get_rxn_weight(rxn_1, reorg_energy=125.0), 0.70, 2)
        # If lambda is larger, shift is smaller: w = 0.5 + 50/(2*500) = 0.55
        self.assertAlmostEqual(get_rxn_weight(rxn_1, reorg_energy=500.0), 0.55, 2)
        # Asymmetric lambdas: (lambda_exo, lambda_endo)
        # For endothermic (+50), lambda_endo controls: 50/(2*300)=0.0833 -> w=0.5833
        self.assertAlmostEqual(get_rxn_weight(rxn_1, reorg_energy=(250.0, 300.0)), 0.58, 2)

        rxn_1.r_species[0].e0 = 150
        rxn_1.r_species[1].e0 = 100
        rxn_1.p_species[0].e0 = 100
        rxn_1.p_species[1].e0 = 100
        # With lambda_exo=300: w = 0.5 - 50/(2*300) = 0.4167
        self.assertAlmostEqual(get_rxn_weight(rxn_1, reorg_energy=(300.0, 250.0)), 0.42, 2)

    def test_get_weight_grid_no_hammond(self):
        """Without Hammond weighting, the grid is exactly BASE_WEIGHT_GRID sorted."""
        # Use rxn_2 (isomerization); energies don't matter because include_hammond=False.
        grid = get_weight_grid(self.rxn_2, include_hammond=False)
        self.assertEqual(grid, sorted(set(round(w, 3) for w in BASE_WEIGHT_GRID)))

    def test_get_weight_grid_no_energies(self):
        """When no energies are available, Hammond estimate is skipped → only BASE_WEIGHT_GRID."""
        r = ARCSpecies(label='R', smiles='CCO[O]')
        p = ARCSpecies(label='P', smiles='[CH2]COO')
        rxn = ARCReaction(r_species=[r], p_species=[p])
        # No e0 / e_elect set → get_rxn_weight returns None → Hammond branch is skipped.
        grid = get_weight_grid(rxn, include_hammond=True)
        expected = sorted(set(round(w, 3) for w in BASE_WEIGHT_GRID))
        self.assertEqual(grid, expected)

    def test_get_weight_grid_with_energies(self):
        """With energies, Hammond weight ± delta is merged into the grid."""
        r = ARCSpecies(label='R', smiles='CCO[O]')
        p = ARCSpecies(label='P', smiles='[CH2]COO')
        r.e0 = 100.0
        p.e0 = 200.0  # +100 kJ/mol endothermic
        rxn = ARCReaction(r_species=[r], p_species=[p])
        grid = get_weight_grid(rxn, include_hammond=True)
        # get_rxn_weight with delta_e=+100 kJ/mol and default params (delta_e_sat=150):
        #   lam_endo = 150 / (2*(0.70-0.5)) = 375; w0 = 0.5 + 100/(2*375) ≈ 0.633
        w0 = get_rxn_weight(rxn)
        self.assertIsNotNone(w0)
        # Grid must be sorted.
        self.assertEqual(grid, sorted(grid))
        # Grid must be unique.
        self.assertEqual(len(grid), len(set(grid)))
        # All values in [0, 1].
        self.assertTrue(all(0.0 <= w <= 1.0 for w in grid))
        # Hammond center w0 must appear.
        self.assertIn(round(w0, 3), grid)
        # Flanking values w0±delta must also appear (or be clipped to bounds).
        expected_lo = round(max(0.0, w0 - HAMMOND_DELTA), 3)
        expected_hi = round(min(1.0, w0 + HAMMOND_DELTA), 3)
        self.assertIn(expected_lo, grid)
        self.assertIn(expected_hi, grid)
        # Base grid values must still be present.
        for w in BASE_WEIGHT_GRID:
            self.assertIn(round(w, 3), grid)

    def test_get_weight_grid_dedup(self):
        """If the Hammond weight coincides with a base-grid value, no duplicate entry is produced."""
        r = ARCSpecies(label='R', smiles='CCO[O]')
        p = ARCSpecies(label='P', smiles='[CH2]COO')
        # Thermoneutral reaction → w0 = 0.5, which is already in BASE_WEIGHT_GRID.
        r.e0 = 100.0
        p.e0 = 100.0
        rxn = ARCReaction(r_species=[r], p_species=[p])
        grid = get_weight_grid(rxn, include_hammond=True)
        # No duplicate 0.5 entries.
        self.assertEqual(len(grid), len(set(grid)))
        self.assertIn(0.5, grid)

    def test_get_weight_grid_clips_at_bounds(self):
        """Hammond-derived weights outside [0, 1] are clipped, not omitted."""
        r = ARCSpecies(label='R', smiles='CCO[O]')
        p = ARCSpecies(label='P', smiles='[CH2]COO')
        # Strongly exothermic → w0 hits w_min=0.30; w0-delta would go below 0.
        r.e0 = 400.0
        p.e0 = 100.0
        rxn = ARCReaction(r_species=[r], p_species=[p])
        grid = get_weight_grid(rxn, include_hammond=True,
                               hammond_delta=0.35)  # large delta ensures clipping
        self.assertEqual(grid, sorted(grid))
        self.assertTrue(all(0.0 <= w <= 1.0 for w in grid))

    def test_get_weight_grid_custom_params(self):
        """Custom base_grid and hammond_delta are respected."""
        r = ARCSpecies(label='R', smiles='CCO[O]')
        p = ARCSpecies(label='P', smiles='[CH2]COO')
        r.e0 = 100.0
        p.e0 = 150.0  # mildly endothermic
        rxn = ARCReaction(r_species=[r], p_species=[p])
        custom_grid = (0.4, 0.6)
        grid = get_weight_grid(rxn, include_hammond=True,
                               base_grid=custom_grid,
                               hammond_delta=0.05)
        # Custom base values must appear.
        for w in custom_grid:
            self.assertIn(round(w, 3), grid)
        # Default BASE_WEIGHT_GRID values (0.35, 0.50, 0.65) must NOT appear
        # (unless they coincide with Hammond values, which they don't at delta=0.05).
        for w in BASE_WEIGHT_GRID:
            if round(w, 3) not in [round(cw, 3) for cw in custom_grid]:
                w0 = get_rxn_weight(rxn)
                hammond_values = {round(_clip01(w0 - 0.05), 3),
                                  round(w0, 3),
                                  round(_clip01(w0 + 0.05), 3)}
                if round(w, 3) not in hammond_values:
                    self.assertNotIn(round(w, 3), grid)
        # Always sorted and unique.
        self.assertEqual(grid, sorted(grid))
        self.assertEqual(len(grid), len(set(grid)))

    def test_interpolate_dispatches_to_isomerization(self):
        """interpolate() calls interpolate_isomerization() for isomerization reactions."""
        # self.rxn_2 is a 1-reactant → 1-product isomerization (CCON=O → CC[N+](=O)[O-]).
        result = interpolate(self.rxn_2, weight=0.5)
        # Must return a list (possibly empty after filtering, but not None).
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)

    def test_interpolate_returns_none_for_bimolecular(self):
        """interpolate() returns None for non-isomerization (bimolecular) reactions."""
        r1 = ARCSpecies(label='H', smiles='[H]')
        r2 = ARCSpecies(label='CH4', smiles='C')
        p1 = ARCSpecies(label='H2', smiles='[H][H]')
        p2 = ARCSpecies(label='CH3', smiles='[CH3]')
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p1, p2])
        self.assertIsNone(interpolate(rxn, weight=0.5))

    def test_interpolate_weight_out_of_range(self):
        """interpolate() with weight outside [0, 1] returns None."""
        self.assertIsNone(interpolate(self.rxn_2, weight=-0.1))
        self.assertIsNone(interpolate(self.rxn_2, weight=1.1))

    def test_interpolate_weight_boundary_values(self):
        """interpolate() with weight exactly 0.0 or 1.0 does not raise and returns a list."""
        for w in (0.0, 1.0):
            result = interpolate(self.rxn_2, weight=w)
            self.assertIsNotNone(result,
                                 msg=f'interpolate returned None for boundary weight {w}')
            self.assertIsInstance(result, list)

    # -----------------------------------------------------------------------
    # Tests for specific RMG families ***
    # -----------------------------------------------------------------------

    def test_interpolate_1_plus_2_cycloaddition(self):
        """Test the interpolate_addition() function for 1+2_Cycloaddition: CH2 + C=C=C <=> C=C1CC1"""
        ch2_xyz = """C       0.00000000    0.00000000    0.10513200
H       0.00000000    0.98826300   -0.31539600
H       0.00000000   -0.98826300   -0.31539600"""
        c3h4_xyz = """C       1.29697653    0.02233190    0.00658756
C       0.00000000   -0.00000034    0.00000210
C      -1.29697654   -0.02233198   -0.00658580
H       1.86532844   -0.70256077   -0.56460908
H       1.83420869    0.76626329    0.58339481
H      -1.85591941    0.54211003   -0.74397783
H      -1.84361771   -0.60581213    0.72518823"""
        c4h6_xyz = """C       1.59999925   -0.11618654   -0.14166302
C       0.29517860   -0.02143486   -0.02613492
C      -0.92013120   -0.71833111    0.10894610
C      -0.81238032    0.84414025    0.04444949
H       2.21797993    0.77036923   -0.22897655
H       2.09015362   -1.08321135   -0.15246324
H      -1.12327237   -1.17593811    1.06705013
H      -1.28992770   -1.23997489   -0.76270297
H      -0.94547237    1.40230195    0.96062403
H      -1.11212744    1.33826544   -0.86912905"""
        # Singlet carbene: must use adjlist (u0 p1), SMILES [CH2] gives triplet.
        r_1 = ARCSpecies(label='R1', adjlist="""multiplicity 1
1 C u0 p1 c0 {2,S} {3,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
""", xyz=ch2_xyz)
        r_2 = ARCSpecies(label='R2', smiles='C=C=C', xyz=c3h4_xyz)
        p = ARCSpecies(label='P', smiles='C=C1CC1', xyz=c4h6_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p])

        # Verify the family is detected.
        self.assertTrue(any(pd['family'] == '1+2_Cycloaddition' for pd in rxn.product_dicts))

        ts_xyzs = interpolate_addition(rxn, weight=0.5)
        self.assertIsNotNone(ts_xyzs)
        self.assertGreater(len(ts_xyzs), 0)

        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 10)
            self.assertFalse(colliding_atoms(ts_xyz),
                             msg=f'Collision in 1+2_Cycloaddition TS:\n{xyz_to_str(ts_xyz)}')

        expected_ts = """C       1.59999925   -0.11618654   -0.14166302
C       0.29517860   -0.02143486   -0.02613492
C      -1.15821797   -1.12490772    0.14486040
C      -0.81238032    0.84414025    0.04444949
H       2.21797993    0.77036923   -0.22897655
H       2.09015362   -1.08321135   -0.15246324
H      -1.36135914   -1.58251472    1.10296443
H      -1.52801447   -1.64655150   -0.72678867
H      -0.94547237    1.40230195    0.96062403
H      -1.11212744    1.33826544   -0.86912905"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

        # The TS should have extended forming bonds compared to the product.
        # Get atom map to find the carbene atom in the product.
        atom_map = map_rxn(rxn=rxn)
        p_carbene = atom_map[0]  # *3 (R atom 0) → P atom
        ts_coords = np.array(ts_xyzs[0]['coords'], dtype=float)
        p_coords = np.array(p.get_xyz()['coords'], dtype=float)
        carbene_neighbors = []
        for atom in p.mol.atoms:
            idx = p.mol.atoms.index(atom)
            if idx == p_carbene:
                for neighbor in atom.edges:
                    n_idx = p.mol.atoms.index(neighbor)
                    if p.mol.atoms[n_idx].symbol != 'H':
                        carbene_neighbors.append(n_idx)
        for n_idx in carbene_neighbors:
            d_product = float(np.linalg.norm(p_coords[p_carbene] - p_coords[n_idx]))
            d_ts = float(np.linalg.norm(ts_coords[p_carbene] - ts_coords[n_idx]))
            self.assertGreater(d_ts, d_product,
                               msg=f'Forming bond C[{p_carbene}]-C[{n_idx}] not extended: '
                                   f'd_product={d_product:.3f}, d_ts={d_ts:.3f}')

        # Also test via the top-level interpolate() dispatcher.
        ts_xyzs_dispatch = interpolate(rxn, weight=0.5)
        self.assertIsNotNone(ts_xyzs_dispatch)
        self.assertEqual(len(ts_xyzs_dispatch), len(ts_xyzs))

    def test_interpolate_1_2_insertion_co(self):
        """Test the interpolate_isomerization() function for 1,2_Insertion_CO: C4H10 + CO <=> C5H10O"""
        c4h10_xyz = """C       1.13904106    0.89856200    0.03177638
C       0.01134000   -0.02965608    0.47749851
C       0.00146739   -0.17089790    1.99799846
C       0.14702643   -1.39740814   -0.18794137
H       2.11225156    0.50558156    0.34441627
H       1.15407629    1.01303671   -1.05750152
H       1.02190701    1.89304220    0.47517768
H      -0.94290276    0.41246158    0.16588855
H      -0.15056169    0.80181581    2.47761405
H       0.95376978   -0.57545110    2.35684776
H      -0.79826233   -0.84133228    2.33087525
H       1.10114256   -1.86533009    0.07668567
H      -0.65843268   -2.07254688    0.12073351
H       0.11143087   -1.30417866   -1.27851792"""
        co_xyz = """C       0.00000000    0.00000000    0.56470000
O       0.00000000    0.00000000   -0.56470000"""
        c5h10o_xyz = """C      -0.40644161    0.93028658   -1.13219412
C       0.06717401    0.01697425    0.00280795
C       1.47259752    0.42702876    0.45367616
C       0.06882160   -1.44190079   -0.46444263
C      -0.86152161    0.12342123    1.20607262
O      -1.85425595    0.84265863    1.25983328
H       0.25665086    0.85928946   -2.00150834
H      -1.41834393    0.66505644   -1.45957610
H      -0.43147701    1.97894049   -0.81412156
H       1.48279018    1.45890167    0.82353577
H       1.83237630   -0.21160269    1.26879518
H       2.19094503    0.35635213   -0.37050110
H       0.73390517   -1.58346878   -1.32345883
H       0.40195890   -2.11601880    0.33323885
H      -0.93707925   -1.76281070   -0.75913154
H      -0.58902376   -0.50657502    2.07187062"""
        r_1 = ARCSpecies(label='R1', smiles="CC(C)C", xyz=c4h10_xyz)
        r_2 = ARCSpecies(label='R2', smiles='[C-]#[O+]', xyz=co_xyz)
        p = ARCSpecies(label='P', smiles='CC(C)(C)C=O', xyz=c5h10o_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p])
        self.assertTrue(any(pd['family'] == '1,2_Insertion_CO' for pd in rxn.product_dicts))
        ts_xyzs = interpolate_addition(rxn, weight=0.5)
        self.assertIsNotNone(ts_xyzs)
        self.assertGreaterEqual(len(ts_xyzs), 1)
        # Check 3-membered ring distances in the TS guess.
        # Product ordering: P[4]=CO carbon (*1, ins), P[1]=quaternary C (*2, sub),
        # P[15]=migrating H (*3, mig). Targets: C-C ~ sbl+0.42 = 1.96, C-H ~ 1.51.
        ts_coords = np.array(ts_xyzs[0]['coords'])
        d_cc = float(np.linalg.norm(ts_coords[4] - ts_coords[1]))   # *1-*2 C-C forming
        d_ch1 = float(np.linalg.norm(ts_coords[4] - ts_coords[15]))  # *1-*3 C-H forming
        d_ch2 = float(np.linalg.norm(ts_coords[1] - ts_coords[15]))  # *2-*3 C-H breaking
        self.assertAlmostEqual(d_cc, 1.96, delta=0.05)
        self.assertAlmostEqual(d_ch1, 1.51, delta=0.05)
        self.assertAlmostEqual(d_ch2, 1.51, delta=0.05)
        # Verify the TS guess also works through the main interpolate() dispatcher.
        ts_xyzs_dispatch = interpolate(rxn, weight=0.5)
        self.assertIsNotNone(ts_xyzs_dispatch)
        self.assertEqual(len(ts_xyzs_dispatch), len(ts_xyzs))
        expected_ts = """C      -0.40644161    0.93028658   -1.13219412
C       0.06717401    0.01697425    0.00280795
C       1.47259752    0.42702876    0.45367616
C       0.06882160   -1.44190079   -0.46444263
C      -1.12744899    0.15390179    1.55062148
O      -2.12018333    0.87313919    1.60438214
H       0.25665086    0.85928946   -2.00150834
H      -1.41834393    0.66505644   -1.45957610
H      -0.43147701    1.97894049   -0.81412156
H       1.48279018    1.45890167    0.82353577
H       1.83237630   -0.21160269    1.26879518
H       2.19094503    0.35635213   -0.37050110
H       0.73390517   -1.58346878   -1.32345883
H       0.40195890   -2.11601880    0.33323885
H      -0.93707925   -1.76281070   -0.75913154
H       0.11567616   -0.67766058    1.34266988"""
        self.assertTrue(almost_equal_coords(ts_xyzs_dispatch[0], str_to_xyz(expected_ts)))
        # for ts_xyz in ts_xyzs:
        #     print('\n\n***********')
        #     print(xyz_to_str(ts_xyz))

    def test_interpolate_1_2_insertion_carbene(self):
        """Test the interpolate_addition() function for 1,2_Insertion_carbene: CH2 + C4H6 <=> C5H8"""
        ch2_xyz = """C       0.00000000    0.00000000    0.10513200
H       0.00000000    0.98826300   -0.31539600
H       0.00000000   -0.98826300   -0.31539600"""
        r2_xyz = """C      -1.82234933    0.19305506   -0.03778728
C      -0.54107201    0.22136175   -0.42195352
C       0.54107201   -0.22136176    0.42195351
C       1.82234934   -0.19305496    0.03778732
H      -2.60510293    0.53666984   -0.70674764
H      -2.12527780   -0.16783462    0.93989164
H      -0.29424944    0.59353135   -1.41415681
H       0.29424943   -0.59353144    1.41415677
H       2.12527782    0.16783480   -0.93989157
H       2.60510294   -0.53666975    0.70674767"""
        p_xyz = """C       1.97753426   -0.34691463   -0.12195850
C       0.96032171    0.45485914   -0.46215363
C      -0.43629664    0.27157147   -0.09968556
C      -1.35584640    1.15966116   -0.51269091
C      -0.83651671   -0.91436221    0.73635894
H       2.98719352   -0.11575642   -0.44772907
H       1.84910220   -1.24076974    0.47792776
H       1.19368072    1.33006788   -1.06832846
H      -2.40510842    1.04750710   -0.25687679
H      -1.09525737    2.02366247   -1.11636739
H      -0.32888591   -0.89422114    1.70676182
H      -1.91408642   -0.93005704    0.93479551
H      -0.58767904   -1.85093188    0.22577726"""
        # Singlet carbene: must use adjlist (u0 p1), SMILES [CH2] gives triplet.
        r_1 = ARCSpecies(label='R1', adjlist="""multiplicity 1
1 C u0 p1 c0 {2,S} {3,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
""", xyz=ch2_xyz)
        r_2 = ARCSpecies(label='R2', smiles='C=CC=C', xyz=r2_xyz)
        p = ARCSpecies(label='P', smiles='C=CC(=C)C', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p])
        self.assertTrue(any(pd['family'] == '1,2_Insertion_carbene' for pd in rxn.product_dicts))
        ts_xyzs = interpolate_addition(rxn, weight=0.5)
        self.assertIsNotNone(ts_xyzs)
        self.assertGreaterEqual(len(ts_xyzs), 1)
        # 3-membered ring: P[4]=CH2 carbon (*1, ins), P[2]=central C (*2, sub),
        # P[10]=migrating H (*3, mig).
        ts_coords = np.array(ts_xyzs[0]['coords'])
        d_cc = float(np.linalg.norm(ts_coords[4] - ts_coords[2]))   # *1-*2 C-C forming
        d_ch1 = float(np.linalg.norm(ts_coords[4] - ts_coords[10]))  # *1-*3 C-H forming
        d_ch2 = float(np.linalg.norm(ts_coords[2] - ts_coords[10]))  # *2-*3 C-H breaking
        self.assertAlmostEqual(d_cc, 1.96, delta=0.05)
        self.assertAlmostEqual(d_ch1, 1.51, delta=0.05)
        self.assertAlmostEqual(d_ch2, 1.51, delta=0.05)
        # Verify the dispatcher routes correctly.
        ts_xyzs_dispatch = interpolate(rxn, weight=0.5)
        self.assertIsNotNone(ts_xyzs_dispatch)
        self.assertEqual(len(ts_xyzs_dispatch), len(ts_xyzs))
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        expected_ts_1 = """C       1.97753426   -0.34691463   -0.12195850
C       0.96032171    0.45485914   -0.46215363
C      -0.43629664    0.27157147   -0.09968556
C      -1.35584640    1.15966116   -0.51269091
C      -0.95744903   -1.27270934    0.98898196
H       2.98719352   -0.11575642   -0.44772907
H       1.84910220   -1.24076974    0.47792776
H       1.19368072    1.33006788   -1.06832846
H      -2.40510842    1.04750710   -0.25687679
H      -1.09525737    2.02366247   -1.11636739
H      -0.44981823   -1.25256827    1.95938484
H      -2.03501874   -1.28840417    1.18741853
H      -0.30202710   -1.20798959   -0.36981657"""
        self.assertTrue(almost_equal_coords(ts_xyzs[1], str_to_xyz(expected_ts_1)))

    def test_interpolate_1_2_nh3_elimination(self):
        """Test the interpolate_addition() function for 1,2_NH3_elimination: NNN <=> H2NN(S) + NH3."""
        n3_xyz = """N      -1.26709244   -0.00392551   -0.17821516
N      -0.00831159    0.62912211   -0.22607923
N      -0.03650217    1.66537185    0.72488290
H      -1.36396603   -0.52480010    0.69598616
H      -1.33497366   -0.72150540   -0.90528855
H       0.20276134    1.00409437   -1.16407646
H       0.01517757    1.28943240    1.67165685
H      -0.93213409    2.15501337    0.67312449"""
        h2nn_xyz = """N       1.24087876    0.00949543    0.60790318
N      -0.09033762   -0.00069128    0.02459641
H      -0.47927195   -0.84665038   -0.39226764
H      -0.67126919    0.83784623    0.01648883"""
        nh3_xyz = """N       0.00064924   -0.00099698    0.29559292
H      -0.41786606    0.84210396   -0.09477452
H      -0.52039228   -0.78225292   -0.10002797
H       0.93760911   -0.05885406   -0.10079043"""
        # A -> B + C: triazene decomposes to aminonitrene + NH3.
        r = ARCSpecies(label='triazene', smiles='NNN', xyz=n3_xyz)
        # H2NN(S) = singlet aminonitrene; use adjlist for charged resonance structure.
        p1 = ARCSpecies(label='H2NNs', adjlist="""multiplicity 1
1 N u0 p0 c+1 {2,S} {3,S} {4,D}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 N u0 p2 c-1 {1,D}
""", xyz=h2nn_xyz)
        p2 = ARCSpecies(label='NH3', smiles='N', xyz=nh3_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        self.assertTrue(any(pd['family'] == '1,2_NH3_elimination' for pd in rxn.product_dicts))
        ts_xyzs = interpolate_addition(rxn, weight=0.5)
        self.assertIsNotNone(ts_xyzs)
        self.assertGreaterEqual(len(ts_xyzs), 1)
        # 3-membered ring in reactant ordering: *1=N0 (NH2), *2=N1 (central), *4=H5 (mig).
        # Targets: N-N ~ sbl+0.42 = 1.87, N-H ~ sbl+0.42 = 1.46.
        ts_coords = np.array(ts_xyzs[0]['coords'])
        d_nn = float(np.linalg.norm(ts_coords[0] - ts_coords[1]))  # *1-*2 N-N breaking
        d_nh1 = float(np.linalg.norm(ts_coords[1] - ts_coords[5]))  # *2-*4 N-H breaking
        d_nh2 = float(np.linalg.norm(ts_coords[0] - ts_coords[5]))  # *1-*4 N-H forming
        self.assertAlmostEqual(d_nn, 1.87, delta=0.05)
        self.assertAlmostEqual(d_nh1, 1.46, delta=0.05)
        self.assertAlmostEqual(d_nh2, 1.46, delta=0.05)
        # Verify the dispatcher routes correctly.
        ts_xyzs_dispatch = interpolate(rxn, weight=0.5)
        self.assertIsNotNone(ts_xyzs_dispatch)
        self.assertEqual(len(ts_xyzs_dispatch), len(ts_xyzs))
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        expected_ts_1 = """N      -1.26709244   -0.00392551   -0.17821516
N      -0.00831159    0.62912211   -0.22607923
N      -0.04578555    2.00661712    1.03804228
H      -1.36396603   -0.52480010    0.69598616
H      -1.33497366   -0.72150540   -0.90528855
H       0.20944918    2.06247116   -0.39838925
H       0.00589419    1.63067767    1.98481623
H      -0.94141747    2.49625864    0.98628387"""
        self.assertTrue(almost_equal_coords(ts_xyzs[1], str_to_xyz(expected_ts_1)))

    def test_interpolate_1_2_xy_interchange(self):
        """Test the interpolate_isomerization() function for 1,2_XY_interchange: CC(O)C(F)CC <=> CC(F)C(O)CC"""
        r_xyz = """C      -2.19165340   -1.06342386    0.64288457
C      -1.10438235   -0.01824702    0.85411769
O      -1.64525065    1.26348244    0.52357215
C       0.13970677   -0.31671404    0.00437792
F      -0.23628791   -0.30509710   -1.31059964
C       1.24160324    0.72514346    0.18806220
C       2.48523729    0.36649099   -0.61207922
H      -3.06903964   -0.82895209    1.25533763
H      -2.53109848   -1.07583207   -0.39863056
H      -1.83804884   -2.06410142    0.90888898
H      -0.83109839    0.00685736    1.91457298
H      -1.74921350    1.27798481   -0.44704531
H       0.51885825   -1.31951225    0.23044429
H       0.88976298    1.70723131   -0.14754737
H       1.50673852    0.80849260    1.24797123
H       3.26058055    1.12528022   -0.46687147
H       2.26228063    0.30987172   -1.68227239
H       2.89130493   -0.59895508   -0.29374226"""
        p_xyz = """C       2.53742425    0.14940355    0.36879540
C       1.21622905   -0.59544292    0.44908313
F       0.98130513   -0.88050298    1.76522888
C       0.03158131    0.22549251   -0.08312007
O      -0.09071355    1.44798127    0.65011523
C      -1.28095263   -0.55904285    0.02626355
C      -2.45764617    0.19716527   -0.57658345
H       2.77523406    0.42742762   -0.66215915
H       3.34511206   -0.48070738    0.75517095
H       2.52072347    1.05380514    0.98520590
H       1.29306919   -1.54783677   -0.08596677
H       0.21494561    0.49118221   -1.13045255
H      -0.11044331    1.20302091    1.59489770
H      -1.50963062   -0.76175655    1.07952940
H      -1.18063581   -1.52460749   -0.48269885
H      -2.63770052    1.13994225   -0.05073603
H      -2.27931973    0.42024052   -1.63330287
H      -3.36858178   -0.40576429   -0.50639881"""
        r = ARCSpecies(label='R', smiles='CC(O)C(F)CC', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='CC(F)C(O)CC', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        # 1,2_XY_interchange is in the 'halogens' family set, not the default set.
        pds = rxn.get_product_dicts(rmg_family_set='halogens')
        self.assertTrue(any(pd['family'] == '1,2_XY_interchange' for pd in pds))
        rxn._product_dicts = pds
        rxn.family = '1,2_XY_interchange'
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertIsNotNone(ts_xyzs)
        self.assertGreater(len(ts_xyzs), 0)
        # r_label_map: *2:1(C), *3:3(C), *1:2(O), *4:4(F)
        # In the TS all 4 bonds (O-C1, O-C3, F-C1, F-C3) should be at Pauling
        # TS estimates: sbl + 0.42 A. C-O single bond = 1.43, C-F = 1.35.
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 18)
            self.assertFalse(colliding_atoms(ts_xyz),
                             msg=f'Collision detected in 1,2_XY_interchange TS guess:\n{xyz_to_str(ts_xyz)}')
            coords = np.array(ts_xyz['coords'], dtype=float)
            d_o_c1 = float(np.linalg.norm(coords[2] - coords[1]))  # O-C1 (breaking)
            d_o_c3 = float(np.linalg.norm(coords[2] - coords[3]))  # O-C3 (forming)
            d_f_c3 = float(np.linalg.norm(coords[4] - coords[3]))  # F-C3 (breaking)
            d_f_c1 = float(np.linalg.norm(coords[4] - coords[1]))  # F-C1 (forming)
            d_cc = float(np.linalg.norm(coords[1] - coords[3]))    # C-C backbone
            self.assertAlmostEqual(d_o_c1, 1.85, delta=0.05, msg=f'O-C1 dist {d_o_c1:.3f}')
            self.assertAlmostEqual(d_o_c3, 1.85, delta=0.05, msg=f'O-C3 dist {d_o_c3:.3f}')
            self.assertAlmostEqual(d_f_c3, 1.77, delta=0.05, msg=f'F-C3 dist {d_f_c3:.3f}')
            self.assertAlmostEqual(d_f_c1, 1.77, delta=0.05, msg=f'F-C1 dist {d_f_c1:.3f}')
            self.assertAlmostEqual(d_cc, 1.54, delta=0.10, msg=f'C-C dist {d_cc:.3f}')
        expected_ts = """C      -2.19165340   -1.06342386    0.64288457
C      -1.10438235   -0.01824702    0.85411769
O      -0.64883361    1.32151972   -0.33752001
C       0.13970677   -0.31671404    0.00437792
F       0.39598914   -0.28189864    1.75537970
C       1.24160324    0.72514346    0.18806220
C       2.48523729    0.36649099   -0.61207922
H      -3.06903964   -0.82895209    1.25533763
H      -2.53109848   -1.07583207   -0.39863056
H      -1.83804884   -2.06410142    0.90888898
H      -0.83109839    0.00685736    1.91457298
H      -0.75279646    1.33602209   -1.30813747
H       0.51885825   -1.31951225    0.23044429
H       0.88976298    1.70723131   -0.14754737
H       1.50673852    0.80849260    1.24797123
H       3.26058055    1.12528022   -0.46687147
H       2.26228063    0.30987172   -1.68227239
H       2.89130493   -0.59895508   -0.29374226"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_1_2_shift_c(self):
        """Test the interpolate_isomerization() function for 1,2_shiftC: CC[C]1C=CC=C1 <=> [CH2]C1(C)C=CC=C1"""
        r_xyz = """C      -2.08011725   -0.87098529   -0.24102896
                   C      -1.38616808    0.31243567    0.41701874
                   C       0.09289885    0.19281646    0.35695343
                   C       0.92864438    0.68782411   -0.70819340
                   C       2.18636908    0.35957487   -0.37255721
                   C       2.18107732   -0.34427638    0.89885251
                   C       0.92008966   -0.45002583    1.34718072
                   H      -1.81540032   -1.81207279    0.25290661
                   H      -1.80896484   -0.95285941   -1.29907735
                   H      -3.16674193   -0.75248342   -0.17993048
                   H      -1.70601347    1.23815887   -0.07595913
                   H      -1.71241303    0.38717620    1.46117876
                   H       0.59841756    1.21177196   -1.58944757
                   H       3.07727387    0.57336525   -0.94230522
                   H       3.06757417   -0.71677188    1.38813917
                   H       0.58240767   -0.91755259    2.25688920"""
        p_xyz = """C      -0.91419261   -0.92211886    1.28775915
                   C      -0.38593444   -0.06230282    0.18302891
                   C       0.67135826    0.91653743    0.70043366
                   C      -1.50477869    0.67123329   -0.52120219
                   C      -1.56600546    0.29578439   -1.80779070
                   C      -0.54194075   -0.68042899   -2.06986329
                   C       0.15393453   -0.90959193   -0.94583904
                   H      -1.87479029   -1.41555103    1.18169570
                   H      -0.24773685   -1.28139411    2.06376191
                   H       1.52757979    0.38651768    1.13544157
                   H       1.05855879    1.55868444   -0.10049234
                   H       0.25983545    1.57373963    1.47630068
                   H      -2.15472843    1.39365373   -0.04968993
                   H      -2.26617185    0.65598910   -2.54596855
                   H      -0.37671673   -1.14563680   -3.02965012
                   H       0.98207416   -1.59686236   -0.85449771"""
        r = ARCSpecies(label='R', smiles='CC[C]1C=CC=C1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[CH2]C1(C)C=CC=C1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        r_label_dict = rxn.product_dicts[0]['r_label_map']
        bb, fb = rxn.get_expected_changing_bonds(r_label_dict=r_label_dict)
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertIsNotNone(ts_xyzs)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 16)
            self.assertFalse(colliding_atoms(ts_xyz),
                             msg=f'Collision detected in 1,2_shift_C TS guess:\n{xyz_to_str(ts_xyz)}')
            # Non-reactive H's on the migrating CH3 must point away from the backbone.
            self.assertFalse(
                has_inward_migrating_group_h(ts_xyz, r.mol, list(bb), list(fb)),
                msg=f'Migrating group H atoms point inward in 1,2_shift_C TS guess:\n{xyz_to_str(ts_xyz)}')
            # Every H must be within 1.2× sbl of its nearest heavy atom (no detached H's).
            coords_arr = np.array(ts_xyz['coords'], dtype=float)
            for h_idx, sym in enumerate(ts_xyz['symbols']):
                if sym != 'H':
                    continue
                dists = [float(np.linalg.norm(coords_arr[h_idx] - coords_arr[j]))
                         for j, sj in enumerate(ts_xyz['symbols']) if sj != 'H']
                nearest_dist = min(dists)
                self.assertLess(
                    nearest_dist, get_single_bond_length('H', 'C') * 1.2,
                    msg=f'H[{h_idx}] is detached: nearest heavy atom at {nearest_dist:.3f} A\n'
                        f'{xyz_to_str(ts_xyz)}')
        self.assertEqual(len(ts_xyzs), 2)
        expected_ts_0 = """C      -0.30959679    0.71818290   -2.43296062
                           C       1.04291313    0.60286397   -0.95092867
                           C      -0.22666985    0.03570512   -0.41369378
                           C      -0.22804391   -1.36837041    0.04376404
                           C      -0.22804391   -1.36837041    1.38625090
                           C      -0.22804391    0.00155172    1.87125677
                           C      -0.22784416    0.84090103    0.82376965
                           H      -0.65613352    1.74712773   -2.52931847
                           H       0.14924810    0.40013143   -3.36912547
                           H      -1.15541125    0.06983043   -2.20418786
                           H       2.03188245    0.90117186   -1.29885405
                           H       1.33398396    1.64007766   -0.78489998
                           H       0.54978504   -1.83796886    0.62224425
                           H      -0.22912753   -2.23699494    2.02618885
                           H      -0.22900952    0.27402069    2.91518735
                           H      -0.22813027    1.91703112    0.86985230"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_0)))

    def test_interpolate_1_2_shift_s(self):
        """Test the interpolate_isomerization() function for 1,2_shiftS: CS[C]1C=CC=C1 <=> [S]C1(C)C=CC=C1"""
        r_xyz = """C       2.48151944    0.27794621    0.08255979
S       1.54240287   -1.26466055    0.05386695
C       1.92879739   -1.90704889    1.58198871
C       1.76008399   -1.25021474    2.85397303
C       2.09667504   -2.15608935    3.78713266
C       2.46843387   -3.40285567    3.13910853
C       2.35859071   -3.26370702    1.80771576
H       3.53858518    0.08313555    0.28471229
H       2.08690550    0.96047702    0.84007270
H       2.40049305    0.76499434   -0.89303041
H       1.41614096   -0.24313760    3.01704222
H       2.07945794   -2.01105055    4.85632253
H       2.76626324   -4.29587844    3.66656769
H       2.54649707   -4.00689008    1.05107429"""
        p_xyz = """S      -0.26535355    0.21493749    2.39170985
C      -0.33543568    0.05497042    0.59777663
C      -1.77224426    0.01601073    0.08709074
C       0.45698680   -1.15342461    0.17777910
C       1.48686167   -0.78349801   -0.60068673
C       1.49352176    0.65086177   -0.72885049
C       0.46773983    1.16241160   -0.02914684
H      -2.31416437    0.93433281    0.34768607
H      -1.81752205   -0.08138872   -1.00531844
H      -2.33838929   -0.82062285    0.51653057
H       0.22173502   -2.16371121    0.47462621
H       2.22670481   -1.43948493   -1.02660566
H       2.23902606    1.21410373   -1.26371074
H       0.24205001    2.21144523    0.08369465"""
        r = ARCSpecies(label='R', smiles='CS[C]1C=CC=C1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[S]C1(C)C=CC=C1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts_0 = """C       1.44144128   -0.30929849   -1.68182421
S      -0.68240384    0.51368349   -1.51998530
C       0.06410061    0.15174720    0.02342821
C       0.05197499   -1.24850121    0.48097363
C       0.05197499   -1.24850121    1.82436099
C       0.05197499    0.12210930    2.30808941
C       0.05459145    0.95790903    1.25605282
H       2.00298968    0.27514423   -2.41065211
H       1.93089676   -0.24533601   -0.71000008
H       1.40418291   -1.35052625   -2.00206875
H       0.18541928   -2.11126912   -0.14903969
H       0.04018156   -2.11684556    2.46493812
H       0.06661891    0.39766615    3.35128611
H       0.04832973    2.03433889    1.29342045"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_0)))

    def test_interpolate_1_3_insertion_co2(self):
        """Test the interpolate_isomerization() function for 1,3_Insertion_CO2: CC=C(CC)C(C)(C)C(=O)O <=> O=C=O + CC=C(CC)C(C)C"""
        r_xyz = """C      -2.49526563   -1.71744655   -0.05070502
C      -1.02439736   -1.43442148   -0.09903098
C      -0.03369873   -2.05724101    0.58071768
C      -0.33706853   -3.21011130    1.53261903
C      -0.85997546   -2.75301776    2.88970062
C       1.47028993   -1.69663277    0.40874963
C       2.07012721   -1.30878861    1.77489924
C       1.74154080   -0.52957843   -0.56586530
C       2.20380863   -2.91010712   -0.19503352
O       1.73766923   -3.76432521   -0.93387645
O       3.51955595   -2.95521685    0.10563464
H      -3.10560600   -0.84943191    0.20522867
H      -2.69351927   -2.49442175    0.69448241
H      -2.82934183   -2.08783564   -1.02499122
H      -0.78561006   -0.62690380   -0.78689260
H       0.55596394   -3.82539962    1.69076067
H      -1.06313426   -3.89101221    1.07201521
H      -1.79380282   -2.19600714    2.76086637
H      -0.14816985   -2.09644367    3.39930225
H      -1.06676022   -3.60034589    3.54974021
H       2.00787775   -2.13013981    2.49728135
H       3.13273869   -1.05110182    1.68998787
H       1.55881292   -0.43721490    2.19959957
H       1.26204457    0.39674910   -0.22955829
H       2.81639214   -0.32714424   -0.65140849
H       1.38896771   -0.75762301   -1.57886997
H       3.82821421   -3.76916475   -0.34544391"""
        p1_xyz = """O      -1.37316735    0.24657196    0.00000000
C      -0.00000000   -0.05081069    0.00000000
O       1.37316735   -0.34819332    0.00000000"""
        p2_xyz = """ C                 -2.38749724   -2.07681559   -0.24769962
 C                 -0.91223049   -1.65569429   -0.38128430
 C                  0.00974358   -2.13096778    0.49086589
 C                 -0.41782495   -3.09217448    1.61552887
 C                 -0.80439231   -2.28004191    2.86557141
 C                  1.48500992   -1.70984460    0.35728267
 C                  2.14707486   -1.71091779    1.74770306
 C                  1.56241137   -0.29548780   -0.24703833
 H                 -3.01706611   -1.28917857   -0.60570935
 H                 -2.61083914   -2.27262544    0.78024828
 H                 -2.55960948   -2.96124003   -0.82482247
 H                 -0.61515337   -0.98784419   -1.16270699
 H                  0.39428846   -3.74720075    1.85283123
 H                 -1.25842619   -3.66927357    1.29111215
 H                 -1.61650572   -1.62501564    2.62826906
 H                  0.03620892   -1.70294282    3.18998814
 H                 -1.10146915   -2.94789332    3.64699310
 H                  2.09329593   -2.69362024    2.16758843
 H                  3.17209758   -1.41831921    1.65488875
 H                  1.63583722   -1.02155958    2.38670333
 H                  1.05117373    0.39387041    0.39196194
 H                  2.58743409   -0.00288922   -0.33985264
 H                  1.10240521   -0.29474214   -1.21310964
 H                  1.99624756   -2.39920281   -0.28171759"""
        r = ARCSpecies(label='R', smiles='CC=C(CC)C(C)(C)C(=O)O', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='O=C=O', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='CC=C(CC)C(C)C', xyz=p2_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts = """C      -2.49526563   -1.71744655   -0.05070502
C      -1.02439736   -1.43442148   -0.09903098
C      -0.03369873   -2.05724101    0.58071768
C      -0.33706853   -3.21011130    1.53261903
C      -0.85997546   -2.75301776    2.88970062
C       1.47028993   -1.69663277    0.40874963
C       2.07012721   -1.30878861    1.77489924
C       1.74154080   -0.52957843   -0.56586530
C       2.40316640   -3.23990858   -0.35913140
O       1.93702700   -4.09412667   -1.09797433
O       3.71891372   -3.28501831   -0.05846324
H      -3.10560600   -0.84943191    0.20522867
H      -2.69351927   -2.49442175    0.69448241
H      -2.82934183   -2.08783564   -1.02499122
H      -0.78561006   -0.62690380   -0.78689260
H       0.55596394   -3.82539962    1.69076067
H      -1.06313426   -3.89101221    1.07201521
H      -1.79380282   -2.19600714    2.76086637
H      -0.14816985   -2.09644367    3.39930225
H      -1.06676022   -3.60034589    3.54974021
H       2.00787775   -2.13013981    2.49728135
H       3.13273869   -1.05110182    1.68998787
H       1.55881292   -0.43721490    2.19959957
H       1.26204457    0.39674910   -0.22955829
H       2.81639214   -0.32714424   -0.65140849
H       1.38896771   -0.75762301   -1.57886997
H       2.85557208   -2.29609554    0.36706348"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_1_3_insertion_ror(self):
        """Test the interpolate_isomerization() function for 1,3_Insertion_ROR: CCOCOCC <=> C=C + CCOCO"""
        r_xyz = """C       2.84371327   -0.12259426   -0.80959323
C       1.78429293    0.33611891    0.17319475
O       0.98774248    1.33390916   -0.46087418
C      -0.04273064    1.81069282    0.39683874
O       0.45956373    2.55313719    1.50198404
C       1.03786821    3.78918928    1.08984338
C       1.53368595    4.51749018    2.32370039
H       3.47128951    0.71926584   -1.12035847
H       3.48059889   -0.89276333   -0.36559694
H       2.38025762   -0.52625224   -1.71588438
H       1.15557568   -0.51308510    0.46266314
H       2.26572175    0.74853212    1.06638262
H      -0.72450592    2.43427115   -0.19225856
H      -0.62749309    0.96759961    0.78148155
H       0.28574202    4.39906139    0.57741117
H       1.87659808    3.60554649    0.40980601
H       2.27586518    3.91304280    2.85564919
H       0.70988426    4.69809127    3.02214828
H       1.98588747    5.47639445    2.05549614"""
        p1_xyz = """C      -0.63422754   -0.20894058   -0.01346068
C       0.63422754    0.20894058    0.01346068
H      -1.30426171   -0.01843680    0.81903872
H      -1.02752125   -0.74974821   -0.86852786
H       1.02752125    0.74974821    0.86852786
H       1.30426171    0.01843680   -0.81903872"""
        p2_xyz = """C      -1.84212476   -0.71378394    0.47665318
C      -0.65235263    0.02855812   -0.09986792
O       0.36443730    0.08927367    0.89809302
C       1.52397691    0.75639354    0.43324228
O       1.27548635    2.14080392    0.30460644
H      -2.22007229   -0.20284632    1.36840285
H      -1.55349776   -1.72370780    0.78627131
H      -2.64953175   -0.78674209   -0.25731493
H      -0.95561943    1.03979140   -0.39163212
H      -0.27680324   -0.50308610   -0.98115751
H       1.85311802    0.35031520   -0.52912452
H       2.32963217    0.60754117    1.15866489
H       1.17124840    2.45844005    1.21692406"""
        r = ARCSpecies(label='R', smiles='CCOCOCC', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='C=C', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='CCOCO', xyz=p2_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts_1 = """C       3.08090013   -0.41970379   -0.62078809
C       2.02147979    0.03900938    0.36199989
O       0.98774248    1.33390916   -0.46087418
C      -0.04273064    1.81069282    0.39683874
O       0.45956373    2.55313719    1.50198404
C       1.03786821    3.78918928    1.08984338
C       1.53368595    4.51749018    2.32370039
H       3.70847637    0.42215631   -0.93155333
H       3.71778575   -1.18987286   -0.17679180
H       1.79842117    0.31756596   -0.92374194
H       1.39276254   -0.81019463    0.65146828
H       2.50290861    0.45142259    1.25518776
H      -0.72450592    2.43427115   -0.19225856
H      -0.62749309    0.96759961    0.78148155
H       0.28574202    4.39906139    0.57741117
H       1.87659808    3.60554649    0.40980601
H       2.27586518    3.91304280    2.85564919
H       0.70988426    4.69809127    3.02214828
H       1.98588747    5.47639445    2.05549614"""
        self.assertTrue(almost_equal_coords(ts_xyzs[1], str_to_xyz(expected_ts_1)))

    def test_interpolate_1_3_insertion_rsr(self):
        """Test the interpolate_isomerization() function for 1,3_Insertion_RSR: H2S + C=C=O <=> C=C(S)O"""
        r1_xyz = """S      -0.00070557    0.61325178    0.00000000
H      -0.97540522   -0.30774854    0.00000000
H       0.97611080   -0.30550323    0.00000000"""
        r2_xyz = """C      -0.53132348   -0.00073450    0.32762745
C       0.75409290    0.00104246    0.48880326
O       1.91943650    0.00265342    0.63492336
H      -1.07761398    0.93232184    0.30157730
H      -1.06459194   -0.93528323    0.21831383"""
        p_xyz = """C      -0.52979235   -0.97049747    0.10629971
C      -0.10826701    0.29287574    0.05045664
S       1.48758014    0.73425675   -0.41792577
O      -0.96330721    1.31352025    0.37581729
H       0.10405125   -1.81396569   -0.14003149
H      -1.54925629   -1.18740449    0.40936382
H       1.98856684   -0.48879943   -0.64555827
H      -0.43054237    2.12019619    0.25847314"""
        r1 = ARCSpecies(label='R1`', smiles='S', xyz=r1_xyz)
        r2 = ARCSpecies(label='R2', smiles='C=C=O', xyz=r2_xyz)
        p = ARCSpecies(label='P', smiles='C=C(S)O', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts_1 = """C      -0.52979235   -0.97049747    0.10629971
C      -0.10826701    0.29287574    0.05045664
S       1.95988414    0.86488707   -0.55654737
O      -0.96330721    1.31352025    0.37581729
H       0.10405125   -1.81396569   -0.14003149
H      -1.54925629   -1.18740449    0.40936382
H       2.46087084   -0.35816911   -0.78417987
H       0.35489815    1.35917387   -0.02995404"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_1)))

    def test_interpolate_1_3_nh3_elimination(self):
        """Test the interpolate_isomerization() function for 1,3_NH3_elimination: CCN <=> C=C + NH3"""
        r_xyz = """C       1.14981017    0.04138987   -0.06722786
C      -0.25415691   -0.17696939    0.46881798
N      -0.38312147    0.39227542    1.80366803
H       1.89791609   -0.44343932    0.56909864
H       1.38984772    1.10839783   -0.12647258
H       1.23928774   -0.38052643   -1.07342059
H      -0.98187243    0.29596310   -0.19835733
H      -0.47689047   -1.24854874    0.50220986
H       0.27600194   -0.06032721    2.43576220
H      -1.31312338    0.19118810    2.16873833"""
        p1_xyz = """C      -0.63422754   -0.20894058   -0.01346068
C       0.63422754    0.20894058    0.01346068
H      -1.30426171   -0.01843680    0.81903872
H      -1.02752125   -0.74974821   -0.86852786
H       1.02752125    0.74974821    0.86852786
H       1.30426171    0.01843680   -0.81903872"""
        p2_xyz = """N       0.00064924   -0.00099698    0.29559292
H      -0.41786606    0.84210396   -0.09477452
H      -0.52039228   -0.78225292   -0.10002797
H       0.93760911   -0.05885406   -0.10079043"""
        r = ARCSpecies(label='R', smiles='CCN', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='C=C', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='N', xyz=p2_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts_3 = """C       1.14981017    0.04138987   -0.06722786
C      -0.25415691   -0.17696939    0.46881798
N      -0.42146184    0.56150838    2.20051069
H       1.89791609   -0.44343932    0.56909864
H       0.45937257    0.77701765    1.05627011
H       1.23928774   -0.38052643   -1.07342059
H      -0.98187243    0.29596310   -0.19835733
H      -0.47689047   -1.24854874    0.50220986
H       0.23766157    0.10890575    2.83260486
H      -1.35146375    0.36042106    2.56558099"""
        self.assertTrue(almost_equal_coords(ts_xyzs[1], str_to_xyz(expected_ts_3)))

    def test_interpolate_1_3_sigmatropic_rearrangement(self):
        """Test the interpolate_isomerization() function for 1,3_sigmatropic_rearrangement: c1ncc[nH]1 <=> N=CN1C=C1"""
        r_xyz = """C      -0.96405208   -0.58870010   -0.35675666
                   N       0.09948347   -1.35699528   -0.30406608
                   C       1.08781769   -0.57088551    0.22943180
                   C       0.61245126    0.68985747    0.50218591
                   N      -0.70083129    0.66320502    0.12207481
                   H      -1.93870511   -0.87854432   -0.72608823
                   H       2.08729155   -0.95482079    0.38815067
                   H       1.07812779    1.57128662    0.91862266
                   H      -1.36158329    1.42559689    0.18141711"""
        p_xyz = """N       0.76582385   -0.14849540   -1.32485588
                   C       0.78208226    0.49284271   -0.20399502
                   N      -0.04861443    0.34490826    0.88039960
                   C      -0.56227958   -0.84609375    1.31645778
                   C      -1.38522743    0.06039446    0.80970400
                   H       1.52092135    0.20130809   -1.92536405
                   H       1.53681129    1.27833147   -0.02452505
                   H      -0.33519514   -1.78256934    0.82247210
                   H      -1.89445111   -0.06503499   -0.13767862"""
        r = ARCSpecies(label='R', smiles='c1ncc[nH]1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='N=CN1C=C1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertIsNotNone(ts_xyzs)
        self.assertIsInstance(ts_xyzs, list)
        expected_ts = """C      -0.23641275   -0.34267593    0.97003367
                         N      -0.23641275    0.83737533    0.32772628
                         C      -0.96859257    0.86919017   -0.82935514
                         C       1.00242139    0.15855582   -0.95991875
                         N       0.50913146   -1.34562885    0.51866657
                         H      -0.23641275   -0.34267593    2.05186727
                         H      -2.01507831    1.14561545   -0.83092758
                         H       1.82425052    0.73828207   -0.53966363
                         H       1.04338235   -1.08869229   -0.33582720"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_1_4_cyclic_birad_scission(self):
        """Test the interpolate_isomerization() function for 1,4_Cyclic_birad_scission: C=C=CCC1=CC=CC=C1 <=> C=C1[CH]C[C]2C=CC=CC21"""
        r_xyz = """C      -1.71276869   -2.14835263   -0.29600082
C      -1.30379477   -0.91506552    0.02297736
C       0.02166928   -0.61873233    0.53428358
C       0.12717819    0.82647389    0.91669270
C      -1.28264593    1.32277593    0.84036811
C      -1.85754229    2.26155916    1.60812418
C      -3.29934285    2.39956966    1.60188376
C      -4.08920980    1.47079635    1.03722857
C      -3.52758183    0.33798706    0.32703925
C      -2.07314208    0.38964129   -0.05559628
H      -2.71169105   -2.33428156   -0.67752001
H      -1.05967066   -3.00703332   -0.18456792
H       0.81708685   -1.33998293    0.63942836
H       0.76667301    1.35549222    0.20321493
H       0.54320924    0.92870099    1.92345007
H      -1.28510222    2.85939041    2.30922346
H      -3.72854139    3.24346240    2.13325690
H      -5.17022073    1.55123850    1.09315776
H      -4.19994276   -0.36461812   -0.14850570
H      -1.99779884    0.76292039   -1.08682170"""
        p_xyz = """C       4.02239745    0.99112778   -0.51172218
C       2.99867210    0.19793387   -0.43514244
C       1.97408045   -0.60057022   -0.35843389
C       0.86276945   -0.47965174    0.64266584
C       0.66594244   -1.74662174    1.44082381
C       1.50458787   -2.03704659    2.52675501
C       1.33469712   -3.21351665    3.25813434
C       0.32902177   -4.11291347    2.90923170
C      -0.50576822   -3.83844901    1.82742278
C      -0.33746731   -2.66244233    1.09422711
H       3.92502565    1.97999484   -0.94447322
H       4.99376579    0.67541658   -0.14899387
H       1.89710222   -1.40848243   -1.08099539
H      -0.06428664   -0.22956820    0.11177135
H       1.04470794    0.35316397    1.33438815
H       2.29799691   -1.34625282    2.80438716
H       1.98935922   -3.42892914    4.09838766
H       0.19790147   -5.02919762    3.47855443
H      -1.28761242   -4.54175731    1.55312124
H      -0.99449818   -2.46537571    0.25025301"""
        r = ARCSpecies(label='R', smiles='C=C=CCC1=CC=CC=C1', xyz=r_xyz)
        p = ARCSpecies(label='P', adjlist="""multiplicity 1
1  C u0 p0 c0 {2,D} {11,S} {12,S}
2  C u0 p0 c0 {1,D} {3,S} {10,S}
3  C u1 p0 c0 {2,S} {4,S} {13,S}
4  C u0 p0 c0 {3,S} {5,S} {14,S} {15,S}
5  C u1 p0 c0 {4,S} {6,S} {10,S}
6  C u0 p0 c0 {5,S} {7,D} {16,S}
7  C u0 p0 c0 {6,D} {8,S} {17,S}
8  C u0 p0 c0 {7,S} {9,D} {18,S}
9  C u0 p0 c0 {8,D} {10,S} {19,S}
10 C u0 p0 c0 {2,S} {5,S} {9,S} {20,S}
11 H u0 p0 c0 {1,S}
12 H u0 p0 c0 {1,S}
13 H u0 p0 c0 {3,S}
14 H u0 p0 c0 {4,S}
15 H u0 p0 c0 {4,S}
16 H u0 p0 c0 {6,S}
17 H u0 p0 c0 {7,S}
18 H u0 p0 c0 {8,S}
19 H u0 p0 c0 {9,S}
20 H u0 p0 c0 {10,S}
""", xyz=p_xyz, multiplicity=1)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts_0 = """C       0.40785983   -3.07475167   -0.01682026
C       0.40785983   -1.94674190   -0.69777642
C       0.56046962   -1.23821765   -1.86772153
C      -0.09556311    0.10714284   -1.96011932
C      -0.20640314    0.67108350   -0.57086920
C       0.05803952    1.97992508   -0.25434547
C       0.07996567    2.39039356    1.10698005
C      -0.21182984    1.45227884    2.13030111
C      -0.45190836    0.09108931    1.79636617
C      -0.52461277   -0.29079423    0.39901033
H       0.40785983   -3.07475167    1.06852417
H       0.44406118   -4.03321525   -0.52310457
H       1.14725771   -1.62497723   -2.69138822
H      -1.08883563   -0.00387350   -2.40615710
H       0.48271872    0.77823617   -2.60223828
H       0.30272090    2.70452498   -1.02589269
H       0.22968212    3.44185145    1.33476939
H      -0.16652518    1.74607229    3.17489327
H      -0.69280023   -0.61529450    2.58345167
H      -1.35044181   -1.00228968    0.31313344"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_0)))

    def test_interpolate_1_4_linear_birad_scission(self):
        """Test the interpolate_isomerization() function for 1,4_Linear_birad_scission: C=C[CH]CC[CH]C=C <=> 2 * C=CC=C"""
        r_xyz = """C       2.31370655    0.15643901    3.32551009
C       1.59622998    0.58189265    2.28072491
C       1.90251261    1.79483739    1.57174749
C       1.10131796    2.20825233    0.37541984
C       1.82975157    1.87655577   -0.92972510
C       1.05206206    2.33621898   -2.12468685
C       0.27040276    1.40861807   -2.89686464
C      -0.47125439    1.78651122   -3.94304892
H       2.05119299   -0.76524044    3.83537533
H       3.17151218    0.70580561    3.70018358
H       0.74453349   -0.00480207    1.94311197
H       2.75484648    2.39750827    1.86896553
H       0.11292791    1.73194873    0.38920030
H       0.92477528    3.28941089    0.43888724
H       2.80815876    2.37307729   -0.94661245
H       2.02842823    0.79908706   -0.98938000
H       1.04303838    3.39278251   -2.37291342
H       0.29098789    0.36007273   -2.60753743
H      -1.04834025    1.05626254   -4.50168832
H      -0.52469319    2.81976569   -4.27099000"""
        p_xyz = """C      -1.82234933    0.19305506   -0.03778728
C      -0.54107201    0.22136175   -0.42195352
C       0.54107201   -0.22136176    0.42195351
C       1.82234934   -0.19305496    0.03778732
H      -2.60510293    0.53666984   -0.70674764
H      -2.12527780   -0.16783462    0.93989164
H      -0.29424944    0.59353135   -1.41415681
H       0.29424943   -0.59353144    1.41415677
H       2.12527782    0.16783480   -0.93989157
H       2.60510294   -0.53666975    0.70674767"""
        r = ARCSpecies(label='R', adjlist="""multiplicity 1
1  C u0 p0 c0 {2,D} {9,S} {10,S}
2  C u0 p0 c0 {1,D} {3,S} {11,S}
3  C u1 p0 c0 {2,S} {4,S} {12,S}
4  C u0 p0 c0 {3,S} {5,S} {13,S} {14,S}
5  C u0 p0 c0 {4,S} {6,S} {15,S} {16,S}
6  C u1 p0 c0 {5,S} {7,S} {17,S}
7  C u0 p0 c0 {6,S} {8,D} {18,S}
8  C u0 p0 c0 {7,D} {19,S} {20,S}
9  H u0 p0 c0 {1,S}
10 H u0 p0 c0 {1,S}
11 H u0 p0 c0 {2,S}
12 H u0 p0 c0 {3,S}
13 H u0 p0 c0 {4,S}
14 H u0 p0 c0 {4,S}
15 H u0 p0 c0 {5,S}
16 H u0 p0 c0 {5,S}
17 H u0 p0 c0 {6,S}
18 H u0 p0 c0 {7,S}
19 H u0 p0 c0 {8,S}
20 H u0 p0 c0 {8,S}
""", xyz=r_xyz, multiplicity=1)
        p1 = ARCSpecies(label='P1', smiles='C=CC=C', xyz=p_xyz)
        p2 = ARCSpecies(label='P2', smiles='C=CC=C', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts = """C       2.10960889    0.24937609    3.69119479
C       1.39213232    0.67482973    2.64640961
C       1.69841495    1.88777447    1.93743219
C       0.89722030    2.30118941    0.74110454
C       1.82975157    1.87655577   -0.92972510
C       1.05206206    2.33621898   -2.12468685
C       0.27040276    1.40861807   -2.89686464
C      -0.47125439    1.78651122   -3.94304892
H       1.84709533   -0.67230336    4.20106003
H       2.96741452    0.79874269    4.06586828
H       0.54043583    0.08813501    2.30879667
H       2.55074882    2.49044535    2.23465023
H      -0.09116975    1.82488581    0.75488500
H       0.72067762    3.38234797    0.80457194
H       2.80815876    2.37307729   -0.94661245
H       2.02842823    0.79908706   -0.98938000
H       1.04303838    3.39278251   -2.37291342
H       0.29098789    0.36007273   -2.60753743
H      -1.04834025    1.05626254   -4.50168832
H      -0.52469319    2.81976569   -4.27099000"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_2_plus_2_cycloaddition(self):
        """Test the interpolate_isomerization() function for 2+2_cycloaddition: CC1(C)CCO1 <=> C=C(C)C + C=O"""
        r_xyz = """C      -0.13115146    1.50252715    0.73546911
C      -0.20787932    0.18760574   -0.02702719
C      -1.62791936   -0.34626546   -0.14857518
C       0.84747885   -0.85869460    0.37427139
C       1.40805028   -0.64229608   -1.02737576
O       0.38919784    0.33317518   -1.34236129
H      -0.51513998    1.38484842    1.75432119
H       0.90049618    1.86443879    0.80640249
H      -0.72129285    2.28005640    0.23863695
H      -2.27074754    0.36387933   -0.67973156
H      -2.06342103   -0.52321307    0.84055723
H      -1.65386185   -1.29272709   -0.69979737
H       0.47114240   -1.86843265    0.56122890
H       1.52392109   -0.56805081    1.18303912
H       1.35227248   -1.50873254   -1.69224550
H       2.40891708   -0.20357298   -1.06815150"""
        p1_xyz = """C      -0.78808508   -1.31685309    0.20647447
C      -0.10495450   -0.17537411    0.02749750
C      -0.75338720    1.17147536    0.16820738
C       1.35432280   -0.16734028   -0.32564966
H      -1.84354122   -1.31372629    0.46178918
H      -0.31134041   -2.28697931    0.10277938
H      -0.26058038    1.74962274    0.95655854
H      -1.81081982    1.05845304    0.42834727
H      -0.69761470    1.75386040   -0.75753528
H       1.51026426    0.33854032   -1.28396988
H       1.73082508   -1.19212091   -0.40738892
H       1.95720910    0.35086319    0.42752746"""
        p2_xyz = """C      -0.01220431    0.00177310   -0.00001455
O       1.19970303   -0.17429820    0.00143077
H      -0.72773088   -0.83593982   -0.00083194
H      -0.45976783    1.00846493   -0.00058428"""
        r = ARCSpecies(label='R', smiles='CC1(C)CCO1', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='C=C(C)C', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='C=O', xyz=p2_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts = """C      -0.13115146    1.50252715    0.73546911
C      -0.20787932    0.18760574   -0.02702719
C      -1.62791936   -0.34626546   -0.14857518
C       0.84747885   -0.85869460    0.37427139
C       1.57012824   -0.59161833   -1.40776994
O       0.55127580    0.38385293   -1.72275547
H      -0.51513998    1.38484842    1.75432119
H       0.90049618    1.86443879    0.80640249
H      -0.72129285    2.28005640    0.23863695
H      -2.27074754    0.36387933   -0.67973156
H      -2.06342103   -0.52321307    0.84055723
H      -1.65386185   -1.29272709   -0.69979737
H       0.47114240   -1.86843265    0.56122890
H       1.52392109   -0.56805081    1.18303912
H       1.51435044   -1.45805479   -2.07263968
H       2.57099504   -0.15289523   -1.44854568"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_6_membered_central_c_c_shift(self):
        """Test the interpolate_isomerization() function for 6_membered_central_C-C_shift: C#CCCC#C <=> C=C=CC=C=C"""
        r_xyz = """C       3.03272979   -0.11060195   -0.24229461
                   C       1.85599055   -0.34675713   -0.20247149
                   C       0.41485966   -0.64142590   -0.15352412
                   C      -0.41485965    0.64142578   -0.17240633
                   C      -1.85599061    0.34675702   -0.12346178
                   C      -3.03272995    0.11060190   -0.08364096
                   H       4.07762286    0.09693448   -0.27758589
                   H       0.19106566   -1.21954180    0.75163518
                   H       0.14301783   -1.27648597   -1.00582442
                   H      -0.19106412    1.21954271   -1.07756459
                   H      -0.14301928    1.27648492    0.67989514
                   H      -4.07762310   -0.09693448   -0.04835177"""
        p_xyz = """C      -3.03124363    0.21595810   -0.01068883
                   C      -1.77136356   -0.00875193   -0.22839960
                   C      -0.51035344   -0.23538255   -0.44913569
                   C       0.51035356    0.23538291    0.44913621
                   C       1.77136365    0.00875234    0.22839985
                   C       3.03124358   -0.21595777    0.01068824
                   H      -3.50880107    1.10742857   -0.40051872
                   H      -3.62554573   -0.48341738    0.56587595
                   H      -0.21235801   -0.79338469   -1.33170668
                   H       0.21235823    0.79338484    1.33170737
                   H       3.50880076   -1.10742925    0.40051615
                   H       3.62554580    0.48341866   -0.56587535"""
        r = ARCSpecies(label='R', smiles='C#CCCC#C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C=C=CC=C=C', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = []
        for w in get_weight_grid(rxn):
            result = interpolate_isomerization(rxn, weight=w)
            if result:
                ts_xyzs.extend(result)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts_0 = """C      -0.76204028    1.05907311    1.42221741
C      -0.76204028    1.05907311    0.18662557
C      -0.76204028    1.05907311   -1.22437195
C       1.31080316   -0.58466070   -1.19036536
C       0.81934425   -1.15049420    0.00513214
C       0.38898017   -1.64598713    1.05201352
H      -0.76204028    1.05907311    2.51221741
H      -0.44469703    2.03651776   -1.58768339
H      -1.64145208    0.52734039   -1.58768339
H       2.21879876   -0.02001848   -0.97865056
H       1.21475151   -1.30732481   -2.00069286
H       0.00932660   -2.08309531    1.97553916"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_0)))

    def test_interpolate_baeyer_villiger_step2(self):
        """Test the interpolate_isomerization() function for Baeyer-Villiger_step2: CC(=O)OOC(C)(C)O <=> COC(C)=O + CC(=O)O"""
        r_xyz = """C       3.24017953   -0.08055947    0.04152133
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
H       0.45889563   -0.75760809    2.70737582"""
        p1_xyz = """C       1.84132160   -0.12640362   -0.04096862
O       0.49846342   -0.57881995    0.13657708
C       0.16776463   -1.64442042   -0.64320962
C      -1.25300465   -2.04486529   -0.38785528
O       0.91103962   -2.20153671   -1.43841944
H       2.00107206    0.20549505   -1.07165741
H       2.51133583   -0.96701748    0.15958434
H       2.07207904    0.67666423    0.66601746
H      -1.92199213   -1.21750580   -0.63744023
H      -1.37677453   -2.28540712    0.67195610
H      -1.49996746   -2.93147666   -0.97716863"""
        p2_xyz = """C      -0.96060243   -0.07364779   -0.00090814
C       0.48592618    0.29350855   -0.03718140
O       1.29644959    0.14042367    0.86002941
O       0.84722479    0.84115942   -1.21216716
H      -1.56574114    0.82659924   -0.14053059
H      -1.18541862   -0.77413428   -0.80892649
H      -1.20711897   -0.51300925    0.96878687
H       1.80251143    1.03132880   -1.10238169"""
        r = ARCSpecies(label='R', smiles='CC(=O)OOC(C)(C)O', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='COC(C)=O', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='CC(=O)O', xyz=p2_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts = """C       3.65102740   -0.13596094   -0.08044385
C       2.22814803   -0.04033353    0.37774175
O       1.81543082    0.78714154    1.18259985
O       1.48909954   -1.03169489   -0.21336761
O      -0.31730507   -0.78810723    0.32288555
C      -0.58258035   -1.70130366    1.39489078
C      -1.89563521   -1.27490637    2.04480283
C      -0.69557405   -3.11930934    0.84358915
O       0.44369603   -1.67793216    2.38179554
H       4.08918008   -1.07473389    0.26711323
H       4.22325274    0.69661238    0.33872753
H       3.69571498   -0.07005316   -1.17053838
H      -2.71999820   -1.28689033    1.32385177
H      -1.81855771   -0.25064710    2.42803142
H      -2.14921491   -1.91945835    2.89304406
H       0.25555438   -3.44404747    0.40665792
H      -1.45267433   -3.18096845    0.05484522
H      -0.94129844   -3.83481939    1.63561633
H       1.06124365   -0.56816924    1.84192409"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_birad_recombination(self):
        """Test the interpolate_isomerization() function for Birad_recombination: [CH2]C=CCC[CH]C=C <=> C=CC1CC=CCC1"""
        r_xyz = """C      -3.42138076    0.29243388   -0.27263839
C      -2.35329501    1.00253201   -0.91475611
C      -2.26681367    2.33969988   -0.91475581
C      -1.15291243    3.10388081   -1.57636523
C      -1.67988798    3.99943131   -2.70135989
C      -0.57837573    4.79487010   -3.33204422
C      -0.00406804    4.39231449   -4.58735676
C       1.00786735    5.05073600   -5.16166572
H      -3.43435567   -0.79065851   -0.30578932
H      -4.21355207    0.81736825    0.24757267
H      -1.59831501    0.39883423   -1.41404612
H      -3.02370217    2.93515557   -0.40787658
H      -0.39507200    2.41612623   -1.97286786
H      -0.65427541    3.71452896   -0.81315604
H      -2.19975990    3.39366020   -3.45418412
H      -2.42271002    4.70136943   -2.30194855
H      -0.18009923    5.65501064   -2.80329725
H      -0.41769654    3.51439040   -5.07908540
H       1.41503188    4.71576696   -6.11055468
H       1.45458039    5.93140659   -4.71125121"""
        p_xyz = """C                 -2.65521195    1.87064119   -0.02446857
 C                 -1.75380051    1.72509734   -1.02588828
 C                 -1.68502567    2.76737849   -2.15748893
 C                 -0.69117350    3.87873009   -1.80743110
 C                 -0.81374282    4.97784207   -2.87784360
 C                 -0.64598633    4.39470644   -4.29601029
 C                 -0.95261513    3.09679062   -4.56884843
 C                 -1.18973519    2.07384720   -3.43901935
 H                 -2.70299514    1.14646035    0.76177495
 H                 -3.31913858    2.70973702   -0.02003917
 H                 -1.08987499    0.88600063   -1.03031856
 H                 -2.65503312    3.19514367   -2.30241132
 H                  0.29828156    3.47209637   -1.78445952
 H                 -0.92091525    4.30230089   -0.85207436
 H                 -1.79045635    5.40958666   -2.81055850
 H                 -0.07363596    5.73055783   -2.70302702
 H                 -0.29345520    5.02690638   -5.08401095
 H                 -1.01810056    2.77655786   -5.58770202
 H                 -1.90841763    1.34528125   -3.75140886
 H                 -0.25775968    1.59319508   -3.22619270"""
        r = ARCSpecies(label='R', smiles='[CH2]C=CCC[CH]C=C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C=CC1CC=CCC1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = []
        for w in get_weight_grid(rxn):
            result = interpolate_isomerization(rxn, weight=w)
            if result:
                ts_xyzs.extend(result)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts = """C      -3.42138076    0.29243388   -0.27263839
C      -2.35329501    1.00253201   -0.91475611
C      -2.37485302    2.34059810   -0.84684933
C      -3.09184975    2.96225599    0.32016937
C      -3.38737296    2.06374739    1.52460918
C      -2.48907594    0.88475036    1.74104228
C      -2.98797526   -0.32091656    2.34532423
C      -2.20917897   -1.37879488    2.59378488
H      -4.35304411    0.81373267   -0.05274985
H      -3.32095412   -0.75896176   -0.00322849
H      -1.59692473    0.39772253   -1.41496555
H      -1.89425690    2.93478064   -1.62407161
H      -4.04742493    3.38092690    0.00443097
H      -2.49151734    3.79555128    0.68529942
H      -3.37990998    2.65193259    2.44225970
H      -4.38948670    1.64411965    1.43640141
H      -1.43375920    0.96874721    1.48152137
H      -4.04514257   -0.36114745    2.60777183
H      -2.62447478   -2.27551454    3.05370144
H      -1.14680228   -1.38795604    2.35012148"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))
        self.assertEqual(len(ts_xyzs), 1)

    def test_interpolate_concerted_intra_diels_alder_monocyclic_1_2_shifth(self):
        """Test the interpolate_isomerization() function for Concerted_Intra_diels_alder_mono-cyclic_1,2_shiftH: C#CC=CC=C <=> [C]1C=CC=CC1"""
        r_xyz = """C       3.25931086   -0.62469725   -0.85172041
                   C       2.21431341   -0.47587355   -0.28012458
                   C       0.97982054   -0.30103766    0.40159285
                   C      -0.13573006    0.04859177   -0.25399636
                   C      -1.40841692    0.23576534    0.40283655
                   C      -2.52147798    0.58481406   -0.25261333
                   H       4.18607967   -0.75650475   -1.35982959
                   H       0.97405553   -0.46156800    1.47539715
                   H      -0.10113962    0.20314181   -1.33112279
                   H      -1.45501699    0.08400757    1.47935301
                   H      -3.45759474    0.71479393    0.28157503
                   H      -2.53403295    0.74932608   -1.32550473"""
        # P is a singlet carbene (cyclohexadienylidene); must specify multiplicity=1 so
        # that RMG interprets [C] as a lone-pair carbene rather than a triplet biradical.
        r = ARCSpecies(label='R', smiles='C#CC=CC=C', xyz=r_xyz)
        p = ARCSpecies(label='P', adjlist="""1     C u0 p0 c0 {2,S} {6,S} {7,S} {8,S}
2     C u0 p0 c0 {1,S} {3,D} {9,S}
3     C u0 p0 c0 {2,D} {4,S} {10,S}
4     C u0 p0 c0 {3,S} {5,D} {11,S}
5     C u0 p0 c0 {4,D} {6,S} {12,S}
6     C u0 p1 c0 {1,S} {5,S}
7     H u0 p0 c0 {1,S}
8     H u0 p0 c0 {1,S}
9     H u0 p0 c0 {2,S}
10    H u0 p0 c0 {3,S}
11    H u0 p0 c0 {4,S}
12    H u0 p0 c0 {5,S}""")
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertIsNotNone(ts_xyzs)
        self.assertIsInstance(ts_xyzs, list)
        self.assertGreater(len(ts_xyzs), 0, 'Expected at least one TS guess for Concerted_Intra_Diels_alder_monocyclic_1,2_shiftH')
        expected_ts = """C       3.25931086   -0.62469725   -0.85172041
C       2.21431341   -0.47587355   -0.28012458
C       1.98480797   -0.63419881    1.11326670
C       2.89460169   -0.97770151    2.03563992
C       4.29204937   -1.25631974    1.79961635
C       4.92103178   -1.21988996    0.61918910
H       2.12504139   -0.23109108   -1.76747526
H       0.97240698   -0.46160558    1.47844588
H       2.59583552   -1.06787309    3.08000952
H       4.91203689   -1.52458145    2.65504088
H       4.97724376   -2.11141797   -0.00540519
H       5.39439525   -0.31403670    0.24043207"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))
        # Verify the migrating H (atom 6) is properly positioned between
        # the donor C0 and acceptor C1 — must be ≈ equidistant for a 1,2-shiftH TS.
        coords = np.array(ts_xyzs[0]['coords'])
        d_C0_H6 = float(np.linalg.norm(coords[0] - coords[6]))
        d_C1_H6 = float(np.linalg.norm(coords[1] - coords[6]))
        self.assertAlmostEqual(d_C0_H6, d_C1_H6, places=1,
                               msg='Migrating H should be equidistant from donor C0 and acceptor C1')

    def test_interpolate_cyclic_ether_formation(self):
        """Test the interpolate_isomerization() function for Cyclic_Ether_Formation: [CH2]C(=C)OO <=> C=C1CO1 + OH"""
        r_xyz = """C      -1.16599016   -1.08405507   -0.12511934
C      -0.13200032   -0.08872899   -0.09899028
C       0.96424520   -0.23895222    0.65844877
O      -0.42464665    1.01074511   -0.88710064
O       0.80541183    1.75280799   -1.11832697
H      -2.07689019   -0.89522149   -0.68069521
H      -1.08196804   -1.99246915    0.45784734
H       1.73039711    0.52686865    0.71024800
H       1.11996807   -1.12302241    1.26725079
H       0.98119234    1.37833135   -1.99878574"""
        p1_xyz = """C       1.38510201   -0.01635125   -0.06013688
C       0.08743442   -0.21093250    0.02765715
C      -1.24562336    0.19764018    0.02666851
O      -0.75483022   -1.21460370    0.21611443
H       1.79164035    0.97382819   -0.22688377
H       2.07415766   -0.84800941    0.03335039
H      -1.71019704    0.42239623   -0.92142615
H      -1.62768381    0.69603226    0.90465634"""
        p2_xyz = """O       0.00000000    0.00000000    0.61310000
H       0.00000000    0.00000000   -0.61310000"""
        r = ARCSpecies(label='R', smiles='[CH2]C(=C)OO', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='C=C1CO1', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='[OH]', xyz=p2_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        expected_ts = """C      -1.19597022   -1.03733085   -0.17267513
C      -0.21889171   -0.09678803   -0.14798422
C       1.15457463   -0.28500002    0.80099757
O       0.48369452    1.15860461   -0.41011226
O       0.93622737    2.28816293   -1.86932849
H      -2.11229997   -0.84737167   -0.73156269
H      -1.11137750   -1.95191402    0.41425051
H       1.32109836   -0.33955289    1.87681997
H       2.00250068   -0.96891489    0.76382419
H       1.10969504    1.91861349   -2.73820257"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_cyclic_thioether_formation(self):
        """Test the interpolate_isomerization() function for Cyclic_Thioether_Formation: [CH2]C(=C)SOC <=> C=C1CS1 + C[O]"""
        r_xyz = """C      -2.38998392    1.02801938    0.53011734
C      -1.59458008    0.13221986   -0.26760278
C      -1.26470130   -1.08353960    0.18831467
S      -1.09404654    0.61661164   -1.86393561
O      -0.71801591    2.21563102   -1.57500019
C       0.66541738    2.45495600   -1.35446238
H      -2.75410264    1.95870709    0.11013214
H      -2.70765363    0.74987333    1.52717668
H      -0.67877532   -1.77763900   -0.40662986
H      -1.57107852   -1.42699131    1.17095577
H       1.25567694    2.16776159   -2.23113917
H       0.80258723    3.52699316   -1.18607057
H       1.01985923    1.92132948   -0.46620689"""
        p1_xyz = """C       1.37716827   -0.01852287   -0.16648584
C       0.10011977    0.25084724   -0.01648424
C      -1.19831097   -0.25951331    0.14965168
S      -0.91645919    1.54517459    0.08416411
H       1.73267577   -1.04204091   -0.19176699
H       2.10749633    0.77698062   -0.26876426
H      -1.48965477   -0.61044483    1.12696076
H      -1.71303523   -0.64248052   -0.71727522"""
        p2_xyz = """C       0.03807240    0.00035621   -0.00484242
O       1.35198769    0.01264937   -0.17195885
H      -0.33965241   -0.14992727    1.02079480
H      -0.51702680    0.90828035   -0.29592912
H      -0.53338088   -0.77135867   -0.54806440"""
        r = ARCSpecies(label='R', smiles='[CH2]C(=C)SOC', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='C=C1CS1', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='C[O]', xyz=p2_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts_0 = """C      -2.49431891    1.11277273    0.66895182
C      -1.56587336    0.06713900   -0.26219745
C      -1.24169463   -1.12761308    0.18584210
S      -1.66605420    1.48782593   -1.52728511
O       0.06231531    2.66052949   -1.60215304
C       1.44574860    2.89985447   -1.38161523
H      -3.47048819    1.11133912    1.15391726
H      -2.42808794    1.57300574    1.65480121
H      -0.65352232   -1.82437353   -0.41138333
H      -1.54945959   -1.47262047    1.17293411
H       2.03320901    2.61402200   -2.25413461
H       1.58243982    3.96815090   -1.21381101
H       1.79851918    2.36874413   -0.49754808"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_0)))

    def test_interpolate_cyclopentadiene_scission(self):
        """Test the interpolate_isomerization() function for Cyclopentadiene_scission: C1=CC2CC2=C1 <=> [C]1C=CC=CC1"""
        r_xyz = """C       1.46526410    0.35550989    0.15268159
                   C       0.45835467    1.13529910   -0.26555553
                   C      -0.75500438    0.35970165   -0.56989350
                   C      -1.48532781   -0.35660657    0.46119178
                   C      -0.34144780   -1.06077923   -0.11686057
                   C       0.98794173   -1.00683992    0.12489717
                   H       2.46308379    0.66299943    0.41975788
                   H       0.51108826    2.21009512   -0.37348204
                   H      -1.11928864    0.38428608   -1.58978132
                   H      -2.45322496   -0.77587088    0.21588387
                   H      -1.38590137   -0.05438209    1.49711542
                   H       1.65446241   -1.85341259    0.04404524"""
        p_xyz = """C       0.32377429   -1.34449694   -0.14122370
                   C       1.47572737   -0.75313342   -0.07853288
                   C       1.30166202    0.68051955    0.07213267
                   C      -0.03314415    1.28528351    0.13512766
                   C      -1.14328725    0.51323467    0.05345670
                   C      -1.10099711   -1.00983052   -0.10666836
                   H       2.43679364   -1.24237771   -0.12954979
                   H       2.17524509    1.31823771    0.13957240
                   H      -0.09926704    2.36242259    0.24835509
                   H      -2.12527706    0.97366262    0.10143239
                   H      -1.60600902   -1.48495894    0.73931727
                   H      -1.60522077   -1.29856310   -1.03341945"""
        # P is a singlet carbene (cyclohexadienylidene); must specify multiplicity=1 so
        # that RMG interprets [C] as a lone-pair carbene rather than a triplet biradical.
        r = ARCSpecies(label='R', smiles='C1=CC2CC2=C1', xyz=r_xyz)
        p = ARCSpecies(label='P', adjlist="""multiplicity 1
1  C u0 p1 c0 {2,S} {6,S}
2  C u0 p0 c0 {1,S} {3,D} {7,S}
3  C u0 p0 c0 {2,D} {4,S} {8,S}
4  C u0 p0 c0 {3,S} {5,D} {9,S}
5  C u0 p0 c0 {4,D} {6,S} {10,S}
6  C u0 p0 c0 {1,S} {5,S} {11,S} {12,S}
7  H u0 p0 c0 {2,S}
8  H u0 p0 c0 {3,S}
9  H u0 p0 c0 {4,S}
10 H u0 p0 c0 {5,S}
11 H u0 p0 c0 {6,S}
12 H u0 p0 c0 {6,S}""", xyz=p_xyz, multiplicity=1)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertIsNotNone(ts_xyzs)
        self.assertIsInstance(ts_xyzs, list)
        self.assertGreaterEqual(len(ts_xyzs), 2)
        good_guesses = 0
        h_c_threshold = get_single_bond_length('H', 'C') * 1.35
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 12)
            self.assertFalse(colliding_atoms(ts_xyz),
                             msg=f'Collision detected in Cyclopentadiene_scission TS guess:\n{xyz_to_str(ts_xyz)}')
            # Count guesses where every H is within 1.35× sbl of its nearest heavy atom.
            coords_arr = np.array(ts_xyz['coords'], dtype=float)
            all_h_ok = True
            for h_idx, sym in enumerate(ts_xyz['symbols']):
                if sym != 'H':
                    continue
                dists = [float(np.linalg.norm(coords_arr[h_idx] - coords_arr[j]))
                         for j, sj in enumerate(ts_xyz['symbols']) if sj != 'H']
                if min(dists) >= h_c_threshold:
                    all_h_ok = False
                    break
            if all_h_ok:
                good_guesses += 1
        self.assertGreaterEqual(good_guesses, 1,
                                msg='No Cyclopentadiene_scission TS guess has all H atoms properly bonded.')
        # The breaking C2-C4 bond (cyclopropane → 6-ring) should be stretched in the TS
        # beyond its R value (1.547 Å) but shorter than P (2.451 Å).
        coords_0 = np.array(ts_xyzs[0]['coords'], dtype=float)
        d_breaking = float(np.linalg.norm(coords_0[2] - coords_0[4]))
        self.assertGreater(d_breaking, 1.6, msg=f'Breaking bond C2-C4 not stretched: {d_breaking:.3f} A')
        self.assertLess(d_breaking, 2.4, msg=f'Breaking bond C2-C4 too stretched: {d_breaking:.3f} A')
        expected_ts_0 = """C      -0.42349082   -1.10881401    0.80262047
C      -0.42349082   -1.10881401   -0.53784960
C       0.89515869   -0.28477860   -0.56148250
C       0.74487960    1.13467640   -0.90323448
C      -0.33334789    1.10045305    0.08523425
C      -0.42349082    0.27094090    1.22792263
H      -0.46220619   -1.96458110    1.45659866
H      -0.44236790   -1.99522514   -1.15715124
H      -0.01231185    0.07605252   -1.04562930
H       0.78417824    2.06095352   -1.46292151
H       1.71794091    0.71254954   -0.68114545
H      -0.79144499    0.57952750    2.19583601"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_0)))

    def test_interpolate_diels_alder_addition(self):
        """Test the interpolate_isomerization() function for Diels_alder_addition: C=CC(=C)C + C=CC=O <=> CC1=CCC(C=O)CC1"""
        r1_xyz = """C       1.97753426   -0.34691463   -0.12195850
C       0.96032171    0.45485914   -0.46215363
C      -0.43629664    0.27157147   -0.09968556
C      -1.35584640    1.15966116   -0.51269091
C      -0.83651671   -0.91436221    0.73635894
H       2.98719352   -0.11575642   -0.44772907
H       1.84910220   -1.24076974    0.47792776
H       1.19368072    1.33006788   -1.06832846
H      -2.40510842    1.04750710   -0.25687679
H      -1.09525737    2.02366247   -1.11636739
H      -0.32888591   -0.89422114    1.70676182
H      -1.91408642   -0.93005704    0.93479551
H      -0.58767904   -1.85093188    0.22577726"""
        r2_xyz = """C      -1.22034116   -0.10890246    0.02353603
C      -0.04004107    0.51094374   -0.08149118
C       1.22322531   -0.24393463    0.03286276
O       2.30875132    0.31445302   -0.06186255
H      -1.30612429   -1.17741471    0.19480533
H      -2.14393224    0.45618508   -0.06217786
H       0.04657041    1.57753840   -0.25245803
H       1.13189173   -1.32886845    0.20678550"""
        p_xyz = """C       2.60098776   -0.04177774    0.73723478
C       1.20465630    0.10105432    0.20245819
C       0.16278370   -0.55312927    0.74494799
C      -1.24024239   -0.46705077    0.21761600
C      -1.33954822    0.16452081   -1.17701034
C      -1.06935354   -0.87644399   -2.25040126
O      -0.50075393   -0.64415323   -3.31363975
C      -0.41124651    1.37364733   -1.29938488
C       1.04721460    1.02438027   -0.98148987
H       3.26841747   -0.42094194   -0.04336972
H       2.64920967   -0.73532885    1.58328037
H       2.97843218    0.92762356    1.07822844
H       0.31418172   -1.19138627    1.61332708
H      -1.82762138    0.12846013    0.92764672
H      -1.67646259   -1.47309646    0.21384290
H      -2.37737283    0.48324136   -1.33650826
H      -1.50255476   -1.87505625   -2.06737417
H      -0.75069363    2.15000964   -0.60076538
H      -0.46865428    1.81280411   -2.30253884
H       1.51473571    0.55339822   -1.85465668
H       1.59082870    1.95894204   -0.79688170"""
        r1 = ARCSpecies(label='R1`', smiles='C=CC(=C)C', xyz=r1_xyz)
        r2 = ARCSpecies(label='R2', smiles='C=CC=O', xyz=r2_xyz)
        p = ARCSpecies(label='P', smiles='CC1=CCC(C=O)CC1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        self.assertEqual(len(ts_xyzs), 1)
        expected_ts_0 = """C       2.60098776   -0.04177774    0.73723478
C       1.20465630    0.10105432    0.20245819
C       0.16278370   -0.55312927    0.74494799
C      -1.24024239   -0.46705077    0.21761600
C      -1.60373374    0.33086364   -1.46744104
C      -1.33353906   -0.71010116   -2.54083196
O      -0.76493945   -0.47781040   -3.60407045
C      -0.67543203    1.53999016   -1.58981558
C       1.04721460    1.02438027   -0.98148987
H       3.26841747   -0.42094194   -0.04336972
H       2.64920967   -0.73532885    1.58328037
H       2.97843218    0.92762356    1.07822844
H       0.31418172   -1.19138627    1.61332708
H      -1.82762138    0.12846013    0.92764672
H      -1.67646259   -1.47309646    0.21384290
H      -2.64155835    0.64958419   -1.62693896
H      -1.76674028   -1.70871342   -2.35780487
H      -1.01487915    2.31635247   -0.89119608
H      -0.73283980    1.97914694   -2.59296954
H       1.51473571    0.55339822   -1.85465668
H       1.59082870    1.95894204   -0.79688170"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_0)))

    def test_interpolate_diels_alder_addition_aromatic(self):
        """Test the interpolate_isomerization() function for Diels_alder_addition_Aromatic: c1ccc(c2ccccc2)cc1 + C#C <=> C1=CC2=C3C=CC=CC3C=CC2C=C1"""
        r1_xyz = """C      -3.54205523   -0.07502317    0.30513224
C      -2.74043148   -0.29957961    1.42136446
C      -1.34835104   -0.27048190    1.30036082
C      -0.73330045   -0.01553155    0.06317038
C       0.73330040    0.01553172   -0.06317075
C       1.52075913   -1.06312546    0.37310262
C       2.91293649   -1.03477669    0.25303904
C       3.54205545    0.07502283   -0.30513215
C       2.78069878    1.15537130   -0.74352022
C       1.38854647    1.12474630   -0.62372646
C      -1.56095452    0.20886131   -1.04973751
C      -2.95320368    0.17898471   -0.93088346
H      -4.62453844   -0.09795097    0.39838347
H      -3.19618377   -0.49503794    2.38842155
H      -0.74031612   -0.43715107    2.18730171
H       1.04824701   -1.94475257    0.80160610
H       3.50339361   -1.88128316    0.59348104
H       4.62453868    0.09795049   -0.39838310
H       3.26815880    2.02470885   -1.17681965
H       0.81310763    1.98417762   -0.96195305
H      -1.12103893    0.39772702   -2.02695465
H      -3.57536879    0.35161193   -1.80508243"""
        r2_xyz = """C       0.59049692   -0.03011357   -0.00178706
C      -0.59049692   -0.24451188    0.00178707
H       1.63915622    0.16026061   -0.00496073
H      -1.63915622   -0.43488608    0.00496071"""
        p_xyz = """C       2.84332690    1.43980589    0.27287352
C       1.60930875    1.53809494   -0.24535975
C       0.72702818    0.37582523   -0.37603061
C      -0.62707838    0.45548717   -0.45882195
C      -1.37683626    1.71376941   -0.42793406
C      -2.66761289    1.76401272   -0.06406835
C      -3.38652298    0.58335309    0.34272824
C      -2.85413040   -0.63087834    0.14324093
C      -1.50141516   -0.81577713   -0.51586378
C      -0.79508961   -2.03171780    0.03545841
C       0.54207612   -2.11038280    0.11721475
C       1.44728414   -0.98924836   -0.33557607
C       2.72223988   -0.95893399    0.48418891
C       3.36464775    0.18618427    0.75550242
H       3.46972453    2.32060219    0.37740412
H       1.25324186    2.51706066   -0.54382089
H      -0.87526333    2.64228029   -0.67395693
H      -3.19485345    2.71267712   -0.03007075
H      -4.36833544    0.70345176    0.78898062
H      -3.42981675   -1.51272906    0.40658279
H      -1.70908628   -1.02855689   -1.57512766
H      -1.39270442   -2.89762608    0.30597226
H       0.99715134   -3.03822026    0.45209207
H       1.75573450   -1.23239206   -1.36328350
H       3.15470656   -1.90009373    0.80917363
H       4.29227483    0.19395179    1.31850164"""
        r1 = ARCSpecies(label='R1`', smiles='c1ccc(c2ccccc2)cc1', xyz=r1_xyz)
        r2 = ARCSpecies(label='R2', smiles='C#C', xyz=r2_xyz)
        p = ARCSpecies(label='P', smiles='C1=CC2=C3C=CC=CC3C=CC2C=C1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        self.assertEqual(len(ts_xyzs), 1)
        expected_ts_0 = """C       2.84332690    1.43980589    0.27287352
C       1.60930875    1.53809494   -0.24535975
C       0.72702818    0.37582523   -0.37603061
C      -0.62707838    0.45548717   -0.45882195
C      -1.37683626    1.71376941   -0.42793406
C      -2.66761289    1.76401272   -0.06406835
C      -3.38652298    0.58335309    0.34272824
C      -2.85413040   -0.63087834    0.14324093
C      -1.50141516   -0.81577713   -0.51586378
C      -0.83013445   -2.44353093    0.21239193
C       0.50703128   -2.52219593    0.29414827
C       1.44728414   -0.98924836   -0.33557607
C       2.72223988   -0.95893399    0.48418891
C       3.36464775    0.18618427    0.75550242
H       3.46972453    2.32060219    0.37740412
H       1.25324186    2.51706066   -0.54382089
H      -0.87526333    2.64228029   -0.67395693
H      -3.19485345    2.71267712   -0.03007075
H      -4.36833544    0.70345176    0.78898062
H      -3.42981675   -1.51272906    0.40658279
H      -1.70908628   -1.02855689   -1.57512766
H      -1.42774926   -3.30943921    0.48290578
H       0.96210650   -3.45003339    0.62902559
H       1.75573450   -1.23239206   -1.36328350
H       3.15470656   -1.90009373    0.80917363
H       4.29227483    0.19395179    1.31850164"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_0)))

    def test_interpolate_ho2_elimination_from_peroxyradical(self):
        """Test the interpolate_isomerization() function for HO2_Elimination_from_PeroxyRadical: CC(O)O[O] <=> CC=O + [O]O"""
        r_xyz = """C      -0.52792548   -0.77076774    0.66788608
C       0.05785510    0.24175984   -0.31171822
O      -0.39237493   -0.04439467   -1.61114071
O       1.45907821    0.15459538   -0.27029612
O       1.91141077    1.18016050   -0.96705788
H      -0.22397300   -1.80064794    0.38283077
H      -0.16353301   -0.55780966    1.69534964
H      -1.63684613   -0.70714141    0.66410656
H      -0.27242547    1.26269634    0.00221505
H      -1.32539776    0.29011420   -1.67064732"""
        p1_xyz = """C      -0.63794889   -0.02183602   -0.12665179
C       0.82935756    0.03421668    0.16760747
O       1.28869095    0.11790235    1.30180467
H      -0.90823756    0.82628235   -0.75995134
H      -1.19318200    0.02888417    0.81280164
H      -0.88147907   -0.95354223   -0.64315594
H       1.49366985   -0.00469550   -0.71211622"""
        p2_xyz = """O       0.99505451   -0.18344310    0.00000000
O      -0.15754548    0.45167049    0.00000000
H      -0.83750903   -0.26822739    0.00000000"""
        r = ARCSpecies(label='R', smiles='CC(O)O[O]', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='CC=O', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='[O]O', xyz=p2_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertEqual(len(ts_xyzs), 1)
        expected_xyz = """C      -0.52792548   -0.77076774    0.66788608
C       0.05785510    0.24175984   -0.31171822
O      -0.39237493   -0.04439467   -1.61114071
O       1.90348295    0.12695075   -0.25715890
O       2.35581551    1.15251587   -0.95392066
H      -0.22397300   -1.80064794    0.38283077
H      -0.16353301   -0.55780966    1.69534964
H      -1.63684613   -0.70714141    0.66410656
H      -0.27242547    1.26269634    0.00221505
H       0.84348057    0.49385350   -1.31559023"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_xyz)))

    def test_interpolate_intra_2_plus_2_cycloaddition_cd(self):
        """Test the interpolate_isomerization() function for intra_2+2_cycloaddition_Cd: C=CC=C <=> C1=CCC1"""
        r_xyz = """C       1.82756305    0.09282277   -0.10513133
                   C       0.58818449   -0.41022877   -0.07453536
                   C      -0.58817345    0.41021329    0.07471351
                   C      -1.82755191   -0.09284018    0.10530914
                   H       2.02221248    1.15701349   -0.01869217
                   H       2.68495777   -0.56301880   -0.21927041
                   H       0.45128207   -1.48580231   -0.16562548
                   H      -0.45127003    1.48578749    0.16580454
                   H      -2.02220285   -1.15703226    0.01886889
                   H      -2.68494614    0.56300277    0.21944957"""
        p_xyz = """C      -0.06902948    0.84060039    0.77804358
                   C      -1.09821853    0.32180183    0.08417575
                   C      -0.22613156   -0.65945413   -0.66625740
                   C       0.95345020   -0.06484505    0.12900378
                   H       0.01489202    1.62040295    1.52931163
                   H      -2.16551579    0.52129254    0.05930486
                   H      -0.41639366   -1.72015227   -0.46931593
                   H      -0.16106002   -0.51376156   -1.75004960
                   H       1.45633649   -0.77613764    0.79325853
                   H       1.71167034    0.43025295   -0.48747520"""
        r = ARCSpecies(label='R', smiles='C=CC=C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C1=CCC1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertIsNotNone(ts_xyzs)
        self.assertIsInstance(ts_xyzs, list)
        self.assertEqual(len(ts_xyzs), 2)
        expected_ts_0 = """C      -0.15879255   -0.65216693   -1.07687008
                           C       0.12683547    0.66992610   -0.74741966
                           C       0.12683547    0.66992610    0.69452995
                           C      -0.15879278   -0.65216718    1.14308572
                           H       0.21286829   -1.67415250   -1.00261573
                           H       0.04464552   -0.47618317   -2.13315729
                           H       0.12683471    1.63121728   -1.25714261
                           H       0.12683547    1.63121836    1.20425265
                           H       0.20572116   -1.65450578    0.91825928
                           H       0.06119390   -0.46186762    2.19355797"""
        expected_ts_1 = """C       0.15879344   -0.61792555   -1.10373556
                           C      -0.12683630    0.70416711   -0.67763656
                           C      -0.12683630    0.70416711    0.66766447
                           C       0.15879333   -0.61792588    1.11622025
                           H       0.89948945   -1.40267782   -1.25746680
                           H      -0.79444861   -1.13667640   -1.20535823
                           H      -0.12683645    1.48738111   -1.43004232
                           H      -0.12683630    1.48738100    1.42007052
                           H      -0.79580439   -1.13741448    1.19973543
                           H       0.90190611   -1.40523957    1.24279255"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_0)))
        self.assertTrue(almost_equal_coords(ts_xyzs[1], str_to_xyz(expected_ts_1)))

    def test_interpolate_intra_5_membered_conjugated_c_c_c_c_addition(self):
        """Test the interpolate_isomerization() function for Intra_5_membered_conjugated_C=C_C=C_addition: C=C=CC=C=C <=> C=C1C=C[C]C1"""
        r_xyz = """C      -3.03124363    0.21595810   -0.01068883
C      -1.77136356   -0.00875193   -0.22839960
C      -0.51035344   -0.23538255   -0.44913569
C       0.51035356    0.23538291    0.44913621
C       1.77136365    0.00875234    0.22839985
C       3.03124358   -0.21595777    0.01068824
H      -3.50880107    1.10742857   -0.40051872
H      -3.62554573   -0.48341738    0.56587595
H      -0.21235801   -0.79338469   -1.33170668
H       0.21235823    0.79338484    1.33170737
H       3.50880076   -1.10742925    0.40051615
H       3.62554580    0.48341866   -0.56587535"""
        p_xyz = """C      -1.75380171    0.48873088   -0.19068706
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
H      -0.36583394   -1.89034834    0.81324667"""
        # P is a singlet carbene (methylenecyclobutenyl); must specify multiplicity=1 so
        # that RMG interprets [C] as a lone-pair carbene rather than a triplet biradical.
        r = ARCSpecies(label='R', smiles='C=C=CC=C=C', xyz=r_xyz)
        p = ARCSpecies(label='P', adjlist="""multiplicity 1
1  C u0 p0 c0 {2,D} {7,S} {8,S}
2  C u0 p0 c0 {1,D} {3,S} {6,S}
3  C u0 p0 c0 {2,S} {4,D} {9,S}
4  C u0 p0 c0 {3,D} {5,S} {10,S}
5  C u0 p1 c0 {4,S} {6,S}
6  C u0 p0 c0 {2,S} {5,S} {11,S} {12,S}
7  H u0 p0 c0 {1,S}
8  H u0 p0 c0 {1,S}
9  H u0 p0 c0 {3,S}
10 H u0 p0 c0 {4,S}
11 H u0 p0 c0 {6,S}
12 H u0 p0 c0 {6,S}""", xyz=p_xyz, multiplicity=1)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        # Ring closure with cumulated-bond flex weighting + collinear-substituent
        # bending: the 5-membered ring C1-C2-C3-C4-C5 forms (C1-C5 ≈ 2.3 Å),
        # while the exocyclic C0=C1 is bent to ~120° (sp2) away from the ring.
        expected_ts_0 = """C      -2.66296545   -0.44995581   -1.06241133
C      -1.77136356   -0.00875193   -0.22839960
C      -0.51035344   -0.23538255   -0.44913569
C       0.12419475    0.40075232    0.67471320
C      -0.88676040    0.87806315    1.33832186
C      -2.07366887    1.21190935    1.74450585
H      -2.98644976    0.16150401   -1.90477396
H      -3.10386210   -1.43835763   -0.93290459
H      -0.21122278   -0.79551043   -1.33506888
H       1.21100627    0.36736162    0.75104005
H      -2.63995829    0.55525007    2.40497150
H      -2.52256615    2.15509630    1.43307430"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_0)))

    def test_interpolate_intra_diels_alder_monocyclic(self):
        """Test the interpolate_isomerization() function for Intra_Diels_alder_monocyclic: C=C1C=CC=C1 <=> C1=CC2CC2=C1"""
        r_xyz = """C       1.98835311   -0.06142285   -0.00200142
C       0.65433874   -0.02021339   -0.00065871
C      -0.15871220    1.17598961   -0.05259916
C      -1.43846254    0.76969575   -0.03122551
C      -1.48316252   -0.67944287    0.03416677
C      -0.23088987   -1.16395429    0.05299117
H       2.52365896   -1.00407776    0.03918308
H       2.58073849    0.84639620   -0.04432075
H       0.20218412    2.19006475   -0.09915008
H      -2.31139440    1.40309362   -0.05766688
H      -2.39347065   -1.25775412    0.06240301
H       0.06681877   -2.19837465    0.09887848"""
        p_xyz = """C       1.46526410    0.35550989    0.15268159
C       0.45835467    1.13529910   -0.26555553
C      -0.75500438    0.35970165   -0.56989350
C      -1.48532781   -0.35660657    0.46119178
C      -0.34144780   -1.06077923   -0.11686057
C       0.98794173   -1.00683992    0.12489717
H       2.46308379    0.66299943    0.41975788
H       0.51108826    2.21009512   -0.37348204
H      -1.11928864    0.38428608   -1.58978132
H      -2.45322496   -0.77587088    0.21588387
H      -1.38590137   -0.05438209    1.49711542
H       1.65446241   -1.85341259    0.04404524"""
        r = ARCSpecies(label='R', smiles='C=C1C=CC=C1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C1=CC2CC2=C1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertIsNotNone(ts_xyzs)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 12)
            self.assertFalse(colliding_atoms(ts_xyz))
        # At least one TS guess must preserve the 5-membered ring scaffold (atoms 1-5).
        ring_bonds = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
        has_good = False
        for ts_xyz in ts_xyzs:
            coords = np.array(ts_xyz['coords'])
            if all(1.2 < np.linalg.norm(coords[a] - coords[b]) < 2.0
                   for a, b in ring_bonds):
                has_good = True
        self.assertTrue(has_good, 'No TS guess preserves the 5-membered ring scaffold.')

    def test_interpolate_intra_disproportionation(self):  # TODO: 0 guesses (singlet biradical: constraint lock + backbone drift rejection)
        """Test the interpolate_isomerization() function for Intra_Disproportionation: C=C1[CH]C[C]2C=CC=CC21 <=> C=C1CCC2=C1C=CC=C2"""
        r_xyz = """C      -1.71276869   -2.14835263   -0.29600082
C      -1.30379477   -0.91506552    0.02297736
C       0.02166928   -0.61873233    0.53428358
C       0.12717819    0.82647389    0.91669270
C      -1.28264593    1.32277593    0.84036811
C      -1.85754229    2.26155916    1.60812418
C      -3.29934285    2.39956966    1.60188376
C      -4.08920980    1.47079635    1.03722857
C      -3.52758183    0.33798706    0.32703925
C      -2.07314208    0.38964129   -0.05559628
H      -2.71169105   -2.33428156   -0.67752001
H      -1.05967066   -3.00703332   -0.18456792
H       0.81708685   -1.33998293    0.63942836
H       0.76667301    1.35549222    0.20321493
H       0.54320924    0.92870099    1.92345007
H      -1.28510222    2.85939041    2.30922346
H      -3.72854139    3.24346240    2.13325690
H      -5.17022073    1.55123850    1.09315776
H      -4.19994276   -0.36461812   -0.14850570
H      -1.99779884    0.76292039   -1.08682170"""
        p_xyz = """C      -1.90026741    1.94710942    0.40100525
C      -1.30574300    0.75204573    0.31181024
C      -1.99313175   -0.58484391    0.45348151
C      -0.98838922   -1.64148659   -0.03156125
C       0.29382707   -0.87655954   -0.13843703
C       0.10406239    0.49722480    0.03030449
C       1.17495026    1.37400095   -0.06610665
C       2.44499563    0.85716362   -0.33800141
C       2.63110871   -0.52146580   -0.51101002
C       1.55015774   -1.39914318   -0.41441853
H      -2.96372588    2.02443126    0.60394418
H      -1.35388672    2.87517220    0.27404180
H      -2.92224070   -0.63686629   -0.12431988
H      -2.23780717   -0.76317323    1.50792828
H      -0.90998364   -2.46834709    0.68116405
H      -1.27794500   -2.03248169   -1.01293936
H       1.04129865    2.44161336    0.06970297
H       3.29558312    1.52972246   -0.41593008
H       3.62447660   -0.90832138   -0.72468077
H       1.69266033   -2.46579510   -0.55597779"""
        r = ARCSpecies(label='R', adjlist="""multiplicity 1
1  C u0 p0 c0 {2,D} {11,S} {12,S}
2  C u0 p0 c0 {1,D} {3,S} {10,S}
3  C u1 p0 c0 {2,S} {4,S} {13,S}
4  C u0 p0 c0 {3,S} {5,S} {14,S} {15,S}
5  C u1 p0 c0 {4,S} {6,S} {10,S}
6  C u0 p0 c0 {5,S} {7,D} {16,S}
7  C u0 p0 c0 {6,D} {8,S} {17,S}
8  C u0 p0 c0 {7,S} {9,D} {18,S}
9  C u0 p0 c0 {8,D} {10,S} {19,S}
10 C u0 p0 c0 {2,S} {5,S} {9,S} {20,S}
11 H u0 p0 c0 {1,S}
12 H u0 p0 c0 {1,S}
13 H u0 p0 c0 {3,S}
14 H u0 p0 c0 {4,S}
15 H u0 p0 c0 {4,S}
16 H u0 p0 c0 {6,S}
17 H u0 p0 c0 {7,S}
18 H u0 p0 c0 {8,S}
19 H u0 p0 c0 {9,S}
20 H u0 p0 c0 {10,S}
""", xyz=r_xyz, multiplicity=1)
        p = ARCSpecies(label='P', smiles='C=C1CCC2=C1C=CC=C2', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_intra_rh_add_endocyclic(self):
        """Test the interpolate_isomerization() function for Intra_RH_Add_Endocyclic: OCCCC=C <=> C1OCCCC1"""
        r_xyz = """O      -2.82721787   -0.50660445    0.17006246
C      -1.42423100   -0.54684140   -0.06325185
C      -0.87209394    0.86948372    0.03307759
C       0.61720459    0.95856824   -0.29789433
C       1.48253511    0.23212057    0.69200350
C       2.29552591   -0.78361542    0.37581471
H      -3.14871804   -1.42308221    0.12085976
H      -0.97202705   -1.20419596    0.68543030
H      -1.25175899   -0.96687073   -1.05925336
H      -1.43076345    1.51736264   -0.65450031
H      -1.06765501    1.27028264    1.03611030
H       0.78939439    0.58580466   -1.31508343
H       0.91578668    2.01385902   -0.29802098
H       1.44262464    0.57498713    1.72449826
H       2.37872656   -1.15563129   -0.64063021
H       2.90404934   -1.26413200    1.13615558"""
        p_xyz = """C       1.49877622    0.35148592    0.11170549
O       0.76070649    1.27450052   -0.68915109
C      -0.59292278    1.40225747   -0.25390393
C      -1.32229318    0.06922018   -0.35968935
C      -0.58250938   -1.01416260    0.41788069
C       0.89132432   -1.04279750    0.02723025
H       1.52973169    0.71083537    1.14705425
H       2.52718630    0.33292459   -0.26270713
H      -1.08027028    2.14514168   -0.89325682
H      -0.61092541    1.78620120    0.77288739
H      -2.34982872    0.16634301    0.00594495
H      -1.37187159   -0.22970950   -1.41399977
H      -0.67028618   -0.81482402    1.49296781
H      -1.04113828   -1.99160041    0.23257309
H       1.44210014   -1.73854431    0.66873858
H       0.97222065   -1.40727159   -1.00427440"""
        r = ARCSpecies(label='R', smiles='OCCCC=C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C1OCCCC1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertIsNotNone(ts_xyzs)
        self.assertIsInstance(ts_xyzs, list)
        print(ts_xyzs)

    def test_interpolate_intra_rh_add_exocyclic(self):
        """Test the interpolate_isomerization() function for Intra_RH_Add_Exocyclic: CCCCC=O <=> OC1CCCC1"""
        r_xyz = """C       2.37622772    0.13188036   -0.10043648
C       1.09261098   -0.13889783    0.66802441
C       1.03542486   -1.57990839    1.17276674
C      -0.25353686   -1.84746630    1.94299618
C      -0.34013642   -3.26490651    2.45557060
O       0.51794350   -4.12410013    2.28438051
H       2.45715808   -0.52452154   -0.97274743
H       2.39602471    1.16814509   -0.45211398
H       3.25421560   -0.02861535    0.53337596
H       1.02698751    0.55499156    1.51415509
H       0.23430311    0.06179345    0.01629135
H       1.89975831   -1.78033522    1.81824290
H       1.10934752   -2.27212144    0.32467502
H      -0.32818699   -1.17971331    2.80802984
H      -1.12507868   -1.67556129    1.30220487
H      -1.25959266   -3.49936639    3.01936650"""
        p_xyz = """O       1.75068632   -0.09029920    0.90108167
C       1.01083061   -0.47823340   -0.24992932
C      -0.13989723   -1.38014711    0.16899350
C       0.50198581   -2.75218262    0.36911590
C       1.79430404   -2.74382894   -0.44803623
C       1.85007389   -1.38351632   -1.13893127
H       2.51562180    0.42592005    0.59460101
H       0.66664036    0.42343058   -0.76453768
H      -0.63895723   -1.04070286    1.08184893
H      -0.88290961   -1.43979207   -0.63498445
H       0.73240574   -2.91763342    1.42790171
H      -0.17319142   -3.55390951    0.05283667
H       2.65512384   -2.86755656    0.21945810
H       1.82352837   -3.56221500   -1.17443990
H       1.39657664   -1.45944699   -2.13415997
H       2.88277623   -1.03926992   -1.24983116"""
        r = ARCSpecies(label='R', smiles='CCCCC=O', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='OC1CCCC1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_intra_r_add_endocyclic(self):
        """Test the interpolate_isomerization() function for Intra_R_Add_Endocyclic: [CH2]C(=C)CC=C <=> C=C1C[CH]CC1"""
        r_xyz = """C      -1.27823088    1.00022129    0.80097658
C      -1.01917913   -0.22955078    0.09026125
C      -0.02594686   -0.29346779   -0.80997590
C      -1.88403832   -1.42653204    0.42498207
C      -3.27677978   -1.29535870   -0.12996190
C      -4.39315838   -1.32133148    0.61022768
H      -2.11036075    1.06181657    1.49325792
H      -0.68646022    1.88869955    0.61532565
H       0.59840554    0.56437329   -1.04036938
H       0.18912948   -1.21154808   -1.34816174
H      -1.90337050   -1.57047520    1.51308516
H      -1.44242217   -2.34157966    0.00954267
H      -3.36129222   -1.17979512   -1.20866901
H      -4.36029388   -1.43688890    1.68899607
H      -5.36907143   -1.22324048    0.14478524"""
        p_xyz = """C       2.12758194   -0.29064626    0.21857156
C       0.82704110   -0.11599145   -0.04685445
C       0.19232545    1.17528779   -0.47347761
C      -1.25056831    0.99030456   -0.15006809
C      -1.50165492   -0.34966012    0.46000320
C      -0.24270762   -1.15659552    0.14365759
H       2.50915234   -1.24701207    0.56215113
H       2.84083494    0.51851416    0.09633547
H       0.31431662    1.33095577   -1.54957658
H       0.60289815    2.02934507    0.07300159
H      -1.99716612    1.76344322   -0.25056055
H      -1.63529725   -0.24203803    1.54187312
H      -2.39462477   -0.82390929    0.04316348
H      -0.01285709   -1.86758050    0.94338245
H      -0.37927447   -1.71441733   -0.79057319"""
        r = ARCSpecies(label='R', smiles='[CH2]C(=C)CC=C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C=C1C[CH]CC1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts_1 = """C       0.08529154   -0.02047020    2.12080314
C       0.08529154   -0.02047020    0.78201213
C       0.60479283   -1.07806229    0.30802780
C       0.02862768    1.20134210   -0.08760707
C      -0.45046973    0.68544216   -1.40102540
C      -0.21329249   -0.67814168   -1.70312787
H       0.08529154    0.90762085    2.68388303
H       0.08732806   -0.94820117    2.68435418
H       1.41931794   -1.80062633    0.25752967
H      -0.11625283   -1.85368618    0.56609896
H       1.02326745    1.64240461   -0.20275442
H      -0.66448483    1.94939660    0.30816137
H      -0.68822312    1.31227568   -2.24707710
H      -0.33262802   -1.49626625   -2.41343147
H      -1.24657590   -0.88001569   -1.42086510"""
        self.assertTrue(almost_equal_coords(ts_xyzs[1], str_to_xyz(expected_ts_1)))

    def test_interpolate_intra_r_add_exotetcyclic(self):
        """Test the interpolate_isomerization() function for Intra_R_Add_ExoTetCyclic: [CH2]CCOOC <=> C1COOC1 + [CH3]"""
        """recipe(actions=[
            ['BREAK_BOND', '*2', 1, '*3'],
            ['FORM_BOND', '*1', 1, '*2'],
            ['LOSE_RADICAL', '*1', '1'],
            ['GAIN_RADICAL', '*3', '1'],
        ])"""
        r_xyz = """C       2.05985381    0.44673859    0.77745680
C       1.69169687   -0.75637966   -0.03331801
C       2.68187206   -1.04448709   -1.16032059
O       3.90557799   -1.53262290   -0.60696723
O       4.78049880   -1.77643754   -1.74910194
C       5.93016348   -0.97143884   -1.52266066
H       1.28828146    1.00280214    1.29749868
H       3.09693103    0.63419897    1.03470223
H       0.69893990   -0.58798681   -0.46472459
H       1.62812074   -1.61807258    0.64019037
H       2.26930074   -1.81581019   -1.82029089
H       2.87980796   -0.14270731   -1.75135770
H       6.41753818   -1.26803447   -0.58942003
H       5.64822776    0.08466497   -1.48132427
H       6.62752728   -1.12057626   -2.35077938"""
        p1_xyz = """C       0.28314743   -0.70497886    0.07380387
C      -1.13816845   -0.28675737   -0.14290495
O      -1.00530960    0.85101967   -0.98865614
O       0.15544463    1.58680743   -0.46124000
C       0.97001273    0.62575479    0.21836293
H       0.66032786   -1.20754429   -0.82483477
H       0.43049357   -1.37331196    0.92545264
H      -1.63459922    0.01358641    0.78559102
H      -1.73197656   -1.05648003   -0.64229664
H       1.03516272    0.92769411    1.26837942
H       1.97546489    0.62421011   -0.21165735"""
        p2_xyz = """C       0.00000000    0.00000001   -0.00000000
H       1.06690511   -0.17519582    0.05416493
H      -0.68531716   -0.83753536   -0.02808565
H      -0.38158795    1.01273118   -0.02607927"""
        r = ARCSpecies(label='R', smiles='[CH2]CCOOC', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='C1COOC1', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='[CH3]', xyz=p2_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        # Ring closure with leaving group: C0 attacks O4 (forming bond ~2.1 Å),
        # C5-O4 bond breaks (breaking bond ~2.1 Å), CH3 H atoms point outward.
        expected_ts = """C       2.05985381    0.44673859    0.77745680
C       1.69169687   -0.75637966   -0.03331801
C       2.73345911   -0.96554051   -1.13084741
O       3.94229347   -0.29400190   -0.77072507
O       3.64344642    1.09516249   -0.43858589
C       5.34170720    2.28428985   -0.10409153
H       1.51876981    1.38222451    0.63535046
H       2.69678047    0.34138829    1.65570833
H       0.70388866   -0.58882623   -0.46257409
H       1.62844124   -1.61372859    0.63679506
H       2.94682617   -2.02977118   -1.23078529
H       2.37533185   -0.58759500   -2.08844966
H       5.82738003    1.98872990    0.82589030
H       5.06076895    3.33665721   -0.06290139
H       6.03724721    2.13554246   -0.93004450"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_intra_r_add_exo_scission(self):
        """Test the interpolate_isomerization() function for Intra_R_Add_Exo_scission: C#C[CH]Cc1ccccc1 <=> C#CC([CH2])c1ccccc1"""
        r_xyz = """C       4.23346824   -0.94993099    0.35203386
                   C       3.27701964   -0.24080855    0.20374022
                   C       2.14744760    0.59958012    0.02913587
                   C       1.05580292    0.28887172   -0.94496833
                   C       0.81218753    1.42065783   -1.91405738
                   C       1.64631371    1.59084870   -3.02865931
                   C       1.43302606    2.64455795   -3.91865828
                   C       0.38802147    3.54057215   -3.70164600
                   C      -0.44293222    3.38571167   -2.59350729
                   C      -0.23130881    2.33270646   -1.70172217
                   H       5.08357894   -1.57799142    0.48427170
                   H       2.06445726    1.49172813    0.64082561
                   H       1.28121839   -0.62374363   -1.51126960
                   H       0.13557426    0.07471597   -0.38754578
                   H       2.47003331    0.90233884   -3.20563155
                   H       2.08451055    2.76729073   -4.77982138
                   H       0.22310907    4.36126939   -4.39461454
                   H      -1.25568049    4.08672428   -2.42251918
                   H      -0.88637790    2.22866854   -0.83978334"""
        p_xyz = """C       2.36461930    2.47614099   -0.28244424
                   C       1.99604231    1.33229290   -0.27285987
                   C       1.54413882   -0.07391351   -0.26910014
                   C       2.23688538   -0.81857550    0.83009369
                   C       0.03108541   -0.18363640   -0.17163035
                   C      -0.69688127   -0.82392311   -1.18801718
                   C      -2.08662674   -0.93685605   -1.10708839
                   C      -2.76791137   -0.41299498   -0.01050932
                   C      -2.06176492    0.22460524    1.00687539
                   C      -0.67253825    0.33861613    0.92823056
                   H       2.68644091    3.49178442   -0.29058723
                   H       1.87049420   -0.52966109   -1.21308108
                   H       3.24746658   -0.55865176    1.12244838
                   H       1.78706966   -1.70886735    1.25422569
                   H      -0.18547358   -1.23966395   -2.05345409
                   H      -2.63770469   -1.43352626   -1.90142994
                   H      -3.84933084   -0.50063791    0.05076577
                   H      -2.59152597    0.63498118    1.86245464
                   H      -0.13614794    0.84056704    1.73128084"""
        r = ARCSpecies(label='R', smiles='C#C[CH]Cc1ccccc1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C#CC([CH2])c1ccccc1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertIsNotNone(ts_xyzs)
        self.assertIsInstance(ts_xyzs, list)
        self.assertGreaterEqual(len(ts_xyzs), 1)
        # At least one TS guess must preserve the phenyl ring (atoms 4-9)
        # and the CH2 group (C3 with H12, H13), following the RMG recipe.
        ring_bonds = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 4)]
        has_good_guess = False
        for ts_xyz in ts_xyzs:
            coords = np.array(ts_xyz['coords'])
            ring_ok = all(1.2 < np.linalg.norm(coords[a] - coords[b]) < 1.7
                          for a, b in ring_bonds)
            ch2_ok = all(0.9 < np.linalg.norm(coords[3] - coords[h]) < 1.3
                         for h in [12, 13])
            if ring_ok and ch2_ok:
                has_good_guess = True
        self.assertTrue(has_good_guess, 'No TS guess preserves the phenyl ring and CH2 group.')
        expected_ts_0 = """C      -0.01958657    0.23765749    4.34549193
C      -0.01958657    0.23765749    3.14564232
C      -0.02291819    0.22886612    1.70122440
C      -0.01958657    1.57156099    1.04013811
C      -0.02636485    0.20416114   -0.42908818
C      -1.17605149   -0.42955527   -0.92570659
C      -1.16498692   -1.03973654   -2.18143365
C      -0.00569938   -1.02581660   -2.95462771
C       1.25169757   -0.41193265   -2.40488495
C       1.16893805    0.22542641   -1.16560710
H      -0.01958657    0.23765749    5.41068463
H       1.06707188    0.22493546    1.69873421
H      -0.31207664    2.41244609    1.66994386
H       0.68652122    1.66765510    0.21475497
H      -2.08866097   -0.45325070   -0.33392806
H      -2.06126681   -1.52763503   -2.55524488
H       0.00177103   -1.50194894   -3.93145306
H       2.15818820   -0.40291726   -3.00439205
H       2.04413067    0.73977373   -0.77527056"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_0)))

    def test_interpolate_intra_r_add_exocyclic(self):
        """Test the interpolate_isomerization() function for Intra_R_Add_Exocyclic: [CH2]C1CC=CC1=C <=> [CH2]C12C=CCC1C2"""
        r_xyz = """C      -0.10963719   -1.94278328    0.14292941
C      -0.01615129   -0.62011917   -0.55371008
C       1.43738698   -0.10430460   -0.63116148
C       1.69330445    0.06562185   -2.09623303
C       0.63003145   -0.26150822   -2.83431380
C      -0.46489166   -0.68557609   -2.00067251
C      -1.66629105   -1.05892346   -2.45632044
H      -0.17711562   -1.97633031    1.22368923
H       0.10151773   -2.86109453   -0.39408440
H      -0.63261635    0.10871952   -0.01235661
H       2.16203120   -0.79979630   -0.19330736
H       1.53472101    0.85907739   -0.11813416
H       2.63393891    0.41423020   -2.49874047
H       0.58604951   -0.21568141   -3.91331394
H      -2.45097997   -1.36443019   -1.77155996
H      -1.89603539   -1.07075910   -3.51660759"""
        p_xyz = """C       1.78867186    0.18808600    1.08254456
C       0.72594276    0.24149539    0.09992962
C       0.63194514   -0.86489224   -0.86783199
C      -0.61469912   -1.34274785   -1.02500739
C      -1.56795638   -0.57091392   -0.14698479
C      -0.70277748    0.44742414    0.57012130
C       0.02163384    1.54157581   -0.17172013
H       2.67773199    0.79478683    0.95855116
H       1.71925168   -0.48078299    1.93253868
H       1.49635915   -1.24478006   -1.39801088
H      -0.90323661   -2.14800147   -1.68296229
H      -2.33896337   -0.07556696   -0.74496608
H      -2.04619474   -1.24318736    0.57234486
H      -0.92514177    0.57122850    1.61936714
H      -0.27905859    1.76601624   -1.18928709
H       0.31648739    2.42027069    0.39137850"""
        r = ARCSpecies(label='R', smiles='[CH2]C1CC=CC1=C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[CH2]C12C=CCC1C2', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertIsNotNone(ts_xyzs)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts = """C       0.01909243   -1.40427604    1.55009942
 C       0.01909243   -0.12811694    0.81427431
 C       1.38422849    0.37784121    0.41892541
 C       1.38889314    0.66070638   -0.97324203
 C       0.04019582    0.37418352   -1.58501883
 C      -0.78556596   -0.10436055   -0.47180513
 C      -1.87584527    0.31767355    0.23897377
 H       0.01909243   -1.40427604    2.63356059
 H       0.11013837   -2.34445001    1.01861260
 H       1.18868646    0.88091053    1.95372104
 H       2.41705544    0.39627647   -0.77946122
 H      -0.68392635    1.15074231   -1.32043321
 H       0.15742156    0.06734541   -2.62914871
 H      -0.62850617   -0.73146389   -1.33653857
 H      -2.24889538    1.01651874    0.98028943
 H      -2.59444815   -0.14669139   -0.42781129"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_intra_retro_diels_alder_bicyclic(self):  # TODO: takes too long (9 min) replace with somethign smaller?
        """Test the interpolate_isomerization() function for Intra_Retro_Diels_alder_bicyclic: CCC1(C)C=CC2(CC)CCC2C1 <=> C=CCCC(=CC=C(C)CC)CC"""
        r_xyz = """C       2.62820311    1.95455895    0.66045522
C       1.77963459    1.16510642   -0.33200375
C       0.35816981    1.72923026   -0.60513220
C      -0.45729836    1.77743789    0.70462023
C       0.43251347    3.13231258   -1.18585164
C       0.38882196    3.43816959   -2.49158845
C       0.35096691    2.37435960   -3.54434274
C       1.46022619    2.56671941   -4.59996970
C       2.86189546    2.43436587   -4.02006202
C      -1.05979297    2.21688592   -4.20089259
C      -0.87396007    0.68758355   -4.13161183
C       0.16360383    0.91203107   -3.00339370
C      -0.40384902    0.80096811   -1.58584947
H       2.18248455    1.95934644    1.65897564
H       3.61805755    1.49397396    0.74771491
H       2.77358272    2.98905091    0.33630917
H       2.34381094    1.10271367   -1.26966875
H       1.69418120    0.13539759    0.03914495
H      -0.02645570    2.47169284    1.43424676
H      -0.50662948    0.78762624    1.17259490
H      -1.48665193    2.10939979    0.52211188
H       0.52219727    3.95395463   -0.47620644
H       0.43093842    4.47574097   -2.81247537
H       1.35412093    1.83190623   -5.40740366
H       1.35948250    3.55385237   -5.06914874
H       3.02531704    1.44037547   -3.59432461
H       3.04873102    3.17396935   -3.23570126
H       3.60782376    2.58679176   -4.80681493
H      -1.17605333    2.62402136   -5.21010284
H      -1.89835842    2.57696674   -3.59144783
H      -0.44517101    0.25438875   -5.04276551
H      -1.76763050    0.11524416   -3.86760729
H       1.04720012    0.27122863   -3.11093426
H      -1.46764758    1.07700362   -1.59099328
H      -0.36320195   -0.24253788   -1.24921386"""
        p_xyz = """ C                 -0.97031193    5.50216532    0.12514659
 C                  0.24916744    4.56181161    0.13925687
 C                 -0.22351539    3.10564462   -0.02729997
 C                 -0.56554744    2.26039541    1.21373297
 C                 -0.33849353    2.56803475   -1.26597845
 C                  0.00354351    3.41327828   -2.50701692
 C                 -0.11143646    2.87566216   -3.74569244
 C                  0.23060473    3.72089733   -4.98673458
 C                 -1.03242234    4.45760651   -5.47007428
 C                 -0.58414242    1.41949886   -3.91223922
 C                  0.63532085    0.47912157   -3.89810962
 C                  0.16260644   -0.97703899   -4.06466198
 C                  0.04761548   -1.51464546   -5.30334099
 H                 -1.62739924    5.24621532    0.92989830
 H                 -0.64188608    6.51391674    0.24087131
 H                 -1.48895140    5.39972863   -0.80513332
 H                  0.90625559    4.81776278   -0.66549482
 H                  0.76780433    4.66424998    1.06953741
 H                 -1.60153827    2.38074613    1.45276716
 H                  0.03123683    2.58362121    2.04094116
 H                 -0.36398709    1.22953504    1.00976664
 H                 -0.66692573    1.55628175   -1.38169956
 H                  0.33197308    4.42503082   -2.39129917
 H                  0.98582039    4.43564700   -4.73438360
 H                  0.59059924    3.08155001   -5.76554188
 H                 -1.39241813    5.09695276   -4.69126694
 H                 -1.78763606    3.74285550   -5.72242582
 H                 -0.79477177    5.04487912   -6.33235686
 H                 -1.10277322    1.31706029   -4.84252333
 H                 -1.24124298    1.16356647   -3.10749160
 H                  1.15395228    0.58155688   -2.96782540
 H                  1.29242248    0.73505433   -4.70285672
 H                 -0.07505125   -1.56431834   -3.20238382
 H                  0.28527050   -0.92736481   -6.16561749
 H                 -0.28082774   -2.52639180   -5.41906505"""
        r = ARCSpecies(label='R', smiles='CCC1(C)C=CC2(CC)CCC2C1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C=CCCC(=CC=C(C)CC)CC', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_intra_ene_reaction(self):
        """Test the interpolate_isomerization() function for Intra_ene_reaction: [CH]1C=CCC1C1C=CC=C1 <=> [CH]1C=CCC1C1=CC=CC1"""
        r_xyz = """C      -1.36262485   -1.33614811   -0.08702211
C      -2.70350854   -0.92818151   -0.37050725
C      -2.82526762    0.39244641   -0.22866599
C      -1.53162451    1.04129650    0.15534834
C      -0.55045629   -0.13880493    0.34325202
C       0.75806531    0.04144663   -0.45229836
C       1.56883117    1.23494814    0.00700816
C       2.78039338    0.83865071    0.42147165
C       2.89382351   -0.58708003    0.27767673
C       1.75271705   -1.07741548   -0.22616482
H      -1.05051909   -2.36554120    0.00006046
H      -3.49586637   -1.60940735   -0.64527347
H      -3.74292673    0.94272427   -0.38363872
H      -1.21830081    1.73449687   -0.63264320
H      -1.64384351    1.60511830    1.08761833
H      -0.32513128   -0.24664896    1.41393869
H       0.55020854    0.13069214   -1.52629302
H       1.22315660    2.25787247   -0.01144886
H       3.56816493    1.47549428    0.79467916
H       3.77674406   -1.15486684    0.52950354
H       1.56743535   -2.11665067   -0.45350367"""
        p_xyz = """C       1.55366820    1.03635804   -0.37383560
C       2.63079531    0.53077850    0.42091537
C       2.55443245   -0.79857991    0.51237427
C       1.37944943   -1.35137258   -0.23811366
C       0.68735565   -0.10328814   -0.83089129
C      -0.73645682    0.05951324   -0.40031363
C      -1.82811265    0.06502298   -1.18415319
C      -3.00360769    0.24633528   -0.37442144
C      -2.64580020    0.35570522    0.91219748
C      -1.15271513    0.25068735    1.03706722
H       1.42462012    2.07140794   -0.64873786
H       3.39167286    1.15259470    0.86987740
H       3.25529926   -1.41683639    1.05573837
H       0.73217040   -1.93146639    0.42772748
H       1.72555758   -2.01696275   -1.03709643
H       0.73432541   -0.15827843   -1.92650695
H      -1.84899398   -0.04738546   -2.25782761
H      -4.00957440    0.28602957   -0.76372185
H      -3.31941194    0.50012284    1.74306462
H      -0.73715017    1.16894496    1.46262037
H      -0.87738606   -0.60600160    1.65882609"""
        r = ARCSpecies(label='R', smiles='[CH]1C=CCC1C1C=CC=C1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[CH]1C=CCC1C1=CC=CC1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertIsNotNone(ts_xyzs)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 21)
            self.assertFalse(colliding_atoms(ts_xyz))

    def test_interpolate_ketoenol(self):
        """Test the interpolate_isomerization() function for Ketoenol: C=C(O)C(C)=O <=> CC(=O)C(C)=O"""
        """recipe(actions=[
            ['CHANGE_BOND', '*1', -1, '*2'],
            ['CHANGE_BOND', '*2', 1, '*3'],
            ['BREAK_BOND', '*3', 1, '*4'],
            ['FORM_BOND', '*4', 1, '*1'],
        ])"""
        r_xyz = """C      -1.22159321    1.04727051   -0.17415136
C      -0.91942196   -0.22292232    0.10686940
O      -1.90815211   -1.19044441    0.03582385
C       0.44706209   -0.69500604    0.50466920
C       1.58344574    0.28457331    0.60825302
O       0.58794013   -1.89820854    0.73687102
H      -0.49981586    1.85374604   -0.13887749
H      -2.23786585    1.30815995   -0.45506231
H      -1.45680525   -2.02766524    0.28003915
H       1.36171699    1.03932868    1.36703308
H       1.76618606    0.75296893   -0.36219962
H       2.49123487   -0.24743541    0.90868545"""
        p_xyz = """C      -1.93946768    0.13069384    0.34771794
C      -0.48384025   -0.00830375    0.68082393
O       0.36752942    0.81884109    0.38053113
C      -0.07907203   -1.26409906    1.43682150
C       1.37655471   -1.40309464    1.76993156
O      -0.93044148   -2.09124536    1.73711143
H      -2.25482502   -0.69806891   -0.29100638
H      -2.52729594    0.10422792    1.26969543
H      -2.11633486    1.08661628   -0.15147829
H       1.69190855   -0.57433232    2.40865816
H       1.96438577   -1.37662611    0.84795578
H       1.55342226   -2.35901745    2.26912698"""
        r = ARCSpecies(label='R', smiles='C=C(O)C(C)=O', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='CC(=O)C(C)=O', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertIsNotNone(ts_xyzs)
        self.assertIsInstance(ts_xyzs, list)

    def test_interpolate_korcek_step1(self):
        """Test the interpolate_isomerization() function for Korcek_step1: """
        """recipe(actions=[
            ['BREAK_BOND', '*1', 1, '*2'],
            ['CHANGE_BOND', '*3', -1, '*4'],
            ['FORM_BOND', '*2', 1, '*3'],
            ['FORM_BOND', '*1', 1, '*4'],
        ])"""
        pass

    def test_interpolate_korcek_step2(self):
        """Test the interpolate_isomerization() function for Korcek_step2: """
        """recipe(actions=[
            ['BREAK_BOND', '*1', 1, '*6'],
            ['BREAK_BOND', '*4', 1, '*5'],
            ['BREAK_BOND', '*2', 1, '*3'],
            ['CHANGE_BOND', '*3', 1, '*4'],
            ['CHANGE_BOND', '*1', 1, '*5'],
            ['FORM_BOND', '*2', 1, '*6'],
        ])"""
        pass

    def test_interpolate_r_addition_com(self):
        """Test the interpolate_isomerization() function for R_Addition_COm: [CH]=C + [C-]#[O+] <=> C=C[C]=O"""
        """recipe(actions=[
            ['LOSE_PAIR', '*1', '1'],
            ['CHANGE_BOND', '*1', -1, '*3'],
            ['GAIN_PAIR', '*3', '1'],
            ['GAIN_RADICAL', '*1', '1'],
            ['FORM_BOND', '*1', 1, '*2'],
            ['LOSE_RADICAL', '*2', '1'],
        ])"""
        r1_xyz = {'symbols': ('C', 'C', 'H', 'H', 'H'), 'isotopes': (12, 12, 1, 1, 1),
                  'coords': ((0.051314, 0.73513, 0.0), (0.051314, -0.598032, 0.0), (-0.705056, 1.524894, 0.0),
                             (-0.892532, -1.168418, 0.0), (0.981824, -1.17906, 0.0))}
        r2_xyz = {'symbols': ('C', 'O'), 'isotopes': (12, 16), 'coords': ((0.0, 0.0, 0.5647), (0.0, 0.0, -0.5647))}
        p_xyz = """C      -1.05582353   -0.20017926    0.05937677
C       0.21933868    0.45983151    0.08061640
C       0.34767542    1.72488610    0.34913916
O       0.46724271    2.86884055    0.59161118
H      -1.13119639   -1.25826023   -0.16835790
H      -1.95511829    0.36523106    0.27350088
H       1.10527012   -0.12500938   -0.13651192"""
        r1 = ARCSpecies(label='R1`', smiles='[CH]=C', xyz=r1_xyz)
        r2 = ARCSpecies(label='R2', smiles='[C-]#[O+]', xyz=r2_xyz)
        p = ARCSpecies(label='P', smiles='C=C[C]=O', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_r_addition_csm(self):
        """Test the interpolate_isomerization() function for R_Addition_CSm: [CH3] + [C-]#[S+] <=> C[C]=S"""
        """recipe(actions=[
            ['LOSE_PAIR', '*1', '1'],
            ['CHANGE_BOND', '*1', -1, '*3'],
            ['GAIN_RADICAL', '*1', '1'],
            ['GAIN_PAIR', '*3', '1'],
            ['FORM_BOND', '*1', 1, '*2'],
            ['LOSE_RADICAL', '*2', '1'],
        ])"""
        r1_xyz = """C       0.00000000    0.00000001   -0.00000000
H       1.06690511   -0.17519582    0.05416493
H      -0.68531716   -0.83753536   -0.02808565
H      -0.38158795    1.01273118   -0.02607927"""
        r2_xyz = """C       0.00000000    0.00000000    0.77520000
S       0.00000000    0.00000000   -0.77520000"""
        p_xyz = """C      -0.58067438    0.11747765    0.02320051
C       0.85092830   -0.17215331   -0.03399838
S       2.61321593   -0.52868580   -0.10440968
H      -1.14036868   -0.55960354   -0.62877927
H      -0.78261811    1.14344481   -0.29833385
H      -0.96048302   -0.00047957    1.04232058"""
        r1 = ARCSpecies(label='R1`', smiles='[CH3]', xyz=r1_xyz)
        r2 = ARCSpecies(label='R2', smiles='[C-]#[S+]', xyz=r2_xyz)
        p = ARCSpecies(label='P', smiles='C[C]=S', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_r_addition_multiplebond_1(self):
        """Test the interpolate_isomerization() function for R_Addition_MultipleBond: C=O + [CH2]C(C)=O <=> CC(=O)CC[O]"""
        r1_xyz = """C      -0.01220431    0.00177310   -0.00001455
O       1.19970303   -0.17429820    0.00143077
H      -0.72773088   -0.83593982   -0.00083194
H      -0.45976783    1.00846493   -0.00058428"""
        r2_xyz = """C       1.28242245   -0.40311539   -0.31476291
C       0.33726067    0.50483207   -0.02244297
C      -1.07563684    0.01997511    0.23258484
O       0.43588038    1.75468657    0.07986633
H       2.30034132   -0.07770341   -0.50085798
H       1.06793973   -1.46154871   -0.37332434
H      -1.41028390    0.28344461    1.24404472
H      -1.79479734    0.46558782   -0.46537064
H      -1.15047994   -1.06842351    0.11324141"""
        p_xyz = """C       1.62458648   -0.41336107    0.24073228
C       0.75327725    0.59078853   -0.46708745
O       1.25002311    1.38064666   -1.28107905
C      -0.72353764    0.65982686   -0.10343780
C      -1.47299076   -0.69213267   -0.18920265
O      -1.30633763   -1.32241954    0.96352558
H       1.49067492   -0.32707593    1.32171956
H       2.67446260   -0.20643293    0.01168983
H       1.39130757   -1.42351669   -0.10222412
H      -0.81704287    1.14442902    0.87742131
H      -1.18187886    1.34316111   -0.83194099
H      -2.50397095   -0.42364573   -0.47146465
H      -1.11944178   -1.17790761   -1.11253294"""
        r1 = ARCSpecies(label='R1`', smiles='C=O', xyz=r1_xyz)
        r2 = ARCSpecies(label='R2', smiles='[CH2]C(C)=O', xyz=r2_xyz)
        p = ARCSpecies(label='P', smiles='CC(=O)CC[O]', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts = """C       1.62458648   -0.41336107    0.24073228
C       0.75327725    0.59078853   -0.46708745
O       1.25002311    1.38064666   -1.28107905
C      -0.72353764    0.65982686   -0.10343780
C      -1.67235345   -1.05176863   -0.21201703
O      -1.50570032   -1.68205550    0.94071120
H       1.49067492   -0.32707593    1.32171956
H       2.67446260   -0.20643293    0.01168983
H       1.39130757   -1.42351669   -0.10222412
H      -0.81704287    1.14442902    0.87742131
H      -1.18187886    1.34316111   -0.83194099
H      -2.70333364   -0.78328169   -0.49427903
H      -1.31880447   -1.53754357   -1.13534732"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_r_addition_multiplebond_2(self):
        """Test the interpolate_isomerization() function for R_Addition_MultipleBond: C=CC1C=CC=C1 + H <=> [CH2]CC1C=CC=C1"""
        r1_xyz = """C       2.59051067   -0.39664881    0.15146030
C       1.35175484   -0.28840983   -0.34892754
C       0.14492095    0.11375786    0.45017703
C      -0.54581199    1.31613735   -0.13940028
C      -1.80598656    1.00242846   -0.47556998
C      -2.05538266   -0.37899125   -0.15698291
C      -0.94940622   -0.91939258    0.37616396
H       2.80444318   -0.19431785    1.19626487
H       3.42013723   -0.69278929   -0.48318339
H       1.19391309   -0.50338766   -1.40465443
H       0.40154153    0.30911499    1.49814794
H      -0.08603849    2.28559015   -0.25908082
H      -2.53459250    1.66645414   -0.91515688
H      -2.99477591   -0.88254253   -0.32730412
H      -0.84880463   -1.93940460    0.71529341"""
        p_xyz = """C       2.46335861    0.37083408    0.38860691
C       1.36459493   -0.45757246   -0.20195506
C       1.84046369   -1.38964936   -1.32789481
C       0.69299253   -2.13481877   -1.96819635
C       0.88038745   -3.45684499   -1.84825546
C       2.11340869   -3.70490861   -1.15017540
C       2.69219298   -2.53701753   -0.83689410
H       3.23256270    0.79999139   -0.24331878
H       2.38848642    0.70506998    1.41714938
H       0.59895937    0.22619037   -0.58872203
H       0.89378746   -1.03646232    0.60301035
H       2.38311108   -0.82456705   -2.09575634
H      -0.14580528   -1.66261327   -2.45758782
H       0.22464292   -4.23146954   -2.21598066
H       2.49797647   -4.68843588   -0.92662144
H       3.63098393   -2.42155546   -0.31586087"""
        r1 = ARCSpecies(label='R1`', smiles='C=CC1C=CC=C1', xyz=r1_xyz)
        r2 = ARCSpecies(label='R2', smiles='[H]')
        p = ARCSpecies(label='P', smiles='[CH2]CC1C=CC=C1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        expected_ts = """C       2.46335861    0.37083408    0.38860691
C       1.36459493   -0.45757246   -0.20195506
C       1.84046369   -1.38964936   -1.32789481
C       0.69299253   -2.13481877   -1.96819635
C       0.88038745   -3.45684499   -1.84825546
C       2.11340869   -3.70490861   -1.15017540
C       2.69219298   -2.53701753   -0.83689410
H       3.23256270    0.79999139   -0.24331878
H       2.38848642    0.70506998    1.41714938
H       0.59895937    0.22619037   -0.58872203
H       0.71689599   -1.25396243    0.90545142
H       2.38311108   -0.82456705   -2.09575634
H      -0.14580528   -1.66261327   -2.45758782
H       0.22464292   -4.23146954   -2.21598066
H       2.49797647   -4.68843588   -0.92662144
H       3.63098393   -2.42155546   -0.31586087"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))
        self.assertEqual(len(ts_xyzs), 2)
        self.assertFalse(almost_equal_coords(ts_xyzs[0], ts_xyzs[1]))

    def test_interpolate_r_addition_multiplebond_3(self):
        """Test the interpolate_isomerization() function for R_Addition_MultipleBond: C=C1C=CC=C1 + CH3 <=> C=C1[CH]C(C)C=C1"""
        r1_xyz = """C       1.97932444    0.19654056   -0.03116803
C       0.65136753    0.06467871   -0.01025714
C      -0.07922537   -1.18380281   -0.05693893
C      -1.38315055   -0.86560527   -0.01425481
C      -1.52520809    0.57681461    0.06005232
C      -0.30860763    1.14529226    0.06304570
H       2.44990974    1.17322164    0.00743674
H       2.63130983   -0.66867287   -0.08744931
H       0.34898775   -2.17075156   -0.11462111
H      -2.21133058   -1.55678387   -0.03134456
H      -2.47217073    1.09172719    0.10509531
H      -0.08120634    2.19734141    0.11040383"""
        p_xyz = """C       2.64386416   -0.48324268   -0.18939114
C       1.29638613   -0.00585584   -0.16424899
C       0.23978341   -0.71572921   -0.59469595
C      -1.02973354    0.07920831   -0.43182592
C      -2.03363444   -0.61589056    0.47787816
C      -0.50765651    1.35744342    0.16465721
C       0.82332097    1.27896935    0.30484326
H       3.44583867    0.14669152    0.17489732
H       2.87457725   -1.47019342   -0.57050534
H       0.27348474   -1.71263570   -1.00934645
H      -1.47875036    0.27606563   -1.41179703
H      -1.60967387   -0.77096914    1.47618155
H      -2.30879891   -1.59649263    0.07401797
H      -2.95490682   -0.03276860    0.59402070
H      -1.11762199    2.20804367    0.42911650
H       1.46483834    2.05039788    0.70324643"""
        r1 = ARCSpecies(label='R1`', smiles='C=C1C=CC=C1', xyz=r1_xyz)
        r2 = ARCSpecies(label='R2', smiles='[CH3]')
        p = ARCSpecies(label='P', smiles='[CH2]CC1C=CC=C1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        expected_ts = """C       2.64386416   -0.48324268   -0.18939114
C       1.29638613   -0.00585584   -0.16424899
C       0.23978341   -0.71572921   -0.59469595
C      -1.02973354    0.07920831   -0.43182592
C      -2.32196267   -0.81552842    0.73915233
C      -0.50765651    1.35744342    0.16465721
C       0.82332097    1.27896935    0.30484326
H       3.44583867    0.14669152    0.17489732
H       2.87457725   -1.47019342   -0.57050534
H       0.27348474   -1.71263570   -1.00934645
H      -1.47875036    0.27606563   -1.41179703
H      -1.89800210   -0.97060700    1.73745572
H      -2.59712714   -1.79613049    0.33529214
H      -3.24323505   -0.23240646    0.85529487
H      -1.11762199    2.20804367    0.42911650
H       1.46483834    2.05039788    0.70324643"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))
        self.assertEqual(len(ts_xyzs), 1)

    def test_interpolate_r_addition_multiplebond_4(self):
        """Test the interpolate_isomerization() function for R_Addition_MultipleBond (reverse template): [CH]=CCC#C <=> C#C + C#C[CH2]"""
        r_xyz = """C       1.91764289    0.19387814   -1.10551154
C       1.07811511    0.33489973   -0.11946313
C       0.02329397   -0.65855509    0.26395663
C      -1.30424787   -0.03198307    0.19778835
C      -2.39219085    0.47466407    0.14545504
H       2.60151619    0.08130988   -1.91596382
H       1.15730025    1.22603872    0.49800743
H       0.05547754   -1.53532338   -0.39317505
H       0.21953728   -1.01057943    1.28285858
H      -3.35644494    0.92564908    0.09589268"""
        p_1_xyz = """C       0.59049692   -0.03011357   -0.00178706
C      -0.59049692   -0.24451188    0.00178707
H       1.63915622    0.16026061   -0.00496073
H      -1.63915622   -0.43488608    0.00496071"""
        p_2_xyz = """C       1.41766830    0.01780142    0.32725319
C       0.24862469    0.00312196    0.05739235
C      -1.01524414   -0.01274826   -0.23435794
H       2.45552802    0.03083369    0.56683162
H      -1.58373893    0.90887589   -0.27714803
H      -1.52283795   -0.94788470   -0.43997119"""
        r = ARCSpecies(label='R1`', smiles='[CH]=CCC#C', xyz=r_xyz)
        p1 = ARCSpecies(label='R2', smiles='C#C', xyz=p_1_xyz)
        p2 = ARCSpecies(label='P', smiles='C#C[CH2]', xyz=p_2_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        expected_ts_1 = """C       2.24216061    0.49951640   -1.22347136
C       1.40263283    0.64053799   -0.23742295
C       0.02329397   -0.65855509    0.26395663
C      -1.30424787   -0.03198307    0.19778835
C      -2.39219085    0.47466407    0.14545504
H       2.92603391    0.38694814   -2.03392364
H       1.48181797    1.53167698    0.38004761
H       0.05547754   -1.53532338   -0.39317505
H       0.21953728   -1.01057943    1.28285858
H      -3.35644494    0.92564908    0.09589268"""
        self.assertTrue(almost_equal_coords(ts_xyzs[1], str_to_xyz(expected_ts_1)))

    def test_interpolate_r_addition_multiplebond_5(self):
        """Test the interpolate_isomerization() function for R_Addition_MultipleBond: C1=CCC=C1 + [CH]1C=CC=C1 <=> [CH]1CC=CC1C1C=CC=C1"""
        r1_xyz = """C       0.89763000    0.91613516   -0.13413747
C      -0.43135752    1.08157989   -0.19330002
C      -1.11632317   -0.23742810    0.01405712
C       0.05086046   -1.16250690    0.19809135
C       1.19568426   -0.47091316    0.10777775
H       1.64414493    1.68751139   -0.24729726
H      -0.94219675    2.01696507   -0.36370068
H      -1.71007697   -0.51759669   -0.86078044
H      -1.75143728   -0.21862448    0.90436901
H      -0.03089692   -2.22393004    0.37595420
H       2.19396896   -0.87119214    0.19896643"""
        r2_xyz = """C       0.08869894   -1.17089670    0.10288694
C      -1.14249756   -0.42770118    0.03685840
C      -0.79368011    0.86343174   -0.07631927
C       0.65461173    0.97238179   -0.08499512
C       1.19398211   -0.25193584    0.02286164
H       0.16976380   -2.24101816    0.19691845
H      -2.13644877   -0.84111883    0.07255411
H      -1.46541924    1.70445900   -0.15059513
H       1.19203963    1.90437077   -0.16651376
H       2.23894947   -0.51197260    0.04634374"""
        p_xyz = """C      -1.40677186    1.14730627    0.19200308
C      -2.75776970    0.66022906   -0.21886549
C      -2.61256055   -0.82663343   -0.20592156
C      -1.37808270   -1.20320774    0.15662201
C      -0.48216575   -0.02036182    0.44394353
C       0.78712746    0.01108616   -0.44238176
C       1.69513398    1.18786816   -0.14533466
C       2.90783369    0.75572153    0.22758406
C       2.93151818   -0.68129514    0.20762606
C       1.73349088   -1.14283333   -0.17743831
H      -1.15214082    2.18341254    0.35096623
H      -3.01372659    1.02013749   -1.21912422
H      -3.52131312    0.98341458    0.49408442
H      -3.41781737   -1.50542753   -0.44924059
H      -1.05776983   -2.23136784    0.25317211
H      -0.21051999   -0.03385639    1.50751741
H       0.51755306    0.02146793   -1.50682725
H       1.40309007    2.22332019   -0.23669935
H       3.75294290    1.37356520    0.49102396
H       3.79701378   -1.27783481    0.45346069
H       1.47726813   -2.18468574   -0.29876071"""
        r1 = ARCSpecies(label='R1`', smiles='C1=CCC=C1', xyz=r1_xyz)
        r2 = ARCSpecies(label='R2', smiles='[CH]1C=CC=C1', xyz=r2_xyz)
        p = ARCSpecies(label='P', smiles='[CH]1CC=CC1C1C=CC=C1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        expected_ts = """C      -1.40677186    1.14730627    0.19200308
C      -2.75776970    0.66022906   -0.21886549
C      -2.61256055   -0.82663343   -0.20592156
C      -1.37808270   -1.20320774    0.15662201
C      -0.48216575   -0.02036182    0.44394353
C       1.12449291    0.01944472   -0.67795816
C       2.03249943    1.19622672   -0.38091106
C       3.24519914    0.76408009   -0.00799234
C       3.26888363   -0.67293658   -0.02795034
C       2.07085633   -1.13447477   -0.41301471
H      -1.15214082    2.18341254    0.35096623
H      -3.01372659    1.02013749   -1.21912422
H      -3.52131312    0.98341458    0.49408442
H      -3.41781737   -1.50542753   -0.44924059
H      -1.05776983   -2.23136784    0.25317211
H      -0.21051999   -0.03385639    1.50751741
H       0.85491851    0.02982649   -1.74240365
H       1.74045552    2.23167875   -0.47227575
H       4.09030835    1.38192376    0.25544756
H       4.13437923   -1.26947625    0.21788429
H       1.81463358   -2.17632718   -0.53433711"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))
        self.assertEqual(len(ts_xyzs), 1)

    def test_interpolate_r_addition_multiplebond_6(self):
        """Test the interpolate_isomerization() function for R_Addition_MultipleBond: NC=C + H <=> [CH2]CN.
        Testing for both H positions being reactive."""
        r1_xyz = """N       1.18312392    0.25943326    0.12618490
C       0.00733662   -0.40779689    0.18732813
C      -1.17816993    0.05162131   -0.21348055
H       1.21957196    1.19886115   -0.25099292
H       2.03658318   -0.17492328    0.45705725
H       0.08977401   -1.40775289    0.61060413
H      -1.29017956    1.04262246   -0.63841564
H      -2.06804061   -0.56206414   -0.12280232"""
        p_xyz = """C       1.24736746   -0.08660317   -0.11677450
C      -0.11686766    0.45302910    0.06609871
N      -1.08087845   -0.60235697   -0.17399432
H       1.44691564   -0.82172187   -0.88860422
H       2.08994022    0.41635224    0.34202047
H      -0.23400754    0.84446135    1.08133453
H      -0.28245124    1.27419377   -0.63766281
H      -0.97739395   -1.31600031    0.54594411
H      -2.02043853   -0.22627433   -0.06082781"""
        r1 = ARCSpecies(label='R1`', smiles='NC=C', xyz=r1_xyz)
        r2 = ARCSpecies(label='R2', smiles='[H]')
        p = ARCSpecies(label='P', smiles='[CH2]CN', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        expected_ts_0 = """C       1.24736746   -0.08660317   -0.11677450
C      -0.11686766    0.45302910    0.06609871
N      -1.08087845   -0.60235697   -0.17399432
H       1.44691564   -0.82172187   -0.88860422
H       2.08994022    0.41635224    0.34202047
H      -0.27849608    0.99312334    1.46691078
H      -0.28245124    1.27419377   -0.63766281
H      -0.97739395   -1.31600031    0.54594411
H      -2.02043853   -0.22627433   -0.06082781"""
        expected_ts_1 = """C       1.24736746   -0.08660317   -0.11677450
C      -0.11686766    0.45302910    0.06609871
N      -1.08087845   -0.60235697   -0.17399432
H       1.44691564   -0.82172187   -0.88860422
H       2.08994022    0.41635224    0.34202047
H      -0.23400754    0.84446135    1.08133453
H      -0.34539881    1.58636432   -0.90520186
H      -0.97739395   -1.31600031    0.54594411
H      -2.02043853   -0.22627433   -0.06082781"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts_0)))
        self.assertTrue(almost_equal_coords(ts_xyzs[1], str_to_xyz(expected_ts_1)))
        self.assertEqual(len(ts_xyzs), 2)

    def test_interpolate_r_addition_multiplebond_7(self):
        """Test the interpolate_isomerization() function for R_Addition_MultipleBond: C2H4 + NH2 <=> [CH2]CN"""
        r1 = ARCSpecies(label='R1`', smiles='C=C')
        r2 = ARCSpecies(label='R2', smiles='[NH2]')
        p = ARCSpecies(label='P', smiles='[CH2]CN')
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        expected_ts = """C       1.20550548    0.31498584    0.08862396
C      -0.03441480   -0.46440519   -0.11368905
N      -1.50567237    0.71999145   -0.18228778
H       2.08420244   -0.16981367    0.49635832
H       1.32639815    1.28427511   -0.38233620
H       0.03499073   -1.02483988   -1.05076186
H      -0.16672320   -1.18028912    0.70340914
H      -2.35751731    0.18327787   -0.33498548
H      -1.61568244    1.16511681    0.72769492"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))
        self.assertEqual(len(ts_xyzs), 1)

    def test_interpolate_r_addition_multiplebond_8(self):
        """Test the interpolate_isomerization() function for R_Addition_MultipleBond: C#CC=CC=C + [c]1ccccc1 <=> C#C[CH]C(C=C)c1ccccc1"""
        r1_xyz = """C       3.35340265   -0.06408675   -0.52400761
C       2.16382044   -0.14303187   -0.66383512
C       0.75635763   -0.23832579   -0.83485030
C      -0.08964658    0.07503577    0.15639762
C      -1.52525803   -0.00876273    0.02151946
C      -2.36893208    0.30431513    1.01174699
H       4.40879393    0.00629855   -0.39893346
H       0.39124677   -0.57510055   -1.80032219
H       0.30410840    0.41007881    1.11451351
H      -1.93028115   -0.34245758   -0.93173400
H      -2.02162547    0.64256690    1.98297350
H      -3.44198650    0.22663779    0.86653167"""
        r2_xyz = """C       0.15052732   -1.61029831    0.02711992
C      -1.11496876   -1.02908263    0.06191547
C      -1.24634615    0.35938880    0.03858985
C      -0.10892476    1.16524604   -0.01962456
C       1.15829079    0.58260223   -0.05445444
C       1.28652351   -0.80616111   -0.03100712
H      -1.99486553   -1.66371628    0.10717730
H      -2.23369115    0.81271796    0.06574029
H      -0.21007958    2.24737138   -0.03784924
H       2.04462498    1.20985799   -0.09980359
H       2.26890933   -1.26792606   -0.05780387"""
        p_xyz = """C      -3.92872757   -1.47192834    0.63358250
C      -2.94092110   -0.79153032    0.60258970
C      -1.77265412    0.01495047    0.56847031
C      -0.78622480   -0.02895120   -0.56871809
C      -0.77398201    1.22361456   -1.43102540
C      -1.51506692    2.33393107   -1.29713471
C       0.60875694   -0.37520044   -0.05377320
C       1.10304979   -1.68245894   -0.19525834
C       2.36952644   -2.02177601    0.28509994
C       3.15655317   -1.06256660    0.91856841
C       2.67763405    0.23599273    1.07625336
C       1.41145870    0.57775692    0.59730651
H      -4.80587828   -2.07557590    0.66228852
H      -1.58412268    0.68341112    1.40170142
H      -1.09297182   -0.83522501   -1.25118213
H      -0.07170966    1.18775849   -2.26433101
H      -2.24816162    2.45558677   -0.50659652
H      -1.40408512    3.15690637   -1.99714322
H       0.50056932   -2.44848181   -0.67901116
H       2.74006069   -3.03674635    0.16744581
H       4.14157322   -1.32812429    1.29292412
H       3.28886460    0.98389599    1.57446813
H       1.05323163    1.59589194    0.73495846"""
        r1 = ARCSpecies(label='R1`', smiles='C#CC=CC=C', xyz=r1_xyz)
        r2 = ARCSpecies(label='R2', smiles='[c]1ccccc1', xyz=r2_xyz)
        p = ARCSpecies(label='P', smiles='C#C[CH]C(C=C)c1ccccc1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts = """C       1.62458648   -0.41336107    0.24073228
C       0.75327725    0.59078853   -0.46708745
O       1.25002311    1.38064666   -1.28107905
C      -0.72353764    0.65982686   -0.10343780
C      -1.67235345   -1.05176863   -0.21201703
O      -1.50570032   -1.68205550    0.94071120
H       1.49067492   -0.32707593    1.32171956
H       2.67446260   -0.20643293    0.01168983
H       1.39130757   -1.42351669   -0.10222412
H      -0.81704287    1.14442902    0.87742131
H      -1.18187886    1.34316111   -0.83194099
H      -2.70333364   -0.78328169   -0.49427903
H      -1.31880447   -1.53754357   -1.13534732"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_r_addition_multiplebond_9(self):
        """Test the interpolate_isomerization() function for R_Addition_MultipleBond: HO2 + CC=C(C)C <=> C[CH]C(C)(C)OO"""
        r1_xyz = """O      -0.15554002    0.45236503    0.00000000
O       0.99423083   -0.18785602    0.00000000
H      -0.83869082   -0.26450900    0.00000000"""
        r2_xyz = """C       2.07514153   -0.40059594    0.21069287
C       0.90426289    0.51228353    0.03034898
C      -0.40166306    0.19140582   -0.05097236
C      -1.44243076    1.26630411   -0.23485102
C      -0.96617039   -1.20025973    0.02599479
H       2.76192462   -0.28697098   -0.63406909
H       1.80714593   -1.45573073    0.28253094
H       2.61376987   -0.12962449    1.12428940
H       1.18191421    1.56406924   -0.04037437
H      -2.00123706    1.09810543   -1.16135002
H      -2.16444305    1.29263857    0.58811577
H      -0.96528394    2.25033442   -0.28934962
H      -1.65993501   -1.27730188    0.86993115
H      -1.51510802   -1.43352415   -0.89255219
H      -0.20828697   -1.97466986    0.15691347"""
        p_xyz = """C       2.03050444   -0.52296856    0.22495060
C       0.96835651    0.51502679    0.10240257
C      -0.34644185    0.21971897   -0.57612749
C      -1.13751395   -0.81230262    0.23700508
C      -1.19792658    1.48882669   -0.72540710
O      -0.11313877   -0.37619731   -1.86689838
O       0.54613365    0.61816770   -2.71208065
H       2.11784948   -1.12302501   -0.68463262
H       2.99746166   -0.04298943    0.40337489
H       1.81899629   -1.18710088    1.06772185
H       1.02862485    1.37751240    0.75896840
H      -1.33897639   -0.46286073    1.25560773
H      -2.09622102   -1.03405701   -0.24687530
H      -0.60224117   -1.76726090    0.29871577
H      -0.65478594    2.28216013   -1.25178948
H      -2.09968951    1.28424008   -1.31394734
H      -1.50241108    1.88730639    0.24890346
H       1.44579836    0.25735273   -2.62073094"""
        r1 = ARCSpecies(label='R1`', smiles='O[O]', xyz=r1_xyz)
        r2 = ARCSpecies(label='R2', smiles='CC=C(C)C', xyz=r2_xyz)
        p = ARCSpecies(label='P', smiles='C[CH]C(C)(C)OO', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts = """C       2.03050444   -0.52296856    0.22495060
C       0.96835651    0.51502679    0.10240257
C      -0.34644185    0.21971897   -0.57612749
C      -1.13751395   -0.81230262    0.23700508
C      -1.19792658    1.48882669   -0.72540710
O      -0.04685924   -0.54549234   -2.23359602
O       0.61241318    0.44887267   -3.07877829
H       2.11784948   -1.12302501   -0.68463262
H       2.99746166   -0.04298943    0.40337489
H       1.81899629   -1.18710088    1.06772185
H       1.02862485    1.37751240    0.75896840
H      -1.33897639   -0.46286073    1.25560773
H      -2.09622102   -1.03405701   -0.24687530
H      -0.60224117   -1.76726090    0.29871577
H      -0.65478594    2.28216013   -1.25178948
H      -2.09968951    1.28424008   -1.31394734
H      -1.50241108    1.88730639    0.24890346
H       1.51207789    0.08805770   -2.98742858"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_r_addition_multiplebond_10(self):
        """Test the interpolate_isomerization() function for R_Addition_MultipleBond: [CH2]C(=O)OO <=> C=C=O + HO2"""
        r = ARCSpecies(label='R`', smiles='[CH2]C(=O)OO')
        p1 = ARCSpecies(label='P1', smiles='C=C=O')
        p2 = ARCSpecies(label='P2', smiles='[O]O')
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        expected_ts = """C      -1.33600555    0.33912322   -0.58799276
C      -0.04195759   -0.22099605   -0.23170726
O       0.33136575   -1.34095465   -0.54011761
O       0.95772458    1.02947560    0.69536135
O       2.17526558    0.35903872    1.18503599
H      -2.01983324   -0.28052198   -1.14877383
H      -1.59828386    1.33952142   -0.27911177
H       2.32418256   -0.23342468    0.41960556"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_retroene(self):
        """Test the interpolate_isomerization() function for Retroene: CC(=O)OCC(C)C <=> C=C(C)C + CC(=O)O"""
        """recipe(actions=[
            ['CHANGE_BOND', '*1', -1, '*2'],
            ['BREAK_BOND', '*5', 1, '*6'],
            ['BREAK_BOND', '*3', 1, '*4'],
            ['FORM_BOND', '*1', 1, '*6'],
            ['CHANGE_BOND', '*2', 1, '*3'],
            ['CHANGE_BOND', '*4', 1, '*5'],
        ])"""
        r_xyz = """C       3.35667786   -0.45750645    0.53734155
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
H       3.11088096    4.66875130   -0.79438064"""
        p1_xyz = """C      -0.78808508   -1.31685309    0.20647447
C      -0.10495450   -0.17537411    0.02749750
C      -0.75338720    1.17147536    0.16820738
C       1.35432280   -0.16734028   -0.32564966
H      -1.84354122   -1.31372629    0.46178918
H      -0.31134041   -2.28697931    0.10277938
H      -0.26058038    1.74962274    0.95655854
H      -1.81081982    1.05845304    0.42834727
H      -0.69761470    1.75386040   -0.75753528
H       1.51026426    0.33854032   -1.28396988
H       1.73082508   -1.19212091   -0.40738892
H       1.95720910    0.35086319    0.42752746"""
        p2_xyz = """C      -0.96060243   -0.07364779   -0.00090814
C       0.48592618    0.29350855   -0.03718140
O       1.29644959    0.14042367    0.86002941
O       0.84722479    0.84115942   -1.21216716
H      -1.56574114    0.82659924   -0.14053059
H      -1.18541862   -0.77413428   -0.80892649
H      -1.20711897   -0.51300925    0.96878687
H       1.80251143    1.03132880   -1.10238169"""
        r = ARCSpecies(label='R', smiles='CC(=O)OCC(C)C', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='C=C(C)C', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='CC(=O)O', xyz=p2_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_singlet_carbene_intra_disproportionation(self):
        """Test the interpolate_isomerization() function for Singlet_Carbene_Intra_Disproportionation: C=C1C=C[C]C1 <=> C=C1C=CC=C1"""
        r_xyz = """C      -1.75380171    0.48873088   -0.19068706
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
H      -0.36583394   -1.89034834    0.81324667"""
        p_xyz = """C       1.98835311   -0.06142285   -0.00200142
C       0.65433874   -0.02021339   -0.00065871
C      -0.15871220    1.17598961   -0.05259916
C      -1.43846254    0.76969575   -0.03122551
C      -1.48316252   -0.67944287    0.03416677
C      -0.23088987   -1.16395429    0.05299117
H       2.52365896   -1.00407776    0.03918308
H       2.58073849    0.84639620   -0.04432075
H       0.20218412    2.19006475   -0.09915008
H      -2.31139440    1.40309362   -0.05766688
H      -2.39347065   -1.25775412    0.06240301
H       0.06681877   -2.19837465    0.09887848"""
        r = ARCSpecies(label='R', smiles='C=C1C=C[C]C1', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C=C1C=CC=C1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertIsNotNone(ts_xyzs)
        self.assertIsInstance(ts_xyzs, list)

    def test_interpolate_xy_addition_multiplebond(self):
        """Test the interpolate_isomerization() function for XY_Addition_MultipleBond: CC(F)F <=> C=CF + F"""
        """recipe(actions=[
            ['BREAK_BOND', '*3', 1, '*4'],
            ['CHANGE_BOND', '*1', -1, '*2'],
            ['FORM_BOND', '*1', 1, '*3'],
            ['FORM_BOND', '*2', 1, '*4'],
        ])"""
        r_xyz = """C       0.77428372   -0.04996907   -0.02995927
C      -0.73260318   -0.00572049   -0.04005672
F      -1.20770160   -0.46744624    1.14245662
F      -1.20409588   -0.83436495   -1.00353479
H       1.13143005   -1.07211215    0.12426434
H       1.18016366    0.33816078   -0.96843228
H       1.16543618    0.56767746    0.78464237
H      -1.12089093    1.00131310   -0.21289037"""
        p1_xyz = """C      -0.70626780   -0.01102906    0.03896246
C       0.61830293    0.09993211   -0.01691349
F       1.40013458   -0.96548678   -0.26531503
H      -1.33173696    0.85084051    0.23950006
H      -1.18316264   -0.97236019   -0.11642808
H       1.20272990    0.99810341    0.12019409"""
        p2_xyz = """F       0.00000000    0.00000000    0.46000000
H       0.00000000    0.00000000   -0.46000000"""
        r = ARCSpecies(label='R', smiles='CC(F)F', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='C=CF', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='F', xyz=p2_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_xy_elimination_hydroxyl(self):
        """Test the interpolate_isomerization() function for XY_elimination_hydroxyl: CCC(=O)O <=> C=C + [H][H] + O=C=O"""
        """recipe(actions=[
            ['BREAK_BOND', '*1', 1, '*2'],
            ['BREAK_BOND', '*3', 1, '*4'],
            ['BREAK_BOND', '*5', 1, '*6'],
            ['CHANGE_BOND', '*2', 1, '*3'],
            ['CHANGE_BOND', '*4', 1, '*5'],
            ['FORM_BOND', '*1', 1, '*6'],
        ])"""
        r_xyz = """C      -1.44342440    0.21938567    0.14134495
C      -0.17943385   -0.58558878   -0.10310381
C      -0.01901784   -1.69295804    0.90160826
O      -0.76331949   -1.97415266    1.82455783
O       1.10272691   -2.40793854    0.68425738
H      -1.43203982    0.67684331    1.13627537
H      -1.53708941    1.01747004   -0.60148550
H      -2.33303198   -0.41590501    0.07585794
H      -0.21702502   -1.02774537   -1.10407189
H       0.69165336    0.07422934   -0.03456899
H       1.09172878   -3.08582798    1.39221989"""
        p1_xyz = """C      -0.63422754   -0.20894058   -0.01346068
C       0.63422754    0.20894058    0.01346068
H      -1.30426171   -0.01843680    0.81903872
H      -1.02752125   -0.74974821   -0.86852786
H       1.02752125    0.74974821    0.86852786
H       1.30426171    0.01843680   -0.81903872"""
        p2_xyz = {'symbols': ('H', 'H'), 'isotopes': (1, 1), 'coords': ((0.0, 0.0, 0.371517), (0.0, 0.0, -0.371517))}
        p3_xyz = """O      -1.37316735    0.24657196    0.00000000
C      -0.00000000   -0.05081069    0.00000000
O       1.37316735   -0.34819332    0.00000000"""
        r = ARCSpecies(label='R', smiles='CCC(=O)O', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='C=C', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='[H][H]', xyz=p2_xyz)
        p3 = ARCSpecies(label='P3', smiles='O=C=O', xyz=p3_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2, p3])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_halocarbene_recombination(self):
        """Test the interpolate_isomerization() function for halocarbene_recombination: F[C](F)C(F)(F)Cl <=> F[C](F)Cl + F[C]F"""
        """recipe(actions=[
            ['LOSE_RADICAL', '*1', '1'],
            ['LOSE_PAIR', '*2', '1'],
            ['GAIN_RADICAL', '*2', '1'],
            ['FORM_BOND', '*1', 1, '*2'],
        ])"""
        r_xyz = """F       1.48247864    1.01499696    0.77064186
C       0.88290671    0.06028673    0.02371663
F       1.64746993   -0.85881410   -0.60838520
C      -0.59971907    0.01196808   -0.08801457
F      -0.98882913   -0.48874727   -1.29265530
F      -1.14261278    1.25777426   -0.00729830
Cl     -1.28169420   -0.99746584    1.20199401"""
        p1_xyz = """F      -0.84202384   -1.11313967   -0.06075322
C      -0.09041741    0.00606935   -0.01199050
F      -0.67517806    1.21498317   -0.14044750
Cl      1.60761930   -0.10791285    0.21319122"""
        p2_xyz = """F      -1.30420375    0.32043305    0.00000000
C      -0.00000000   -0.00000001    0.00000000
F       1.30420375   -0.32043304    0.00000000"""
        r = ARCSpecies(label='R', smiles='F[C](F)C(F)(F)Cl', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='F[C](F)Cl', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', adjlist="""multiplicity 1
1 F u0 p3 c0 {2,S}
2 C u0 p1 c0 {1,S} {3,S}
3 F u0 p3 c0 {2,S}""", xyz=p2_xyz, multiplicity=1)
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_halocarbene_recombination_double(self):
        """Test the interpolate_isomerization() function for halocarbene_recombination_double: [CH]F + [CH]F <=> FC=CF"""
        """recipe(actions=[
            ['LOSE_PAIR', '*1', '1'],
            ['LOSE_PAIR', '*2', '1'],
            ['FORM_BOND', '*1', 1, '*2'],
            ['CHANGE_BOND', '*1', 1, '*2'],
        ])"""
        r_xyz = """C      -0.09263942   -0.00211919    0.00000000
F       1.25000028    0.02859493    0.00000000
H      -1.15736087   -0.02647574    0.00000000"""
        p_xyz = """F       1.42204379   -0.40820397   -0.99855401
C       0.64154676    0.15618245   -0.05993182
C      -0.64154676   -0.15618245    0.05993180
F      -1.42204376    0.40820397    0.99855402
H       1.18833720    0.86211206    0.54632476
H      -1.18833722   -0.86211206   -0.54632476"""
        r1 = ARCSpecies(label='R1`', adjlist="""multiplicity 1
1 C u0 p1 c0 {2,S} {3,S}
2 F u0 p3 c0 {1,S}
3 H u0 p0 c0 {1,S}
""", xyz=r_xyz, multiplicity=1)
        r2 = ARCSpecies(label='R2', adjlist="""multiplicity 1
1 C u0 p1 c0 {2,S} {3,S}
2 F u0 p3 c0 {1,S}
3 H u0 p0 c0 {1,S}
""", xyz=r_xyz, multiplicity=1)
        p = ARCSpecies(label='P', smiles='FC=CF', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r1, r2], p_species=[p])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_intra_h_migration_cccoo(self):
        """Test the interpolate_isomerization() function for intra H migration: CCCOO (6-membered ring)."""
        r_xyz = """C                 -1.31455963    0.65305704    0.00229593
                   C                  0.17407454    0.87684185    0.32708610
                   O                  0.97540012    0.03343074   -0.50443961
                   O                  2.25137227    0.22524629   -0.22604804
                   H                 -1.56888362   -0.37060266    0.18212958
                   H                 -1.49495314    0.89014604   -1.02539419
                   H                  0.35446804    0.63975284    1.35477623
                   H                  0.42839853    1.90050154    0.14725245
                   C                 -2.17752564    1.56134592    0.89778516
                   H                 -3.21183640    1.40585907    0.67211926
                   H                 -1.99713214    1.32425692    1.92547529
                   H                 -1.92320166    2.58500562    0.71795151"""
        p_xyz = """C                  0.10191448    0.80917231    0.12324900
                   C                  1.63680299    0.68488584    0.13968460
                   O                  2.03194937   -0.20270773    1.18894005
                   O                  3.34756810   -0.30923899    1.20302771
                   H                 -0.33221037   -0.15465524   -0.04249800
                   H                  1.97345768    0.29884684   -0.79975007
                   H                  2.07092784    1.64871339    0.30543160
                   H                  3.73706329    0.55550348    1.35173530
                   H                 -0.23474021    1.19521131    1.06268367
                   C                 -0.32362778    1.76504231   -1.00671841
                   H                 -1.26726146    1.63176387   -1.49322877
                   H                  0.32433693    2.56246418   -1.30531527"""
        r = ARCSpecies(label='R', smiles='CCCO[O]', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[CH2]CCOO', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn, weight=0.5)
        self.assertIsNotNone(ts_xyzs)
        self.assertGreaterEqual(len(ts_xyzs), 1)
        self.assertLessEqual(len(ts_xyzs), 6)
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 12)
            self.assertFalse(colliding_atoms(ts_xyz), msg=f'Collision detected in TS guess:\n{xyz_to_str(ts_xyz)}')
            assert_h_migration_quality(self, ts_xyz)
        assert_unique_guesses(self, ts_xyzs)
        self.assertGreaterEqual(len(ts_xyzs), 3)
        for i in range(len(ts_xyzs)):
            for j in range(i + 1, len(ts_xyzs)):
                self.assertFalse(
                    almost_equal_coords(ts_xyzs[i], ts_xyzs[j]),
                    msg=f'Dedup failed: guesses {i} and {j} are near-identical.',
                )
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_intra_h_migration_cco(self):
        """Test the interpolate_isomerization() function for intra H migration: CCO (4-membered ring)."""
        r_xyz = """ C                 -3.35807020    0.39772754   -0.02139706
 H                 -2.80953191    0.44242278   -0.93900704
 H                 -4.34767471   -0.00900040   -0.00893508
 C                 -2.72326461    0.91878394    1.28133933
 H                 -1.66157493    0.79378755    1.23561273
 H                 -2.95540282    1.95641525    1.40106030
 O                 -3.24245519    0.18293235    2.39213346
 H                 -2.84673223    0.50774673    3.20422887"""
        p_xyz = """ C                 -0.34334771   -0.13590857    0.00000002
 H                  0.01333400    0.36848377   -0.87365124
 H                 -1.41334771   -0.13588640   -0.00000560
 C                  0.16999407    0.59004821    1.25740487
 H                  1.23999407    0.59002603    1.25741049
 H                 -0.18665169    1.59886128    1.25739942
 O                 -0.30669270   -0.08404623    2.42499487
 H                  0.01329805   -1.14472164    0.00000547"""
        r = ARCSpecies(label='R', smiles='[CH2]CO', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='CC[O]', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn, weight=0.5)
        self.assertIsNotNone(ts_xyzs)
        self.assertGreaterEqual(len(ts_xyzs), 1)
        self.assertLessEqual(len(ts_xyzs), 6)
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 8)
            self.assertFalse(colliding_atoms(ts_xyz),
                             msg=f'Collision detected in TS guess:\n{xyz_to_str(ts_xyz)}')
            assert_h_migration_quality(self, ts_xyz)
        assert_unique_guesses(self, ts_xyzs)
        self.assertGreaterEqual(len(ts_xyzs), 1)
        expected_ts = """C      -0.41865872    1.21447395   -0.16611224
H      -0.67018571    1.94949404    0.56967272
H      -0.67018574    1.32368640   -1.20037860
C      -0.41865872   -0.23745201    0.34722104
H      -0.41865872   -0.23745201    1.41722109
H      -1.29230999   -0.74185488   -0.00944557
O       0.74893138   -0.91156055   -0.12944556
H       0.88979859    0.45040550   -0.69723648"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_intra_h_migration_ccoo(self):
        """Test the interpolate_isomerization() function for intra H migration: CCOO (5-membered ring)."""
        r_xyz = """C      -1.05582103   -0.03329574   -0.10080257
                   C       0.41792695    0.17831205    0.21035514
                   O       1.19234020   -0.65389683   -0.61111443
                   O       2.44749684   -0.41401220   -0.28381363
                   H      -1.33614002   -1.09151783    0.08714882
                   H      -1.25953618    0.21489046   -1.16411897
                   H      -1.67410396    0.62341419    0.54699514
                   H       0.59566350   -0.06437686    1.28256640
                   H       0.67254676    1.24676329    0.02676370"""
        p_xyz = """C      -1.40886397    0.22567351   -0.37379668
                   C       0.06280787    0.04097694   -0.38515682
                   O       0.44130326   -0.57668419    0.84260864
                   O       1.89519755   -0.66754203    0.80966180
                   H      -1.87218376    0.90693511   -1.07582340
                   H      -2.03646287   -0.44342165    0.20255768
                   H       0.35571681   -0.60165457   -1.22096147
                   H       0.56095122    1.01161503   -0.47393734
                   H       2.05354047   -0.10415729    1.58865243"""
        r = ARCSpecies(label='R', smiles='CCO[O]', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[CH2]COO', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn, weight=0.5)
        self.assertIsNotNone(ts_xyzs)
        self.assertGreaterEqual(len(ts_xyzs), 1)
        self.assertLessEqual(len(ts_xyzs), 6)
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 9)
            self.assertFalse(colliding_atoms(ts_xyz),
                             msg=f'Collision detected in TS guess:\n{xyz_to_str(ts_xyz)}')
            assert_h_migration_quality(self, ts_xyz)
            # The migrating H (donor=C[0], acceptor=O[3]) must be closest to
            # a reactive heavy atom, not to another H or backbone atom.
            coords_arr = np.array(ts_xyz['coords'], dtype=float)
            for h_idx, sym in enumerate(ts_xyz['symbols']):
                if sym != 'H':
                    continue
                d_donor = float(np.linalg.norm(coords_arr[h_idx] - coords_arr[0]))   # C[0]
                d_acceptor = float(np.linalg.norm(coords_arr[h_idx] - coords_arr[3]))  # O[3]
                if d_acceptor > 2.0:
                    continue  # not the migrating H
                d_reactive_min = min(d_donor, d_acceptor)
                # Nearest atom overall (excluding self).
                dists = [(j, float(np.linalg.norm(coords_arr[h_idx] - coords_arr[j])))
                         for j in range(len(ts_xyz['symbols'])) if j != h_idx]
                nearest_idx, nearest_dist = min(dists, key=lambda x: x[1])
                self.assertLessEqual(
                    d_reactive_min, nearest_dist + 0.01,
                    msg=f'Migrating H[{h_idx}] is closer to atom [{nearest_idx}] '
                        f'({ts_xyz["symbols"][nearest_idx]}, d={nearest_dist:.3f}) than to '
                        f'reactive heavy atoms (C[0]={d_donor:.3f}, O[3]={d_acceptor:.3f})\n'
                        f'{xyz_to_str(ts_xyz)}')
                # Also verify H is farther from backbone O[2] than from acceptor O[3].
                d_backbone_O = float(np.linalg.norm(coords_arr[h_idx] - coords_arr[2]))
                self.assertGreater(
                    d_backbone_O, d_acceptor - 0.05,
                    msg=f'Migrating H[{h_idx}] too close to backbone O[2]: '
                        f'd(H,O[2])={d_backbone_O:.3f} vs d(H,O[3])={d_acceptor:.3f}\n'
                        f'{xyz_to_str(ts_xyz)}')
        assert_unique_guesses(self, ts_xyzs)
        self.assertGreaterEqual(len(ts_xyzs), 1)
        expected_ts = """C      -0.32167303   -0.81819596    1.17869736
C      -0.24530715    0.60203101    0.69541490
O      -0.24530715    0.60203101   -0.70711839
O       0.70246600   -0.33908784   -1.08511413
H       0.84886835   -0.73646147    0.22830147
H       0.49326099   -0.30427880    1.68847348
H      -1.09966706   -0.64511573    0.43514608
H      -0.24530715    1.64665457    1.08127342
H      -1.05171137    1.26607222    1.08127328"""
        self.assertTrue(almost_equal_coords(ts_xyzs[0], str_to_xyz(expected_ts)))

    def test_interpolate_intra_no2_ono_conversion(self):
        """Test the interpolate_isomerization() function for intra_NO2_ONO_conversion: [O-][N+](=O)CC <=> CCON=O"""
        r_xyz = """O       1.77136558   -0.91790626    0.88650594
N       1.34754589   -0.18857388   -0.01862669
O       1.86645005   -0.03906737   -1.13182045
C       0.08946605    0.57559465    0.25484606
C       0.46072863    1.91146690    0.86342166
H      -0.52075344   -0.02737899    0.93392769
H      -0.43797095    0.69242674   -0.69660400
H       1.09014915    2.48001164    0.17179384
H      -0.42932512    2.51112436    1.08295532
H       1.01533324    1.78326517    1.79934783"""
        p_xyz = """C      -1.36894499    0.07118059   -0.24801399
C      -0.01369535    0.17184136    0.42591278
O      -0.03967083   -0.62462610    1.60609048
N       1.23538512   -0.53558048    2.24863846
O       1.25629155   -1.21389295    3.27993827
H      -2.16063255    0.41812452    0.42429392
H      -1.39509985    0.66980796   -1.16284741
H      -1.59800183   -0.96960842   -0.49986392
H       0.19191326    1.21800574    0.68271847
H       0.76371340   -0.19234475   -0.25650067"""
        r = ARCSpecies(label='R', smiles='[O-][N+](=O)CC', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='CCON=O', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertIsNotNone(ts_xyzs)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 10)
            self.assertFalse(colliding_atoms(ts_xyz))

    def test_interpolate_intra_oh_migration(self):
        """Test the interpolate_isomerization() function for Intra_OH_migration reactions."""
        r_xyz = """C      -1.40886397    0.22567351   -0.37379668
C       0.06280787    0.04097694   -0.38515682
O       0.44130326   -0.57668419    0.84260864
O       1.89519755   -0.66754203    0.80966180
H      -1.87218376    0.90693511   -1.07582340
H      -2.03646287   -0.44342165    0.20255768
H       0.35571681   -0.60165457   -1.22096147
H       0.56095122    1.01161503   -0.47393734
H       2.05354047   -0.10415729    1.58865243"""
        p_xyz = """O       0.97298522    1.16961708    0.68631092
C       0.83017736    0.23002128   -0.24518707
C      -0.46505265   -0.55857538    0.09146589
O      -1.54540067    0.36524471    0.24441655
H       1.61381747   -0.53531530   -0.35348282
H       0.69744639    0.56361493   -1.28695526
H      -0.71560487   -1.25802813   -0.71249310
H      -0.36288272   -1.12613201    1.02419042
H      -1.03086141    1.13813060    0.58426610"""
        r = ARCSpecies(label='R', smiles='[CH2]COO', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[O]CCO', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertIsNotNone(ts_xyzs)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 9)
            self.assertFalse(colliding_atoms(ts_xyz))

    def test_interpolate_intra_halogen_migration(self):
        """Test the interpolate_isomerization() function for intra_halogen_migration: FCCC[C](F)F <=> [CH2]CCC(F)(F)F"""
        """recipe(actions=[
            ['BREAK_BOND', '*2', 1, '*3'],
            ['FORM_BOND', '*1', 1, '*3'],
            ['GAIN_RADICAL', '*2', '1'],
            ['LOSE_RADICAL', '*1', '1'],
        ])"""
        r_xyz = """F       1.93592759   -1.04813200    0.17239309
C       1.41395997   -0.06443750   -0.60748935
C       0.46854139    0.77821484    0.23269059
C       1.16469946    1.45000317    1.41577429
C       2.13600384    2.49526387    0.98914077
F       1.69221606    3.70990602    0.60332208
F       3.45162393    2.20655153    0.91224277
H       2.23977740    0.51935595   -1.02311040
H       0.87599990   -0.54232434   -1.43132912
H      -0.01588539    1.53022886   -0.40118094
H      -0.31963114    0.12637206    0.62794629
H       0.40903520    1.92224463    2.05360591
H       1.67965177    0.70327850    2.03007255"""
        p_xyz = """C      -2.10258623    0.28609914   -0.11161659
C      -0.80850454   -0.44729615    0.01949484
C      -0.27209648   -0.40163127    1.44584029
C       1.03111915   -1.15786446    1.56235292
F       1.97934384   -0.63177629    0.75822896
F       0.87880578   -2.45776869    1.23195390
F       1.49664262   -1.10927826    2.83007421
H      -2.25664107    1.23858311    0.38402441
H      -2.81716662   -0.01824459   -0.86926845
H      -0.96292814   -1.48784377   -0.28803477
H      -0.08395313   -0.00553132   -0.67357116
H      -1.00377942   -0.83558539    2.13782580
H      -0.11333646    0.63795904    1.75659256"""
        r = ARCSpecies(label='R', smiles='FCCC[C](F)F', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='[CH2]CCC(F)(F)F', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_intra_substitutioncs_cyclization(self):
        """Test the interpolate_isomerization() function for intra_substitutionCS_cyclization: """
        """recipe(actions=[
            ['BREAK_BOND', '*1', 1, '*2'],
            ['FORM_BOND', '*1', 1, '*3'],
            ['GAIN_RADICAL', '*2', '1'],
            ['LOSE_RADICAL', '*3', '1'],
        ])"""
        pass

    def test_interpolate_intra_substitutioncs_isomerization(self):
        """Test the interpolate_isomerization() function for Intra_substitutionCS_isomerization: [CH2]SC <=> CC[S]"""
        r_xyz = """C       1.49359756    0.26065949    0.12401718
S       0.26300557    0.39883151   -1.09862837
C      -1.16266168   -0.18440138   -0.15528675
H       2.53018726    0.54294783   -0.11178158
H       1.26337431   -0.10426874    1.13589597
H      -1.33584607    0.46046027    0.71053683
H      -0.99959059   -1.21037607    0.18570227
H      -2.05206195   -0.16387486   -0.79046245"""
        p_xyz = """C       0.77758633   -0.01229993    0.03615697
C      -0.74224664   -0.12075638   -0.00763792
S      -1.45914811    1.52171047   -0.37183152
H       1.11694877    0.67695693    0.81818542
H       1.19225093    0.32412348   -0.92127862
H       1.21257173   -0.99402300    0.25412065
H      -1.08603665   -0.52423142    0.95210784
H      -1.01192637   -0.87148015   -0.75982286"""
        r = ARCSpecies(label='R', smiles='[CH2]SC', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='CC[S]', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertIsNotNone(ts_xyzs)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 8)
            self.assertFalse(colliding_atoms(ts_xyz))

    def test_interpolate_intra_substitutions_cyclization(self):
        """Test the interpolate_isomerization() function for intra_substitutionS_cyclization: C[CH]CCCSC <=> CC1CCCS1"""
        """recipe(actions=[
            ['BREAK_BOND', '*1', 1, '*2'],
            ['FORM_BOND', '*1', 1, '*3'],
            ['GAIN_RADICAL', '*2', '1'],
            ['LOSE_RADICAL', '*3', '1'],
        ])"""
        r_xyz = """C      -3.33147467   -0.41330899    0.93687068
C      -2.13790604   -0.63163510    0.07402846
C      -2.32541520   -0.95568346   -1.37476606
C      -1.06208205   -1.56878759   -1.98555502
C      -1.28629342   -1.91107099   -3.45681094
S       0.22406608   -2.53992437   -4.26214101
C       0.33764157   -4.16462509   -3.47260022
H      -4.07068858   -1.20628187    0.79133558
H      -3.03674990   -0.41559284    1.99046133
H      -3.79427355    0.55142730    0.71074741
H      -1.17649306   -0.26532029    0.42123272
H      -3.16065335   -1.65682284   -1.48728704
H      -2.58931203   -0.03693874   -1.91110959
H      -0.22931624   -0.86017174   -1.88828407
H      -0.79159608   -2.46846463   -1.42015621
H      -2.08869602   -2.64719826   -3.57351283
H      -1.59007794   -1.01154471   -4.00308240
H      -0.59885206   -4.71570801   -3.59080849
H       1.13722657   -4.73681109   -3.95100063
H       0.57876160   -4.06472416   -2.41186409"""
        p_xyz = """C       2.12609188   -0.45688636    0.32417521
C       0.88393506    0.34021073   -0.03446926
C       1.18807825    1.53629428   -0.93178724
C      -0.12400601    1.91288456   -1.60355628
C      -0.68058047    0.63413926   -2.20580487
S      -0.30025782   -0.66085451   -0.99327157
H       2.85325448    0.16359903    0.85920907
H       2.61376895   -0.82849161   -0.58361465
H       1.89021722   -1.32413223    0.95109396
H       0.37892066    0.67320874    0.87908640
H       1.93980374    1.28478919   -1.69147877
H       1.58662389    2.37274285   -0.34751913
H       0.02548364    2.67725081   -2.37294810
H      -0.82373021    2.32524926   -0.86544695
H      -1.75666269    0.70147335   -2.38653769
H      -0.17919466    0.39223015   -3.14806886"""
        r = ARCSpecies(label='R', smiles='C[CH]CCCSC', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='CC1CCCS1', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate(rxn)
        self.assertGreater(len(ts_xyzs), 0)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))

    def test_interpolate_intra_substitutions_isomerization(self):
        """Test the interpolate_isomerization() function for Intra_substitutionS_isomerization: [CH2]SSC <=> CSC[S]"""
        r_xyz = """C       2.02473594    0.05810114    0.12967514
S       0.94173618    1.38848441   -0.00439602
S       1.99155683    2.55179194   -1.33352089
C       3.05975458    3.50692441   -0.22777177
H       1.79171393   -0.74186961    0.82204853
H       2.90913559   -0.02956306   -0.49048675
H       3.72773084    2.84617735    0.33119562
H       3.67272000    4.18684912   -0.82584520
H       2.46084746    4.10465096    0.46458235"""
        p_xyz = """C       1.39454780   -0.02562661   -0.01611442
S       0.02609945   -0.59423158   -1.05475409
C       0.59788634    0.03226201   -2.67186639
S       0.44786605    1.84643452   -2.74780581
H       1.38239969    1.06036220    0.10207113
H       1.28884668   -0.47108212    0.97718956
H       2.35414865   -0.34501856   -0.43126943
H      -0.01997740   -0.46631648   -3.42654341
H       1.61593633   -0.33730052   -2.83543977"""
        r = ARCSpecies(label='R', smiles='[CH2]SSC', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='CSC[S]', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn, weight=0.5)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertIsNotNone(ts_xyzs)
        self.assertIsInstance(ts_xyzs, list)

    def test_interpolate_lone_electron_pair_bond(self):
        """Test the interpolate_isomerization() function for lone_electron_pair_bond: """
        """recipe(actions=[
            ['LOSE_PAIR', '*1', '1'],
            ['FORM_BOND', '*1', 1, '*2'],
        ])"""
        pass

    def test_linear_adapter(self):
        """Test the LinearAdapter class."""
        self.assertEqual(self.rxn_1.family, 'Cyclopentadiene_scission')
        linear_1 = LinearAdapter(job_type='tsg',
                                 reactions=[self.rxn_1],
                                 testing=True,
                                 project='test',
                                 project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_linear', 'rxn_1'),
                                 )
        self.assertIsNone(self.rxn_1.ts_species)
        linear_1.execute()
        self.assertGreater(len(self.rxn_1.ts_species.ts_guesses), 0)
        self.assertEqual(self.rxn_1.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'))
        # todo, add actual geometry tests

    def test_linear_adapter_2(self):
        self.rxn_2.family = 'intra_NO2_ONO_conversion'
        self.rxn_2.atom_map = [0, 1, 3, 2, 4, 5, 7, 6, 9, 8]
        self.assertEqual(self.rxn_2.family, 'intra_NO2_ONO_conversion')
        linear_2 = LinearAdapter(job_type='tsg',
                                 reactions=[self.rxn_2],
                                 testing=True,
                                 project='test',
                                 project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_linear', 'rxn_2'),
                                 )
        self.assertIsNone(self.rxn_2.ts_species)
        linear_2.execute()
        self.assertGreater(len(self.rxn_2.ts_species.ts_guesses), 0)
        self.assertEqual(self.rxn_2.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'O', 'N', 'O', 'H', 'H', 'H', 'H', 'H'))
        expected_xyz = 1  # todo

    def test_get_r_constraints(self):
        """Test the get_r_constraints() function."""
        self.assertEqual(get_r_constraints([(1, 5)], [(0, 5)]), {'R_atom': [(5, 1)]})
        # Atoms 1,2,5,7,8 all have frequency 2; atoms 0,4 have frequency 1.
        # New tie-breaking: sort equal-frequency atoms by atom index ascending → [1,2,5,7,8,0,4].
        # Constraint selection picks bonds that introduce a new atom relative to already-seen atoms.
        self.assertEqual(get_r_constraints([(1, 5), (7, 2), (8, 2)], [(0, 5), (7, 4), (8, 1)]),
                         {'R_atom': [(1, 5), (2, 7), (5, 0), (7, 4)]})

    def test_interp_dihedral_deg(self):
        """Test the interp_dihedral_deg() function."""
        self.assertEqual(interp_dihedral_deg(50, 100, 0.5), 75)
        self.assertEqual(interp_dihedral_deg(50, 100, 0.0), 50)
        self.assertEqual(interp_dihedral_deg(50, 100, 1.0), 100)
        self.assertEqual(interp_dihedral_deg(250, 360, 0.5), -55)
        self.assertEqual(interp_dihedral_deg(180, -180, 0.5), -180)
        self.assertEqual(interp_dihedral_deg(178, -178, 0.5), -180)
        self.assertEqual(interp_dihedral_deg(178, -170, 0.5), -176)

    def test_clip01(self):
        """Test the _clip01 helper."""
        self.assertEqual(_clip01(0.5), 0.5)
        self.assertEqual(_clip01(0.0), 0.0)
        self.assertEqual(_clip01(1.0), 1.0)
        self.assertEqual(_clip01(-1.0), 0.0)
        self.assertEqual(_clip01(2.0), 1.0)
        self.assertEqual(_clip01(-0.001), 0.0)
        self.assertEqual(_clip01(1.001), 1.0)

    # -----------------------------------------------------------------------
    # Integration tests for the dual-topology Z-matrix chimera engine
    # -----------------------------------------------------------------------

    def test_interpolate_isomerization_ring_opening_no_crash(self):
        """Integration test 1: ring-opening/scission handles find_smart_anchors gracefully.

        Cyclopentadiene_scission opens a 5-membered ring to give a linear C5 carbene.
        Rings often lack spectator chains long enough for the preferred three-atom
        anchor, exercising find_smart_anchors's fallback logic.

        Assertions:
          * No exception is raised during anchor selection or Z-matrix generation.
          * The function returns a non-empty list (colliding_atoms did not reject all
            candidates).
          * Every surviving guess has the correct atom count and no colliding atoms.
        """
        ts_xyzs = interpolate_isomerization(self.rxn_1, weight=0.5)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertIsNotNone(ts_xyzs)
        self.assertIsInstance(ts_xyzs, list)
        # At least one valid TS guess must survive the collision filter.
        self.assertGreater(
            len(ts_xyzs), 0,
            msg='interpolate_isomerization must produce ≥1 TS guess for '
                'Cyclopentadiene_scission (rxn_1); all candidates were rejected.',
        )
        n_atoms = len(self.rxn_1.r_species[0].get_xyz()['symbols'])
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), n_atoms,
                             msg='TS guess has wrong atom count.')
            self.assertFalse(
                colliding_atoms(ts_xyz),
                msg=f'Collision detected in Cyclopentadiene_scission TS guess:\n'
                    f'{xyz_to_str(ts_xyz)}',
            )

    def test_interpolate_isomerization_degenerate_dedup(self):
        """Integration test 2: degenerate 1,5-H shift — dedup removes near-identical chimeras.

        Penta-1,3-diene (CC=CC=C) undergoes a thermally allowed 1,5-H sigmatropic
        shift that is degenerate: reactant and product are the same molecule (only the
        atom-index labelling changes).

        The test verifies three properties of the dedup pass:

        1. No exception is raised during the full pipeline.
        2. Every surviving guess has the correct atom count and no colliding atoms.
        3. **No two entries in the returned list are near-identical** — if dedup had
           failed, the list would contain redundant copies that
           ``almost_equal_coords`` would flag.  A clean returned list is the
           invariant that the pruning logic guarantees.

        The count of surviving guesses is left unconstrained because non-identity
        atom_maps create genuinely distinct Type-R and Type-P chimeras even for
        degenerate reactions; the essential property is uniqueness within the output.
        """
        # Penta-1,3-diene s-trans planar geometry (5C + 8H = 13 atoms, SMILES CC=CC=C)
        penta_13_diene_xyz = """C    2.6362    0.0000    0.0000
C    1.3442    0.6930    0.0000
C    0.0000    0.0000    0.0000
C   -1.3442    0.6930    0.0000
C   -2.6362    0.0000    0.0000
H    2.5820   -0.6289    0.8928
H    2.5820   -0.6289   -0.8928
H    3.6014    0.5018    0.0000
H    1.3970    1.7729    0.0000
H    0.0000   -1.0847    0.0000
H   -1.3970    1.7729    0.0000
H   -3.6014    0.5018    0.0000
H   -2.6362   -1.0847    0.0000"""
        r = ARCSpecies(label='R', smiles='CC=CC=C', xyz=penta_13_diene_xyz)
        p = ARCSpecies(label='P', smiles='CC=CC=C', xyz=penta_13_diene_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn, weight=0.5)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertIsNotNone(ts_xyzs)
        # Every surviving guess must have the correct atom count and no collisions.
        for ts_xyz in ts_xyzs:
            self.assertEqual(len(ts_xyz['symbols']), 13,
                             msg='TS guess has wrong atom count for penta-1,3-diene.')
            self.assertFalse(colliding_atoms(ts_xyz),
                             msg=f'Collision in penta-1,3-diene TS guess:\n{xyz_to_str(ts_xyz)}')
        # Key invariant: the returned list must contain no near-duplicate entries.
        # If dedup had missed a pair, almost_equal_coords would flag it here.
        for i in range(len(ts_xyzs)):
            for j in range(i + 1, len(ts_xyzs)):
                self.assertFalse(
                    almost_equal_coords(ts_xyzs[i], ts_xyzs[j]),
                    msg=f'Duplicate TS guesses at indices {i} and {j} — '
                        f'the almost_equal_coords dedup pass left a redundant entry.',
                )

    def test_interpolate_isomerization_linear_alkyne_no_nan(self):
        """Integration test 3: sp-hybridised (linear) groups do not produce NaN or crash.

        Terminal alkynes (H-C≡C-) have H-C≡C angles of ~180°.  zmat_from_xyz inserts
        dummy atoms to avoid collinear singularities, producing DX_ / AX_ Z-matrix
        variables.  These must be classified correctly by column position inside
        average_zmat_params (column 2 → dihedral) and must pass through zmat_to_xyz
        without generating NaN/Inf coordinates, ZeroDivisionErrors, or ValueErrors.

        Uses 6_membered_central_C-C_shift with hex-1,5-diyne (C#CCCC#C) →
        hexa-1,2,4,5-tetraene (C=C=CC=C=C).  Both endpoints contain terminal alkynes
        with genuinely linear bond angles.

        Assertions:
          * No exception is raised from dummy-atom arithmetic.
          * Every surviving TS guess has entirely finite (non-NaN, non-Inf) coordinates.
          * No surviving guess contains colliding atoms.
        """
        import math

        r_xyz = """C    3.03272979   -0.11060195   -0.24229461
C    1.85599055   -0.34675713   -0.20247149
C    0.41485966   -0.64142590   -0.15352412
C   -0.41485965    0.64142578   -0.17240633
C   -1.85599061    0.34675702   -0.12346178
C   -3.03272995    0.11060190   -0.08364096
H    4.07762286    0.09693448   -0.27758589
H    0.19106566   -1.21954180    0.75163518
H    0.14301783   -1.27648597   -1.00582442
H   -0.19106412    1.21954271   -1.07756459
H   -0.14301928    1.27648492    0.67989514
H   -4.07762310   -0.09693448   -0.04835177"""
        p_xyz = """C   -3.03124363    0.21595810   -0.01068883
C   -1.77136356   -0.00875193   -0.22839960
C   -0.51035344   -0.23538255   -0.44913569
C    0.51035356    0.23538291    0.44913621
C    1.77136365    0.00875234    0.22839985
C    3.03124358   -0.21595777    0.01068824
H   -3.50880107    1.10742857   -0.40051872
H   -3.62554573   -0.48341738    0.56587595
H   -0.21235801   -0.79338469   -1.33170668
H    0.21235823    0.79338484    1.33170737
H    3.50880076   -1.10742925    0.40051615
H    3.62554580    0.48341866   -0.56587535"""
        r = ARCSpecies(label='R', smiles='C#CCCC#C', xyz=r_xyz)
        p = ARCSpecies(label='P', smiles='C=C=CC=C=C', xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])
        # Must complete without raising ValueError, ZeroDivisionError, or similar.
        ts_xyzs = interpolate_isomerization(rxn, weight=0.5)
        for ts_xyz in ts_xyzs:
            print('\n\n***********')
            print(xyz_to_str(ts_xyz))
        self.assertIsNotNone(ts_xyzs)
        self.assertIsInstance(ts_xyzs, list)
        # Verify that no NaN or Inf coordinates propagate from dummy-atom arithmetic.
        for ts_xyz in ts_xyzs:
            for coord_triple in ts_xyz['coords']:
                for val in coord_triple:
                    self.assertTrue(
                        math.isfinite(val),
                        msg=f'Non-finite coordinate {val!r} in TS guess — '
                            f'dummy-atom arithmetic produced NaN or Inf:\n'
                            f'{xyz_to_str(ts_xyz)}',
                    )
            # No collision in surviving guesses.
            self.assertFalse(
                colliding_atoms(ts_xyz),
                msg=f'Collision detected in TS guess for linear-alkyne system:\n'
                    f'{xyz_to_str(ts_xyz)}',
            )

    # ------------------------------------------------------------------ #
    # Empty stubs — one per linear-supported RMG family lacking a test.  #
    # Fill each with a real reaction and assertions when ready.           #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #                   Unit tests for helper functions                    #
    # ------------------------------------------------------------------ #

    def test_find_split_bonds_by_fragmentation_simple(self):
        """Test _find_split_bonds_by_fragmentation for a simple A -> B + C dissociation."""
        # Ethane C2H6: cutting the C-C bond gives two CH3 fragments.
        uni = ARCSpecies(label='ethane', smiles='CC',
                         xyz=str_to_xyz("""C      -0.75560000    0.00000000    0.00000000
C       0.75560000    0.00000000    0.00000000
H      -1.16560000    0.99270000    0.00000000
H      -1.16560000   -0.49640000    0.85960000
H      -1.16560000   -0.49640000   -0.85960000
H       1.16560000   -0.99270000    0.00000000
H       1.16560000    0.49640000   -0.85960000
H       1.16560000    0.49640000    0.85960000"""))
        p1 = ARCSpecies(label='CH3_1', smiles='[CH3]',
                        xyz=str_to_xyz("""C       0.00000000    0.00000000    0.00000000
H       1.07900000    0.00000000    0.00000000
H      -0.53950000    0.93440000    0.00000000
H      -0.53950000   -0.93440000    0.00000000"""))
        p2 = ARCSpecies(label='CH3_2', smiles='[CH3]',
                        xyz=str_to_xyz("""C       0.00000000    0.00000000    0.00000000
H       1.07900000    0.00000000    0.00000000
H      -0.53950000    0.93440000    0.00000000
H      -0.53950000   -0.93440000    0.00000000"""))
        cuts = _find_split_bonds_by_fragmentation(uni.mol, [p1, p2])
        self.assertEqual(len(cuts), 1)
        self.assertEqual(len(cuts[0]), 1)
        # The C-C bond should be (0, 1).
        bond = cuts[0][0]
        self.assertEqual(bond, (0, 1))

    def test_find_split_bonds_by_fragmentation_no_valid_cut(self):
        """Test _find_split_bonds_by_fragmentation returns empty when no cut matches."""
        # Methane can't be split into two CH3 fragments.
        uni = ARCSpecies(label='methane', smiles='C',
                         xyz=str_to_xyz("""C       0.00000000    0.00000000    0.00000000
H       0.62910000    0.62910000    0.62910000
H      -0.62910000   -0.62910000    0.62910000
H      -0.62910000    0.62910000   -0.62910000
H       0.62910000   -0.62910000   -0.62910000"""))
        p = ARCSpecies(label='CH3', smiles='[CH3]',
                       xyz=str_to_xyz("""C       0.00000000    0.00000000    0.00000000
H       1.07900000    0.00000000    0.00000000
H      -0.53950000    0.93440000    0.00000000
H      -0.53950000   -0.93440000    0.00000000"""))
        cuts = _find_split_bonds_by_fragmentation(uni.mol, [p, p])
        self.assertEqual(cuts, [])

    def test_find_split_bonds_by_fragmentation_relaxed_h(self):
        """Test _find_split_bonds_by_fragmentation with H redistribution (relaxed matching)."""
        # H2O2 -> OH + OH: strict match. Single bond cut O-O gives two OH.
        uni = ARCSpecies(label='H2O2', smiles='OO',
                         xyz=str_to_xyz("""O       0.00000000    0.72720000   -0.05290000
O       0.00000000   -0.72720000   -0.05290000
H       0.79470000    0.89780000    0.47720000
H      -0.79470000   -0.89780000    0.47720000"""))
        p1 = ARCSpecies(label='OH1', smiles='[OH]',
                        xyz=str_to_xyz("""O       0.00000000    0.00000000    0.10890000
H       0.00000000    0.00000000   -0.87120000"""))
        p2 = ARCSpecies(label='OH2', smiles='[OH]',
                        xyz=str_to_xyz("""O       0.00000000    0.00000000    0.10890000
H       0.00000000    0.00000000   -0.87120000"""))
        cuts = _find_split_bonds_by_fragmentation(uni.mol, [p1, p2])
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0][0], (0, 1))

    def test_stretch_bond_simple_dissociation(self):
        """Test _stretch_bond for a simple 2-fragment stretch."""
        # Use ethane: cut C-C bond, stretch the smaller fragment away.
        uni = ARCSpecies(label='ethane', smiles='CC',
                         xyz=str_to_xyz("""C      -0.75560000    0.00000000    0.00000000
C       0.75560000    0.00000000    0.00000000
H      -1.16560000    0.99270000    0.00000000
H      -1.16560000   -0.49640000    0.85960000
H      -1.16560000   -0.49640000   -0.85960000
H       1.16560000   -0.99270000    0.00000000
H       1.16560000    0.49640000   -0.85960000
H       1.16560000    0.49640000    0.85960000"""))
        split_bonds = [(0, 1)]
        ts_xyz = _stretch_bond(uni.get_xyz(), uni.mol, split_bonds, weight=0.5)
        self.assertIsNotNone(ts_xyz)
        # The C-C distance in the TS should be larger than in ethane (~1.54 Å).
        coords = np.array(ts_xyz['coords'])
        d_cc = float(np.linalg.norm(coords[0] - coords[1]))
        self.assertGreater(d_cc, 1.54, msg=f'C-C distance not stretched: {d_cc:.3f} Å')
        # Should not have collisions.
        self.assertFalse(colliding_atoms(ts_xyz))

    def test_stretch_bond_returns_none_for_no_fragmentation(self):
        """Test _stretch_bond returns None when split bonds don't fragment the molecule."""
        # Try to split ethane on a C-H bond — doesn't fragment into 2 heavy fragments,
        # but does give 2 fragments (H and C2H5).
        uni = ARCSpecies(label='ethane', smiles='CC',
                         xyz=str_to_xyz("""C      -0.75560000    0.00000000    0.00000000
C       0.75560000    0.00000000    0.00000000
H      -1.16560000    0.99270000    0.00000000
H      -1.16560000   -0.49640000    0.85960000
H      -1.16560000   -0.49640000   -0.85960000
H       1.16560000   -0.99270000    0.00000000
H       1.16560000    0.49640000   -0.85960000
H       1.16560000    0.49640000    0.85960000"""))
        # Split on a non-existent bond — should not fragment.
        ts_xyz = _stretch_bond(uni.get_xyz(), uni.mol, [(3, 5)], weight=0.5)
        # Bond (3,5) doesn't exist in ethane mol graph, so removing it won't fragment.
        # This should return None since fragments < 2.
        self.assertIsNone(ts_xyz)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_linear'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
