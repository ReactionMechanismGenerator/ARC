#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.species.nd_scan module
"""

import math
import unittest

from arc.exceptions import SchedulerError
from arc.species.nd_scan import (decrement_running_jobs,
                                 finalize_directed_scan_results,
                                 fit_adaptive_surface_model,
                                 format_dihedral_key,
                                 generate_adaptive_candidate_points,
                                 generate_adaptive_seed_points,
                                 get_completed_adaptive_points,
                                 get_continuous_scan_dihedrals,
                                 get_pending_adaptive_points,
                                 get_rotor_dict_by_pivots,
                                 get_torsion_dihedral_grid,
                                 get_adaptive_stopping_reason,
                                 increment_continuous_scan_indices,
                                 initialize_adaptive_scan_state,
                                 initialize_continuous_scan_state,
                                 is_adaptive_eligible,
                                 is_adaptive_enabled,
                                 is_adaptive_scan_complete,
                                 is_continuous_scan_complete,
                                 iter_brute_force_scan_points,
                                 mark_scan_point_completed,
                                 mark_scan_point_failed,
                                 mark_scan_point_invalid,
                                 mark_scan_points_pending,
                                 predict_surface_values,
                                 score_candidate_points,
                                 select_next_adaptive_points,
                                 should_continue_adaptive_scan,
                                 normalize_directed_scan_energies,
                                 record_directed_scan_point,
                                 validate_scan_resolution,
                                 calculate_neighbor_energy_jump,
                                 calculate_neighbor_geometry_rmsd,
                                 classify_neighbor_edge_continuity,
                                 detect_branch_jump_points,
                                 get_sampled_point_neighbors,
                                 iter_sampled_neighbor_edges,
                                 run_adaptive_surface_validation,
                                 update_adaptive_validation_state,
                                 check_periodic_edge_consistency,
                                 extract_adaptive_2d_surface_arrays,
                                 fit_separable_surface_proxy,
                                 calculate_separable_fit_error,
                                 calculate_nonseparability_score,
                                 calculate_cross_term_strength,
                                 calculate_low_energy_path_coupling,
                                 compute_coupling_metrics,
                                 compute_surface_quality_metrics,
                                 calculate_coverage_fraction,
                                 update_nd_classification,
                                 COUPLING_NONSEP_THRESHOLD,
                                 )


class TestNDScan(unittest.TestCase):
    """
    Contains unit tests for the nd_scan module
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None

    def test_validate_scan_resolution(self):
        """Test scan resolution validation."""
        # Valid resolutions
        validate_scan_resolution(8.0)
        validate_scan_resolution(10.0)
        validate_scan_resolution(1.0)
        validate_scan_resolution(120.0)
        validate_scan_resolution(360.0)

        # Invalid resolutions
        with self.assertRaises(SchedulerError):
            validate_scan_resolution(7.0)
        with self.assertRaises(SchedulerError):
            validate_scan_resolution(11.0)
        with self.assertRaises(SchedulerError):
            validate_scan_resolution(13.0)

    def test_get_torsion_dihedral_grid(self):
        """Test generating dihedral grids from xyz coordinates."""
        # Create a simple xyz dict: 4 atoms in a known geometry
        # Using a methanol-like geometry (H-O-C-H dihedral)
        xyz = {'symbols': ('H', 'O', 'C', 'H'),
               'isotopes': (1, 16, 12, 1),
               'coords': ((0.0, 0.0, 1.0),
                           (0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0),
                           (1.5, 1.0, 0.0))}
        torsions = [[0, 1, 2, 3]]
        increment = 120.0  # 3 + 1 = 4 points for easy verification
        grid = get_torsion_dihedral_grid(xyz, torsions, increment)

        self.assertEqual(len(grid), 1)
        key = tuple(torsions[0])
        self.assertIn(key, grid)
        self.assertEqual(len(grid[key]), int(360 / 120) + 1)  # 4 points
        # All angles should be in -180..+180 range
        for angle in grid[key]:
            self.assertGreaterEqual(angle, -180.0)
            self.assertLessEqual(angle, 180.0)

    def test_iter_brute_force_scan_points_1d(self):
        """Test 1D brute-force point generation."""
        dihedrals = {(0, 1, 2, 3): [0.0, 90.0, 180.0, -90.0]}
        torsions = [[0, 1, 2, 3]]
        points = list(iter_brute_force_scan_points(dihedrals, torsions, diagonal=False))
        self.assertEqual(len(points), 4)
        self.assertEqual(points[0], (0.0,))
        self.assertEqual(points[1], (90.0,))
        self.assertEqual(points[2], (180.0,))
        self.assertEqual(points[3], (-90.0,))

    def test_iter_brute_force_scan_points_2d(self):
        """Test 2D brute-force point generation (cartesian product)."""
        dihedrals = {(0, 1, 2, 3): [0.0, 120.0, -120.0],
                     (4, 5, 6, 7): [10.0, 130.0, -110.0]}
        torsions = [[0, 1, 2, 3], [4, 5, 6, 7]]
        points = list(iter_brute_force_scan_points(dihedrals, torsions, diagonal=False))
        # 3 x 3 = 9 combinations
        self.assertEqual(len(points), 9)
        # First point: first angle of each torsion
        self.assertEqual(points[0], (0.0, 10.0))
        # Second point: first torsion stays, second increments
        self.assertEqual(points[1], (0.0, 130.0))
        # Last point
        self.assertEqual(points[-1], (-120.0, -110.0))

    def test_iter_brute_force_scan_points_2d_diagonal(self):
        """Test 2D diagonal brute-force point generation."""
        dihedrals = {(0, 1, 2, 3): [0.0, 120.0, -120.0],
                     (4, 5, 6, 7): [10.0, 130.0, -110.0]}
        torsions = [[0, 1, 2, 3], [4, 5, 6, 7]]
        points = list(iter_brute_force_scan_points(dihedrals, torsions, diagonal=True))
        # diagonal: only 3 points (one per step)
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0], (0.0, 10.0))
        self.assertEqual(points[1], (120.0, 130.0))
        self.assertEqual(points[2], (-120.0, -110.0))

    def test_initialize_continuous_scan_state(self):
        """Test continuous scan state initialization."""
        rotor_dict = {
            'torsion': [[0, 1, 2, 3], [4, 5, 6, 7]],
            'scan': [[1, 2, 3, 4], [5, 6, 7, 8]],
            'cont_indices': list(),
            'original_dihedrals': list(),
        }
        # Non-collinear geometry so dihedral angles are well-defined.
        # scan uses 1-indexed atoms, so scan [1,2,3,4] refers to atoms 0,1,2,3 in 0-index.
        xyz = {'symbols': ('H', 'O', 'C', 'H', 'N', 'C', 'O', 'H'),
               'isotopes': (1, 16, 12, 1, 14, 12, 16, 1),
               'coords': ((0.0, 1.0, 0.5),
                           (0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0),
                           (1.5, 1.0, 0.5),
                           (3.0, 1.0, 0.0),
                           (4.0, 0.0, 0.5),
                           (5.0, 0.0, 0.0),
                           (5.5, 1.0, 0.5))}
        initialize_continuous_scan_state(rotor_dict, xyz)

        self.assertEqual(rotor_dict['cont_indices'], [0, 0])
        self.assertEqual(len(rotor_dict['original_dihedrals']), 2)
        # original_dihedrals should be strings with 2 decimal places
        for d in rotor_dict['original_dihedrals']:
            self.assertIsInstance(d, str)
            self.assertIn('.', d)

    def test_initialize_continuous_scan_state_idempotent(self):
        """Test that initialization doesn't overwrite existing state."""
        rotor_dict = {
            'torsion': [[0, 1, 2, 3]],
            'scan': [[1, 2, 3, 4]],
            'cont_indices': [5],
            'original_dihedrals': ['45.00'],
        }
        xyz = {'symbols': ('H', 'O', 'C', 'H'),
               'isotopes': (1, 16, 12, 1),
               'coords': ((0.0, 0.0, 1.0),
                           (0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0),
                           (1.5, 1.0, 0.0))}
        initialize_continuous_scan_state(rotor_dict, xyz)
        # Should NOT overwrite existing values
        self.assertEqual(rotor_dict['cont_indices'], [5])
        self.assertEqual(rotor_dict['original_dihedrals'], ['45.00'])

    def test_get_continuous_scan_dihedrals(self):
        """Test computing dihedrals for a continuous scan step."""
        rotor_dict = {
            'torsion': [[0, 1, 2, 3]],
            'original_dihedrals': ['0.00'],
            'cont_indices': [3],
        }
        increment = 10.0
        dihedrals = get_continuous_scan_dihedrals(rotor_dict, increment)
        self.assertEqual(len(dihedrals), 1)
        self.assertAlmostEqual(dihedrals[0], 30.0, places=2)

    def test_get_continuous_scan_dihedrals_wrapping(self):
        """Test that continuous scan dihedrals wrap around -180..+180."""
        rotor_dict = {
            'torsion': [[0, 1, 2, 3]],
            'original_dihedrals': ['170.00'],
            'cont_indices': [3],
        }
        increment = 10.0
        dihedrals = get_continuous_scan_dihedrals(rotor_dict, increment)
        # 170 + 30 = 200, wrapped to -160
        self.assertAlmostEqual(dihedrals[0], -160.0, places=2)

    def test_is_continuous_scan_complete(self):
        """Test continuous scan completion detection."""
        increment = 10.0  # 37 grid points per dimension

        # Not complete: last index not at max
        rotor_dict_incomplete = {
            'torsion': [[0, 1, 2, 3], [4, 5, 6, 7]],
            'cont_indices': [36, 0],
        }
        self.assertFalse(is_continuous_scan_complete(rotor_dict_incomplete, increment))

        # Complete: last index at max
        rotor_dict_complete = {
            'torsion': [[0, 1, 2, 3], [4, 5, 6, 7]],
            'cont_indices': [0, 36],
        }
        self.assertTrue(is_continuous_scan_complete(rotor_dict_complete, increment))

        # 1D complete
        rotor_dict_1d_complete = {
            'torsion': [[0, 1, 2, 3]],
            'cont_indices': [36],
        }
        self.assertTrue(is_continuous_scan_complete(rotor_dict_1d_complete, increment))

    def test_increment_continuous_scan_indices_non_diagonal(self):
        """Test non-diagonal continuous scan index incrementing (odometer-style)."""
        increment = 120.0  # 4 grid points, indices 0..3

        # Simple increment of first index
        rotor_dict = {
            'torsion': [[0, 1, 2, 3], [4, 5, 6, 7]],
            'cont_indices': [0, 0],
        }
        increment_continuous_scan_indices(rotor_dict, increment, diagonal=False)
        self.assertEqual(rotor_dict['cont_indices'], [1, 0])

        # First index at max -> rolls over to 0, second increments
        rotor_dict = {
            'torsion': [[0, 1, 2, 3], [4, 5, 6, 7]],
            'cont_indices': [3, 0],
        }
        increment_continuous_scan_indices(rotor_dict, increment, diagonal=False)
        self.assertEqual(rotor_dict['cont_indices'], [0, 1])

        # Middle of scan
        rotor_dict = {
            'torsion': [[0, 1, 2, 3], [4, 5, 6, 7]],
            'cont_indices': [2, 1],
        }
        increment_continuous_scan_indices(rotor_dict, increment, diagonal=False)
        self.assertEqual(rotor_dict['cont_indices'], [3, 1])

    def test_increment_continuous_scan_indices_diagonal(self):
        """Test diagonal continuous scan index incrementing."""
        increment = 120.0

        rotor_dict = {
            'torsion': [[0, 1, 2, 3], [4, 5, 6, 7]],
            'cont_indices': [0, 0],
        }
        increment_continuous_scan_indices(rotor_dict, increment, diagonal=True)
        self.assertEqual(rotor_dict['cont_indices'], [1, 1])

        increment_continuous_scan_indices(rotor_dict, increment, diagonal=True)
        self.assertEqual(rotor_dict['cont_indices'], [2, 2])

    def test_normalize_directed_scan_energies(self):
        """Test energy normalization from a mock directed_scan dict."""
        rotor_dict = {
            'directed_scan_type': 'brute_force_opt',
            'scan': [[1, 2, 3, 4]],
            'directed_scan': {
                ('0.00',): {'energy': -100.5, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
                ('120.00',): {'energy': -100.0, 'xyz': {}, 'is_isomorphic': True, 'trsh': ['some_method']},
                ('-120.00',): {'energy': -100.3, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
            },
        }
        results, trshed_points = normalize_directed_scan_energies(rotor_dict)
        self.assertEqual(trshed_points, 1)
        self.assertEqual(results['directed_scan_type'], 'brute_force_opt')
        self.assertEqual(results['scans'], [[1, 2, 3, 4]])
        # Minimum energy is -100.5, so ('0.00',) should be 0.0
        self.assertAlmostEqual(results['directed_scan'][('0.00',)]['energy'], 0.0)
        self.assertAlmostEqual(results['directed_scan'][('120.00',)]['energy'], 0.5)
        self.assertAlmostEqual(results['directed_scan'][('-120.00',)]['energy'], 0.2)

    def test_normalize_directed_scan_energies_with_none(self):
        """Test energy normalization when some energies are None."""
        rotor_dict = {
            'directed_scan_type': 'brute_force_sp',
            'scan': [[1, 2, 3, 4]],
            'directed_scan': {
                ('0.00',): {'energy': -50.0, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
                ('90.00',): {'energy': None, 'xyz': {}, 'is_isomorphic': False, 'trsh': ['method1']},
                ('180.00',): {'energy': -45.0, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
            },
        }
        results, trshed_points = normalize_directed_scan_energies(rotor_dict)
        self.assertEqual(trshed_points, 1)
        self.assertAlmostEqual(results['directed_scan'][('0.00',)]['energy'], 0.0)
        self.assertIsNone(results['directed_scan'][('90.00',)]['energy'])
        self.assertAlmostEqual(results['directed_scan'][('180.00',)]['energy'], 5.0)

    def test_normalize_directed_scan_energies_2d(self):
        """Test energy normalization for a 2D scan."""
        rotor_dict = {
            'directed_scan_type': 'brute_force_opt',
            'scan': [[1, 2, 3, 4], [5, 6, 7, 8]],
            'directed_scan': {
                ('0.00', '0.00'): {'energy': -200.0, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
                ('0.00', '120.00'): {'energy': -195.0, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
                ('120.00', '0.00'): {'energy': -198.0, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
                ('120.00', '120.00'): {'energy': -190.0, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
            },
        }
        results, trshed_points = normalize_directed_scan_energies(rotor_dict)
        self.assertEqual(trshed_points, 0)
        self.assertAlmostEqual(results['directed_scan'][('0.00', '0.00')]['energy'], 0.0)
        self.assertAlmostEqual(results['directed_scan'][('0.00', '120.00')]['energy'], 5.0)
        self.assertAlmostEqual(results['directed_scan'][('120.00', '0.00')]['energy'], 2.0)
        self.assertAlmostEqual(results['directed_scan'][('120.00', '120.00')]['energy'], 10.0)


    def test_format_dihedral_key(self):
        """Test legacy string-tuple key formatting."""
        key = format_dihedral_key([180.0, -170.0])
        self.assertEqual(key, ('180.00', '-170.00'))

        key_1d = format_dihedral_key([0.0])
        self.assertEqual(key_1d, ('0.00',))

        key_3d = format_dihedral_key([45.123, -90.456, 0.0])
        self.assertEqual(key_3d, ('45.12', '-90.46', '0.00'))

    def test_record_directed_scan_point(self):
        """Test that record_directed_scan_point writes the exact legacy shape."""
        rotor_dict = {'directed_scan': {}}
        record_directed_scan_point(
            rotor_dict=rotor_dict,
            dihedrals=[180.0, -170.0],
            energy=-100.5,
            xyz={'symbols': ('H',), 'coords': ((0.0, 0.0, 0.0),)},
            is_isomorphic=True,
            trsh=['method1'],
        )
        key = ('180.00', '-170.00')
        self.assertIn(key, rotor_dict['directed_scan'])
        entry = rotor_dict['directed_scan'][key]
        self.assertEqual(entry['energy'], -100.5)
        self.assertEqual(entry['xyz'], {'symbols': ('H',), 'coords': ((0.0, 0.0, 0.0),)})
        self.assertTrue(entry['is_isomorphic'])
        self.assertEqual(entry['trsh'], ['method1'])

    def test_record_directed_scan_point_1d(self):
        """Test record for a 1D scan point."""
        rotor_dict = {'directed_scan': {}}
        record_directed_scan_point(
            rotor_dict=rotor_dict,
            dihedrals=[90.0],
            energy=-50.0,
            xyz=None,
            is_isomorphic=False,
            trsh=[],
        )
        key = ('90.00',)
        self.assertIn(key, rotor_dict['directed_scan'])
        self.assertIsNone(rotor_dict['directed_scan'][key]['xyz'])
        self.assertFalse(rotor_dict['directed_scan'][key]['is_isomorphic'])

    def test_record_directed_scan_point_overwrites(self):
        """Test that recording the same point again overwrites the previous entry."""
        rotor_dict = {'directed_scan': {
            ('0.00',): {'energy': -100.0, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
        }}
        record_directed_scan_point(
            rotor_dict=rotor_dict,
            dihedrals=[0.0],
            energy=-200.0,
            xyz={'new': True},
            is_isomorphic=False,
            trsh=['trsh1'],
        )
        self.assertEqual(rotor_dict['directed_scan'][('0.00',)]['energy'], -200.0)
        self.assertFalse(rotor_dict['directed_scan'][('0.00',)]['is_isomorphic'])

    def test_get_rotor_dict_by_pivots_found(self):
        """Test pivot lookup when the pivots exist."""
        rotors_dict = {
            0: {'pivots': [1, 2], 'scan': [[1, 2, 3, 4]]},
            1: {'pivots': [3, 4], 'scan': [[3, 4, 5, 6]]},
        }
        match = get_rotor_dict_by_pivots(rotors_dict, [3, 4])
        self.assertIsNotNone(match)
        idx, rd = match
        self.assertEqual(idx, 1)
        self.assertEqual(rd['pivots'], [3, 4])

    def test_get_rotor_dict_by_pivots_not_found(self):
        """Test pivot lookup when the pivots do not exist."""
        rotors_dict = {
            0: {'pivots': [1, 2], 'scan': [[1, 2, 3, 4]]},
        }
        match = get_rotor_dict_by_pivots(rotors_dict, [99, 100])
        self.assertIsNone(match)

    def test_get_rotor_dict_by_pivots_nested(self):
        """Test pivot lookup with nested pivots (list of lists)."""
        rotors_dict = {
            0: {'pivots': [[1, 2], [3, 4]], 'scan': [[1, 2, 3, 4], [3, 4, 5, 6]]},
        }
        match = get_rotor_dict_by_pivots(rotors_dict, [[1, 2], [3, 4]])
        self.assertIsNotNone(match)
        idx, rd = match
        self.assertEqual(idx, 0)

    def test_finalize_directed_scan_results_non_ess(self):
        """Test finalize produces the same payload as normalize for non-ESS scans."""
        rotor_dict = {
            'directed_scan_type': 'brute_force_opt',
            'scan': [[1, 2, 3, 4]],
            'directed_scan': {
                ('0.00',): {'energy': -100.5, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
                ('120.00',): {'energy': -100.0, 'xyz': {}, 'is_isomorphic': True, 'trsh': ['m']},
                ('-120.00',): {'energy': -100.3, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
            },
        }
        results, trshed = finalize_directed_scan_results(rotor_dict)
        self.assertEqual(trshed, 1)
        self.assertAlmostEqual(results['directed_scan'][('0.00',)]['energy'], 0.0)
        self.assertAlmostEqual(results['directed_scan'][('120.00',)]['energy'], 0.5)
        self.assertEqual(results['directed_scan_type'], 'brute_force_opt')
        self.assertEqual(results['scans'], [[1, 2, 3, 4]])

    def test_finalize_directed_scan_results_ess(self):
        """Test finalize delegates to parser for ESS scans."""
        mock_results = {'directed_scan_type': 'ess', 'scans': [[1, 2, 3, 4]], 'directed_scan': {}}
        rotor_dict = {
            'directed_scan_type': 'ess',
            'scan_path': '/fake/path.log',
            'scan': [[1, 2, 3, 4]],
            'directed_scan': {},
        }

        def mock_parse(log_file_path):
            self.assertEqual(log_file_path, '/fake/path.log')
            return [mock_results]

        results, trshed = finalize_directed_scan_results(rotor_dict, parse_nd_scan_energies_func=mock_parse)
        self.assertEqual(trshed, 0)
        self.assertIs(results, mock_results)

    def test_finalize_directed_scan_results_ess_no_func_raises(self):
        """Test that ESS finalize raises if no parser func is given."""
        rotor_dict = {
            'directed_scan_type': 'ess',
            'scan_path': '/fake/path.log',
            'scan': [[1, 2, 3, 4]],
            'directed_scan': {},
        }
        with self.assertRaises(ValueError):
            finalize_directed_scan_results(rotor_dict)

    def test_decrement_running_jobs(self):
        """Test brute-force job counter decrement."""
        rotor_dict = {'number_of_running_jobs': 3}
        self.assertFalse(decrement_running_jobs(rotor_dict))
        self.assertEqual(rotor_dict['number_of_running_jobs'], 2)

        self.assertFalse(decrement_running_jobs(rotor_dict))
        self.assertEqual(rotor_dict['number_of_running_jobs'], 1)

        self.assertTrue(decrement_running_jobs(rotor_dict))
        self.assertEqual(rotor_dict['number_of_running_jobs'], 0)

    def test_decrement_running_jobs_already_zero(self):
        """Test that decrementing past zero clamps to 0 and signals done."""
        rotor_dict = {'number_of_running_jobs': 0}
        result = decrement_running_jobs(rotor_dict)
        self.assertTrue(result)  # clamped to 0, treated as done
        self.assertEqual(rotor_dict['number_of_running_jobs'], 0)


class TestAdaptiveNDScan(unittest.TestCase):
    """
    Contains unit tests for the adaptive sparse 2D scan functionality in nd_scan module.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.maxDiff = None
        # Non-collinear 8-atom geometry for 2D scans
        cls.xyz = {
            'symbols': ('H', 'O', 'C', 'H', 'N', 'C', 'O', 'H'),
            'isotopes': (1, 16, 12, 1, 14, 12, 16, 1),
            'coords': ((0.0, 1.0, 0.5),
                       (0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0),
                       (1.5, 1.0, 0.5),
                       (3.0, 1.0, 0.0),
                       (4.0, 0.0, 0.5),
                       (5.0, 0.0, 0.0),
                       (5.5, 1.0, 0.5)),
        }

    def _make_2d_rotor_dict(self, scan_type='brute_force_opt', policy='dense'):
        """Helper to create a 2D rotor dict for testing."""
        rd = {
            'pivots': [[1, 2], [5, 6]],
            'top': [[0], [7]],
            'scan': [[1, 2, 3, 4], [5, 6, 7, 8]],
            'torsion': [[0, 1, 2, 3], [4, 5, 6, 7]],
            'number_of_running_jobs': 0,
            'success': None,
            'invalidation_reason': '',
            'times_dihedral_set': 0,
            'trsh_counter': 0,
            'trsh_methods': list(),
            'scan_path': '',
            'directed_scan_type': scan_type,
            'directed_scan': dict(),
            'dimensions': 2,
            'original_dihedrals': list(),
            'cont_indices': list(),
            'symmetry': None,
            'max_e': None,
        }
        if policy == 'adaptive':
            rd['sampling_policy'] = 'adaptive'
        return rd

    # -- Eligibility tests --

    def test_is_adaptive_eligible_brute_force_2d(self):
        """Test eligibility for a 2D brute_force_opt rotor."""
        rd = self._make_2d_rotor_dict('brute_force_opt')
        self.assertTrue(is_adaptive_eligible(rd))

    def test_is_adaptive_eligible_brute_force_sp_2d(self):
        """Test eligibility for a 2D brute_force_sp rotor."""
        rd = self._make_2d_rotor_dict('brute_force_sp')
        self.assertTrue(is_adaptive_eligible(rd))

    def test_is_adaptive_ineligible_ess(self):
        """Test ESS scans are not eligible."""
        rd = self._make_2d_rotor_dict('ess')
        self.assertFalse(is_adaptive_eligible(rd))

    def test_is_adaptive_ineligible_cont_opt(self):
        """Test continuous scans are not eligible."""
        rd = self._make_2d_rotor_dict('cont_opt')
        self.assertFalse(is_adaptive_eligible(rd))

    def test_is_adaptive_ineligible_diagonal(self):
        """Test diagonal brute-force scans are not eligible."""
        rd = self._make_2d_rotor_dict('brute_force_opt_diagonal')
        self.assertFalse(is_adaptive_eligible(rd))

    def test_is_adaptive_ineligible_1d(self):
        """Test 1D scans are not eligible."""
        rd = self._make_2d_rotor_dict('brute_force_opt')
        rd['dimensions'] = 1
        self.assertFalse(is_adaptive_eligible(rd))

    def test_is_adaptive_enabled_default_dense(self):
        """Test that adaptive is disabled by default (dense policy)."""
        rd = self._make_2d_rotor_dict('brute_force_opt')
        self.assertFalse(is_adaptive_enabled(rd))

    def test_is_adaptive_enabled_with_policy(self):
        """Test that adaptive is enabled when policy is 'adaptive'."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        self.assertTrue(is_adaptive_enabled(rd))

    # -- Initialization tests --

    def test_initialize_adaptive_scan_state(self):
        """Test adaptive state initialization."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        state = rd['adaptive_scan']
        self.assertTrue(state['enabled'])
        self.assertEqual(state['phase'], 'seed')
        self.assertGreater(len(state['seed_points']), 0)
        self.assertEqual(state['stopping_reason'], None)
        self.assertEqual(len(state['completed_points']), 0)

    def test_initialize_adaptive_scan_state_idempotent(self):
        """Test that initialization is idempotent."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        seeds_1 = list(rd['adaptive_scan']['seed_points'])
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        seeds_2 = rd['adaptive_scan']['seed_points']
        self.assertEqual(seeds_1, seeds_2)

    # -- Seed generation tests --

    def test_seed_points_include_origin(self):
        """Test that seed points include the current-geometry point."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        seeds = generate_adaptive_seed_points(rd, self.xyz, increment=120.0)
        self.assertGreater(len(seeds), 0)
        # All seeds should be 2-element tuples
        for s in seeds:
            self.assertEqual(len(s), 2)

    def test_seed_points_no_duplicates(self):
        """Test that seed points are deduplicated."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        seeds = generate_adaptive_seed_points(rd, self.xyz, increment=10.0)
        keys = [tuple(f'{a:.2f}' for a in s) for s in seeds]
        self.assertEqual(len(keys), len(set(keys)))

    def test_seed_points_reasonable_count(self):
        """Test that seed count is between expected bounds for 8-degree resolution."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        seeds = generate_adaptive_seed_points(rd, self.xyz, increment=8.0)
        # With increment=8: coarse grid ~16x16=256, plus 1D cuts ~2x46=92, minus overlaps
        # Should be well under the full grid of 46*46=2116 but substantial
        self.assertGreater(len(seeds), 50)
        self.assertLess(len(seeds), 500)

    # -- Bookkeeping tests --

    def test_mark_scan_points_pending(self):
        """Test marking points as pending."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        points = [[0.0, 0.0], [120.0, 120.0]]
        mark_scan_points_pending(rd, points)
        self.assertEqual(len(get_pending_adaptive_points(rd)), 2)

    def test_mark_scan_points_pending_no_duplicates(self):
        """Test that pending doesn't get duplicates."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        points = [[0.0, 0.0]]
        mark_scan_points_pending(rd, points)
        mark_scan_points_pending(rd, points)
        self.assertEqual(len(get_pending_adaptive_points(rd)), 1)

    def test_mark_scan_point_completed(self):
        """Test completing a point moves it from pending and writes legacy."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        mark_scan_points_pending(rd, [[45.0, -90.0]])
        mark_scan_point_completed(rd, [45.0, -90.0], energy=-100.0, xyz={},
                                  is_isomorphic=True, trsh=[])
        self.assertEqual(len(get_pending_adaptive_points(rd)), 0)
        self.assertEqual(len(get_completed_adaptive_points(rd)), 1)
        # Also in legacy directed_scan
        self.assertIn(('45.00', '-90.00'), rd['directed_scan'])
        entry = rd['directed_scan'][('45.00', '-90.00')]
        self.assertEqual(entry['energy'], -100.0)

    def test_mark_scan_point_failed(self):
        """Test failing a point removes it from pending."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        mark_scan_points_pending(rd, [[45.0, -90.0]])
        mark_scan_point_failed(rd, [45.0, -90.0])
        self.assertEqual(len(get_pending_adaptive_points(rd)), 0)
        self.assertEqual(len(rd['adaptive_scan']['failed_points']), 1)

    def test_mark_scan_point_invalid(self):
        """Test invalidating a point."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        mark_scan_points_pending(rd, [[45.0, -90.0]])
        mark_scan_point_invalid(rd, [45.0, -90.0])
        self.assertEqual(len(get_pending_adaptive_points(rd)), 0)
        self.assertEqual(len(rd['adaptive_scan']['invalid_points']), 1)

    # -- Surrogate / model tests --

    def test_fit_adaptive_surface_model(self):
        """Test fitting a surface model from completed points."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        # Simulate some completed points
        for phi0, phi1, e in [(0.0, 0.0, -100.0), (120.0, 0.0, -95.0), (0.0, 120.0, -90.0)]:
            mark_scan_point_completed(rd, [phi0, phi1], energy=e, xyz={},
                                      is_isomorphic=True, trsh=[])
        model = fit_adaptive_surface_model(rd)
        self.assertEqual(model['type'], 'idw')
        self.assertEqual(len(model['centers']), 3)
        self.assertEqual(len(model['values']), 3)

    def test_predict_surface_values(self):
        """Test surface prediction at query points."""
        model = {
            'type': 'idw',
            'centers': [[0.0, 0.0], [120.0, 0.0]],
            'values': [0.0, 10.0],
            'length_scale': 30.0,
        }
        preds = predict_surface_values(model, [[0.0, 0.0], [120.0, 0.0], [60.0, 0.0]])
        # At exact centers, should return close to center values
        self.assertAlmostEqual(preds[0], 0.0, places=1)
        self.assertAlmostEqual(preds[1], 10.0, places=1)
        # Midpoint should be somewhere between
        self.assertGreater(preds[2], 0.0)
        self.assertLess(preds[2], 10.0)

    def test_predict_surface_values_empty_model(self):
        """Test prediction with empty model returns None."""
        model = {'centers': [], 'values': [], 'length_scale': 30.0}
        preds = predict_surface_values(model, [[0.0, 0.0]])
        self.assertIsNone(preds[0])

    def test_score_candidate_points(self):
        """Test scoring prefers distant points."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        # Complete one point at origin
        mark_scan_point_completed(rd, [0.0, 0.0], energy=-100.0, xyz={},
                                  is_isomorphic=True, trsh=[])
        fit_adaptive_surface_model(rd)
        # Score: a far point vs a near point
        scores = score_candidate_points(rd, [[180.0, 180.0], [1.0, 1.0]])
        self.assertGreater(scores[0], scores[1])

    # -- Candidate generation tests --

    def test_generate_adaptive_candidate_points(self):
        """Test candidate generation excludes visited points."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        rd['original_dihedrals'] = ['0.00', '0.00']
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        # Complete all seed points
        for s in list(rd['adaptive_scan']['seed_points']):
            mark_scan_point_completed(rd, s, energy=-100.0, xyz={},
                                      is_isomorphic=True, trsh=[])
        candidates = generate_adaptive_candidate_points(rd, increment=120.0)
        # No candidate should be in completed
        completed_keys = {tuple(f'{a:.2f}' for a in c)
                          for c in get_completed_adaptive_points(rd)}
        for c in candidates:
            key = tuple(f'{a:.2f}' for a in c)
            self.assertNotIn(key, completed_keys)

    # -- Selection tests --

    def test_select_next_adaptive_points_seed_phase(self):
        """Test that seed phase returns seed points."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0, batch_size=5)
        points = select_next_adaptive_points(rd, increment=120.0)
        self.assertGreater(len(points), 0)
        self.assertLessEqual(len(points), 5)

    def test_select_next_adaptive_points_no_resubmit(self):
        """Test that completed points are never resubmitted."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        rd['original_dihedrals'] = ['0.00', '0.00']
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0, batch_size=100)
        # Get all seeds
        batch1 = select_next_adaptive_points(rd, increment=120.0, batch_size=1000)
        mark_scan_points_pending(rd, batch1)
        for p in batch1:
            mark_scan_point_completed(rd, p, energy=-100.0, xyz={},
                                      is_isomorphic=True, trsh=[])
        # Now get adaptive batch
        batch2 = select_next_adaptive_points(rd, increment=120.0)
        batch1_keys = {tuple(f'{a:.2f}' for a in p) for p in batch1}
        for p in batch2:
            key = tuple(f'{a:.2f}' for a in p)
            self.assertNotIn(key, batch1_keys)

    # -- Stopping tests --

    def test_stopping_max_points(self):
        """Test stopping when max_points reached."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0, max_points=3)
        # Simulate 3 completed points
        for i, (a, b) in enumerate([(0.0, 0.0), (120.0, 0.0), (0.0, 120.0)]):
            mark_scan_point_completed(rd, [a, b], energy=float(-100 + i),
                                      xyz={}, is_isomorphic=True, trsh=[])
        reason = get_adaptive_stopping_reason(rd, increment=120.0)
        self.assertEqual(reason, 'max_points_reached')

    def test_stopping_grid_exhausted(self):
        """Test stopping when all grid points are visited."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        rd['original_dihedrals'] = ['0.00', '0.00']
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0,
                                       max_points=10000)
        # Complete all 4x4=16 grid points
        n = int(360 / 120.0) + 1
        for i in range(n):
            for j in range(n):
                a0 = round(((i * 120.0) + 180.0) % 360.0 - 180.0, 2)
                a1 = round(((j * 120.0) + 180.0) % 360.0 - 180.0, 2)
                mark_scan_point_completed(rd, [a0, a1], energy=-100.0,
                                          xyz={}, is_isomorphic=True, trsh=[])
        self.assertFalse(should_continue_adaptive_scan(rd, increment=120.0))

    def test_is_adaptive_scan_complete_with_pending(self):
        """Test that scan is not complete when pending points exist."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0, max_points=2)
        mark_scan_points_pending(rd, [[0.0, 0.0]])
        mark_scan_point_completed(rd, [120.0, 0.0], energy=-100.0, xyz={},
                                  is_isomorphic=True, trsh=[])
        # max_points=2, completed+pending=2, but pending > 0
        self.assertFalse(is_adaptive_scan_complete(rd, increment=120.0))

    # -- Restart / serialization tests --

    def test_adaptive_state_is_yaml_serializable(self):
        """Test that adaptive state contains only YAML-safe types."""
        import json
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        # Simulate some activity
        seeds = rd['adaptive_scan']['seed_points'][:3]
        mark_scan_points_pending(rd, seeds)
        mark_scan_point_completed(rd, seeds[0], energy=-100.0, xyz={},
                                  is_isomorphic=True, trsh=[])
        mark_scan_point_failed(rd, seeds[1])
        # json.dumps will raise if not serializable
        json.dumps(rd['adaptive_scan'])
        json.dumps(rd['sampling_policy'])

    def test_dense_rotor_unchanged_by_adaptive_code(self):
        """Test that a dense rotor dict has no adaptive_scan key."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='dense')
        self.assertNotIn('adaptive_scan', rd)
        self.assertFalse(is_adaptive_enabled(rd))

    def test_directed_scan_type_values_unchanged(self):
        """Verify that no new directed_scan_type values are introduced."""
        valid_types = {
            'ess', 'brute_force_sp', 'brute_force_opt', 'cont_opt',
            'brute_force_sp_diagonal', 'brute_force_opt_diagonal', 'cont_opt_diagonal',
        }
        # Check eligibility function doesn't accept anything outside legacy types
        for dst in valid_types:
            rd = self._make_2d_rotor_dict(dst)
            # Just verify it doesn't crash
            is_adaptive_eligible(rd)
        # A hypothetical new type should not be eligible
        rd = self._make_2d_rotor_dict('adaptive_brute_force')
        self.assertFalse(is_adaptive_eligible(rd))

    def test_finalize_adds_sparse_metadata_for_adaptive(self):
        """Test that finalize_directed_scan_results adds sparse metadata for adaptive scans."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='adaptive')
        initialize_adaptive_scan_state(rd, self.xyz, increment=120.0)
        # Complete some points
        for phi0, phi1, e in [(0.0, 0.0, -100.0), (120.0, 0.0, -95.0), (0.0, 120.0, -90.0)]:
            mark_scan_point_completed(rd, [phi0, phi1], energy=e, xyz={},
                                      is_isomorphic=True, trsh=[])
        mark_scan_point_failed(rd, [60.0, 60.0])
        results, trshed = finalize_directed_scan_results(rd)
        self.assertEqual(results['sampling_policy'], 'adaptive')
        self.assertIn('adaptive_scan_summary', results)
        summary = results['adaptive_scan_summary']
        self.assertEqual(summary['completed_count'], 3)
        self.assertEqual(summary['failed_count'], 1)
        self.assertEqual(len(summary['failed_points']), 1)

    def test_finalize_no_sparse_metadata_for_dense(self):
        """Test that finalize_directed_scan_results has no sparse metadata for dense scans."""
        rd = self._make_2d_rotor_dict('brute_force_opt', policy='dense')
        # Add some scan points directly
        rd['directed_scan'] = {
            ('0.00', '0.00'): {'energy': -100.0, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
            ('120.00', '0.00'): {'energy': -95.0, 'xyz': {}, 'is_isomorphic': True, 'trsh': []},
        }
        results, _ = finalize_directed_scan_results(rd)
        self.assertNotIn('sampling_policy', results)
        self.assertNotIn('adaptive_scan_summary', results)


class TestSurfaceValidation(unittest.TestCase):
    """
    Contains unit tests for adaptive 2D surface validation in nd_scan module.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.maxDiff = None

    def _make_validated_rotor(self, increment=120.0, energy_fn=None):
        """
        Helper: create a 2D adaptive rotor with completed points on a 120-degree grid
        and mock xyz data. ``energy_fn(phi0, phi1) -> float`` sets the energy.
        """
        if energy_fn is None:
            energy_fn = lambda a, b: abs(a) + abs(b)  # smooth
        rd = {
            'pivots': [[1, 2], [5, 6]],
            'top': [[0], [7]],
            'scan': [[1, 2, 3, 4], [5, 6, 7, 8]],
            'torsion': [[0, 1, 2, 3], [4, 5, 6, 7]],
            'number_of_running_jobs': 0,
            'success': None,
            'invalidation_reason': '',
            'times_dihedral_set': 0,
            'trsh_counter': 0,
            'trsh_methods': [],
            'scan_path': '',
            'directed_scan_type': 'brute_force_opt',
            'directed_scan': {},
            'dimensions': 2,
            'original_dihedrals': ['0.00', '0.00'],
            'cont_indices': [],
            'symmetry': None,
            'max_e': None,
            'sampling_policy': 'adaptive',
            'adaptive_scan': {
                'enabled': True,
                'phase': 'complete',
                'batch_size': 10,
                'candidate_points': [],
                'pending_points': [],
                'completed_points': [],
                'failed_points': [],
                'invalid_points': [],
                'seed_points': [],
                'selected_points_history': [],
                'stopping_reason': 'max_points_reached',
                'max_points': 200,
                'min_points': 20,
                'fit_metadata': {},
                'surface_model': {},
            },
        }
        # Build a simple mock xyz (4 atoms, tetrahedron-like)
        base_coords = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                        (0.5, 0.87, 0.0), (0.5, 0.29, 0.82))
        base_xyz = {
            'symbols': ('C', 'H', 'H', 'H'),
            'isotopes': (12, 1, 1, 1),
            'coords': base_coords,
        }
        n = int(360 / increment) + 1
        for i in range(n):
            for j in range(n):
                a0 = round((i * increment + 180.0) % 360.0 - 180.0, 2)
                a1 = round((j * increment + 180.0) % 360.0 - 180.0, 2)
                key = (f'{a0:.2f}', f'{a1:.2f}')
                e = energy_fn(a0, a1)
                rd['directed_scan'][key] = {
                    'energy': e,
                    'xyz': base_xyz,
                    'is_isomorphic': True,
                    'trsh': [],
                }
                rd['adaptive_scan']['completed_points'].append([a0, a1])
        return rd

    # -- Neighbor helpers --

    def test_get_sampled_point_neighbors(self):
        """Test finding neighbors of a sampled point."""
        rd = self._make_validated_rotor(increment=120.0)
        neighbors = get_sampled_point_neighbors(rd, [0.0, 0.0], increment=120.0)
        # 0.0 has neighbors at ±120 in each dimension
        self.assertGreaterEqual(len(neighbors), 2)

    def test_iter_sampled_neighbor_edges(self):
        """Test iterating unique neighbor edges."""
        rd = self._make_validated_rotor(increment=120.0)
        edges = list(iter_sampled_neighbor_edges(rd, increment=120.0))
        self.assertGreater(len(edges), 0)
        # Each edge should be unique
        edge_keys = set()
        for a, b in edges:
            key = tuple(sorted([
                (f'{a[0]:.2f}', f'{a[1]:.2f}'),
                (f'{b[0]:.2f}', f'{b[1]:.2f}')
            ]))
            self.assertNotIn(key, edge_keys)
            edge_keys.add(key)

    # -- Energy jump --

    def test_calculate_neighbor_energy_jump(self):
        """Test energy jump calculation."""
        rd = self._make_validated_rotor(increment=120.0)
        jump = calculate_neighbor_energy_jump(rd, [0.0, 0.0], [120.0, 0.0])
        self.assertIsNotNone(jump)
        self.assertIsInstance(jump, float)
        self.assertGreaterEqual(jump, 0.0)

    def test_calculate_neighbor_energy_jump_missing(self):
        """Test energy jump returns None for missing point."""
        rd = self._make_validated_rotor(increment=120.0)
        jump = calculate_neighbor_energy_jump(rd, [0.0, 0.0], [999.0, 999.0])
        self.assertIsNone(jump)

    # -- Geometry RMSD --

    def test_calculate_neighbor_geometry_rmsd_same_xyz(self):
        """Test RMSD is ~0 for identical geometries."""
        rd = self._make_validated_rotor(increment=120.0)
        rmsd = calculate_neighbor_geometry_rmsd(rd, [0.0, 0.0], [120.0, 0.0])
        # Same base_xyz for all points, so RMSD should be ~0
        self.assertIsNotNone(rmsd)
        self.assertAlmostEqual(rmsd, 0.0, places=4)

    def test_calculate_neighbor_geometry_rmsd_missing(self):
        """Test RMSD returns None for missing geometry."""
        rd = self._make_validated_rotor(increment=120.0)
        # Overwrite one point's xyz to None
        rd['directed_scan'][('0.00', '0.00')]['xyz'] = None
        rmsd = calculate_neighbor_geometry_rmsd(rd, [0.0, 0.0], [120.0, 0.0])
        self.assertIsNone(rmsd)

    # -- Edge classification --

    def test_classify_continuous_edge(self):
        """Test that a smooth edge is classified as continuous."""
        rd = self._make_validated_rotor(increment=120.0,
                                        energy_fn=lambda a, b: 0.1 * (a + b))
        result = classify_neighbor_edge_continuity(rd, [0.0, 0.0], [120.0, 0.0])
        self.assertTrue(result['continuous'])
        self.assertEqual(result['reasons'], [])

    def test_classify_discontinuous_edge_energy(self):
        """Test that a huge energy jump flags the edge as discontinuous."""
        rd = self._make_validated_rotor(increment=120.0)
        # Inject a massive energy at one point
        rd['directed_scan'][('120.00', '0.00')]['energy'] = 9999.0
        result = classify_neighbor_edge_continuity(rd, [0.0, 0.0], [120.0, 0.0],
                                                    energy_threshold=10.0)
        self.assertFalse(result['continuous'])
        self.assertTrue(any('energy_jump' in r for r in result['reasons']))

    # -- Periodic consistency --

    def test_check_periodic_edge_consistency_same_geometry(self):
        """Test periodic check with identical geometries is consistent."""
        rd = self._make_validated_rotor(increment=120.0)
        # -120 and 120 differ by 240 on the number line but are neighbors via wrap
        result = check_periodic_edge_consistency(rd, [-120.0, 0.0], [120.0, 0.0])
        self.assertTrue(result['consistent'])

    # -- Branch-jump detection --

    def test_detect_branch_jump_points_smooth(self):
        """Test no branch jumps on a genuinely smooth surface (constant energy)."""
        rd = self._make_validated_rotor(increment=120.0,
                                        energy_fn=lambda a, b: 5.0)
        flagged = detect_branch_jump_points(rd, increment=120.0, energy_threshold=50.0)
        self.assertEqual(flagged, [])

    def test_detect_branch_jump_points_spike(self):
        """Test that a spike point surrounded by smooth neighbors gets flagged."""
        rd = self._make_validated_rotor(increment=120.0,
                                        energy_fn=lambda a, b: 0.0)
        # Inject a huge spike at one point
        rd['directed_scan'][('0.00', '0.00')]['energy'] = 999.0
        flagged = detect_branch_jump_points(rd, increment=120.0,
                                            energy_threshold=10.0,
                                            min_suspicious_edges=2)
        flagged_keys = {(f'{p[0]:.2f}', f'{p[1]:.2f}') for p in flagged}
        self.assertIn(('0.00', '0.00'), flagged_keys)

    # -- Full validation orchestration --

    def test_run_adaptive_surface_validation_smooth(self):
        """Test validation on a smooth surface."""
        rd = self._make_validated_rotor(increment=120.0,
                                        energy_fn=lambda a, b: 0.01 * abs(a + b))
        val = run_adaptive_surface_validation(rd, increment=120.0)
        self.assertEqual(val['status'], 'complete')
        self.assertGreater(val['neighbor_edges_checked'], 0)
        self.assertEqual(len(val['discontinuous_edges']), 0)
        self.assertEqual(len(val['branch_jump_points']), 0)
        self.assertIn('Surface passed all continuity checks.', val['notes'])

    def test_run_adaptive_surface_validation_with_spike(self):
        """Test validation detects a discontinuity."""
        rd = self._make_validated_rotor(increment=120.0,
                                        energy_fn=lambda a, b: 0.0)
        rd['directed_scan'][('0.00', '0.00')]['energy'] = 999.0
        val = run_adaptive_surface_validation(rd, increment=120.0)
        self.assertEqual(val['status'], 'complete')
        self.assertGreater(len(val['discontinuous_edges']), 0)

    def test_run_adaptive_surface_validation_empty(self):
        """Test validation with no edges."""
        rd = self._make_validated_rotor(increment=120.0)
        # Clear all but one point
        keys = list(rd['directed_scan'].keys())
        for k in keys[1:]:
            del rd['directed_scan'][k]
        val = run_adaptive_surface_validation(rd, increment=120.0)
        self.assertEqual(val['status'], 'no_edges')

    # -- Serializability --

    def test_validation_state_yaml_safe(self):
        """Test that validation state is JSON/YAML-serializable."""
        import json
        rd = self._make_validated_rotor(increment=120.0)
        val = run_adaptive_surface_validation(rd, increment=120.0)
        json.dumps(val)  # should not raise

    # -- Integration --

    def test_update_adaptive_validation_state(self):
        """Test that update writes validation into rotor state."""
        rd = self._make_validated_rotor(increment=120.0)
        update_adaptive_validation_state(rd, increment=120.0)
        self.assertIn('validation', rd['adaptive_scan'])
        self.assertEqual(rd['adaptive_scan']['validation']['status'], 'complete')

    def test_update_skips_dense(self):
        """Test that update does nothing for dense scans."""
        rd = self._make_validated_rotor(increment=120.0)
        rd['sampling_policy'] = 'dense'
        update_adaptive_validation_state(rd, increment=120.0)
        self.assertNotIn('validation', rd['adaptive_scan'])

    def test_finalize_includes_validation_summary(self):
        """Test that finalize includes validation_summary for adaptive scans."""
        rd = self._make_validated_rotor(increment=120.0,
                                        energy_fn=lambda a, b: 0.0)
        results, _ = finalize_directed_scan_results(rd, increment=120.0)
        self.assertIn('validation_summary', results)
        self.assertEqual(results['validation_summary']['status'], 'complete')

    def test_finalize_no_validation_for_dense(self):
        """Test that finalize omits validation_summary for dense scans."""
        rd = self._make_validated_rotor(increment=120.0)
        rd['sampling_policy'] = 'dense'
        results, _ = finalize_directed_scan_results(rd, increment=120.0)
        self.assertNotIn('validation_summary', results)

    def test_edge_classification_output_shape(self):
        """Test that classify_neighbor_edge_continuity returns all expected keys."""
        rd = self._make_validated_rotor(increment=120.0)
        result = classify_neighbor_edge_continuity(rd, [0.0, 0.0], [120.0, 0.0])
        self.assertIn('continuous', result)
        self.assertIn('energy_jump', result)
        self.assertIn('geometry_rmsd', result)
        self.assertIn('reasons', result)
        self.assertIsInstance(result['reasons'], list)


class TestCouplingAndClassification(unittest.TestCase):
    """
    Tests for coupling metrics, surface quality, and ND rotor classification.
    Uses synthetic 2D surfaces with known separable/coupled structure.
    """

    def _make_rotor_with_surface(self, energy_fn, increment=120.0, n_failed=0, n_invalid=0):
        """
        Helper: create a fully populated adaptive 2D rotor dict with a given energy function.
        energy_fn(phi0_deg, phi1_deg) -> energy in kJ/mol (already normalized, min ~0).
        """
        rd = {
            'pivots': [[1, 2], [5, 6]],
            'top': [[0], [7]],
            'scan': [[1, 2, 3, 4], [5, 6, 7, 8]],
            'torsion': [[0, 1, 2, 3], [4, 5, 6, 7]],
            'number_of_running_jobs': 0,
            'success': None,
            'invalidation_reason': '',
            'times_dihedral_set': 0,
            'trsh_counter': 0,
            'trsh_methods': [],
            'scan_path': '',
            'directed_scan_type': 'brute_force_opt',
            'directed_scan': {},
            'dimensions': 2,
            'original_dihedrals': ['0.00', '0.00'],
            'cont_indices': [],
            'symmetry': None,
            'max_e': None,
            'sampling_policy': 'adaptive',
            'adaptive_scan': {
                'enabled': True,
                'phase': 'complete',
                'batch_size': 10,
                'candidate_points': [],
                'pending_points': [],
                'completed_points': [],
                'failed_points': [],
                'invalid_points': [],
                'seed_points': [],
                'selected_points_history': [],
                'stopping_reason': 'max_points_reached',
                'max_points': 200,
                'min_points': 20,
                'fit_metadata': {},
                'surface_model': {},
            },
        }
        n = int(360 / increment) + 1
        count = 0
        for i in range(n):
            for j in range(n):
                a0 = round((i * increment + 180.0) % 360.0 - 180.0, 2)
                a1 = round((j * increment + 180.0) % 360.0 - 180.0, 2)
                if count < n_failed:
                    rd['adaptive_scan']['failed_points'].append([a0, a1])
                    count += 1
                    continue
                if count < n_failed + n_invalid:
                    rd['adaptive_scan']['invalid_points'].append([a0, a1])
                    count += 1
                    continue
                key = (f'{a0:.2f}', f'{a1:.2f}')
                e = energy_fn(a0, a1)
                rd['directed_scan'][key] = {
                    'energy': e, 'xyz': {}, 'is_isomorphic': True, 'trsh': [],
                }
                rd['adaptive_scan']['completed_points'].append([a0, a1])
                count += 1
        return rd

    # --- Data extraction ---

    def test_extract_surface_arrays(self):
        """Test surface array extraction."""
        rd = self._make_rotor_with_surface(lambda a, b: abs(a) + abs(b), increment=120.0)
        data = extract_adaptive_2d_surface_arrays(rd)
        self.assertGreater(data['n_points'], 0)
        self.assertEqual(len(data['phi0']), data['n_points'])
        self.assertEqual(len(data['energy']), data['n_points'])

    # --- Separable fit ---

    def test_separable_fit_on_separable_surface(self):
        """A purely separable surface should have near-zero fit error."""
        rd = self._make_rotor_with_surface(
            lambda a, b: 10.0 * (1 - math.cos(math.radians(a))) + 5.0 * (1 - math.cos(math.radians(b))),
            increment=60.0)
        data = extract_adaptive_2d_surface_arrays(rd)
        sep_fit = fit_separable_surface_proxy(data)
        error = calculate_separable_fit_error(data, sep_fit)
        self.assertLess(error, 0.05, 'Separable surface should have small fit error')

    def test_separable_fit_on_coupled_surface(self):
        """A coupled surface should have larger fit error."""
        rd = self._make_rotor_with_surface(
            lambda a, b: 10.0 * math.cos(math.radians(a - b)),
            increment=60.0)
        data = extract_adaptive_2d_surface_arrays(rd)
        sep_fit = fit_separable_surface_proxy(data)
        error = calculate_separable_fit_error(data, sep_fit)
        self.assertGreater(error, 0.05, 'Coupled surface should have larger fit error')

    # --- Nonseparability score ---

    def test_nonseparability_separable(self):
        """Separable surface should have low nonseparability score."""
        rd = self._make_rotor_with_surface(
            lambda a, b: 10.0 * (1 - math.cos(math.radians(a))) + 5.0 * (1 - math.cos(math.radians(b))),
            increment=60.0)
        data = extract_adaptive_2d_surface_arrays(rd)
        sep_fit = fit_separable_surface_proxy(data)
        score = calculate_nonseparability_score(data, sep_fit)
        self.assertLess(score, COUPLING_NONSEP_THRESHOLD)

    def test_nonseparability_coupled(self):
        """Coupled surface should have high nonseparability score."""
        rd = self._make_rotor_with_surface(
            lambda a, b: 10.0 * math.cos(math.radians(a - b)),
            increment=60.0)
        data = extract_adaptive_2d_surface_arrays(rd)
        sep_fit = fit_separable_surface_proxy(data)
        score = calculate_nonseparability_score(data, sep_fit)
        self.assertGreater(score, COUPLING_NONSEP_THRESHOLD)

    # --- Cross-term strength ---

    def test_cross_term_separable(self):
        """Separable surface should have low cross-term strength."""
        rd = self._make_rotor_with_surface(
            lambda a, b: abs(a) / 180.0 * 10.0 + abs(b) / 180.0 * 5.0,
            increment=60.0)
        data = extract_adaptive_2d_surface_arrays(rd)
        sep_fit = fit_separable_surface_proxy(data)
        ct = calculate_cross_term_strength(data, sep_fit)
        self.assertLess(ct, 0.15)

    # --- Low-energy-path coupling ---

    def test_low_energy_path_separable(self):
        """For a separable surface, low-energy path should show low correlation."""
        rd = self._make_rotor_with_surface(
            lambda a, b: 10.0 * (1 - math.cos(math.radians(a))) + 5.0 * (1 - math.cos(math.radians(b))),
            increment=30.0)
        data = extract_adaptive_2d_surface_arrays(rd)
        coupling = calculate_low_energy_path_coupling(data)
        # For a truly separable surface the low-energy ridge is at phi0~0 regardless of phi1
        # so sin/cos correlation should be low
        self.assertLess(coupling, 0.7)

    # --- Complete coupling metrics ---

    def test_compute_coupling_metrics_separable(self):
        """Coupling metrics for a separable surface."""
        rd = self._make_rotor_with_surface(
            lambda a, b: 10.0 * (1 - math.cos(math.radians(a))) + 5.0 * (1 - math.cos(math.radians(b))),
            increment=60.0)
        metrics = compute_coupling_metrics(rd)
        self.assertEqual(metrics['status'], 'complete')
        self.assertLess(metrics['nonseparability_score'], COUPLING_NONSEP_THRESHOLD)

    def test_compute_coupling_metrics_insufficient(self):
        """Coupling metrics with too few points."""
        rd = self._make_rotor_with_surface(lambda a, b: 0.0, increment=120.0)
        # Only 16 points but some might be enough. Let's clear most.
        keys_to_remove = list(rd['directed_scan'].keys())[5:]
        for k in keys_to_remove:
            del rd['directed_scan'][k]
        rd['adaptive_scan']['completed_points'] = rd['adaptive_scan']['completed_points'][:5]
        metrics = compute_coupling_metrics(rd)
        self.assertEqual(metrics['status'], 'insufficient_data')

    # --- Surface quality ---

    def test_surface_quality_good(self):
        """Quality metrics for a clean scan."""
        rd = self._make_rotor_with_surface(lambda a, b: 0.0, increment=120.0)
        # Run validation first (needed for warning fraction)
        update_adaptive_validation_state(rd, increment=120.0)
        metrics = compute_surface_quality_metrics(rd, increment=120.0)
        self.assertEqual(metrics['status'], 'complete')
        self.assertGreater(metrics['quality_score'], 0.5)
        self.assertAlmostEqual(metrics['failed_fraction'], 0.0)

    def test_surface_quality_many_failures(self):
        """Quality metrics with many failed points."""
        rd = self._make_rotor_with_surface(lambda a, b: 0.0, increment=120.0, n_failed=6)
        metrics = compute_surface_quality_metrics(rd, increment=120.0)
        self.assertGreater(metrics['failed_fraction'], 0.1)

    def test_coverage_fraction(self):
        """Test coverage calculation."""
        rd = self._make_rotor_with_surface(lambda a, b: 0.0, increment=120.0)
        cov = calculate_coverage_fraction(rd, increment=120.0)
        # Grid formula gives (360/120+1)^2 = 16, but -180 and 180 share a key
        # after normalization, so 9 unique keys out of 16 grid points.
        self.assertGreater(cov, 0.5)
        self.assertLessEqual(cov, 1.0)

    # --- ND classification ---

    def test_classify_separable(self):
        """Clean separable surface should be classified as separable."""
        rd = self._make_rotor_with_surface(
            lambda a, b: 10.0 * (1 - math.cos(math.radians(a))) + 5.0 * (1 - math.cos(math.radians(b))),
            increment=60.0)
        update_adaptive_validation_state(rd, increment=60.0)
        update_nd_classification(rd, increment=60.0)
        cls = rd['adaptive_scan']['nd_classification']
        self.assertEqual(cls['classification'], 'separable')
        self.assertEqual(cls['recommended_action'], 'treat_as_separable_1d_like')

    def test_classify_coupled(self):
        """Clean coupled surface should be classified as coupled."""
        rd = self._make_rotor_with_surface(
            lambda a, b: 10.0 * math.cos(math.radians(a - b)),
            increment=60.0)
        update_adaptive_validation_state(rd, increment=60.0)
        update_nd_classification(rd, increment=60.0)
        cls = rd['adaptive_scan']['nd_classification']
        self.assertEqual(cls['classification'], 'coupled')
        self.assertEqual(cls['recommended_action'], 'retain_as_coupled_2d_surface')

    def test_classify_unreliable_many_failures(self):
        """Surface with many failures should be classified as unreliable."""
        rd = self._make_rotor_with_surface(lambda a, b: 0.0, increment=60.0, n_failed=15)
        update_adaptive_validation_state(rd, increment=60.0)
        update_nd_classification(rd, increment=60.0)
        cls = rd['adaptive_scan']['nd_classification']
        self.assertEqual(cls['classification'], 'unreliable')
        self.assertEqual(cls['recommended_action'], 'fallback_due_to_surface_quality')

    def test_classify_unreliable_insufficient_data(self):
        """Very few points should be classified as unreliable."""
        rd = self._make_rotor_with_surface(lambda a, b: 0.0, increment=120.0)
        # Remove most points
        keys = list(rd['directed_scan'].keys())[3:]
        for k in keys:
            del rd['directed_scan'][k]
        rd['adaptive_scan']['completed_points'] = rd['adaptive_scan']['completed_points'][:3]
        update_adaptive_validation_state(rd, increment=120.0)
        update_nd_classification(rd, increment=120.0)
        cls = rd['adaptive_scan']['nd_classification']
        self.assertEqual(cls['classification'], 'unreliable')

    # --- Dense unchanged ---

    def test_classify_skips_dense(self):
        """update_nd_classification should do nothing for dense scans."""
        rd = self._make_rotor_with_surface(lambda a, b: 0.0, increment=120.0)
        rd['sampling_policy'] = 'dense'
        update_nd_classification(rd, increment=120.0)
        self.assertNotIn('nd_classification', rd.get('adaptive_scan', {}))

    # --- Finalization integration ---

    def test_finalize_includes_classification_summary(self):
        """Finalization should include classification summary for adaptive scans."""
        rd = self._make_rotor_with_surface(
            lambda a, b: 10.0 * (1 - math.cos(math.radians(a))),
            increment=60.0)
        results, _ = finalize_directed_scan_results(rd, increment=60.0)
        self.assertIn('classification_summary', results)
        self.assertIn('coupling_summary', results)
        self.assertIn('surface_quality_summary', results)
        self.assertIsNotNone(results['classification_summary']['classification'])

    def test_finalize_no_classification_for_dense(self):
        """Finalization omits classification for dense scans."""
        rd = self._make_rotor_with_surface(lambda a, b: 0.0, increment=120.0)
        rd['sampling_policy'] = 'dense'
        results, _ = finalize_directed_scan_results(rd, increment=120.0)
        self.assertNotIn('classification_summary', results)
        self.assertNotIn('coupling_summary', results)

    # --- Serializability ---

    def test_classification_metadata_yaml_safe(self):
        """All new metadata should be JSON/YAML-serializable."""
        import json
        rd = self._make_rotor_with_surface(
            lambda a, b: 10.0 * math.cos(math.radians(a - b)),
            increment=60.0)
        update_adaptive_validation_state(rd, increment=60.0)
        update_nd_classification(rd, increment=60.0)
        state = rd['adaptive_scan']
        json.dumps(state.get('coupling_metrics', {}))
        json.dumps(state.get('surface_quality', {}))
        json.dumps(state.get('nd_classification', {}))


if __name__ == '__main__':
    unittest.main()
