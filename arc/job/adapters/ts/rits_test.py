#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for the RitS TS-guess adapter (``arc.job.adapters.ts.rits_ts``).

Tier-1 (always runs):
    * settings resolution and finder helpers
    * pure-Python helpers: ``write_xyz_file``, ``parse_multi_frame_xyz``,
      ``process_rits_tsg`` dedup
    * adapter instantiation with ``testing=True``, file-path layout
    * graceful skip when ``rits_env`` / checkpoint are missing
    * input.yml writer (mocked subprocess)

Tier-2 (gated on ``_rits_environment_ready()``):
    * end-to-end ``execute_incore`` against the real ``rits_env`` for a
      handful of family-diverse reactions sourced from
      ``arc/job/adapters/ts/linear_test.py``.

The Tier-2 tests are skipped automatically on CI runners that did not run
``install_rits.sh`` — the matching CI lane (``rits-install`` in
``.github/workflows/ci.yml``) installs the env and exercises them.
"""

import importlib
import math
import os
import shutil
import sys
import unittest
from collections import Counter
from unittest import mock

import arc.job.adapters.ts.rits_ts as rits_mod
from arc.common import ARC_TESTING_PATH, read_yaml_file
from arc.job.adapters.ts.rits_ts import (RitSAdapter,
                                         _rits_environment_ready,
                                         process_rits_tsg,
                                         write_xyz_file,
                                         )
from arc.reaction import ARCReaction
from arc.species.converter import str_to_xyz, compare_confs
from arc.species.species import ARCSpecies, TSGuess

HAS_RITS = _rits_environment_ready()


def _build_rxn_isomerization_propyl():
    """nC3H7 → iC3H7. The simplest isomerization in ARC's test suite."""
    return ARCReaction(r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
                       p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')])


def _build_rxn_diels_alder():
    """C=CC(=C)C + C=CC=O → CC1=CCC(C=O)CC1 — bimolecular Diels-Alder."""
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
    r1 = ARCSpecies(label='R1', smiles='C=CC(=C)C', xyz=r1_xyz)
    r2 = ARCSpecies(label='R2', smiles='C=CC=O', xyz=r2_xyz)
    p = ARCSpecies(label='P', smiles='CC1=CCC(C=O)CC1', xyz=p_xyz)
    return ARCReaction(r_species=[r1, r2], p_species=[p])


def _build_rxn_one_plus_two_cycloaddition():
    """Singlet CH2 + C=C=C → C=C1CC1 — bimolecular addition with carbene."""
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
    r1 = ARCSpecies(label='CH2_singlet', adjlist="""multiplicity 1
1 C u0 p1 c0 {2,S} {3,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
""", xyz=ch2_xyz)
    r2 = ARCSpecies(label='allene', smiles='C=C=C', xyz=c3h4_xyz)
    p = ARCSpecies(label='methylene_cyclopropane', smiles='C=C1CC1', xyz=c4h6_xyz)
    return ARCReaction(r_species=[r1, r2], p_species=[p])


def _build_rxn_nh3_elimination():
    """NNN → H2NN(s) + NH3 — 1 reactant → 2 products elimination."""
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
    r = ARCSpecies(label='triazene', smiles='NNN', xyz=n3_xyz)
    p1 = ARCSpecies(label='H2NNs', adjlist="""multiplicity 1
1 N u0 p0 c+1 {2,S} {3,S} {4,D}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 N u0 p2 c-1 {1,D}
""", xyz=h2nn_xyz)
    p2 = ARCSpecies(label='NH3', smiles='N', xyz=nh3_xyz)
    return ARCReaction(r_species=[r], p_species=[p1, p2])


# === Group A: 1<->1 isomerizations =========================================

def _build_rxn_vinyl_alcohol_to_acetaldehyde():
    """Keto-enol tautomerization C2H4O: C=CO -> CC=O (6 atoms, 1,3-H shift)."""
    r = ARCSpecies(label='vinyl_alcohol', smiles='C=CO')
    p = ARCSpecies(label='acetaldehyde', smiles='CC=O')
    return ARCReaction(r_species=[r], p_species=[p])


def _build_rxn_propenol_to_acetone():
    """Keto-enol tautomerization C3H6O: OC(=C)C -> CC(=O)C (10 atoms)."""
    r = ARCSpecies(label='propen_2_ol', smiles='OC(=C)C')
    p = ARCSpecies(label='acetone', smiles='CC(=O)C')
    return ARCReaction(r_species=[r], p_species=[p])


def _build_rxn_cyclobutene_to_butadiene():
    """Electrocyclic ring opening C4H6: C1=CCC1 -> C=CC=C (10 atoms)."""
    r = ARCSpecies(label='cyclobutene', smiles='C1=CCC1')
    p = ARCSpecies(label='1_3_butadiene', smiles='C=CC=C')
    return ARCReaction(r_species=[r], p_species=[p])


def _build_rxn_methoxy_to_hydroxymethyl():
    """1,2-H migration in CH3O radical: [O]C -> O[CH2] (5 atoms)."""
    r = ARCSpecies(label='methoxy', smiles='[O]C')
    p = ARCSpecies(label='hydroxymethyl', smiles='O[CH2]')
    return ARCReaction(r_species=[r], p_species=[p])


def _build_rxn_ethoxy_to_alpha_hydroxyethyl():
    """1,2-H migration in CH3CH2O radical: CC[O] -> [CH2]CO (8 atoms)."""
    r = ARCSpecies(label='ethoxy', smiles='CC[O]')
    p = ARCSpecies(label='alpha_hydroxyethyl', smiles='[CH2]CO')
    return ARCReaction(r_species=[r], p_species=[p])


def _build_rxn_cyclopropane_to_propene():
    """Ring opening C3H6: C1CC1 -> C=CC (9 atoms)."""
    r = ARCSpecies(label='cyclopropane', smiles='C1CC1')
    p = ARCSpecies(label='propene', smiles='C=CC')
    return ARCReaction(r_species=[r], p_species=[p])


# === Group B: 1<->2 / 2<->1 (eliminations / cycloadditions) ================

def _build_rxn_cyclobutane_retro_22():
    """Retro [2+2] C4H8 -> 2 C2H4 (cyclobutane -> 2 ethene), 12 atoms."""
    r = ARCSpecies(label='cyclobutane', smiles='C1CCC1')
    p1 = ARCSpecies(label='ethene_a', smiles='C=C')
    p2 = ARCSpecies(label='ethene_b', smiles='C=C')
    return ARCReaction(r_species=[r], p_species=[p1, p2])


def _build_rxn_da_butadiene_ethene():
    """Small Diels-Alder C4H6 + C2H4 -> cyclohexene C6H10 (16 atoms)."""
    r1 = ARCSpecies(label='1_3_butadiene', smiles='C=CC=C')
    r2 = ARCSpecies(label='ethene', smiles='C=C')
    p = ARCSpecies(label='cyclohexene', smiles='C1=CCCCC1')
    return ARCReaction(r_species=[r1, r2], p_species=[p])


def _build_rxn_ethanol_dehydration():
    """β-elimination CCO -> C=C + H2O (9 atoms)."""
    r = ARCSpecies(label='ethanol', smiles='CCO')
    p1 = ARCSpecies(label='ethene', smiles='C=C')
    p2 = ARCSpecies(label='water', smiles='O')
    return ARCReaction(r_species=[r], p_species=[p1, p2])


def _build_rxn_methylamine_dehydrogenation():
    """1,2-dehydrogenation CN -> C=N + H2 (7 atoms total)."""
    r = ARCSpecies(label='methylamine', smiles='CN')
    p1 = ARCSpecies(label='methyleneamine', smiles='C=N')
    p2 = ARCSpecies(label='dihydrogen', smiles='[H][H]')
    return ARCReaction(r_species=[r], p_species=[p1, p2])


def _build_rxn_ethyl_peroxy_ho2_elimination():
    """β-scission CCO[O] -> C=C + O[O] (9 atoms)."""
    r = ARCSpecies(label='ethyl_peroxy', smiles='CCO[O]')
    p1 = ARCSpecies(label='ethene', smiles='C=C')
    p2 = ARCSpecies(label='hydroperoxyl', smiles='O[O]')
    return ARCReaction(r_species=[r], p_species=[p1, p2])


# === Group C: 2<->2 H-abstractions =========================================

def _build_rxn_hab_ch4_oh():
    """H-abstraction CH4 + OH -> CH3 + H2O (6 atoms each side)."""
    r1 = ARCSpecies(label='methane', smiles='C')
    r2 = ARCSpecies(label='hydroxyl', smiles='[OH]')
    p1 = ARCSpecies(label='methyl', smiles='[CH3]')
    p2 = ARCSpecies(label='water', smiles='O')
    return ARCReaction(r_species=[r1, r2], p_species=[p1, p2])


def _build_rxn_hab_c2h6_h():
    """H-abstraction C2H6 + H -> C2H5 + H2 (9 atoms)."""
    r1 = ARCSpecies(label='ethane', smiles='CC')
    r2 = ARCSpecies(label='H_atom', smiles='[H]')
    p1 = ARCSpecies(label='ethyl', smiles='C[CH2]')
    p2 = ARCSpecies(label='dihydrogen', smiles='[H][H]')
    return ARCReaction(r_species=[r1, r2], p_species=[p1, p2])


def _build_rxn_hab_nh3_oh():
    """H-abstraction NH3 + OH -> NH2 + H2O (6 atoms)."""
    r1 = ARCSpecies(label='ammonia', smiles='N')
    r2 = ARCSpecies(label='hydroxyl', smiles='[OH]')
    p1 = ARCSpecies(label='amidogen', smiles='[NH2]')
    p2 = ARCSpecies(label='water', smiles='O')
    return ARCReaction(r_species=[r1, r2], p_species=[p1, p2])


def _build_rxn_hab_ch3oh_h():
    """H-abstraction CH3OH + H -> CH2OH + H2 (7 atoms; abstracts α-CH)."""
    r1 = ARCSpecies(label='methanol', smiles='CO')
    r2 = ARCSpecies(label='H_atom', smiles='[H]')
    p1 = ARCSpecies(label='hydroxymethyl', smiles='[CH2]O')
    p2 = ARCSpecies(label='dihydrogen', smiles='[H][H]')
    return ARCReaction(r_species=[r1, r2], p_species=[p1, p2])


# ---------------------------------------------------------------------------
# Pure-python helpers + plumbing
# ---------------------------------------------------------------------------

class TestRitSHelpers(unittest.TestCase):
    """Helper-function unit tests that don't need rits_env."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = os.path.join(ARC_TESTING_PATH, 'rits_helpers')
        os.makedirs(cls.tmp_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_write_xyz_file_round_trip(self):
        """write_xyz_file should produce a parseable XYZ file with correct atom count."""
        xyz_dict = {
            'symbols': ('C', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 1, 1, 1, 1),
            'coords': (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
            ),
        }
        path = os.path.join(self.tmp_dir, 'methane.xyz')
        write_xyz_file(xyz_dict, path, comment='methane test')
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            lines = f.read().splitlines()
        # Header
        self.assertEqual(int(lines[0]), 5)
        self.assertEqual(lines[1], 'methane test')
        # Body — 5 coordinate lines starting with the right symbols
        body_symbols = [ln.split()[0] for ln in lines[2:7]]
        self.assertEqual(body_symbols, ['C', 'H', 'H', 'H', 'H'])
        # Round-trip via str_to_xyz
        rt = str_to_xyz(path)
        self.assertEqual(rt['symbols'], xyz_dict['symbols'])

    def test_write_xyz_file_strips_newlines_in_comment(self):
        """A multi-line comment must not corrupt the XYZ format."""
        xyz_dict = {
            'symbols': ('H', 'H'),
            'isotopes': (1, 1),
            'coords': ((0.0, 0.0, 0.0), (0.74, 0.0, 0.0)),
        }
        path = os.path.join(self.tmp_dir, 'h2.xyz')
        write_xyz_file(xyz_dict, path, comment='line1\nline2\nline3')
        with open(path) as f:
            lines = f.read().splitlines()
        # Header is exactly 2 lines + 2 atoms = 4 lines minimum
        self.assertEqual(int(lines[0]), 2)
        self.assertNotIn('\n', lines[1])
        self.assertEqual(len(lines), 4)

    def test_process_rits_tsg_failed_entry(self):
        """A failed-sentinel dict should not produce a TSGuess."""
        ts_species = ARCSpecies(label='TS', is_ts=True)
        added = process_rits_tsg(
            tsg_dict={'method': 'RitS', 'method_direction': 'F', 'method_index': 0,
                      'initial_xyz': None, 'success': False, 'execution_time': '0:00:00.0'},
            local_path=self.tmp_dir,
            ts_species=ts_species,
        )
        self.assertFalse(added)
        self.assertEqual(len(ts_species.ts_guesses), 0)

    def test_process_rits_tsg_dedup_against_existing(self):
        """A RitS guess that matches an existing GCN guess should not be appended;
        the existing guess should be re-labeled to credit RitS as well."""
        ts_species = ARCSpecies(label='TS', is_ts=True)
        # Plant a GCN guess first.
        existing_xyz_str = """C    0.0  0.0  0.0
H    1.0  0.0  0.0
H   -1.0  0.0  0.0
H    0.0  1.0  0.0
H    0.0 -1.0  0.0"""
        existing = TSGuess(method='GCN', method_direction='F', method_index=0,
                           index=0, success=True)
        existing.process_xyz(str_to_xyz(existing_xyz_str))
        ts_species.ts_guesses.append(existing)
        # Submit a RitS guess with identical coordinates.
        added = process_rits_tsg(
            tsg_dict={'method': 'RitS', 'method_direction': 'F', 'method_index': 0,
                      'initial_xyz': existing_xyz_str, 'success': True,
                      'execution_time': '0:00:01.0'},
            local_path=self.tmp_dir,
            ts_species=ts_species,
        )
        self.assertFalse(added)  # not appended
        self.assertEqual(len(ts_species.ts_guesses), 1)
        # The existing guess should now credit both methods. Note: TSGuess
        # lowercases the method string on construction.
        merged = ts_species.ts_guesses[0].method.lower()
        self.assertIn('rits', merged)
        self.assertIn('gcn', merged)

    def test_process_rits_tsg_unique_guess_appended(self):
        """A unique non-colliding guess should be appended."""
        ts_species = ARCSpecies(label='TS', is_ts=True)
        unique_xyz = """C    0.0  0.0  0.0
H    1.5  0.0  0.0
H   -1.5  0.0  0.0
H    0.0  1.5  0.0
H    0.0 -1.5  0.0"""
        added = process_rits_tsg(
            tsg_dict={'method': 'RitS', 'method_direction': 'F', 'method_index': 2,
                      'initial_xyz': unique_xyz, 'success': True,
                      'execution_time': '0:00:02.0'},
            local_path=self.tmp_dir,
            ts_species=ts_species,
        )
        self.assertTrue(added)
        self.assertEqual(len(ts_species.ts_guesses), 1)
        # TSGuess lowercases method on construction.
        self.assertEqual(ts_species.ts_guesses[0].method.lower(), 'rits')
        self.assertEqual(ts_species.ts_guesses[0].method_index, 2)
        self.assertTrue(ts_species.ts_guesses[0].success)

    def test_process_rits_tsg_collision_rejected(self):
        """A guess where two atoms overlap must be rejected by colliding_atoms."""
        ts_species = ARCSpecies(label='TS', is_ts=True)
        bad_xyz = """C    0.0  0.0  0.0
H    0.0  0.0  0.0
H   -1.5  0.0  0.0
H    0.0  1.5  0.0
H    0.0 -1.5  0.0"""
        added = process_rits_tsg(
            tsg_dict={'method': 'RitS', 'method_direction': 'F', 'method_index': 0,
                      'initial_xyz': bad_xyz, 'success': True,
                      'execution_time': '0:00:00.5'},
            local_path=self.tmp_dir,
            ts_species=ts_species,
        )
        self.assertFalse(added)
        self.assertEqual(len(ts_species.ts_guesses), 0)

    def test_process_rits_tsg_dedup_catches_rigid_rotation(self):
        """A rigidly rotated + translated copy of an existing TSGuess must be
        deduped. This is the whole point of switching from byte-level
        almost_equal_coords to distance-matrix compare_confs — RitS samples
        each TS in its own random orientation, so rotated copies are common.
        """
        ts_species = ARCSpecies(label='TS', is_ts=True)
        # Plant the original (use atypical CH bond lengths so we can be sure
        # the assertion isn't accidentally matching some default geometry).
        original_xyz = """C    0.000   0.000   0.000
H    0.700   0.700   0.700
H   -0.700  -0.700   0.700
H   -0.700   0.700  -0.700
H    0.700  -0.700  -0.700"""
        first = process_rits_tsg(
            tsg_dict={'method': 'RitS', 'method_direction': 'F', 'method_index': 0,
                      'initial_xyz': original_xyz, 'success': True,
                      'execution_time': '0:00:00.0'},
            local_path=self.tmp_dir,
            ts_species=ts_species,
        )
        self.assertTrue(first)
        self.assertEqual(len(ts_species.ts_guesses), 1)

        # Build a 37° z-axis rotation + translation of the same molecule.
        theta = math.radians(37.0)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        original_coords = [
            (0.000, 0.000, 0.000),
            (0.700, 0.700, 0.700),
            (-0.700, -0.700, 0.700),
            (-0.700, 0.700, -0.700),
            (0.700, -0.700, -0.700),
        ]
        rotated = []
        for x, y, z in original_coords:
            rx = cos_t * x - sin_t * y + 10.0   # also translate by (+10, +5, -3)
            ry = sin_t * x + cos_t * y + 5.0
            rz = z - 3.0
            rotated.append((rx, ry, rz))
        symbols = ('C', 'H', 'H', 'H', 'H')
        rotated_xyz_str = '\n'.join(
            f'{s}  {x:.6f}  {y:.6f}  {z:.6f}' for s, (x, y, z) in zip(symbols, rotated)
        )

        added = process_rits_tsg(
            tsg_dict={'method': 'RitS', 'method_direction': 'F', 'method_index': 1,
                      'initial_xyz': rotated_xyz_str, 'success': True,
                      'execution_time': '0:00:00.5'},
            local_path=self.tmp_dir,
            ts_species=ts_species,
        )
        self.assertFalse(added,
                         'rotated+translated duplicate of an existing RitS guess '
                         'must be deduped via compare_confs (distance-matrix RMSD)')
        self.assertEqual(len(ts_species.ts_guesses), 1,
                         'no new TSGuess should be appended for a rotated duplicate')


class TestRitSScriptParser(unittest.TestCase):
    """Direct unit tests for arc/job/adapters/scripts/rits_script.py:parse_multi_frame_xyz."""

    @classmethod
    def setUpClass(cls):
        # Import the standalone script as a module so we can call its helpers directly.
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(rits_mod.__file__))), 'scripts')
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        cls.rits_script = importlib.import_module('rits_script')
        cls.tmp_dir = os.path.join(ARC_TESTING_PATH, 'rits_script_parser')
        os.makedirs(cls.tmp_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _write(self, name: str, body: str) -> str:
        path = os.path.join(self.tmp_dir, name)
        with open(path, 'w') as f:
            f.write(body)
        return path

    def test_single_frame_xyz(self):
        body = "3\n\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH -1.0 0.0 0.0\n"
        frames = self.rits_script.parse_multi_frame_xyz(self._write('one.xyz', body))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].splitlines()[0].split()[0], 'C')

    def test_multi_frame_xyz(self):
        body = ("3\n\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH -1.0 0.0 0.0\n"
                "3\n\nC 0.1 0.0 0.0\nH 1.1 0.0 0.0\nH -0.9 0.0 0.0\n")
        frames = self.rits_script.parse_multi_frame_xyz(self._write('two.xyz', body))
        self.assertEqual(len(frames), 2)
        # Frame 0 starts at the origin; frame 1 is shifted by +0.1 in x
        self.assertAlmostEqual(float(frames[0].splitlines()[0].split()[1]), 0.0)
        self.assertAlmostEqual(float(frames[1].splitlines()[0].split()[1]), 0.1)

    def test_missing_file_returns_empty_list(self):
        frames = self.rits_script.parse_multi_frame_xyz(os.path.join(self.tmp_dir, 'nope.xyz'))
        self.assertEqual(frames, list())

    def test_garbage_does_not_loop_forever(self):
        body = "this is not an xyz\nat all\n"
        frames = self.rits_script.parse_multi_frame_xyz(self._write('garbage.xyz', body))
        self.assertEqual(frames, list())


class TestRitSAdapterInstantiation(unittest.TestCase):
    """Verify the adapter constructs and lays out files even without rits_env."""

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.output_dir = os.path.join(ARC_TESTING_PATH, 'RitS', 'instantiation')
        os.makedirs(cls.output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(ARC_TESTING_PATH, 'RitS'), ignore_errors=True)

    def _build_adapter(self, project_dir: str, n_samples: int = 5):
        rxn = _build_rxn_isomerization_propyl()
        return RitSAdapter(
            job_type='tsg',
            reactions=[rxn],
            testing=True,
            project='test_rits',
            project_directory=project_dir,
            args={'keyword': {'n_samples': n_samples}},
        )

    def test_instantiation_sets_paths_and_metadata(self):
        proj = os.path.join(self.output_dir, 'paths')
        adapter = self._build_adapter(proj, n_samples=7)
        self.assertEqual(adapter.job_adapter, 'rits')
        self.assertEqual(adapter.execution_type, 'incore')
        self.assertEqual(adapter.url, 'https://github.com/isayevlab/RitS')
        self.assertEqual(adapter.incore_capacity, 1)
        self.assertEqual(adapter.n_samples, 7)
        # File paths should all live under the local_path the adapter set up
        self.assertTrue(adapter.reactant_xyz_path.endswith('reactant.xyz'))
        self.assertTrue(adapter.product_xyz_path.endswith('product.xyz'))
        self.assertTrue(adapter.ts_out_xyz_path.endswith('rits_ts.xyz'))
        self.assertTrue(adapter.yml_in_path.endswith('input.yml'))
        self.assertTrue(adapter.yml_out_path.endswith('output.yml'))
        # All five paths should share a parent directory
        parents = {os.path.dirname(p) for p in (adapter.reactant_xyz_path,
                                                adapter.product_xyz_path,
                                                adapter.ts_out_xyz_path,
                                                adapter.yml_in_path,
                                                adapter.yml_out_path)}
        self.assertEqual(len(parents), 1)

    def test_default_n_samples(self):
        proj = os.path.join(self.output_dir, 'default_samples')
        adapter = RitSAdapter(
            job_type='tsg',
            reactions=[_build_rxn_isomerization_propyl()],
            testing=True,
            project='test_rits',
            project_directory=proj,
        )
        self.assertEqual(adapter.n_samples, rits_mod.DEFAULT_N_SAMPLES)

    def test_n_samples_invalid_args_falls_back_to_default(self):
        proj = os.path.join(self.output_dir, 'bad_samples')
        adapter = RitSAdapter(
            job_type='tsg',
            reactions=[_build_rxn_isomerization_propyl()],
            testing=True,
            project='test_rits',
            project_directory=proj,
            args={'keyword': {'n_samples': 'not-a-number'}},
        )
        self.assertEqual(adapter.n_samples, rits_mod.DEFAULT_N_SAMPLES)

    def test_missing_reactions_raises(self):
        proj = os.path.join(self.output_dir, 'no_reactions')
        with self.assertRaises(ValueError):
            RitSAdapter(job_type='tsg', reactions=None, testing=True,
                        project='test_rits', project_directory=proj)


class TestRitSGracefulSkip(unittest.TestCase):
    """When rits_env / checkpoint are missing, execute_incore must NOT raise."""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = os.path.join(ARC_TESTING_PATH, 'RitS', 'graceful_skip')
        os.makedirs(cls.output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(ARC_TESTING_PATH, 'RitS'), ignore_errors=True)

    def test_missing_python_logs_and_returns(self):
        rxn = _build_rxn_isomerization_propyl()
        adapter = RitSAdapter(
            job_type='tsg',
            reactions=[rxn],
            testing=True,
            project='test_rits',
            project_directory=os.path.join(self.output_dir, 'no_python'),
        )
        # Patch the module-level constants to simulate a host without rits_env.
        with mock.patch.object(rits_mod, 'RITS_PYTHON', None), \
             mock.patch.object(rits_mod, 'RITS_REPO_PATH', '/nonexistent/RitS'), \
             mock.patch.object(rits_mod, 'RITS_CKPT_PATH', '/nonexistent/rits.ckpt'):
            # Should not raise
            adapter.execute_incore()
        # No TS guesses should have been created
        if rxn.ts_species is not None:
            self.assertEqual(len(rxn.ts_species.ts_guesses), 0)

    def test_missing_checkpoint_logs_and_returns(self):
        rxn = _build_rxn_isomerization_propyl()
        adapter = RitSAdapter(
            job_type='tsg',
            reactions=[rxn],
            testing=True,
            project='test_rits',
            project_directory=os.path.join(self.output_dir, 'no_ckpt'),
        )
        with mock.patch.object(rits_mod, 'RITS_CKPT_PATH', '/nonexistent/ckpt'):
            adapter.execute_incore()
        if rxn.ts_species is not None:
            self.assertEqual(len(rxn.ts_species.ts_guesses), 0)


class TestRitSInputYamlWritten(unittest.TestCase):
    """Verify input.yml is written correctly without invoking the real subprocess."""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = os.path.join(ARC_TESTING_PATH, 'RitS', 'input_yml')
        os.makedirs(cls.output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(ARC_TESTING_PATH, 'RitS'), ignore_errors=True)

    def test_input_yml_contents(self):
        """A successful execute_incore should write input.yml with all required keys.

        We mock subprocess.run so the test does not depend on rits_env actually
        being installed."""
        rxn = _build_rxn_diels_alder()
        adapter = RitSAdapter(
            job_type='tsg',
            reactions=[rxn],
            testing=True,
            project='test_rits',
            project_directory=os.path.join(self.output_dir, 'da'),
            args={'keyword': {'n_samples': 4}},
        )

        # Pretend the env is fully ready, but make subprocess.run a no-op so we
        # never actually invoke RitS — we only care about input.yml + the
        # mapped reactant.xyz / product.xyz files we wrote.
        fake_completed = mock.Mock(returncode=0)
        with mock.patch.object(rits_mod, '_rits_environment_ready', return_value=True), \
             mock.patch.object(rits_mod, 'RITS_PYTHON', '/fake/python'), \
             mock.patch.object(rits_mod, 'RITS_REPO_PATH', '/fake/RitS'), \
             mock.patch.object(rits_mod, 'RITS_CKPT_PATH', '/fake/rits.ckpt'), \
             mock.patch('arc.job.adapters.ts.rits_ts.subprocess.run',
                        return_value=fake_completed) as run_mock:
            adapter.execute_incore()

        self.assertTrue(run_mock.called)
        # input.yml should exist with the keys our standalone script expects
        self.assertTrue(os.path.isfile(adapter.yml_in_path))
        in_dict = read_yaml_file(adapter.yml_in_path)
        for key in ('reactant_xyz_path', 'product_xyz_path', 'rits_repo_path',
                    'ckpt_path', 'output_xyz_path', 'yml_out_path',
                    'config_path', 'n_samples', 'batch_size', 'charge', 'device'):
            self.assertIn(key, in_dict, f'missing key {key} in input.yml')
        self.assertEqual(in_dict['n_samples'], 4)
        self.assertEqual(in_dict['device'], 'auto')
        self.assertEqual(in_dict['rits_repo_path'], '/fake/RitS')
        self.assertEqual(in_dict['ckpt_path'], '/fake/rits.ckpt')
        self.assertTrue(in_dict['config_path'].endswith('rits.yaml'))
        # The reactant + product XYZ files should be on disk and have matching atom counts
        self.assertTrue(os.path.isfile(adapter.reactant_xyz_path))
        self.assertTrue(os.path.isfile(adapter.product_xyz_path))
        with open(adapter.reactant_xyz_path) as f:
            r_n = int(f.readline())
        with open(adapter.product_xyz_path) as f:
            p_n = int(f.readline())
        self.assertEqual(r_n, p_n)
        # Diels-Alder C=CC(=C)C + C=CC=O → CC1=CCC(C=O)CC1 has 21 atoms
        self.assertEqual(r_n, 21)


# ---------------------------------------------------------------------------
# End-to-end runs against the real rits_env (skipped without it)
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_RITS, 'rits_env / checkpoint not installed; run `make install-rits` to enable.')
class TestRitSEndToEnd(unittest.TestCase):
    """End-to-end runs through subprocess into the real rits_env.

    These tests are gated on `_rits_environment_ready()` so a CI runner that
    skipped install_rits.sh still gets a green run. The matching CI lane
    `rits-install` in .github/workflows/ci.yml installs the env and exercises
    them on every PR.

    Each test asks for a small number of samples (n_samples=2) so the runtime
    stays reasonable: even on CPU, two samples per reaction completes in well
    under a minute on the model RitS ships.
    """

    @classmethod
    def setUpClass(cls):
        cls.output_dir = os.path.join(ARC_TESTING_PATH, 'RitS', 'e2e')
        os.makedirs(cls.output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(ARC_TESTING_PATH, 'RitS'), ignore_errors=True)

    def _run_e2e(self, rxn, label: str, expected_n_atoms: int, n_samples: int = 2,
                 expect_success: bool = True):
        """Helper: build adapter, execute, return the (rxn, adapter) pair after assertions.

        Args:
            rxn: The ARCReaction to feed to RitS.
            label: Subdirectory name under the test output dir.
            expected_n_atoms: Atom count both reactant and product XYZs should match.
            n_samples: Number of TS samples to ask RitS for.
            expect_success: When True, assert at least one usable TSGuess was produced.
                When False, assert only that the adapter handled RitS's failure
                gracefully (output.yml exists, failed-sentinel entry inside, no
                crash). Used for reactions RitS cannot handle by design — e.g.
                charged/zwitterionic species, where its OpenBabel bond inference
                trips RDKit sanitization.
        """
        proj = os.path.join(self.output_dir, label)
        adapter = RitSAdapter(
            job_type='tsg',
            reactions=[rxn],
            testing=True,
            project='test_rits',
            project_directory=proj,
            args={'keyword': {'n_samples': n_samples}},
        )
        adapter.execute_incore()

        # The reactant + product XYZ that ARC fed to RitS must have matching atom counts
        with open(adapter.reactant_xyz_path) as f:
            r_n = int(f.readline())
        with open(adapter.product_xyz_path) as f:
            p_n = int(f.readline())
        self.assertEqual(r_n, expected_n_atoms)
        self.assertEqual(p_n, expected_n_atoms)
        # The reactant and product elements must match as multisets — atoms are
        # neither created nor destroyed across an elementary reaction.
        r_xyz_dict = str_to_xyz(adapter.reactant_xyz_path)
        p_xyz_dict = str_to_xyz(adapter.product_xyz_path)
        expected_formula = Counter(r_xyz_dict['symbols'])
        self.assertEqual(expected_formula, Counter(p_xyz_dict['symbols']),
                         f'reactant and product element multisets disagree for {label}')

        # The output YAML should exist and be readable in either case
        self.assertTrue(os.path.isfile(adapter.yml_out_path),
                        f'rits_script.py did not write {adapter.yml_out_path}')
        out = read_yaml_file(adapter.yml_out_path) or list()
        self.assertGreater(len(out), 0, f'rits_script.py produced 0 entries for {label}')
        successes = [tsg for tsg in out if tsg.get('success') and tsg.get('initial_xyz')]

        if expect_success:
            self.assertGreater(len(successes), 0,
                               f'RitS produced 0 successful TSGuess entries for {label}')
            # Strict check: EVERY successful TS must have the same atom count
            # AND the same element multiset as the reactants. Catches both
            # atom-count mismatches and element-shuffling bugs.
            for i, tsg_dict in enumerate(successes):
                ts_xyz = str_to_xyz(tsg_dict['initial_xyz'])
                self.assertEqual(
                    len(ts_xyz['symbols']), expected_n_atoms,
                    f'{label} TS sample {i}: atom count {len(ts_xyz["symbols"])} '
                    f'!= expected {expected_n_atoms}',
                )
                actual_formula = Counter(ts_xyz['symbols'])
                self.assertEqual(
                    actual_formula, expected_formula,
                    f'{label} TS sample {i}: molecular formula '
                    f'{dict(actual_formula)} does not match reactant '
                    f'{dict(expected_formula)}',
                )
        else:
            # Failure path: there should be exactly one failed sentinel entry,
            # and the adapter must not have created any successful TS guesses
            # on the reaction object.
            self.assertEqual(len(successes), 0,
                             f'Expected RitS to fail on {label}, but got '
                             f'{len(successes)} successful guess(es)')
            self.assertTrue(all(not tsg.get('success') for tsg in out))
        return adapter

    def test_e2e_isomerization_propyl(self):
        """nC3H7 → iC3H7 (10 atoms, isomerization).

        With ``n_samples=2`` RitS produces two TS guesses. We assert that
        BOTH survive the distance-matrix dedup — verified empirically:
        the two samples differ along the reaction coordinate (C-C of the
        donor side: 1.52 Å vs 1.76 Å; migrating-H acceptor distance:
        1.28 Å vs 1.16 Å), with a distance-matrix RMSD of ~0.56 Å — well
        above the 0.1 Å dedup threshold. They represent two diverse
        starting points for downstream Gaussian/ORCA TS optimization,
        which is exactly the value of asking for ``n_samples > 1``.
        Rotated/translated *exact* copies would be merged — see
        TestRitSHelpers.test_process_rits_tsg_dedup_catches_rigid_rotation.
        """
        adapter = self._run_e2e(_build_rxn_isomerization_propyl(),
                                label='isom_propyl', expected_n_atoms=10)
        rxn = adapter.reactions[0]
        successful = [tsg for tsg in rxn.ts_species.ts_guesses if tsg.success]
        # Both samples should survive — they are structurally distinct.
        self.assertEqual(
            len(successful), 2,
            f'Expected 2 unique TS guesses for nC3H7→iC3H7 (each from a '
            f'separate point on the reaction coordinate), got {len(successful)}.',
        )
        # Sanity-check they ARE distinct under compare_confs (else dedup is broken).
        self.assertFalse(
            compare_confs(successful[0].initial_xyz, successful[1].initial_xyz),
            'The two propyl TS guesses unexpectedly compare equal — RitS may have '
            'collapsed onto a single saddle, or the dedup is mis-tuned.',
        )

    def test_e2e_diels_alder(self):
        """Diels-Alder bimolecular addition (21 atoms)."""
        self._run_e2e(_build_rxn_diels_alder(),
                      label='diels_alder', expected_n_atoms=21)

    def test_e2e_one_plus_two_cycloaddition(self):  # fails
        """1+2 cycloaddition with singlet carbene (10 atoms, bimolecular)."""
        self._run_e2e(_build_rxn_one_plus_two_cycloaddition(),
                      label='one_plus_two', expected_n_atoms=10)

    def test_e2e_nh3_elimination_graceful_failure(self):  # fails (as planned)
        """1,2-NH3 elimination NNN → H2NN(s) + NH3 — RitS cannot handle this
        because its OpenBabel bond inference rejects the zwitterionic
        aminonitrene product (4-valent N+). The adapter must:

        * still write input.yml + reactant.xyz + product.xyz
        * still get a non-empty output.yml back
        * write a failed-sentinel TSGuess entry
        * NOT raise

        This test pins the graceful-failure code path so it doesn't regress.
        """
        adapter = self._run_e2e(_build_rxn_nh3_elimination(),
                                label='nh3_elim_graceful', expected_n_atoms=8,
                                expect_success=False)
        # The reaction's ts_species should still exist but have no successful TSGuesses.
        rxn = adapter.reactions[0]
        self.assertIsNotNone(rxn.ts_species)
        successful = [tsg for tsg in rxn.ts_species.ts_guesses if tsg.success]
        self.assertEqual(len(successful), 0)

    # ----- Group A: 1<->1 isomerizations -------------------------------------

    def test_e2e_vinyl_alcohol_to_acetaldehyde(self):
        """Keto-enol tautomerization C2H4O (7 atoms: 2C + 4H + 1O)."""
        self._run_e2e(_build_rxn_vinyl_alcohol_to_acetaldehyde(),
                      label='vinyl_alcohol_to_acetaldehyde', expected_n_atoms=7)

    def test_e2e_propenol_to_acetone(self):
        """Keto-enol tautomerization C3H6O (10 atoms)."""
        self._run_e2e(_build_rxn_propenol_to_acetone(),
                      label='propenol_to_acetone', expected_n_atoms=10)

    def test_e2e_cyclobutene_to_butadiene(self):
        """Electrocyclic ring opening C4H6 (10 atoms)."""
        self._run_e2e(_build_rxn_cyclobutene_to_butadiene(),
                      label='cyclobutene_to_butadiene', expected_n_atoms=10)

    def test_e2e_methoxy_to_hydroxymethyl(self):
        """1,2-H migration in CH3O radical (5 atoms)."""
        self._run_e2e(_build_rxn_methoxy_to_hydroxymethyl(),
                      label='methoxy_to_hydroxymethyl', expected_n_atoms=5)

    def test_e2e_ethoxy_to_alpha_hydroxyethyl(self):
        """1,2-H migration in CH3CH2O radical (8 atoms)."""
        self._run_e2e(_build_rxn_ethoxy_to_alpha_hydroxyethyl(),
                      label='ethoxy_to_alpha_hydroxyethyl', expected_n_atoms=8)

    def test_e2e_cyclopropane_to_propene(self):
        """Cyclopropane ring opening C3H6 (9 atoms)."""
        self._run_e2e(_build_rxn_cyclopropane_to_propene(),
                      label='cyclopropane_to_propene', expected_n_atoms=9)

    # ----- Group B: 1<->2 / 2<->1 (eliminations / cycloadditions) -----------

    def test_e2e_cyclobutane_retro_22(self):
        """Retro [2+2] cyclobutane -> 2 ethene (12 atoms)."""
        self._run_e2e(_build_rxn_cyclobutane_retro_22(),
                      label='cyclobutane_retro_22', expected_n_atoms=12)

    def test_e2e_da_butadiene_ethene(self):
        """Small Diels-Alder butadiene + ethene -> cyclohexene (16 atoms)."""
        self._run_e2e(_build_rxn_da_butadiene_ethene(),
                      label='da_butadiene_ethene', expected_n_atoms=16)

    def test_e2e_ethanol_dehydration(self):
        """β-elimination ethanol -> ethene + water (9 atoms)."""
        self._run_e2e(_build_rxn_ethanol_dehydration(),
                      label='ethanol_dehydration', expected_n_atoms=9)

    def test_e2e_methylamine_dehydrogenation(self):
        """1,2-dehydrogenation methylamine -> methyleneamine + H2 (7 atoms)."""
        self._run_e2e(_build_rxn_methylamine_dehydrogenation(),
                      label='methylamine_dehydrogenation', expected_n_atoms=7)

    def test_e2e_ethyl_peroxy_ho2_elimination(self):
        """β-scission ethyl peroxy -> ethene + HO2 (9 atoms)."""
        self._run_e2e(_build_rxn_ethyl_peroxy_ho2_elimination(),
                      label='ethyl_peroxy_ho2_elimination', expected_n_atoms=9)

    # ----- Group C: 2<->2 H-abstractions ------- -----------------------------

    def test_e2e_hab_ch4_oh(self):
        """H-abstraction CH4 + OH -> CH3 + H2O (7 atoms total: 1C + 5H + 1O)."""
        self._run_e2e(_build_rxn_hab_ch4_oh(),
                      label='hab_ch4_oh', expected_n_atoms=7)

    def test_e2e_hab_c2h6_h(self):
        """H-abstraction C2H6 + H -> C2H5 + H2 (9 atoms)."""
        self._run_e2e(_build_rxn_hab_c2h6_h(),
                      label='hab_c2h6_h', expected_n_atoms=9)

    def test_e2e_hab_nh3_oh(self):
        """H-abstraction NH3 + OH -> NH2 + H2O (6 atoms)."""
        self._run_e2e(_build_rxn_hab_nh3_oh(),
                      label='hab_nh3_oh', expected_n_atoms=6)

    def test_e2e_hab_ch3oh_h(self):
        """H-abstraction CH3OH + H -> CH2OH + H2 (7 atoms; abstracts α-CH)."""
        self._run_e2e(_build_rxn_hab_ch3oh_h(),
                      label='hab_ch3oh_h', expected_n_atoms=7)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
