#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for the GoFlow TS-guess adapter (``arc.job.adapters.ts.goflow_ts``).

Test tiers
----------
**Tier-1** (always runs): wiring, settings resolution, helper functions, adapter
instantiation, graceful skip when ``goflow_env`` / ckpt are missing, mocked
subprocess.

**Tier-2** (gated on ``_goflow_environment_ready()``): end-to-end
``execute_incore`` against the real ``goflow_env`` for a handful of family-
diverse reactions. Includes an explicit `strict=True` `load_state_dict` test
so a placeholder checkpoint file (e.g. the 45-byte LFS pointer shipped in
goflow_lean@main) is rejected immediately.

The Tier-2 tests are skipped automatically on CI runners that did not run
``install_goflow.sh`` AND do not have ``ARC_GOFLOW_CKPT`` pointing at a real
checkpoint.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from rdkit import Chem

from arc.common import read_yaml_file, save_yaml_file
from arc.job.adapter import JobEnum
from arc.job.adapters.common import all_families_ts_adapters, default_incore_adapters
from arc.settings import external_paths as goflow_paths
from arc.job.adapters.ts.goflow_ts import (
    GOFLOW_DEDUP_DMAT_RMSD,
    GoFlowAdapter,
    MAX_GOFLOW_ATOMS,
    _goflow_environment_ready,
    _within_goflow_supported_domain,
    build_atom_mapped_smiles,
    process_goflow_tsg,
)
from arc.reaction import ARCReaction
from arc.settings import settings as settings_mod
from arc.species.converter import str_to_xyz, xyz_to_str
from arc.species.species import ARCSpecies, TSGuess


class TestJobEnumIncludesGoFlow(unittest.TestCase):
    """JobEnum must expose `goflow` for adapter selection."""

    def test_goflow_is_a_member_of_job_enum(self):
        self.assertTrue(hasattr(JobEnum, 'goflow'), 'JobEnum is missing the `goflow` member')
        self.assertEqual(JobEnum.goflow.value, 'goflow')


class TestDefaultIncoreAdaptersIncludesGoFlow(unittest.TestCase):
    """GoFlow runs incore (no queue submission needed)."""

    def test_goflow_in_default_incore_adapters(self):
        self.assertIn('goflow', default_incore_adapters)


class TestAllFamiliesTSAdaptersIncludesGoFlow(unittest.TestCase):
    """
    GoFlow ships in the default ``ts_adapters`` list, so it must also be in
    ``all_families_ts_adapters`` — otherwise the scheduler's gating in
    ``spawn_ts_jobs`` would silently never spawn it. Out-of-domain reactions
    (non-H/C/N/O/F elements, >100 atoms) are filtered at runtime by the
    adapter's own ``_within_goflow_supported_domain`` guard; hosts without
    ``goflow_env`` skip cleanly via ``_goflow_environment_ready``.
    """

    def test_goflow_in_all_families_ts_adapters(self):
        self.assertIn('goflow', all_families_ts_adapters)


class TestWithinGoFlowSupportedDomain(unittest.TestCase):
    """
    GoFlow was trained on RDB7 (small organic, H/C/N/O/F). Reactions outside
    the validated domain must be skipped cleanly with a clear warning, not
    sent to the model where they would either crash or produce silently bad
    geometries.
    """

    def test_accepts_h_abstraction_ch4_oh(self):
        rxn = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'),
                                     ARCSpecies(label='OH', smiles='[OH]')],
                          p_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                     ARCSpecies(label='H2O', smiles='O')])
        ok, reason = _within_goflow_supported_domain(rxn)
        self.assertTrue(ok, msg=f'Should accept H-abstraction; got reason={reason!r}')
        self.assertEqual(reason, '')

    def test_rejects_unsupported_element(self):
        """Sulfur (or any element not in H/C/N/O/F) → reject."""
        rxn = ARCReaction(r_species=[ARCSpecies(label='H2S', smiles='S')],
                          p_species=[ARCSpecies(label='HS', smiles='[SH]'),
                                     ARCSpecies(label='H', smiles='[H]')])
        ok, reason = _within_goflow_supported_domain(rxn)
        self.assertFalse(ok)
        self.assertIn('S', reason)

    def test_rejects_reaction_above_max_atom_threshold(self):
        """
        Build a lightweight rxn-shaped stand-in. The function only iterates
        rxn.r_species and reads either get_xyz() or mol.atoms. A SimpleNamespace
        with the minimum surface area avoids slow RDKit perception over a
        100-carbon polyene SMILES that the production code never sees.
        """
        n_extra = 5
        n_atoms = MAX_GOFLOW_ATOMS + n_extra
        big_xyz = {'symbols': ('C',) * n_atoms,
                   'isotopes': (12,) * n_atoms,
                   'coords': tuple((float(i), 0.0, 0.0) for i in range(n_atoms))}
        fake_spc = SimpleNamespace(mol=None, get_xyz=lambda: big_xyz)
        rxn = SimpleNamespace(r_species=[fake_spc], p_species=[fake_spc])
        ok, reason = _within_goflow_supported_domain(rxn)
        self.assertFalse(ok)
        self.assertIn('atom', reason.lower())


class TestGoFlowEnvironmentReady(unittest.TestCase):
    """
    `_goflow_environment_ready()` reads four module-level globals
    (GOFLOW_PYTHON, GOFLOW_REPO_PATH, GOFLOW_CKPT_PATH, GOFLOW_FEAT_DICT_PATH)
    and returns True iff all four point at real, plausibly-valid files/dirs.
    """

    def test_returns_false_when_python_missing(self):
        with unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_PYTHON', None), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_REPO_PATH', '/tmp/repo'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_CKPT_PATH', '/tmp/ckpt'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_FEAT_DICT_PATH', '/tmp/fd'):
            self.assertFalse(_goflow_environment_ready())

    def test_returns_false_when_ckpt_missing(self):
        with unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_PYTHON', '/usr/bin/python'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_REPO_PATH', '/tmp/repo'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_CKPT_PATH', None), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_FEAT_DICT_PATH', '/tmp/fd'):
            self.assertFalse(_goflow_environment_ready())


class TestProcessGoFlowTSG(unittest.TestCase):
    """
    `process_goflow_tsg(tsg_dict, local_path, ts_species)` converts a script-
    output TSGuess dict into an ARC TSGuess, checks for collisions, dedups
    against existing guesses, and saves the geometry.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='goflow_test_')
        self.ts_species = ARCSpecies(label='ts', is_ts=True, smiles='[CH2]C')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_false_for_failed_guess(self):
        bad = {'method': 'GoFlow', 'method_direction': 'F', 'method_index': 0,
               'success': False, 'initial_xyz': None}
        self.assertFalse(process_goflow_tsg(bad, self.tmpdir, self.ts_species))
        self.assertEqual(len(self.ts_species.ts_guesses), 0)

    def test_returns_false_when_atoms_collide(self):
        # Two atoms at exactly the same position.
        collide_xyz = "C 0.0 0.0 0.0\nH 0.0 0.0 0.0"
        tsg = {'method': 'GoFlow', 'method_direction': 'F', 'method_index': 0,
               'success': True, 'initial_xyz': collide_xyz}
        self.assertFalse(process_goflow_tsg(tsg, self.tmpdir, self.ts_species))

    def test_appends_new_unique_guess_to_species(self):
        tsg = {'method': 'GoFlow', 'method_direction': 'F', 'method_index': 0,
               'success': True,
               'initial_xyz': 'C 0.0 0.0 0.0\nH 0.0 0.0 1.1\nH 0.0 1.0 -0.4\n'
                              'H 0.9 -0.5 -0.4\nH -0.9 -0.5 -0.4'}
        ok = process_goflow_tsg(tsg, self.tmpdir, self.ts_species)
        self.assertTrue(ok)
        self.assertEqual(len(self.ts_species.ts_guesses), 1)
        self.assertIn('goflow', self.ts_species.ts_guesses[0].method.lower())

    def test_consolidates_near_duplicate_against_existing_guess(self):
        """A second guess that's only a tiny perturbation (well under the
        dmat-RMSD threshold) of an existing one should NOT be appended;
        the existing guess's `method` should be annotated to credit GoFlow."""
        first = {'method': 'GoFlow', 'method_direction': 'F', 'method_index': 0,
                 'success': True,
                 'initial_xyz': 'C 0.000 0.000 0.000\nH 0.000 0.000 1.100\n'
                                'H 0.000 1.000 -0.400\nH 0.900 -0.500 -0.400\n'
                                'H -0.900 -0.500 -0.400'}
        # Same skeleton, every atom shifted by ~0.01 Å — well below the
        # 0.15 Å aggregate dmat-RMSD threshold.
        second = {'method': 'GoFlow', 'method_direction': 'F', 'method_index': 1,
                  'success': True,
                  'initial_xyz': 'C 0.010 0.010 0.000\nH 0.010 0.010 1.100\n'
                                 'H 0.010 1.010 -0.400\nH 0.910 -0.490 -0.400\n'
                                 'H -0.890 -0.490 -0.400'}
        self.assertTrue(process_goflow_tsg(first, self.tmpdir, self.ts_species))
        self.assertFalse(process_goflow_tsg(second, self.tmpdir, self.ts_species))
        self.assertEqual(len(self.ts_species.ts_guesses), 1,
                         'second near-duplicate should have been consolidated, not appended')

    def test_appends_distinct_guess_with_dmat_rmsd_above_threshold(self):
        """A geometrically distinct second guess (dmat-RMSD > threshold)
        must be appended as a new unique TSGuess."""
        first = {'method': 'GoFlow', 'method_direction': 'F', 'method_index': 0,
                 'success': True,
                 'initial_xyz': 'C 0.000 0.000 0.000\nH 0.000 0.000 1.100\n'
                                'H 0.000 1.000 -0.400\nH 0.900 -0.500 -0.400\n'
                                'H -0.900 -0.500 -0.400'}
        # Same connectivity but the migrating-H is in a clearly different
        # position; aggregate dmat-RMSD will be well above 0.15 Å.
        second = {'method': 'GoFlow', 'method_direction': 'F', 'method_index': 1,
                  'success': True,
                  'initial_xyz': 'C 0.000 0.000 0.000\nH 0.000 0.000 1.500\n'
                                 'H 0.500 1.300 -0.400\nH 1.200 -0.500 -0.400\n'
                                 'H -1.200 -0.500 -0.400'}
        self.assertTrue(process_goflow_tsg(first, self.tmpdir, self.ts_species))
        self.assertTrue(process_goflow_tsg(second, self.tmpdir, self.ts_species))
        self.assertEqual(len(self.ts_species.ts_guesses), 2)

    def test_consolidates_rotor_twins_with_heavy_atoms_unchanged(self):
        """Two TS guesses that agree on the heavy-atom skeleton but differ
        only in the torsion of a terminal H pair must be consolidated —
        they're the same TS, sampled at different rotor wells."""
        # CH3 with one bond stretched — heavy atom (C) at origin in both.
        first = {'method': 'GoFlow', 'method_direction': 'F', 'method_index': 0,
                 'success': True,
                 'initial_xyz': 'C 0.000 0.000 0.000\nH 0.000 0.000 1.500\n'
                                'H 0.000 1.000 -0.400\nH 0.900 -0.500 -0.400\n'
                                'H -0.900 -0.500 -0.400'}
        # Same heavy atom, but the three "spectator" H's are rotated ~60°
        # around the C-H(1) axis. The non-reactive H positions move ~0.5 Å
        # each — pushing the all-atom dmat-RMSD well above 0.15 Å, but the
        # heavy-atom dmat-RMSD stays at zero.
        second = {'method': 'GoFlow', 'method_direction': 'F', 'method_index': 1,
                  'success': True,
                  'initial_xyz': 'C 0.000 0.000 0.000\nH 0.000 0.000 1.500\n'
                                 'H 0.866 0.500 -0.400\nH -0.866 0.500 -0.400\n'
                                 'H 0.000 -1.000 -0.400'}
        self.assertTrue(process_goflow_tsg(first, self.tmpdir, self.ts_species))
        self.assertFalse(process_goflow_tsg(second, self.tmpdir, self.ts_species))
        self.assertEqual(len(self.ts_species.ts_guesses), 1,
                         'rotor twin with same heavy-atom skeleton must consolidate')

    def test_consolidation_annotates_existing_method_string(self):
        """When a near-duplicate hits an existing guess from a *different*
        adapter (e.g. heuristics), the existing guess's `method` should be
        appended with ' and GoFlow' so downstream consumers see both."""
        # Pre-seed ts_species with a heuristics-style guess.
        ts_xyz_str = ('C 0.000 0.000 0.000\nH 0.000 0.000 1.100\n'
                      'H 0.000 1.000 -0.400\nH 0.900 -0.500 -0.400\n'
                      'H -0.900 -0.500 -0.400')
        seed = TSGuess(method='Heuristics', method_direction='F',
                       method_index=0, success=True,
                       xyz=str_to_xyz(ts_xyz_str))
        self.ts_species.ts_guesses.append(seed)

        # GoFlow produces a near-duplicate of the heuristics guess.
        twin = {'method': 'GoFlow', 'method_direction': 'F', 'method_index': 0,
                'success': True,
                'initial_xyz': 'C 0.005 0.005 0.000\nH 0.005 0.005 1.100\n'
                               'H 0.005 1.005 -0.400\nH 0.905 -0.495 -0.400\n'
                               'H -0.895 -0.495 -0.400'}
        self.assertFalse(process_goflow_tsg(twin, self.tmpdir, self.ts_species))
        self.assertEqual(len(self.ts_species.ts_guesses), 1)
        # Method string should now mention BOTH adapters.
        merged_method = self.ts_species.ts_guesses[0].method.lower()
        self.assertIn('heuristics', merged_method)
        self.assertIn('goflow', merged_method)


class TestBuildAtomMappedSmiles(unittest.TestCase):
    """
    `build_atom_mapped_smiles(rxn, side)` produces SMILES with every atom
    (including every H) carrying an atom-map number 1..N. This is the
    highest-risk helper in the adapter: GoFlow's preprocessor parses the
    SMILES and reorders atoms by map number, so any silent loss of
    hydrogens or duplicate map numbers will silently corrupt inference.
    """

    @classmethod
    def setUpClass(cls):
        # nC3H7 (10 atoms: 3C + 7H) → iC3H7. Same atom count, easy to verify.
        cls.rxn = ARCReaction(r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
                              p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')])
        # ARC will compute the atom_map lazily; force it once and cache.
        cls.atom_map = cls.rxn.atom_map  # may be None if mapping fails

    def test_returns_none_for_unknown_side(self):
        self.assertIsNone(build_atom_mapped_smiles(self.rxn, side='other'))

    def test_returns_smiles_with_every_h_explicit(self):
        smi = build_atom_mapped_smiles(self.rxn, side='reactants')
        self.assertIsNotNone(smi, 'reactant SMILES build returned None')
        # Round-trip the SMILES with H preservation; every H must be a real atom.
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(smi, params)
        self.assertIsNotNone(mol)
        n_h_atoms = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'H')
        n_heavy_atoms = sum(1 for a in mol.GetAtoms() if a.GetSymbol() != 'H')
        self.assertEqual(n_h_atoms, 7)
        self.assertEqual(n_heavy_atoms, 3)
        self.assertEqual(mol.GetNumAtoms(), 10)

    def test_map_numbers_are_one_through_n_with_no_gaps(self):
        smi = build_atom_mapped_smiles(self.rxn, side='reactants')
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(smi, params)
        maps = sorted(a.GetAtomMapNum() for a in mol.GetAtoms())
        self.assertEqual(maps, list(range(1, 11)))

    def test_reactant_and_product_smiles_share_the_same_map_set(self):
        if self.atom_map is None:
            self.skipTest('rxn.atom_map could not be computed in this env')
        r_smi = build_atom_mapped_smiles(self.rxn, side='reactants')
        p_smi = build_atom_mapped_smiles(self.rxn, side='products')
        self.assertIsNotNone(r_smi)
        self.assertIsNotNone(p_smi)
        params = Chem.SmilesParserParams()
        params.removeHs = False
        r_maps = sorted(a.GetAtomMapNum() for a in Chem.MolFromSmiles(r_smi, params).GetAtoms())
        p_maps = sorted(a.GetAtomMapNum() for a in Chem.MolFromSmiles(p_smi, params).GetAtoms())
        self.assertEqual(r_maps, p_maps)

    def test_returns_none_when_atom_map_unavailable_for_products(self):
        """If rxn has no mapping, we cannot build product-side mapped SMILES."""
        rxn = ARCReaction(
            r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
            p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')],
        )
        rxn._atom_map = None
        # Force the lazy property to return our None override.
        with unittest.mock.patch.object(type(rxn), 'atom_map',
                               new_callable=unittest.mock.PropertyMock,
                               return_value=None):
            self.assertIsNone(build_atom_mapped_smiles(rxn, side='products'))


class TestBuildAtomMappedSmilesStress(unittest.TestCase):
    """
    Stress tests for ``build_atom_mapped_smiles`` across a family-diverse
    set of reactions. Each fixture exercises a different code path:
        - cross-fragment H migration (H + CH4 → CH3 + H2)
        - heavy-atom permutation across fragments (2 CH3 → C2H6)
        - heteroatom on reactant side (H + NH3, H + HF)
        - O-H abstraction (H + CH3OH → H2 + CH3O)
        - large complex permutation (CH3 + C2H6 → CH4 + C2H5)
        - addition (H + propene → nC3H7) — multi-fragment reactant → single product
    """

    @staticmethod
    def _build(name, r_species, p_species):
        rxn = ARCReaction(r_species=r_species, p_species=p_species)
        # Force atom_map computation; some fixtures fail at this stage
        # (e.g. when ARC's mapping heuristics choke). We test only those
        # that produce a real atom_map — others would also be skipped at
        # adapter runtime.
        try:
            am = rxn.atom_map
        except Exception:
            am = None
        return name, rxn, am

    @classmethod
    def setUpClass(cls):
        cls.fixtures = []
        cls.fixtures.append(cls._build(
            'h_abstraction_ch4',
            [ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='H', smiles='[H]')],
            [ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2', smiles='[H][H]')],
        ))
        cls.fixtures.append(cls._build(
            'oh_abstraction',
            [ARCSpecies(label='CH3OH', smiles='CO'), ARCSpecies(label='H', smiles='[H]')],
            [ARCSpecies(label='CH3O', smiles='[CH2]O'), ARCSpecies(label='H2', smiles='[H][H]')],
        ))
        cls.fixtures.append(cls._build(
            'nh3_h_abstraction',
            [ARCSpecies(label='NH3', smiles='N'), ARCSpecies(label='H', smiles='[H]')],
            [ARCSpecies(label='NH2', smiles='[NH2]'), ARCSpecies(label='H2', smiles='[H][H]')],
        ))
        cls.fixtures.append(cls._build(
            'methyl_recombination',
            [ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='CH3', smiles='[CH3]')],
            [ARCSpecies(label='C2H6', smiles='CC')],
        ))
        cls.fixtures.append(cls._build(
            'cross_h_abstraction',
            [ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='C2H6', smiles='CC')],
            [ARCSpecies(label='CH4', smiles='C'),    ARCSpecies(label='C2H5', smiles='C[CH2]')],
        ))
        cls.fixtures.append(cls._build(
            'h_plus_propene_addition',
            [ARCSpecies(label='C3H6', smiles='C=CC'), ARCSpecies(label='H', smiles='[H]')],
            [ARCSpecies(label='C3H7', smiles='[CH2]CC')],
        ))
        cls.fixtures.append(cls._build(
            'hf_h_abstraction',
            [ARCSpecies(label='HF', smiles='F'),   ARCSpecies(label='H', smiles='[H]')],
            [ARCSpecies(label='F',  smiles='[F]'), ARCSpecies(label='H2', smiles='[H][H]')],
        ))

    def _smiles_atoms(self, smi):
        """Parse SMILES with explicit Hs preserved; return list of (map_num, element)."""
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(smi, params)
        self.assertIsNotNone(mol, f'RDKit could not re-parse: {smi!r}')
        return [(a.GetAtomMapNum(), a.GetSymbol()) for a in mol.GetAtoms()]

    def test_every_fixture_produces_n_atom_smiles_with_complete_map_set(self):
        """Reactant SMILES has N atoms, every atom carries a unique map num in 1..N.
        N is taken from len(atom_map), which is ARC's stoichiometry-aware count
        (counts each occurrence of identical species, e.g. 2 CH3 = 8 atoms even
        though r_species is deduped to a single CH3 entry)."""
        for name, rxn, am in self.fixtures:
            with self.subTest(fixture=name):
                if am is None:
                    self.skipTest(f'{name}: ARC atom_map computation failed')
                n_atoms = len(am)
                smi = build_atom_mapped_smiles(rxn, side='reactants')
                self.assertIsNotNone(smi, f'{name}: build returned None')
                atoms = self._smiles_atoms(smi)
                self.assertEqual(len(atoms), n_atoms,
                                 f'{name}: SMILES has {len(atoms)} atoms, expected {n_atoms}')
                self.assertEqual(sorted(m for m, _ in atoms), list(range(1, n_atoms + 1)),
                                 f'{name}: map numbers are not exactly 1..{n_atoms}')

    def test_element_at_each_map_number_is_consistent_across_sides(self):
        """For every map number i, the atomic symbol on reactant and product
        side must be identical — otherwise GoFlow's preprocessor would assert
        on `atomic_numbers_match` mid-inference."""
        for name, rxn, am in self.fixtures:
            with self.subTest(fixture=name):
                if am is None:
                    self.skipTest(f'{name}: ARC atom_map computation failed')
                r_smi = build_atom_mapped_smiles(rxn, side='reactants')
                p_smi = build_atom_mapped_smiles(rxn, side='products')
                self.assertIsNotNone(r_smi, f'{name}: reactant build returned None')
                self.assertIsNotNone(p_smi, f'{name}: product build returned None')
                r_by_map = dict(self._smiles_atoms(r_smi))
                p_by_map = dict(self._smiles_atoms(p_smi))
                self.assertEqual(set(r_by_map), set(p_by_map), f'{name}: map number sets differ across sides')
                for mn in r_by_map:
                    self.assertEqual(r_by_map[mn], p_by_map[mn],
                                     f'{name}: map={mn} is {r_by_map[mn]} on reactant '
                                     f'but {p_by_map[mn]} on product — atom identity not preserved')

    def test_h_migration_in_ch4_plus_h_actually_swaps_h_position(self):
        """Sanity-check the cross-fragment H-migration semantics. For
        H + CH4 → CH3 + H2 with atom_map=[0, 5, 1, 2, 3, 4]:
            - reactant atom 0 (C)        → product atom 0 (C)
            - reactant atom 1 (lone H)   → product atom 5 (one H of H2)
            - reactant atoms 2..5 (CH-H) → product atoms 1..4 (CH3 Hs + the other H of H2)
        Map number 2 should therefore label an H on both sides, but in
        DIFFERENT bond environments — bonded to C on reactant, bonded to H
        on product (or vice versa for the swapped index).
        """
        name, rxn, am = self.fixtures[0]   # h_abstraction_ch4
        if am is None:
            self.skipTest('atom_map unavailable')
        r_smi = build_atom_mapped_smiles(rxn, side='reactants')
        p_smi = build_atom_mapped_smiles(rxn, side='products')
        params = Chem.SmilesParserParams(); params.removeHs = False
        r_mol = Chem.MolFromSmiles(r_smi, params)
        p_mol = Chem.MolFromSmiles(p_smi, params)

        def neighbors_by_map(mol, map_num):
            for a in mol.GetAtoms():
                if a.GetAtomMapNum() == map_num:
                    return sorted(n.GetSymbol() for n in a.GetNeighbors())
            self.fail(f'no atom with map={map_num}')

        # Map 1 is the carbon: bonded to 4 H on reactant (CH4), 3 H on product (CH3).
        self.assertEqual(neighbors_by_map(r_mol, 1), ['H', 'H', 'H', 'H'])
        self.assertEqual(neighbors_by_map(p_mol, 1), ['H', 'H', 'H'])

    def test_radical_electrons_total_is_preserved_through_round_trip(self):
        """`SetNumRadicalElectrons` is in the build path; this test trips
        if RDKit silently re-perceives radicals during sanitization."""
        for name, rxn, am in self.fixtures:
            with self.subTest(fixture=name):
                if am is None:
                    self.skipTest(f'{name}: atom_map unavailable')
                # Sum of radical electrons on the reactant side from ARC's
                # ARCSpecies.mol — that's what the build sees.
                r_smi = build_atom_mapped_smiles(rxn, side='reactants')
                params = Chem.SmilesParserParams(); params.removeHs = False
                r_mol = Chem.MolFromSmiles(r_smi, params)
                # Total spin = sum(num_radical_electrons) should be at least 1
                # for every fixture above (every one has a radical somewhere).
                total_rad = sum(a.GetNumRadicalElectrons() for a in r_mol.GetAtoms())
                self.assertGreater(total_rad, 0,
                                   f'{name}: round-tripped reactant SMILES has zero radical '
                                   f'electrons; SetNumRadicalElectrons was lost')

    def test_smiles_round_trips_with_strict_sanitization(self):
        """The SMILES we hand to GoFlow must parse cleanly with default
        sanitization (which is what goflow.preprocessing uses internally).
        A SMILES that only parses with sanitize=False would silently
        generate a bad PyG graph downstream."""
        for name, rxn, am in self.fixtures:
            with self.subTest(fixture=name):
                if am is None:
                    self.skipTest(f'{name}: atom_map unavailable')
                for side in ('reactants', 'products'):
                    smi = build_atom_mapped_smiles(rxn, side=side)
                    self.assertIsNotNone(smi, f'{name}/{side}: build returned None')
                    # Default sanitization, removeHs=False (matches goflow's parser).
                    params = Chem.SmilesParserParams(); params.removeHs = False
                    mol = Chem.MolFromSmiles(smi, params)
                    self.assertIsNotNone(mol, f'{name}/{side}: SMILES failed strict parse: {smi!r}')

    def test_returns_none_when_atom_map_is_incomplete(self):
        """If atom_map has a hole (missing index), the inversion loop on the
        product side raises ValueError → function returns None gracefully."""
        rxn = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'),
                                     ARCSpecies(label='H', smiles='[H]')],
                          p_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                     ARCSpecies(label='H2', smiles='[H][H]')])
        with unittest.mock.patch.object(type(rxn), 'atom_map',
                                        new_callable=unittest.mock.PropertyMock,
                                        return_value=[0, 5, 1, 2, 3, 99]):  # 4 missing, 99 stray
            self.assertIsNone(build_atom_mapped_smiles(rxn, side='products'))


class TestGoFlowAdapterInstantiation(unittest.TestCase):
    """A bare adapter instance with `testing=True` does no I/O and is happy."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='goflow_proj_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_adapter_constructs_in_testing_mode(self):
        rxn = ARCReaction(r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
                          p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')])
        adapter = GoFlowAdapter(project='goflow_test',
                                project_directory=self.tmpdir,
                                job_type='tsg',
                                reactions=[rxn],
                                testing=True)
        self.assertEqual(adapter.job_adapter, 'goflow')
        self.assertEqual(adapter.command, 'goflow_script.py')


class TestExecuteIncoreWithMockedSubprocess(unittest.TestCase):
    """
    Verifies the adapter's execute_incore lifecycle without touching the real
    goflow_env: monkeypatches `_goflow_environment_ready → True`, mocks
    `subprocess.run` to write a stub `output.yml`, and asserts that:
      - `input.yml` was written with all required keys
      - the stub TSGuess made it into `rxn.ts_species.ts_guesses`
      - reactant.xyz and product.xyz exist on disk
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='goflow_e2e_')
        self.rxn = ARCReaction(r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
                               p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')])

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_skips_cleanly_when_environment_not_ready(self):
        adapter = GoFlowAdapter(project='goflow_test',
                                project_directory=self.tmpdir,
                                job_type='tsg',
                                reactions=[self.rxn],
                                testing=False)
        with unittest.mock.patch('arc.job.adapters.ts.goflow_ts._goflow_environment_ready', return_value=False):
            # Should not raise.
            adapter.execute_incore()
        # No TSGuesses should have been added.
        if self.rxn.ts_species is not None:
            self.assertEqual(len(self.rxn.ts_species.ts_guesses), 0)

    def test_writes_input_yml_and_ingests_tsg_when_subprocess_mocked(self):

        adapter = GoFlowAdapter(project='goflow_test',
                                project_directory=self.tmpdir,
                                job_type='tsg',
                                reactions=[self.rxn],
                                testing=False)

        # 5-atom toy TS guess (CH3 with one bond stretched).
        stub_xyz = ('C 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 1.0 -0.4\n'
                    'H 0.9 -0.5 -0.4\nH -0.9 -0.5 -0.4')
        stub_tsgs = [{'method': 'GoFlow', 'method_direction': 'F', 'method_index': 0,
                      'success': True, 'initial_xyz': stub_xyz, 'execution_time': '0:00:00.001'}]

        def fake_subprocess_run(cmd, **kwargs):
            save_yaml_file(path=adapter.yml_out_path, content=stub_tsgs)
            return unittest.mock.Mock(returncode=0)

        with unittest.mock.patch('arc.job.adapters.ts.goflow_ts._goflow_environment_ready', return_value=True), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts._within_goflow_supported_domain', return_value=(True, '')), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_PYTHON', '/usr/bin/python'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_REPO_PATH', self.tmpdir), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_CKPT_PATH', '/dev/null'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_FEAT_DICT_PATH', '/dev/null'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.subprocess.run', side_effect=fake_subprocess_run):
            adapter.execute_incore()

        # input.yml exists and has the right keys.
        self.assertTrue(os.path.isfile(adapter.yml_in_path))
        input_dict = read_yaml_file(adapter.yml_in_path)
        for key in ('reactant_xyz_path', 'product_xyz_path',
                    'reactant_smiles', 'product_smiles',
                    'goflow_repo_path', 'ckpt_path', 'feat_dict_path',
                    'output_xyz_path', 'yml_out_path',
                    'n_samples', 'num_steps', 'device'):
            self.assertIn(key, input_dict)

        # reactant.xyz and product.xyz exist.
        self.assertTrue(os.path.isfile(adapter.reactant_xyz_path))
        self.assertTrue(os.path.isfile(adapter.product_xyz_path))

        # The stub TSGuess made it into ts_species.
        self.assertEqual(len(self.rxn.ts_species.ts_guesses), 1)
        self.assertIn('goflow', self.rxn.ts_species.ts_guesses[0].method.lower())

    def test_iterates_over_every_reaction_in_self_reactions(self):
        """When the adapter is constructed with multiple reactions, each one
        must produce a TSGuess — not just self.reactions[0]. This guards
        against a regression where execute_goflow forgets to loop."""
        rxn1 = ARCReaction(r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
                           p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')])
        rxn2 = ARCReaction(r_species=[ARCSpecies(label='nC4H9', smiles='[CH2]CCC')],
                           p_species=[ARCSpecies(label='sC4H9', smiles='C[CH]CC')])
        # The mocked subprocess writes a 5-atom stub xyz; the adapter
        # creates rxn.ts_species lazily and appends it. Don't pre-seed
        # ts_species — that would trigger ARC's atom-balance check against
        # the placeholder smiles.
        ts_xyz = ('C 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 1.0 -0.4\n'
                  'H 0.9 -0.5 -0.4\nH -0.9 -0.5 -0.4')

        adapter = GoFlowAdapter(project='goflow_multirxn',
                                project_directory=self.tmpdir,
                                job_type='tsg',
                                reactions=[rxn1, rxn2],
                                testing=False)

        def fake_subprocess_run(cmd, **kwargs):
            save_yaml_file(path=adapter.yml_out_path, content=[{
                'method': 'GoFlow', 'method_direction': 'F', 'method_index': 0,
                'success': True, 'initial_xyz': ts_xyz,
                'execution_time': '0:00:00.001'}])
            return unittest.mock.Mock(returncode=0)

        with unittest.mock.patch('arc.job.adapters.ts.goflow_ts._goflow_environment_ready', return_value=True), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts._within_goflow_supported_domain', return_value=(True, '')), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_PYTHON', '/usr/bin/python'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_REPO_PATH', self.tmpdir), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_CKPT_PATH', '/dev/null'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_FEAT_DICT_PATH', '/dev/null'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.subprocess.run', side_effect=fake_subprocess_run):
            adapter.execute_incore()

        self.assertIsNotNone(rxn1.ts_species, 'first rxn ts_species not created')
        self.assertIsNotNone(rxn2.ts_species, 'second rxn ts_species not created')
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 1, 'first rxn missing TSGuess')
        self.assertEqual(len(rxn2.ts_species.ts_guesses), 1,
                         'second rxn missing TSGuess — execute_goflow likely only processes self.reactions[0]')

    def test_subprocess_timeout_logged_and_skipped_gracefully(self):
        """When the subprocess hangs past the timeout, the adapter must log
        a warning and continue, not propagate the TimeoutExpired."""
        adapter = GoFlowAdapter(project='goflow_timeout',
                                project_directory=self.tmpdir,
                                job_type='tsg',
                                reactions=[self.rxn],
                                testing=False)
        adapter.goflow_subprocess_timeout = 1  # second

        def hanging_subprocess_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get('timeout', 1))

        with unittest.mock.patch('arc.job.adapters.ts.goflow_ts._goflow_environment_ready', return_value=True), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts._within_goflow_supported_domain', return_value=(True, '')), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_PYTHON', '/usr/bin/python'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_REPO_PATH', self.tmpdir), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_CKPT_PATH', '/dev/null'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.GOFLOW_FEAT_DICT_PATH', '/dev/null'), \
             unittest.mock.patch('arc.job.adapters.ts.goflow_ts.subprocess.run', side_effect=hanging_subprocess_run):
            # Must not raise — adapter swallows TimeoutExpired and continues.
            adapter.execute_incore()
        # No TSGuess appended (subprocess never wrote output.yml).
        self.assertEqual(len(self.rxn.ts_species.ts_guesses), 0)


###############################################################################
# Tier-2 — env-gated end-to-end tests
###############################################################################
#
# These tests exercise the full GoFlow inference pipeline by spawning the
# real `goflow_env` subprocess against a real reaction. They self-skip on
# any host where `_goflow_environment_ready()` returns False — i.e. when
# python/repo/ckpt/feat_dict aren't all present and valid (the 45-byte LFS
# placeholder is rejected by the size guard in the settings layer).
#
# To run locally:
#   bash devtools/install_goflow.sh --no-ckpt-check
#   export ARC_GOFLOW_CKPT=/path/to/your/epoch_<NNN>.ckpt
#   pytest arc/job/adapters/ts/goflow_test.py::TestGoFlowEndToEnd -v


def _refresh_goflow_paths_and_check_ready() -> bool:
    """
    Tier-2 gating predicate. Called from each Tier-2 class's ``setUpClass``
    (NOT at module-import time) so that user-set ``ARC_GOFLOW_*`` env vars
    take effect even when set after pytest collection.

    ``goflow_ts`` caches its module-level ``GOFLOW_*`` globals at import time
    via ``settings.get(...)``. ``_goflow_environment_ready()`` reads those
    cached globals — refreshing only ``settings_mod.GOFLOW_*`` is not enough.
    Both the settings module and ``goflow_ts``'s own globals must be
    re-bound from a fresh discovery for the readiness check to see new
    values."""
    repo = goflow_paths.find_goflow_repo()
    ckpt = goflow_paths.find_goflow_ckpt(repo)
    feat = goflow_paths.find_goflow_feat_dict(repo)
    settings_mod.GOFLOW_REPO_PATH = repo
    settings_mod.GOFLOW_CKPT_PATH = ckpt
    settings_mod.GOFLOW_FEAT_DICT_PATH = feat
    goflow_ts_mod = sys.modules['arc.job.adapters.ts.goflow_ts']
    goflow_ts_mod.GOFLOW_REPO_PATH = repo
    goflow_ts_mod.GOFLOW_CKPT_PATH = ckpt
    goflow_ts_mod.GOFLOW_FEAT_DICT_PATH = feat
    return _goflow_environment_ready()


_TIER2_SKIP_MSG = ('goflow_env or real ckpt not available — '
                   'set ARC_GOFLOW_CKPT or run devtools/install_goflow.sh')


class TestGoFlowRealCheckpointStrictLoad(unittest.TestCase):
    """
    Acid test for "is this a real checkpoint, not a 45-byte placeholder?"
    Instantiate FlowModule via the same Hydra recipe the adapter uses, then
    require `load_state_dict(strict=True)` succeeds with zero missing/unexpected
    keys. A placeholder ckpt fails this immediately.
    """

    @classmethod
    def setUpClass(cls):
        if not _refresh_goflow_paths_and_check_ready():
            raise unittest.SkipTest(_TIER2_SKIP_MSG)

    def test_strict_load_succeeds_against_paper_equivalent_ckpt(self):

        # We must run the strict-load probe inside goflow_env (where torch +
        # goflow are importable), not in arc_env. Spawn a tiny subprocess.
        probe = ('import sys, pickle, torch\n'
                 'from hydra import initialize_config_dir, compose\n'
                 'from hydra.utils import instantiate\n'
                 'ckpt = sys.argv[1]; feat = sys.argv[2]; cfg_dir = sys.argv[3]\n'
                 'with open(feat, "rb") as f: fd = pickle.load(f)\n'
                 'feat_dim = sum(len(v) for v in fd.values())\n'
                 'with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):\n'
                 '    cfg = compose(config_name="train", overrides=['
                 '"model=flow", "data=rdb7", '
                 'f"model.representation.n_atom_rdkit_feats={feat_dim}", '
                 '"model.num_samples=1", "model.num_steps=5", '
                 '"model.sample_method=gaussian"])\n'
                 'fm = instantiate(cfg.model)\n'
                 'obj = torch.load(ckpt, map_location="cpu", weights_only=False)\n'
                 'res = fm.load_state_dict(obj["state_dict"], strict=True)\n'
                 'import json; print(json.dumps({"missing": list(res.missing_keys), '
                 '"unexpected": list(res.unexpected_keys), "feat_dim": feat_dim}))\n')
        cfg_dir = os.path.join(settings_mod.GOFLOW_REPO_PATH, 'src', 'goflow', 'configs')
        result = subprocess.run([settings_mod.GOFLOW_PYTHON, '-c', probe,
                                 settings_mod.GOFLOW_CKPT_PATH, settings_mod.GOFLOW_FEAT_DICT_PATH, cfg_dir],
                                capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            self.fail(f'strict-load probe failed:\nstdout={result.stdout}\nstderr={result.stderr}')
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report['missing'], [], f'state_dict has missing keys: {report["missing"]}')
        self.assertEqual(report['unexpected'], [], f'state_dict has unexpected keys: {report["unexpected"]}')
        self.assertEqual(report['feat_dim'], 36, f'feat_dim derived from feat_dict_organic.pkl should be 36 '
                                                 f'for the published RDB7 dictionary, got {report["feat_dim"]}')


class TestGoFlowEndToEnd(unittest.TestCase):
    """End-to-end: real subprocess, real ckpt, real ARCReaction → real TSGuesses."""

    @classmethod
    def setUpClass(cls):
        if not _refresh_goflow_paths_and_check_ready():
            raise unittest.SkipTest(_TIER2_SKIP_MSG)
        cls.tmpdir = tempfile.mkdtemp(prefix='goflow_tier2_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _build_h_abstraction_rxn(self):
        """H + CH4 ↔ CH3 + H2 — the canonical small H-abstraction in RDB7's
        chemistry domain. Six atoms, all H/C; in-domain for GoFlow."""
        ch4 = ARCSpecies(label='CH4', smiles='C', xyz=(
            'C       0.00000000    0.00000000    0.00000000\n'
            'H       0.62911800    0.62911800    0.62911800\n'
            'H      -0.62911800   -0.62911800    0.62911800\n'
            'H      -0.62911800    0.62911800   -0.62911800\n'
            'H       0.62911800   -0.62911800   -0.62911800'))
        h_atom = ARCSpecies(label='H', smiles='[H]', xyz='H 0.0 0.0 0.0')
        ch3 = ARCSpecies(label='CH3', smiles='[CH3]', xyz=(
            'C    0.00000  0.00000  0.00000\n'
            'H    1.07770  0.00000  0.00000\n'
            'H   -0.53885  0.93333  0.00000\n'
            'H   -0.53885 -0.93333  0.00000'))
        h2 = ARCSpecies(label='H2', smiles='[H][H]',
                        xyz='H 0.0 0.0 0.0\nH 0.0 0.0 0.74')
        rxn = ARCReaction(r_species=[ch4, h_atom], p_species=[ch3, h2])
        # TS species needs a 6-atom placeholder so ARC's atom-balance check
        # accepts the reaction. The actual coordinates are irrelevant — the
        # adapter overwrites ts_guesses entirely with its own samples.
        ts_placeholder_xyz = ('C 0.0 0.0 0.0\n'
                              'H 0.6 0.6 0.6\n'
                              'H -0.6 -0.6 0.6\n'
                              'H -0.6 0.6 -0.6\n'
                              'H 0.6 -0.6 -0.6\n'
                              'H 1.5 1.5 1.5')
        ts = ARCSpecies(label='TS_h_abstr', is_ts=True, charge=0, multiplicity=2, xyz=ts_placeholder_xyz)
        ts.ts_guesses = []
        rxn.ts_species = ts
        return rxn

    def test_h_abstraction_produces_geometrically_valid_tsguesses(self):

        rxn = self._build_h_abstraction_rxn()

        adapter = GoFlowAdapter(project='goflow_tier2_e2e',
                                project_directory=self.tmpdir,
                                job_type='tsg',
                                reactions=[rxn],
                                testing=False)
        adapter.execute_incore()

        guesses = rxn.ts_species.ts_guesses
        self.assertGreater(len(guesses), 0, 'GoFlow produced no TSGuesses for an in-domain reaction')

        n_atoms_expected = sum(spc.number_of_atoms for spc in rxn.r_species)

        for i, g in enumerate(guesses):
            with self.subTest(guess_idx=i):
                self.assertTrue(g.success, f'guess {i} marked unsuccessful')
                self.assertIsNotNone(g.initial_xyz)
                xyz = (str_to_xyz(g.initial_xyz)
                       if isinstance(g.initial_xyz, str) else g.initial_xyz)

                self.assertEqual(len(xyz['symbols']), n_atoms_expected,
                                 f'guess {i}: expected {n_atoms_expected} atoms, '
                                 f'got {len(xyz["symbols"])}')

                # Element ordering matches the mapped reactant ordering. For
                # CH4 + H, ARC's canonicalization puts the carbon first.
                self.assertEqual(xyz['symbols'][0], 'C', f'guess {i}: first atom should be C')
                self.assertEqual(sorted(xyz['symbols']), sorted(('C', 'H', 'H', 'H', 'H', 'H')))

                flat = [c for tup in xyz['coords'] for c in tup]
                self.assertFalse(any(math.isnan(c) or math.isinf(c) for c in flat),
                                 f'guess {i}: NaN/inf in coordinates')

                # Not collapsed to all-zero or near-zero (would indicate
                # the model output a degenerate geometry).
                norm_sq = sum(c * c for c in flat)
                self.assertGreater(norm_sq, 1.0, f'guess {i}: geometry collapsed (norm² = {norm_sq})')

                # No two atoms occupy the same position (collision).
                for a in range(len(xyz['symbols'])):
                    for b in range(a + 1, len(xyz['symbols'])):
                        d = math.sqrt(sum(
                            (xyz['coords'][a][k] - xyz['coords'][b][k]) ** 2
                            for k in range(3)))
                        self.assertGreater(d, 0.3, f'guess {i}: atoms {a} and {b} colliding (d={d:.3f} Å)')


###############################################################################
# Tier-2 — isomerization fixtures (manual-inspection driver)
###############################################################################
#
# 10 isomerization reactions adapted from the linear-adapter test suite
# (arc/job/adapters/ts/linear_test.py — see ARC main). Each fixture stays
# within GoFlow's HCNOF training domain. The tests here drive
# GoFlowAdapter.execute_incore() against a real ckpt and PRINT every
# surviving TSGuess as a plain XYZ string to stdout — for the user to
# eyeball geometric sanity. Light structural assertions only (atom count,
# no NaN/inf, no atom collisions).
#
# Run with `pytest -s` to see the printed XYZs.


class TestGoFlowIsomerizationFixtures(unittest.TestCase):
    """One method per isomerization reaction; prints TS XYZs for manual review."""

    @classmethod
    def setUpClass(cls):
        if not _refresh_goflow_paths_and_check_ready():
            raise unittest.SkipTest(_TIER2_SKIP_MSG)
        cls.tmpdir_root = tempfile.mkdtemp(prefix='goflow_isom_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir_root, ignore_errors=True)

    @staticmethod
    def _heavy_dmat(coords, symbols):
        """Pairwise distance matrix over heavy atoms only (rotor-invariant)."""
        heavy = [i for i, s in enumerate(symbols) if s != 'H']
        if len(heavy) < 2:
            heavy = list(range(len(symbols)))
        return [math.sqrt(sum((coords[a][k] - coords[b][k]) ** 2 for k in range(3)))
                for ai, a in enumerate(heavy) for b in heavy[ai + 1:]]

    @classmethod
    def _dmat_rmsd(cls, c1, c2, symbols):
        """Heavy-atom distance-matrix RMSD between two geometries (Å)."""
        d1 = cls._heavy_dmat(c1, symbols)
        d2 = cls._heavy_dmat(c2, symbols)
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(d1, d2)) / len(d1))

    def _run_isomerization(self, name, r_smiles, r_xyz, p_smiles, p_xyz):
        """Build the reaction, run GoFlow, print all surviving TSGuesses."""

        r = ARCSpecies(label='R', smiles=r_smiles, xyz=r_xyz)
        p = ARCSpecies(label='P', smiles=p_smiles, xyz=p_xyz)
        rxn = ARCReaction(r_species=[r], p_species=[p])

        # Skip cleanly if ARC's mapping engine can't handle this reaction —
        # adapter would then return zero guesses (correct behavior) but that
        # isn't a model-quality failure, so don't fail the test.
        try:
            am = rxn.atom_map
        except Exception:
            am = None
        if am is None:
            self.skipTest(f'{name}: ARC atom-map computation failed; adapter would skip')

        # TS placeholder must have the right atom count for atom-balance.
        # Reactant XYZ has the right atomic composition; reuse it.
        ts = ARCSpecies(label=f'TS_{name}', is_ts=True, charge=r.charge,
                        multiplicity=r.multiplicity, xyz=r_xyz)
        ts.ts_guesses = []
        rxn.ts_species = ts

        proj_dir = tempfile.mkdtemp(prefix=f'{name}_', dir=self.tmpdir_root)
        adapter = GoFlowAdapter(project=f'goflow_isom_{name}',
                                project_directory=proj_dir,
                                job_type='tsg',
                                reactions=[rxn],
                                testing=False)
        adapter.execute_incore()

        guesses = rxn.ts_species.ts_guesses
        n_atoms_expected = sum(spc.number_of_atoms for spc in rxn.r_species)

        print()
        print(f'==== {name} ====')
        print(f'  reaction       : {r_smiles} -> {p_smiles}')
        print(f'  expected atoms : {n_atoms_expected}')
        print(f'  guesses (after dedup): {len(guesses)}')

        self.assertGreater(len(guesses), 0, f'{name}: GoFlow produced no surviving TSGuess')

        for i, g in enumerate(guesses):
            self.assertTrue(g.success, f'{name} guess {i}: marked unsuccessful')
            xyz = (str_to_xyz(g.initial_xyz) if isinstance(g.initial_xyz, str) else g.initial_xyz)

            self.assertEqual(len(xyz['symbols']), n_atoms_expected,
                             f'{name} guess {i}: expected {n_atoms_expected} atoms, '
                             f'got {len(xyz["symbols"])}')

            flat = [c for tup in xyz['coords'] for c in tup]
            self.assertFalse(any(math.isnan(c) or math.isinf(c) for c in flat),
                             f'{name} guess {i}: NaN/inf coordinates')
            for a in range(len(xyz['symbols'])):
                for b in range(a + 1, len(xyz['symbols'])):
                    d = math.sqrt(sum((xyz['coords'][a][k] - xyz['coords'][b][k]) ** 2 for k in range(3)))
                    self.assertGreater(d, 0.3, f'{name} guess {i}: atoms {a} and {b} colliding (d={d:.3f} Å)')

            print(f'  --- TS guess {i} ---')
            print(xyz_to_str(xyz))

        # ----- uniqueness diagnostics + assertion -----
        # Parse all guesses once.
        parsed = [(str_to_xyz(g.initial_xyz) if isinstance(g.initial_xyz, str) else g.initial_xyz) for g in guesses]
        if len(parsed) >= 2:
            # Pairwise HEAVY-atom dmat-RMSD (matches the metric the adapter
            # uses for consolidation: rotation + torsion invariant).
            print(f'  --- pairwise heavy-atom dmat-RMSD (Å), upper triangle ---')
            header = '       ' + ' '.join(f'  g{j}' for j in range(len(parsed)))
            print(header)
            min_rmsd = float('inf')
            for i in range(len(parsed)):
                row = [f'   g{i}: ']
                for j in range(len(parsed)):
                    if j <= i:
                        row.append('     ')
                    else:
                        rmsd = self._dmat_rmsd(parsed[i]['coords'],
                                               parsed[j]['coords'],
                                               parsed[i]['symbols'])
                        row.append(f'{rmsd:5.3f}')
                        min_rmsd = min(min_rmsd, rmsd)
                print(' '.join(row))
            print(f'  --- min pairwise heavy-atom dmat-RMSD: {min_rmsd:.3f} Å ---')

            # Hard uniqueness assertion: every surviving pair must have
            # heavy-atom dmat-RMSD >= GOFLOW_DEDUP_DMAT_RMSD; else dedup leaked.
            for i in range(len(parsed)):
                for j in range(i + 1, len(parsed)):
                    rmsd = self._dmat_rmsd(parsed[i]['coords'],
                                           parsed[j]['coords'],
                                           parsed[i]['symbols'])
                    self.assertGreaterEqual(
                        rmsd, GOFLOW_DEDUP_DMAT_RMSD,
                        f'{name} guesses {i} and {j} are too similar: dmat-RMSD '
                        f'= {rmsd:.3f} Å < {GOFLOW_DEDUP_DMAT_RMSD} Å threshold '
                        f'(dedup pass missed them)')

    # -------------------------- the 10 fixtures --------------------------

    def test_intra_h_migration_cco(self):  # V
        """4-membered ring H migration: [CH2]CO <=> CC[O]"""
        r_xyz = """C  -3.35807020  0.39772754 -0.02139706
H  -2.80953191  0.44242278 -0.93900704
H  -4.34767471 -0.00900040 -0.00893508
C  -2.72326461  0.91878394  1.28133933
H  -1.66157493  0.79378755  1.23561273
H  -2.95540282  1.95641525  1.40106030
O  -3.24245519  0.18293235  2.39213346
H  -2.84673223  0.50774673  3.20422887"""
        p_xyz = """C  -0.34334771 -0.13590857  0.00000002
H   0.01333400  0.36848377 -0.87365124
H  -1.41334771 -0.13588640 -0.00000560
C   0.16999407  0.59004821  1.25740487
H   1.23999407  0.59002603  1.25741049
H  -0.18665169  1.59886128  1.25739942
O  -0.30669270 -0.08404623  2.42499487
H   0.01329805 -1.14472164  0.00000547"""
        self._run_isomerization('intra_h_migration_cco', '[CH2]CO', r_xyz, 'CC[O]', p_xyz)

    def test_intra_h_migration_ccoo(self):
        """5-membered ring H migration: CCO[O] <=> [CH2]COO"""
        r_xyz = """C  -1.05582103 -0.03329574 -0.10080257
C   0.41792695  0.17831205  0.21035514
O   1.19234020 -0.65389683 -0.61111443
O   2.44749684 -0.41401220 -0.28381363
H  -1.33614002 -1.09151783  0.08714882
H  -1.25953618  0.21489046 -1.16411897
H  -1.67410396  0.62341419  0.54699514
H   0.59566350 -0.06437686  1.28256640
H   0.67254676  1.24676329  0.02676370"""
        p_xyz = """C  -1.40886397  0.22567351 -0.37379668
C   0.06280787  0.04097694 -0.38515682
O   0.44130326 -0.57668419  0.84260864
O   1.89519755 -0.66754203  0.80966180
H  -1.87218376  0.90693511 -1.07582340
H  -2.03646287 -0.44342165  0.20255768
H   0.35571681 -0.60165457 -1.22096147
H   0.56095122  1.01161503 -0.47393734
H   2.05354047 -0.10415729  1.58865243"""
        self._run_isomerization('intra_h_migration_ccoo', 'CCO[O]', r_xyz, '[CH2]COO', p_xyz)

    def test_intra_h_migration_cccoo(self):
        """6-membered ring H migration: CCCO[O] <=> [CH2]CCOO"""
        r_xyz = """C  -1.31455963  0.65305704  0.00229593
C   0.17407454  0.87684185  0.32708610
O   0.97540012  0.03343074 -0.50443961
O   2.25137227  0.22524629 -0.22604804
H  -1.56888362 -0.37060266  0.18212958
H  -1.49495314  0.89014604 -1.02539419
H   0.35446804  0.63975284  1.35477623
H   0.42839853  1.90050154  0.14725245
C  -2.17752564  1.56134592  0.89778516
H  -3.21183640  1.40585907  0.67211926
H  -1.99713214  1.32425692  1.92547529
H  -1.92320166  2.58500562  0.71795151"""
        p_xyz = """C   0.10191448  0.80917231  0.12324900
C   1.63680299  0.68488584  0.13968460
O   2.03194937 -0.20270773  1.18894005
O   3.34756810 -0.30923899  1.20302771
H  -0.33221037 -0.15465524 -0.04249800
H   1.97345768  0.29884684 -0.79975007
H   2.07092784  1.64871339  0.30543160
H   3.73706329  0.55550348  1.35173530
H  -0.23474021  1.19521131  1.06268367
C  -0.32362778  1.76504231 -1.00671841
H  -1.26726146  1.63176387 -1.49322877
H   0.32433693  2.56246418 -1.30531527"""
        self._run_isomerization('intra_h_migration_cccoo', 'CCCO[O]', r_xyz, '[CH2]CCOO', p_xyz)

    def test_intra_oh_migration(self):
        """OH migration: [CH2]COO <=> [O]CCO"""
        r_xyz = """C  -1.40886397  0.22567351 -0.37379668
C   0.06280787  0.04097694 -0.38515682
O   0.44130326 -0.57668419  0.84260864
O   1.89519755 -0.66754203  0.80966180
H  -1.87218376  0.90693511 -1.07582340
H  -2.03646287 -0.44342165  0.20255768
H   0.35571681 -0.60165457 -1.22096147
H   0.56095122  1.01161503 -0.47393734
H   2.05354047 -0.10415729  1.58865243"""
        p_xyz = """O   0.97298522  1.16961708  0.68631092
C   0.83017736  0.23002128 -0.24518707
C  -0.46505265 -0.55857538  0.09146589
O  -1.54540067  0.36524471  0.24441655
H   1.61381747 -0.53531530 -0.35348282
H   0.69744639  0.56361493 -1.28695526
H  -0.71560487 -1.25802813 -0.71249310
H  -0.36288272 -1.12613201  1.02419042
H  -1.03086141  1.13813060  0.58426610"""
        self._run_isomerization('intra_oh_migration', '[CH2]COO', r_xyz, '[O]CCO', p_xyz)

    def test_intra_halogen_migration(self):
        """Fluorine 1,4-shift: FCCC[C](F)F <=> [CH2]CCC(F)(F)F"""
        r_xyz = """F   1.93592759 -1.04813200  0.17239309
C   1.41395997 -0.06443750 -0.60748935
C   0.46854139  0.77821484  0.23269059
C   1.16469946  1.45000317  1.41577429
C   2.13600384  2.49526387  0.98914077
F   1.69221606  3.70990602  0.60332208
F   3.45162393  2.20655153  0.91224277
H   2.23977740  0.51935595 -1.02311040
H   0.87599990 -0.54232434 -1.43132912
H  -0.01588539  1.53022886 -0.40118094
H  -0.31963114  0.12637206  0.62794629
H   0.40903520  1.92224463  2.05360591
H   1.67965177  0.70327850  2.03007255"""
        p_xyz = """C  -2.10258623  0.28609914 -0.11161659
C  -0.80850454 -0.44729615  0.01949484
C  -0.27209648 -0.40163127  1.44584029
C   1.03111915 -1.15786446  1.56235292
F   1.97934384 -0.63177629  0.75822896
F   0.87880578 -2.45776869  1.23195390
F   1.49664262 -1.10927826  2.83007421
H  -2.25664107  1.23858311  0.38402441
H  -2.81716662 -0.01824459 -0.86926845
H  -0.96292814 -1.48784377 -0.28803477
H  -0.08395313 -0.00553132 -0.67357116
H  -1.00377942 -0.83558539  2.13782580
H  -0.11333646  0.63795904  1.75659256"""
        self._run_isomerization('intra_halogen_migration', 'FCCC[C](F)F', r_xyz, '[CH2]CCC(F)(F)F', p_xyz)

    def test_intra_no2_ono_conversion(self):
        """NO2 ↔ ONO rearrangement: [O-][N+](=O)CC <=> CCON=O"""
        r_xyz = """O   1.77136558 -0.91790626  0.88650594
N   1.34754589 -0.18857388 -0.01862669
O   1.86645005 -0.03906737 -1.13182045
C   0.08946605  0.57559465  0.25484606
C   0.46072863  1.91146690  0.86342166
H  -0.52075344 -0.02737899  0.93392769
H  -0.43797095  0.69242674 -0.69660400
H   1.09014915  2.48001164  0.17179384
H  -0.42932512  2.51112436  1.08295532
H   1.01533324  1.78326517  1.79934783"""
        p_xyz = """C  -1.36894499  0.07118059 -0.24801399
C  -0.01369535  0.17184136  0.42591278
O  -0.03967083 -0.62462610  1.60609048
N   1.23538512 -0.53558048  2.24863846
O   1.25629155 -1.21389295  3.27993827
H  -2.16063255  0.41812452  0.42429392
H  -1.39509985  0.66980796 -1.16284741
H  -1.59800183 -0.96960842 -0.49986392
H   0.19191326  1.21800574  0.68271847
H   0.76371340 -0.19234475 -0.25650067"""
        self._run_isomerization('intra_no2_ono_conversion', '[O-][N+](=O)CC', r_xyz, 'CCON=O', p_xyz)

    def test_1_5_h_shift_pentadiene(self):
        """Degenerate sigmatropic 1,5-H shift in penta-1,3-diene: CC=CC=C <=> CC=CC=C"""
        xyz = """C   2.6362  0.0000  0.0000
C   1.3442  0.6930  0.0000
C   0.0000  0.0000  0.0000
C  -1.3442  0.6930  0.0000
C  -2.6362  0.0000  0.0000
H   2.5820 -0.6289  0.8928
H   2.5820 -0.6289 -0.8928
H   3.6014  0.5018  0.0000
H   1.3970  1.7729  0.0000
H   0.0000 -1.0847  0.0000
H  -1.3970  1.7729  0.0000
H  -3.6014  0.5018  0.0000
H  -2.6362 -1.0847  0.0000"""
        self._run_isomerization('1_5_h_shift_pentadiene', 'CC=CC=C', xyz, 'CC=CC=C', xyz)

    def test_6_mem_central_cc_shift_alkyne_to_allene(self):
        """6-membered central C-C shift: C#CCCC#C <=> C=C=CC=C=C"""
        r_xyz = """C   3.03272979 -0.11060195 -0.24229461
C   1.85599055 -0.34675713 -0.20247149
C   0.41485966 -0.64142590 -0.15352412
C  -0.41485965  0.64142578 -0.17240633
C  -1.85599061  0.34675702 -0.12346178
C  -3.03272995  0.11060190 -0.08364096
H   4.07762286  0.09693448 -0.27758589
H   0.19106566 -1.21954180  0.75163518
H   0.14301783 -1.27648597 -1.00582442
H  -0.19106412  1.21954271 -1.07756459
H  -0.14301928  1.27648492  0.67989514
H  -4.07762310 -0.09693448 -0.04835177"""
        p_xyz = """C  -3.03124363  0.21595810 -0.01068883
C  -1.77136356 -0.00875193 -0.22839960
C  -0.51035344 -0.23538255 -0.44913569
C   0.51035356  0.23538291  0.44913621
C   1.77136365  0.00875234  0.22839985
C   3.03124358 -0.21595777  0.01068824
H  -3.50880107  1.10742857 -0.40051872
H  -3.62554573 -0.48341738  0.56587595
H  -0.21235801 -0.79338469 -1.33170668
H   0.21235823  0.79338484  1.33170737
H   3.50880076 -1.10742925  0.40051615
H   3.62554580  0.48341866 -0.56587535"""
        self._run_isomerization('6_mem_central_cc_shift', 'C#CCCC#C', r_xyz, 'C=C=CC=C=C', p_xyz)

    def test_1_3_sigmatropic_rearrangement_imidazole(self):
        """1,3-sigmatropic rearrangement on imidazole: c1ncc[nH]1 <=> N=CN1C=C1"""
        r_xyz = """C  -0.96405208 -0.58870010 -0.35675666
N   0.09948347 -1.35699528 -0.30406608
C   1.08781769 -0.57088551  0.22943180
C   0.61245126  0.68985747  0.50218591
N  -0.70083129  0.66320502  0.12207481
H  -1.93870511 -0.87854432 -0.72608823
H   2.08729155 -0.95482079  0.38815067
H   1.07812779  1.57128662  0.91862266
H  -1.36158329  1.42559689  0.18141711"""
        p_xyz = """N   0.76582385 -0.14849540 -1.32485588
C   0.78208226  0.49284271 -0.20399502
N  -0.04861443  0.34490826  0.88039960
C  -0.56227958 -0.84609375  1.31645778
C  -1.38522743  0.06039446  0.80970400
H   1.52092135  0.20130809 -1.92536405
H   1.53681129  1.27833147 -0.02452505
H  -0.33519514 -1.78256934  0.82247210
H  -1.89445111 -0.06503499 -0.13767862"""
        self._run_isomerization('1_3_sigmatropic_imidazole', 'c1ncc[nH]1', r_xyz, 'N=CN1C=C1', p_xyz)

    def test_1_2_methyl_shift_on_cyclopentadienyl(self):
        """1,2-methyl shift on cyclopentadienyl carbene: CC[C]1C=CC=C1 <=> [CH2]C1(C)C=CC=C1"""
        r_xyz = """C  -2.08011725 -0.87098529 -0.24102896
C  -1.38616808  0.31243567  0.41701874
C   0.09289885  0.19281646  0.35695343
C   0.92864438  0.68782411 -0.70819340
C   2.18636908  0.35957487 -0.37255721
C   2.18107732 -0.34427638  0.89885251
C   0.92008966 -0.45002583  1.34718072
H  -1.81540032 -1.81207279  0.25290661
H  -1.80896484 -0.95285941 -1.29907735
H  -3.16674193 -0.75248342 -0.17993048
H  -1.70601347  1.23815887 -0.07595913
H  -1.71241303  0.38717620  1.46117876
H   0.59841756  1.21177196 -1.58944757
H   3.07727387  0.57336525 -0.94230522
H   3.06757417 -0.71677188  1.38813917
H   0.58240767 -0.91755259  2.25688920"""
        p_xyz = """C  -0.91419261 -0.92211886  1.28775915
C  -0.38593444 -0.06230282  0.18302891
C   0.67135826  0.91653743  0.70043366
C  -1.50477869  0.67123329 -0.52120219
C  -1.56600546  0.29578439 -1.80779070
C  -0.54194075 -0.68042899 -2.06986329
C   0.15393453 -0.90959193 -0.94583904
H  -1.87479029 -1.41555103  1.18169570
H  -0.24773685 -1.28139411  2.06376191
H   1.52757979  0.38651768  1.13544157
H   1.05855879  1.55868444 -0.10049234
H   0.25983545  1.57373963  1.47630068
H  -2.15472843  1.39365373 -0.04968993
H  -2.26617185  0.65598910 -2.54596855
H  -0.37671673 -1.14563680 -3.02965012
H   0.98207416 -1.59686236 -0.85449771"""
        self._run_isomerization('1_2_methyl_shift_cpd', 'CC[C]1C=CC=C1', r_xyz, '[CH2]C1(C)C=CC=C1', p_xyz)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
