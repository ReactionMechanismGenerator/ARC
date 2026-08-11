#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the ring_pucker_seed_conformers() function in
arc.species.conformers, and of the ring_pucker.ring_mean_plane() helper it relies on.
"""

import copy
import unittest
from unittest.mock import patch

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import arc.species.conformers as conformers
import arc.species.converter as converter
import arc.species.ring_pucker as ring_pucker
from arc.exceptions import ConformerError
from arc.molecule.molecule import Molecule
from arc.species.perceive import perceive_molecule_from_xyz


def build_seed_geometry(smiles, seed=42):
    """Build an RMG Molecule and a matching 3D xyz dict for ``smiles`` via RDKit.

    Args:
        smiles (str): The SMILES string of the molecule.
        seed (int): The RDKit embedding random seed.

    Returns:
        tuple: (mol, xyz, rd_mol) where ``mol`` is an RMG Molecule and ``xyz`` is an xyz dict,
            with atoms in the same order as ``mol.atoms`` and ``rd_mol``.
    """
    rd_mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(rd_mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(rd_mol)
    conf = rd_mol.GetConformer()
    coords = [tuple(conf.GetAtomPosition(i)) for i in range(rd_mol.GetNumAtoms())]
    symbols = tuple(atom.GetSymbol() for atom in rd_mol.GetAtoms())
    xyz = converter.xyz_from_data(coords=coords, symbols=symbols)
    mol = perceive_molecule_from_xyz(xyz)
    for i, atom in enumerate(mol.atoms):
        assert atom.element.symbol == symbols[i], \
            f'RMG Molecule atom order does not match xyz/RDKit atom order at index {i}.'
    return mol, xyz, rd_mol


def _conformers_without_dmat(conformer_list):
    """Strip the cached ``dmat`` numpy array (a dedup-distance cache, not part of conformer
    identity) so conformer dicts can be compared with ``assertEqual``.

    Args:
        conformer_list (list): A list of conformer dicts, possibly containing a ``dmat`` key.

    Returns:
        list: The same conformer dicts, each without a ``dmat`` key.
    """
    return [{key: value for key, value in conformer.items() if key != 'dmat'} for conformer in conformer_list]


class TestRingPuckerSeedConformers(unittest.TestCase):
    """
    Contains unit tests for ring_pucker_seed_conformers() and ring_pucker.ring_mean_plane().
    """

    def test_cyclohexane_chair_global_min_recovered_from_any_base(self):
        """The chair global minimum must be recovered regardless of the base geometry's basin.

        Seed 0 embeds/MMFF-optimizes to a chair base geometry; seed 42 embeds/MMFF-optimizes to a
        twist-boat base geometry. An unconstrained-relaxation seeding scheme erases the raw
        z-morphed pucker and every seed collapses back into the base geometry's own basin, so from
        a twist-boat base the chair global minimum is missed entirely. This test must pass from
        BOTH bases.
        """
        for seed in (0, 42):
            mol, xyz, rd_mol = build_seed_geometry('C1CCCCC1', seed=seed)
            ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

            xyzs, energies = conformers.ring_pucker_seed_conformers(
                label='cyclohexane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)
            self.assertTrue(len(xyzs) > 0, f'No conformers returned from seed={seed} base geometry.')

            found_chair = False
            for conf_xyz in xyzs:
                ring_coords = np.array(conf_xyz['coords'])[ring_atom_indices]
                if ring_pucker.classify_pucker(ring_coords) == 'chair':
                    amplitude = ring_pucker.puckering_amplitude(ring_coords)
                    if 0.5 <= amplitude <= 0.65:
                        found_chair = True
                        break
            self.assertTrue(found_chair, f'Chair global minimum not recovered from seed={seed} base geometry.')

    def test_cyclohexane_covers_multiple_puckers(self):
        """The returned polished set (from a chair base) should span multiple pucker states."""
        mol, xyz, rd_mol = build_seed_geometry('C1CCCCC1', seed=0)
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        xyzs, energies = conformers.ring_pucker_seed_conformers(
            label='cyclohexane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)

        labels = set()
        for conf_xyz in xyzs:
            ring_coords = np.array(conf_xyz['coords'])[ring_atom_indices]
            labels.add(ring_pucker.classify_pucker(ring_coords))
        self.assertGreaterEqual(len(labels), 2, f'Expected at least 2 distinct pucker labels, got {labels}.')
        self.assertIn('chair', labels)
        self.assertIn('twist-boat', labels)

    def test_returns_energies_paired_with_xyzs(self):
        """Returned energies must be paired 1:1 with xyzs, finite, and include a low-energy chair."""
        mol, xyz, rd_mol = build_seed_geometry('C1CCCCC1', seed=0)
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        xyzs, energies = conformers.ring_pucker_seed_conformers(
            label='cyclohexane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)

        self.assertEqual(len(xyzs), len(energies))
        self.assertGreater(len(energies), 0)
        for energy in energies:
            self.assertIsInstance(energy, float)
            self.assertTrue(np.isfinite(energy))

        chair_energies = list()
        for conf_xyz, energy in zip(xyzs, energies):
            ring_coords = np.array(conf_xyz['coords'])[ring_atom_indices]
            if ring_pucker.classify_pucker(ring_coords) == 'chair':
                chair_energies.append(energy)
        self.assertGreater(len(chair_energies), 0)
        self.assertAlmostEqual(min(chair_energies), min(energies), delta=1e-6)

    def test_raw_seed_count_matches_full_phase_wheel_6_ring(self):
        """Every entry of the 14-point 6-ring phase wheel must produce exactly one raw seed.

        ``optimize_conformer_with_frozen_ring`` is monkeypatched to pass its input straight
        through (energy 0.0), isolating the raw-seed generation loop from FF-relaxation collapse,
        so the returned count/labels reflect ``canonical_pucker_wheel`` directly.
        """
        mol, xyz, rd_mol = build_seed_geometry('C1CCCCC1', seed=0)
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        with patch('arc.species.conformers.optimize_conformer_with_frozen_ring',
                  side_effect=lambda mol, seed_xyz, ring_idx, force_field='MMFF94s': (seed_xyz, 0.0)):
            xyzs, energies = conformers.ring_pucker_seed_conformers(
                label='cyclohexane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)

        wheel = ring_pucker.canonical_pucker_wheel(6)
        self.assertEqual(len(xyzs), len(wheel))
        self.assertEqual(len(energies), len(wheel))

        labels = [ring_pucker.classify_pucker(np.array(conf_xyz['coords'])[ring_atom_indices])
                 for conf_xyz in xyzs]
        self.assertEqual(labels.count('chair'), 2)
        self.assertEqual(labels.count('boat'), 6)
        self.assertEqual(labels.count('twist-boat'), 6)

    def test_raw_seed_count_matches_full_phase_wheel_5_ring(self):
        """Every entry of the 20-point 5-ring phase wheel must produce exactly one raw seed."""
        mol, xyz, rd_mol = build_seed_geometry('C1CCCC1', seed=0)  # cyclopentane
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        with patch('arc.species.conformers.optimize_conformer_with_frozen_ring',
                  side_effect=lambda mol, seed_xyz, ring_idx, force_field='MMFF94s': (seed_xyz, 0.0)):
            xyzs, energies = conformers.ring_pucker_seed_conformers(
                label='cyclopentane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)

        wheel = ring_pucker.canonical_pucker_wheel(5)
        self.assertEqual(len(xyzs), len(wheel))
        self.assertEqual(len(energies), len(wheel))

        labels = [ring_pucker.classify_pucker(np.array(conf_xyz['coords'])[ring_atom_indices])
                 for conf_xyz in xyzs]
        self.assertEqual(labels.count('envelope'), 10)
        self.assertEqual(labels.count('twist'), 10)

    def test_methyltetrahydrofuran_seeding_smoke(self):
        """Smoke test (not a proof): a substituted 5-ring should survive seeding with more than
        one distinct phase-resolved pucker state among its polished conformers.

        This is deliberately lenient. Empirically, the envelope family is an FF saddle point for
        this substituted ring (mirroring boat being a saddle for 6-rings): all 20 raw wheel seeds
        relax into the twist family regardless of base seed, so ``classify_pucker``'s coarse label
        alone never shows 2 distinct labels here. Using the phase-resolved ``pucker_state_id``
        instead checks that the denser wheel still yields more than one distinct twist phase bin,
        rather than collapsing to a single point.
        """
        mol, xyz, rd_mol = build_seed_geometry('CC1CCCO1', seed=0)  # 2-methyltetrahydrofuran
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        xyzs, energies = conformers.ring_pucker_seed_conformers(
            label='2-methyltetrahydrofuran', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)
        self.assertGreater(len(xyzs), 0)

        state_ids = {ring_pucker.pucker_state_id(np.array(conf_xyz['coords'])[ring_atom_indices])
                    for conf_xyz in xyzs}
        self.assertGreaterEqual(len(state_ids), 2,
                                f'Expected at least 2 distinct phase-resolved pucker states, got {state_ids}.')

    def test_substituted_5_ring_raw_seeds_span_full_phase_wheel(self):
        """The RAW (pre-optimization) seeds for a SUBSTITUTED 5-ring must span all 20 phase bins
        of the Cremer-Pople pucker wheel, proving the seeder installs the full wheel for a
        substituted ring rather than only for symmetric cyclopentane.

        ``optimize_conformer_with_frozen_ring`` is monkeypatched to pass its input straight
        through (``(seed_xyz, 0.0)``), so ``xyzs`` here are the raw wheel-displaced seeds
        themselves, not FF-relaxed geometries; this isolates the seeding/wheel-installation step
        from any downstream FF-relaxation collapse (see the lenient
        ``test_methyltetrahydrofuran_seeding_smoke`` above, which instead checks the polished,
        post-optimization output and is deliberately tolerant of relaxation collapsing distinct
        raw phases into a shared basin).

        This proves the seeder emits all 20 phase bins at the raw-seed installation stage; it
        does NOT prove a substituted 5-ring recovers a non-base global-minimum pucker after real
        force-field optimization plus the count cap (no reference method exists for that inside a
        unit test; that broader claim is covered by a separate smoke test exercising
        post-FF behavior).
        """
        mol, xyz, rd_mol = build_seed_geometry('CC1CCCO1', seed=0)  # 2-methyltetrahydrofuran
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        with patch('arc.species.conformers.optimize_conformer_with_frozen_ring',
                  side_effect=lambda mol, seed_xyz, ring_idx, force_field='MMFF94s': (seed_xyz, 0.0)):
            xyzs, energies = conformers.ring_pucker_seed_conformers(
                label='2-methyltetrahydrofuran', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)

        wheel = ring_pucker.canonical_pucker_wheel(5)
        self.assertEqual(len(xyzs), len(wheel))

        state_ids = {ring_pucker.pucker_state_id(np.array(conf_xyz['coords'])[ring_atom_indices])
                    for conf_xyz in xyzs}
        self.assertEqual(len(state_ids), len(wheel),
                         f'Expected raw seeds to span all {len(wheel)} phase bins, got {len(state_ids)}: '
                         f'{state_ids}.')

    def test_fused_ring_is_gated_out(self):
        """Fused-ring systems (e.g., decalin) must be hard-gated out, returning ([], [])."""
        mol, xyz, rd_mol = build_seed_geometry('C1CCC2CCCCC2C1', seed=0)  # decalin
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        xyzs, energies = conformers.ring_pucker_seed_conformers(
            label='decalin', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)
        self.assertEqual(xyzs, [])
        self.assertEqual(energies, [])

    def test_acyclic_and_wrong_size_return_empty(self):
        """Wrong ring sizes and acyclic atom selections must both yield empty lists."""
        mol7, xyz7, rd_mol7 = build_seed_geometry('C1CCCCCC1', seed=0)  # cycloheptane
        ring_atom_indices_7 = list(rd_mol7.GetRingInfo().AtomRings()[0])
        self.assertEqual(len(ring_atom_indices_7), 7)

        xyzs, energies = conformers.ring_pucker_seed_conformers(
            label='cycloheptane', mol=mol7, ring_atom_indices=ring_atom_indices_7, base_xyz=xyz7)
        self.assertEqual((xyzs, energies), ([], []))

        mol_chain, xyz_chain, _ = build_seed_geometry('CCCCCC', seed=0)  # hexane, acyclic
        chain_indices = [atom_index for atom_index, atom in enumerate(mol_chain.atoms)
                         if atom.element.symbol == 'C']
        self.assertEqual(len(chain_indices), 6)

        xyzs, energies = conformers.ring_pucker_seed_conformers(
            label='hexane', mol=mol_chain, ring_atom_indices=chain_indices, base_xyz=xyz_chain)
        self.assertEqual((xyzs, energies), ([], []))

    def test_atom_order_mismatch_raises(self):
        """A base_xyz whose symbols are permuted relative to mol.atoms must raise ConformerError."""
        mol, xyz, rd_mol = build_seed_geometry('C1CCCCC1', seed=0)
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        bad_symbols = list(xyz['symbols'])
        h_index = bad_symbols.index('H')
        bad_symbols[0], bad_symbols[h_index] = bad_symbols[h_index], bad_symbols[0]
        bad_xyz = converter.xyz_from_data(coords=xyz['coords'], symbols=bad_symbols)

        with self.assertRaises(ConformerError):
            conformers.ring_pucker_seed_conformers(
                label='cyclohexane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=bad_xyz)

    def test_substituent_bonds_preserved_by_morph(self):
        """The methyl substituent must stay bonded to its ring carbon in every returned conformer."""
        mol, xyz, rd_mol = build_seed_geometry('CC1CCCCC1', seed=0)  # methylcyclohexane
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])
        ring_set = set(ring_atom_indices)

        methyl_c_index = None
        methyl_anchor_index = None
        for atom_index, atom in enumerate(mol.atoms):
            if atom_index in ring_set:
                continue
            if atom.element.symbol != 'C':
                continue
            for neighbor in atom.edges.keys():
                neighbor_index = mol.atoms.index(neighbor)
                if neighbor_index in ring_set:
                    methyl_c_index = atom_index
                    methyl_anchor_index = neighbor_index
                    break
            if methyl_c_index is not None:
                break
        self.assertIsNotNone(methyl_c_index)

        xyzs, energies = conformers.ring_pucker_seed_conformers(
            label='methylcyclohexane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)
        self.assertGreater(len(xyzs), 0)

        for conf_xyz in xyzs:
            coords = np.array(conf_xyz['coords'])
            bond_length = np.linalg.norm(coords[methyl_c_index] - coords[methyl_anchor_index])
            self.assertLess(bond_length, 1.7)

    def test_returned_conformers_are_converged(self):
        """Every returned conformer must already sit at a force-field-converged geometry."""
        mol, xyz, rd_mol = build_seed_geometry('C1CCCCC1', seed=0)
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        xyzs, energies = conformers.ring_pucker_seed_conformers(
            label='cyclohexane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)
        self.assertGreater(len(xyzs), 0)

        for conf_xyz, energy in zip(xyzs, energies):
            _, check_rd_mol = converter.rdkit_conf_from_mol(mol, conf_xyz)
            mol_properties = AllChem.MMFFGetMoleculeProperties(check_rd_mol, mmffVariant='MMFF94s')
            ff = AllChem.MMFFGetMoleculeForceField(check_rd_mol, mol_properties)
            energy_before = ff.CalcEnergy()
            AllChem.MMFFOptimizeMolecule(check_rd_mol, mmffVariant='MMFF94s', maxIters=500)
            ff_after = AllChem.MMFFGetMoleculeForceField(check_rd_mol, mol_properties)
            energy_after = ff_after.CalcEnergy()
            self.assertAlmostEqual(energy_before, energy_after, delta=1e-3,
                                   msg=f'Returned conformer was not converged: {energy_before} -> {energy_after}')

    def test_frozen_ring_drops_seed_when_stage_two_could_not_be_set_up(self):
        """RDKit's ForceField.Minimize() convention is v==1 not-converged / v==0 converged /
        v==-1 could-not-set-up. A v==-1 stage-2 result means the force field never actually ran,
        so its CalcEnergy() must not be trusted as a converged geometry/energy; the seed must be
        dropped (None returned), just like the v==1 not-converged case."""
        mol, xyz, rd_mol = build_seed_geometry('C1CCCCC1', seed=0)
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        real_get_ff = AllChem.MMFFGetMoleculeForceField
        call_count = {'n': 0}

        class _CouldNotSetUpForceField:
            """Proxies a real RDKit ForceField but forces Minimize() to report v==-1."""

            def __init__(self, real_ff):
                self._real_ff = real_ff

            def Minimize(self, *args, **kwargs):
                return -1

            def __getattr__(self, name):
                return getattr(self._real_ff, name)

        def fake_get_ff(*args, **kwargs):
            call_count['n'] += 1
            real_ff = real_get_ff(*args, **kwargs)
            if call_count['n'] == 1:
                return real_ff
            return _CouldNotSetUpForceField(real_ff)

        with patch('arc.species.conformers.AllChem.MMFFGetMoleculeForceField', side_effect=fake_get_ff):
            result = conformers.optimize_conformer_with_frozen_ring(mol, xyz, ring_atom_indices)
        self.assertIsNone(result)

    def test_no_boat_saddle_survives_seeding(self):
        """A symmetric boat seed sits exactly on the boat<->twist-boat saddle; MMFF's Minimize()
        reports spurious zero-gradient convergence there. Stage-2 polishing must symmetry-break
        the geometry before minimizing so no returned conformer is stuck at the boat saddle."""
        mol, xyz, rd_mol = build_seed_geometry('C1CCCCC1', seed=0)
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        xyzs, energies = conformers.ring_pucker_seed_conformers(
            label='cyclohexane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)
        self.assertGreater(len(xyzs), 0)

        labels = list()
        for conf_xyz in xyzs:
            ring_coords = np.array(conf_xyz['coords'])[ring_atom_indices]
            labels.append(ring_pucker.classify_pucker(ring_coords))
        self.assertNotIn('boat', labels, f'A boat saddle point survived polishing: {labels}.')

    def test_wrong_cyclic_order_raises(self):
        """Ring indices permuted out of connectivity order must raise ConformerError."""
        mol, xyz, rd_mol = build_seed_geometry('C1CCCCC1', seed=0)
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])
        permuted_indices = [ring_atom_indices[i] for i in (0, 2, 4, 1, 3, 5)]

        with self.assertRaises(ConformerError):
            conformers.ring_pucker_seed_conformers(
                label='cyclohexane', mol=mol, ring_atom_indices=permuted_indices, base_xyz=xyz)

    def test_partial_amplitudes_raises(self):
        """A partial amplitude map missing required pucker labels must raise ConformerError."""
        mol, xyz, rd_mol = build_seed_geometry('C1CCCCC1', seed=0)
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        with self.assertRaises(ConformerError):
            conformers.ring_pucker_seed_conformers(
                label='cyclohexane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz,
                amplitudes={'chair': 0.6})

    def test_ring_mean_plane_matches_cp_z(self):
        """ring_mean_plane()'s z-displacements should reproduce the CP puckering amplitude."""
        ring_coords = ring_pucker.ideal_pucker_geometry(6, 'chair')
        centroid, normal, z = ring_pucker.ring_mean_plane(ring_coords)
        self.assertEqual(centroid.shape, (3,))
        self.assertEqual(normal.shape, (3,))
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=6)

        amplitude_from_plane = float(np.sqrt(np.sum(z ** 2)))
        amplitude_from_cp = ring_pucker.puckering_amplitude(ring_coords)
        self.assertAlmostEqual(amplitude_from_plane, amplitude_from_cp, places=6)

    def test_displace_ring_pucker_phase_deg_none_matches_omitted_phase(self):
        """``_displace_ring_pucker(..., phase_deg=None, ...)`` must produce byte-identical
        coordinates to the equivalent construction that omits ``phase_deg`` entirely from
        ``ring_pucker.ideal_pucker_geometry``, proving the reduced cross-product path (which
        always passes ``phase_deg=None`` for non-equatorial labels) is equivalent to the older
        no-``phase_deg`` call style rather than silently diverging.
        """
        mol, xyz, rd_mol = build_seed_geometry('C1CCCCC1', seed=0)
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])
        coords = np.array(xyz['coords'])
        atom_to_index = {id(atom): i for i, atom in enumerate(mol.atoms)}

        plan = conformers._ring_pucker_plan('cyclohexane', mol, ring_atom_indices, coords, atom_to_index)
        self.assertIsNotNone(plan)

        amplitude_map = ring_pucker.DEFAULT_PUCKER_AMPLITUDES
        label_state = 'chair'
        pole_sign = 1

        coords_via_none = conformers._displace_ring_pucker(
            coords, plan, label_state, None, pole_sign, amplitude_map)

        q = pole_sign * amplitude_map[label_state]
        z_target = ring_pucker.ideal_pucker_geometry(plan['n'], label_state, amplitude=q)[:, 2]
        expected_coords = coords.copy()
        for ring_position, atom_idx in enumerate(plan['ring_idx']):
            delta_p = (z_target[ring_position] - plan['z_real'][ring_position]) * plan['normal']
            expected_coords[atom_idx] = expected_coords[atom_idx] + delta_p
            for other_atom_idx, anchor_p in plan['anchor_of'].items():
                if anchor_p == ring_position:
                    expected_coords[other_atom_idx] = expected_coords[other_atom_idx] + delta_p

        np.testing.assert_array_equal(coords_via_none, expected_coords)


def build_base_conformers(smiles, seed=0):
    """Build a one-entry base conformer list (as ``generate_conformers`` would hold internally).

    Args:
        smiles (str): The SMILES string of the molecule.
        seed (int): The RDKit embedding random seed.

    Returns:
        tuple: (mol, base_conformers) where ``mol`` is an RMG Molecule and ``base_conformers``
            is a list with a single conformer dictionary with 'xyz' and 'FF energy' keys.
    """
    mol, xyz, _ = build_seed_geometry(smiles, seed=seed)
    base_conformers = [{'xyz': xyz, 'index': 0, 'FF energy': 0.0, 'source': 'test'}]
    return mol, base_conformers


class TestRingPuckerBaseConformers(unittest.TestCase):
    """
    Contains unit tests for ring_pucker_base_conformers() and ring_is_saturated().
    """

    def test_acyclic_molecule_returns_empty(self):
        """An acyclic molecule (n-butane) with a non-empty base list must return an empty list."""
        mol, base_conformers = build_base_conformers('CCCC', seed=0)
        result = conformers.ring_pucker_base_conformers(label='butane', mol=mol,
                                                         base_conformers=base_conformers)
        self.assertEqual(result, [])

    def test_cyclohexane_returns_base_conformers(self):
        """Cyclohexane, given one base conformer, must return a non-empty list of well-formed dicts."""
        mol, base_conformers = build_base_conformers('C1CCCCC1', seed=0)
        result = conformers.ring_pucker_base_conformers(label='cyclohexane', mol=mol,
                                                         base_conformers=base_conformers)
        self.assertGreater(len(result), 0)
        num_atoms = len(mol.atoms)
        for conf in result:
            self.assertIn('xyz', conf)
            self.assertIn('FF energy', conf)
            self.assertIsInstance(conf['FF energy'], float)
            self.assertEqual(conf['source'], 'ring pucker')
            self.assertEqual(len(conf['xyz']['coords']), num_atoms)

    def test_multiple_bases_stay_within_the_pucker_base_cap_and_recover_both_chairs(self):
        """Seeding cyclohexane (an unsubstituted, symmetric ring) from a second base conformer
        keeps the result bounded by ``PUCKER_MAX_BASES`` and still recovers both chair basins,
        even though the surviving COUNT is not monotonic in the number of base conformers.

        With ``PUCKER_MAX_BASES=20`` a single base's raw seeds already span the full 14-entry
        6-ring Cremer-Pople phase wheel with no truncation (``get_lowest_confs`` is called with
        ``e=None``, so only the count cap applies): every one of the 14 phase-distinct geometries
        survives, i.e. ``len(result_one) == 14``. Adding a second base conformer roughly doubles
        the raw candidate pool (2 x 14), but ``get_lowest_confs``'s dedup is a greedy,
        insertion-order-dependent distance-DISTANCE comparison (not a phase-identity comparison,
        see FIX C docstring notes) -- interleaving a second base's geometries by FF energy can
        cause the greedy scan to reject candidates it would otherwise have kept, so
        ``len(result_two)`` can legitimately be SMALLER than ``len(result_one)`` (empirically 8
        here). This is a pre-existing property of the distance-based dedup, not something this
        fix changes, so a strict "more bases never yield fewer results" invariant does not hold;
        this test instead pins the actual observed counts and re-asserts the property that
        matters: both chair basins are still recovered in both cases.
        """
        mol, xyz, _ = build_seed_geometry('C1CCCCC1', seed=0)
        _, xyz_alt, _ = build_seed_geometry('C1CCCCC1', seed=1)
        base_one = [{'xyz': xyz, 'index': 0, 'FF energy': 0.0, 'source': 'test'}]
        base_two = base_one + [{'xyz': xyz_alt, 'index': 1, 'FF energy': 0.0, 'source': 'test'}]

        result_one = conformers.ring_pucker_base_conformers(label='cyclohexane', mol=mol,
                                                             base_conformers=base_one)
        result_two = conformers.ring_pucker_base_conformers(label='cyclohexane', mol=mol,
                                                             base_conformers=base_two)
        self.assertLessEqual(len(result_one), conformers.PUCKER_MAX_BASES)
        self.assertLessEqual(len(result_two), conformers.PUCKER_MAX_BASES)
        self.assertEqual(len(result_one), 14)
        self.assertEqual(len(result_two), 8)

        atom_to_index = {id(atom): i for i, atom in enumerate(mol.atoms)}
        ring = [ring for ring in mol.get_deterministic_sssr() if len(ring) == 6][0]
        ring_indices = [atom_to_index[id(atom)] for atom in ring]
        for result in (result_one, result_two):
            labels = {ring_pucker.classify_pucker(np.array(conf['xyz']['coords'])[ring_indices])
                      for conf in result}
            self.assertIn('chair', labels)

    def test_many_base_conformers_are_capped(self):
        """The per-ring fallback path (the only path a monocyclic molecule can take) must be
        bounded by the same caps as the cross-product path: only the lowest-FF-energy
        PUCKER_MAX_CROSS_BASE_CONFORMERS bases are seeded, and the combined result is capped at
        PUCKER_MAX_BASES, so it cannot become an uncapped, unbounded seeding path."""
        mol, xyz, _ = build_seed_geometry('C1CCCCC1', seed=0)
        base_conformers = [{'xyz': xyz, 'index': i, 'FF energy': float(i), 'source': 'test'}
                           for i in range(20)]

        real_seed_conformers = conformers.ring_pucker_seed_conformers
        with patch('arc.species.conformers.ring_pucker_seed_conformers',
                   side_effect=real_seed_conformers) as mock_seed:
            result = conformers.ring_pucker_base_conformers(label='cyclohexane', mol=mol,
                                                             base_conformers=base_conformers)

        self.assertLessEqual(len(result), conformers.PUCKER_MAX_BASES)
        # A single ring means one ring_pucker_seed_conformers() call per seeded base.
        self.assertLessEqual(mock_seed.call_count, conformers.PUCKER_MAX_CROSS_BASE_CONFORMERS)

    def test_benzene_aromatic_ring_returns_empty(self):
        """An aromatic ring (benzene) must be skipped, returning an empty list."""
        mol, base_conformers = build_base_conformers('c1ccccc1', seed=0)
        result = conformers.ring_pucker_base_conformers(label='benzene', mol=mol,
                                                         base_conformers=base_conformers)
        self.assertEqual(result, [])

    def test_cyclohexene_unsaturated_ring_returns_empty(self):
        """A ring with an in-ring C=C double bond (cyclohexene) must be skipped."""
        mol, base_conformers = build_base_conformers('C1=CCCCC1', seed=0)
        result = conformers.ring_pucker_base_conformers(label='cyclohexene', mol=mol,
                                                         base_conformers=base_conformers)
        self.assertEqual(result, [])

    def test_decalin_fused_ring_returns_empty(self):
        """A fused bicyclic system (decalin) must return an empty list."""
        mol, base_conformers = build_base_conformers('C1CCC2CCCCC2C1', seed=0)
        result = conformers.ring_pucker_base_conformers(label='decalin', mol=mol,
                                                         base_conformers=base_conformers)
        self.assertEqual(result, [])

    def test_empty_base_conformers_returns_empty(self):
        """An empty base_conformers list must short-circuit to an empty list."""
        mol, _ = build_base_conformers('C1CCCCC1', seed=0)
        result = conformers.ring_pucker_base_conformers(label='cyclohexane', mol=mol, base_conformers=[])
        self.assertEqual(result, [])

    def test_chirality_propagated_to_pucker_bases(self):
        """A 'chirality' key on a base conformer dict must be copied onto every returned dict."""
        mol, xyz, _ = build_seed_geometry('C1CCCCC1', seed=0)
        chirality = {0: 'R'}
        base_conformers = [{'xyz': xyz, 'index': 0, 'FF energy': 0.0, 'source': 'test', 'chirality': chirality}]
        result = conformers.ring_pucker_base_conformers(label='cyclohexane', mol=mol,
                                                         base_conformers=base_conformers)
        self.assertGreater(len(result), 0)
        for conf in result:
            self.assertIn('chirality', conf)
            self.assertEqual(conf['chirality'], chirality)

    def test_ring_is_saturated(self):
        """ring_is_saturated() must be True for cyclohexane, False for benzene and cyclohexene."""
        mol, _ = build_base_conformers('C1CCCCC1', seed=0)
        sssr = mol.get_deterministic_sssr()
        self.assertEqual(len(sssr), 1)
        self.assertTrue(conformers.ring_is_saturated(mol, sssr[0]))

        mol_benzene, _ = build_base_conformers('c1ccccc1', seed=0)
        sssr_benzene = mol_benzene.get_deterministic_sssr()
        self.assertFalse(conformers.ring_is_saturated(mol_benzene, sssr_benzene[0]))

        mol_ene, _ = build_base_conformers('C1=CCCCC1', seed=0)
        sssr_ene = mol_ene.get_deterministic_sssr()
        self.assertFalse(conformers.ring_is_saturated(mol_ene, sssr_ene[0]))


class TestRingPuckerIntegration(unittest.TestCase):
    """
    Integration tests confirming that ring-pucker base geometries flow through
    deduce_new_conformers() / generate_conformers() and that the chair global
    minimum is recovered for cyclohexane.
    """

    def test_cyclohexane_chair_recovered_via_generate_conformers(self):
        """generate_conformers() on cyclohexane must return a chair as the lowest conformer."""
        mol = Molecule(smiles='C1CCCCC1')
        lowest_confs = conformers.generate_conformers(mol_list=mol, label='cyclohexane', n_confs=5)
        self.assertGreater(len(lowest_confs), 0)

        classified = list()
        for conf in lowest_confs:
            xyz = conf['xyz']
            perceived_mol = perceive_molecule_from_xyz(xyz)
            for ring in perceived_mol.get_deterministic_sssr():
                if len(ring) == 6:
                    ring_indices = [perceived_mol.atoms.index(atom) for atom in ring]
                    ring_coords = np.array(xyz['coords'])[ring_indices]
                    classified.append(ring_pucker.classify_pucker(ring_coords))
                    break

        self.assertIn('chair', classified, f'No chair conformer recovered; got pucker labels: {classified}')
        self.assertEqual(classified[0], 'chair',
                         f'Lowest-energy conformer is not a chair; got pucker labels (lowest first): {classified}')

    def test_ethylcyclohexane_pucker_times_rotamer(self):
        """A substituted ring must recover the chair global minimum together with exocyclic rotamers.

        This exercises the fix that routes ring-pucker base geometries through the exocyclic-torsion
        combination machinery: the returned ensemble should hold several chair conformers that differ
        in their side-chain rotamers, with the lowest-energy conformer being a chair.
        """
        mol = Molecule(smiles='CCC1CCCCC1')
        lowest_confs = conformers.generate_conformers(mol_list=mol, label='ethylcyclohexane', n_confs=10)
        self.assertGreaterEqual(len(lowest_confs), 2,
                                'Expected multiple (pucker x rotamer) conformers for ethylcyclohexane.')

        classified = list()
        for conf in lowest_confs:
            xyz = conf['xyz']
            perceived_mol = perceive_molecule_from_xyz(xyz)
            for ring in perceived_mol.get_deterministic_sssr():
                if len(ring) == 6:
                    ring_indices = [perceived_mol.atoms.index(atom) for atom in ring]
                    ring_coords = np.array(xyz['coords'])[ring_indices]
                    classified.append(ring_pucker.classify_pucker(ring_coords))
                    break

        self.assertEqual(classified[0], 'chair',
                         f'Lowest-energy conformer is not a chair; got pucker labels (lowest first): {classified}')


class TestETKDGv3Backstop(unittest.TestCase):
    """
    Contains unit tests for the ETKDGv3 ring-aware embedding backstop.
    """

    def test_gate_acyclic_false(self):
        """mol_has_ring_unsupported_by_cp() must be False for an acyclic molecule."""
        mol = Molecule(smiles='CCCCCC')
        self.assertFalse(conformers.mol_has_ring_unsupported_by_cp(mol))

    def test_gate_supported_monocyclic_false(self):
        """mol_has_ring_unsupported_by_cp() must be False for CP-supported monocyclic rings."""
        for smiles in ['C1CCCCC1', 'C1CCOCC1', 'CCC1CCCCC1']:
            mol = Molecule(smiles=smiles)
            self.assertFalse(conformers.mol_has_ring_unsupported_by_cp(mol),
                             f'Expected False for CP-supported ring in {smiles}.')

    def test_gate_unsaturated_true(self):
        """mol_has_ring_unsupported_by_cp() must be True for in-ring-unsaturated, non-aromatic rings."""
        for smiles in ['C1CCC=CC1']:
            mol = Molecule(smiles=smiles)
            self.assertTrue(conformers.mol_has_ring_unsupported_by_cp(mol),
                            f'Expected True for unsaturated ring in {smiles}.')

    def test_gate_exocyclic_sp2_true(self):
        """mol_has_ring_unsupported_by_cp() must be True for saturated 5/6 rings bearing an
        exocyclic sp2 atom (ketone, lactam), since CP cannot seed the resulting half-chair."""
        for smiles in ['O=C1CCCCC1', 'O=C1CCCCN1']:
            mol = Molecule(smiles=smiles)
            self.assertTrue(conformers.mol_has_ring_unsupported_by_cp(mol),
                            f'Expected True for exocyclic-sp2-bearing ring in {smiles}.')

    def test_gate_fully_aromatic_false(self):
        """mol_has_ring_unsupported_by_cp() must be False for a fully aromatic ring (no pucker
        freedom), and must not be forced True by an aromatic ring fused only through a
        substituent bond to a clean saturated ring."""
        for smiles in ['c1ccccc1', 'c1ccccc1C1CCCCC1']:
            mol = Molecule(smiles=smiles)
            self.assertFalse(conformers.mol_has_ring_unsupported_by_cp(mol),
                             f'Expected False for {smiles}.')

    def test_gate_wrong_size_true(self):
        """mol_has_ring_unsupported_by_cp() must be True for rings larger than 6, which CP does
        not support at all."""
        for smiles in ['C1CCCCCC1']:
            mol = Molecule(smiles=smiles)
            self.assertTrue(conformers.mol_has_ring_unsupported_by_cp(mol),
                            f'Expected True for wrong-size ring in {smiles}.')

    def test_gate_small_ring_false(self):
        """mol_has_ring_unsupported_by_cp() must be False for rings smaller than 5, whose pucker
        freedom is negligible and which the backstop need not cover."""
        mol = Molecule(smiles='C1CCC1')
        self.assertFalse(conformers.mol_has_ring_unsupported_by_cp(mol),
                         'Expected False for a small (4-membered) ring.')

    def test_gate_fused_bridged_spiro_true(self):
        """mol_has_ring_unsupported_by_cp() must be True for fused, bridged, and spiro ring systems."""
        for smiles in ['C1CCC2CCCCC2C1', 'C1CC2CCC1C2', 'C1CCC2(CC1)CCCC2']:
            mol = Molecule(smiles=smiles)
            self.assertTrue(conformers.mol_has_ring_unsupported_by_cp(mol),
                            f'Expected True for fused/bridged/spiro ring in {smiles}.')

    def test_embed_etkdgv3_returns_confs(self):
        """embed_rdkit() with use_etkdg_v3=True must return an RDMol with embedded conformers."""
        mol = Molecule(smiles='C1CCC2CCCCC2C1')
        rd_mol = conformers.embed_rdkit('decalin', mol, num_confs=5, use_etkdg_v3=True)
        self.assertIsNotNone(rd_mol)
        self.assertGreaterEqual(rd_mol.GetNumConformers(), 1)
        self.assertEqual(rd_mol.GetNumAtoms(), len(mol.atoms))

    def test_generate_ff_conformers_gated_source(self):
        """generate_force_field_conformers() must add ETKDGv3-sourced conformers only when the
        molecule has a ring unsupported by the CP seeder."""
        decalin_conformers = conformers.generate_force_field_conformers(
            'decalin', [Molecule(smiles='C1CCC2CCCCC2C1')], torsion_num=0, charge=0, multiplicity=1, num_confs=5)
        self.assertTrue(any(conf['source'] == 'ETKDGv3' for conf in decalin_conformers),
                        'Expected at least one ETKDGv3-sourced conformer for decalin.')

        hexane_conformers = conformers.generate_force_field_conformers(
            'hexane', [Molecule(smiles='CCCCCC')], torsion_num=0, charge=0, multiplicity=1, num_confs=5)
        self.assertFalse(any(conf['source'] == 'ETKDGv3' for conf in hexane_conformers),
                         'Did not expect any ETKDGv3-sourced conformer for hexane (acyclic gate).')

    def test_generate_conformers_fused_endtoend(self):
        """generate_conformers() must return a non-empty list for a fused bicyclic ring system without raising."""
        result = conformers.generate_conformers(mol_list=Molecule(smiles='C1CCC2CCCCC2C1'), label='decalin', n_confs=5)
        self.assertTrue(result)

    def test_etkdg_excluded_from_torsion_sampling(self):
        """deduce_new_conformers() must not let ETKDGv3-sourced conformers pollute torsion-well
        learning: an ETKDGv3 conformer tagged with a distinct torsion well must not change the
        learned symmetries nor the number of newly generated conformers relative to a baseline
        pool that lacks it."""
        mol = Molecule(smiles='CCCC')
        torsions, tops = conformers.determine_rotors([mol])
        baseline_conformers = conformers.generate_force_field_conformers(
            'butane', [mol], torsion_num=len(torsions), charge=0, multiplicity=1, num_confs=5)

        baseline_new_conformers, baseline_symmetries = conformers.deduce_new_conformers(
            label='butane', conformers=baseline_conformers, torsions=torsions, tops=tops, mol_list=[mol],
            combination_threshold=10)

        etkdg_conformer = copy.deepcopy(baseline_conformers[0])
        etkdg_conformer['source'] = 'ETKDGv3'
        torsion = tuple(torsions[0])
        distinct_angle = etkdg_conformer['torsion_dihedrals'][torsion] + 90.0
        if distinct_angle > 180.0:
            distinct_angle -= 360.0
        etkdg_conformer['torsion_dihedrals'][torsion] = distinct_angle
        polluted_conformers = baseline_conformers + [etkdg_conformer]

        polluted_new_conformers, polluted_symmetries = conformers.deduce_new_conformers(
            label='butane', conformers=polluted_conformers, torsions=torsions, tops=tops, mol_list=[mol],
            combination_threshold=10)

        self.assertEqual(baseline_symmetries, polluted_symmetries)
        # The ETKDG conformer is injected as an additional ring-conformer base (not a torsion-well
        # sample), so it legitimately contributes its own combinations on top of the baseline ones.
        # What must NOT change is the combinations generated from the baseline (non-ETKDG) bases,
        # which are emitted first and are identical to the unpolluted baseline run.
        self.assertEqual(baseline_new_conformers, polluted_new_conformers[:len(baseline_new_conformers)])
        self.assertGreater(len(polluted_new_conformers), len(baseline_new_conformers))

    def test_pucker_and_etkdg_bases_receive_full_combination_threshold(self):
        """deduce_new_conformers() must pass the full, undivided combination_threshold to
        generate_conformer_combinations() for every base regardless of source. The prior
        mechanism divided the threshold by (num_pucker_bases + 1) / (num_etkdg_bases + 1),
        which could silently flip a pucker/ETKDG base onto the lossy iterative-combination
        path and risk missing the global minimum; it was removed (C5), and cost is now bounded
        by the base-count caps (PUCKER_MAX_BASES, PUCKER_MAX_CROSS_BASE_CONFORMERS) instead."""
        mol = Molecule(smiles='CCC1CCCCC1')
        torsions, tops = conformers.determine_rotors([mol])
        baseline_conformers = conformers.generate_force_field_conformers(
            'ethylcyclohexane', [mol], torsion_num=len(torsions), charge=0, multiplicity=1, num_confs=5)

        captured_thresholds = list()
        original_generate_combinations = conformers.generate_conformer_combinations

        def _capture(*args, **kwargs):
            captured_thresholds.append(kwargs.get('combination_threshold'))
            return original_generate_combinations(*args, **kwargs)

        with patch.object(conformers, 'generate_conformer_combinations', side_effect=_capture):
            conformers.deduce_new_conformers(
                label='ethylcyclohexane', conformers=baseline_conformers, torsions=torsions, tops=tops,
                mol_list=[mol], combination_threshold=8)

        self.assertTrue(captured_thresholds)
        self.assertTrue(all(threshold == 8 for threshold in captured_thresholds),
                        f'Expected every base to receive the full combination_threshold; got {captured_thresholds}.')

    def test_etkdg_confs_become_bases(self):
        """generate_conformers() must incorporate ETKDGv3 ring conformers as bases and still
        return a non-empty result for a fused ring system with no rotatable exocyclic torsions."""
        decalin_smiles = 'C1CCC2CCCCC2C1'
        self.assertTrue(conformers.mol_has_ring_unsupported_by_cp(Molecule(smiles=decalin_smiles)))

        decalin_conformers = conformers.generate_force_field_conformers(
            'decalin', [Molecule(smiles=decalin_smiles)], torsion_num=0, charge=0, multiplicity=1, num_confs=5)
        self.assertGreaterEqual(sum(1 for conf in decalin_conformers if conf['source'] == 'ETKDGv3'), 1)

        result = conformers.generate_conformers(mol_list=Molecule(smiles=decalin_smiles), label='decalin', n_confs=6)
        self.assertTrue(result)


class TestAcyclicRegressionGuard(unittest.TestCase):
    """
    Guards that the ring-pucker and ETKDGv3 machinery is inert on acyclic molecules,
    keeping the acyclic conformer-generation path identical to the pre-feature behavior.
    """

    ACYCLIC_SPECIES = [('heptane', 'CCCCCCC'),
                       ('hexanol', 'CCCCCCO'),
                       ('iso_octane', 'CC(C)CC(C)(C)C'),
                       ('glycerol', 'OCC(O)CO')]

    def test_ring_machinery_inert_on_acyclic_molecules(self):
        """mol_has_ring_unsupported_by_cp() and ring_pucker_base_conformers() must be no-ops for acyclic molecules."""
        for label, smiles in self.ACYCLIC_SPECIES:
            mol, base_conformers = build_base_conformers(smiles, seed=0)
            self.assertFalse(conformers.mol_has_ring_unsupported_by_cp(mol),
                             f'Expected False for acyclic {label}.')
            result = conformers.ring_pucker_base_conformers(label=label, mol=mol,
                                                            base_conformers=base_conformers)
            self.assertEqual(result, [], f'Expected no ring-pucker bases for acyclic {label}.')

    def test_generate_conformers_has_no_ring_sources_on_acyclic_molecules(self):
        """generate_conformers() must not emit ETKDGv3- or ring-pucker-sourced conformers for acyclic molecules."""
        for label, smiles in self.ACYCLIC_SPECIES:
            lowest_confs, all_confs = conformers.generate_conformers(
                mol_list=Molecule(smiles=smiles), label=label, n_confs=10, return_all_conformers=True)
            self.assertTrue(lowest_confs, f'Expected a non-empty lowest conformers list for {label}.')
            for conf in lowest_confs + all_confs:
                self.assertNotIn(conf.get('source'), ('ETKDGv3', 'ring pucker'),
                                 f'Unexpected {conf.get("source")}-sourced conformer for acyclic {label}.')

    def test_heptane_generate_conformers_deterministic(self):
        """Two independent generate_conformers() calls on heptane must return identical FF energies."""
        energies = list()
        for _ in range(2):
            lowest_confs = conformers.generate_conformers(mol_list=Molecule(smiles='CCCCCCC'),
                                                          label='heptane', n_confs=10)
            energies.append([conf['FF energy'] for conf in lowest_confs])
        self.assertEqual(energies[0], energies[1])


class TestMultiRingPuckerCrossProduct(unittest.TestCase):
    """
    Contains unit tests for the bounded multi-independent-ring pucker cross product in
    ring_pucker_base_conformers().
    """

    def test_dicyclohexyl_cross_product_covers_both_rings_chair(self):
        """Both independent 6-rings of dicyclohexyl must be simultaneously recoverable as chairs.

        Seed 5 embeds/MMFF-optimizes to a base geometry where BOTH rings classify as
        'twist-boat' (confirmed against the pre-cross-product code: it returns 12 per-ring
        pucker bases, none of which have both rings 'chair' simultaneously). The single-ring
        seeder can only pucker one ring at a time against this base, so it cannot produce a
        combined (chair, chair) state; only the cross product can.
        """
        mol, xyz, rd_mol = build_seed_geometry('C1CCC(CC1)C1CCCCC1', seed=5)
        ring_index_lists = [list(ring) for ring in rd_mol.GetRingInfo().AtomRings()]
        self.assertEqual(len(ring_index_lists), 2)

        base_conformers = [{'xyz': xyz, 'index': 0, 'FF energy': 0.0, 'source': 'test'}]
        result = conformers.ring_pucker_base_conformers(label='dicyclohexyl', mol=mol,
                                                         base_conformers=base_conformers)
        self.assertGreater(len(result), 0)

        both_chair = False
        for conf in result:
            coords = np.array(conf['xyz']['coords'])
            labels = [ring_pucker.classify_pucker(coords[ring_indices]) for ring_indices in ring_index_lists]
            if all(pucker_label == 'chair' for pucker_label in labels):
                both_chair = True
                break
        self.assertTrue(both_chair, 'Expected a combined (chair, chair) base among the cross-product results.')

    def test_cross_product_falls_back_to_per_ring_when_a_ring_plan_fails(self):
        """A base for which one ring's plan fails to build must not be dropped outright; the
        cross product must fall back to per-ring seeding for that base so the other ring's pucker
        seeds are not lost.

        This guards the P2a coverage-regression fix in _ring_pucker_cross_product_bases: the
        pre-fix code did `continue` on the whole base whenever any single ring's plan came back
        None, silently dropping seeds for rings that DID plan successfully. Ring B's plan is
        forced to fail via a monkeypatched _ring_pucker_plan; ring A's real plan is left intact,
        and the test asserts ring A still shows displaced (non-base) pucker geometry among the
        returned bases, rather than the whole base being skipped.
        """
        mol, xyz, rd_mol = build_seed_geometry('C1CCC(CC1)C1CCCCC1', seed=5)
        ring_index_lists = [list(ring) for ring in rd_mol.GetRingInfo().AtomRings()]
        self.assertEqual(len(ring_index_lists), 2)
        ring_a, ring_b = ring_index_lists
        ring_b_set = frozenset(ring_b)

        real_plan = conformers._ring_pucker_plan

        def fake_plan(label, mol, ring_atom_indices, coords, atom_to_index):
            if frozenset(ring_atom_indices) == ring_b_set:
                return None
            return real_plan(label, mol, ring_atom_indices, coords, atom_to_index)

        base_conformers = [{'xyz': xyz, 'index': 0, 'FF energy': 0.0, 'source': 'test'}]
        with patch('arc.species.conformers._ring_pucker_plan', side_effect=fake_plan):
            result = conformers.ring_pucker_base_conformers(label='dicyclohexyl', mol=mol,
                                                             base_conformers=base_conformers)

        self.assertGreater(len(result), 0)
        ring_pucker_results = [conf for conf in result if conf.get('source') == 'ring pucker']
        self.assertGreater(len(ring_pucker_results), 0)

        base_coords = np.array(xyz['coords'])
        displaced = False
        for conf in ring_pucker_results:
            coords = np.array(conf['xyz']['coords'])
            if not np.allclose(coords[ring_a], base_coords[ring_a], atol=1e-3):
                displaced = True
                break
        self.assertTrue(displaced, 'Expected ring A to still be puckered via the per-ring fallback.')

    def test_cross_product_bounded_by_cap(self):
        """The number of returned dicyclohexyl pucker bases must never exceed PUCKER_MAX_BASES."""
        mol, base_conformers = build_base_conformers('C1CCC(CC1)C1CCCCC1', seed=5)
        result = conformers.ring_pucker_base_conformers(label='dicyclohexyl', mol=mol,
                                                         base_conformers=base_conformers)
        self.assertLessEqual(len(result), conformers.PUCKER_MAX_BASES)

    def test_cross_product_base_conformer_count_bounded(self):
        """Only PUCKER_MAX_CROSS_BASE_CONFORMERS of the supplied base conformers may receive
        cross-product pucker seeding, bounding worst-case FF-opt cost.

        Eight duplicated dicyclohexyl base conformer dicts (varying only in 'FF energy', a cheap
        synthetic stand-in for real diastereomers) are supplied; the point is testing the count
        bound, not diastereomer diversity. _ring_pucker_plan is wrapped (not replaced) so it still
        does real per-ring planning, letting its call count serve as a proxy for how many bases
        were processed: with both rings planning successfully every processed base costs exactly
        len(ring_index_lists) == 2 calls.
        """
        mol, xyz, rd_mol = build_seed_geometry('C1CCC(CC1)C1CCCCC1', seed=5)
        ring_index_lists = [list(ring) for ring in rd_mol.GetRingInfo().AtomRings()]
        self.assertEqual(len(ring_index_lists), 2)

        base_conformers = [{'xyz': xyz, 'index': i, 'FF energy': float(i), 'source': 'test'}
                          for i in range(8)]

        real_plan = conformers._ring_pucker_plan
        with patch('arc.species.conformers._ring_pucker_plan', side_effect=real_plan) as mock_plan:
            result = conformers.ring_pucker_base_conformers(label='dicyclohexyl', mol=mol,
                                                             base_conformers=base_conformers)

        self.assertEqual(mock_plan.call_count,
                         conformers.PUCKER_MAX_CROSS_BASE_CONFORMERS * len(ring_index_lists)
                         + len(base_conformers[:conformers.PUCKER_MAX_CROSS_BASE_CONFORMERS])
                         * len(ring_index_lists))
        self.assertLessEqual(len(result), conformers.PUCKER_MAX_BASES)

    def test_cross_product_path_stays_reduced_not_full_wheel(self):
        """The independent-rings cross product must keep its reduced (3-state x 2-sign = 6
        options/ring) enumeration, NOT the 14-entry single-ring phase wheel, per the locked design
        decision that a full wheel there would blow the 36-combo cap (14**2 == 196 >> 36).

        Since ``ring_pucker_base_conformers`` now returns the UNION of the cross product and the
        per-ring full-wheel independent seeds, ``_displace_ring_pucker`` calls come from two
        sources: the cross product runs first (via ``_ring_pucker_cross_product_bases``) and the
        independent full wheel runs second (via ``_seed_rings_independently``). The first
        ``PUCKER_MAX_CROSS_COMBINATIONS * len(ring_index_lists)`` calls (the cross-product's own
        budget) are asserted to all be reduced (``phase_deg=None``, labels drawn only from
        ``canonical_pucker_states(6)``); the remaining calls (the unioned independent wheel) are
        asserted to include at least one non-None ``phase_deg``, proving the wheel is still used.
        """
        mol, xyz, rd_mol = build_seed_geometry('C1CCC(CC1)C1CCCCC1', seed=5)
        ring_index_lists = [list(ring) for ring in rd_mol.GetRingInfo().AtomRings()]
        self.assertEqual(len(ring_index_lists), 2)

        base_conformers = [{'xyz': xyz, 'index': 0, 'FF energy': 0.0, 'source': 'test'}]
        real_displace = conformers._displace_ring_pucker
        with patch('arc.species.conformers._displace_ring_pucker', side_effect=real_displace) as mock_displace:
            conformers.ring_pucker_base_conformers(label='dicyclohexyl', mol=mol,
                                                    base_conformers=base_conformers)

        n_cross_calls = conformers.PUCKER_MAX_CROSS_COMBINATIONS * len(ring_index_lists)
        self.assertGreater(len(mock_displace.call_args_list), n_cross_calls,
                           'Expected the union to add calls beyond the cross product alone.')
        cross_calls = mock_displace.call_args_list[:n_cross_calls]
        independent_calls = mock_displace.call_args_list[n_cross_calls:]

        seen_labels = set()
        for call in cross_calls:
            _, plan, label_state, phase_deg, pole_sign, _ = call.args
            self.assertIsNone(phase_deg)
            seen_labels.add(label_state)
        self.assertEqual(seen_labels, set(ring_pucker.canonical_pucker_states(6)))

        self.assertTrue(any(call.args[3] is not None for call in independent_calls),
                        'Expected the unioned independent full-wheel path to use non-None phases.')

    def test_monocyclic_pucker_bases_are_count_capped_chair_lowest(self):
        """ring_pucker_base_conformers()'s output for monocyclic molecules is capped by COUNT
        only (get_lowest_confs(..., e=None)), not by a pre-torsion energy window, since a single
        ring never enters the cross-product branch (len(rings) >= 2 is required).

        This supersedes the old "byte-identical to pre-refactor" golden: seeding now enumerates
        the full 14-entry phase wheel (2 chair poles + 12 equatorial phase bins), and pinning exact
        FF energies encoded the now-removed default 5.0 kcal/mol energy pruning, which could
        silently discard a distinct pucker phase before torsion seeding ever ran. Instead this
        asserts only structural, non-brittle properties: the base count stays within budget, and
        chair (cyclohexane's/ethylcyclohexane's known global-minimum pucker) is both present and
        the lowest-energy base.
        """
        for smiles in ('C1CCCCC1', 'CCC1CCCCC1'):
            mol, base_conformers = build_base_conformers(smiles, seed=0)
            result = conformers.ring_pucker_base_conformers(label='monocyclic', mol=mol,
                                                             base_conformers=base_conformers)
            self.assertGreaterEqual(len(result), 1, f'Expected at least one base for {smiles}.')
            self.assertLessEqual(len(result), conformers.PUCKER_MAX_BASES,
                                 f'Expected at most PUCKER_MAX_BASES bases for {smiles}.')

            atom_to_index = {id(atom): i for i, atom in enumerate(mol.atoms)}
            ring = [ring for ring in mol.get_deterministic_sssr() if len(ring) == 6][0]
            ring_indices = [atom_to_index[id(atom)] for atom in ring]
            labels = [ring_pucker.classify_pucker(np.array(conf['xyz']['coords'])[ring_indices])
                     for conf in result]
            self.assertIn('chair', labels, f'Expected at least one chair base for {smiles}.')

            lowest = min(result, key=lambda conf: conf['FF energy'])
            lowest_label = ring_pucker.classify_pucker(np.array(lowest['xyz']['coords'])[ring_indices])
            self.assertEqual(lowest_label, 'chair',
                             f'Expected the global-minimum base to be a chair for {smiles}.')

    def test_fused_ring_still_returns_empty(self):
        """A fused bicyclic system (decalin) must not enter the cross product and must still
        return an empty list, exactly as the pre-cross-product code does (see also
        TestRingPuckerBaseConformers.test_decalin_fused_ring_returns_empty)."""
        mol, base_conformers = build_base_conformers('C1CCC2CCCCC2C1', seed=0)
        result = conformers.ring_pucker_base_conformers(label='decalin', mol=mol,
                                                         base_conformers=base_conformers)
        self.assertEqual(result, [])

    def test_three_independent_rings_falls_back_to_per_ring_path(self):
        """When the combination count exceeds PUCKER_MAX_CROSS_COMBINATIONS, the cross product
        must not be attempted; each ring is puckered independently against the base instead.

        Tricyclohexylmethane (SMILES 'C(C1CCCCC1)(C1CCCCC1)C1CCCCC1') has three independent
        6-rings, needing 6**3 = 216 (pucker state x pole) combinations -- well above the
        36-combination cap -- so this exercises a real invocation of the >36-combos fallback
        branch in ring_pucker_base_conformers() (the check at the top of that function, distinct
        from _ring_pucker_cross_product_bases' own per-base "a ring's plan failed" fallback added
        for the P2a fix). _ring_pucker_cross_product_bases is mocked only to assert it is never
        called -- confirming the branch decision -- while the per-ring path itself (three real
        ring_pucker_seed_conformers() calls) still runs for real and is asserted to yield seeds.
        Embedding tricyclohexylmethane and running the real per-ring path took ~2s locally, well
        under the ~15s budget, so no cheaper synthetic construction was needed.
        """
        mol, base_conformers = build_base_conformers('C(C1CCCCC1)(C1CCCCC1)C1CCCCC1', seed=0)
        sssr = mol.get_deterministic_sssr()
        rings = [ring for ring in sssr if len(ring) in (5, 6) and conformers.ring_is_saturated(mol, ring)]
        self.assertEqual(len(rings), 3)

        n_combos = 1
        for ring in rings:
            n_combos *= 2 * len(ring_pucker.canonical_pucker_states(len(ring)))
        self.assertEqual(n_combos, 216)
        self.assertGreater(n_combos, conformers.PUCKER_MAX_CROSS_COMBINATIONS)

        with patch('arc.species.conformers._ring_pucker_cross_product_bases') as mock_cross:
            result = conformers.ring_pucker_base_conformers(label='tricyclohexylmethane', mol=mol,
                                                             base_conformers=base_conformers)

        mock_cross.assert_not_called()
        self.assertGreater(len(result), 0)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
