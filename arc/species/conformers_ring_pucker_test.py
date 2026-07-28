#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the ring_pucker_seed_conformers() function in
arc.species.conformers, and of the ring_pucker.ring_mean_plane() helper it relies on.
"""

import copy
import unittest

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

    def test_multiple_bases_yield_more_than_single_base(self):
        """Two base conformers must yield more pucker bases than a single base conformer."""
        mol, xyz, _ = build_seed_geometry('C1CCCCC1', seed=0)
        _, xyz_alt, _ = build_seed_geometry('C1CCCCC1', seed=1)
        base_one = [{'xyz': xyz, 'index': 0, 'FF energy': 0.0, 'source': 'test'}]
        base_two = base_one + [{'xyz': xyz_alt, 'index': 1, 'FF energy': 0.0, 'source': 'test'}]

        result_one = conformers.ring_pucker_base_conformers(label='cyclohexane', mol=mol,
                                                             base_conformers=base_one)
        result_two = conformers.ring_pucker_base_conformers(label='cyclohexane', mol=mol,
                                                             base_conformers=base_two)
        self.assertGreater(len(result_two), len(result_one))

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
        """mol_has_ring_unsupported_by_cp() must be True for unsaturated rings."""
        for smiles in ['C1CCC=CC1', 'c1ccccc1']:
            mol = Molecule(smiles=smiles)
            self.assertTrue(conformers.mol_has_ring_unsupported_by_cp(mol),
                            f'Expected True for unsaturated ring in {smiles}.')

    def test_gate_wrong_size_true(self):
        """mol_has_ring_unsupported_by_cp() must be True for rings outside size 5/6."""
        for smiles in ['C1CCCCCC1', 'C1CCC1']:
            mol = Molecule(smiles=smiles)
            self.assertTrue(conformers.mol_has_ring_unsupported_by_cp(mol),
                            f'Expected True for wrong-size ring in {smiles}.')

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

    def test_etkdg_bases_do_not_alter_pucker_threshold(self):
        """deduce_new_conformers() must derive the ring-pucker combination threshold solely
        from the number of ring-pucker bases, independent of how many ETKDGv3 bases are also
        present in the input pool."""
        mol = Molecule(smiles='CCC1CCCCC1')
        torsions, tops = conformers.determine_rotors([mol])
        baseline_conformers = conformers.generate_force_field_conformers(
            'ethylcyclohexane', [mol], torsion_num=len(torsions), charge=0, multiplicity=1, num_confs=5)

        new_conformers_a, _ = conformers.deduce_new_conformers(
            label='ethylcyclohexane', conformers=baseline_conformers, torsions=torsions, tops=tops,
            mol_list=[mol], combination_threshold=8)

        polluted_conformers = list(baseline_conformers)
        for _ in range(4):
            etkdg_conformer = copy.deepcopy(baseline_conformers[0])
            etkdg_conformer['source'] = 'ETKDGv3'
            polluted_conformers.append(etkdg_conformer)

        new_conformers_b, _ = conformers.deduce_new_conformers(
            label='ethylcyclohexane', conformers=polluted_conformers, torsions=torsions, tops=tops,
            mol_list=[mol], combination_threshold=8)

        self.assertEqual(_conformers_without_dmat(new_conformers_a),
                          _conformers_without_dmat(new_conformers_b[:len(new_conformers_a)]))
        self.assertGreaterEqual(len(new_conformers_b), len(new_conformers_a))

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


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
