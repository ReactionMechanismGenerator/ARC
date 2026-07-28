#!/usr/bin/env python3
# encoding: utf-8

"""
Tests for the global-min-recovery metric (``ring_conformer_metric``) and its supporting
Cremer-Pople diagnostic helpers (``pucker_state_id``, ``pucker_label_counts``) added in
milestone 3 of the ring-puckering conformer feature.
"""

import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from arc.species import conformers, ring_pucker
from arc.species.converter import xyz_from_data
from arc.species.perceive import perceive_molecule_from_xyz


def build_cyclohexane_pool(seed=0):
    """Build a pooled cyclohexane conformer set mixing ring-pucker seeds with plain embeds.

    Args:
        seed (int, optional): The RDKit random seed for the plain-embedded conformers.

    Returns:
        tuple: ``(pool, ring_atom_indices, mol)`` where ``pool`` is a list of
               ``{'xyz': ..., 'FF energy': ...}`` conformer dicts, ``ring_atom_indices``
               are the ring atom indices within each conformer's xyz, and ``mol`` is the
               corresponding RMG Molecule object.
    """
    rd_mol = Chem.AddHs(Chem.MolFromSmiles('C1CCCCC1'))
    AllChem.EmbedMolecule(rd_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rd_mol)
    conf = rd_mol.GetConformer()
    coords = [tuple(conf.GetAtomPosition(i)) for i in range(rd_mol.GetNumAtoms())]
    symbols = tuple(atom.GetSymbol() for atom in rd_mol.GetAtoms())
    xyz = xyz_from_data(coords=coords, symbols=symbols)
    mol = perceive_molecule_from_xyz(xyz)
    ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

    pool = list()
    seed_xyzs, seed_energies = conformers.ring_pucker_seed_conformers(
        label='cyclohexane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)
    for conf_xyz, energy in zip(seed_xyzs, seed_energies):
        pool.append({'xyz': conf_xyz, 'FF energy': energy})

    rd_embed = embed_and_optimize(seed)
    if rd_embed is not None:
        for i in range(rd_embed.GetNumConformers()):
            embed_conf = rd_embed.GetConformer(i)
            embed_coords = [tuple(embed_conf.GetAtomPosition(j)) for j in range(rd_embed.GetNumAtoms())]
            embed_xyz = xyz_from_data(coords=embed_coords, symbols=symbols)
            embed_energy = compute_mmff_energy(rd_embed, i)
            if embed_energy is not None:
                pool.append({'xyz': embed_xyz, 'FF energy': embed_energy})

    return pool, ring_atom_indices, mol


def embed_and_optimize(seed, num_confs=5):
    """Embed and MMFF-optimize a few plain cyclohexane conformers via RDKit.

    Args:
        seed (int): The RDKit random seed.
        num_confs (int, optional): The number of conformers to embed.

    Returns:
        Chem.Mol | None: An RDKit molecule with optimized conformers.
    """
    rd_mol = Chem.AddHs(Chem.MolFromSmiles('C1CCCCC1'))
    AllChem.EmbedMultipleConfs(rd_mol, numConfs=num_confs, randomSeed=seed)
    AllChem.MMFFOptimizeMoleculeConfs(rd_mol)
    return rd_mol


def compute_mmff_energy(rd_mol, conf_id):
    """Compute the MMFF94s energy of a given RDKit conformer.

    Args:
        rd_mol (Chem.Mol): The RDKit molecule.
        conf_id (int): The conformer index.

    Returns:
        float | None: The MMFF94s energy in kcal/mol, or ``None`` if unavailable.
    """
    mol_properties = AllChem.MMFFGetMoleculeProperties(rd_mol, mmffVariant='MMFF94s')
    if mol_properties is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(rd_mol, mol_properties, confId=conf_id)
    if ff is None:
        return None
    return ff.CalcEnergy()


class TestPuckerStateId(unittest.TestCase):
    """Tests for ring_pucker.pucker_state_id()."""

    def test_boat_and_twist_boat_are_distinct(self):
        boat_coords = ring_pucker.ideal_pucker_geometry(6, 'boat')
        twist_boat_coords = ring_pucker.ideal_pucker_geometry(6, 'twist-boat')
        boat_id = ring_pucker.pucker_state_id(boat_coords)
        twist_boat_id = ring_pucker.pucker_state_id(twist_boat_coords)
        self.assertNotEqual(boat_id, twist_boat_id)

    def test_planar_hexagon_returns_planar_unbinned(self):
        idx = np.arange(6)
        angles = 2.0 * np.pi * idx / 6
        planar_coords = np.column_stack([1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(6)])
        self.assertEqual(ring_pucker.pucker_state_id(planar_coords), 'planar')

    def test_chair_returns_unbinned(self):
        chair_coords = ring_pucker.ideal_pucker_geometry(6, 'chair')
        self.assertEqual(ring_pucker.pucker_state_id(chair_coords), 'chair')


class TestPuckerLabelCounts(unittest.TestCase):
    """Tests for ring_pucker.pucker_label_counts()."""

    def test_mixed_pucker_geometries_are_counted(self):
        chair_coords = ring_pucker.ideal_pucker_geometry(6, 'chair')
        twist_boat_coords = ring_pucker.ideal_pucker_geometry(6, 'twist-boat')
        counts = ring_pucker.pucker_label_counts([chair_coords, twist_boat_coords])
        self.assertIn('chair', counts)
        self.assertIn('twist-boat', counts)
        self.assertEqual(counts['chair'], 1)
        self.assertEqual(counts['twist-boat'], 1)


class TestRingConformerMetric(unittest.TestCase):
    """Tests for conformers.ring_conformer_metric()."""

    def test_pooled_cyclohexane_metric(self):
        pool, ring_atom_indices, mol = build_cyclohexane_pool()
        metric = conformers.ring_conformer_metric('cyclohexane', mol, pool, ring_atom_indices)

        self.assertTrue(np.isfinite(metric['min_energy']))
        self.assertIn('chair', metric['arc_dedup_pucker_labels'])
        # Secondary diagnostic: ARC's own (non-symmetry-reduced) dedup should find at least one minimum.
        self.assertGreaterEqual(metric['arc_dedup_unique_confs'], 1)

        lowest_conf = min(pool, key=lambda c: c['FF energy'])
        lowest_ring_coords = np.array(lowest_conf['xyz']['coords'])[ring_atom_indices]
        self.assertEqual(ring_pucker.classify_pucker(lowest_ring_coords), 'chair')

        metric = conformers.ring_conformer_metric(
            'cyclohexane', mol, pool, ring_atom_indices, reference_conformer=lowest_conf)
        self.assertTrue(metric['global_min_hit'])

    def test_global_min_hit_true_for_matching_chair_reference(self):
        pool, ring_atom_indices, mol = build_cyclohexane_pool()
        reference = min(pool, key=lambda c: c['FF energy'])
        metric = conformers.ring_conformer_metric(
            'cyclohexane', mol, pool, ring_atom_indices, reference_conformer=reference)
        self.assertTrue(metric['global_min_hit'])

    def test_global_min_hit_symmetry_aware(self):
        """A ring-flip chair pole should hit a reference of the OTHER chair pole via
        symmetry-aware RMSD, even though the two poles' atom coordinates differ substantially
        under the fixed atom ordering (a naive, non-symmetry-aware RMSD would miss the match)."""
        rd_mol = Chem.AddHs(Chem.MolFromSmiles('C1CCCCC1'))
        AllChem.EmbedMolecule(rd_mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(rd_mol)
        conf = rd_mol.GetConformer()
        coords = [tuple(conf.GetAtomPosition(i)) for i in range(rd_mol.GetNumAtoms())]
        symbols = tuple(atom.GetSymbol() for atom in rd_mol.GetAtoms())
        xyz = xyz_from_data(coords=coords, symbols=symbols)
        mol = perceive_molecule_from_xyz(xyz)
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        seed_xyzs, seed_energies = conformers.ring_pucker_seed_conformers(
            label='cyclohexane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)
        chairs = list()
        for conf_xyz, energy in zip(seed_xyzs, seed_energies):
            ring_coords = np.array(conf_xyz['coords'])[ring_atom_indices]
            if ring_pucker.classify_pucker(ring_coords) == 'chair':
                chairs.append({'xyz': conf_xyz, 'FF energy': energy})
        self.assertGreaterEqual(len(chairs), 2)

        pole_a, pole_b = chairs[0], chairs[1]
        naive_rmsd = np.sqrt(np.mean(np.sum(
            (np.array(pole_a['xyz']['coords']) - np.array(pole_b['xyz']['coords'])) ** 2, axis=1)))
        self.assertGreater(naive_rmsd, 0.125)

        metric = conformers.ring_conformer_metric(
            'cyclohexane', mol, [pole_a], ring_atom_indices, reference_conformer=pole_b)
        self.assertTrue(metric['global_min_hit'])

    def test_global_min_hit_false_for_different_pucker_reference(self):
        rd_mol = Chem.AddHs(Chem.MolFromSmiles('C1CCCCC1'))
        AllChem.EmbedMolecule(rd_mol, randomSeed=0)
        AllChem.MMFFOptimizeMolecule(rd_mol)
        conf = rd_mol.GetConformer()
        coords = [tuple(conf.GetAtomPosition(i)) for i in range(rd_mol.GetNumAtoms())]
        symbols = tuple(atom.GetSymbol() for atom in rd_mol.GetAtoms())
        xyz = xyz_from_data(coords=coords, symbols=symbols)
        mol = perceive_molecule_from_xyz(xyz)
        ring_atom_indices = list(rd_mol.GetRingInfo().AtomRings()[0])

        chair_xyzs, chair_energies = conformers.ring_pucker_seed_conformers(
            label='cyclohexane', mol=mol, ring_atom_indices=ring_atom_indices, base_xyz=xyz)
        chair_conf = None
        twist_boat_only_pool = list()
        for conf_xyz, energy in zip(chair_xyzs, chair_energies):
            ring_coords = np.array(conf_xyz['coords'])[ring_atom_indices]
            label = ring_pucker.classify_pucker(ring_coords)
            if label == 'chair' and chair_conf is None:
                chair_conf = {'xyz': conf_xyz, 'FF energy': energy}
            if label == 'twist-boat':
                twist_boat_only_pool.append({'xyz': conf_xyz, 'FF energy': energy})

        self.assertIsNotNone(chair_conf)
        self.assertGreater(len(twist_boat_only_pool), 0)

        metric = conformers.ring_conformer_metric(
            'cyclohexane', mol, twist_boat_only_pool, ring_atom_indices, reference_conformer=chair_conf)
        self.assertFalse(metric['global_min_hit'])

    def test_metric_does_not_mutate_input(self):
        pool, ring_atom_indices, mol = build_cyclohexane_pool()
        original_keys = [set(conf.keys()) for conf in pool]
        conformers.ring_conformer_metric('cyclohexane', mol, pool, ring_atom_indices)
        for conf, keys_before in zip(pool, original_keys):
            self.assertEqual(set(conf.keys()), keys_before)

    def test_status_field(self):
        metric = conformers.ring_conformer_metric('cyclohexane', None, [], [0, 1, 2, 3, 4, 5])
        self.assertEqual(metric['status'], 'empty')
        self.assertEqual(metric['arc_dedup_unique_confs'], 0)
        self.assertIsNone(metric['min_energy'])

        pool, ring_atom_indices, mol = build_cyclohexane_pool()
        metric = conformers.ring_conformer_metric('cyclohexane', mol, pool, ring_atom_indices)
        self.assertEqual(metric['status'], 'ok')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
