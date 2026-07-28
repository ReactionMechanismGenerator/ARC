#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the ring_pucker_seed_conformers() function in
arc.species.conformers, and of the ring_pucker.ring_mean_plane() helper it relies on.
"""

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


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
