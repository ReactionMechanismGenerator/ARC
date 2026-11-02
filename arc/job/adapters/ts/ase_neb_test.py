#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.adapters.ts.ase_neb module.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from arc.common import read_yaml_file
from arc.exceptions import DependencyError
from arc.job.adapters.common import has_one_molecule_side
from arc.job.adapters.ts.ase_neb import (ASENEBAdapter,
                                         FRAGMENT_SEPARATION,
                                         candidate_endpoints,
                                         h_transfers,
                                         moved_h_product,
                                         kabsch_align,
                                         neb_endpoints,
                                         resolve_level,
                                         side_fragments,
                                         )
from arc.level import Level
from arc.reaction import ARCReaction
from arc.species import ARCSpecies


ETHYL = """C      -0.79286713    0.00021434   -0.00488532
C       0.69009402   -0.00048893    0.01264458
H      -1.19389316    0.88538716    0.50369548
H      -1.15736023    0.00335563   -1.03735733
H      -1.19455587   -0.88746226    0.49685810
H       1.23960066    0.92705716   -0.01845001
H       1.23940418   -0.92770023   -0.01993618"""
ETHYLENE = """C       0.66546301    0.00000000    0.00000000
C      -0.66546301    0.00000000    0.00000000
H       1.23926316    0.92465700    0.00000000
H       1.23926316   -0.92465700    0.00000000
H      -1.23926316    0.92465700    0.00000000
H      -1.23926316   -0.92465700    0.00000000"""


def distances(coords: np.ndarray) -> np.ndarray:
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)


class TestASENEB(unittest.TestCase):
    """
    Contains unit tests for the ASE NEB TS adapter.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.project_directory = tempfile.mkdtemp(prefix='test_ASENEBAdapter_')
        cls.addClassCleanup(shutil.rmtree, cls.project_directory, ignore_errors=True)
        cls.dissociation = ARCReaction(r_species=[ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=ETHYL)],
                                       p_species=[ARCSpecies(label='C2H4', smiles='C=C', xyz=ETHYLENE),
                                                  ARCSpecies(label='H', smiles='[H]')])
        cls.isomerization = ARCReaction(r_species=[ARCSpecies(label='nC3H7', smiles='CC[CH2]')],
                                        p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')])
        cls.abstraction = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='H', smiles='[H]')],
                                      p_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                                 ARCSpecies(label='H2', smiles='[H][H]')])
        cls.uma = Level(method='uma-s-1p2')
        cls.ho2_elimination = ARCReaction(r_species=[ARCSpecies(label='C2H5O2', smiles='CCO[O]')],
                                          p_species=[ARCSpecies(label='C2H4', smiles='C=C'),
                                                     ARCSpecies(label='HO2', smiles='[O]O')])
        cls.h_shift = ARCReaction(r_species=[ARCSpecies(label='C2H5O2', smiles='CCO[O]')],
                                  p_species=[ARCSpecies(label='C2H4OOH', smiles='[CH2]COO')])

    def test_kabsch_align(self):
        """A rotated and translated copy is aligned back onto the original."""
        target = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.1, 0.3], [-0.4, -0.2, 1.0]])
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        mobile = target @ rotation.T + np.array([3.0, -2.0, 1.0])
        np.testing.assert_allclose(kabsch_align(mobile, target), target, atol=1e-10)
        # A single atom only moves.
        np.testing.assert_allclose(kabsch_align(np.array([[5.0, 5.0, 5.0]]), np.array([[1.0, 2.0, 3.0]])),
                                   [[1.0, 2.0, 3.0]])

    def test_has_one_molecule_side(self):
        """NEB endpoints are built from a one-molecule side, which an abstraction lacks."""
        self.assertTrue(has_one_molecule_side(self.dissociation))
        self.assertTrue(has_one_molecule_side(self.isomerization))
        self.assertFalse(has_one_molecule_side(self.abstraction))
        with self.assertRaises(DependencyError):
            neb_endpoints(self.abstraction)

    def test_dissociation_endpoints(self):
        """C2H5 -> C2H4 + H: the products sit where they are in ethyl, rigidly, and FRAGMENT_SEPARATION apart."""
        r_xyz, p_xyz = neb_endpoints(self.dissociation)
        self.assertEqual(r_xyz['symbols'], p_xyz['symbols'])
        r, p = np.array(r_xyz['coords']), np.array(p_xyz['coords'])
        np.testing.assert_allclose(distances(r), distances(np.array(self.dissociation.get_reactants_xyz(
            return_format=dict)['coords'])), atol=1e-8)  # the reactant is unchanged
        fragments = side_fragments(self.dissociation, well=1)
        self.assertEqual(sorted(len(f) for f in fragments), [1, 6])
        ethylene, hydrogen = sorted(fragments, key=len, reverse=True)
        self.assertEqual({r_xyz['symbols'][i] for i in hydrogen}, {'H'})
        # Ethylene keeps its own geometry.
        np.testing.assert_allclose(np.sort(distances(p[ethylene]).ravel()),
                                   np.sort(distances(np.array(ETHYLENE_COORDS)).ravel()), atol=1e-6)
        # It sits where those atoms are in ethyl.
        self.assertLess(np.sqrt(((p[ethylene] - r[ethylene]) ** 2).sum(axis=1).mean()), 0.6)
        closest = np.min(np.linalg.norm(p[ethylene] - p[hydrogen][0], axis=1))
        self.assertGreaterEqual(closest, FRAGMENT_SEPARATION)
        self.assertLess(closest, FRAGMENT_SEPARATION + 0.2)
        # The H leaves from the CH3 group of ethyl (a beta C-H): its carbon carries three H atoms, it included.
        carbons = [i for i, s in enumerate(r_xyz['symbols']) if s == 'C']
        nearest_carbon = min(carbons, key=lambda c: np.linalg.norm(r[c] - r[hydrogen[0]]))
        self.assertEqual(len([j for j in range(len(r)) if r_xyz['symbols'][j] == 'H'
                              and np.linalg.norm(r[j] - r[nearest_carbon]) < 1.2]), 3)

    def test_isomerization_endpoints(self):
        """n-C3H7 -> i-C3H7: the product is aligned onto the reactant in the reactant's atom order."""
        r_xyz, p_xyz = neb_endpoints(self.isomerization)
        self.assertEqual(r_xyz['symbols'], p_xyz['symbols'])
        r, p = np.array(r_xyz['coords']), np.array(p_xyz['coords'])
        unaligned = np.array(self.isomerization.get_products_xyz(return_format=dict)['coords'])
        rmsd = lambda a, b: np.sqrt(((a - b) ** 2).sum(axis=1).mean())
        self.assertLessEqual(rmsd(p, r), rmsd(unaligned, r) + 1e-8)
        np.testing.assert_allclose(distances(p), distances(unaligned), atol=1e-8)  # a rigid motion only

    def test_ho2_elimination_endpoints(self):
        """C2H5O2 -> C2H4 + HO2: HO2, whose three atoms all change bonds, keeps its own geometry and leaves C2H4."""
        r_xyz, p_xyz = neb_endpoints(self.ho2_elimination)
        p = np.array(p_xyz['coords'])
        fragments = side_fragments(self.ho2_elimination, well=1)
        ho2 = next(f for f in fragments if len(f) == 3)
        ethylene = next(f for f in fragments if len(f) == 6)
        self.assertEqual(sorted(p_xyz['symbols'][i] for i in ho2), ['H', 'O', 'O'])
        h = next(i for i in ho2 if p_xyz['symbols'][i] == 'H')
        self.assertLess(min(np.linalg.norm(p[h] - p[i]) for i in ho2 if i != h), 1.05)  # an O-H bond
        self.assertGreaterEqual(np.min(np.linalg.norm(p[ho2][:, None] - p[ethylene][None], axis=2)),
                                FRAGMENT_SEPARATION)

    def test_near_attack_folding(self):
        """C2H5O2 -> CH2CH2OOH (1,4-H shift): the reactant endpoint is folded toward the H transfer."""
        r_xyz, p_xyz = neb_endpoints(self.h_shift)
        formed, _ = self.h_shift.get_formed_and_broken_bonds()
        (i, j), = formed
        folded = np.linalg.norm(np.array(r_xyz['coords'][i]) - np.array(r_xyz['coords'][j]))
        unfolded_xyz = self.h_shift.get_reactants_xyz(return_format=dict)
        unfolded = np.linalg.norm(np.array(unfolded_xyz['coords'][i]) - np.array(unfolded_xyz['coords'][j]))
        self.assertLessEqual(folded, unfolded + 1e-6)
        self.assertLess(folded, 3.0)
        self.assertEqual(r_xyz['symbols'], p_xyz['symbols'])

    def test_candidate_endpoints(self):
        """Several atom maps give distinct bands, shortest first, and the reaction keeps its own map."""
        original = list(self.dissociation.atom_map)
        bands = candidate_endpoints(self.dissociation, n_bands=3)
        self.assertGreaterEqual(len(bands), 2)  # which of the three methyl H atoms leaves
        self.assertLessEqual(len(bands), 3)
        self.assertEqual([b['path_rmsd'] for b in bands], sorted(b['path_rmsd'] for b in bands))
        self.assertEqual(len({tuple(b['atom_map']) for b in bands}), len(bands))
        for band in bands:
            self.assertEqual(band['xyz']['symbols'], band['product_xyz']['symbols'])
        self.assertEqual(list(self.dissociation.atom_map), original)
        self.assertEqual(len(candidate_endpoints(self.dissociation, n_bands=1)), 1)

    def test_moved_h_product(self):
        """A hydrogen transfer gets a product endpoint in which only the migrating H moved, onto its acceptor."""
        r_xyz, _ = neb_endpoints(self.h_shift)
        formed, broken = self.h_shift.get_formed_and_broken_bonds()
        transfers = h_transfers(self.h_shift, formed, broken, tuple(r_xyz['symbols']))
        self.assertEqual(len(transfers), 1)
        (h, acceptor), = transfers
        self.assertEqual((r_xyz['symbols'][h], r_xyz['symbols'][acceptor]), ('H', 'O'))
        mol = self.h_shift.r_species[0].mol
        neighbors = {i: {mol.atoms.index(n) for n in atom.edges} for i, atom in enumerate(mol.atoms)}
        moved = moved_h_product(r_xyz, transfers, neighbors)
        r, p = np.array(r_xyz['coords']), np.array(moved['coords'])
        self.assertAlmostEqual(float(np.linalg.norm(p[h] - p[acceptor])), 0.97, places=6)
        for n in neighbors[acceptor] - {h}:  # a sensible angle at the acceptor
            cosine = np.dot(p[h] - p[acceptor], p[n] - p[acceptor]) / 0.97 / np.linalg.norm(p[n] - p[acceptor])
            self.assertGreaterEqual(np.degrees(np.arccos(cosine)), 99.9)
        others = [i for i in range(len(r)) if i != h]
        np.testing.assert_allclose(p[others], r[others])
        # Not a hydrogen transfer.
        formed, broken = self.dissociation.get_formed_and_broken_bonds()
        self.assertIsNone(h_transfers(self.dissociation, formed, broken, tuple(r_xyz['symbols'])))
        # The candidates include one moved-H band at most.
        bands = candidate_endpoints(self.h_shift, n_bands=3)
        self.assertLessEqual(sum(b['moved_h'] for b in bands), 1)
        for band in bands:
            self.assertEqual('keep_bonds' in band, band['moved_h'])

    def test_resolve_level(self):
        """The band runs at an ASE level; a DFT TS-guess level is not one."""
        self.assertEqual(resolve_level(self.uma).method, 'uma-s-1p2')
        with self.assertRaises(DependencyError):
            resolve_level(Level(method='wb97xd', basis='def2svp'))
        with self.assertRaises(DependencyError):
            resolve_level(None)

    def test_write_input_file(self):
        """The adapter writes a 'neb' job for ase_script.py with both endpoints and the band settings."""
        job = ASENEBAdapter(execution_type='incore',
                            job_type='tsg',
                            project='test_ase_neb',
                            project_directory=os.path.join(self.project_directory, 'test_ase_neb'),
                            reactions=[self.dissociation],
                            level=self.uma,
                            tsg=0,
                            testing=True,
                            )
        self.assertEqual(job.job_adapter, 'ase_neb')
        os.makedirs(job.local_path, exist_ok=True)
        job.write_input_file()
        content = read_yaml_file(os.path.join(job.local_path, 'input.yml'))
        self.assertEqual(content['job_type'], 'neb')
        self.assertEqual(tuple(content['xyz']['symbols']), tuple(content['product_xyz']['symbols']))
        self.assertEqual(content['multiplicity'], 2)
        self.assertEqual(content['settings']['calculator'], 'uma')
        self.assertEqual(content['settings']['model'], 'uma-s-1p2')
        self.assertIn('neb_images', content['settings'])
        self.assertNotIn('neb_bands', content['settings'])  # the adapter's, not ase_script's
        self.assertGreaterEqual(len(content['bands']), 1)
        self.assertEqual(content['bands'][0]['xyz']['coords'], content['xyz']['coords'])
        with self.assertRaises(DependencyError):
            ASENEBAdapter(execution_type='incore', job_type='tsg', project='test_ase_neb',
                          project_directory=os.path.join(self.project_directory, 'test_ase_neb'),
                          reactions=[self.abstraction], level=self.uma, tsg=0, testing=True)


ETHYLENE_COORDS = [[float(x) for x in line.split()[1:]] for line in ETHYLENE.splitlines()]


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
