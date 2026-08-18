#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for the arc.job.adapters.ts.xy_addition module (XY_Addition_MultipleBond TS seed builder).
"""

import math
import unittest

from arc.job.adapters.ts.xy_addition import xy_addition
from arc.job.adapters.ts.seed_hub import get_ts_seeds, get_wrapper_constraints
from arc.reaction import ARCReaction
from arc.species import ARCSpecies


class TestXYAdditionSeed(unittest.TestCase):
    """Tests for the XY_Addition_MultipleBond 4-center TS seed builder."""

    def test_xy_addition_seed_is_four_center(self):
        """The seed for C=C + HCl -> CH3CH2Cl must be a 4-center arrangement:
        X (H) and Y (Cl) both approaching the (former) double bond, with the H-Cl bond retained."""
        ethylene = ARCSpecies(label='R1', smiles='C=C', multiplicity=1,
                              xyz='C 0 0 0.667\nC 0 0 -0.667\nH 0 0.921 1.232\n'
                                  'H 0 -0.921 1.232\nH 0 0.921 -1.232\nH 0 -0.921 -1.232')
        hcl = ARCSpecies(label='R2', smiles='Cl', multiplicity=1, xyz='Cl 0 0 0.071\nH 0 0 -1.211')
        chloroethane = ARCSpecies(label='P1', smiles='CCCl', multiplicity=1,
                                  xyz='C 1.61 -0.36 0\nC 0.49 0.66 0\nCl -1.14 -0.15 0\nH 1.57 -0.99 -0.88\n'
                                      'H 2.57 0.16 0\nH 0.53 1.30 0.88\nH 0.53 1.30 -0.88\nH -0.34 -0.5 0')
        rxn = ARCReaction(r_species=[ethylene, hcl], p_species=[chloroethane])
        # Reactants are [ethylene(0-5), HCl(6=Cl, 7=H)]; *1,*2 are the carbons, *3=X=H, *4=Y=Cl.
        label_maps = [
            {'*2': 0, '*1': 1, '*3': 7, '*4': 6},
            {'*2': 1, '*1': 0, '*3': 7, '*4': 6},
        ]
        rxn.product_dicts = [{'r_label_map': label_map} for label_map in label_maps]
        rxn.family = 'XY_Addition_MultipleBond'

        seeds = xy_addition(reaction=rxn)
        self.assertEqual(len(seeds), 2)
        self.assertEqual(seeds[0]['method'], 'Heuristics-XY')
        self.assertEqual(
            [seed['metadata']['reactive_atoms'] for seed in seeds],
            label_maps,
        )
        hub_seeds = get_ts_seeds(reaction=rxn)
        self.assertEqual([seed['metadata']['reactive_atoms'] for seed in hub_seeds], label_maps)
        self.assertEqual(
            [get_wrapper_constraints('crest', reaction=rxn, seed=seed) for seed in hub_seeds],
            [
                {
                    'atoms': tuple(label_map[label] for label in ('*1', '*2', '*3', '*4')),
                    'distance_pairs': (
                        (label_map['*1'], label_map['*3']),
                        (label_map['*2'], label_map['*4']),
                        (label_map['*3'], label_map['*4']),
                    ),
                }
                for label_map in label_maps
            ],
        )
        xyz = seeds[0]['xyz']
        self.assertEqual(len(xyz['symbols']), 8)  # atom count preserved

        coords = [tuple(c) for c in xyz['coords']]

        def dist(i, j):
            return math.sqrt(sum((coords[i][k] - coords[j][k]) ** 2 for k in range(3)))

        carbons = [i for i, s in enumerate(xyz['symbols']) if s == 'C']
        cl = [i for i, s in enumerate(xyz['symbols']) if s == 'Cl'][0]
        hydrogens = [i for i, s in enumerate(xyz['symbols']) if s == 'H']
        transferring_h = min(hydrogens, key=lambda h: dist(h, cl))

        min_cl_c = min(dist(cl, c) for c in carbons)
        min_h_c = min(dist(transferring_h, c) for c in carbons)
        # 4-center TS region: both Cl and the transferring H engage the carbons (forming bonds),
        # while the H-Cl bond is still present (breaking). Contrast with an H-transfer saddle where
        # Cl would be a spectator (> 2.9 A from every carbon).
        self.assertLess(min_cl_c, 2.6, 'Cl should be forming a bond to a carbon (not a spectator)')
        self.assertLess(min_h_c, 1.9, 'the transferring H should be forming a bond to a carbon')
        self.assertLess(dist(transferring_h, cl), 1.9, 'the breaking H-Cl bond should still be present')


if __name__ == '__main__':
    unittest.main()
