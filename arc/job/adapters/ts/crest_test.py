#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for arc.job.adapters.ts.crest
"""

import os
import tempfile
import types
import unittest
from unittest.mock import patch

from arc import constants
from arc.job.adapters.common import ts_adapters_by_rmg_family
from arc.job.adapters.ts import crest as crest_mod
from arc.job.adapters.ts.crest import CrestAdapter
from arc.job.adapters.ts.seed_hub import get_ts_seeds, get_wrapper_constraints
from arc.reaction import ARCReaction
from arc.settings.settings import ts_adapters
from arc.species.converter import str_to_xyz, xyz_to_str
from arc.species.species import ARCSpecies, TSGuess

WATER_XYZ = str_to_xyz("""O 0.0 0.0 0.0
                          H 0.0 0.0 0.96
                          H 0.9 0.0 0.0""")

OH_OH_XYZ = str_to_xyz("""O 0.00000000 -0.02752832 -1.20590500
                          H 0.00000000 -0.02752832 -0.03383145
                          O 0.00000000 -0.02752832  1.12142787
                          H 0.00000000  0.90131726  1.37454478""")

OH_OH_CONSTRAINTS = {'atoms': (0, 1, 2),
                     'distance_pairs': ((0, 1), (1, 2)),
                     'angle_atoms': (0, 1, 2),
                     }

CH4_OH_SEED_XYZ = str_to_xyz("""C  0.00000000  0.00000000  0.00000000
                                H  0.00000000  0.00000000  1.30000000
                                H  1.03000000  0.00000000 -0.36000000
                                H -0.51000000 -0.89000000 -0.36000000
                                H -0.51000000  0.89000000 -0.36000000
                                O  0.00000000  0.00000000  2.58000000
                                H  0.90000000  0.00000000  2.90000000""")

CH4_OH_RESULT_XYZ = str_to_xyz("""C  0.01000000  0.00000000  0.00000000
                                  H  0.00000000  0.01000000  1.31000000
                                  H  1.04000000  0.00000000 -0.36000000
                                  H -0.51000000 -0.90000000 -0.36000000
                                  H -0.51000000  0.90000000 -0.35000000
                                  O  0.00000000  0.01000000  2.59000000
                                  H  0.91000000  0.00000000  2.90000000""")

CH4_OH_SEED = {'xyz': CH4_OH_SEED_XYZ,
               'family': 'H_Abstraction',
               'method': 'Heuristics',
               'source_adapter': 'heuristics',
               'metadata': {'reactive_atoms': {'A': 0, 'H': 1, 'B': 5}},
               }


HYDROLYSIS_FAMILIES = ('carbonyl_based_hydrolysis', 'ether_hydrolysis', 'nitrile_hydrolysis')

# A methyl formate + water four-center hydrolysis TS seed, as heuristics.hydrolysis() generates it.
# Its atom roles are a=2 (the carbonyl carbon), b=1 (the leaving ester oxygen), e=3, o=8 (the water
# oxygen), h1=9 (the transferring water hydrogen) and d=7; atom 10 is the spectator water hydrogen.
HYDROLYSIS_SEED_XYZ = str_to_xyz("""C  0.14934638 -1.04077992 -1.29688828
                                    O  0.14934638 -1.04077992  0.13108654
                                    C  0.14934638  0.57218659  0.84211324
                                    O -0.41188584  1.56260964  0.40276657
                                    H -0.83434378 -0.75219743 -1.68052037
                                    H  0.36843817 -2.05708179 -1.63593228
                                    H  0.92676740 -0.37254154 -1.68052028
                                    H  0.14934645  0.47946490  1.93926498
                                    O  1.84045116 -0.06034836  1.24530570
                                    H  1.33683018 -0.95702439  0.60778079
                                    H  1.85273271 -0.25279783  2.19594358""")

HYDROLYSIS_POSITIONAL_INDICES = [2, 1, 3, 8, 9, 7]

HYDROLYSIS_REACTIVE_ATOMS = {'a': 2, 'b': 1, 'o': 8, 'h1': 9}

# All six pairwise separations among a, b, o and h1, zero-based.
HYDROLYSIS_DISTANCE_PAIRS = ((2, 1), (2, 8), (2, 9), (1, 8), (1, 9), (8, 9))


def make_ch4_oh_reaction() -> ARCReaction:
    """Return a CH4 + OH <=> CH3 + H2O ARCReaction."""
    return ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'),
                                  ARCSpecies(label='OH', smiles='[OH]')],
                       p_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                  ARCSpecies(label='H2O', smiles='O')],
                       )


def make_hydrolysis_seed(family: str = 'carbonyl_based_hydrolysis',
                         indices=None,
                         reactive_atoms: bool = True,
                         ) -> dict:
    """
    Return a hydrolysis seed entry in the schema that get_ts_seeds() produces.

    Args:
        family (str, optional): The hydrolysis family to label the seed with.
        indices (optional): The raw ``indices`` metadata entry. Defaults to the positional
                            ``(a, b, e, o, h1, d)`` sequence that hydrolysis() emits.
        reactive_atoms (bool, optional): Whether to also carry the named ``reactive_atoms`` dict.

    Returns:
        dict: The seed entry.
    """
    metadata = {'indices': list(HYDROLYSIS_POSITIONAL_INDICES) if indices is None else indices}
    if reactive_atoms:
        metadata['reactive_atoms'] = dict(HYDROLYSIS_REACTIVE_ATOMS)
    return {'xyz': HYDROLYSIS_SEED_XYZ,
            'family': family,
            'method': 'Heuristics',
            'source_adapter': 'heuristics',
            'metadata': metadata,
            }


def make_methyl_formate_hydrolysis_reaction() -> ARCReaction:
    """Return a methyl formate + water <=> formic acid + methanol ARCReaction."""
    return ARCReaction(r_species=[ARCSpecies(label='ester', smiles='COC=O'),
                                  ARCSpecies(label='H2O', smiles='O')],
                       p_species=[ARCSpecies(label='formicacid', smiles='C(=O)O'),
                                  ARCSpecies(label='methanol', smiles='CO')],
                       family='carbonyl_based_hydrolysis',
                       )


class CrestTestCase(unittest.TestCase):
    """A base class providing a temporary directory and CREST module-global patching."""

    def setUp(self):
        """Create a temporary directory that is removed after the test."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def patch_crest_module(self, **overrides):
        """
        Patch the CREST module globals for the duration of the test.

        Args:
            overrides: Module global names mapped to the values to patch them with. Any global that
                       is not overridden gets a local-PBS default suitable for input generation.

        Returns:
            module: The patched ``arc.job.adapters.ts.crest`` module.
        """
        values = {'settings': {'submit_filenames': {'PBS': 'submit.sh', 'HTCondor': 'submit.sub'}},
                  'submit_scripts': {'local': {}},
                  'CREST_PATH': '/usr/bin/crest',
                  'CREST_ENV_PATH': '',
                  'SERVERS': {'local': {'cluster_soft': 'pbs', 'cpus': 4, 'memory': 8, 'queue': 'testq'}},
                  }
        values.update(overrides)
        for name, value in values.items():
            patcher = patch.object(crest_mod, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return crest_mod


class TestCrestInputGeneration(CrestTestCase):
    """Tests for CREST input generation."""

    def test_creates_valid_input_files(self):
        """Ensure CREST inputs are written with the expected content and format."""
        self.patch_crest_module(submit_scripts={'local': {
            'crest': ('#PBS -q {queue}\n'
                      '#PBS -N {name}\n'
                      '#PBS -l select=1:ncpus={cpus}:mem={memory}gb\n'),
            'crest_job': '{activation_line}\ncd {path}\n{commands}\n',
        }})

        crest_dir = crest_mod.crest_ts_conformer_search(WATER_XYZ, 0, 1, 2, self.tmpdir.name, 0)

        coords_path = os.path.join(crest_dir, 'coords.ref')
        constraints_path = os.path.join(crest_dir, 'constraints.inp')
        submit_path = os.path.join(crest_dir, 'submit.sh')

        self.assertTrue(os.path.exists(coords_path))
        self.assertTrue(os.path.exists(constraints_path))
        self.assertTrue(os.path.exists(submit_path))

        with open(coords_path) as f:
            coords = f.read().strip().splitlines()
        self.assertEqual(coords[0].strip(), '$coord')
        self.assertEqual(coords[-1].strip(), '$end')
        self.assertEqual(len(coords) - 2, len(WATER_XYZ['symbols']))

        with open(constraints_path) as f:
            constraints = f.read()
        self.assertIn('atoms: 1, 2, 3', constraints)
        self.assertIn('force constant=0.5', constraints)
        self.assertIn('reference=coords.ref', constraints)
        self.assertIn('distance: 1, 2, auto', constraints)
        self.assertIn('distance: 2, 3, auto', constraints)
        self.assertTrue(constraints.strip().endswith('$end'))

    def test_coords_ref_is_in_bohr_with_turbomole_column_order(self):
        """coords.ref holds ``x y z SYMBOL`` lines whose coordinates are in Bohr, not Angstrom."""
        self.patch_crest_module()

        crest_dir = crest_mod.crest_ts_conformer_search(xyz_guess=WATER_XYZ,
                                                        constraints={'atoms': (0, 1, 2),
                                                                     'distance_pairs': ((0, 1), (1, 2))},
                                                        path=self.tmpdir.name,
                                                        xyz_crest_int=0,
                                                        )
        with open(os.path.join(crest_dir, 'coords.ref')) as f:
            lines = f.read().strip().splitlines()
        atom_lines = lines[1:-1]
        self.assertEqual(len(atom_lines), 3)

        parsed = list()
        for line, symbol in zip(atom_lines, WATER_XYZ['symbols']):
            tokens = line.split()
            self.assertEqual(len(tokens), 4)
            self.assertEqual(tokens[3], symbol)
            parsed.append(tuple(float(token) for token in tokens[:3]))

        for parsed_coords, expected_coords in zip(parsed, WATER_XYZ['coords']):
            for parsed_coord, expected_coord in zip(parsed_coords, expected_coords):
                self.assertAlmostEqual(parsed_coord, expected_coord * constants.angstrom_to_bohr, places=6)

        self.assertAlmostEqual(parsed[1][2], 0.96 * constants.angstrom_to_bohr, places=6)
        self.assertNotAlmostEqual(parsed[1][2], 0.96, places=3)

    def test_charge_and_multiplicity_flags(self):
        """--chrg and --uhf are written for a charged open-shell system and omitted otherwise."""
        self.patch_crest_module()
        constraints = {'atoms': (0, 1, 2), 'distance_pairs': ((0, 1), (1, 2))}

        anion_dir = crest_mod.crest_ts_conformer_search(xyz_guess=WATER_XYZ,
                                                        constraints=constraints,
                                                        path=self.tmpdir.name,
                                                        xyz_crest_int=0,
                                                        charge=-1,
                                                        multiplicity=2,
                                                        )
        with open(os.path.join(anion_dir, 'submit.sh')) as f:
            anion_submit = f.read()
        self.assertIn('--chrg -1', anion_submit)
        self.assertIn('--uhf 1', anion_submit)

        neutral_dir = crest_mod.crest_ts_conformer_search(xyz_guess=WATER_XYZ,
                                                          constraints=constraints,
                                                          path=self.tmpdir.name,
                                                          xyz_crest_int=1,
                                                          charge=0,
                                                          multiplicity=1,
                                                          )
        with open(os.path.join(neutral_dir, 'submit.sh')) as f:
            neutral_submit = f.read()
        self.assertNotIn('--chrg', neutral_submit)
        self.assertNotIn('--uhf', neutral_submit)

    def test_h_abstraction_pins_three_distances_and_writes_no_angle(self):
        """The H-abstraction $constrain block pins A--H, H--B and A--B, which fixes the triad."""
        self.patch_crest_module()

        crest_path = crest_mod.crest_ts_conformer_search(xyz_guess=OH_OH_XYZ,
                                                         constraints=OH_OH_CONSTRAINTS,
                                                         path=self.tmpdir.name,
                                                         xyz_crest_int=3,
                                                         )
        with open(os.path.join(crest_path, 'constraints.inp')) as f:
            constraints_text = f.read()
        self.assertIn('distance: 1, 2, auto', constraints_text)
        self.assertIn('distance: 2, 3, auto', constraints_text)
        self.assertIn('distance: 1, 3, auto', constraints_text)
        self.assertNotIn('angle:', constraints_text)
        self.assertIn('$metadyn\n  atoms: 4\n', constraints_text)

    def test_no_metadyn_block_when_every_atom_is_reactive(self):
        """An atom-less $metadyn block is not written; the reactive zone covering all atoms is logged."""
        self.patch_crest_module()

        with self.assertLogs('arc', level='WARNING') as log:
            crest_path = crest_mod.crest_ts_conformer_search(xyz_guess=WATER_XYZ,
                                                             constraints={'atoms': (0, 1, 2),
                                                                          'distance_pairs': ((0, 1), (1, 2))},
                                                             path=self.tmpdir.name,
                                                             xyz_crest_int=0,
                                                             )
        with open(os.path.join(crest_path, 'constraints.inp')) as f:
            constraints_text = f.read()
        self.assertNotIn('$metadyn', constraints_text)
        self.assertIn('$constrain', constraints_text)
        self.assertTrue(constraints_text.strip().endswith('$end'))
        self.assertTrue(any('$metadyn' in message for message in log.output))

    def test_stale_crest_best_is_removed_before_launching(self):
        """A crest_best.xyz left by an earlier run must not survive into a new run's directory."""
        self.patch_crest_module()
        stale_dir = os.path.join(self.tmpdir.name, 'crest_0')
        os.makedirs(stale_dir)
        stale_path = os.path.join(stale_dir, 'crest_best.xyz')
        with open(stale_path, 'w') as f:
            f.write(f"4\nstale geometry\n{xyz_to_str(OH_OH_XYZ)}\n")

        crest_path = crest_mod.crest_ts_conformer_search(xyz_guess=OH_OH_XYZ,
                                                         constraints=OH_OH_CONSTRAINTS,
                                                         path=self.tmpdir.name,
                                                         xyz_crest_int=0,
                                                         )
        self.assertEqual(crest_path, stale_dir)
        self.assertFalse(os.path.exists(stale_path))

    def test_creates_submit_file_without_crest_templates(self):
        """Fallback submit template generation works when submit.py has no CREST templates."""
        self.patch_crest_module()

        crest_dir = crest_mod.crest_ts_conformer_search(xyz_guess=WATER_XYZ,
                                                        constraints=OH_OH_CONSTRAINTS,
                                                        path=self.tmpdir.name,
                                                        xyz_crest_int=1,
                                                        )
        submit_path = os.path.join(crest_dir, 'submit.sh')
        self.assertTrue(os.path.exists(submit_path))
        with open(submit_path) as f:
            submit_text = f.read()
        self.assertIn('#PBS -q testq', submit_text)
        self.assertIn('coords.ref --cinp constraints.inp --noreftopo -T 4', submit_text)

    def test_creates_xy_distance_constraints_and_validates_completed_geometry(self):
        """Write all three XY recipe distances and reject a geometry that loses one."""
        reference_xyz = str_to_xyz("""C  0.0000 0.0000  0.6670
                                      C  0.0000 0.0000 -0.6670
                                      H  0.0000 0.9210  1.2320
                                      H  0.0000 -0.9210 1.2320
                                      H  0.0000 0.9210 -1.2320
                                      H  0.0000 -0.9210 -1.2320
                                      Cl 0.0000 2.1000 -0.6670
                                      H  0.0000 1.6000  0.6670""")
        constraints = {'atoms': (1, 0, 7, 6),
                       'distance_pairs': ((1, 7), (0, 6), (7, 6)),
                       }
        self.patch_crest_module()

        crest_path = crest_mod.crest_ts_conformer_search(xyz_guess=reference_xyz,
                                                         constraints=constraints,
                                                         path=self.tmpdir.name,
                                                         xyz_crest_int=2,
                                                         )
        with open(os.path.join(crest_path, 'constraints.inp')) as f:
            constraints_text = f.read()
        self.assertIn('atoms: 2, 1, 8, 7', constraints_text)
        self.assertIn('distance: 2, 8, auto', constraints_text)
        self.assertIn('distance: 1, 7, auto', constraints_text)
        self.assertIn('distance: 8, 7, auto', constraints_text)
        self.assertIn('atoms: 3, 4, 5, 6', constraints_text)

        crest_best_path = os.path.join(crest_path, 'crest_best.xyz')
        jobs = {'123': {'path': crest_path, 'status': 'done'}}
        references = {crest_path: {'xyz': reference_xyz, 'constraints': constraints}}
        with open(crest_best_path, 'w') as f:
            f.write(f"8\nCREST geometry\n{xyz_to_str(reference_xyz)}\n")
        self.assertEqual(crest_mod.process_completed_jobs(jobs, references), [reference_xyz])

        dissociated_xyz = dict(reference_xyz)
        dissociated_coords = list(reference_xyz['coords'])
        dissociated_coords[6] = (0.0, 8.0, -0.6670)
        dissociated_xyz['coords'] = tuple(dissociated_coords)
        with open(crest_best_path, 'w') as f:
            f.write(f"8\nCREST geometry\n{xyz_to_str(dissociated_xyz)}\n")
        self.assertEqual(crest_mod.process_completed_jobs(jobs, references), [])

    def test_unsupported_cluster_software_skips_instead_of_raising(self):
        """An unsupported local queue must skip CREST, not abort the ARC run.

        crest_ts_conformer_search() is reached from job.execute(), which Scheduler.run_job calls
        with no try/except, so raising there kills the whole project.
        """
        self.patch_crest_module(SERVERS={'local': {'cluster_soft': 'Slurm', 'cpus': 4}})
        result = crest_mod.crest_ts_conformer_search(WATER_XYZ,
                                                     path=self.tmpdir.name,
                                                     constraints={'atoms': [0, 1],
                                                                  'distance_pairs': [(0, 1)]},
                                                     )
        self.assertIsNone(result)

    def test_constructor_and_legacy_argument_validation(self):
        """The adapter rejects a missing reaction, and the legacy call rejects incomplete indices."""
        with self.assertRaises(ValueError):
            CrestAdapter(job_type='tsg',
                         reactions=None,
                         testing=True,
                         project='test_CrestAdapter',
                         project_directory=self.tmpdir.name,
                         )
        with self.assertRaises(ValueError):
            crest_mod.crest_ts_conformer_search(xyz_guess=WATER_XYZ,
                                                a_atom=None,
                                                h_atom=1,
                                                b_atom=2,
                                                path=self.tmpdir.name,
                                                )


class TestCrestBackupSeeds(CrestTestCase):
    """Tests for seeding CREST from TS guesses that other adapters produced."""

    def test_get_backup_ts_seeds_selects_successful_non_crest_guesses(self):
        """The backup seed picker skips CREST/failed guesses, prefers opt_xyz, and dedups."""
        from arc.job.adapters.ts.seed_hub import get_backup_ts_seeds

        seed_geom = OH_OH_XYZ
        other_geom = str_to_xyz("""O 0.10000000 -0.02752832 -1.20590500
                                   H 0.00000000 -0.02752832 -0.03383145
                                   O 0.00000000 -0.02752832  1.12142787
                                   H 0.00000000  0.90131726  1.37454478""")

        def make_tsg(method, success, initial_xyz, opt_xyz=None):
            return types.SimpleNamespace(method=method, success=success,
                                         initial_xyz=initial_xyz, opt_xyz=opt_xyz)

        ts_guesses = [
            make_tsg('crest', True, other_geom),
            make_tsg('autotst', False, other_geom),
            make_tsg('autotst', True, other_geom, opt_xyz=seed_geom),
            make_tsg('heuristics', True, seed_geom),
        ]
        reaction = types.SimpleNamespace(family='H_Abstraction',
                                         ts_species=types.SimpleNamespace(ts_guesses=ts_guesses),
                                         )

        seeds = get_backup_ts_seeds(reaction, exclude_method='crest')
        self.assertEqual(len(seeds), 1)
        seed = seeds[0]
        self.assertEqual(seed['xyz'], seed_geom)
        self.assertEqual(seed['source_adapter'], 'autotst')
        self.assertEqual(seed['family'], 'H_Abstraction')
        self.assertEqual(seed['metadata'], {})

        self.assertEqual(get_backup_ts_seeds(types.SimpleNamespace(family='H_Abstraction')), [])
        empty_rxn = types.SimpleNamespace(family='H_Abstraction',
                                          ts_species=types.SimpleNamespace(ts_guesses=[]))
        self.assertEqual(get_backup_ts_seeds(empty_rxn), [])

    def test_backup_seed_drives_constraint_derivation_and_input_generation(self):
        """An external TS guess seeds CREST: constraints are re-derived from its geometry."""
        from arc.job.adapters.ts.seed_hub import get_backup_ts_seeds, get_wrapper_constraints

        reaction = types.SimpleNamespace(
            family='H_Abstraction',
            ts_species=types.SimpleNamespace(ts_guesses=[
                types.SimpleNamespace(method='autotst', success=True,
                                      initial_xyz=OH_OH_XYZ, opt_xyz=None),
            ]),
        )

        seeds = get_backup_ts_seeds(reaction, exclude_method='crest')
        self.assertEqual(len(seeds), 1)
        constraints = get_wrapper_constraints(wrapper='crest', reaction=reaction, seed=seeds[0])
        self.assertIsNotNone(constraints)
        self.assertIn('atoms', constraints)
        self.assertIn('distance_pairs', constraints)
        self.assertIn('angle_atoms', constraints)

        self.patch_crest_module()
        crest_path = crest_mod.crest_ts_conformer_search(xyz_guess=seeds[0]['xyz'],
                                                         constraints=constraints,
                                                         path=self.tmpdir.name,
                                                         xyz_crest_int=0,
                                                         )
        self.assertTrue(os.path.exists(os.path.join(crest_path, 'coords.ref')))
        with open(os.path.join(crest_path, 'constraints.inp')) as f:
            constraints_text = f.read()
        self.assertIn('atoms:', constraints_text)
        self.assertIn('reference=coords.ref', constraints_text)
        self.assertIn('$metadyn', constraints_text)


class TestCrestAvailability(CrestTestCase):
    """Tests for crest_available()."""

    def test_crest_available_matrix(self):
        """CREST is available only with a local server and at least one CREST path configured."""
        with patch.object(crest_mod, 'SERVERS', {}), \
                patch.object(crest_mod, 'CREST_PATH', '/usr/bin/crest'), \
                patch.object(crest_mod, 'CREST_ENV_PATH', ''):
            self.assertFalse(crest_mod.crest_available())

        with patch.object(crest_mod, 'SERVERS', {'local': {'cluster_soft': 'pbs'}}), \
                patch.object(crest_mod, 'CREST_PATH', None), \
                patch.object(crest_mod, 'CREST_ENV_PATH', None):
            self.assertFalse(crest_mod.crest_available())

        with patch.object(crest_mod, 'SERVERS', {'local': {'cluster_soft': 'pbs'}}), \
                patch.object(crest_mod, 'CREST_PATH', '/usr/bin/crest'), \
                patch.object(crest_mod, 'CREST_ENV_PATH', None):
            self.assertTrue(crest_mod.crest_available())

        with patch.object(crest_mod, 'SERVERS', {'local': {'cluster_soft': 'pbs'}}), \
                patch.object(crest_mod, 'CREST_PATH', None), \
                patch.object(crest_mod, 'CREST_ENV_PATH', 'conda activate crest_env'):
            self.assertTrue(crest_mod.crest_available())


class TestCrestGating(CrestTestCase):
    """Tests for the reactive-core gate and the seed cap."""

    @staticmethod
    def make_rxn(family, reactant_atom_counts):
        """Return a minimal reaction stand-in carrying a family and reactant atom counts."""
        return types.SimpleNamespace(
            family=family,
            r_species=[types.SimpleNamespace(number_of_atoms=n) for n in reactant_atom_counts],
        )

    def test_tiny_system_gate_skips_crest_only_for_whole_molecule_core(self):
        """CREST is gated off only when at most one atom sits outside the H-abstraction triad."""
        self.assertTrue(crest_mod._crest_reactive_core_covers_molecule(self.make_rxn('H_Abstraction', [2, 2])))
        self.assertTrue(crest_mod._crest_reactive_core_covers_molecule(self.make_rxn('H_Abstraction', [1, 2])))
        # 5 atoms, i.e. 2 spectators, is the first case that must NOT be gated off.
        self.assertFalse(crest_mod._crest_reactive_core_covers_molecule(self.make_rxn('H_Abstraction', [3, 2])))
        self.assertFalse(crest_mod._crest_reactive_core_covers_molecule(self.make_rxn('H_Abstraction', [5, 2])))
        self.assertFalse(
            crest_mod._crest_reactive_core_covers_molecule(self.make_rxn('XY_Addition_MultipleBond', [2, 2]))
        )
        self.assertFalse(crest_mod._crest_reactive_core_covers_molecule(self.make_rxn('H_Abstraction', [None, 2])))

    def test_tiny_system_gate_covers_the_four_center_hydrolysis_core(self):
        """A hydrolysis core of four atoms is gated off only when at most one spectator remains."""
        for family in HYDROLYSIS_FAMILIES:
            # 5 atoms, i.e. 1 spectator outside the a/b/o/h1 core, is gated off.
            self.assertTrue(crest_mod._crest_reactive_core_covers_molecule(self.make_rxn(family, [2, 3])),
                            msg=f'{family} was not gated off at 5 atoms.')
            self.assertTrue(crest_mod._crest_reactive_core_covers_molecule(self.make_rxn(family, [1, 3])))
            # 6 atoms, e.g. HCN + H2O, is the smallest real substrate and must NOT be gated off.
            self.assertFalse(crest_mod._crest_reactive_core_covers_molecule(self.make_rxn(family, [3, 3])),
                             msg=f'{family} was wrongly gated off at 6 atoms.')
            self.assertFalse(crest_mod._crest_reactive_core_covers_molecule(self.make_rxn(family, [8, 3])))
            self.assertFalse(crest_mod._crest_reactive_core_covers_molecule(self.make_rxn(family, [None, 3])))
        self.assertFalse(crest_mod._crest_reactive_core_covers_molecule(self.make_rxn('intra_H_migration', [2, 3])))

    def test_select_diverse_seeds_keeps_distinct_geometries(self):
        """The seed selector keeps geometrically distinct seeds and drops near-duplicates."""
        def seed(offset):
            coords = tuple(tuple(coord + offset for coord in atom_coords) for atom_coords in WATER_XYZ['coords'])
            return {'xyz': {'symbols': WATER_XYZ['symbols'], 'isotopes': WATER_XYZ['isotopes'], 'coords': coords}}

        seeds = [seed(0.0), seed(0.001), seed(0.002), seed(5.0)]
        selected = crest_mod._select_diverse_seeds(seeds=seeds, max_seeds=2)
        self.assertEqual(len(selected), 2)
        self.assertIn(seeds[0], selected)
        self.assertIn(seeds[3], selected)

        self.assertEqual(crest_mod._select_diverse_seeds(seeds=seeds, max_seeds=10), seeds)

        incomparable = [{'xyz': WATER_XYZ}, {'xyz': OH_OH_XYZ}, {'xyz': WATER_XYZ}]
        self.assertEqual(crest_mod._select_diverse_seeds(seeds=incomparable, max_seeds=2), incomparable[:2])


class TestCrestJobLifecycle(CrestTestCase):
    """Test the parts of the CREST lifecycle that can hang or abort the whole ARC run."""

    def test_monitor_terminates_on_an_errored_job(self):
        """A queue reporting 'errored' must end the wait.

        check_job_status() reports 'done', 'running' or 'errored' and never 'failed', so a terminal
        set of ('done', 'failed') left an errored job polled forever, blocking the scheduler.
        """
        crest_jobs = {'1': {'path': os.path.join(self.tmpdir.name, 'crest_0'), 'status': 'running'}}
        calls = {'n': 0}

        def fake_status(job_id):
            calls['n'] += 1
            return 'errored'

        with patch.object(crest_mod, 'check_job_status', fake_status):
            crest_mod.monitor_crest_jobs(crest_jobs, check_interval=0, max_time_seconds=1)
        self.assertEqual(crest_jobs['1']['status'], 'errored')
        # Exactly one poll: 'errored' is terminal. If it were not, the loop would spin until the
        # wall-clock deadline and poll many times, which is the hang this guards against.
        self.assertEqual(calls['n'], 1)

    def test_monitor_cancels_and_ingests_on_the_wall_time(self):
        """A timed-out job is cancelled; a geometry it already wrote is still ingested."""
        finished_path = os.path.join(self.tmpdir.name, 'crest_0')
        empty_path = os.path.join(self.tmpdir.name, 'crest_1')
        os.makedirs(finished_path)
        os.makedirs(empty_path)
        with open(os.path.join(finished_path, 'crest_best.xyz'), 'w') as f:
            f.write(f"4\nCREST geometry\n{xyz_to_str(OH_OH_XYZ)}\n")
        crest_jobs = {'1': {'path': finished_path, 'status': 'running'},
                      '2': {'path': empty_path, 'status': 'running'},
                      }
        deleted = list()

        with patch.object(crest_mod, 'check_job_status', lambda job_id: 'running'), \
                patch.object(crest_mod, 'delete_job', deleted.append):
            crest_mod.monitor_crest_jobs(crest_jobs, check_interval=0, max_time_seconds=0)

        self.assertEqual(sorted(deleted), ['1', '2'])
        self.assertEqual(crest_jobs['1']['status'], 'done')
        self.assertEqual(crest_jobs['2']['status'], 'errored')

        references = {finished_path: {'xyz': OH_OH_XYZ, 'constraints': OH_OH_CONSTRAINTS}}
        self.assertEqual(crest_mod.process_completed_jobs(crest_jobs, references), [OH_OH_XYZ])

    def test_monitor_logs_a_cancellation_failure_and_keeps_going(self):
        """One failed cancellation must not stop the remaining jobs from being cancelled."""
        crest_jobs = {'1': {'path': os.path.join(self.tmpdir.name, 'crest_0'), 'status': 'running'},
                      '2': {'path': os.path.join(self.tmpdir.name, 'crest_1'), 'status': 'running'},
                      }
        deleted = list()

        def fake_delete(job_id):
            deleted.append(job_id)
            if job_id == '1':
                raise RuntimeError(f'Could not delete job {job_id}')

        with patch.object(crest_mod, 'check_job_status', lambda job_id: 'running'), \
                patch.object(crest_mod, 'delete_job', fake_delete), \
                self.assertLogs('arc', level='ERROR') as log:
            crest_mod.monitor_crest_jobs(crest_jobs, check_interval=0, max_time_seconds=0)

        self.assertEqual(sorted(deleted), ['1', '2'])
        self.assertEqual(crest_jobs['1']['status'], 'errored')
        self.assertEqual(crest_jobs['2']['status'], 'errored')
        self.assertTrue(any('Could not cancel the timed-out CREST job 1' in message for message in log.output))

    def test_failed_submission_does_not_collapse_jobs(self):
        """submit_job() signals a failed submission as (None, None) or as ('errored', '')."""
        submissions = [(None, None), ('errored', ''), ('running', '17'), ('running', '18')]
        paths = [os.path.join(self.tmpdir.name, f'crest_{i}') for i in range(4)]
        calls = {'n': 0}

        def fake_submit(path):
            calls['n'] += 1
            return submissions[calls['n'] - 1]

        with patch.object(crest_mod, 'submit_job', fake_submit):
            jobs = crest_mod.submit_crest_jobs(paths)

        self.assertNotIn(None, jobs)
        self.assertNotIn('', jobs)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(sorted(info['path'] for info in jobs.values()), sorted(paths[2:]))

    def test_process_completed_jobs_rejects_bad_geometries(self):
        """Colliding, non-finite and empty geometries are rejected; a valid one is accepted."""
        crest_path = os.path.join(self.tmpdir.name, 'crest_0')
        os.makedirs(crest_path)
        crest_best_path = os.path.join(crest_path, 'crest_best.xyz')
        jobs = {'123': {'path': crest_path, 'status': 'done'}}
        references = {crest_path: {'xyz': OH_OH_XYZ, 'constraints': OH_OH_CONSTRAINTS}}

        def write(content):
            with open(crest_best_path, 'w') as f:
                f.write(content)

        colliding_coords = list(OH_OH_XYZ['coords'])
        colliding_coords[3] = (colliding_coords[2][0] + 0.3, colliding_coords[2][1], colliding_coords[2][2])
        colliding_xyz = dict(OH_OH_XYZ)
        colliding_xyz['coords'] = tuple(colliding_coords)
        write(f"4\nCREST geometry\n{xyz_to_str(colliding_xyz)}\n")
        self.assertEqual(crest_mod.process_completed_jobs(jobs, references), [])

        write("""4
CREST geometry
O 0.00000000 -0.02752832 -1.20590500
H 0.00000000 -0.02752832 -0.03383145
O 0.00000000 -0.02752832  1.12142787
H 0.00000000  nan         1.37454478
""")
        self.assertEqual(crest_mod.process_completed_jobs(jobs, references), [])

        write('0\nCREST geometry\n')
        with self.assertLogs('arc', level='WARNING') as log:
            self.assertEqual(crest_mod.process_completed_jobs(jobs, references), [])
        self.assertTrue(any('is empty' in message for message in log.output),
                        msg=f'A zero-atom geometry must be rejected as empty:\n{log.output}')

        write(f"4\nCREST geometry\n{xyz_to_str(OH_OH_XYZ)}\n")
        self.assertEqual(crest_mod.process_completed_jobs(jobs, references), [OH_OH_XYZ])

    def test_process_completed_jobs_rejects_a_dissociated_reactive_triad(self):
        """Do not accept a crest_best.xyz whose acceptor has separated from the transferring H."""
        dissociated_xyz = str_to_xyz("""O -1.1644 0.0000 0.0000
                                        H  0.0000 0.0000 0.0000
                                        O  4.9000 0.0000 0.0000
                                        H  5.8703 0.0000 0.0000""")
        zeus_bad_xyz = str_to_xyz("""O -0.71236464  0.03765902 -0.02937463
                                     H -0.60136223 -0.77534746  0.43444583
                                     O  0.69301187  0.05895500  0.02917997
                                     H  0.90856791 -0.75830305 -0.43135588""")
        crest_path = os.path.join(self.tmpdir.name, 'crest_0')
        os.makedirs(crest_path)
        crest_best_path = os.path.join(crest_path, 'crest_best.xyz')
        with open(crest_best_path, 'w') as f:
            f.write(f"4\nCREST geometry\n{xyz_to_str(dissociated_xyz)}\n")

        jobs = {'123': {'path': crest_path, 'status': 'done'}}
        references = {crest_path: {'xyz': OH_OH_XYZ, 'constraints': OH_OH_CONSTRAINTS}}
        self.assertEqual(crest_mod.process_completed_jobs(jobs, crest_references={}), [])
        self.assertEqual(crest_mod.process_completed_jobs(jobs, crest_references=references), [])

        with open(crest_best_path, 'w') as f:
            f.write(f"4\nCREST geometry\n{xyz_to_str(zeus_bad_xyz)}\n")
        self.assertEqual(crest_mod.process_completed_jobs(jobs, crest_references=references), [])

        with open(crest_best_path, 'w') as f:
            f.write(f"4\nCREST geometry\n{xyz_to_str(OH_OH_XYZ)}\n")
        self.assertEqual(crest_mod.process_completed_jobs(jobs, crest_references=references), [OH_OH_XYZ])

    def test_preserves_reactive_constraints_validates_indices(self):
        """Malformed constraint indices are rejected rather than trusted."""
        valid = {'atoms': (0, 1, 2), 'distance_pairs': ((0, 1), (1, 2))}
        self.assertTrue(crest_mod._preserves_reactive_constraints(xyz=OH_OH_XYZ,
                                                                  reference_xyz=OH_OH_XYZ,
                                                                  constraints=valid))
        for constraints in [{'atoms': (0, 1, 1), 'distance_pairs': ((0, 1),)},
                            {'atoms': (0, 1, 9), 'distance_pairs': ((0, 1),)},
                            {'atoms': (0, 1, -1), 'distance_pairs': ((0, 1),)},
                            {'atoms': tuple(), 'distance_pairs': tuple()},
                            {'atoms': (0, 1, 2), 'distance_pairs': ((0, 3),)},
                            {'atoms': (0, 1, 2), 'distance_pairs': ((0, 1, 2),)},
                            ]:
            self.assertFalse(crest_mod._preserves_reactive_constraints(xyz=OH_OH_XYZ,
                                                                       reference_xyz=OH_OH_XYZ,
                                                                       constraints=constraints),
                             msg=f'{constraints} should have been rejected')


class TestCrestExecuteIncore(CrestTestCase):
    """Tests for the execute_incore() orchestration."""

    def setUp(self):
        """Build a CH4 + OH reaction and a CREST adapter writing into the temporary directory."""
        super().setUp()
        self.rxn = make_ch4_oh_reaction()
        self.patch_crest_module()
        self.adapter = CrestAdapter(job_type='tsg',
                                    reactions=[self.rxn],
                                    testing=True,
                                    project='test_CrestAdapter',
                                    project_directory=self.tmpdir.name,
                                    )
        os.makedirs(self.adapter.local_path, exist_ok=True)

    def run_execute_incore(self, crest_results, seeds=CH4_OH_SEED, submitted=None):
        """
        Run execute_incore() with the submission, monitoring and ingestion steps patched out.

        Args:
            crest_results (list): The geometries process_completed_jobs() should return.
            seeds: The seed entry, or list of seed entries, get_ts_seeds() should return.
            submitted (list, optional): A list collecting the CREST job directories submitted.

        Returns:
            list: The CREST job directories that were prepared.
        """
        prepared = submitted if submitted is not None else list()
        seed_list = seeds if isinstance(seeds, list) else [seeds]

        def fake_submit(crest_paths):
            prepared.extend(crest_paths)
            return {str(i): {'path': path, 'status': 'done'} for i, path in enumerate(crest_paths)}

        with patch.object(crest_mod, 'crest_available', lambda: True), \
                patch.object(crest_mod, 'get_ts_seeds', lambda **kwargs: list(seed_list)), \
                patch.object(crest_mod, 'submit_crest_jobs', fake_submit), \
                patch.object(crest_mod, 'monitor_crest_jobs', lambda *args, **kwargs: None), \
                patch.object(crest_mod, 'process_completed_jobs',
                             lambda *args, **kwargs: list(crest_results)):
            self.adapter.execute_incore()
        return prepared

    def test_execute_incore_appends_a_crest_ts_guess(self):
        """A CREST geometry becomes a successful TSGuess and is saved to the job directory."""
        self.run_execute_incore(crest_results=[CH4_OH_RESULT_XYZ])

        self.assertIsNotNone(self.rxn.ts_species)
        self.assertEqual(len(self.rxn.ts_species.ts_guesses), 1)
        ts_guess = self.rxn.ts_species.ts_guesses[0]
        self.assertEqual(ts_guess.method.lower(), 'crest')
        self.assertTrue(ts_guess.success)
        self.assertEqual(ts_guess.family, 'H_Abstraction')
        self.assertEqual(ts_guess.initial_xyz['symbols'], CH4_OH_RESULT_XYZ['symbols'])
        for guess_coords, expected_coords in zip(ts_guess.initial_xyz['coords'], CH4_OH_RESULT_XYZ['coords']):
            for guess_coord, expected_coord in zip(guess_coords, expected_coords):
                self.assertAlmostEqual(guess_coord, expected_coord, places=6)
        self.assertTrue(os.path.isfile(os.path.join(self.adapter.local_path, 'CREST_0.xyz')))

    def test_execute_incore_merges_a_duplicate_geometry(self):
        """A CREST geometry equal to an existing guess adds 'crest' instead of a new guess."""
        self.rxn.ts_species = ARCSpecies(label='TS', is_ts=True, charge=0, multiplicity=2)
        existing = TSGuess(method='autotst', success=True, family='H_Abstraction', xyz=CH4_OH_RESULT_XYZ)
        self.rxn.ts_species.append_ts_guess(existing)

        self.run_execute_incore(crest_results=[CH4_OH_RESULT_XYZ])

        self.assertEqual(len(self.rxn.ts_species.ts_guesses), 1)
        merged = self.rxn.ts_species.ts_guesses[0]
        self.assertEqual(merged.method, 'autotst')
        self.assertIn('crest', merged.method_sources)

    def test_execute_incore_caps_the_number_of_seeds(self):
        """Only MAX_CREST_SEEDS seeds are submitted, and dropping the rest is logged."""
        seeds = list()
        for index in range(crest_mod.MAX_CREST_SEEDS * 3):
            coords = tuple(tuple(coord + 0.05 * index for coord in atom_coords)
                           for atom_coords in CH4_OH_SEED_XYZ['coords'])
            xyz = {'symbols': CH4_OH_SEED_XYZ['symbols'],
                   'isotopes': CH4_OH_SEED_XYZ['isotopes'],
                   'coords': coords,
                   }
            seeds.append(dict(CH4_OH_SEED, xyz=xyz))

        submitted = list()
        with self.assertLogs('arc', level='INFO') as log:
            self.run_execute_incore(crest_results=[], seeds=seeds, submitted=submitted)

        self.assertEqual(len(submitted), crest_mod.MAX_CREST_SEEDS)
        self.assertEqual(len(set(submitted)), crest_mod.MAX_CREST_SEEDS)
        self.assertTrue(any(f'dropping the other {len(seeds) - crest_mod.MAX_CREST_SEEDS}' in message
                            for message in log.output),
                        msg=f'The dropped seeds were not logged:\n{log.output}')

    def test_execute_incore_names_job_directories_per_reaction(self):
        """The CREST job directory name carries the reaction index, so reactions cannot collide."""
        submitted = list()
        self.run_execute_incore(crest_results=[], submitted=submitted)
        self.assertEqual([os.path.basename(path) for path in submitted], ['crest_0_0'])

    def test_execute_incore_falls_back_to_an_external_ts_guess(self):
        """With no heuristic seed, CREST is seeded from another adapter's successful TS guess."""
        self.rxn.ts_species = ARCSpecies(label='TS', is_ts=True, charge=0, multiplicity=2)
        self.rxn.ts_species.append_ts_guess(
            TSGuess(method='autotst', success=True, family='H_Abstraction', xyz=CH4_OH_SEED_XYZ)
        )
        recorded = list()

        def fake_search(xyz_guess, **kwargs):
            recorded.append({'xyz': xyz_guess, 'constraints': kwargs.get('constraints')})
            crest_path = os.path.join(self.tmpdir.name, f"crest_{kwargs.get('xyz_crest_int')}")
            os.makedirs(crest_path, exist_ok=True)
            return crest_path

        with patch.object(crest_mod, 'crest_available', lambda: True), \
                patch.object(crest_mod, 'get_ts_seeds', lambda **kwargs: []), \
                patch.object(crest_mod, 'crest_ts_conformer_search', fake_search), \
                patch.object(crest_mod, 'submit_crest_jobs', lambda paths: {}), \
                patch.object(crest_mod, 'monitor_crest_jobs', lambda *args, **kwargs: None), \
                patch.object(crest_mod, 'process_completed_jobs', lambda *args, **kwargs: []):
            self.adapter.execute_incore()

        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]['xyz']['symbols'], CH4_OH_SEED_XYZ['symbols'])
        constraints = recorded[0]['constraints']
        self.assertIsNotNone(constraints)
        self.assertEqual(set(constraints['atoms']), {0, 1, 5})
        self.assertEqual(constraints['angle_atoms'][1], 1)
        for pair in constraints['distance_pairs']:
            self.assertEqual(len(pair), 2)
            self.assertTrue(all(atom in constraints['atoms'] for atom in pair))


class TestCrestHydrolysisConstraints(CrestTestCase):
    """Tests for the four-center hydrolysis CREST constraint specification."""

    def test_six_distance_pairs_for_every_hydrolysis_family(self):
        """Each hydrolysis family pins all six pairwise distances among a, b, o and h1."""
        for family in HYDROLYSIS_FAMILIES:
            rxn = types.SimpleNamespace(family=family)
            constraints = get_wrapper_constraints(wrapper='crest',
                                                  reaction=rxn,
                                                  seed=make_hydrolysis_seed(family=family),
                                                  )
            self.assertIsNotNone(constraints, msg=f'No CREST constraints were built for {family}.')
            self.assertEqual(constraints['atoms'], (2, 1, 8, 9))
            self.assertEqual(len(constraints['distance_pairs']), 6)
            self.assertEqual({frozenset(pair) for pair in constraints['distance_pairs']},
                             {frozenset(pair) for pair in HYDROLYSIS_DISTANCE_PAIRS})
            self.assertNotIn('angle_atoms', constraints)

    def test_six_pairs_are_derived_from_the_positional_metadata_alone(self):
        """A seed carrying only the positional indices still yields the full six-distance core."""
        seed = make_hydrolysis_seed(reactive_atoms=False)
        self.assertNotIn('reactive_atoms', seed['metadata'])
        constraints = get_wrapper_constraints(wrapper='crest',
                                              reaction=types.SimpleNamespace(family='carbonyl_based_hydrolysis'),
                                              seed=seed,
                                              )
        self.assertEqual(constraints['atoms'], (2, 1, 8, 9))
        self.assertEqual({frozenset(pair) for pair in constraints['distance_pairs']},
                         {frozenset(pair) for pair in HYDROLYSIS_DISTANCE_PAIRS})

    def test_seed_metadata_gains_the_named_reactive_atom_dict(self):
        """get_ts_seeds() converts the positional hydrolysis indices into a named role dict."""
        rxn = types.SimpleNamespace(family='ether_hydrolysis')
        with patch('arc.job.adapters.ts.heuristics.hydrolysis',
                   return_value=([HYDROLYSIS_SEED_XYZ],
                                 ['ether_hydrolysis'],
                                 [list(HYDROLYSIS_POSITIONAL_INDICES)])):
            seeds = get_ts_seeds(reaction=rxn, base_adapter='heuristics')
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]['metadata']['reactive_atoms'], HYDROLYSIS_REACTIVE_ATOMS)
        self.assertEqual(seeds[0]['metadata']['indices'], HYDROLYSIS_POSITIONAL_INDICES)

    def test_seed_metadata_accepts_an_already_named_index_mapping(self):
        """A hydrolysis generator emitting a role-keyed mapping is honoured as-is."""
        named = {'a': 2, 'b': 1, 'e': 3, 'o': 8, 'h1': 9, 'd': 7}
        rxn = types.SimpleNamespace(family='nitrile_hydrolysis')
        with patch('arc.job.adapters.ts.heuristics.hydrolysis',
                   return_value=([HYDROLYSIS_SEED_XYZ], ['nitrile_hydrolysis'], [named])):
            seeds = get_ts_seeds(reaction=rxn, base_adapter='heuristics')
        self.assertEqual(seeds[0]['metadata']['reactive_atoms'], HYDROLYSIS_REACTIVE_ATOMS)
        constraints = get_wrapper_constraints(wrapper='crest', reaction=rxn, seed=seeds[0])
        self.assertEqual(constraints['atoms'], (2, 1, 8, 9))

    def test_constraints_are_none_for_missing_or_malformed_metadata(self):
        """A hydrolysis seed whose roles cannot be resolved yields no constraints, with a warning."""
        rxn = types.SimpleNamespace(family='carbonyl_based_hydrolysis')
        broken_metadata = [None,
                           {},
                           {'indices': None},
                           {'indices': [2, 1, 8]},
                           {'indices': {'a': 2, 'b': 1, 'o': 8}},
                           {'indices': [2, 1, 3, 8, 99, 7]},
                           {'indices': [2, 1, 3, 8, 8, 7]},
                           {'indices': [2, 1, 3, 9, 8, 7]},
                           {'reactive_atoms': {'a': 2, 'b': 1, 'o': 9, 'h1': 8}},
                           {'reactive_atoms': {'A': 2, 'H': 9, 'B': 1}},
                           ]
        for metadata in broken_metadata:
            seed = {'xyz': HYDROLYSIS_SEED_XYZ,
                    'family': 'carbonyl_based_hydrolysis',
                    'metadata': metadata,
                    }
            with self.assertLogs('arc', level='WARNING'):
                constraints = get_wrapper_constraints(wrapper='crest', reaction=rxn, seed=seed)
            self.assertIsNone(constraints, msg=f'Metadata {metadata!r} should not yield constraints.')

    def test_constraints_inp_holds_one_based_indices(self):
        """A hydrolysis seed round trips into a $constrain block of six one-based distances."""
        self.patch_crest_module()
        constraints = get_wrapper_constraints(wrapper='crest',
                                              reaction=types.SimpleNamespace(family='carbonyl_based_hydrolysis'),
                                              seed=make_hydrolysis_seed(),
                                              )
        crest_path = crest_mod.crest_ts_conformer_search(xyz_guess=HYDROLYSIS_SEED_XYZ,
                                                         constraints=constraints,
                                                         path=self.tmpdir.name,
                                                         xyz_crest_int='hydro',
                                                         )
        with open(os.path.join(crest_path, 'constraints.inp')) as f:
            constraints_text = f.read()

        self.assertIn('atoms: 3, 2, 9, 10\n', constraints_text)
        for atom_1, atom_2 in HYDROLYSIS_DISTANCE_PAIRS:
            self.assertIn(f'distance: {atom_1 + 1}, {atom_2 + 1}, auto', constraints_text,
                          msg=f'The {atom_1}--{atom_2} distance is missing from:\n{constraints_text}')
        self.assertEqual(constraints_text.count('distance:'), 6)
        self.assertNotIn('angle:', constraints_text)
        self.assertIn('$metadyn\n  atoms: 1, 4, 5, 6, 7, 8, 11\n', constraints_text)
        self.assertNotIn('distance: 2, 1,', constraints_text)
        self.assertNotIn('distance: 8, 9,', constraints_text)

    def test_a_hydrolysis_geometry_survives_the_reactive_constraint_check(self):
        """The six-distance core accepts the seed geometry and rejects a dissociated one."""
        constraints = get_wrapper_constraints(wrapper='crest',
                                              reaction=types.SimpleNamespace(family='nitrile_hydrolysis'),
                                              seed=make_hydrolysis_seed(family='nitrile_hydrolysis'),
                                              )
        self.assertTrue(crest_mod._preserves_reactive_constraints(xyz=HYDROLYSIS_SEED_XYZ,
                                                                  reference_xyz=HYDROLYSIS_SEED_XYZ,
                                                                  constraints=constraints,
                                                                  ))
        dissociated_coords = list(HYDROLYSIS_SEED_XYZ['coords'])
        dissociated_coords[8] = (9.0, -0.06034836, 1.24530570)
        dissociated_xyz = dict(HYDROLYSIS_SEED_XYZ, coords=tuple(dissociated_coords))
        self.assertFalse(crest_mod._preserves_reactive_constraints(xyz=dissociated_xyz,
                                                                   reference_xyz=HYDROLYSIS_SEED_XYZ,
                                                                   constraints=constraints,
                                                                   ))


class TestCrestHydrolysisExecution(CrestTestCase):
    """Tests for driving execute_incore() with hydrolysis seeds."""

    def setUp(self):
        """Build a methyl formate hydrolysis reaction and a CREST adapter."""
        super().setUp()
        self.rxn = make_methyl_formate_hydrolysis_reaction()
        self.patch_crest_module()
        self.adapter = CrestAdapter(job_type='tsg',
                                    reactions=[self.rxn],
                                    testing=True,
                                    project='test_CrestAdapter',
                                    project_directory=self.tmpdir.name,
                                    )
        os.makedirs(self.adapter.local_path, exist_ok=True)

    def run_with_seeds(self, seeds):
        """
        Run execute_incore() over ``seeds`` and return the recorded constraint specifications.

        Args:
            seeds (list): The seed entries get_ts_seeds() should return.

        Returns:
            list: One entry per prepared CREST job, holding its constraint specification.
        """
        recorded = list()

        def fake_search(xyz_guess, **kwargs):
            recorded.append(kwargs.get('constraints'))
            crest_path = os.path.join(self.tmpdir.name, f"crest_{kwargs.get('xyz_crest_int')}")
            os.makedirs(crest_path, exist_ok=True)
            return crest_path

        with patch.object(crest_mod, 'crest_available', lambda: True), \
                patch.object(crest_mod, 'get_ts_seeds', lambda **kwargs: list(seeds)), \
                patch.object(crest_mod, 'get_backup_ts_seeds', lambda *args, **kwargs: []), \
                patch.object(crest_mod, 'crest_ts_conformer_search', fake_search), \
                patch.object(crest_mod, 'submit_crest_jobs', lambda paths: {}), \
                patch.object(crest_mod, 'monitor_crest_jobs', lambda *args, **kwargs: None), \
                patch.object(crest_mod, 'process_completed_jobs', lambda *args, **kwargs: []):
            self.adapter.execute_incore()
        return recorded

    def test_a_hydrolysis_seed_reaches_crest_with_six_distances(self):
        """CREST accepts the registered hydrolysis family and constrains the four-center core."""
        recorded = self.run_with_seeds([make_hydrolysis_seed()])
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]['atoms'], (2, 1, 8, 9))
        self.assertEqual({frozenset(pair) for pair in recorded[0]['distance_pairs']},
                         {frozenset(pair) for pair in HYDROLYSIS_DISTANCE_PAIRS})

    def test_a_malformed_hydrolysis_seed_is_skipped_cleanly(self):
        """A seed with unusable metadata is skipped and logged rather than aborting the run."""
        seed = make_hydrolysis_seed(indices=[2, 1, 8], reactive_atoms=False)
        with self.assertLogs('arc', level='WARNING') as log:
            recorded = self.run_with_seeds([seed])
        self.assertEqual(recorded, [])
        self.assertTrue(any('CREST hydrolysis reactive atoms' in message for message in log.output),
                        msg=f'The unusable hydrolysis metadata was not logged:\n{log.output}')
        self.assertTrue(any('Skipping this CREST seed' in message for message in log.output),
                        msg=f'The skipped seed was not logged:\n{log.output}')


class TestCrestHydrolysisRegistration(unittest.TestCase):
    """Tests for the CREST family registration of the hydrolysis families."""

    def test_crest_registered_for_every_hydrolysis_family(self):
        """CREST is an eligible TS adapter for all three hydrolysis families."""
        for family in HYDROLYSIS_FAMILIES:
            self.assertIn('crest', ts_adapters_by_rmg_family[family])

    def test_heuristics_stays_registered_for_every_hydrolysis_family(self):
        """Registering CREST must not displace the heuristics adapter."""
        for family in HYDROLYSIS_FAMILIES:
            self.assertIn('heuristics', ts_adapters_by_rmg_family[family])

    def test_hydrolysis_families_match_family_sets(self):
        """The hub's local family tuple must stay in step with heuristics.FAMILY_SETS.

        seed_hub holds its own copy so that resolving the families does not import heuristics.
        This test is what makes that duplication safe.
        """
        from arc.job.adapters.ts.heuristics import FAMILY_SETS
        from arc.job.adapters.ts.seed_hub import get_hydrolysis_families

        from_family_sets = sorted(family for families in FAMILY_SETS.values() for family in families)
        self.assertEqual(sorted(get_hydrolysis_families()), from_family_sets)
        self.assertEqual(sorted(HYDROLYSIS_FAMILIES), from_family_sets)

    def test_crest_is_not_a_default_ts_adapter(self):
        """CREST stays opt-in: it is absent from the default ts_adapters list.

        An adapter runs only if it is both registered for the family and listed in ts_adapters,
        so registering the hydrolysis families does not change ARC's default behaviour.
        """
        self.assertNotIn('crest', [adapter.lower() for adapter in ts_adapters])


if __name__ == '__main__':
    unittest.main()
