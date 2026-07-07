#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.qst2 module.
"""

import datetime
import os
import shutil
import unittest

from arc.common import ARC_TESTING_PATH
from arc.job.adapters.ts.qst2 import QST2Adapter
from arc.level import Level
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies


class TestQST2Adapter(unittest.TestCase):
    """
    Contains unit tests for the QST2Adapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        for i in range(10):
            cls.addClassCleanup(shutil.rmtree, os.path.join(ARC_TESTING_PATH, f'test_QST2Adapter_{i}'),
                                ignore_errors=True)

        # A 1,2-halogen migration: unimolecular <-> unimolecular (a QST2-friendly reaction).
        # R = O[CH]CCl, P = [CH2]C(O)Cl. The atoms keep the same identity and order across
        # the reaction, so the atom map is the identity permutation.
        cls.r_xyz = {'symbols': ('O', 'C', 'C', 'Cl', 'H', 'H', 'H', 'H'),
                     'isotopes': (16, 12, 12, 35, 1, 1, 1, 1),
                     'coords': ((-1.95574208, -0.61841681, 0.08130563),
                                (-1.27101508, 0.47222919, -0.31974537),
                                (-0.07165208, 0.90591219, 0.32866963),
                                (1.43978292, -0.26297181, -0.06795737),
                                (-1.52857108, -1.02344581, 0.84821663),
                                (-1.57959308, 0.83127419, -1.29027737),
                                (0.26922792, 1.87651319, -0.01033137),
                                (-0.09175408, 0.84551219, 1.41369363))}
        cls.p_xyz = {'symbols': ('O', 'C', 'C', 'Cl', 'H', 'H', 'H', 'H'),
                     'isotopes': (16, 12, 12, 35, 1, 1, 1, 1),
                     'coords': ((-1.19607055, -1.10305652, -0.17298926),
                                (-0.62305755, -0.00083652, 0.40430474),
                                (-1.19066255, 1.23097748, -0.09069726),
                                (1.28331745, 0.04100148, -0.05770126),
                                (-0.77288255, -1.90184752, 0.16462374),
                                (-0.52506955, -0.06107952, 1.48622674),
                                (-1.65785655, 1.24294248, -1.06456526),
                                (-0.99373755, 2.15657148, 0.42718974))}
        cls.level = Level(method='b3lyp', basis='def2tzvp')

    def _make_reaction(self) -> ARCReaction:
        """Build a mapped unimolecular <-> unimolecular reaction for QST2."""
        r_species = ARCSpecies(label='R', smiles='O[CH]CCl', xyz=self.r_xyz, multiplicity=2)
        p_species = ARCSpecies(label='P', smiles='[CH2]C(O)Cl', xyz=self.p_xyz, multiplicity=2)
        reaction = ARCReaction(r_species=[r_species], p_species=[p_species])
        # The atoms keep the same order across the reaction (intramolecular migration).
        reaction.atom_map = list(range(len(self.r_xyz['symbols'])))
        return reaction

    def test_adapter_constructs(self):
        """Test that the QST2 adapter constructs and creates a dummy TS species."""
        job = QST2Adapter(project='test_0',
                          job_type='tsg',
                          project_directory=os.path.join(ARC_TESTING_PATH, 'test_QST2Adapter_0'),
                          reactions=[self._make_reaction()],
                          level=self.level,
                          server='local',
                          )
        self.assertEqual(job.job_adapter, 'qst2')
        self.assertEqual(job.execution_type, 'queue')
        self.assertIsNotNone(job.reactions[0].ts_species)
        self.assertTrue(job.reactions[0].ts_species.is_ts)

    def test_no_reactions_raises(self):
        """Test that instantiating without reactions raises a ValueError."""
        with self.assertRaises(ValueError):
            QST2Adapter(project='test_err',
                        job_type='tsg',
                        project_directory=os.path.join(ARC_TESTING_PATH, 'test_QST2Adapter_err'),
                        reactions=None,
                        level=self.level,
                        server='local',
                        )

    def test_write_input_file(self):
        """Test writing a valid Gaussian QST2 input file with two matching geometry blocks."""
        job = QST2Adapter(project='test_1',
                          job_type='tsg',
                          project_directory=os.path.join(ARC_TESTING_PATH, 'test_QST2Adapter_1'),
                          reactions=[self._make_reaction()],
                          level=self.level,
                          server='local',
                          )
        input_path = os.path.join(job.local_path, 'input.gjf')
        self.assertTrue(os.path.isfile(input_path))
        with open(input_path, 'r') as f:
            content = f.read()

        # 1. The QST2 opt keyword and a frequency job are present.
        self.assertIn('opt=(qst2', content)
        self.assertIn('freq', content)
        self.assertIn('b3lyp/def2tzvp', content)

        # 2. There are exactly two molecule specifications (reactant and product).
        self.assertIn('QST2 R__=__P_TS reactant', content)
        self.assertIn('QST2 R__=__P_TS product', content)
        self.assertEqual(content.count('opt=(qst2'), 1)

        # 3. Split the file into the two geometry blocks and compare atom counts and order.
        reactant_block = content.split('reactant')[1].split('product')[0]
        product_block = content.split('product')[1]
        reactant_symbols = self._parse_block_symbols(reactant_block)
        product_symbols = self._parse_block_symbols(product_block)

        expected_symbols = list(self.r_xyz['symbols'])
        self.assertEqual(len(reactant_symbols), len(expected_symbols))
        self.assertEqual(len(product_symbols), len(expected_symbols))
        self.assertEqual(len(reactant_symbols), len(product_symbols))
        # The two blocks must be in the same atom order (critical for QST2).
        self.assertEqual(reactant_symbols, product_symbols)
        self.assertEqual(reactant_symbols, expected_symbols)

    @staticmethod
    def _parse_block_symbols(block: str) -> list:
        """Extract the element symbols (first token of each coordinate line) from a geometry block."""
        symbols = list()
        for line in block.splitlines():
            tokens = line.split()
            if len(tokens) == 4:
                element, x, y, z = tokens
                try:
                    float(x), float(y), float(z)
                except ValueError:
                    continue
                symbols.append(element)
        return symbols

    def test_process_run_no_output_no_success(self):
        """Test that process_run appends an unsuccessful TSGuess when no output file exists."""
        job = QST2Adapter(project='test_2',
                          job_type='tsg',
                          project_directory=os.path.join(ARC_TESTING_PATH, 'test_QST2Adapter_2'),
                          reactions=[self._make_reaction()],
                          level=self.level,
                          server='local',
                          )
        job.reactions[0].ts_species.ts_guesses = list()
        # Point the output file at a path that does not exist.
        job.local_path_to_output_file = os.path.join(job.local_path, 'does_not_exist.log')
        job.initial_time = datetime.datetime.now()
        job.final_time = datetime.datetime.now()
        job.process_run()
        guesses = job.reactions[0].ts_species.ts_guesses
        self.assertEqual(len(guesses), 1)
        self.assertEqual(guesses[0].method, 'qst2')
        self.assertFalse(guesses[0].success)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        for i in range(10):
            shutil.rmtree(os.path.join(ARC_TESTING_PATH, f'test_QST2Adapter_{i}'), ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_TESTING_PATH, 'test_QST2Adapter_err'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
