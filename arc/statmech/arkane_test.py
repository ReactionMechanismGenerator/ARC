#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for ARC's statmech.arkane modele
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH
from arc.level import Level
from arc.reaction import ARCReaction
from arc.species import ARCSpecies
from arc.statmech.arkane import ArkaneAdapter


class TestArkaneAdapter(unittest.TestCase):
    """
    Contains unit tests for ArkaneAdapter.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        path_1 = os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_tests_delete', 'arkane_1')
        path_2 = os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_tests_delete', 'arkane_2')
        path_3 = os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_tests_delete', 'arkane_3')
        for path in [path_1, path_2, path_3]:
            if not os.path.isdir(path):
                os.makedirs(path)
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='CH3NH', smiles='C[NH]')],
                            p_species=[ARCSpecies(label='CH2NH2', smiles='[CH2]N')])
        rxn_1.ts_species = ARCSpecies(label='TS1', is_ts=True, xyz="""C      -0.68121000   -0.03232800    0.00786900
                                                                      H      -1.26057500    0.83953400   -0.27338400
                                                                      H      -1.14918300   -1.01052100   -0.06991000
                                                                      H       0.10325500    0.27373300    0.96444200
                                                                      N       0.74713700    0.12694800   -0.09842700
                                                                      H       1.16150600   -0.79827600    0.01973500""")
        cls.arkane_1 = ArkaneAdapter(output_directory=path_1, output_dict=dict(), bac_type=None, reaction=rxn_1)
        cls.arkane_2 = ArkaneAdapter(output_directory=path_2, output_dict=dict(), bac_type=None,
                                     species=rxn_1.r_species[0])
        cls.ic3h7 = ARCSpecies(label='iC3H7', smiles='C[CH]C')
        cls.ic3h7.e_elect = 150.1
        opt_path = os.path.join(ARC_PATH, 'arc', 'testing', 'statmech', 'iC3H7_xtb', 'opt_a1596', 'output.out')
        freq_path = os.path.join(ARC_PATH, 'arc', 'testing', 'statmech', 'iC3H7_xtb', 'freq_a1597', 'output.out')
        cls.arkane_3 = ArkaneAdapter(output_directory=path_3,
                                     output_dict={'iC3H7': {'paths': {'freq': freq_path,
                                                                      'sp': opt_path,
                                                                      'opt': opt_path,
                                                                      'composite': '',
                                                                      }}},
                                     bac_type=None,
                                     species=cls.ic3h7,
                                     sp_level=Level('gfn2'),
                                     )

    def test__str__(self):
        """Test the __str__ function"""
        for arkane in [self.arkane_1, self.arkane_2, self.arkane_3]:
            repr = arkane.__str__()
            self.assertIn('ArkaneAdapter(', repr)
            self.assertIn(f'output_directory={arkane.output_directory}, ', repr)
            self.assertIn(f'bac_type={arkane.bac_type}, ', repr)
            self.assertIn(f'freq_scale_factor={arkane.freq_scale_factor}, ', repr)
            self.assertIn(f'species={arkane.species}, ', repr)
            self.assertIn(f'reaction={arkane.reaction}, ', repr)
            self.assertIn(f'T_min={arkane.T_min}, ', repr)
            self.assertIn(f'T_max={arkane.T_max}, ', repr)
            self.assertIn(f'T_count={arkane.T_count})', repr)
            if arkane.sp_level is not None:
                self.assertIn(f'sp_level={arkane.sp_level.simple()}', repr)
    
    def test_arkane_has_en_corr(self):
        """Test the arkane_has_en_corr() function"""
        self.arkane_1.sp_level = Level('CBS-QB3')
        self.assertTrue(self.arkane_1.arkane_has_en_corr())

        # B3LYP/6-31G(d,p) does not support nitrogen AECs
        self.arkane_1.sp_level = Level(method='B3LYP', basis='6-31G(d,p)')
        self.assertFalse(self.arkane_1.arkane_has_en_corr())

        self.arkane_1.sp_level = Level(method='fake', basis='basis')
        self.assertFalse(self.arkane_1.arkane_has_en_corr())

        self.arkane_2.sp_level = Level('CBS-QB3')
        self.assertTrue(self.arkane_2.arkane_has_en_corr())

        self.arkane_2.sp_level = Level(method='fake', basis='basis')
        self.assertFalse(self.arkane_2.arkane_has_en_corr())

    def test_run_statmech_using_molecular_properties(self):
        """Test running statmech using molecular properties."""
        self.arkane_3.compute_thermo()
        self.assertTrue(os.path.isfile(os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_tests_delete', 'arkane_3',
                                                    'Species', 'iC3H7', 'thermo_properties_plot.pdf')))
        self.assertAlmostEqual(self.ic3h7.thermo.E0.value_si, 25996536.997005656)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_tests_delete'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
