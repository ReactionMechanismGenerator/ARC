import unittest

from arc.common import ARC_PATH
from arc.species import ic_params

class TestPerceive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_AnchorSpec_ensure_primary_when_not_list(self):
        """test the method AnchorSpec.ensure_primary"""
        a = ic_params.AnchorSpec(m2i="not-a-list", m1i=[7, 8])
        a.ensure_primary(required_first_m2i=0)
        self.assertEqual(a.m2i, [0])

    def test_AnchorSpec_ensure_primary_move_to_front(self):
        """test the method AnchorSpec.ensure_primary"""
        a = ic_params.AnchorSpec(m2i=[2, 0, 3], m1i=[7, 8])
        a.ensure_primary(0)
        self.assertEqual(a.m2i, [0, 2, 3])

    def test_AnchorSpec_ensure_primary_noop_if_already_first(self):
        """test the method AnchorSpec.ensure_primary"""
        a = ic_params.AnchorSpec(m2i=[1, 4, 5], m1i=[7, 8])
        a.ensure_primary(1)
        self.assertEqual(a.m2i, [1, 4, 5])

    def test_AnchorSpec_ensure_primary_casts_to_int(self):
        """test the method AnchorSpec.ensure_primary"""
        a = ic_params.AnchorSpec(m2i=["1", "3"], m1i=[7, 8])
        a.ensure_primary(1)
        self.assertEqual(a.m2i, [1, 3])

    def test_PlacementParam_ensure_primary_m2i_sets_index_if_none(self):
        """test the method ensure_primary_m2i"""
        p = ic_params.PlacementParam(
            key=ic_params.ParamKey.R,
            val=1.23,
            anchors=ic_params.AnchorSpec(m2i=[3, 0, 2], m1i=[9]),
            index=None,
        )
        p.ensure_primary_m2i(0)
        self.assertEqual(p.anchors.m2i[0], 0)
        self.assertEqual(p.index, 0)

    def test_PlacementParam_set_base_n_sets_final_index(self):
        """test the method set_base_n"""
        p = ic_params.PlacementParam(
            key=ic_params.ParamKey.A,
            val=109.5,
            anchors=ic_params.AnchorSpec(m2i=[1, 0], m1i=[5]),
            index=1,
        )
        p.set_base_n(10)
        self.assertEqual(p.final_index, 11)

    def test_PlacementParam_set_base_n_noop_if_index_none(self):
        """test the method set_base_n"""
        p = ic_params.PlacementParam(
            key=ic_params.ParamKey.D,
            val=180.0,
            anchors=ic_params.AnchorSpec(m2i=[2, 1, 0], m1i=[4, 5]),
            index=None,
        )
        p.set_base_n(7)
        self.assertIsNone(p.final_index)

    def test_PlacementParam_build_key_shifts_only_m2(self):
        """test the method build_key"""
        p = ic_params.PlacementParam(
            key=ic_params.ParamKey.A,
            val=109.5,
            anchors=ic_params.AnchorSpec(m2i=[0, 1, 2], m1i=[5, 6]),
            index=0,
        )
        key = p.build_key(base_n=10)
        self.assertEqual(key, "A_10_11_12_5_6")

    def test_Atom1Params_post_init_enforces_m2i0_and_keys(self):
        """test the class Atom1Params """
        R = ic_params.PlacementParam(ic_params.ParamKey.R, 1.0, ic_params.AnchorSpec([3, 0], [100]), index=None)
        A = ic_params.PlacementParam(ic_params.ParamKey.A, 90.0, ic_params.AnchorSpec([2, 0], [101, 102]), index=None)
        D = ic_params.PlacementParam(ic_params.ParamKey.D, 180.0, ic_params.AnchorSpec([5, 0, 4], [103, 104, 105]), index=None)
        atom1 = ic_params.Atom1Params(R=R, A=A, D=D)
        self.assertEqual(atom1.R.anchors.m2i[0], 0)
        self.assertEqual(atom1.A.anchors.m2i[0], 0)
        self.assertEqual(atom1.D.anchors.m2i[0], 0)
        self.assertEqual(atom1.R.index, 0)
        self.assertEqual(atom1.A.index, 0)
        self.assertEqual(atom1.D.index, 0)

    def test_Atom1Params_set_base_n_propagates_final_index(self):
        """test the method set_base_n for Atom1Params"""
        atom1 = ic_params.Atom1Params(
            R=ic_params.PlacementParam(ic_params.ParamKey.R, 1.1, ic_params.AnchorSpec([0], [1]), index=None),
            A=ic_params.PlacementParam(ic_params.ParamKey.A, 109.5, ic_params.AnchorSpec([0, 1], [2]), index=None),
            D=ic_params.PlacementParam(ic_params.ParamKey.D, 180.0, ic_params.AnchorSpec([0, 1, 2], [3]), index=None),
        )
        atom1.set_base_n(7)
        self.assertEqual(atom1.R.final_index, 7)
        self.assertEqual(atom1.A.final_index, 7)
        self.assertEqual(atom1.D.final_index, 7)

    def test_Atom1Params_invalid_keys_raise(self):
        """test the method Atom1Params raises ValueError on invalid keys"""
        with self.assertRaises(ValueError):
            ic_params.Atom1Params(
                R=ic_params.PlacementParam(ic_params.ParamKey.A, 1.0, ic_params.AnchorSpec([0], [1])),
                A=ic_params.PlacementParam(ic_params.ParamKey.R, 90.0, ic_params.AnchorSpec([0, 1], [2])),
                D=ic_params.PlacementParam(ic_params.ParamKey.D, 180.0, ic_params.AnchorSpec([0, 1, 2], [3])),
            )

    def test_Atom2Params_post_init_enforces_m2i1_and_keys(self):
        """test that the method enforces m2i[0] == 1 and correct keys"""
        A = ic_params.PlacementParam(ic_params.ParamKey.A, 120.0, ic_params.AnchorSpec([4, 1, 0], [9, 10]), index=None)
        D = ic_params.PlacementParam(ic_params.ParamKey.D, 60.0, ic_params.AnchorSpec([2, 1, 0], [11, 12, 13]), index=None)
        atom2 = ic_params.Atom2Params(A=A, D=D)
        self.assertEqual(atom2.A.anchors.m2i[0], 1)
        self.assertEqual(atom2.D.anchors.m2i[0], 1)
        self.assertEqual(atom2.A.index, 1)
        self.assertEqual(atom2.D.index, 1)

    def test_Atom2Params_set_base_n_propagates_final_index(self):
        """test the method Atom2Params.set_base_n"""
        atom2 = ic_params.Atom2Params(
            A=ic_params.PlacementParam(ic_params.ParamKey.A, 100.0, ic_params.AnchorSpec([1, 0], [2]), index=None),
            D=ic_params.PlacementParam(ic_params.ParamKey.D, 180.0, ic_params.AnchorSpec([1, 0, 2], [3]), index=None),
        )
        atom2.set_base_n(20)
        self.assertEqual(atom2.A.final_index, 21)
        self.assertEqual(atom2.D.final_index, 21)

    def test_Atom2Params_invalid_keys_raise(self):
        """test that the method Atom2Params raises ValueError on invalid keys"""
        with self.assertRaises(ValueError):
            ic_params.Atom2Params(
                A=ic_params.PlacementParam(ic_params.ParamKey.R, 1.0, ic_params.AnchorSpec([1, 0], [2])),
                D=ic_params.PlacementParam(ic_params.ParamKey.D, 180.0, ic_params.AnchorSpec([1, 0, 2], [3])),
            )

    def test_Atom3Params_post_init_enforces_m2i2_and_key(self):
        """test that the class Atom3Params enforces m2i[0] == 2 and correct key"""
        D = ic_params.PlacementParam(ic_params.ParamKey.D, 179.9, ic_params.AnchorSpec([5, 2, 1, 0], [7, 8, 9]), index=None)
        atom3 = ic_params.Atom3Params(D=D)
        self.assertEqual(atom3.D.anchors.m2i[0], 2)
        self.assertEqual(atom3.D.index, 2)

    def test_Atom3Params_set_base_n_propagates_final_index(self):
        """test the method Atom3Params.set_base_n"""
        atom3 = ic_params.Atom3Params(
            D=ic_params.PlacementParam(ic_params.ParamKey.D, 120.0, ic_params.AnchorSpec([2, 1, 0], [4, 5, 6]), index=None)
        )
        atom3.set_base_n(30)
        self.assertEqual(atom3.D.final_index, 32)

    def test_Atom3Params_invalid_key_raises(self):
        """test the class Atom3Params"""
        with self.assertRaises(ValueError):
            ic_params.Atom3Params(D=ic_params.PlacementParam(ic_params.ParamKey.A, 90.0, ic_params.AnchorSpec([2, 1, 0], [4, 5])))

    def test_build_key_with_string_indices_and_shift(self):
        """test the method build_key"""
        p = ic_params.PlacementParam(
            key=ic_params.ParamKey.R,
            val=1.0,
            anchors=ic_params.AnchorSpec(m2i=["2", "0"], m1i=["10", "11"]),
            index=2,
        )
        self.assertEqual(p.build_key(5), "R_7_5_10_11")
    
if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))