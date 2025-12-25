from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ParamKey(str, Enum):
    """Internal-coordinate parameter type."""
    R = "R"   # bond length
    A = "A"   # bond angle
    D = "D"   # dihedral


@dataclass
class AnchorSpec:
    """
    Indices that define this parameter relative to the two molecules:
      - m2i: indices in xyz2 (local to the second structure being added)
      - m1i: indices in xyz1 (anchors in the first structure)
    """
    m2i: List[int]
    m1i: List[int]

    def ensure_primary(self, required_first_m2i: int) -> None:
        """
        Enforce that the first entry in m2i is `required_first_m2i`.
        If absent, insert it; if present but not first, move it to front.
        """
        if not isinstance(self.m2i, list):
            self.m2i = [required_first_m2i]
            return
        self.m2i = [int(x) for x in self.m2i]
        if self.m2i and self.m2i[0] == required_first_m2i:
            return
        if required_first_m2i in self.m2i:
            self.m2i.remove(required_first_m2i)
        self.m2i.insert(0, required_first_m2i)


@dataclass
class PlacementParam:
    """
    One IC parameter (R/A/D) for a single atom placement.

    Attributes
    ----------
    key : ParamKey
        Type of parameter (R/A/D).
    val : float
        Numeric value of the parameter (Å, degrees).
    anchors : AnchorSpec
        Anchor indices in xyz2 (m2i) and xyz1 (m1i).
    index : Optional[int]
        Local atom index in xyz2 this parameter primarily refers to. If not
        given, it is inferred as anchors.m2i[0] after normalization.
    final_index : Optional[int]
        The shifted index in the combined structure (n + index), set by
        calling `set_base_n(n)` where n = len(xyz1).
    """
    key: ParamKey
    val: float
    anchors: AnchorSpec
    index: Optional[int] = None
    final_index: Optional[int] = None

    def ensure_primary_m2i(self, required_first_m2i: int) -> None:
        self.anchors.ensure_primary(required_first_m2i)
        if self.index is None:
            self.index = self.anchors.m2i[0]

    def set_base_n(self, base_n: int) -> None:
        if self.index is not None:
            self.final_index = base_n + int(self.index)

    def build_key(self, base_n: int) -> str:
        """
        Build zmat variable name like 'A_<m2 shifted>_<m1 raw>...'
        m2 indices are shifted by base_n; m1 indices are unchanged.
        """
        m2_part = [str(int(i) + base_n) for i in self.anchors.m2i]
        m1_part = [str(int(i)) for i in self.anchors.m1i]
        return "_".join([self.key.value] + m2_part + m1_part)


@dataclass
class Atom1Params:
    """Placement parameters for the FIRST atom of xyz2 (must target m2i[0] == 0)."""
    R: PlacementParam
    A: PlacementParam
    D: PlacementParam

    def __post_init__(self) -> None:
        self.R.ensure_primary_m2i(0)
        self.A.ensure_primary_m2i(0)
        self.D.ensure_primary_m2i(0)
        # Key sanity:
        if self.R.key is not ParamKey.R or self.A.key is not ParamKey.A or self.D.key is not ParamKey.D:
            raise ValueError("Atom1Params expects keys R, A, D respectively.")

    def set_base_n(self, base_n: int) -> None:
        self.R.set_base_n(base_n)
        self.A.set_base_n(base_n)
        self.D.set_base_n(base_n)


@dataclass
class Atom2Params:
    """Placement parameters for the SECOND atom of xyz2 (must target m2i[0] == 1)."""
    A: PlacementParam
    D: PlacementParam

    def __post_init__(self) -> None:
        self.A.ensure_primary_m2i(1)
        self.D.ensure_primary_m2i(1)
        if self.A.key is not ParamKey.A or self.D.key is not ParamKey.D:
            raise ValueError("Atom2Params expects keys A, D respectively.")

    def set_base_n(self, base_n: int) -> None:
        self.A.set_base_n(base_n)
        self.D.set_base_n(base_n)


@dataclass
class Atom3Params:
    """Placement parameters for the THIRD atom of xyz2 (must target m2i[0] == 2)."""
    D: PlacementParam

    def __post_init__(self) -> None:
        self.D.ensure_primary_m2i(2)
        if self.D.key is not ParamKey.D:
            raise ValueError("Atom3Params expects key D.")

    def set_base_n(self, base_n: int) -> None:
        self.D.set_base_n(base_n)
