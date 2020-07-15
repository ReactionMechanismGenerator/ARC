"""
A TS adapter for QST2 method in Gaussian.
"""

from typing import TYPE_CHECKING, List, Optional

from arc.species.converter import check_xyz_dict
from arc.ts.adapter import TSAdapter
from arc.ts.factory import register_ts_adapter

if TYPE_CHECKING:
    from arc.reaction import ARCReaction


class QST2Adapter(TSAdapter):
    """
    A class for representing user guesses for a transition state.
    """

    def __init__(self,
                 user_guesses: Optional[List[dict]] = None,
                 arc_reaction: Optional['ARCReaction'] = None,
                 dihedral_increment: Optional[float] = None,
                 ) -> None:
        """
        Initializes a UserAdapter instance.

        Args:
            user_guesses (List[dict], optional): Entries are dictionary representations of Cartesian coordinate.
            arc_reaction (ARCReaction, optional): The ARC reaction object with the family attribute populated,
                                                  not used by the UserAdapter.
            dihedral_increment (Optional[float], optional): The scan dihedral increment to use when generating guesses,
                                                            not used by the UserAdapter.
        """
        self.user_guesses = user_guesses

    def __repr__(self) -> str:
        """
        A short representation of the current UserAdapter.

        Returns: str
            The desired representation.
        """
        return f"UserAdapter(user_guesses={self.user_guesses})"

    def generate_guesses(self) -> List[dict]:
        """
        Generate TS guesses using the user guesses.

        Returns: List[dict]
            Entries are Cartesian coordinate dictionaries of TS guesses.
        """
        if self.user_guesses is None or not self.user_guesses:
            return list()

        if not isinstance(self.user_guesses, list):
            self.user_guesses = [self.user_guesses]

        results = list()
        for user_guess in self.user_guesses:
            results.append(check_xyz_dict(user_guess))
        return results


register_ts_adapter('user', UserAdapter)
