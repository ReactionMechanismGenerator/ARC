"""
A TS adapter for user guesses.
"""

from typing import TYPE_CHECKING

from arc.species.converter import check_xyz_dict
from .factory import register_ts_adapter
from .adapter import TSAdapter

if TYPE_CHECKING:
    from arc.reaction import ARCReaction


class UserAdapter(TSAdapter):
    """
    A class for representing user guesses for a transition state.
    """

    def __init__(self, user_guesses: list = None,
                 arc_reaction: 'ARCReaction' = None,
                 dihedral_increment: float = 20,
                 ) -> None:
        """
        Initializes a UserAdapter instance.

        Args:
            user_guesses (list): TS user guesses.
            arc_reaction: (ARCReaction, optional): The ARC Reaction object, not used in the UserAdapter class.
            dihedral_increment: (float, optional): The scan dihedral increment to use when generating guesses,
                                                   not used in the UserAdapter class.
        """
        if user_guesses is not None and not isinstance(user_guesses, list):
            raise TypeError(f'user_guessed must be a list, got\n'
                            f'{user_guesses}\n'
                            f'which is a {type(user_guesses)}.')
        self.user_guesses = user_guesses

    def __repr__(self) -> str:
        """
        A short representation of the current UserAdapter.

        Returns:
            str: The desired representation.
        """
        return f"UserAdapter(user_guesses={self.user_guesses})"

    def generate_guesses(self) -> list:
        """
        Generate TS guesses using the user guesses.

        Returns:
            list: Entries are Cartresian coordinates of TS guesses.
        """
        if self.user_guesses is None:
            return list()

        if not isinstance(self.user_guesses, list):
            self.user_guesses = [self.user_guesses]

        results = list()
        for user_guess in self.user_guesses:
            results.append(check_xyz_dict(user_guess))
        return results


register_ts_adapter('user', UserAdapter)
