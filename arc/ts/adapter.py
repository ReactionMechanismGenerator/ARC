"""
A module for the abstract TSAdapter class
"""

from abc import ABC, abstractmethod


class TSAdapter(ABC):
    """
    An abstract TS Adapter class
    """

    @abstractmethod
    def generate_guesses(self) -> list:
        """
        Generate TS guesses.

        Returns:
            list: Entries are Cartresian coordinates of TS guesses.
        """
        pass
