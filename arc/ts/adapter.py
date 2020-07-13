"""
A module for the abstract TSAdapter class
"""

from abc import ABC, abstractmethod
from typing import List


class TSAdapter(ABC):
    """
    An abstract TS Adapter class
    """

    @abstractmethod
    def generate_guesses(self) -> List[dict]:
        """
        Generate TS guesses.

        Returns: List[dict]
            Entries are Cartesian coordinate dictionaries of TS guesses.
        """
        pass
