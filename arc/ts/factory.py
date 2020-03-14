"""
A module for generating TS search adapters.
"""

from typing import Type, TYPE_CHECKING

from .adapter import TSAdapter

if TYPE_CHECKING:
    from arc.reaction import ARCReaction


_registered_ts_adapters = {}


def register_ts_adapter(ts_method: str,
                        ts_method_class: Type[TSAdapter],
                        ) -> None:
    """
    A register for TS search methods adapters.

    Args:
        ts_method (TSMethodsEnum): A string representation for a TS search adapter.
        ts_method_class (Type[TSAdapter]): The TS search method adapter class.
    """
    if not issubclass(ts_method_class, TSAdapter):
        raise TypeError(f'{ts_method_class} is not a TSAdapter.')
    _registered_ts_adapters[ts_method] = ts_method_class


def ts_method_factory(ts_adapter: str,
                      user_guesses: list = None,
                      arc_reaction: 'ARCReaction' = None,
                      dihedral_increment = None,
                      ) -> Type[TSAdapter]:
    """
    A factory generating the TS search method adapter corresponding to ``ts_adapter``.

    Args:
        ts_adapter (TSMethodsEnum): A string representation for a TS search adapter.
        user_guesses (list, optional): Entries are string representations of Cartesian coordinate.
        arc_reaction (ARCReaction, optional): The ARC reaction object with the family attribute populated.
        dihedral_increment (float, optional): The scan dihedral increment to use when generating guesses.

    Returns:
        Type[TSAdapter]: The requested TSAdapter child, initialized with the respective arguments,
    """
    ts_method = _registered_ts_adapters[ts_adapter](user_guesses=user_guesses,
                                                    arc_reaction=arc_reaction,
                                                    dihedral_increment=dihedral_increment
                                                    )
    return ts_method
