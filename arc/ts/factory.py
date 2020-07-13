"""
A module for generating TS search adapters.
"""

from typing import TYPE_CHECKING, List, Optional, Type

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
                      user_guesses: Optional[List[dict]] = None,
                      arc_reaction: Optional['ARCReaction'] = None,
                      dihedral_increment: Optional[float] = None,
                      ) -> Type[TSAdapter]:
    """
    A factory generating the TS search method adapter corresponding to ``ts_adapter``.

    Args:
        ts_adapter (TSMethodsEnum): A string representation for a TS search adapter.
        user_guesses (List[dict], optional): Entries are dictionary representations of Cartesian coordinate.
        arc_reaction (ARCReaction, optional): The ARC reaction object with the family attribute populated.
        dihedral_increment (Optional[float], optional): The scan dihedral increment to use when generating guesses.

    Returns: Type[TSAdapter]
        The requested TSAdapter child instance, initialized with the respective arguments.
    """
    return _registered_ts_adapters[ts_adapter](user_guesses=user_guesses,
                                               arc_reaction=arc_reaction,
                                               dihedral_increment=dihedral_increment
                                               )
