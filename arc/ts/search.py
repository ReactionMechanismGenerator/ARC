"""
A module for performing reaction transition state searches.
"""

from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from arc.common import colliding_atoms
from arc.reaction import ARCReaction
from arc.rmgdb import determine_reaction_family
from arc.ts.factory import _registered_ts_adapters, ts_method_factory

if TYPE_CHECKING:
    from rmgpy.data.rmg import RMGDatabase


class TSMethodsEnum(str, Enum):
    """
    The supported methods for a TS search. The methods which are available are a finite set.
    """
    autotst = 'autotst'  # AutoTST
    gsm = 'gsm'  # double ended growing string method (DE-GSM)
    pygsm = 'pygsm'  # double ended growing string method (DE-GSM)
    heuristics = 'heuristics'  # brute force heuristics
    kinbot = 'kinbot'  # KinBot
    gnn_isomerization = 'gnn_isomerization'  # Graph neural network ML for isomerization reactions, https://doi.org/10.1021/acs.jpclett.0c00500
    neb_ase = 'neb_ase'  # NEB in ASE
    neb_terachem = 'neb_terachem'  # NEB in TeraChem
    qst2 = 'qst2'  # Synchronous Transit-Guided Quasi-Newton (STQN) implemented in Gaussian
    user = 'user'  # user guesses


class TSSearch(object):
    """
    A class for performing reaction transition state searches.

    Args:
        arc_reaction (ARCReaction): The ARC Reaction object.
        methods (List[str]): The TS search methods to carry out.
                             Allowed values are those specified under the TSMethodsEnum class.
        user_guesses (list, optional): Entries are string representations of Cartesian coordinate.
        rmg_db (RMGDatabase, optional): The RMG database object, mandatory for the following methods:
                                        'autotst', 'heuristics', 'kinbot'.
        dihedral_increment (float, optional): The scan dihedral increment to use when generating guesses.
    """

    def __init__(self,
                 arc_reaction: ARCReaction,
                 methods: List[str],
                 user_guesses: Optional[List[dict]] = None,
                 rmg_db: Optional['RMGDatabase'] = None,
                 dihedral_increment: float = 20,
                 ):
        if not isinstance(arc_reaction, ARCReaction):
            raise TypeError(f'arc_reaction must be an ARCReaction object, '
                            f'got {arc_reaction} which is a {type(arc_reaction)}.')
        self.arc_reaction = arc_reaction
        if self.arc_reaction.rmg_reaction is None:
            self.arc_reaction.rmg_reaction_from_arc_species()
        if arc_reaction.family is None:
            self.arc_reaction.family = determine_reaction_family(self.rmg_db, self.arc_reaction.rmg_reaction)[0]

        self.methods = [TSMethodsEnum(method.lower()) for method in methods]  # pass methods through the enumeration
        for method in self.methods:
            if method.value not in list(_registered_ts_adapters.keys()):
                raise ValueError(f'Could not interpret unregistered method {method}.')

        self.user_guesses = user_guesses
        self.rmg_db = rmg_db
        self.dihedral_increment = dihedral_increment
        self.ts_guesses = list()

    def __repr__(self) -> str:
        """
        A short representation of the current TSSearch.

        Returns: str
            The desired representation.
        """
        return f"TSSearch(arc_reaction='{self.arc_reaction}', " \
               f"methods='{self.methods}', " \
               f"user_guesses={self.user_guesses}, " \
               f"rmgdb = {self.rmg_db})"

    def execute(self):
        """
        Execute a TS search.

        Returns: List[dict]
            Entries are Cartesian coordinate dictionaries of TS guesses generated using the requested method(s).
        """
        self.search()
        return self.ts_guesses

    def search(self) -> None:
        """
        Execute the selected TS search methods.
        Populates the self.ts_guesses list.
        """
        for method in self.methods:
            ts_adapter = ts_method_factory(ts_adapter=method.value,
                                           user_guesses=self.user_guesses,
                                           arc_reaction=self.arc_reaction,
                                           dihedral_increment=self.dihedral_increment,
                                           )
            ts_guesses = ts_adapter.generate_guesses()
            for ts_guess in ts_guesses:
                if not colliding_atoms(xyz=ts_guess):
                    self.ts_guesses.append(ts_guess)
