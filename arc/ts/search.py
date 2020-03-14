"""
A module for performing reaction transition state searches.
"""

from enum import Enum
from typing import TYPE_CHECKING


import qcelemental as qcel

from arc.common import colliding_atoms
from arc.reaction import ARCReaction
from arc.rmgdb import determine_reaction_family
from .factory import _registered_ts_adapters, ts_method_factory

if TYPE_CHECKING:
    from rmgpy.data.rmg import RMGDatabase


class TSMethodsEnum(str, Enum):
    """
    The supported methods for a TS search. The methods which are available are a finite set.
    """

    autotst = 'autotst'  # AutoTST
    gsm = 'gsm'  # double ended growing string method (DE-GSM)
    heuristics = 'heuristics'  # brute force heuristics
    kinbot = 'kinbot'  # KinBot
    ml = 'ml'  # machine learning  Todo: we'll probably have more than one ML module, expand
    neb_ase = 'neb_ase'  # NEB in ASE
    neb_terachem = 'neb_terachem'  # NEB in TeraChem
    qst2 = 'qst2'  # Synchronous Transit-Guided Quasi-Newton (STQN) implemented in Gaussian
    user = 'user'  # user guesses


class TSJobTypesEnum(str, Enum):
    """
    The available job types in a TS search. The job types which are available are a finite set.
    """

    opt = 'opt'  # geometry optimization
    freq = 'freq'  # frequency calculation
    sp = 'sp'  # single point calculation
    irc = 'irc'  # internal redundant coordinate calculation


class TSSearch(object):
    """
    A class for performing reaction transition state searches.

    Args:
        arc_reaction (ARCReaction): The ARC Reaction object.
        methods (list): The TS search methods to carry out.
                        Allowed values: 'autotst', 'gsm', 'heuristics', 'kinbot', 'ml', 'neb_ase',
                        'neb_terachem', 'qst2', 'user'.
        levels (dict): Keys are job types (allowed values are 'opt', 'freq', 'sp', 'irc'),
                       values are the corresponding levels of theory dictionaries.
                       Note: IRC should be done at the same level as the geometry optimization,
                       if a different level is specified an additional optimization will be spawned
                       prior to the IRC calculation.
        user_guesses (list, optional): Entries are string representations of Cartesian coordinate.
        rmg_db (RMGDatabase, optional): The RMG database object, mandatory for the following methods:
                                        'autotst', 'heuristics', 'kinbot'.
        dihedral_increment (float, optional): The scan dihedral increment to use when generating guesses.
    """

    def __init__(
            self,
            arc_reaction: ARCReaction,
            methods: list,
            levels: dict,
            user_guesses: list = None,
            rmg_db: 'RMGDatabase' = None,
            dihedral_increment: float = 20,
    ) -> None:

        if not isinstance(arc_reaction, ARCReaction):
            raise TypeError(f'arc_reaction must be an ARCReaction object, '
                            f'got {arc_reaction} which is a {type(arc_reaction)}')

        self.rmg_db = rmg_db
        self.arc_reaction = arc_reaction
        if self.arc_reaction.rmg_reaction is None:
            self.arc_reaction.rmg_reaction_from_arc_species()
        if arc_reaction.family is None:
            self.arc_reaction.family = determine_reaction_family(self.rmg_db, self.arc_reaction.rmg_reaction)[0]

        self.methods = [TSMethodsEnum(method) for method in methods]
        for method in self.methods:
            if method not in list(_registered_ts_adapters.keys()):
                raise ValueError(f'Could not interpret unregistered method {method}')
        self.levels = {TSJobTypesEnum(key): val for key, val in levels.items()}

        self.user_guesses = user_guesses
        self.dihedral_increment = dihedral_increment
        self.ts_guesses = list()

    def execute(self):
        """
        Execute a TS search.

        Returns:
            dict: Entries are dictionaries  Cartesian coordinates of TS guesses generated using the requested methods.
        """

        self.search()
        return self.ts_guesses

    def __repr__(self) -> str:
        """
        A short representation of the current TSSearch.

        Returns:
            str: The desired representation.
        """

        return f"TSSearch(arc_reaction='{self.arc_reaction}', " \
               f"methods='{self.methods}', levels={self.levels}, " \
               f"user_guesses={self.user_guesses}, rmgdb = {self.rmg_db})"

    def search(self) -> None:
        """
        Execute the selected TS search methods.
        Populates the self.ts_guesses list.
        """

        for method in self.methods:
            if found(method):
                ts_adapter = ts_method_factory(ts_adapter=method,
                                               user_guesses=self.user_guesses,
                                               arc_reaction=self.arc_reaction,
                                               dihedral_increment=self.dihedral_increment,
                                               )

                ts_guesses = ts_adapter.generate_guesses()
                for ts_guess in ts_guesses:
                    if not colliding_atoms(xyz=ts_guess):
                        self.ts_guesses.append(ts_guess)


def found(method: TSMethodsEnum,
          raise_error: bool = False,
          ) -> bool:
    """
    Check whether a TSSearch method exists.

    Args:
    method (TSMethodsEnum): The method to check.
    raise_error (bool, optional): Whether to raise an error if the module was not found.

    Returns:
    bool: Whether the module was found,
    """

    # Hard coding heuristics and user guesses, since they always exist
    if method in [TSMethodsEnum.heuristics, TSMethodsEnum.user]:
        return True
    # Hard coding ML to return False, this repository is not up yet
    if method in [TSMethodsEnum.ml]:
        # The ML method is not implemented yet
        return False

    # Todo: fix "which"
    # software_dict = {TSMethodsEnum.autotst: 'autotst',
    #                  TSMethodsEnum.gsm: 'gsm',
    #                  TSMethodsEnum.kinbot: 'kinbot',
    #                  TSMethodsEnum.neb_ase: 'ase',
    #                  TSMethodsEnum.neb_terachem: 'terachem',
    #                  TSMethodsEnum.qst2: 'gaussian',
    #                  }
    # url_dict = {TSMethodsEnum.autotst: 'https://github.com/ReactionMechanismGenerator/AutoTST',
    #             TSMethodsEnum.gsm: 'https://github.com/ZimmermanGroup/molecularGSM',
    #             TSMethodsEnum.kinbot: 'https://github.com/zadorlab/KinBot',
    #             TSMethodsEnum.neb_ase: 'https://wiki.fysik.dtu.dk/ase/install.html',
    #             TSMethodsEnum.neb_terachem: 'http://www.petachem.com/products.html',
    #             TSMethodsEnum.qst2: 'https://gaussian.com/',
    #             }
    #
    # return qcel.util.which(software_dict[method],
    #                        return_bool=True,
    #                        raise_error=raise_error,
    #                        raise_msg=f'Please install {software_dict[method]} to use the {method} method, '
    #                                  f'see {url_dict[method]} for more information',
    #                        )
