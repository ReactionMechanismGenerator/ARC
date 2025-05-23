"""
A module for representing a reaction.
"""

from typing import Dict, List, Optional, Tuple, Union

from arkane.common import get_element_mass
from rmgpy.reaction import Reaction
from rmgpy.species import Species

from arc.common import get_logger
from arc.exceptions import ReactionError, InputError
from arc.family.family import ReactionFamily, get_reaction_family_products
from arc.species.converter import (check_xyz_dict,
                                   sort_xyz_using_indices,
                                   translate_to_center_of_mass,
                                   translate_xyz,
                                   xyz_to_str,
                                   )
from arc.mapping.driver import map_reaction
from arc.species.species import ARCSpecies, check_atom_balance, check_label


logger = get_logger()


class ARCReaction(object):
    """
    A class for representing a chemical reaction.

    Either give reactants and products (just list of labels corresponding to :ref:`ARCSpecies <species>`),
    a reaction label, or an RMG Reaction object.
    If the reactants and products in the RMG Reaction aren't ARCSpecies, they will be created.

    The ARCReaction object stores the labels corresponding to the reactants, products and TS ARCSpecies objects
    as self.reactants, self.products, and self.ts_label, respectively.

    Args:
        label (str, optional): The reaction's label in the format `r1 + r2 <=> p1 + p2`
                               (or unimolecular on either side, as appropriate).
        reactants (List[str], optional): A list of reactant *labels* corresponding to an :ref:`ARCSpecies <species>`.
        products (List[str], optional): A list of product *labels* corresponding to an :ref:`ARCSpecies <species>`.
        r_species (List[ARCSpecies], optional): A list of reactants :ref:`ARCSpecies <species>` objects.
        p_species (List[ARCSpecies], optional): A list of products :ref:`ARCSpecies <species>` objects.
        ts_label (str, optional): The :ref:`ARCSpecies <species>` label of the respective TS.
        ts_xyz_guess (list, optional): A list of TS XYZ user guesses, each in a string format.
        xyz (list, optional): Identical to `ts_xyz_guess`, used as a shortcut.
        multiplicity (int, optional): The reaction surface multiplicity. A trivial guess will be made unless provided.
        charge (int, optional): The reaction surface charge.
        reaction_dict (dict, optional): A dictionary to create this object from (used when restarting ARC).
        species_list (list, optional): A list of ARCSpecies entries for matching reactants and products
                                       to existing species.
        preserve_param_in_scan (list, optional): Entries are length two iterables of atom indices (1-indexed)
                                                 between which distances and dihedrals of these pivots must be
                                                 preserved. Used for identification of rotors which break a TS.
        kinetics (Dict[str, float], optional): The high pressure limit rate coefficient calculated by ARC.
                                               Keys are 'A' in cm-s-mol units, 'n', and 'Ea' in kJ/mol.

    Attributes:
        label (str): The reaction's label in the format `r1 + r2 <=> p1 + p2`
                     (or unimolecular on either side, as appropriate).
        family (str): The RMG kinetic family, if applicable.
        family_own_reverse (bool): Whether the RMG family is its own reverse.
        reactants (List[str]): A list of reactants labels corresponding to an :ref:`ARCSpecies <species>`.
        products (List[str]): A list of products labels corresponding to an :ref:`ARCSpecies <species>`.
        r_species (List[ARCSpecies]): A list of reactants :ref:`ARCSpecies <species>` objects.
        p_species (List[ARCSpecies]): A list of products :ref:`ARCSpecies <species>` objects.
        ts_species (ARCSpecies): The :ref:`ARCSpecies <species>` corresponding to the reaction's TS.
        dh_rxn298 (float): The heat of reaction at 298K in J/mol.
        kinetics (Dict[str, float]): The high pressure limit rate coefficient calculated by ARC.
                                     Keys are 'A' in cm-s-mol units, 'n', and 'Ea' in kJ/mol.
        rmg_kinetics (List[Dict[str, float]]): The Arrhenius kinetics from RMG's libraries and families.
                                               Each dict has 'A' in cm-s-mol units, 'n', and 'Ea' in kJ/mol as keys,
                                               and a 'comment' key with a description of the source of the kinetics.
        long_kinetic_description (str): A description for the species entry in the thermo library outputted.
        ts_xyz_guess (list): A list of TS XYZ user guesses, each in a string format.
        multiplicity (int): The reaction surface multiplicity. A trivial guess will be made unless provided.
        charge (int): The reaction surface charge.
        index (int): An auto-generated index associating the ARCReaction object with the
                     corresponding TS :ref:`ARCSpecies <species>` object.
        ts_label (str): The :ref:`ARCSpecies <species>` label of the respective TS.
        preserve_param_in_scan (list): Entries are length two iterables of atom indices (1-indexed) between which
                                       distances and dihedrals of these pivots must be preserved.
        atom_map (List[int]): An atom map, mapping the reactant atoms to the product atoms.
                              I.e., an atom map of [0, 2, 1] means that reactant atom 0 matches product atom 0,
                              reactant atom 1 matches product atom 2, and reactant atom 2 matches product atom 1.
        done_opt_r_n_p (bool): Whether the optimization of all reactants and products is complete.
    """
    def __init__(self,
                 label: str = '',
                 reactants: Optional[List[str]] = None,
                 products: Optional[List[str]] = None,
                 r_species: Optional[List[ARCSpecies]] = None,
                 p_species: Optional[List[ARCSpecies]] = None,
                 ts_label: Optional[str] = None,
                 ts_xyz_guess: Optional[list] = None,
                 xyz: Optional[list] = None,
                 multiplicity: Optional[int] = None,
                 charge: Optional[int] = None,
                 reaction_dict: Optional[dict] = None,
                 species_list: Optional[List[ARCSpecies]] = None,
                 preserve_param_in_scan: Optional[list] = None,
                 kinetics: Dict[str, float] = None,
                 ):
        self.arrow = ' <=> '
        self.plus = ' + '
        self.r_species = r_species or list()
        self.p_species = p_species or list()
        self.kinetics = kinetics
        self.rmg_kinetics = None
        self.long_kinetic_description = ''
        self._family = None
        self._family_own_reverse = False
        self.ts_label = ts_label
        self.dh_rxn298 = None
        self.ts_xyz_guess = ts_xyz_guess or xyz or list()
        self.preserve_param_in_scan = preserve_param_in_scan
        self._atom_map = None
        self._charge = charge
        self._multiplicity = multiplicity
        if reaction_dict is not None:
            # Reading from a dictionary
            self.from_dict(reaction_dict=reaction_dict, species_list=species_list)
        else:
            # Not reading from a dictionary
            self.label = label
            self.index = None
            self.ts_species = None
            reactants = reactants or [spc.label for spc in self.r_species]
            self.reactants = [check_label(reactant)[0] for reactant in reactants] if reactants else list()
            products = products or [spc.label for spc in self.p_species] or None
            self.products = [check_label(product)[0] for product in products] if products else list()
            if not self.label \
                    and not (len(self.reactants) * len(self.products)) \
                    and not (len(self.r_species) * len(self.p_species)):
                raise InputError(f'Cannot determine reactants and/or products labels for reaction {self.label}')
            self.set_label_reactants_products()
            self.ts_xyz_guess = ts_xyz_guess or list()
            self.done_opt_r_n_p = None
        if (len(self.reactants) > 3 or len(self.products) > 3) and len(self.ts_xyz_guess) == 0:
            raise ReactionError(f'An ARC Reaction can have up to three reactants / products. got {len(self.reactants)} '
                                f'reactants and {len(self.products)} products for reaction {self.label}.')
        if not isinstance(self.ts_xyz_guess, list):
            self.ts_xyz_guess = [self.ts_xyz_guess]
        self.remove_dup_species()
        self.check_atom_balance()

    @property
    def atom_map(self):
        """The reactants to products atom map"""
        if self._atom_map is None \
                and all(species.get_xyz(generate=False) is not None for species in self.r_species + self.p_species):
            for backend in ["ARC", "QCElemental"]:
                _atom_map = map_reaction(rxn=self, backend=backend)
                if _atom_map is not None:
                    self._atom_map = _atom_map
                    break
                logger.error(f"The requested ARC reaction {self}, and it's reverse, could not be atom mapped using {backend}.")
        if self._atom_map is None:
            logger.error(f"The requested ARC reaction {self} could not be atom mapped.")
        return self._atom_map

    @atom_map.setter
    def atom_map(self, value):
        """Allow setting the atom map"""
        self._atom_map = value

    @property
    def charge(self):
        """The net electric charge of the reaction PES"""
        if self._charge is None:
            if len(self.r_species):
                self._charge = self.get_rxn_charge()
        return self._charge

    @charge.setter
    def charge(self, value):
        """Allow setting the reaction charge"""
        if value is not None and not isinstance(value, int):
            raise InputError(f'Reaction charge must be an integer, got {value} which is a {type(value)}.')
        self._charge = value

    @property
    def multiplicity(self):
        """The electron spin multiplicity of the reaction PES"""
        if self._multiplicity is None:
            self._multiplicity = self.get_rxn_multiplicity()
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, value):
        """Allow setting the reaction multiplicity"""
        self._multiplicity = value
        if value is not None:
            if not isinstance(value, int):
                raise InputError(f'Reaction multiplicity must be an integer, got {value} which is a {type(value)}.')
            logger.info(f'Setting multiplicity of reaction {self.label} to {self._multiplicity}')

    @property
    def family(self):
        """The RMG reaction family"""
        if self._family is None:
            self._family, self._family_own_reverse = self.determine_family()
        return self._family

    @family.setter
    def family(self, value):
        """Allow setting family"""
        self._family = value
        if value is not None and not isinstance(value, str):
            raise InputError(f'Reaction family must be a string, got {value} which is a {type(value)}.')

    @property
    def family_own_reverse(self):
        """The RMG reaction family's own reverse property"""
        if self._family_own_reverse is None:
            self._family, self._family_own_reverse = self.determine_family()
        return self._family_own_reverse

    @family_own_reverse.setter
    def family_own_reverse(self, value):
        """Allow setting family_own_reverse"""
        self._family_own_reverse = bool(value)

    def __str__(self) -> str:
        """Return a string representation of the object"""
        str_representation = f'ARCReaction('
        str_representation += f'label="{self.label}", '
        if self.preserve_param_in_scan is not None:
            str_representation += f'preserve_param_in_scan="{self.preserve_param_in_scan}", '
        str_representation += f'multiplicity={self.multiplicity}, '
        str_representation += f'charge={self.charge})'
        return str_representation

    def as_dict(self,
                reset_atom_ids: bool = False,
                report_family: bool = True,
                ) -> dict:
        """
        A helper function for dumping this object as a dictionary in a YAML file for restarting ARC.

        Args:
            reset_atom_ids (bool, optional): Whether to reset the atom IDs in the .mol Molecule attribute of reactant
                                             and product species. Useful when copying the object to avoid duplicate
                                             atom IDs between different object instances.
            report_family (bool, optional): Whether to report the reaction family.

        Returns:
            dict: The dictionary representation of the object instance.
        """
        reaction_dict = dict()
        reaction_dict['label'] = self.label
        if self.index is not None:
            reaction_dict['index'] = self.index
        reaction_dict['multiplicity'] = self.multiplicity
        if self.charge != 0:
            reaction_dict['charge'] = self.charge
        reaction_dict['reactants'] = self.reactants
        reaction_dict['products'] = self.products
        reaction_dict['r_species'] = [spc.as_dict(reset_atom_ids=reset_atom_ids) for spc in self.r_species]
        reaction_dict['p_species'] = [spc.as_dict(reset_atom_ids=reset_atom_ids) for spc in self.p_species]
        if self.ts_species is not None:
            reaction_dict['ts_species'] = self.ts_species.as_dict()
        if self._atom_map is not None:
            reaction_dict['atom_map'] = self._atom_map
        if self.done_opt_r_n_p is not None:
            reaction_dict['done_opt_r_n_p'] = self.done_opt_r_n_p
        if self.preserve_param_in_scan is not None:
            reaction_dict['preserve_param_in_scan'] = self.preserve_param_in_scan
        if report_family:
            if self.family is not None:
                reaction_dict['family'] = self.family
            if self.family_own_reverse:
                reaction_dict['family_own_reverse'] = self.family_own_reverse
        if self.long_kinetic_description:
            reaction_dict['long_kinetic_description'] = self.long_kinetic_description
        if len(self.ts_xyz_guess):
            reaction_dict['ts_xyz_guess'] = self.ts_xyz_guess
        reaction_dict['label'] = self.label
        if self.ts_label is not None:
            reaction_dict['ts_label'] = self.ts_label
        return reaction_dict

    def from_dict(self,
                  reaction_dict: dict,
                  species_list: Optional[list] = None,
                  ):
        """
        A helper function for loading this object from a dictionary in a YAML file for restarting ARC.
        """
        self.index = reaction_dict['index'] if 'index' in reaction_dict else None
        self.label = reaction_dict['label'] if 'label' in reaction_dict else ''
        self.multiplicity = reaction_dict['multiplicity'] if 'multiplicity' in reaction_dict else None
        self.charge = reaction_dict['charge'] if 'charge' in reaction_dict else 0
        self.reactants = reaction_dict.get('reactants') or list()
        self.products = reaction_dict.get('products') or list()
        if 'family' in reaction_dict and reaction_dict['family'] is not None:
            self.family = reaction_dict['family']
        self.family_own_reverse = reaction_dict['family_own_reverse'] if 'family_own_reverse' in reaction_dict else False
        self.set_label_reactants_products(species_list)
        if not (len(self.reactants) * len(self.products)):
            raise InputError(f'Cannot determine reactants and/or products labels for reaction {self.label}')
        if not (len(self.reactants) * len(self.products)):
            raise InputError(f'All species in a reaction must be labeled (and the labels must correspond '
                             f'to respective Species in ARC).\nProblematic reaction:{self.label}')
        if self.ts_label is None:
            self.ts_label = reaction_dict['ts_label'] if 'ts_label' in reaction_dict else None
        if species_list is not None and 'r_species' in reaction_dict and len(reaction_dict['r_species']) \
                and 'p_species' in reaction_dict and len(reaction_dict['p_species']):
            self.r_species, self.p_species = list(), list()
            for spc in species_list:
                for r_spc_dict in reaction_dict['r_species']:
                    if r_spc_dict['label'] in [spc.label, spc.original_label]:
                        self.r_species.append(spc)
                for p_spc_dict in reaction_dict['p_species']:
                    if p_spc_dict['label'] in [spc.label, spc.original_label]:
                        self.p_species.append(spc)
        else:
            self.r_species = [ARCSpecies(species_dict=r_dict) for r_dict in reaction_dict['r_species']] \
                if 'r_species' in reaction_dict else self.r_species or list()
            self.p_species = [ARCSpecies(species_dict=p_dict) for p_dict in reaction_dict['p_species']] \
                if 'p_species' in reaction_dict else self.p_species or list()
        self.reactants = self.reactants or [spc.label for spc in self.r_species]
        self.products = self.products or [spc.label for spc in self.p_species]
        self.ts_species = ARCSpecies(species_dict=reaction_dict['ts_species']) if 'ts_species' in reaction_dict else None
        self.long_kinetic_description = reaction_dict['long_kinetic_description'] \
            if 'long_kinetic_description' in reaction_dict else ''
        self.ts_xyz_guess = reaction_dict['ts_xyz_guess'] if 'ts_xyz_guess' in reaction_dict \
            else reaction_dict['xyz'] if 'xyz' in reaction_dict else list()
        self.preserve_param_in_scan = reaction_dict['preserve_param_in_scan'] \
            if 'preserve_param_in_scan' in reaction_dict else None
        self.atom_map = reaction_dict['atom_map'] if 'atom_map' in reaction_dict else None
        self.done_opt_r_n_p = reaction_dict['done_opt_r_n_p'] if 'done_opt_r_n_p' in reaction_dict else None

    def copy(self):
        """
        Get a copy of this object instance.

        Returns:
            ARCReaction: A copy of this object instance.
        """
        reaction_dict = self.as_dict(reset_atom_ids=True)
        return ARCReaction(reaction_dict=reaction_dict)

    def flip_reaction(self, report_family: bool = True):
        """
        Get a copy of this object instance with flipped reactants and products.

        Args:
            report_family (bool, optional): Whether to report the reaction family.

        Returns:
            ARCReaction: A copy of this object instance with flipped reactants and products.
        """
        reaction_dict = self.as_dict(reset_atom_ids=True, report_family=report_family)
        reset_keys = ['label', 'index', 'atom_map', 'family', 'family_own_reverse', 'long_kinetic_description']
        if 'r_species' in reaction_dict.keys() and 'p_species' in reaction_dict.keys():
            reaction_dict['r_species'], reaction_dict['p_species'] = reaction_dict['p_species'], reaction_dict['r_species']
        else:
            reset_keys.extend(['r_species', 'p_species'])
        if 'reactants' in reaction_dict.keys() and 'products' in reaction_dict.keys():
            reaction_dict['reactants'], reaction_dict['products'] = reaction_dict['products'], reaction_dict['reactants']
        else:
            reset_keys.extend(['reactants', 'products'])
        for key in reset_keys:
            if key in reaction_dict.keys():
                del reaction_dict[key]
        label_splits = self.label.split(self.arrow)
        reaction_dict['label'] = self.arrow.join([label_splits[1], label_splits[0]])
        flipped_rxn = ARCReaction(reaction_dict=reaction_dict)
        flipped_rxn.set_label_reactants_products()
        return flipped_rxn

    def is_isomerization(self):
        """
        Determine whether this is an isomerization reaction.

        Returns:
            bool: Whether this is an isomerization reaction.
        """
        return True if len(self.r_species) == 1 and len(self.p_species) == 1 else False

    def set_label_reactants_products(self, species_list: Optional[List[ARCSpecies]] = None):
        """A helper function for settings the label, reactants, and products attributes for a Reaction"""
        # First make sure that reactants and products labels are defined (most often used).
        if not len(self.reactants) or not len(self.products):
            if self.label:
                if self.arrow not in self.label:
                    raise ReactionError(f'A reaction label must contain an arrow ("{self.arrow}")')
                reactants, products = self.label.split(self.arrow)
                if self.plus in reactants:
                    self.reactants = reactants.split(self.plus)
                else:
                    self.reactants = [reactants]
                if self.plus in products:
                    self.products = products.split(self.plus)
                else:
                    self.products = [products]
                self.reactants = [reactant.strip() for reactant in self.reactants]
                self.products = [product.strip() for product in self.products]
                if species_list is not None:
                    if len(self.reactants) and len(self.products):
                        labels = [spc.label for spc in species_list]
                        original_labels = [spc.original_label for spc in species_list]
                        for spc_label in self.reactants + self.products:
                            if spc_label not in labels + original_labels:
                                raise ValueError(f'The species {spc_label} appears in the reaction label\n'
                                                 f'{self.label}\n'
                                                 f'Yet no species with a corresponding label was defined in ARC.')
                    if not len(self.r_species) and len(self.reactants):
                        self.r_species = [spc for spc in species_list if spc.label in self.reactants
                                          or spc.original_label in self.reactants]
                    if not len(self.p_species) and len(self.products):
                        self.p_species = [spc for spc in species_list if spc.label in self.products
                                          or spc.original_label in self.products]
            elif len(self.r_species) and len(self.p_species):
                self.reactants = [r.label for r in self.r_species]
                self.products = [p.label for p in self.p_species]
        if not self.label:
            if len(self.reactants) and len(self.products):
                self.label = self.arrow.join([self.plus.join(r for r in self.reactants),
                                              self.plus.join(p for p in self.products)])
            elif len(self.r_species) and len(self.p_species):
                self.label = self.arrow.join([self.plus.join(r.label for r in self.r_species),
                                              self.plus.join(p.label for p in self.p_species)])
        if not self.label and not (len(self.reactants) * len(self.products)):
            raise ReactionError('Either a label or reactants and products lists must be specified')
        self.reactants = [check_label(reactant)[0] for reactant in self.reactants]
        self.products = [check_label(product)[0] for product in self.products]
        if bool(len(self.reactants)) ^ bool(len(self.products)):
            raise ReactionError(f'Both the reactants and products must be specified for a reaction, '
                                f'got: reactants = {self.reactants}, products = {self.products}.')

    def get_rxn_charge(self):
        """A helper function for determining the surface charge"""
        if len(self.r_species):
            return sum([r.charge for r in self.r_species])

    def get_rxn_multiplicity(self):
        """A helper function for determining the surface multiplicity"""
        reactants, products = self.get_reactants_and_products(arc=True)
        multiplicity = None
        ordered_r_mult_list, ordered_p_mult_list = list(), list()
        if len(reactants):
            if len(reactants) == 1:
                return reactants[0].multiplicity
            if len(products) == 1:
                return products[0].multiplicity
            ordered_r_mult_list = sorted([r_spc.multiplicity for r_spc in reactants])
            ordered_p_mult_list = sorted([p_spc.multiplicity for p_spc in products])

        for list_1, list_2 in [(ordered_r_mult_list, ordered_p_mult_list),
                               (ordered_p_mult_list, ordered_r_mult_list)]:
            if all(m == 1 for m in list_1) and multiplicity is None:
                multiplicity = 1  # S + S = S or T
                break
            if all(m == 2 for m in list_1) and len(list_1) == 2 \
                    and all(m == 2 for m in list_2) and len(list_2) == 2 and multiplicity is None:
                multiplicity = 1  # D + D = S or T
                break
            if 2 in list_1 and all(m == 1 for i, m in enumerate(list_1) if i != list_1.index(2)):
                multiplicity = 2  # S + D = D
                break
            if 3 in list_1 and all(m == 1 for i, m in enumerate(list_1) if i != list_1.index(3)):
                multiplicity = 3  # S + T = T
                break
            if 4 in list_1 and all(m == 1 for i, m in enumerate(list_1) if i != list_1.index(4)):
                multiplicity = 4  # S + Q = Q
                break
            if all(m == 2 for m in list_1):
                # D + D = S or T
                # D + D + D = D or Q
                if len(list_1) % 2 == 0:  # even number of D's in list_1, m must be an odd number
                    if any(m > 2 for m in list_2):
                        multiplicity = max(list_2) if max(list_2) % 2 == 1 else max(list_2) - 1
                else:  # odd number of D's in list_1, m must be even
                    multiplicity = max(list_2) if max(list_2) % 2 == 0 else max(list_2) - 1
            if all(m == 3 for m in list_1):
                # T + T = S or P
                # T + T + T = T or 7
                if len(list_1) % 2 == 0:  # even number of T's in list_1, m must be 1 or 5
                    multiplicity = 1
                    logger.warning(f'ASSUMING a multiplicity of 1 (singlet) for reaction {self.label}')
                else:  # odd number of D's in list_1, m must be 3 or 7
                    multiplicity = 3
                    logger.warning(f'ASSUMING a multiplicity of 3 (triplet) for reaction {self.label}')
            if list_1 == [2, 3] and 4 not in list_2:
                # D + T = D or Q
                multiplicity = 2
                logger.warning(f'ASSUMING a multiplicity of 2 (doublet) for reaction {self.label}')

        if multiplicity is None:
            logger.error(f'Could not determine multiplicity for reaction {self.label}')
            return None
        return multiplicity

    def determine_family(self,
                         rmg_family_set: str = 'default',
                         consider_rmg_families: bool = True,
                         consider_arc_families: bool = True,
                         discover_own_reverse_rxns_in_reverse: bool = False,
                         ):
        """
        Determine the RMG reaction family.
        Populates the .family, and .family_own_reverse attributes.

        Args:
            rmg_family_set (str, optional): The RMG family set to use.
            consider_rmg_families (bool, optional): Whether to consider RMG's families in addition to ARC's.
            consider_arc_families (bool, optional): Whether to consider ARC's families in addition to RMG's.
            discover_own_reverse_rxns_in_reverse (bool, optional): Whether to discover own reverse reactions in reverse.
        """
        product_dicts = get_reaction_family_products(rxn=self,
                                                     rmg_family_set=rmg_family_set,
                                                     consider_rmg_families=consider_rmg_families,
                                                     consider_arc_families=consider_arc_families,
                                                     discover_own_reverse_rxns_in_reverse=discover_own_reverse_rxns_in_reverse,
                                                     )
        if len(product_dicts):
            family, family_own_reverse = product_dicts[0]['family'], product_dicts[0]['own_reverse']
            return family, family_own_reverse
        return None, None

    def check_attributes(self):
        """Check that the Reaction object is defined correctly"""
        self.set_label_reactants_products()
        if not self.label:
            raise ReactionError('A reaction seems to not be defined correctly')
        if self.arrow not in self.label:
            raise ReactionError(f'A reaction label must include a double ended arrow with spaces on both '
                                f'sides: "{self.arrow}". Got:{self.label}')
        if '+' in self.label and self.plus not in self.label:
            raise ReactionError(f'Reactants or products in a reaction label must separated with {self.plus} '
                                f'(has spaces on both sides). Got:{self.label}')
        species_labels = self.label.split(self.arrow)
        reactants = [check_label(reactant)[0] for reactant in species_labels[0].split(self.plus)]
        products = [check_label(product)[0] for product in species_labels[1].split(self.plus)]
        if len(self.reactants):
            for reactant in reactants:
                if reactant not in self.reactants:
                    raise ReactionError(f'Reactant {reactant} from the reaction label {self.label} '
                                        f'is not in self.reactants ({self.reactants})')
            for reactant in self.reactants:
                if reactant not in reactants:
                    raise ReactionError(f'Reactant {reactant} is not in the reaction label ({self.label})')
        if len(self.products):
            for product in products:
                if product not in self.products:
                    raise ReactionError(f'Product {product} from the reaction {self.label} '
                                        f'is not in self.products ({self.products})')
            for product in self.products:
                if product not in products:
                    raise ReactionError(f'Product {product} is not in the reaction label ({self.label})')
        if len(self.r_species):
            for reactant in self.r_species:
                if reactant.label not in self.reactants:
                    raise ReactionError(f'Reactant {reactant.label} from {self.label} '
                                        f'is not in self.reactants ({self.reactants})')
            for reactant in reactants:
                if reactant not in [r.label for r in self.r_species]:
                    raise ReactionError(f'Reactant {reactant} from the reaction label {self.label} '
                                        f'is not in self.r_species ({[r.label for r in self.r_species]})')
            for reactant in self.reactants:
                if reactant not in [r.label for r in self.r_species]:
                    raise ReactionError(f'Reactant {reactant} is not in '
                                        f'self.r_species ({[r.label for r in self.r_species]})')
        if len(self.p_species):
            for product in self.p_species:
                if product.label not in self.products:
                    raise ReactionError(f'Product {product.label} from {self.label} '
                                        f'is not in self.products ({self.reactants})')
            for product in products:
                if product not in [p.label for p in self.p_species]:
                    raise ReactionError(f'Product {product} from the reaction label {self.label} '
                                        f'is not in self.p_species ({[p.label for p in self.p_species]})')
            for product in self.products:
                if product not in [p.label for p in self.p_species]:
                    raise ReactionError(f'Product {product} is not in '
                                        f'self.p_species ({[p.label for p in self.p_species]})')

    def remove_dup_species(self):
        """
        Make sure each species is considered only once in reactants, products, r_species, and p_species.
        The same species in the reactants/products is considered through get_species_count().
        """
        self.reactants = sorted(list(set(self.reactants)))
        self.products = sorted(list(set(self.products)))
        self.r_species = remove_dup_species(self.r_species)
        self.p_species = remove_dup_species(self.p_species)

    def check_done_opt_r_n_p(self):
        """
        Check whether the ``final_xyz`` attributes of all ``r_species`` and ``p_species``
        are populated, and flag ``self.done_opt_r_n_p`` as ``True`` if they are.
        Useful to know when to spawn TS search jobs.
        """
        if not self.done_opt_r_n_p:
            self.done_opt_r_n_p = all(spc.final_xyz is not None for spc in self.r_species + self.p_species)

    def check_atom_balance(self,
                           ts_xyz: Optional[dict] = None,
                           raise_error: bool = True,
                           ) -> bool:
        """
        Check atom balance between reactants, TSs, and product wells.

        Args:
            ts_xyz (Optional[dict]): An alternative TS xyz to check.
                                     If unspecified, user guesses and the ts_species will be checked.
            raise_error (bool, optional): Whether to raise an error if an imbalance is found.

        Raises:
            ReactionError: If not all wells and TSs are atom balanced.
                           The exception is not raised if ``raise_error`` is ``False``.

        Returns:
            bool: Whether all wells and TSs are atom balanced.
        """
        balanced_wells, balanced_ts_xyz, balanced_xyz_guess, balanced_ts_species_mol, balanced_ts_species_xyz = \
            True, True, True, True, True
        r_well, p_well = '', ''

        for reactant in self.r_species:
            count = self.get_species_count(species=reactant, well=0)
            xyz = reactant.get_xyz(generate=True)
            if xyz is not None and xyz:
                r_well += (xyz_to_str(xyz) + '\n') * count
            else:
                r_well = ''
                break

        for product in self.p_species:
            count = self.get_species_count(species=product, well=1)
            xyz = product.get_xyz(generate=True)
            if xyz is not None and xyz:
                p_well += (xyz_to_str(xyz) + '\n') * count
            else:
                p_well = ''
                break

        if r_well:
            for xyz_guess in self.ts_xyz_guess:
                balanced_xyz_guess *= check_atom_balance(entry_1=xyz_guess, entry_2=r_well)

            if p_well:
                balanced_wells = check_atom_balance(entry_1=r_well, entry_2=p_well)

            if ts_xyz:
                balanced_ts_xyz = check_atom_balance(entry_1=ts_xyz, entry_2=r_well)

            if self.ts_species is not None:
                if self.ts_species.mol is not None:
                    balanced_ts_species_mol = check_atom_balance(entry_1=self.ts_species.mol, entry_2=r_well)

                ts_xyz = self.ts_species.get_xyz()
                if ts_xyz is not None:
                    balanced_ts_species_xyz = check_atom_balance(entry_1=self.ts_species.get_xyz(), entry_2=r_well)

        if not balanced_wells:
            logger.error(f'The reactant(s) and product(s) wells of reaction {self.label}, are not atom balanced.')
        if not balanced_ts_xyz:
            logger.error(f'The generated TS xyz for reaction {self.label} '
                         f'is not atom balances with the reactant(s) well.')
        if not balanced_ts_species_mol:
            logger.error(f'The TS mol for reaction {self.label} is not atom balances with the reactant(s) well.')
        if not balanced_ts_species_xyz:
            logger.error(f'The TS coordinates for reaction {self.label} '
                         f'are not atom balances with the reactant(s) well.')
        if not balanced_xyz_guess:
            logger.error(f'Check TS xyz user guesses of reaction {self.label}, '
                         f'some are not atom balances with the reactant(s) well.')
        if not all([balanced_wells, balanced_ts_xyz, balanced_ts_species_mol,
                    balanced_ts_species_xyz, balanced_xyz_guess]):
            if raise_error:
                raise ReactionError(f'The Reaction {self.label} is not atom balanced.\n'
                                    f'balanced wells: {balanced_wells}\n'
                                    f'balanced ts xyz: {balanced_ts_xyz}\n'
                                    f'balanced ts species mol: {balanced_ts_species_mol}\n'
                                    f'balanced ts species xyz: {balanced_ts_species_xyz}\n'
                                    f'balanced xyz guess: {bool(balanced_xyz_guess)}')
            return False

        return True

    def get_species_count(self,
                          species: Optional[ARCSpecies] = None,
                          label: Optional[str] = None,
                          well: int = 0,
                          ) -> int:
        """
        Get the number of times a species participates in the reactants or products well.
        Either ``species`` or ``label`` must be given.

        Args:
            species (ARCSpecies, optional): The species to check.
            label (str, optional): The species label.
            well (int, optional): Either ``0`` or ``1`` for the reactants or products well, respectively.

        Returns:
            Optional[int]:
                The number of occurrences of this species in the respective well.
        """
        if species is None and label is None:
            raise ValueError('Called get_species_count without a species nor its label.')
        if well not in [0, 1]:
            raise ValueError(f'Got well = {well}, expected either 0 or 1.')
        label = species.label if species is not None else label
        well_str = self.label.split('<=>')[well]
        wells = [check_label(spc_label)[0] for spc_label in well_str.strip().split(self.plus)]
        count = sum([label == spc_label for spc_label in wells])
        return count

    def get_reactants_and_products(self,
                                   arc: bool = True,
                                   return_copies: bool = True,
                                   ) -> Tuple[List[Union[ARCSpecies, Species]], List[Union[ARCSpecies, Species]]]:
        """
        Get a list of reactant and product species including duplicate species, if any.
        The species could either be ``ARCSpecies`` or ``RMGSpecies`` object instance.

        Args:
            arc (bool, optional): Whether to return the species as ARCSpecies (``True``) or as RMG Species (``False``).
            return_copies (bool, optional): Whether to return unique object instances using the copy() method.

        Returns:
            Tuple[List[Union[ARCSpecies, Species]], List[Union[ARCSpecies, Species]]]:
                The reactants and products.
        """
        reactants, products = list(), list()
        for r_spc in self.r_species:
            if arc:
                for i in range(self.get_species_count(species=r_spc, well=0)):
                    reactants.append(r_spc.copy() if return_copies else r_spc)
            else:
                for i in range(self.get_species_count(species=r_spc, well=0)):
                    reactants.append(Species(label=r_spc.label, molecule=[r_spc.mol.copy(deep=True) if return_copies
                                                                          else r_spc.mol]))
        for p_spc in self.p_species:
            if arc:
                for i in range(self.get_species_count(species=p_spc, well=1)):
                    products.append(p_spc.copy() if return_copies else p_spc)
            else:
                for i in range(self.get_species_count(species=p_spc, well=1)):
                    products.append(Species(label=p_spc.label, molecule=[p_spc.mol.copy(deep=True) if return_copies
                                                                          else p_spc.mol]))
        return reactants, products

    def get_expected_changing_bonds(self,
                                    r_label_dict: Dict[str, int],
                                    ) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
        """
        Get the expected forming and breaking bonds from the RMG reaction template.

        Args:
            r_label_dict (Dict[str, int]): The RMG reaction atom labels and corresponding atom indices
                                           of atoms in a TemplateReaction.

        Returns:
            Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
                A list of tuples of atom indices representing breaking and forming bonds.
        """
        if self.family is None:
            return None, None
        family = ReactionFamily(label=self.family)
        # E.g.: [['BREAK_BOND', '*1', 1, '*2'], ['FORM_BOND', '*2', 1, '*3'], ['GAIN_RADICAL', '*1', '1']]
        expected_breaking_bonds = [tuple(sorted([r_label_dict[action[1]], r_label_dict[action[3]]]))
                                   for action in family.actions if action[0] == 'BREAK_BOND']
        expected_forming_bonds = [tuple(sorted([r_label_dict[action[1]], r_label_dict[action[3]]]))
                                  for action in family.actions if action[0] == 'FORM_BOND']
        return expected_breaking_bonds, expected_forming_bonds

    def get_number_of_atoms_in_reaction_zone(self) -> Optional[int]:
        """
        Get the number of atoms that participate in the reaction zone according to the reaction's RMG recipe.

        Returns:
            int: The number of atoms that participate in the reaction zone.
        """
        if self.family is None:
            return None
        family = ReactionFamily(label=self.family)
        # E.g.: [['BREAK_BOND', '*1', 1, '*2'], ['FORM_BOND', '*2', 1, '*3'], ['GAIN_RADICAL', '*1', '1']
        labels = list()
        for action in family.actions:
            for entry in action:
                if isinstance(entry, str) and '*' in entry:
                    labels.append(entry)
        labels = set(labels)
        return len(labels)

    def get_single_mapped_product_xyz(self) -> Optional[ARCSpecies]:
        """
        Get a copy of the product species with mapped cartesian coordinates of a reaction with a single product.

        Returns:
            Optional[ARCSpecies]: The corresponding ARCSpecies object with mapped coordinates.
        """
        if len(self.p_species) > 1:
            logger.error(f'Can only return a mapped product for reactions with a single product, '
                         f'got {len(self.p_species)}.')
            return None
        mapped_xyz = sort_xyz_using_indices(xyz_dict=self.p_species[0].get_xyz(), indices=self.atom_map)
        mapped_product = ARCSpecies(label=self.p_species[0].label,
                                    mol=self.p_species[0].mol.copy(deep=True),
                                    multiplicity=self.p_species[0].multiplicity,
                                    charge=self.p_species[0].charge,
                                    xyz=mapped_xyz,
                                    )
        return mapped_product

    def get_reactants_xyz(self, return_format='str') -> Union[dict, str]:
        """
        Get a combined string/dict representation of the cartesian coordinates of all reactant species.

        Args:
            return_format (str): Either ``'dict'`` to return a dict format or ``'str'`` to return a string format.
                                 Default: ``'str'``.

        Returns: Union[dict, str]
            The combined cartesian coordinates.

        Todo:
            Orient the fragments according to the reactive site.
        """
        xyz_dict = dict()
        if len(self.r_species) == 1:
            xyz_dict = self.r_species[0].get_xyz()
        elif len(self.r_species) >= 2:
            xyz_dict = {'symbols': tuple(), 'isotopes': tuple(), 'coords': tuple()}
            for i, reactant in enumerate(self.r_species):
                xyz = translate_to_center_of_mass(reactant.get_xyz())
                if i:
                    xyz = translate_xyz(xyz_dict=xyz,
                                        translation=(sum(spc.radius for spc in self.r_species[:i]) * 1.1 * i, 0, 0))
                xyz_dict['symbols'] += xyz['symbols']
                xyz_dict['isotopes'] += xyz['isotopes']
                xyz_dict['coords'] += xyz['coords']
        xyz_dict = translate_to_center_of_mass(check_xyz_dict(xyz_dict))
        if return_format == 'str':
            xyz_dict = xyz_to_str(xyz_dict)
        return xyz_dict

    def get_products_xyz(self, return_format='str') -> Union[dict, str]:
        """
        Get a combined string/dict representation of the cartesian coordinates of all product species.
        The resulting coordinates are ordered as the reactants using an atom map.

        Args:
            return_format (str): Either ``'dict'`` to return a dict format or ``'str'`` to return a string format.
                                 Default: ``'str'``.

        Returns: Union[dict, str]
            The combined cartesian coordinates.

        Todo:
            Orient the fragments according to the reactive site.
        """
        if len(self.p_species) == 1:
            xyz_dict = self.p_species[0].get_xyz()
        else:
            xyz_dict = {'symbols': tuple(), 'isotopes': tuple(), 'coords': tuple()}
            for i, product in enumerate(self.p_species):
                xyz = translate_to_center_of_mass(product.get_xyz())
                if i:
                    xyz = translate_xyz(xyz_dict=xyz,
                                        translation=(sum(spc.radius for spc in self.p_species[:i]) * 1.1 * i, 0, 0))
                xyz_dict['symbols'] += xyz['symbols']
                xyz_dict['isotopes'] += xyz['isotopes']
                xyz_dict['coords'] += xyz['coords']
        xyz_dict = translate_to_center_of_mass(check_xyz_dict(xyz_dict))
        xyz_dict = sort_xyz_using_indices(xyz_dict=xyz_dict, indices=self.atom_map)
        if return_format == 'str':
            xyz_dict = xyz_to_str(xyz_dict)
        return xyz_dict

    def get_element_mass(self) -> List[float]:
        """
        Get the mass of all elements of a reaction. Uses the atom order of the reactants.

        Returns:
            List[float]: The masses of all elements in the reactants.
        """
        masses = list()
        for reactant in self.get_reactants_and_products()[0]:
            for atom in reactant.mol.atoms:
                masses.append(get_element_mass(atom.element.symbol)[0])
        return masses

    def get_bonds(self) -> Tuple[list, list]:
        """
        Get the connectivity of the reactants and products.

        Returns:
            Tuple[List[Tuple[int, int]]]:
                A length-2 tuple is which entries represent reactants and product information, respectively.
                Each entry is a list of tuples, each represents a bond and contains sorted atom indices.
        """
        r_bonds, p_bonds = list(), list()
        for bonds, spc_list in zip([r_bonds, p_bonds], [self.r_species, self.p_species]):
            len_atoms = 0
            for spc in spc_list:
                for i, atom_1 in enumerate(spc.mol.atoms):
                    for atom2, bond12 in atom_1.edges.items():
                        bond = tuple(sorted([i + len_atoms, spc.mol.atoms.index(atom2) + len_atoms]))
                        if bond not in bonds:
                            bonds.append(bond)
                len_atoms += spc.number_of_atoms
        return r_bonds, p_bonds

    def copy_e0_values(self, other_rxn: Optional['ARCReaction']):
        """
        Copy the E0 values from another reaction object instance for the TS
        and for all species if they have corresponding labels.

        Args:
            other_rxn (ARCReaction): An ARCReaction object instance from which E0 values will be copied.
        """
        if other_rxn is not None:
            self.ts_species.e0 = self.ts_species.e0 or other_rxn.ts_species.e0
            for spc in self.r_species + self.p_species:
                for other_spc in other_rxn.r_species + other_rxn.p_species:
                    if spc.label == other_spc.label:
                        spc.e0 = spc.e0 or other_spc.e0

    def get_rxn_smiles(self) -> Optional[str]:
        """
        returns the reaction smiles of the reaction.

    Raises:
        ValueError: If any of the species (reactants or products) has no SMILES (or could not be generated for some reason).

    Returns: string
        The reaction SMILES
        """
        reactants, products = self.get_reactants_and_products(arc=True, return_copies=True)
        smiles_r = [reactant.mol.copy(deep=True).to_smiles() for reactant in reactants]
        smiles_p = [product.mol.copy(deep=True).to_smiles() for product in products]
        if not any(smiles_r) or not any(smiles_p):
            raise ValueError(f"""Could not find smiles for one or more species
                                 got: reactants: {smiles_r}
                                      products: {smiles_p}""")
        return ".".join(smiles_r)+">>"+".".join(smiles_p)


def remove_dup_species(species_list: List[ARCSpecies]) -> List[ARCSpecies]:
    """
    Remove duplicate species from a species list.
    Used when assigning r_species and p_species.

    Args:
        species_list (List[ARCSpecies]): The species list to process.

    Returns:
        List[ARCSpecies]: A list of species without duplicates.
    """
    if species_list is None or not(len(species_list)):
        return list()
    new_species_list = list()
    for species in species_list:
        if species.label not in [spc.label for spc in new_species_list]:
            new_species_list.append(species)
    return new_species_list
