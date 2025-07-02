"""
A module for working with RMG reaction families.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import ast
import os
import re

from arc.common import clean_text, get_logger
from arc.imports import settings
from arc.molecule import Bond, Group, Molecule
from arc.molecule.resonance import generate_resonance_structures_safely

if TYPE_CHECKING:
    from arc.species import ARCSpecies
    from arc.reaction.reaction import ARCReaction

RMG_DB_PATH = settings['RMG_DB_PATH']
ARC_FAMILIES_PATH = settings['ARC_FAMILIES_PATH']


logger = get_logger()


class ReactionFamily(object):
    """
    A class for representing a reaction family.

    Args:
        label (str): The reaction family label.
        consider_arc_families (bool, optional): Whether to consider ARC's custom families
                                                 when searching for the family groups file.

    Attributes:
        label (str): The reaction family label.
    """

    def __init__(self,
                 label: str,
                 consider_arc_families: bool = True,
                 ):
        if label is None:
            raise ValueError('Cannot initialize a ReactionFamily object without a label')
        self.label = label
        self.groups_as_lines = self.get_groups_file_as_lines(consider_arc_families=consider_arc_families)
        self.reversible = is_reversible(self.groups_as_lines)
        self.own_reverse = is_own_reverse(self.groups_as_lines)
        self.reactants = get_reactant_groups_from_template(self.groups_as_lines)
        self.reactant_num = self.get_reactant_num()
        self.product_num = get_product_num(self.groups_as_lines)
        entry_labels = list()
        for reactant_group in self.reactants:
            entry_labels.extend(reactant_group)
        self.entries = get_entries(self.groups_as_lines, entry_labels=entry_labels)
        self.actions = get_recipe_actions(self.groups_as_lines)

    def __str__(self):
        """
        A string representation of the object.
        """
        return f'ReactionFamily(label={self.label})'

    def get_groups_file_as_lines(self, consider_arc_families: bool = True) -> List[str]:
        """
        Get the groups file as a list of lines.
        Precedence is given to RMG families (ARC families should therefore have distinct names than RMG's)

        Args:
            consider_arc_families (bool, optional): Whether to consider ARC's custom families.

        Returns:
            List[str]: The groups file as a list of lines.
        """
        groups_path = os.path.join(RMG_DB_PATH, 'input', 'kinetics', 'families', self.label, 'groups.py')
        if not os.path.isfile(groups_path):
            if consider_arc_families:
                groups_path = os.path.join(ARC_FAMILIES_PATH, f'{self.label}.py')
            if not os.path.isfile(groups_path):
                raise FileNotFoundError(f'Could not find the groups file for family {self.label}')
        with open(groups_path, 'r') as f:
            groups_as_lines = f.readlines()
        return groups_as_lines

    def generate_products(self,
                          reactants: List['ARCSpecies'],
                          ) -> Dict[Union[str, Tuple[str, str]], List[Tuple[List['Molecule'], Dict[int, str]]]]:
        """
        Generate a list of all the possible reaction products of this family starting from the list of ``reactants``.

        reactant_to_group_maps has the following structure::

            {0: [{'group': 1, 'subgroup': <label of group that is subgraphisomorphic with reactant 0>}, ...],
             1: [{'group': 0, 'subgroup': <label of group that is subgraphisomorphic with reactant 1>}, ...]}

        Args:
            reactants (List['ARCSpecies']): The reactants to generate reaction products for.

        Returns:
            Dict[Union[str, Tuple[str, str]], List[Tuple[List['Molecule'], Dict[int, str]]]]:
                The generated reaction products in different possible reaction paths including iso-teleological paths.
                keys are family group labels used to generate the products,
                values are a list of the corresponding possible products and isomorphic subgraphs.
        """
        if self.reactant_num != len(reactants):
            return dict()
        reactant_to_group_maps = dict()
        for reactant_idx, reactant in enumerate(reactants):
            for groups_idx, group_labels in enumerate(self.reactants):
                for group_label in group_labels:
                    group = Group().from_adjacency_list(
                        get_group_adjlist(self.groups_as_lines, entry_label=group_label))
                    for mol in reactant.mol_list or [reactant.mol]:
                        splits = group.split()
                        if mol.is_subgraph_isomorphic(other=group, save_order=True) \
                                or len(splits) > 1 and any(mol.is_subgraph_isomorphic(other=g, save_order=True) for g in splits):
                            if reactant_idx not in reactant_to_group_maps:
                                reactant_to_group_maps[reactant_idx] = list()
                            reactant_to_group_maps[reactant_idx].append({'group': groups_idx, 'subgroup': group_label})
        if self.reactant_num != len(reactant_to_group_maps.keys()):
            return dict()
        return self.generate_products_by_reactant_groups(reactants=reactants,
                                                         reactant_to_group_maps=reactant_to_group_maps)

    def generate_products_by_reactant_groups(self,
                                             reactants: List['ARCSpecies'],
                                             reactant_to_group_maps: Dict[int, List[Dict[str, Union[int, str]]]],
                                             ) -> Dict[Union[str, Tuple[str, str]], List[Tuple[List['Molecule'], Dict[int, str]]]]:
        """
        Generate a list of all the possible reaction products of this family starting from the list of ``reactants``
        and the mapping of reactant indices to family groups.

        Args:
            reactants (List['ARCSpecies']): The reactants to generate reaction products for.
            reactant_to_group_maps (Dict[int, List[Dict[str, Union[int, str]]]]): A dictionary mapping reactant indices
                                                                                  to groups that match them.

        Returns:
            Dict[str, List[Tuple[List['Molecule'], Dict[int, str]]]]:
                The generated reaction products.
                Keys are family group labels used to generate the products.
                Values are lists of tuples with entries:
                (0) a list of the corresponding possible reactions, (1) the isomorphic subgraph.
        """
        if len(reactants) == 1:
            return self.generate_unimolecular_products(reactants=reactants,
                                                       reactant_to_group_maps=reactant_to_group_maps)
        elif len(reactants) == 2:
            return self.generate_bimolecular_products(reactants=reactants,
                                                      reactant_to_group_maps=reactant_to_group_maps)
        logger.error(f'Unsupported number of reactants encountered in generate_products_by_reactant_groups: '
                     f'{len(reactants)}')
        return dict()

    def generate_unimolecular_products(self,
                                       reactants: List['ARCSpecies'],
                                       reactant_to_group_maps: Dict[int, List[Dict[str, Union[int, str]]]],
                                       ) -> Dict[str, List[Tuple[List['Molecule'], Dict[int, str]]]]:
        """
        Generate a list of all the possible unimolecular reaction products of this family starting from
        the list of ``reactants`` and the mapping of reactant indices to family groups.

        Args:
            reactants (List['ARCSpecies']): The reactants to generate reaction products for.
            reactant_to_group_maps (Dict[int, List[Dict[str, Union[int, str]]]]): A dictionary mapping reactant indices
                                                                                  to groups that match them.

        Returns:
            Dict[str, List[Tuple[List['Molecule'], Dict[int, str]]]]:
                The generated reaction products.
                Keys are family group labels used to generate the products.
                Values are lists of tuples with entries:
                (0) a list of the corresponding possible reactions, (1) the isomorphic subgraph.
        """
        isomorphic_subgraph_dicts = list()
        reactant_to_group_maps = reactant_to_group_maps[0]
        for mol in reactants[0].mol_list or [reactants[0].mol]:
            for reactant_to_group_map in reactant_to_group_maps:
                group = Group().from_adjacency_list(get_group_adjlist(self.groups_as_lines,
                                                                      entry_label=reactant_to_group_map['subgroup']))
                isomorphic_subgraphs = mol.find_subgraph_isomorphisms(other=group, save_order=True)
                if len(isomorphic_subgraphs):
                    for isomorphic_subgraph in isomorphic_subgraphs:
                        isomorphic_subgraph_dicts.append(
                            {'mols': [mol],
                             'groups': [group],
                             'subgroup': reactant_to_group_map['subgroup'],
                             'isomorphic_subgraph': {mol.atoms.index(atom): group_atom.label
                                                     for atom, group_atom in isomorphic_subgraph.items()}})
        products_by_group = dict()
        for isomorphic_subgraph_dict in isomorphic_subgraph_dicts:
            try:
                product_list = self.apply_recipe(mols=isomorphic_subgraph_dict['mols'],
                                                 isomorphic_subgraph=isomorphic_subgraph_dict['isomorphic_subgraph'])
            except ValueError:
                continue
            if product_list is not None:
                if isomorphic_subgraph_dict['subgroup'] not in products_by_group:
                    products_by_group[isomorphic_subgraph_dict['subgroup']] = list()
                products_by_group[isomorphic_subgraph_dict['subgroup']].append((product_list,
                                                                                isomorphic_subgraph_dict['isomorphic_subgraph']))
        return products_by_group

    def generate_bimolecular_products(self,
                                      reactants: List['ARCSpecies'],
                                      reactant_to_group_maps: Dict[int, List[Dict[str, Union[int, str]]]],
                                      ) -> Dict[Tuple[str, str], List[Tuple[List['Molecule'], Dict[int, str]]]]:
        """
        Generate a list of all the possible bimolecular reaction products of this family starting from
        the list of ``reactants`` and the mapping of reactant indices to family groups.

        reactant_to_group_maps has the following structure::

            {0: [{'group': 1, 'subgroup': <label of group that is subgraphisomorphic with reactant 0>}, ...],
             1: [{'group': 0, 'subgroup': <label of group that is subgraphisomorphic with reactant 1>}, ...]}

        Args:
            reactants (List['ARCSpecies']): The reactants to generate reaction products for.
            reactant_to_group_maps (Dict[int, List[Dict[str, Union[int, str]]]]): A dictionary mapping reactant indices
                                                                                  to groups that match them.

        Returns:
            Dict[Tuple[str, str], List[Tuple[List['Molecule'], Dict[int, str]]]]:
                The generated reaction products.
                Keys are family group labels used to generate the products.
                Values are lists of tuples with entries:
                (0) a list of the corresponding possible reactions, (1) the isomorphic subgraph.
        """
        if list(reactant_to_group_maps.keys()) != [0, 1]:
            return dict()
        isomorphic_subgraph_dicts = list()
        for mol_1 in reactants[0].mol_list or [reactants[0].mol]:
            for mol_2 in reactants[1].mol_list or [reactants[1].mol]:
                splits = Group().from_adjacency_list(
                    get_group_adjlist(self.groups_as_lines, entry_label=reactant_to_group_maps[0][0]['subgroup'])).split()
                if len(splits) > 1:
                    for i in [0, 1]:
                        isomorphic_subgraphs_1 = mol_1.find_subgraph_isomorphisms(other=splits[i], save_order=True)
                        isomorphic_subgraphs_2 = mol_2.find_subgraph_isomorphisms(other=splits[not i], save_order=True)
                        if len(isomorphic_subgraphs_1) and len(isomorphic_subgraphs_2):
                            for isomorphic_subgraph_1 in isomorphic_subgraphs_1:
                                for isomorphic_subgraph_2 in isomorphic_subgraphs_2:
                                    isomorphic_subgraph_dicts.append(
                                        {'mols': [mol_1, mol_2],
                                         'subgroups': reactant_to_group_maps[0][0]['subgroup'],
                                         'isomorphic_subgraph': get_isomorphic_subgraph(isomorphic_subgraph_1,
                                                                                        isomorphic_subgraph_2,
                                                                                        mol_1,
                                                                                        mol_2)})
                    continue
                for reactant_to_group_map_1 in reactant_to_group_maps[0]:
                    group_1 = Group().from_adjacency_list(get_group_adjlist(self.groups_as_lines,
                                                                            entry_label=reactant_to_group_map_1['subgroup']))
                    for reactant_to_group_map_2 in reactant_to_group_maps[1]:
                        group_2 = Group().from_adjacency_list(get_group_adjlist(self.groups_as_lines,
                                                                                entry_label=reactant_to_group_map_2['subgroup']))
                        isomorphic_subgraphs_1 = mol_1.find_subgraph_isomorphisms(other=group_1, save_order=True)
                        isomorphic_subgraphs_2 = mol_2.find_subgraph_isomorphisms(other=group_2, save_order=True)
                        if len(isomorphic_subgraphs_1) and len(isomorphic_subgraphs_2):
                            for isomorphic_subgraph_1 in isomorphic_subgraphs_1:
                                for isomorphic_subgraph_2 in isomorphic_subgraphs_2:
                                    isomorphic_subgraph_dicts.append(
                                        {'mols': [mol_1, mol_2],
                                         'subgroups': (reactant_to_group_map_1['subgroup'],
                                                       reactant_to_group_map_2['subgroup']),
                                         'isomorphic_subgraph': get_isomorphic_subgraph(isomorphic_subgraph_1,
                                                                                        isomorphic_subgraph_2,
                                                                                        mol_1,
                                                                                        mol_2)})
        products_by_group = dict()
        for isomorphic_subgraph_dict in isomorphic_subgraph_dicts:
            try:
                product_list = self.apply_recipe(mols=isomorphic_subgraph_dict['mols'],
                                                 isomorphic_subgraph=isomorphic_subgraph_dict['isomorphic_subgraph'])
            except ValueError:
                continue
            if product_list is not None:
                if isomorphic_subgraph_dict['subgroups'] not in products_by_group:
                    products_by_group[isomorphic_subgraph_dict['subgroups']] = list()
                products_by_group[isomorphic_subgraph_dict['subgroups']].append((product_list,
                                                                                isomorphic_subgraph_dict['isomorphic_subgraph']))
        return products_by_group

    def apply_recipe(self,
                     mols: List['Molecule'],
                     isomorphic_subgraph: Dict[int, str],
                     ) -> Optional[List['Molecule']]:
        """
        Generate a reaction product of this family from a reactant mol and the isomorphic subgraph
        using the family's recipe.

        Args:
            mols (['Molecule']): The reactant molecule(s).
            isomorphic_subgraph (Dict[int, str]): A dictionary representing the isomorphic subgraph.

        Raises:
            ValueError: If an invalid action is encountered.

        Returns:
            Optional[List['Molecule']]: The generated reaction product(s).
        """
        structure = Molecule()
        for mol in mols:
            structure = structure.merge(mol.copy(deep=True))
        structure = add_labels_to_molecule(structure, isomorphic_subgraph)
        for action in self.actions:
            if action[0] in ['CHANGE_BOND', 'FORM_BOND', 'BREAK_BOND']:
                structure.reset_connectivity_values()
                label_1, info, label_2 = action[1:]
                atom_1 = structure.get_labeled_atoms(label_1)[0]
                atom_2 = structure.get_labeled_atoms(label_2)[0]
                if atom_1 is None or atom_2 is None or atom_1 is atom_2:
                    raise ValueError('Invalid atom labels in reaction recipe.')
                if action[0] == 'CHANGE_BOND':
                    if not structure.has_bond(atom_1, atom_2):
                        raise ValueError('Attempted to change a nonexistent bond.')
                    bond = structure.get_bond(atom_1, atom_2)
                    if bond.is_benzene():
                        structure.props['validAromatic'] = False
                    atom_1.apply_action(['CHANGE_BOND', label_1, info, label_2])
                    atom_2.apply_action(['CHANGE_BOND', label_1, info, label_2])
                    bond.apply_action(['CHANGE_BOND', label_1, info, label_2])
                elif action[0] == 'FORM_BOND':
                    if structure.has_bond(atom_1, atom_2):
                        raise ValueError('Attempted to create an existing bond.')
                    if info != 1:
                        raise ValueError(f'Attempted to create bond of type {info}')
                    bond = Bond(atom_1, atom_2, order=info)
                    structure.add_bond(bond)
                    atom_1.apply_action(['FORM_BOND', label_1, info, label_2])
                    atom_2.apply_action(['FORM_BOND', label_1, info, label_2])
                elif action[0] == 'BREAK_BOND':
                    if not structure.has_bond(atom_1, atom_2):
                        raise ValueError('Attempted to remove a nonexistent bond.')
                    bond = structure.get_bond(atom_1, atom_2)
                    structure.remove_bond(bond)
                    atom_1.apply_action(['BREAK_BOND', label_1, info, label_2])
                    atom_2.apply_action(['BREAK_BOND', label_1, info, label_2])
            elif action[0] in ['LOSE_RADICAL', 'GAIN_RADICAL', 'LOSE_PAIR', 'GAIN_PAIR']:
                label, change = action[1:]
                change = int(change)
                atom = structure.get_labeled_atoms(label)[0]
                if atom is None:
                    raise ValueError(f'Unable to find atom with label "{label}" while applying reaction recipe.')
                atom.apply_action([action[0], label, change])
            else:
                raise ValueError(f'Unknown action "{action[0]}" encountered.')
        if 'validAromatic' in structure.props and not structure.props['validAromatic']:
            structure.kekulize()
        for atom in structure.atoms:
            atom.update_charge()
        structures = structure.split()
        if self.product_num != len(structures):
            return None
        updated_structures = list()
        for structure in structures:
            structure.update(log_species=False, raise_atomtype_exception=False, sort_atoms=False)
            updated_structures.append(structure)
        return updated_structures

    def get_reactant_num(self) -> int:
        """
        Get the number of reactants for this family.

        Returns:
            int: The number of reactants.
        """
        if len(self.reactants) == 1:
            group = Group().from_adjacency_list(get_group_adjlist(self.groups_as_lines, entry_label=self.reactants[0][0]))
            groups = group.split()
            return len(groups)
        else:
            return len(self.reactants)


def get_reaction_family_products(rxn: 'ARCReaction',
                                 rmg_family_set: Optional[Union[List[str], str]] = None,
                                 consider_rmg_families: bool = True,
                                 consider_arc_families: bool = True,
                                 discover_own_reverse_rxns_in_reverse: bool = False,
                                 ) -> List[dict]:
    """
    Determine the RMG reaction family for a given ARC reaction by generating corresponding product dicts.

    Args:
        rxn ('ARCReaction'): The ARC reaction object.
        rmg_family_set (Union[List[str], str], optional): The RMG family set to use from
                                                          RMG-database/input/kinetics/families/recommended.py.
                                                          Can be a name of a defined set, or a list
                                                          of explicit family labels to consider.
                                                          Note that surface families are excluded if 'all' is used.
        consider_rmg_families (bool, optional): Whether to consider RMG's families.
        consider_arc_families (bool, optional): Whether to consider ARC's custom families.
        discover_own_reverse_rxns_in_reverse (bool, optional): Whether to discover reactions belonging to a family
                                                               which is its own reverse (its template applied in
                                                               reverse is the same) from both directions.
                                                               ``True`` will cause the function to discover reactions
                                                               in both directions, ``False`` will cause the function
                                                               to discover reactions only in the forward direction.

    Returns:
        List[dict]: The list of product dictionaries with the reaction family label.
                    Keys are: 'family', 'group_labels', 'products', 'own_reverse', 'discovered_in_reverse', 'actions'.
    """
    family_labels = get_all_families(rmg_family_set=rmg_family_set,
                                     consider_rmg_families=consider_rmg_families,
                                     consider_arc_families=consider_arc_families)
    product_dicts = list()
    for family_label in family_labels:
        # Forward:
        products = determine_possible_reaction_products_from_family(rxn=rxn,
                                                                    family_label=family_label,
                                                                    consider_arc_families=consider_arc_families,
                                                                    reverse=False,
                                                                    )
        if len(products):
            product_dicts.extend(filter_products_by_reaction(rxn=rxn, product_dicts=products))

        # Reverse:
        flipped_rxn = rxn.flip_reaction(report_family=False)
        products = determine_possible_reaction_products_from_family(rxn=flipped_rxn,
                                                                    family_label=family_label,
                                                                    consider_arc_families=consider_arc_families,
                                                                    reverse=True,
                                                                    )
        if len(products):
            filtered_products = filter_products_by_reaction(rxn=flipped_rxn, product_dicts=products)
            if not discover_own_reverse_rxns_in_reverse:
                product_dicts.extend([prod for prod in filtered_products if not prod['own_reverse']])
            else:
                product_dicts.extend(filtered_products)
    return product_dicts


def determine_possible_reaction_products_from_family(rxn: 'ARCReaction',
                                                     family_label: str,
                                                     consider_arc_families: bool = True,
                                                     reverse: bool = False,
                                                     ) -> List[dict]:
    """
    Determine the possible reaction products for a given ARC reaction and a given RMG reaction family.

    Structure of the returned product_dicts::

        [{'family': str: Family label,
          'group_labels': Tuple[str, str]: Group labels used to generate the products,
          'products': List['Molecule']: The generated products,
          'r_label_map': Dict[int, str]: Mapping of reactant atom indices to labels,
          'p_label_map': Dict[str, int]: Mapping of product labels to atom indices
                                         (refers to the given 'products' in this dict
                                         and not to the products of the original reaction),
          'own_reverse': bool: Whether the family's template also represents its own reverse,
          'discovered_in_reverse': bool: Whether the reaction was discovered in reverse},
         ]

    Args:
        rxn ('ARCReaction'): The ARC reaction object.
        family_label (str): The reaction family label.
        consider_arc_families (bool, optional): Whether to consider ARC's custom families.
        reverse (bool, optional): Whether the reaction is in reverse.

    Returns:
        List[dict]: A list of dictionaries, each containing the family label, the group labels, the products,
                    and whether the family's template also represents its own reverse.
    """
    product_dicts = list()
    family = ReactionFamily(label=family_label, consider_arc_families=consider_arc_families)
    products = family.generate_products(reactants=rxn.get_reactants_and_products(return_copies=True)[0])
    if products:
        for group_labels, product_lists in products.items():
            for product_list in product_lists:
                if isomorphic_products(rxn=rxn, products=product_list[0]):
                    product_dicts.append({'family': family_label,
                                          'group_labels': group_labels,
                                          'products': product_list[0],
                                          'r_label_map': {val: key for key, val in product_list[1].items() if val},
                                          'p_label_map': {atom.label: j + sum(len(p.atoms) for p in product_list[0][:i])
                                                          for i, product in enumerate(product_list[0])
                                                          for j, atom in enumerate(product.atoms) if atom.label},
                                          'own_reverse': family.own_reverse,
                                          'discovered_in_reverse': reverse,
                                          })
    return product_dicts


def filter_products_by_reaction(rxn: 'ARCReaction',
                                product_dicts: List[dict],
                                ) -> List[dict]:
    """
    Filter the possible reaction products by the ARC reaction.

    Args:
        rxn ('ARCReaction'): The ARC reaction object.
        product_dicts (List[dict]): A list of dictionaries, each containing the family label, the group labels,
                                    the products, and whether the family's template also represents its own reverse.

    Returns:
        List[dict]: The filtered list of product dictionaries.
    """
    filtered_product_dicts, r_label_maps = list(), list()
    _, p_species = rxn.get_reactants_and_products(return_copies=True)
    for product_dict in product_dicts:
        if len(product_dict['products']) != len(p_species):
            continue
        if product_dict['r_label_map'] in r_label_maps:
            continue
        if check_product_isomorphism(product_dict['products'], p_species):
            filtered_product_dicts.append(product_dict)
            r_label_maps.append(product_dict['r_label_map'])
    return filtered_product_dicts


def check_product_isomorphism(products: List['Molecule'],
                              p_species: List['ARCSpecies'],
                              ) -> bool:
    """
    Check whether the products are isomorphic to the given species.
    Supports unimolecular and bimolecular reactions.

    Args:
        products (Tuple[List['Molecule'], Dict[int, str]]): The products to check.
        p_species (List['ARCSpecies']): The species to check against.

    Returns:
        bool: Whether the products are isomorphic to the species.
    """
    prods_a = [generate_resonance_structures_safely(mol) or [mol.copy(deep=True)] for mol in products]
    prods_b = [spc.mol_list or [spc.mol] for spc in p_species]
    if len(prods_a) == 1:
        prod_a = prods_a[0]
        prod_b = prods_b[0]
        for mol_a in prod_a:
            if any(mol_b.is_isomorphic(mol_a) for mol_b in prod_b):
                return True
    if len(products) == 2:
        isomorphic = [False, False]
        for i, prod_a in enumerate(prods_a):
            skip = False
            for prod_b in prods_b:
                if skip:
                    break
                for mol_a in prod_a:
                    if any(mol_b.is_isomorphic(mol_a) for mol_b in prod_b):
                        isomorphic[i] = True
                        skip = True
        return all(isomorphic)
    return False


def get_all_families(rmg_family_set: Union[List[str], str] = 'default',
                     consider_rmg_families: bool = True,
                     consider_arc_families: bool = True,
                     ) -> List[str]:
    """
    Get all available RMG and ARC families.
    If ``rmg_family_set`` is a list of family labels and does not contain family sets, it will be returned as is.

    Args:
        rmg_family_set (Union[List[str], str], optional): The RMG family set to use.
        consider_rmg_families (bool, optional): Whether to consider RMG's families.
        consider_arc_families (bool, optional): Whether to consider ARC's custom families.

    Returns:
        List[str]: A list of all available families.
    """
    rmg_family_set = rmg_family_set or 'default'
    family_sets = get_rmg_recommended_family_sets()
    if isinstance(rmg_family_set, list) and all(fam not in family_sets for fam in rmg_family_set):
        return rmg_family_set
    rmg_families, arc_families = list(), list()
    if consider_rmg_families:
        if not isinstance(rmg_families, list) and rmg_family_set not in list(family_sets) + ['all']:
            raise ValueError(f'Invalid RMG family set: {rmg_family_set}')
        if rmg_family_set == 'all':
            for family_set_label, families in family_sets.items():
                if 'surface' not in family_set_label:
                    rmg_families.extend(list(families))
        else:
            rmg_families = list(family_sets[rmg_family_set]) \
                if isinstance(rmg_family_set, str) and rmg_family_set in family_sets else rmg_family_set
    if consider_arc_families:
        arc_families = [os.path.splitext(family)[0] for family in os.listdir(ARC_FAMILIES_PATH)]
    rmg_families = [rmg_families] if isinstance(rmg_families, str) else rmg_families
    arc_families = [arc_families] if isinstance(arc_families, str) else arc_families
    return rmg_families + arc_families if rmg_families is not None else arc_families


def get_rmg_recommended_family_sets() -> Dict[str, str]:
    """
    Get the recommended RMG family sets from RMG-database/input/kinetics/families/recommended.py.

    Returns:
        Dict[str, str]: The recommended RMG family sets.
    """
    family_sets = dict()
    recommended_path = os.path.join(RMG_DB_PATH, 'input', 'kinetics', 'families', 'recommended.py')
    if not os.path.isfile(recommended_path):
        raise FileNotFoundError(f'Could not find the recommended RMG families file at {recommended_path}')
    with open(recommended_path, 'r') as f:
        recommended_content = f.read()
    dict_strings = re.findall(r'(\w+)\s*=\s*\{[^}]*\}', recommended_content, re.DOTALL)
    for dict_name in dict_strings:
        pattern = rf'{dict_name}\s*=\s*(\{{[^}}]*\}})'
        match = re.search(pattern, recommended_content, re.DOTALL)
        if match:
            dict_str = match.group(1)
            family_sets[dict_name] = ast.literal_eval(dict_str)
    return family_sets


def add_labels_to_molecule(mol: 'Molecule',
                           isomorphic_subgraph: dict,
                           ) -> 'Molecule':
    """
    Add atom labels to a molecule based on an isomorphic subgraph.

    Args:
         mol ('Molecule'): The molecule to add labels to.
         isomorphic_subgraph (dict): A dictionary representing the isomorphic subgraph. E.g.::
                                         {<Atom 'C'>: <GroupAtom [*2 'R!H']>,
                                          <Atom 'C.'>: <GroupAtom [*1 'R!H']>,
                                          <Atom 'H'>: <GroupAtom [*3 'H']>
                                        }

    Returns:
         'Molecule': The molecule with atom labels added.
    """
    for atom_index, label in isomorphic_subgraph.items():
        mol.atoms[atom_index].label = label
    return mol


def is_reversible(groups_as_lines: List[str]) -> bool:
    """
    Determine whether the reaction family is reversible.

    Returns:
        bool: Whether the reaction family is reversible.
    """
    for line in groups_as_lines:
        if 'reversible = True' in line:
            return True
        if 'reversible = False' in line:
            return False
    return True


def is_own_reverse(groups_as_lines: List[str]) -> bool:
    """
    Determine whether the reaction family's template also represents its own reverse.

    Returns:
        bool: Whether the reaction family's template also represents its own reverse.
    """
    for line in groups_as_lines:
        if 'ownReverse=True' in line:
            return True
        if 'ownReverse=False' in line:
            return False
    return False


def get_reactant_groups_from_template(groups_as_lines: List[str]) -> List[List[str]]:
    """
    Get the reactant groups from a template content string.
    Descends the entries if a group is defined as an OR complex,
    e.g.: group = "OR{Xtrirad_H, Xbirad_H, Xrad_H, X_H}"

    Args:
        groups_as_lines (List[str]): The template content string.

    Returns:
        List[List[str]]: The non-complex reactant groups.
    """
    reactant_labels = get_initial_reactant_labels_from_template(groups_as_lines)
    result = list()
    for reactant_label in reactant_labels:
        if 'OR{' not in get_group_adjlist(groups_as_lines, entry_label=reactant_label):
            result.append([reactant_label])
        else:
            stack = [reactant_label]
            while any('OR{' in get_group_adjlist(groups_as_lines, entry_label=label) for label in stack):
                label = stack.pop(0)
                group_adjlist = get_group_adjlist(groups_as_lines, entry_label=label)
                if 'OR{' not in group_adjlist:
                    stack.append(label)
                else:
                    stack.extend(descent_complex_group(group_adjlist))
            result.append(stack)
    return result


def get_product_num(groups_as_lines: List[str]) -> int:
    """
    Get the number of products from a template content string.

    Args:
        groups_as_lines (List[str]): The template content string.

    Returns:
        int: The number of products.
    """
    return len(get_initial_reactant_labels_from_template(groups_as_lines, products=True))


def descent_complex_group(group: str) -> List[str]:
    """
    Descend a group if it is defined as an OR complex,
    e.g.: group = "OR{Xtrirad_H, Xbirad_H, Xrad_H, X_H}".

    Args:
        group (str): The group to descend.

    Returns:
        List[str]: The non-complex reactant group labels, e.g.: ['Xtrirad_H', 'Xbirad_H', 'Xrad_H', 'X_H'].
    """
    if group.startswith('OR{') and group.endswith('}'):
        group = group[3:-1].split(', ')
    if isinstance(group, str):
        group = [group]
    return group


def get_initial_reactant_labels_from_template(groups_as_lines: List[str],
                                              products: bool = False,
                                              ) -> List[str]:
    """
    Get the initial reactant labels from a template content string.
    Does not descent the entries if the corresponding group is defined as an OR complex.

    Args:
        groups_as_lines (List[str]): The template content string.
        products (bool, optional): Whether to get the product labels instead of the reactant labels.

    Returns:
        List[str]: The reactant groups.
    """
    labels = list()
    for line in groups_as_lines:
        match = re.search(r'products=\[(.*?)\]', line) if products else re.search(r'reactants=\[(.*?)\]', line)
        if match:
            labels = match.group(1).replace('"', '').split(', ')
            break
    return labels


def get_recipe_actions(groups_as_lines: List[str]) -> List[List[str]]:
    """
    Get the recipe actions from a template content string.

    Args:
        groups_as_lines (List[str]): The template content string.

    Returns:
        List[List[str]]: The recipe actions.
    """
    actions = []
    for i in range(len(groups_as_lines)):
        if 'recipe(actions=[' in groups_as_lines[i]:
            j = 0
            while '])' not in groups_as_lines[i + 1 + j]:
                if "['" in groups_as_lines[i + 1 + j]:
                    actions.append(ast.literal_eval(groups_as_lines[i + 1 + j].strip())[0])
                j += 1
            break
    return actions


def get_entries(groups_as_lines: List[str],
                entry_labels: List[str],
                ) -> Dict[str, str]:
    """
    Get the requested entries grom a template content string.

    Args:
        groups_as_lines (List[str]): The template content string.
        entry_labels (List[str]): The entry labels to extract.

    Returns:
        Dict[str, str]: The extracted entries, keys are the labels, values are the groups.
    """
    groups_str = ''.join(groups_as_lines)
    entries = re.findall(r'entry\((.*?)\)', groups_str, re.DOTALL)
    specific_entries = dict()
    for i, entry in enumerate(entries):
        label_match = re.search(r'label = "(.*?)"', entry)
        group_match = re.search(r'group =(.*?)(?=\w+ =)', entry, re.DOTALL)
        if label_match is not None and group_match is not None and label_match.group(1) in entry_labels:
            specific_entries[label_match.group(1)] = clean_text(group_match.group(1))
        if i > 2000:
            break
    return specific_entries


def get_group_adjlist(groups_as_lines: List[str],
                      entry_label: str,
                      ) -> str:
    """
    Get the corresponding group value for the given entry label.

    Args:
        groups_as_lines (List[str]): The template content string.
        entry_label (str): The entry label to extract.

    Returns:
        str: The extracted group.
    """
    specific_entries = get_entries(groups_as_lines, entry_labels=[entry_label])
    return specific_entries[entry_label]


def get_isomorphic_subgraph(isomorphic_subgraph_1: dict,
                            isomorphic_subgraph_2: dict,
                            mol_1: 'Molecule',
                            mol_2: 'Molecule',
                            ) -> dict:
    """
    Get the isomorphic subgraph from two isomorphic subgraphs and the corresponding molecules.

    Args:
        isomorphic_subgraph_1 (dict): The isomorphic subgraph of the first molecule.
        isomorphic_subgraph_2 (dict): The isomorphic subgraph of the second molecule.
        mol_1 ('Molecule'): The first molecule.
        mol_2 ('Molecule'): The second molecule.

    Returns:
        dict: The isomorphic subgraph which is a dict map of atom indices and group labels.
              E.g.: {0: '*3', 4: '*1', 7: '*2'}
    """
    isomorphic_subgraph = dict()
    len_mol_1 = len(mol_1.atoms)
    for atom, group_atom in isomorphic_subgraph_1.items():
        isomorphic_subgraph[mol_1.atoms.index(atom)] = group_atom.label
    for atom, group_atom in isomorphic_subgraph_2.items():
        isomorphic_subgraph[mol_2.atoms.index(atom) + len_mol_1] = group_atom.label
    return isomorphic_subgraph


def isomorphic_products(rxn: 'ARCReaction',
                        products: List['Molecule'],
                        ) -> bool:
    """
    Check whether the reaction products are isomorphic to the family-generated products.

    Args:
        rxn ('ARCReaction'): The ARC reaction object.
        products (List['Molecule']): The products to check.

    Returns:
        bool: Whether the products are isomorphic to the species.
    """
    p_species = rxn.get_reactants_and_products(return_copies=True)[1]
    return check_product_isomorphism(products, p_species)
