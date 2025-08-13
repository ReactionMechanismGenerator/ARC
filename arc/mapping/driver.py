"""
This is the engine part of the atom mapping module.
Here, the edge function for calculation the atom map are located.
Strategy:
    1) The ARCReaction object calls map_reaction
    2) map_reaction make sure that the reaction has family if one can be generated.
    3) If the reaction is supported by RMG, it is sent to the driver. Else, it is mapped with map_general_rxn.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from arc.common import logger
from arc.exceptions import ActionError, AtomTypeError
from arc.mapping.engine import (are_adj_elements_in_agreement,
                                copy_species_list_for_mapping,
                                cut_species_based_on_atom_indices,
                                find_all_breaking_bonds,
                                fingerprint,
                                flip_map,
                                get_template_product_order,
                                glue_maps,
                                iterative_dfs, map_two_species,
                                label_species_atoms,
                                make_bond_changes,
                                map_pairs,
                                pairing_reactants_and_products_for_mapping,
                                reorder_p_label_map,
                                update_xyz,
                                )
from arc.species.converter import check_molecule_list_order

if TYPE_CHECKING:
    from arc.molecule.molecule import Molecule
    from arc.reaction import ARCReaction

MAX_PDI = 25


def map_reaction(rxn: 'ARCReaction',
                 backend: str = 'ARC',
                 flip = False,
                 ) -> Optional[List[int]]:
    """
    Map a reaction.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        backend (str, optional): Currently only supports ``'ARC'``.
        flip (bool, optional): If True, the reaction will be flipped before it is mapped.
                               Useful for reactions that are not atom-mapped correctly in the forward direction.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    def try_mapping(r: 'ARCReaction') -> Optional[List[int]]:
        try:
            return map_rxn(r, backend=backend)
        except ValueError:
            return None
    if flip:
        raw_map = try_mapping(rxn.flip_reaction())
        if raw_map is None:
            return None
        return check_atom_map_and_return(flip_map(raw_map))
    if rxn.family is None:
        logger.warning(f'No family determined for {rxn.label}.\nMapping as a general or isomerization reaction.')
        general_map = map_general_rxn(rxn, backend=backend)
        if general_map is not None:
            return check_atom_map_and_return(general_map)
        return map_reaction(rxn, backend=backend, flip=True)
    raw_map = try_mapping(rxn)
    if raw_map is None:
        return map_reaction(rxn, backend=backend, flip=True)
    return check_atom_map_and_return(raw_map)


def check_atom_map_and_return(atom_map: Optional[List[int]]) -> Optional[List[int]]:
    """
    Check if the atom map is valid and return it.

    Args:
        atom_map (Optional[List[int]]): The atom map to check.

    Returns:
        Optional[List[int]]: The atom map if it is valid, None otherwise.
    """
    if atom_map is None:
        return None
    if not isinstance(atom_map, list):
        logger.error(f'Atom map must be a list. Got type {type(atom_map)}: {atom_map}.')
        return None
    if not all(isinstance(i, int) for i in atom_map):
        logger.error(f'All elements in the atom map must be integers. Got:\n{atom_map}.')
        return None
    if any(i < 0 for i in atom_map):
        logger.error(f'Atom map indices must be non‐negative. Got:\n{atom_map}')
        return None
    if not len(set(atom_map)) == len(atom_map):
        logger.error(f'The atom map should not contain duplicate indices. Got:\n{atom_map}')
        return None
    if set(atom_map) != set(range(len(atom_map))):
        logger.error(f'Atom map must be a permutation of 0..{len(atom_map) - 1}. Got:\n{atom_map}')
        return None
    return atom_map


def map_general_rxn(rxn: 'ARCReaction',
                    backend: str = 'ARC',
                    ) -> Optional[List[int]]:
    """
    Map a general reaction (one that was not categorized into a reaction family by RMG).
    The general method isn't great, a family-specific method should be implemented where possible.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        backend (str, optional): Currently only supports ``'ARC'``.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    atom_map = None
    if rxn.is_isomerization():
        atom_map = map_isomerization_reaction(rxn=rxn)
    return atom_map


def map_isomerization_reaction(rxn: 'ARCReaction') -> Optional[List[int]]:
    """
    Map isomerization reaction that has no corresponding RMG family.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    reactant, product = rxn.r_species[0], rxn.p_species[0]

    # Build edge‐sets for reactant & product
    r_atoms, p_atoms = reactant.mol.atoms, product.mol.atoms
    r_index, p_index = {atom: i for i, atom in enumerate(r_atoms)}, {atom: i for i, atom in enumerate(p_atoms)}
    def edge_set(atoms, index_map):
        """Create a set of edges for the given atoms based on their bonds."""
        edges = set()
        for atom in atoms:
            i_1 = index_map[atom]
            for nbr in atom.bonds:
                i_2 = index_map[nbr]
                edges.add(tuple(sorted((i_1, i_2))))
        return edges

    r_edges, p_edges = edge_set(r_atoms, r_index), edge_set(p_atoms, p_index)
    edge_diff = len(r_edges) - len(p_edges)

    # Only consider the “special” ring‐opening case if exactly one bond changed
    if abs(edge_diff) == 1:
        # 1) Use the fingerprint+DFS “unique‐atom” logic
        pairs, success = None, True
        r_fp, p_fp = fingerprint(spc=reactant), fingerprint(spc=product)
        unique_r_keys, unique_p_keys = list(), list()
        for r_key, r_val in r_fp.items():
            if not any(are_adj_elements_in_agreement(r_val, p_val) for p_val in p_fp.values()):
                unique_r_keys.append(r_key)
        for p_key, p_val in p_fp.items():
            if not any(are_adj_elements_in_agreement(r_val, p_val) for r_val in r_fp.values()):
                unique_p_keys.append(p_key)

        sorted_r, sorted_p = sorted(r_fp[k]['self'] for k in unique_r_keys), sorted(p_fp[k]['self'] for k in unique_p_keys)

        if len(unique_r_keys) == 2 == len(unique_p_keys) and sorted_r == sorted_p:
            # (a) two distinct elements: trivial match
            if len(set(sorted_p)) == 2:
                if r_fp[unique_r_keys[0]]['self'] == p_fp[unique_p_keys[0]]['self']:
                    pairs = [(unique_r_keys[0], unique_p_keys[0]),
                             (unique_r_keys[1], unique_p_keys[1])]
                else:
                    pairs = [(unique_r_keys[0], unique_p_keys[1]),
                             (unique_r_keys[1], unique_p_keys[0])]
            # (b) same element twice: try both DFS seeds
            else:
                c0 = iterative_dfs(r_fp, p_fp, unique_r_keys[0], unique_p_keys[0], allow_first_key_pair_to_disagree=True)
                c1 = iterative_dfs(r_fp, p_fp, unique_r_keys[0], unique_p_keys[1], allow_first_key_pair_to_disagree=True)
                if (c0 is None) ^ (c1 is None):
                    if c1 is None:
                        pairs = [(unique_r_keys[0], unique_p_keys[0]), (unique_r_keys[1], unique_p_keys[1])]
                    else:
                        pairs = [(unique_r_keys[0], unique_p_keys[1]), (unique_r_keys[1], unique_p_keys[0])]

        # 2) If fingerprint+DFS succeeded, do exactly the old removal & update
        if pairs is not None:
            # decide which side had the extra bond
            extra_in_reactant = edge_diff == 1
            r_copy, p_copy = reactant.copy(), product.copy()
            try:
                if extra_in_reactant:
                    b = r_copy.mol.get_bond(r_copy.mol.atoms[pairs[0][0]], r_copy.mol.atoms[pairs[1][0]])
                    r_copy.mol.remove_bond(b)
                    r_copy.mol.update(log_species=False, raise_atomtype_exception=False, sort_atoms=False)
                else:
                    b = p_copy.mol.get_bond(p_copy.mol.atoms[pairs[0][1]], p_copy.mol.atoms[pairs[1][1]])
                    p_copy.mol.remove_bond(b)
                    p_copy.mol.update(log_species=False, raise_atomtype_exception=False, sort_atoms=False)
            except ValueError:
                success = False

            if success:
                for atom in r_copy.mol.atoms + p_copy.mol.atoms:
                    for bd in atom.bonds.values():
                        bd.order = 1
                return map_two_species(r_copy, p_copy, map_type='list')

        # 3) If fingerprint branch *didn’t* yield pairs, still do a deterministic edge‐removal mapping for *any* ring opening
        removed = (r_edges - p_edges) if edge_diff == 1 else (p_edges - r_edges)
        if len(removed) == 1:
            i, j = removed.pop()
            r_copy, p_copy = reactant.copy(), product.copy()
            try:
                if edge_diff == 1:
                    bond = r_copy.mol.get_bond(r_copy.mol.atoms[i], r_copy.mol.atoms[j])
                    r_copy.mol.remove_bond(bond)
                    r_copy.mol.update(log_species=False, raise_atomtype_exception=False, sort_atoms=False)
                else:
                    bond = p_copy.mol.get_bond(p_copy.mol.atoms[i], p_copy.mol.atoms[j])
                    p_copy.mol.remove_bond(bond)
                    p_copy.mol.update(log_species=False, raise_atomtype_exception=False, sort_atoms=False)
            except ValueError:
                pass
            else:
                for atom in r_copy.mol.atoms + p_copy.mol.atoms:
                    for bd in atom.bonds.values():
                        bd.order = 1
                return map_two_species(r_copy, p_copy, map_type='list')

    # 4) Fallback to the general mapping
    return map_two_species(reactant, product, map_type='list')



def map_rxn(rxn: 'ARCReaction',
            backend: str = 'ARC',
            product_dict_index_to_try: int = 0,
            ) -> Optional[List[int]]:
    """
    A wrapper function for mapping reaction, uses databases for mapping with the correct reaction family parameters.
    Strategy:
        1) Scissor the reactant(s) and product(s).
        2) Match pair species.
        3) Map_two_species.
        4) Join maps together.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        backend (str, optional): Currently only supports ``'ARC'``. Currently not used, only one backend is implemented.
        product_dict_index_to_try (int, optional): The index of the reaction family product dictionary to try.
                                                   Defaults to 0, which is the first product dictionary.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    pdi = product_dict_index_to_try
    reactants, products = rxn.get_reactants_and_products(return_copies=False)
    reactants, products = copy_species_list_for_mapping(reactants), copy_species_list_for_mapping(products)
    label_species_atoms(reactants), label_species_atoms(products)

    r_bdes, p_bdes = find_all_breaking_bonds(rxn, r_direction=True, pdi=pdi), find_all_breaking_bonds(rxn, r_direction=False, pdi=pdi)
    r_cuts, p_cuts = cut_species_based_on_atom_indices(reactants, r_bdes), cut_species_based_on_atom_indices(products, p_bdes)

    try:
        r_label_map = rxn.product_dicts[pdi]['r_label_map']
        p_label_map = rxn.product_dicts[pdi]['p_label_map']
        template_products = rxn.product_dicts[pdi]['products']
    except (IndexError, KeyError) as e:
        logger.error(f"No valid template maps for reaction {rxn} ({rxn.family}), cannot atom map. Got:\n{e}")
        return None
    try:
        template_order = get_template_product_order(rxn, template_products)
    except ValueError:
        if rxn.product_dicts is not None and len(rxn.product_dicts) - 1 > pdi < MAX_PDI:
            return map_rxn(rxn, backend=backend, product_dict_index_to_try=pdi + 1)
        else:
            logger.error(f'No valid template order for reaction {rxn} ({rxn.family}), cannot atom map.')
            return None

    updated_p_label_map = reorder_p_label_map(p_label_map=p_label_map,
                                              template_order=template_order,
                                              template_products=template_products,
                                              actual_products=rxn.get_reactants_and_products()[1])
    try:
        make_bond_changes(rxn, r_cuts, r_label_map)
    except (ValueError, IndexError, ActionError, AtomTypeError) as e:
        logger.warning(e)
    r_cuts, p_cuts = update_xyz(r_cuts), update_xyz(p_cuts)
    pairs = pairing_reactants_and_products_for_mapping(r_cuts, p_cuts)
    if p_cuts:
        logger.error(f'Could not find isomorphism for scissored species: {[cut.mol.smiles for cut in p_cuts]}')
        return None

    fragment_maps = map_pairs(pairs)
    total_atoms = sum(len(sp.mol.atoms) for sp in reactants)
    atom_map = glue_maps(maps=fragment_maps,
                         pairs=pairs,
                         r_label_map=r_label_map,
                         p_label_map=updated_p_label_map,
                         total_atoms=total_atoms,
                         )
    if atom_map is None and rxn.product_dicts is not None and len(rxn.product_dicts) - 1 > pdi < MAX_PDI:
        return map_rxn(rxn, backend=backend, product_dict_index_to_try=pdi + 1)
    return atom_map


def convert_label_dict(label_dict: Dict[str, int],
                       reference_mol_list: List['Molecule'],
                       mol_list: List['Molecule'],
                       ) -> Optional[Dict[str, int]]:
    """
    Convert the label dictionary to the correct atom indices in the reaction and reference molecules

    Args:
        label_dict (Dict[str, int]): A dictionary of atom labels (e.g., '*1') to atom indices.
        reference_mol_list (List[Molecule]): The list of molecules to which label_dict values refer.
        mol_list (List[Molecule]): The list of molecules to which label_dict values should be converted.

    Returns:
        Dict[str, int]: The converted label dictionary.
    """
    if len(reference_mol_list) != len(mol_list):
        raise ValueError(f'The number of reference molecules ({len(reference_mol_list)}) '
                         f'does not match the number of molecules ({len(mol_list)}).')
    if len(reference_mol_list) == 1:
        atom_map = map_two_species(reference_mol_list[0], mol_list[0])
        if atom_map is None:
            return None
        return {label: atom_map[index] for label, index in label_dict.items()}
    elif len(reference_mol_list) == 2:
        ordered = check_molecule_list_order(mols_1=reference_mol_list, mols_2=mol_list)
        atom_map_1 = map_two_species(reference_mol_list[0], mol_list[0]) if ordered else map_two_species(reference_mol_list[1], mol_list[0])
        atom_map_2 = map_two_species(reference_mol_list[1], mol_list[1]) if ordered else map_two_species(reference_mol_list[0], mol_list[1])
        if atom_map_1 is None or atom_map_2 is None:
            return None
        atom_map = atom_map_1 + [index + len(atom_map_1) for index in atom_map_2] if ordered else \
            atom_map_2 + [index + len(atom_map_2) for index in atom_map_1]
        return {label: atom_map[index] for label, index in label_dict.items()}
    return None
