"""
This is the engine part of the atom mapping module.
Here, the edge function for calculation the atom map are located.
Strategy:
    1) The ARCReaction object calls map_reaction
    2) map_reaction make sure that the reaction has family if one can be generated.
    3) If the reaction is supported by RMG, it is sent to the driver. Else, it is mapped with map_general_rxn.
"""

from typing import TYPE_CHECKING, List, Optional

from arc.mapping.engine import (assign_labels_to_products,
                                are_adj_elements_in_agreement,
                                create_qc_mol,
                                flip_map,
                                fingerprint,
                                get_atom_indices_of_labeled_atoms_in_a_reaction,
                                glue_maps,
                                label_species_atoms,
                                make_bond_changes,
                                map_pairs,
                                iterative_dfs, map_two_species,
                                pairing_reactants_and_products_for_mapping,
                                copy_species_list_for_mapping,
                                find_all_bdes,
                                cut_species_based_on_atom_indices,
                                update_xyz,
                                RESERVED_FINGERPRINT_KEYS,
                                )
from arc.common import logger

from rmgpy.exceptions import ActionError, AtomTypeError

if TYPE_CHECKING:
    from arc.reaction import ARCReaction


def map_reaction(rxn: 'ARCReaction',
                 backend: str = 'ARC',
                 flip: bool = False
                 ) -> Optional[List[int]]:
    """
    Map a reaction.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        flip (bool, optional): Whether to attempts fliping the reaction.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    if flip:
        logger.warning(f"The requested ARC reaction {rxn} could not be atom mapped using {backend}. Trying again with the flipped reaction.")
        try:
            _map = flip_map(map_rxn(rxn.flip_reaction(), backend=backend))
        except ValueError:
            return None
        return _map
    else:
        if rxn.family is None:
            logger.warning(f'Could not determine the reaction family for {rxn.label}. '
                           f'Mapping as a general or isomerization reaction.')
            _map = map_general_rxn(rxn)
            return _map if _map is not None else map_reaction(rxn, backend=backend, flip=True)
        try:
            _map = map_rxn(rxn, backend=backend)
        except ValueError:
            return map_reaction(rxn, backend=backend, flip=True)
        return _map if _map is not None else map_reaction(rxn, backend=backend, flip=True)


def map_general_rxn(rxn: 'ARCReaction',
                    ) -> Optional[List[int]]:
    """
    Map a general reaction (one that was not categorized into a reaction family by RMG).
    The general method isn't great, a family-specific method should be implemented where possible.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    if rxn.is_isomerization():
        atom_map = map_isomerization_reaction(rxn=rxn)
        if atom_map is not None:
            return atom_map

    # If the reaction is not a known RMG template and is not isomerization, use fragments via the QCElemental backend.
    qcmol_1 = create_qc_mol(species=[spc.copy() for spc in rxn.r_species],
                            charge=rxn.charge,
                            multiplicity=rxn.multiplicity,
                            )
    qcmol_2 = create_qc_mol(species=[spc.copy() for spc in rxn.p_species],
                            charge=rxn.charge,
                            multiplicity=rxn.multiplicity,
                            )
    if qcmol_1 is None or qcmol_2 is None:
        return None
    data = qcmol_2.align(ref_mol=qcmol_1, verbose=0)[1]
    atom_map = data['mill'].atommap.tolist()
    return atom_map


def map_isomerization_reaction(rxn: 'ARCReaction',
                               ) -> Optional[List[int]]:
    """
    Map isomerization reaction that has no corresponding RMG family.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    # 1. Check if this is a ring opening reaction (only one bond scission occurs).
    if abs(len(rxn.r_species[0].mol.get_all_edges()) - len(rxn.p_species[0].mol.get_all_edges())) == 1:
        # Identify the bond that was removed, and map a modified version of the species (without this bond).
        pairs, success = None, True
        r_fingerprint, p_fingerprint = fingerprint(spc=rxn.r_species[0]), fingerprint(spc=rxn.p_species[0])
        unique_r_keys, unique_p_keys = list(), list()
        for r_key, r_val in r_fingerprint.items():
            for p_val in p_fingerprint.values():
                if are_adj_elements_in_agreement(r_val, p_val):
                    break
            else:
                unique_r_keys.append(r_key)
        for p_key, p_val in p_fingerprint.items():
            for r_key, r_val in r_fingerprint.items():
                if are_adj_elements_in_agreement(r_val, p_val):
                    break
            else:
                unique_p_keys.append(p_key)
        sorted_symbols_r = sorted([r_fingerprint[key]['self'] for key in unique_r_keys])
        sorted_symbols_p = sorted([p_fingerprint[key]['self'] for key in unique_p_keys])
        if len(unique_r_keys) == len(unique_p_keys) == 2 and sorted_symbols_r == sorted_symbols_p:
            if len(set(sorted_symbols_p)) == 2:
                # Only two unique elements, easy to match.
                if r_fingerprint[unique_r_keys[0]]['self'] == p_fingerprint[unique_p_keys[0]]['self']:
                    pairs = [(unique_r_keys[0], unique_p_keys[0]), (unique_r_keys[1], unique_p_keys[1])]
                else:
                    pairs = [(unique_r_keys[0], unique_p_keys[1]), (unique_r_keys[1], unique_p_keys[0])]
            else:
                # Two elements of the same type. Use DFS graph traversal to pair them.
                candidate_0 = iterative_dfs(r_fingerprint,
                                            p_fingerprint,
                                            unique_r_keys[0],
                                            unique_p_keys[0],
                                            allow_first_key_pair_to_disagree=True,
                                            )
                candidate_1 = iterative_dfs(r_fingerprint,
                                            p_fingerprint,
                                            unique_r_keys[0],
                                            unique_p_keys[1],
                                            allow_first_key_pair_to_disagree=True,
                                            )
                if bool(candidate_0 is None) != bool(candidate_1 is None):
                    if candidate_1 is None:
                        pairs = [(unique_r_keys[0], unique_p_keys[0]), (unique_r_keys[1], unique_p_keys[1])]
                    else:
                        pairs = [(unique_r_keys[0], unique_p_keys[1]), (unique_r_keys[1], unique_p_keys[0])]
        if pairs is not None:
            r_copy = rxn.r_species[0].copy()
            p_copy = rxn.p_species[0].copy()
            extra_bond_in_reactant = False
            if sum([len(v) for k, v in r_fingerprint[pairs[0][0]].items() if k not in RESERVED_FINGERPRINT_KEYS]) > \
                    sum([len(v) for k, v in p_fingerprint[pairs[0][1]].items() if k not in RESERVED_FINGERPRINT_KEYS]):
                extra_bond_in_reactant = True
            if extra_bond_in_reactant:
                try:
                    bond = r_copy.mol.get_bond(r_copy.mol.atoms[unique_r_keys[0]], r_copy.mol.atoms[unique_r_keys[1]])
                except ValueError:
                    success = False
                else:
                    r_copy.mol.remove_bond(bond)
                    r_copy.mol.update(log_species=False, raise_atomtype_exception=False, sort_atoms=False)
            else:
                try:
                    bond = p_copy.mol.get_bond(p_copy.mol.atoms[unique_p_keys[0]], p_copy.mol.atoms[unique_p_keys[1]])
                except ValueError:
                    success = False
                else:
                    p_copy.mol.remove_bond(bond)
                    p_copy.mol.update(log_species=False, raise_atomtype_exception=False, sort_atoms=False)
            for atom in r_copy.mol.atoms + p_copy.mol.atoms:
                for bond in atom.bonds.values():
                    bond.order = 1
            if success:
                return map_two_species(r_copy, p_copy, map_type='list')

    # Fallback to the general mapping algorithm.
    return map_two_species(rxn.r_species[0], rxn.p_species[0], map_type='list')


def map_rxn(rxn: 'ARCReaction',
            backend: str = 'ARC',
            ) -> Optional[List[int]]:
    """
    A wrapper function for mapping reaction, uses databases for mapping with the correct reaction family parameters.
    Strategy:
        1) get_rmg_reactions_from_arc_reaction, get_atom_indices_of_labeled_atoms_in_an_rmg_reaction.
        2) (For bimolecular reactions) Find the species in which the bond is broken.
        3) Scissor the reactant(s) and product(s).
        4) Match pair species.
        5) Map_two_species.
        6) Join maps together.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        
    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    r_label_dict, p_label_dict = get_atom_indices_of_labeled_atoms_in_a_reaction(arc_reaction=rxn)

    assign_labels_to_products(rxn=rxn, products=rxn.get_family_products())

    reactants, products = copy_species_list_for_mapping(rxn.r_species), copy_species_list_for_mapping(rxn.p_species)
    label_species_atoms(reactants), label_species_atoms(products)
    
    r_bdes, p_bdes = find_all_bdes(rxn, True), find_all_bdes(rxn, False)

    r_cuts = cut_species_based_on_atom_indices(reactants, r_bdes)
    p_cuts = cut_species_based_on_atom_indices(products, p_bdes)

    try:
        make_bond_changes(rxn, r_cuts, r_label_dict)
    except (ValueError, IndexError, ActionError, AtomTypeError) as e:
        logger.warning(e)

    r_cuts, p_cuts = update_xyz(r_cuts), update_xyz(p_cuts)

    pairs_of_reactant_and_products = pairing_reactants_and_products_for_mapping(r_cuts, p_cuts)
    for p_tup in pairs_of_reactant_and_products:
        print(f'\npairs_of_reactant_and_products: {[s.mol for s in p_tup]}\n\n\n')
    if len(p_cuts):
        logger.error(f"Could not find isomorphism for scissored species: {[cut.mol.smiles for cut in p_cuts]}")
        return None
    maps = map_pairs(pairs_of_reactant_and_products)

    return glue_maps(maps, pairs_of_reactant_and_products)
