"""
This is the engine part of the atom mapping module.
Here, the edge function for calculation the atom map are located.
Strategy:
    1) The ARCReaction object calls map_reaction
    2) map_reaction make sure that the reaction has family if one can be generated.
    3) If the reaction is supported by RMG, it is sent to the driver. Else, it is mapped with map_general_rxn.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from arc.family import determine_possible_reaction_products_from_family
from arc.mapping.engine import (RESERVED_FINGERPRINT_KEYS,
                                are_adj_elements_in_agreement,
                                create_qc_mol,
                                flip_map,
                                fingerprint,
                                glue_maps,
                                label_species_atoms,
                                make_bond_changes,
                                map_pairs,
                                iterative_dfs, map_two_species,
                                pairing_reactants_and_products_for_mapping,
                                copy_species_list_for_mapping,
                                find_all_breaking_bonds,
                                cut_species_based_on_atom_indices,
                                update_xyz,
                                )
from arc.common import logger
from arc.species.converter import check_molecule_list_order

from arc.exceptions import ActionError, AtomTypeError

if TYPE_CHECKING:
    from arc.molecule.molecule import Molecule
    from arc.reaction import ARCReaction


def map_reaction(rxn: 'ARCReaction',
                 backend: str = 'ARC',
                 flip = False
                 ) -> Optional[List[int]]:
    """
    Map a reaction.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.

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
            _map = map_general_rxn(rxn, backend=backend)
            return _map if _map is not None else map_reaction(rxn, backend=backend, flip=True)
        try:
            _map = map_rxn(rxn, backend=backend)
        except ValueError:
            return map_reaction(rxn, backend=backend, flip=True)
        return _map if _map is not None else map_reaction(rxn, backend=backend, flip=True)


def map_general_rxn(rxn: 'ARCReaction',
                    backend: str = 'ARC',
                    ) -> Optional[List[int]]:
    """
    Map a general reaction (one that was not categorized into a reaction family by RMG).
    The general method isn't great, a family-specific method should be implemented where possible.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.

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
        1) Scissor the reactant(s) and product(s).
        2) Match pair species.
        3) Map_two_species.
        4) Join maps together.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        
    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    reactants, products = rxn.get_reactants_and_products(return_copies=False)
    reactants, products = copy_species_list_for_mapping(reactants), copy_species_list_for_mapping(products)
    label_species_atoms(reactants), label_species_atoms(products)
    
    r_bdes, p_bdes = find_all_breaking_bonds(rxn, True), find_all_breaking_bonds(rxn, False)

    r_cuts = cut_species_based_on_atom_indices(reactants, r_bdes)
    p_cuts = cut_species_based_on_atom_indices(products, p_bdes)

    product_dicts = determine_possible_reaction_products_from_family(rxn, family_label=rxn.family)
    try:
        r_label_dict = product_dicts[0]['r_label_map']
        make_bond_changes(rxn, r_cuts, r_label_dict)
    except (ValueError, IndexError, ActionError, AtomTypeError) as e:
        logger.warning(e)

    r_cuts, p_cuts = update_xyz(r_cuts), update_xyz(p_cuts)

    pairs_of_reactant_and_products = pairing_reactants_and_products_for_mapping(r_cuts, p_cuts)
    if len(p_cuts):
        logger.error(f"Could not find isomorphism for scissored species: {[cut.mol.smiles for cut in p_cuts]}")
        return None
    maps = map_pairs(pairs_of_reactant_and_products)

    return glue_maps(maps, pairs_of_reactant_and_products)


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
            print(f'Could not map {reference_mol_list[0].copy(deep=True).to_smiles()} to {mol_list[0].copy(deep=True).to_smiles()}')
            return None
        return {label: atom_map[index] for label, index in label_dict.items()}
    elif len(reference_mol_list) == 2:
        ordered = check_molecule_list_order(mols_1=reference_mol_list, mols_2=mol_list)
        atom_map_1 = map_two_species(reference_mol_list[0], mol_list[0]) if ordered else map_two_species(reference_mol_list[1], mol_list[0])
        atom_map_2 = map_two_species(reference_mol_list[1], mol_list[1]) if ordered else map_two_species(reference_mol_list[0], mol_list[1])
        if atom_map_1 is None or atom_map_2 is None:
            print(f'Could not map {reference_mol_list[0].copy(deep=True).to_smiles()} to {mol_list[0].copy(deep=True).to_smiles()} '
                  f'or {reference_mol_list[1].copy(deep=True).to_smiles()} to {mol_list[1].copy(deep=True).to_smiles()}')
            return None
        atom_map = atom_map_1 + [index + len(atom_map_1) for index in atom_map_2] if ordered else \
            atom_map_2 + [index + len(atom_map_2) for index in atom_map_1]
        return {label: atom_map[index] for label, index in label_dict.items()}
    return None
