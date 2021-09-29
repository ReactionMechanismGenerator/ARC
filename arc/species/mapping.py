"""
A module for atom-mapping a species or a set of species.

Species atom-map logic:
1. Determine adjacent elements for each heavy atom
2. Identify and loop superposition possibilities
3. Recursively modify dihedrals until the structures overlap to some tolerance
4. Determine RMSD to backbone, if good then determine RMSD to H's
5. When mapping H's on terminal heavy atoms, check whether rotating this rotor will reduce the overall RMSD
   if there's more than one H on that terminal atom
"""

from collections import deque
from itertools import product
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from qcelemental.exceptions import ValidationError
from qcelemental.models.molecule import Molecule as QCMolecule

from rmgpy.molecule import Molecule
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.common import convert_list_index_0_to_1, extremum_list, logger
from arc.exceptions import SpeciesError
from arc.species import ARCSpecies
from arc.species.conformers import determine_chirality
from arc.species.converter import compare_confs, sort_xyz_using_indices, translate_xyz, xyz_from_data, xyz_to_str
from arc.species.vectors import calculate_angle, calculate_dihedral_angle, calculate_distance, get_delta_angle


if TYPE_CHECKING:
    from rmgpy.data.kinetics.family import TemplateReaction
    from rmgpy.data.rmg import RMGDatabase
    from rmgpy.molecule.molecule import Atom
    from rmgpy.reaction import Reaction
    from arc.reaction import ARCReaction


RESERVED_FINGERPRINT_KEYS = ['self', 'chirality', 'label']


def map_reaction(rxn: 'ARCReaction',
                 backend: str = 'ARC',
                 db: Optional['RMGDatabase'] = None,
                 ) -> Optional[List[int]]:
    """
    Map a reaction.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    if rxn.family is None:
        rmgdb.determine_family(reaction=rxn, db=db)
    if rxn.family is None:
        return map_general_rxn(rxn, backend=backend)

    fam_func_dict = {'Disproportionation': map_h_abstraction,  # Mapping a disprop reaction is similar to H_Abs.
                     'H_Abstraction': map_h_abstraction,
                     'HO2_Elimination_from_PeroxyRadical': map_ho2_elimination_from_peroxy_radical,
                     'intra_H_migration': map_intra_h_migration,
                     }

    if rxn.family.label not in fam_func_dict.keys():
        logger.info(f'Using a generic mapping algorithm for {rxn} of family {rxn.family.label}')

    map_func = fam_func_dict.get(rxn.family.label, map_general_rxn)

    return map_func(rxn, backend=backend, db=db)


def map_general_rxn(rxn: 'ARCReaction',
                    backend: str = 'ARC',
                    db: Optional['RMGDatabase'] = None,
                    ) -> Optional[List[int]]:
    """
    Map a general reaction (one that was not categorized into a reaction family by RMG).
    The general method isn't great, a family-specific method should be implemented where possible.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    if rxn.is_isomerization():
        for backend in ['ARC', 'QCElemental']:
            atom_map = map_isomerization_reaction(rxn, backend, db)
            if atom_map is not None:
                break
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
                               backend: str = 'ARC',
                               db: Optional['RMGDatabase'] = None,
                               ) -> Optional[List[int]]:
    """
    Map isomerization reaction that has no corresponding RMG family.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        db (RMGDatabase, optional): The RMG database instance.

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
            for p_key, p_val in p_fingerprint.items():
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
                return map_two_species(r_copy, p_copy, map_type='list', backend=backend)

    # Fallback to the general mapping algorithm.
    return map_two_species(rxn.r_species[0], rxn.p_species[0], map_type='list', backend=backend)


# Family-specific mapping functions:


def map_h_abstraction(rxn: 'ARCReaction',
                      backend: str = 'ARC',
                      db: Optional['RMGDatabase'] = None,
                      ) -> Optional[List[int]]:
    """
    Map a hydrogen abstraction or a disproportionation reaction (they are principally similar for mapping).

    Strategy:
        H_Abstraction: Map species R(*1)-H(*2) to species R(*1)j and map species R(*3)j to species R(*3)-H(*2).
        Disprop: Map species R(*1)j to species R(*1)-H(*4) and map species R(*3)j-R(*2)-H(*4) to species R(*3)=R(*2).
        Use scissors to map the backbone.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    disprop = False
    if not check_family_for_mapping_function(rxn=rxn, db=db, family='H_Abstraction'):
        if check_family_for_mapping_function(rxn=rxn, db=db, family='Disproportionation'):
            disprop = True
        else:
            return None

    atom_labels = ('*1', '*2', '*3') if not disprop else ('*2', '*4', '*1')  # (H, H-anchor, attacking rad)

    rmg_reactions = get_rmg_reactions_from_arc_reaction(arc_reaction=rxn, backend=backend)
    r_label_dict, p_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                                      rmg_reaction=rmg_reactions[0])
    r_h_index = r_label_dict[atom_labels[1]]
    p_h_index = p_label_dict[atom_labels[1]]
    len_r1, len_p1 = rxn.r_species[0].number_of_atoms, rxn.p_species[0].number_of_atoms
    r1_h2 = 0 if r_h_index < len_r1 else 1  # Identify R(*1)-H(*2), it's either reactant 0 or reactant 1.
    r3 = 1 - r1_h2  # Identify R(*3) in the reactants.
    r3_h2 = 0 if p_h_index < len_p1 else 1  # Identify R(*3)-H(*2), it's either product 0 or product 1.
    r1 = 1 - r3_h2  # Identify R(*1) in the products.

    spc_r1_h2 = ARCSpecies(label='R1-H2',
                           mol=rxn.r_species[r1_h2].mol.copy(deep=True),
                           xyz=rxn.r_species[r1_h2].get_xyz(),
                           bdes=[(r_label_dict[atom_labels[0]] + 1 - r1_h2 * len_r1,
                                  r_label_dict[atom_labels[1]] + 1 - r1_h2 * len_r1)],  # Mark the R(*1)-H(*2) bond for scission.
                           )
    spc_r1_h2.final_xyz = spc_r1_h2.get_xyz()  # Scissors requires the .final_xyz attribute to be populated.
    try:
        spc_r1_h2_cuts = spc_r1_h2.scissors()
    except SpeciesError:
        return None
    spc_r1_h2_cut = [spc for spc in spc_r1_h2_cuts if spc.label != 'H'][0] \
        if any(spc.label != 'H' for spc in spc_r1_h2_cuts) else spc_r1_h2_cuts[0]  # Treat H2 as well :)
    spc_r3_h2 = ARCSpecies(label='R3-H2',
                           mol=rxn.p_species[r3_h2].mol.copy(deep=True),
                           xyz=rxn.p_species[r3_h2].get_xyz(),
                           bdes=[(p_label_dict[atom_labels[2]] + 1 - r3_h2 * len_p1,
                                  p_label_dict[atom_labels[1]] + 1 - r3_h2 * len_p1)],  # Mark the R(*3)-H(*2) bond for scission.
                           )
    spc_r3_h2.final_xyz = spc_r3_h2.get_xyz()
    try:
        spc_r3_h2_cuts = spc_r3_h2.scissors()
    except SpeciesError:
        return None
    spc_r3_h2_cut = [spc for spc in spc_r3_h2_cuts if spc.label != 'H'][0] \
        if any(spc.label != 'H' for spc in spc_r3_h2_cuts) else spc_r3_h2_cuts[0]  # Treat H2 as well :)
    map_1 = map_two_species(spc_r1_h2_cut, rxn.p_species[r1], backend=backend)
    map_2 = map_two_species(rxn.r_species[r3], spc_r3_h2_cut, backend=backend)

    result = {r_h_index: p_h_index}
    for r_increment, p_increment, map_i, j in zip([r1_h2 * len_r1, r3 * len_r1],
                                                  [r1 * len_p1, r3_h2 * len_p1],
                                                  [map_1, map_2],
                                                  [0, 1]):
        for i, entry in enumerate(map_i):
            r_index = i + r_increment + (1 - j) * int(i + r_increment >= r_h_index)
            p_index = entry + p_increment + j * int(i + p_increment >= p_h_index)
            result[r_index] = p_index
    return [val for key, val in sorted(result.items(), key=lambda item: item[0])]


def map_ho2_elimination_from_peroxy_radical(rxn: 'ARCReaction',
                                            backend: str = 'ARC',
                                            db: Optional['RMGDatabase'] = None,
                                            ) -> Optional[List[int]]:
    """
    Map an HO2 elimination from peroxy radical reaction.
    Strategy: Remove the O(*3), O(*4), and H(*5) atoms from the reactant and map to the R(*1)=R(*2) product.
    Note that two consecutive scissions must be performed.

    Todo: This function only deals with a single direction of HO2 elimination, consider the reverse direction as well and test.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    if not check_family_for_mapping_function(rxn=rxn, db=db, family='HO2_Elimination_from_PeroxyRadical'):
        return None

    rmg_reactions = get_rmg_reactions_from_arc_reaction(arc_reaction=rxn, backend=backend)
    r_label_dict, p_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                                      rmg_reaction=rmg_reactions[0])

    if len(rxn.r_species) == 1 and len(rxn.p_species) == 2:
        r_o3_index = r_label_dict['*3']
        r_o4_index = r_label_dict['*4']
        r_h5_index = r_label_dict['*5']
        len_p1 = rxn.p_species[0].number_of_atoms
        r1dr2 = 0 if p_label_dict['*1'] < len_p1 else 1  # Identify R(*1)=R(*2), it's either product 0 or product 1.

        mol_r_mod = rxn.r_species[0].mol.copy(deep=True)
        xyz_r_mod = rxn.r_species[0].get_xyz()
        vertex_indices = sorted([r_o3_index, r_o4_index, r_h5_index], reverse=True)
        for vertex_index in vertex_indices:
            mol_r_mod.vertices.remove(mol_r_mod.vertices[vertex_index])
        xyz_r_mod['symbols'] = tuple(symbol for i, symbol in enumerate(xyz_r_mod['symbols']) if i not in vertex_indices)
        xyz_r_mod['isotopes'] = tuple(isotope for i, isotope in enumerate(xyz_r_mod['isotopes']) if i not in vertex_indices)
        xyz_r_mod['coords'] = tuple(coord for i, coord in enumerate(xyz_r_mod['coords']) if i not in vertex_indices)
        spc_r_mod = ARCSpecies(label='R', mol=mol_r_mod, xyz=xyz_r_mod)
        spc_r_mod.final_xyz = xyz_r_mod  # .set_dihedral() requires the .final_xyz attribute.

        # Different dihedral angles in the reactant and product will make mapping H atoms hard.
        # Fix dihedrals between 4 heavy atom sequences.
        spc_r_mod.determine_rotors()
        map_1 = map_two_species(spc_r_mod, rxn.p_species[r1dr2], backend=backend)
        if spc_r_mod.rotors_dict is not None:
            for rotor in spc_r_mod.rotors_dict.values():
                torsion = rotor['torsion']
                if spc_r_mod.mol.atoms[torsion[0]].is_non_hydrogen() and spc_r_mod.mol.atoms[torsion[1]].is_non_hydrogen():
                    spc_r_mod.set_dihedral(scan=convert_list_index_0_to_1(torsion),
                                           deg_abs=calculate_dihedral_angle(coords=rxn.p_species[r1dr2].get_xyz(),
                                                                            torsion=[map_1[t] for t in torsion]),
                                           chk_rotor_list=False)
                    spc_r_mod.final_xyz = spc_r_mod.initial_xyz
        map_2 = map_two_species(spc_r_mod, rxn.p_species[r1dr2], backend=backend)
        new_map, added_ho2_atoms = list(), list()
        star_map = {r_o3_index: '*3', r_o4_index: '*4', r_h5_index: '*5'}
        for i, entry in enumerate(map_2):
            for j in [i, i + 1, i + 2]:
                # Check three consecutive indices, we don't know whether the HO2 atoms are mapped consecutively or not.
                if j in star_map.keys() and j not in added_ho2_atoms:
                    new_map.append(p_label_dict[star_map[j]])
                    added_ho2_atoms.append(j)
                else:
                    break
            new_map.append(entry)
        return new_map


def map_intra_h_migration(rxn: 'ARCReaction',
                          backend: str = 'ARC',
                          db: Optional['RMGDatabase'] = None,
                          ) -> Optional[List[int]]:
    """
    Map an intra hydrogen migration reaction.
    Strategy: Remove the *3 H atom from both the reactant and product to have the same backbone.
    Map the backbone and add the (known) *3 H atom.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    if not check_family_for_mapping_function(rxn=rxn, db=db, family='intra_H_migration'):
        return None

    rmg_reactions = get_rmg_reactions_from_arc_reaction(arc_reaction=rxn, backend=backend)
    r_label_dict, p_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                                      rmg_reaction=rmg_reactions[0])

    r_h_index = r_label_dict['*3']
    p_h_index = p_label_dict['*3']

    spc_r = ARCSpecies(label='R',
                       mol=rxn.r_species[0].mol.copy(deep=True),
                       xyz=rxn.r_species[0].get_xyz(),
                       bdes=[(r_label_dict['*2'] + 1, r_label_dict['*3'] + 1)],  # Mark the R(*2)-H(*3) bond for scission.
                       )
    spc_r.final_xyz = spc_r.get_xyz()  # Scissors requires the .final_xyz attribute to be populated.
    try:
        spc_r_dot = [spc for spc in spc_r.scissors() if spc.label != 'H'][0]
    except SpeciesError:
        return None
    spc_p = ARCSpecies(label='P',
                       mol=rxn.p_species[0].mol.copy(deep=True),
                       xyz=rxn.p_species[0].get_xyz(),
                       bdes=[(p_label_dict['*1'] + 1, p_label_dict['*3'] + 1)],  # Mark the R(*1)-H(*3) bond for scission.
                       )
    spc_p.final_xyz = spc_p.get_xyz()
    try:
        spc_p_dot = [spc for spc in spc_p.scissors() if spc.label != 'H'][0]
    except SpeciesError:
        return None
    map_ = map_two_species(spc_r_dot, spc_p_dot, backend=backend)

    new_map = list()
    for i, entry in enumerate(map_):
        if i == r_h_index:
            new_map.append(p_h_index)
        new_map.append(entry if entry < p_h_index else entry + 1)
    return new_map


# Mapping functions:


def check_family_for_mapping_function(rxn: 'ARCReaction',
                                      family: str,
                                      db: Optional['RMGDatabase'] = None,
                                      ) -> bool:
    """
    Check that the actual reaction family and the desired reaction family are the same.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        family (str): The desired reaction family to check for.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        bool: Whether the reaction family and the desired ``family`` are consistent.
    """
    if rxn.family is None:
        rmgdb.determine_family(reaction=rxn, db=db)
    if rxn.family is None or rxn.family.label != family:
        return False
    return True


def get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction: 'ARCReaction',
                                                         rmg_reaction: 'TemplateReaction',
                                                         ) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """
    Get the RMG reaction atom labels and the corresponding 0-indexed atom indices
    for all labeled atoms in a TemplateReaction.

    Args:
        arc_reaction (ARCReaction): An ARCReaction object instance.
        rmg_reaction (TemplateReaction): A respective RMG family TemplateReaction object instance.

    Returns:
        Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
            The tuple entries relate to reactants and products.
            Keys are labels (e.g., '*1'), values are corresponding 0-indices atoms.
    """
    if not hasattr(rmg_reaction, 'labeled_atoms') or not rmg_reaction.labeled_atoms:
        return None, None

    for mol in rmg_reaction.reactants + rmg_reaction.products:
        mol.generate_resonance_structures(save_order=True)

    r_map, p_map = map_arc_rmg_species(arc_reaction=arc_reaction, rmg_reaction=rmg_reaction)

    reactant_index_dict, product_index_dict = dict(), dict()
    reactant_atoms, product_atoms = list(), list()
    rmg_reactant_order = [val[0] for key, val in sorted(r_map.items(), key=lambda item: item[0])]
    rmg_product_order = [val[0] for key, val in sorted(p_map.items(), key=lambda item: item[0])]
    for i in rmg_reactant_order:
        reactant_atoms.extend([atom for atom in rmg_reaction.reactants[i].atoms])
    for i in rmg_product_order:
        product_atoms.extend([atom for atom in rmg_reaction.products[i].atoms])

    for labeled_atom_dict, atom_list, index_dict in zip([rmg_reaction.labeled_atoms['reactants'],
                                                         rmg_reaction.labeled_atoms['products']],
                                                        [reactant_atoms, product_atoms],
                                                        [reactant_index_dict, product_index_dict]):
        for label, atom_1 in labeled_atom_dict.items():
            for i, atom_2 in enumerate(atom_list):
                if atom_1.id == atom_2.id:
                    index_dict[label] = i
                    break
    return reactant_index_dict, product_index_dict


def map_arc_rmg_species(arc_reaction: 'ARCReaction',
                        rmg_reaction: Union['Reaction', 'TemplateReaction'],
                        concatenate: bool = True,
                        ) -> Tuple[Dict[int, Union[List[int], int]], Dict[int, Union[List[int], int]]]:
    """
    Map the species pairs in an ARC reaction to those in a respective RMG reaction
    which is defined in the same direction.

    Args:
        arc_reaction (ARCReaction): An ARCReaction object instance.
        rmg_reaction (Union[Reaction, TemplateReaction]): A respective RMG family TemplateReaction object instance.
        concatenate (bool, optional): Whether to return isomorphic species as a single list (``True``, default),
                                      or to return isomorphic species separately (``False``).

    Returns:
        Tuple[Dict[int, Union[List[int], int]], Dict[int, Union[List[int], int]]]:
            The first tuple entry refers to reactants, the second to products.
            Keys are specie indices in the ARC reaction,
            values are respective indices in the RMG reaction.
            If ``concatenate`` is ``True``, values are lists of integers. Otherwise, values are integers.
    """
    if rmg_reaction.is_isomerization():
        if concatenate:
            return {0: [0]}, {0: [0]}
        else:
            return {0: 0}, {0: 0}
    r_map, p_map = dict(), dict()
    arc_reactants, arc_products = arc_reaction.get_reactants_and_products(arc=True)
    for spc_map, rmg_species, arc_species in [(r_map, rmg_reaction.reactants, arc_reactants),
                                              (p_map, rmg_reaction.products, arc_products)]:
        for i, arc_spc in enumerate(arc_species):
            for j, rmg_obj in enumerate(rmg_species):
                rmg_spc = Species(molecule=[rmg_obj]) if isinstance(rmg_obj, Molecule) else rmg_obj
                if not isinstance(rmg_spc, Species):
                    raise ValueError(f'Expected an RMG object instances of Molecule or Species, '
                                     f'got {rmg_obj} which is a {type(rmg_obj)}.')
                rmg_spc.generate_resonance_structures(save_order=True)
                rmg_spc_based_on_arc_spc = Species(molecule=arc_spc.mol_list)
                rmg_spc_based_on_arc_spc.generate_resonance_structures(save_order=True)
                if rmg_spc.is_isomorphic(rmg_spc_based_on_arc_spc, save_order=True):
                    if i in spc_map.keys() and concatenate:
                        spc_map[i].append(j)
                    elif concatenate:
                        spc_map[i] = [j]
                    else:
                        spc_map[i] = j
                        break
    if not r_map or not p_map:
        raise ValueError(f'Could not match some of the RMG Reaction {rmg_reaction} to the ARC Reaction {arc_reaction}.')
    return r_map, p_map


def find_equivalent_atoms_in_reactants(arc_reaction: 'ARCReaction',
                                       backend: str = 'ARC',
                                       ) -> Optional[List[List[int]]]:
    """
    Find atom indices that are equivalent in the reactants of an ARCReaction
    in the sense that they represent degenerate reaction sites that are indifferentiable in 2D.
    Bridges between RMG reaction templates and ARC's 3D TS structures.
    Running indices in the returned structure relate to reactant_0 + reactant_1 + ...

    Args:
        arc_reaction ('ARCReaction'): The ARCReaction object instance.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.

    Returns:
        Optional[List[List[int]]]: Entries are lists of 0-indices, each such list represents equivalent atoms.
    """
    rmg_reactions = get_rmg_reactions_from_arc_reaction(arc_reaction, backend=backend)
    dicts = [get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(rmg_reaction=rmg_reaction,
                                                                  arc_reaction=arc_reaction)[0]
             for rmg_reaction in rmg_reactions]
    equivalence_map = dict()
    for index_dict in dicts:
        for key, value in index_dict.items():
            if key in equivalence_map:
                equivalence_map[key].append(value)
            else:
                equivalence_map[key] = [value]
    equivalent_indices = list(list(set(equivalent_list)) for equivalent_list in equivalence_map.values())
    return equivalent_indices


def get_rmg_reactions_from_arc_reaction(arc_reaction: 'ARCReaction',
                                        backend: str = 'ARC',
                                        db: Optional['RMGDatabase'] = None,
                                        ) -> Optional[List['TemplateReaction']]:
    """
    A helper function for getting RMG reactions from an ARC reaction.
    This function calls ``map_two_species()`` so that each species in the RMG reaction is correctly mapped
    to the corresponding species in the ARC reaction. It does not attempt to map reactants to products.

    Args:
        arc_reaction (ARCReaction): The ARCReaction object instance.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[TemplateReaction]]:
            The respective RMG TemplateReaction object instances (considering resonance structures).
    """
    if arc_reaction.family is None:
        rmgdb.determine_family(reaction=arc_reaction, db=db)
    if arc_reaction.family is None:
        return None
    if not arc_reaction.family.save_order:
        raise ValueError('Must have the save_order attribute of the family set to True.')
    rmg_reactions = arc_reaction.family.generate_reactions(reactants=[spc.mol for spc in arc_reaction.r_species],
                                                           products=[spc.mol for spc in arc_reaction.p_species],
                                                           prod_resonance=True,
                                                           delete_labels=False,
                                                           relabel_atoms=False,
                                                           )
    for rmg_reaction in rmg_reactions:
        r_map, p_map = map_arc_rmg_species(arc_reaction=arc_reaction, rmg_reaction=rmg_reaction, concatenate=False)
        ordered_rmg_reactants = [rmg_reaction.reactants[r_map[i]] for i in range(len(rmg_reaction.reactants))]
        ordered_rmg_products = [rmg_reaction.products[p_map[i]] for i in range(len(rmg_reaction.products))]
        mapped_rmg_reactants, mapped_rmg_products = list(), list()
        for ordered_rmg_mols, arc_species, mapped_mols in zip([ordered_rmg_reactants, ordered_rmg_products],
                                                              [arc_reaction.r_species, arc_reaction.p_species],
                                                              [mapped_rmg_reactants, mapped_rmg_products],
                                                              ):
            for rmg_mol, arc_spc in zip(ordered_rmg_mols, arc_species):
                mol = arc_spc.copy().mol
                # The RMG molecule will get a random 3D conformer, don't consider chirality when mapping.
                atom_map = map_two_species(mol, rmg_mol, map_type='dict', backend=backend, consider_chirality=False)
                new_atoms_list = list()
                for i in range(len(rmg_mol.atoms)):
                    rmg_mol.atoms[atom_map[i]].id = mol.atoms[i].id
                    new_atoms_list.append(rmg_mol.atoms[atom_map[i]])
                rmg_mol.atoms = new_atoms_list
                mapped_mols.append(rmg_mol)
        rmg_reaction.reactants, rmg_reaction.products = mapped_rmg_reactants, mapped_rmg_products
    return rmg_reactions


def map_two_species(spc_1: Union[ARCSpecies, Species, Molecule],
                    spc_2: Union[ARCSpecies, Species, Molecule],
                    map_type: str = 'list',
                    backend: str = 'ARC',
                    consider_chirality: bool = True,
                    allow_backend_shift: bool = True,
                    verbose: bool = False,
                    ) -> Optional[Union[List[int], Dict[int, int]]]:
    """
    Map the atoms in spc1 to the atoms in spc2.
    All indices are 0-indexed.
    If a dict type atom map is returned, it cold conveniently be used to map ``spc_2`` -> ``spc_1`` by doing::

        ordered_spc1.atoms = [spc_2.atoms[atom_map[i]] for i in range(len(spc_2.atoms))]

    Args:
        spc_1 (Union[ARCSpecies, Species, Molecule]): Species 1.
        spc_2 (Union[ARCSpecies, Species, Molecule]): Species 2.
        map_type (str, optional): Whether to return a 'list' or a 'dict' map type.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        allow_backend_shift (bool, optional): Whether to try QCElemental's method if ARC'd method cannot identify candidates.
        consider_chirality (bool, optional): Whether to consider chirality when fingerprinting.
        verbose (bool, optional): Whether to use logging.

    Returns:
        Optional[Union[List[int], Dict[int, int]]]:
            The atom map. By default, a list is returned.
            If the map is of ``list`` type, entry indices are atom indices of ``spc_1``, entry values are atom indices of ``spc_2``.
            If the map is of ``dict`` type, keys are atom indices of ``spc_1``, values are atom indices of ``spc_2``.
    """
    spc_1, spc_2 = get_arc_species(spc_1), get_arc_species(spc_2)
    if not check_species_before_mapping(spc_1, spc_2, verbose=verbose) and backend != 'QCElemental':
        if verbose:
            logger.warning(f'Could not map species {spc_1} and {spc_2}.')
        return None

    # A shortcut for mono-atomic species.
    if spc_1.number_of_atoms == spc_2.number_of_atoms == 1:
        if map_type == 'dict':
            return {0: 0}
        return [0]

    # A shortcut for homonuclear diatomic species.
    if spc_1.number_of_atoms == spc_2.number_of_atoms == 2 \
            and len(set([atom.element.symbol for atom in spc_1.mol.atoms])) == 1:
        if map_type == 'dict':
            return {0: 0, 1: 1}
        return [0, 1]

    # A shortcut for species with all different elements:
    if len(set([atom.element.symbol for atom in spc_1.mol.atoms])) == spc_1.number_of_atoms:
        atom_map = {}
        for i, atom_1 in enumerate(spc_1.mol.atoms):
            for j, atom_2 in enumerate(spc_2.mol.atoms):
                if atom_1.element.symbol == atom_2.element.symbol:
                    atom_map[i] = j
                    break
        if map_type == 'list':
            atom_map = [v for k, v in sorted(atom_map.items(), key=lambda item: item[0])]
        return atom_map

    if backend.lower() not in ['qcelemental', 'arc']:
        raise ValueError(f'The backend method could be either "QCElemental" or "ARC", got {backend}.')
    atom_map = None

    if backend.lower() == 'arc':
        fingerprint_1 = fingerprint(spc_1, consider_chirality=consider_chirality)
        fingerprint_2 = fingerprint(spc_2, consider_chirality=consider_chirality)
        candidates = identify_superimposable_candidates(fingerprint_1, fingerprint_2)
        if not len(candidates):
            if allow_backend_shift:
                backend = 'QCElemental'
            else:
                return None
        else:
            rmsds, fixed_spcs = list(), list()
            for candidate in candidates:
                fixed_spc_1, fixed_spc_2 = fix_dihedrals_by_backbone_mapping(spc_1, spc_2, backbone_map=candidate)
                fixed_spcs.append((fixed_spc_1, fixed_spc_2))
                backbone_1, backbone_2 = set(list(candidate.keys())), set(list(candidate.values()))
                xyz1, xyz2 = fixed_spc_1.get_xyz(), fixed_spc_2.get_xyz()
                xyz1 = xyz_from_data(coords=[xyz1['coords'][i] for i in range(fixed_spc_1.number_of_atoms) if i in backbone_1],
                                     symbols=[xyz1['symbols'][i] for i in range(fixed_spc_1.number_of_atoms) if i in backbone_1],
                                     isotopes=[xyz1['isotopes'][i] for i in range(fixed_spc_1.number_of_atoms) if i in backbone_1])
                xyz2 = xyz_from_data(coords=[xyz2['coords'][i] for i in range(fixed_spc_2.number_of_atoms) if i in backbone_2],
                                     symbols=[xyz2['symbols'][i] for i in range(fixed_spc_2.number_of_atoms) if i in backbone_2],
                                     isotopes=[xyz2['isotopes'][i] for i in range(fixed_spc_2.number_of_atoms) if i in backbone_2])
                no_gap_candidate = remove_gaps_from_values(candidate)
                xyz2 = sort_xyz_using_indices(xyz2, indices=[v for k, v in sorted(no_gap_candidate.items(), key=lambda item: item[0])])
                rmsds.append(compare_confs(xyz1=xyz1, xyz2=xyz2, rmsd_score=True))
            chosen_candidate_index = rmsds.index(min(rmsds))
            fixed_spc_1, fixed_spc_2 = fixed_spcs[chosen_candidate_index]
            atom_map = map_hydrogens(fixed_spc_1, fixed_spc_2, candidate)
            if map_type == 'list':
                atom_map = [v for k, v in sorted(atom_map.items(), key=lambda item: item[0])]
        if atom_map is None and allow_backend_shift:
            backend = 'QCElemental'

    if backend.lower() == 'qcelemental':
        qcmol_1 = create_qc_mol(species=spc_1.copy())
        qcmol_2 = create_qc_mol(species=spc_2.copy())
        if qcmol_1 is None or qcmol_2 is None:
            return None
        if len(qcmol_1.symbols) != len(qcmol_2.symbols):
            raise ValueError(f'The number of atoms in spc1 ({spc_1.number_of_atoms}) must be equal '
                             f'to the number of atoms in spc1 ({spc_2.number_of_atoms}).')
        data = qcmol_2.align(ref_mol=qcmol_1, verbose=0, uno_cutoff=0.01)
        atom_map = data[1]['mill'].atommap.tolist()
        if map_type == 'dict':
            atom_map = {key: val for key, val in enumerate(atom_map)}
    return atom_map


def get_arc_species(spc: Union[ARCSpecies, Species, Molecule]) -> ARCSpecies:
    """
    Convert an object to an ARCSpecies object.

    Args:
        spc (Union[ARCSpecies, Species, Molecule]): An input object.

    Returns:
        ARCSpecies: The corresponding ARCSpecies object.
    """
    if isinstance(spc, ARCSpecies):
        return spc
    if isinstance(spc, Species):
        return ARCSpecies(label='S', mol=spc.molecule[0])
    if isinstance(spc, Molecule):
        return ARCSpecies(label='S', mol=spc)
    raise ValueError(f'Species entries may only be ARCSpecies, RMG Species, or RMG Molecule.\n'
                     f'Got {spc} which is a {type(spc)}.')


def create_qc_mol(species: Union[ARCSpecies, Species, Molecule, List[Union[ARCSpecies, Species, Molecule]]],
                  charge: Optional[int] = None,
                  multiplicity: Optional[int] = None,
                  ) -> Optional[QCMolecule]:
    """
    Create a single QCMolecule object instance from ARCSpecies object instance(s).

    Args:
        species (List[Union[ARCSpecies, Species, Molecule]]): Entries are ARCSpecies / RMG Species / RMG Molecule
                                                              object instances.
        charge (int, optional): The overall charge of the surface.
        multiplicity (int, optional): The overall electron multiplicity of the surface.

    Returns:
        Optional[QCMolecule]: The respective QCMolecule object instance.
    """
    species_list = [species] if not isinstance(species, list) else species
    if charge is None:
        charge = sum([spc.charge for spc in species_list])
    if multiplicity is None:
        # Trivial guess.
        multiplicity = sum([spc.multiplicity - 1 for spc in species_list]) + 1
    radius = max([spc.radius for spc in species_list]) if len(species_list) > 1 else 0
    qcmol = None
    data = '\n--\n'.join([xyz_to_str(translate_xyz(spc.get_xyz(), translation=(i * radius, 0, 0)))
                          for i, spc in enumerate(species_list)]) \
        if len(species_list) > 1 else xyz_to_str(species_list[0].get_xyz())
    try:
        qcmol = QCMolecule.from_data(
            data=data,
            molecular_charge=charge,
            molecular_multiplicity=multiplicity,
            fragment_charges=[spc.charge for spc in species_list],
            fragment_multiplicities=[spc.multiplicity for spc in species_list],
            orient=False,
        )
    except ValidationError as err:
        logger.warning(f'Could not get atom map for {[spc.label for spc in species_list]}, got:\n{err}')
    return qcmol


def check_species_before_mapping(spc_1: ARCSpecies,
                                 spc_2: ARCSpecies,
                                 verbose: bool = False,
                                 ) -> bool:
    """
    Perform general checks before mapping two species.

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        verbose (bool, optional): Whether to use logging.

    Returns:
        bool: ``True`` if all checks passed, ``False`` otherwise.
    """
    # Check number of atoms > 0.
    if spc_1.number_of_atoms == 0 or spc_2.number_of_atoms == 0:
        if verbose:
            logger.warning(f'The number of atoms must be larger than 0, '
                           f'got {spc_1.number_of_atoms} and {spc_2.number_of_atoms}.')
        return False
    # Check the same number of atoms.
    if spc_1.number_of_atoms != spc_2.number_of_atoms:
        if verbose:
            logger.warning(f'The number of atoms must be identical between the two species, '
                           f'got {spc_1.number_of_atoms} and {spc_2.number_of_atoms}.')
        return False
    # Check the same number of each element.
    element_dict_1, element_dict_2 = dict(), dict()
    for atom in spc_1.mol.vertices:
        element_dict_1[atom.element.symbol] = element_dict_1.get(atom.element.symbol, 0) + 1
    for atom in spc_2.mol.vertices:
        element_dict_2[atom.element.symbol] = element_dict_2.get(atom.element.symbol, 0) + 1
    for key, val in element_dict_1.items():
        if val != element_dict_2[key]:
            if verbose:
                logger.warning(f'The chemical formula of the two species is not identical, got the following elements:\n'
                               f'{element_dict_1}\n{element_dict_2}')
            return False
    # Check the same number of bonds between similar elements (ignore bond order).
    bonds_dict_1, bonds_dict_2 = get_bonds_dict(spc_1), get_bonds_dict(spc_2)
    for key, val in bonds_dict_1.items():
        if key not in bonds_dict_2.keys() or val != bonds_dict_2[key]:
            if verbose:
                logger.warning(f'The chemical bonds in the two species are not identical, got the following bonds:\n'
                               f'{bonds_dict_1}\n{bonds_dict_2}')
            return False
    # Check whether both species are cyclic.
    if spc_1.mol.is_cyclic() != spc_2.mol.is_cyclic():
        if verbose:
            logger.warning(f'Both species should be either linear or non-linear, got:\n'
                           f'linear = {spc_1.mol.is_linear()} and linear = {spc_2.mol.is_linear()}.')
        return False
    return True


def get_bonds_dict(spc: ARCSpecies) -> Dict[str, int]:
    """
    Get a dictionary of bonds by elements in the species ignoring bond orders.

    Args:
        spc (ARCSpecies): The species to examine.

    Returns:
        Dict[str, int]: Keys are 'A-B' strings of element symbols sorted alphabetically (e.g., 'C-H' or 'H-O'),
                        values are the number of such bonds (ignoring bond order).
    """
    bond_dict = dict()
    bonds = spc.mol.get_all_edges()
    for bond in bonds:
        key = '-'.join(sorted([bond.atom1.element.symbol, bond.atom2.element.symbol]))
        bond_dict[key] = bond_dict.get(key, 0) + 1
    return bond_dict


def fingerprint(spc: ARCSpecies,
                consider_chirality: bool = True
                ) -> Dict[int, Dict[str, Union[str, List[int]]]]:
    """
    Determine the species fingerprint.
    For any heavy atom in the ``spc`` its element (``self``) will be determined,
    the element types and numbers of adjacent atoms are determined,
    and any chirality information will be determined if relevant.

    Args:
        spc (ARCSpecies): The input species.
        consider_chirality (bool, optional): Whether to consider chirality when fingerprinting.

    Returns:
        Dict[int, Dict[str, List[int]]]: Keys are indices of heavy atoms, values are dicts. keys are element symbols,
                                         values are indices of adjacent atoms corresponding to this element.
    """
    fingerprint_dict = dict()
    chirality_dict = determine_chirality(conformers=[{'xyz': spc.get_xyz()}],
                                         label=spc.label,
                                         mol=spc.mol)[0]['chirality'] if consider_chirality else {}
    for i, atom_1 in enumerate(spc.mol.atoms):
        if atom_1.is_non_hydrogen():
            atom_fingerprint = {'self': atom_1.element.symbol}
            if consider_chirality:
                for key, val in chirality_dict.items():
                    if i in key:
                        atom_fingerprint['chirality'] = val
            # if atom_1.label:
            #     atom_fingerprint['label'] = atom_1.label
            for atom_2 in atom_1.edges.keys():
                if atom_2.element.symbol not in atom_fingerprint .keys():
                    atom_fingerprint[atom_2.element.symbol] = list()
                atom_fingerprint[atom_2.element.symbol] = sorted(atom_fingerprint[atom_2.element.symbol]
                                                                 + [spc.mol.atoms.index(atom_2)])
            fingerprint_dict[i] = atom_fingerprint
    return fingerprint_dict


def identify_superimposable_candidates(fingerprint_1: Dict[int, Dict[str, Union[str, List[int]]]],
                                       fingerprint_2: Dict[int, Dict[str, Union[str, List[int]]]],
                                       ) -> List[Dict[int, int]]:
    """
    Identify candidate ordering of heavy atoms (only) that could potentially be superimposed.

    Args:
        fingerprint_1 (Dict[int, Dict[str, Union[str, List[int]]]]): Adjacent element dict for species 1.
        fingerprint_2 (Dict[int, Dict[str, Union[str, List[int]]]]): Adjacent element dict for species 2.

    Returns:
        List[Dict[int, int]]: Entries are superimposable candidate dicts. Keys are atom indices of heavy atoms
                              of species 1, values are potentially mapped atom indices of species 2.
    """
    candidates = list()
    for key_1, val_1 in fingerprint_1.items():
        for key_2, val_2 in fingerprint_2.items():
            # Try all combinations of heavy atoms.
            result = iterative_dfs(fingerprint_1, fingerprint_2, key_1, key_2)
            if result is not None:
                candidates.append(result)
    return prune_identical_dicts(candidates)


def are_adj_elements_in_agreement(fingerprint_1: Dict[str, Union[str, List[int]]],
                                  fingerprint_2: Dict[str, Union[str, List[int]]],
                                  ) -> bool:
    """
    Check whether two dictionaries representing adjacent elements are in agreement
    w.r.t. the type and number of adjacent elements.
    Also checks the identity of the parent ("self") element.

    Args:
          fingerprint_1 (Dict[str, List[int]]): Adjacent elements dictionary 1.
          fingerprint_2 (Dict[str, List[int]]): Adjacent elements dictionary 2.

    Returns:
        bool: ``True`` if the two dicts represent identical adjacent elements, ``False`` otherwise.
    """
    if len(list(fingerprint_1.keys())) != len(list(fingerprint_2.keys())):
        return False
    if fingerprint_1['self'] != fingerprint_2['self']:
        return False
    if fingerprint_1.get('chirality', 1) != fingerprint_2.get('chirality', 1):
        return False
    if fingerprint_1.get('label', 1) != fingerprint_2.get('label', 1):
        return False
    for key, val in fingerprint_1.items():
        if key not in RESERVED_FINGERPRINT_KEYS and (key not in fingerprint_2 or len(val) != len(fingerprint_2[key])):
            return False
    return True


def iterative_dfs(fingerprint_1: Dict[int, Dict[str, List[int]]],
                  fingerprint_2: Dict[int, Dict[str, List[int]]],
                  key_1: int,
                  key_2: int,
                  allow_first_key_pair_to_disagree: bool = False,
                  ) -> Optional[Dict[int, int]]:
    """
    A depth first search (DFS) graph traversal algorithm to determine possible superimposable ordering of heavy atoms.
    This is an iterative and not a recursive algorithm since Python doesn't have a great support for recursion
    since it lacks Tail Recursion Elimination and because there is a limit of recursion stack depth (by default is 1000).

    Args:
        fingerprint_1 (Dict[int, Dict[str, List[int]]]): Adjacent elements dictionary 1 (graph 1).
        fingerprint_2 (Dict[int, Dict[str, List[int]]]): Adjacent elements dictionary 2 (graph 2).
        key_1 (int): The starting index for graph 1.
        key_2 (int): The starting index for graph 2.
        allow_first_key_pair_to_disagree (bool, optional): ``True`` to not enforce agreement between the fingerprint
                                                           of ``key_1`` and ``key_2``.

    Returns:
        Optional[Dict[int, int]]: ``None`` if this is an invalid superimposable candidate. Keys are atom indices of
                                  heavy atoms of species 1, values are potentially mapped atom indices of species 2.
    """
    visited_1, visited_2 = list(), list()
    stack_1, stack_2 = deque(), deque()
    stack_1.append(key_1)
    stack_2.append(key_2)
    result = dict()
    i = 2
    while stack_1 and stack_2:
        if i:
            i -= 1
        key_1 = stack_1.pop()
        key_2 = stack_2.pop()
        if key_1 in visited_1 or key_2 in visited_2:
            continue
        visited_1.append(key_1)
        visited_2.append(key_2)
        if not are_adj_elements_in_agreement(fingerprint_1[key_1], fingerprint_2[key_2]) \
                and not (i and allow_first_key_pair_to_disagree):
            continue
        result[key_1] = key_2
        for symbol in fingerprint_1[key_1].keys():
            if symbol not in RESERVED_FINGERPRINT_KEYS + ['H']:
                for combination_tuple in product(fingerprint_1[key_1][symbol], fingerprint_2[key_2][symbol]):
                    if combination_tuple[0] not in visited_1 and combination_tuple[1] not in visited_2:
                        stack_1.append(combination_tuple[0])
                        stack_2.append(combination_tuple[1])
    if len(result.keys()) != len(fingerprint_1.keys()):
        return None
    return result


def prune_identical_dicts(dicts_list: List[dict]) -> List[dict]:
    """
    Return a list of unique dictionaries.

    Args:
        dicts_list (List[dict]): A list of dicts to prune.

    Returns:
        List[dict]: A list of unique dicts.
    """
    new_dicts_list = list()
    for new_dict in dicts_list:
        unique = True
        for existing_dict in new_dicts_list:
            if unique:
                for new_key, new_val in new_dict.items():
                    if new_key not in existing_dict.keys() or new_val == existing_dict[new_key]:
                        unique = False
                        break
        if unique:
            new_dicts_list.append(new_dict)
    return new_dicts_list


def remove_gaps_from_values(data: Dict[int, int]) -> Dict[int, int]:
    """
    Return a dictionary of integer keys and values with consecutive values starting at 0.

    Args:
        data (Dict[int, int]): A dictionary of integers as keys and values.

    Returns:
        Dict[int, int]: A dictionary of integers as keys and values with consecutive values starting at 0.
    """
    new_data = dict()
    val = 0
    for key, _ in sorted(data.items(), key=lambda item: item[1]):
        new_data[key] = val
        val += 1
    return new_data


def fix_dihedrals_by_backbone_mapping(spc_1: ARCSpecies,
                                      spc_2: ARCSpecies,
                                      backbone_map: Dict[int, int],
                                      ) -> Tuple[ARCSpecies, ARCSpecies]:
    """
    Fix the dihedral angles of two mapped species to align them.

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): The backbone map.

    Returns:
        Tuple[ARCSpecies, ARCSpecies]: The corresponding species with aligned dihedral angles.
    """
    if not spc_1.rotors_dict or not spc_2.rotors_dict:
        spc_1.determine_rotors()
        spc_2.determine_rotors()
    spc_1_copy, spc_2_copy = spc_1.copy(), spc_2.copy()
    torsions = get_backbone_dihedral_angles(spc_1, spc_2, backbone_map)
    deviations = [get_backbone_dihedral_deviation_score(spc_1, spc_2, backbone_map, torsions=torsions)]
    # Loop while the deviation improves by more than 1 degree:
    while len(torsions) and (len(deviations) < 2 or deviations[-2] - deviations[-1] > 1):
        for torsion_dict in torsions:
            angle = 0.5 * sum([torsion_dict['angle 1'], torsion_dict['angle 2']])
            spc_1_copy.set_dihedral(scan=convert_list_index_0_to_1(torsion_dict['torsion 1']),
                                    deg_abs=angle, count=False, chk_rotor_list=False, xyz=spc_1_copy.get_xyz())
            spc_2_copy.set_dihedral(scan=convert_list_index_0_to_1(torsion_dict['torsion 2']),
                                    deg_abs=angle, count=False, chk_rotor_list=False, xyz=spc_2_copy.get_xyz())
            spc_1_copy.final_xyz, spc_2_copy.final_xyz = spc_1_copy.initial_xyz, spc_2_copy.initial_xyz
        torsions = get_backbone_dihedral_angles(spc_1_copy, spc_2_copy, backbone_map)
        deviations.append(get_backbone_dihedral_deviation_score(spc_1_copy, spc_2_copy, backbone_map, torsions=torsions))
    return spc_1_copy, spc_2_copy


def get_backbone_dihedral_deviation_score(spc_1: ARCSpecies,
                                          spc_2: ARCSpecies,
                                          backbone_map: Dict[int, int],
                                          torsions: Optional[List[Dict[str, Union[float, List[int]]]]] = None
                                          ) -> float:
    """
    Determine a deviation score for dihedral angles of torsions.
    We don't consider here "terminal" torsions, just pure backbone torsions.

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): The backbone map.
        torsions (Optional[List[Dict[str, Union[float, List[int]]]]], optional): The backbone dihedral angles.
    """
    torsions = torsions or get_backbone_dihedral_angles(spc_1, spc_2, backbone_map)
    return sum([abs(torsion_dict['angle 1'] - torsion_dict['angle 2']) for torsion_dict in torsions])


def get_backbone_dihedral_angles(spc_1: ARCSpecies,
                                 spc_2: ARCSpecies,
                                 backbone_map: Dict[int, int],
                                 ) -> List[Dict[str, Union[float, List[int]]]]:
    """
    Determine the dihedral angles of the backbone torsions of two backbone mapped species.
    The output has the following format::

        torsions = [{'torsion 1': [0, 1, 2, 3],  # The first torsion in terms of species 1 indices.
                     'torsion 2': [5, 7, 2, 4],  # The first torsion in terms of species 2 indices.
                     'angle 1': 60.0,  # The corresponding dihedral angle to 'torsion 1'.
                     'angle 2': 125.1,  # The corresponding dihedral angle to 'torsion 2'.
                    },
                    {}, ...  # The second torsion, and so on.
                   ]

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): The backbone map.

    Returns:
        List[Dict[str, Union[float, List[int]]]]: The corresponding species with aligned dihedral angles.
    """
    torsions = list()
    if not spc_1.rotors_dict or not spc_2.rotors_dict:
        spc_1.determine_rotors()
        spc_2.determine_rotors()
    if spc_1.rotors_dict is not None and spc_2.rotors_dict is not None:
        for rotor_dict_1 in spc_1.rotors_dict.values():
            torsion_1 = rotor_dict_1['torsion']
            if spc_1.mol.atoms[torsion_1[0]].is_non_hydrogen() \
                    and spc_1.mol.atoms[torsion_1[3]].is_non_hydrogen():
                # This is not a "terminal" torsion.
                for rotor_dict_2 in spc_2.rotors_dict.values():
                    torsion_2 = [backbone_map[t_1] for t_1 in torsion_1]
                    if all(pivot_2 in [torsion_2[1], torsion_2[2]]
                           for pivot_2 in [rotor_dict_2['torsion'][1], rotor_dict_2['torsion'][2]]):
                        torsions.append({'torsion 1': torsion_1,
                                         'torsion 2': torsion_2,
                                         'angle 1': calculate_dihedral_angle(coords=spc_1.get_xyz(), torsion=torsion_1),
                                         'angle 2': calculate_dihedral_angle(coords=spc_2.get_xyz(), torsion=torsion_2)})
    return torsions


def map_lists(list_1: List[float],
              list_2: List[float],
              ) -> Dict[int, int]:
    """
    Map two lists with equal lengths of floats by proximity of entries.
    Assuming no two entries in a list are identical.
    Note: values are treated as cyclic in a 0-360 range (i.e., 1 is closer to 259 than to 5)!

    Args:
        list_1 (List[float]): List 1.
        list_2 (List[float]): List 2.

    Returns:
        Dict[int, int]: The lists map. Keys correspond to entry indices in ``list_1``,
                        Values correspond to respective entry indices in ``list_2``.
    """
    if len(list_1) != len(list_2):
        raise ValueError(f'Lists must be of the same length, got:\n{list_1}\n{list_2}\n'
                         f'with lengths {len(list_1)} and {len(list_2)}.')
    list_map = dict()
    dict_1, dict_2 = {k: True for k in list_1}, {k: True for k in list_2}
    while any(dict_1.values()):
        min_1_index = list_1.index(min([k for k, v in dict_1.items() if v]))
        angle_deltas = [get_delta_angle(list_1[min_1_index], angle_2) if dict_2[angle_2] else None for angle_2 in list_2]
        min_2_index = angle_deltas.index(extremum_list(angle_deltas, return_min=True))
        dict_1[list_1[min_1_index]], dict_2[list_2[min_2_index]] = False, False
        list_map[min_1_index] = min_2_index
    return list_map


def map_hydrogens(spc_1: ARCSpecies,
                  spc_2: ARCSpecies,
                  backbone_map: Dict[int, int],
                  ) -> Dict[int, int]:
    """
    Atom map hydrogen atoms between two species with a known mapped heavy atom backbone.
    If only a single hydrogen atom is bonded to a given heavy atom, it is straight-forwardly mapped.
    If more than one hydrogen atom is bonded to a given heavy atom forming a "terminal" internal rotor,
    an internal rotation will be attempted to find the closest match, e.g., in cases such as::

        C -- H1     |         H1
         \          |       /
          H2        |     C -- H2

    To avoid mapping H2 to H1 due to small RMSD, but H1 to H2 although the RMSD is huge.
    Further, we assume that each H has but one covalent bond, and that there are maximum 4 hydrogen atoms per heavy atom
    (e.g., CH4 or SiH4).

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): The backbone map.

    Returns:
        Dict[int, int]: The atom map. Keys are 0-indices in ``spc_1``, values are corresponding 0-indices in ``spc_2``.
    """
    atom_map = backbone_map
    atoms_1, atoms_2 = spc_1.mol.atoms, spc_2.mol.atoms
    for hydrogen_1 in atoms_1:
        if hydrogen_1.is_hydrogen() and atoms_1.index(hydrogen_1) not in atom_map.keys():
            success = False
            heavy_atom_1 = list(hydrogen_1.edges.keys())[0]
            heavy_atom_2 = atoms_2[backbone_map[atoms_1.index(heavy_atom_1)]]
            num_hydrogens_1 = len([atom for atom in heavy_atom_1.edges.keys() if atom.is_hydrogen()])
            if num_hydrogens_1 == 1:
                # We know that num_hydrogens_2 == 1 because the candidate map resulted from are_adj_elements_in_agreement().
                hydrogen_2 = [atom for atom in heavy_atom_2.edges.keys() if atom.is_hydrogen()][0]
                atom_map[atoms_1.index(hydrogen_1)] = atoms_2.index(hydrogen_2)
                success = True
            # Consider 2/3/4 hydrogen atoms on this heavy atom.
            # 1. Check for a heavy atom with only H atoms adjacent to it (CH4, NH3, H2).
            if not success:
                if all(atom.is_hydrogen() for atom in heavy_atom_1.edges.keys()):
                    for atom_1, atom_2 in zip([atom for atom in atoms_1 if atom.is_hydrogen()],
                                              [atom for atom in atoms_2 if atom.is_hydrogen()]):
                        atom_map[atoms_1.index(atom_1)] = atoms_2.index(atom_2)
                    success = True
            # 2. Check for a torsion involving heavy_atom_1 as a pivotal atom (most common case).
            if not success:
                if spc_1.rotors_dict is not None:
                    heavy_atom_1_index = atoms_1.index(heavy_atom_1)
                    for rotor_dict in spc_1.rotors_dict.values():
                        if heavy_atom_1_index in [rotor_dict['torsion'][1], rotor_dict['torsion'][2]]:
                            atom_map = add_adjacent_hydrogen_atoms_to_map_based_on_a_specific_torsion(
                                spc_1=spc_1,
                                spc_2=spc_2,
                                heavy_atom_1=heavy_atom_1,
                                heavy_atom_2=heavy_atom_2,
                                torsion=rotor_dict['torsion'],
                                atom_map=atom_map,
                                find_torsion_end_to_map=True,
                            )
                            success = True
                            break
            # 3. Check for a pseudo-torsion (may involve multiple bonds) with heavy_atom_1 as a pivot.
            if not success:
                pseudo_torsion = list()
                for atom_1_3 in heavy_atom_1.edges.keys():
                    if atom_1_3.is_non_hydrogen():
                        for atom_1_4 in atom_1_3.edges.keys():
                            if atom_1_4.is_non_hydrogen() and atom_1_4 is not heavy_atom_1:
                                pseudo_torsion = [atoms_1.index(atom) for atom in [hydrogen_1, heavy_atom_1, atom_1_3, atom_1_4]]
                                break
                        if not len(pseudo_torsion):
                            # Compromise for a hydrogen atom in position 4.
                            for atom_1_4 in atom_1_3.edges.keys():
                                if atom_1_4 is not heavy_atom_1:
                                    pseudo_torsion = [atoms_1.index(atom) for atom in [hydrogen_1, heavy_atom_1, atom_1_3, atom_1_4]]
                                    break
                        if len(pseudo_torsion):
                            atom_map = add_adjacent_hydrogen_atoms_to_map_based_on_a_specific_torsion(
                                spc_1=spc_1,
                                spc_2=spc_2,
                                heavy_atom_1=heavy_atom_1,
                                heavy_atom_2=heavy_atom_2,
                                torsion=pseudo_torsion[::-1],
                                atom_map=atom_map,
                                find_torsion_end_to_map=False,
                            )
                            success = True
                            break
            # 4. Check by angles and bond lengths (search for 2 consecutive heavy atoms).
            if not success:
                atom_1_3, angle_1, bond_length_1 = None, None, None
                for atom_1_3 in heavy_atom_1.edges.keys():
                    if atom_1_3.is_non_hydrogen():
                        heavy_atom_1_index, hydrogen_1_index = atoms_1.index(heavy_atom_1), atoms_1.index(hydrogen_1)
                        angle_1 = calculate_angle(coords=spc_1.get_xyz(),
                                                  atoms=[atoms_1.index(atom_1_3), heavy_atom_1_index, hydrogen_1_index])
                        bond_length_1 = calculate_distance(coords=spc_1.get_xyz(),
                                                           atoms=[heavy_atom_1_index, hydrogen_1_index])
                        break
                if atom_1_3 is not None:
                    atom_2_3_index = atom_map[atoms_1.index(atom_1_3)]
                    angle_deviations, bond_length_deviations, hydrogen_indices_2 = list(), list(), list()
                    for hydrogen_2 in heavy_atom_2.edges.keys():
                        if hydrogen_2.is_hydrogen() and atoms_2.index(hydrogen_2) not in atom_map.values():
                            heavy_atom_2_index, hydrogen_2_index = atoms_2.index(heavy_atom_2), atoms_2.index(hydrogen_2)
                            angle_2 = calculate_angle(coords=spc_2.get_xyz(),
                                                      atoms=[atom_2_3_index, heavy_atom_2_index, hydrogen_2_index])
                            bond_length_2 = calculate_distance(coords=spc_2.get_xyz(),
                                                               atoms=[heavy_atom_2_index, hydrogen_2_index])
                            angle_deviations.append(abs(angle_1 - angle_2))
                            bond_length_deviations.append(abs(bond_length_1 - bond_length_2))
                            hydrogen_indices_2.append(hydrogen_2_index)
                    deviations = [bond_length_deviations[i] * hydrogen_indices_2[i] for i in range(len(angle_deviations))]
                    atom_map[atoms_1.index(hydrogen_1)] = hydrogen_indices_2[deviations.index(min(deviations))]
    return atom_map


def add_adjacent_hydrogen_atoms_to_map_based_on_a_specific_torsion(spc_1: ARCSpecies,
                                                                   spc_2: ARCSpecies,
                                                                   heavy_atom_1: 'Atom',
                                                                   heavy_atom_2: 'Atom',
                                                                   torsion: List[int],
                                                                   atom_map: Dict[int, int],
                                                                   find_torsion_end_to_map: bool = True,
                                                                   ) -> Dict[int, int]:
    """
    Map hydrogen atoms around one end of a specific torsion (or pseudo-torsion) by matching dihedral angles.

    Args:
          spc_1 (ARCSpecies): Species 1.
          spc_2 (ARCSpecies): Species 2.
          heavy_atom_1 (Atom): The heavy atom from ``spc_1`` around which hydrogen atoms will be mapped.
          heavy_atom_2 (Atom): The heavy atom from ``spc_2`` around which hydrogen atoms will be mapped.
          torsion (List[int]): 0-indices of 4 consecutive atoms in ``spc_1``.
          atom_map (Dict[int, int]): A partial atom map between ``spc_1`` and ``spc_2``.
          find_torsion_end_to_map (bool, optional): Whether to automatically identify the side of the torsion that
                                                    requires mapping. If ``False``, the last atom, ``torsion[-1]``
                                                    is assumed to be on the side that requires mapping.

    Returns:
        Dict[int, int]: The updated atom map.
    """
    atoms_1, atoms_2 = spc_1.mol.atoms, spc_2.mol.atoms
    hydrogen_indices_1 = [atoms_1.index(atom)
                          for atom in heavy_atom_1.edges.keys() if atom.is_hydrogen()]
    hydrogen_indices_2 = [atoms_2.index(atom)
                          for atom in heavy_atom_2.edges.keys() if atom.is_hydrogen()]
    torsion_1_base = torsion[::-1][:-1] if torsion[-1] not in hydrogen_indices_1 and find_torsion_end_to_map else torsion[:-1]
    torsions_1 = [torsion_1_base + [h_index] for h_index in hydrogen_indices_1]
    if torsion_1_base[0] not in atom_map.keys():
        # There's an unmapped hydrogen atom in torsion_1_base. Randomly map it to a respective hydrogen.
        atom_map[torsion_1_base[0]] = [atoms_2.index(atom)
                                       for atom in atoms_2[atom_map[torsion_1_base[1]]].edges.keys()
                                       if atom.is_hydrogen()][0]
    torsions_2 = [[atom_map[t] for t in torsion_1_base] + [h_index] for h_index in hydrogen_indices_2]
    dihedrals_1 = [calculate_dihedral_angle(coords=spc_1.get_xyz(), torsion=torsion)
                   for torsion in torsions_1]
    dihedrals_2 = [calculate_dihedral_angle(coords=spc_2.get_xyz(), torsion=torsion)
                   for torsion in torsions_2]
    dihedral_map_dict = map_lists(dihedrals_1, dihedrals_2)
    for key, val in dihedral_map_dict.items():
        atom_map[hydrogen_indices_1[key]] = hydrogen_indices_2[val]
    return atom_map
