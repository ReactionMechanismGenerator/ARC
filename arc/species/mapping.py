"""
A module for atom-mapping a species or a set of species.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from qcelemental.exceptions import ValidationError
from qcelemental.models.molecule import Molecule as QCMolecule

from rmgpy.molecule import Molecule
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.common import convert_list_index_0_to_1, logger
from arc.species import ARCSpecies
from arc.species.converter import translate_xyz, xyz_to_str
from arc.species.vectors import calculate_dihedral_angle


if TYPE_CHECKING:
    from rmgpy.data.kinetics.family import TemplateReaction
    from rmgpy.data.rmg import RMGDatabase
    from rmgpy.reaction import Reaction
    from arc.reaction import ARCReaction


def map_reaction(rxn: 'ARCReaction',
                 db: Optional['RMGDatabase'] = None,
                 ) -> Optional[List[int]]:
    """
    Map a reaction.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    if rxn.family is None:
        rmgdb.determine_family(reaction=rxn, db=db)
    if rxn.family is None:
        return map_general_rxn(rxn)

    fam_func_dict = {'H_Abstraction': map_h_abstraction,
                     'HO2_Elimination_from_PeroxyRadical': map_ho2_elimination_from_peroxy_radical,
                     'intra_H_migration': map_intra_h_migration,
                     }

    if rxn.family.label not in fam_func_dict.keys():
        logger.info(f'Using a generic mapping algorithm for {rxn} of family {rxn.family.label}')

    map_func = fam_func_dict.get(rxn.family.label, map_general_rxn)

    return map_func(rxn, db)


def map_general_rxn(rxn: 'ARCReaction',
                    db: Optional['RMGDatabase'] = None,
                    ) -> Optional[List[int]]:
    """
    Map a general reaction (one that was not categorized into a reaction family by RMG).
    The general method isn't great, a family-specific method should be implemented where possible.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants, entry values are running atom indices of the products.
    """
    if rxn.is_isomerization():
        return map_two_species(rxn.r_species[0], rxn.p_species[0], map_type='list')

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


# Family-specific mapping functions:


def map_h_abstraction(rxn: 'ARCReaction',
                      db: Optional['RMGDatabase'] = None,
                      ) -> Optional[List[int]]:
    """
    Map a hydrogen abstraction reaction.
    Strategy: Map species R(*1)-H(*2) to species R(*1)j and map species R(*3)j to species R(*3)-H(*2).
    Use scissors to map the backbone.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants, entry values are running atom indices of the products.
    """
    if not check_family_for_mapping_function(rxn=rxn, db=db, family='H_Abstraction'):
        return None

    rmg_reactions = _get_rmg_reactions_from_arc_reaction(arc_reaction=rxn)
    r_label_dict, p_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                                      rmg_reaction=rmg_reactions[0])
    r_h_index = r_label_dict['*2']
    p_h_index = p_label_dict['*2']
    len_r1, len_p1 = rxn.r_species[0].number_of_atoms, rxn.p_species[0].number_of_atoms
    r1_h2 = 0 if r_h_index < len_r1 else 1  # Identify R(*1)-H(*2), it's either reactant 0 or reactant 1.
    r3 = 1 - r1_h2  # Identify R(*3) in the reactants.
    r3_h2 = 0 if p_h_index < len_p1 else 1  # Identify R(*3)-H(*2), it's either product 0 or product 1.
    r1 = 1 - r3_h2  # Identify R(*1) in the products.

    spc_r1_h2 = ARCSpecies(label='R1-H2',
                           mol=rxn.r_species[r1_h2].mol.copy(deep=True),
                           xyz=rxn.r_species[r1_h2].get_xyz(),
                           bdes=[(r_label_dict['*1'] + 1 - r1_h2 * len_r1,
                                  r_label_dict['*2'] + 1 - r1_h2 * len_r1)],  # Mark the R(*1)-H(*2) bond for scission.
                           )
    spc_r1_h2.final_xyz = spc_r1_h2.get_xyz()  # Scissors require the .final_xyz attribute to be populated.
    spc_r1_h2_cuts = spc_r1_h2.scissors()
    spc_r1_h2_cut = [spc for spc in spc_r1_h2_cuts if spc.label != 'H'][0] \
        if any(spc.label != 'H' for spc in spc_r1_h2_cuts) else spc_r1_h2_cuts[0]  # Treat H2 as well :)
    spc_r3_h2 = ARCSpecies(label='R3-H2',
                           mol=rxn.p_species[r3_h2].mol.copy(deep=True),
                           xyz=rxn.p_species[r3_h2].get_xyz(),
                           bdes=[(p_label_dict['*3'] + 1 - r3_h2 * len_p1,
                                  p_label_dict['*2'] + 1 - r3_h2 * len_p1)],  # Mark the R(*3)-H(*2) bond for scission.
                           )
    spc_r3_h2.final_xyz = spc_r3_h2.get_xyz()  # Scissors require the .final_xyz attribute to be populated.
    spc_r3_h2_cuts = spc_r3_h2.scissors()
    spc_r3_h2_cut = [spc for spc in spc_r3_h2_cuts if spc.label != 'H'][0] \
        if any(spc.label != 'H' for spc in spc_r3_h2_cuts) else spc_r3_h2_cuts[0]  # Treat H2 as well :)
    map_1 = map_two_species(spc_r1_h2_cut, rxn.p_species[r1])
    map_2 = map_two_species(rxn.r_species[r3], spc_r3_h2_cut)

    result = {r_h_index: p_h_index}
    for r_increment, p_increment, map_ in zip([r1_h2 * len_r1, (1 - r1_h2) * len_r1],
                                              [(1 - r3_h2) * len_p1, r3_h2 * len_p1],
                                              [map_1, map_2]):
        for i, entry in enumerate(map_):
            r_index = i + r_increment + int(i + r_increment >= r_h_index)
            p_index = entry + p_increment
            result[r_index] = p_index
    return [val for key, val in sorted(result.items(), key=lambda item: item[0])]


def map_ho2_elimination_from_peroxy_radical(rxn: 'ARCReaction',
                                            db: Optional['RMGDatabase'] = None,
                                            ) -> Optional[List[int]]:
    """
    Map an HO2 elimination from peroxy radical reaction.
    Strategy: Remove the O(*3), O(*4), and H(*5) atoms from the reactant and map to the R(*1)=R(*2) product.
    Note that two consecutive scissions must be performed.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants, entry values are running atom indices of the products.
    """
    if not check_family_for_mapping_function(rxn=rxn, db=db, family='HO2_Elimination_from_PeroxyRadical'):
        return None

    rmg_reactions = _get_rmg_reactions_from_arc_reaction(arc_reaction=rxn)
    r_label_dict, p_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                                      rmg_reaction=rmg_reactions[0])

    # reverse = False
    # if len(rxn.p_species) == 1 and len(rxn.r_species) == 2:
    #     reverse = True
    #     r_label_dict, p_label_dict = p_label_dict, r_label_dict

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
        map_1 = map_two_species(spc_r_mod, rxn.p_species[r1dr2])
        for rotor in spc_r_mod.rotors_dict.values():
            torsion = rotor['torsion']
            if not spc_r_mod.mol.atoms[torsion[0]].is_hydrogen() and not spc_r_mod.mol.atoms[torsion[1]].is_hydrogen():
                spc_r_mod.set_dihedral(scan=convert_list_index_0_to_1(torsion),
                                       deg_abs=calculate_dihedral_angle(coords=rxn.p_species[r1dr2].get_xyz(),
                                                                        torsion=[map_1[t] for t in torsion]),
                                       chk_rotor_list=False)
                spc_r_mod.final_xyz = spc_r_mod.initial_xyz
        map_2 = map_two_species(spc_r_mod, rxn.p_species[r1dr2])
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
                          db: Optional['RMGDatabase'] = None,
                          ) -> Optional[List[int]]:
    """
    Map an intra hydrogen migration reaction.
    Strategy: Remove the *3 H atom from both the reactant and product to have the same backbone.
    Map the backbone and add the (known) *3 H atom.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants, entry values are running atom indices of the products.
    """
    if not check_family_for_mapping_function(rxn=rxn, db=db, family='intra_H_migration'):
        return None

    rmg_reactions = _get_rmg_reactions_from_arc_reaction(arc_reaction=rxn)
    r_label_dict, p_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                                      rmg_reaction=rmg_reactions[0])

    r_h_index = r_label_dict['*3']
    p_h_index = p_label_dict['*3']

    spc_r = ARCSpecies(label='R',
                       mol=rxn.r_species[0].mol.copy(deep=True),
                       xyz=rxn.r_species[0].get_xyz(),
                       bdes=[(r_label_dict['*2'] + 1, r_label_dict['*3'] + 1)],  # Mark the R(*2)-H(*3) bond for scission.
                       )
    spc_r.final_xyz = spc_r.get_xyz()  # Scissors require the .final_xyz attribute to be populated.
    spc_r_dot = [spc for spc in spc_r.scissors() if spc.label != 'H'][0]
    spc_p = ARCSpecies(label='P',
                       mol=rxn.p_species[0].mol.copy(deep=True),
                       xyz=rxn.p_species[0].get_xyz(),
                       bdes=[(p_label_dict['*1'] + 1, p_label_dict['*3'] + 1)],  # Mark the R(*1)-H(*3) bond for scission.
                       )
    spc_p.final_xyz = spc_p.get_xyz()  # Scissors require the .final_xyz attribute to be populated.
    spc_p_dot = [spc for spc in spc_p.scissors() if spc.label != 'H'][0]
    map_ = map_two_species(spc_r_dot, spc_p_dot)

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


def map_two_species(spc_1: Union[ARCSpecies, Species, Molecule],
                    spc_2: Union[ARCSpecies, Species, Molecule],
                    map_type: str = 'list',
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

    Returns:
        Optional[Union[List[int], Dict[int, int]]]:
            The atom map. By default, a list is returned.
            If the map is of ``list`` type, entry indices are atom indices of ``spc_1``, entry values are atom indices of ``spc_2``.
            If the map is of ``dict`` type, keys are atom indices of ``spc_1``, values are atom indices of ``spc_2``.
    """
    qcmol_1 = create_qc_mol(species=spc_1.copy())
    qcmol_2 = create_qc_mol(species=spc_2.copy())
    if qcmol_1 is None or qcmol_2 is None:
        return None
    if len(qcmol_1.symbols) != len(qcmol_2.symbols):
        raise ValueError(f'The number of atoms in spc1 ({spc_1.number_of_atoms}) must be equal '
                         f'to the number of atoms in spc1 ({spc_2.number_of_atoms}).')
    data = qcmol_2.align(ref_mol=qcmol_1, verbose=0)
    atom_map = data[1]['mill'].atommap.tolist()
    if map_type == 'dict':  # ** Todo: test
        atom_map = {key: val for key, val in enumerate(atom_map)}
    return atom_map


def create_qc_mol(species: Union[ARCSpecies, Species, Molecule, List[Union[ARCSpecies, Species, Molecule]]],
                  charge: Optional[int] = None,
                  multiplicity: Optional[int] = None,
                  ) -> Optional[QCMolecule]:
    """
    Create a single QCMolecule object instance from a ARCSpecies object instances.

    Args:
        species (List[Union[ARCSpecies, Species, Molecule]]): Entries are ARCSpecies / RMG Species / RMG Molecule
                                                              object instances.
        charge (int, optional): The overall charge of the surface.
        multiplicity (int, optional): The overall electron multiplicity of the surface.

    Returns:
        Optional[QCMolecule]: The respective QCMolecule object instance.
    """
    species = [species] if not isinstance(species, list) else species
    species_list = list()
    for spc in species:
        if isinstance(spc, ARCSpecies):
            species_list.append(spc)
        elif isinstance(spc, Species):
            species_list.append(ARCSpecies(label='S', mol=spc.molecule[0]))
        elif isinstance(spc, Molecule):
            species_list.append(ARCSpecies(label='S', mol=spc))
        else:
            raise ValueError(f'Species entries may only be ARCSpecies, RMG Species, or RMG Molecule, '
                             f'got {spc} which is a {type(spc)}.')
    if len(species_list) == 1:
        if charge is None:
            charge = species_list[0].charge
        if multiplicity is None:
            multiplicity = species_list[0].multiplicity
    if charge is None or multiplicity is None:
        raise ValueError(f'An overall charge and multiplicity must be specified for multiple species, '
                         f'got: {charge} and {multiplicity}, respectively')
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
                if isinstance(rmg_obj, Molecule):
                    rmg_spc = Species(molecule=[rmg_obj])
                elif isinstance(rmg_obj, Species):
                    rmg_spc = rmg_obj
                else:
                    raise ValueError(f'Expected an RMG object instance of Molecule() or Species(),'
                                     f'got {rmg_obj} which is a {type(rmg_obj)}.')
                rmg_spc.generate_resonance_structures(save_order=True)
                if rmg_spc.is_isomorphic(arc_spc.mol, save_order=True):
                    if i in spc_map.keys() and concatenate:  # ** Todo: test
                        spc_map[i].append(j)
                    elif concatenate:
                        spc_map[i] = [j]
                    else:
                        spc_map[i] = j
                        break
    return r_map, p_map


def find_equivalent_atoms_in_reactants(arc_reaction: 'ARCReaction') -> Optional[List[List[int]]]:
    """
    Find atom indices that are equivalent in the reactants of an ARCReaction
    in the sense that they represent degenerate reaction sites that are indifferentiable in 2D.
    Bridges between RMG reaction templates and ARC's 3D TS structures.
    Running indices in the returned structure relate to reactant_0 + reactant_1 + ...

    Args:
        arc_reaction ('ARCReaction'): The ARCReaction object instance.

    Returns:
        Optional[List[List[int]]]: Entries are lists of 0-indices, each such list represents equivalent atoms.
    """
    rmg_reactions = _get_rmg_reactions_from_arc_reaction(arc_reaction)
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


def _get_rmg_reactions_from_arc_reaction(arc_reaction: 'ARCReaction',
                                         db: Optional['RMGDatabase'] = None,
                                         ) -> Optional[List['TemplateReaction']]:
    """
    A helper function for getting RMG reactions from an ARC reaction.

    Args:
        arc_reaction (ARCReaction): The ARCReaction object instance.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[TemplateReaction]]:
            The respective RMG TemplateReaction object instances (considering resonance structures).
    """
    if arc_reaction.family is None:
        rmgdb.determine_family(reaction=arc_reaction, db=db)
    if arc_reaction.family is None:
        return None
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
                atom_map = map_two_species(mol, rmg_mol, map_type='dict')
                new_atoms_list = list()
                for i in range(len(rmg_mol.atoms)):
                    rmg_mol.atoms[atom_map[i]].id = mol.atoms[i].id
                    new_atoms_list.append(rmg_mol.atoms[atom_map[i]])
                rmg_mol.atoms = new_atoms_list
                mapped_mols.append(rmg_mol)
        rmg_reaction.reactants, rmg_reaction.products = mapped_rmg_reactants, mapped_rmg_products
    return rmg_reactions
