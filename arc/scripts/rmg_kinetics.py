#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to run RMG
and get kinetic rate coefficients for reactions
"""

import os
from typing import List, Optional, Tuple

from common import parse_command_line_arguments, read_yaml_file, save_yaml_file

from rmgpy.data.kinetics.common import find_degenerate_reactions
from rmgpy.data.kinetics.family import KineticsFamily
from rmgpy.data.rmg import RMGDatabase
from rmgpy import settings as rmg_settings
from rmgpy.reaction import same_species_lists, Reaction
from rmgpy.species import Species

DB_PATH = rmg_settings['database.directory']


def main():
    """
    Get reaction rate coefficients from RMG.

    The input YAML file should be a list of species dictionaries.
    Each species dictionary should have the following keys:
        - reactants: A list of adjacency lists of the reactants.
        - products: A list of adjacency lists of the products.
        - dh_rxn298: The heat of reaction at 298 K in J/mol.
    The returned YAML file will have the same species dictionaries with the following additional keys:
        - kinetics: The reaction kinetics as a dictionary with A, n, Ea, T_min, T_max, and comment keys.
    """
    args = parse_command_line_arguments()
    input_file = args.file
    reaction_list = read_yaml_file(path=input_file)
    if not isinstance(reaction_list, list):
        raise ValueError(f'The content of {input_file} must be a list, got {reaction_list} which is a {type(reaction_list)}')
    result = get_rate_coefficients(reaction_list)
    save_yaml_file(path=input_file, content=result)


def get_rate_coefficients(reaction_list: List[dict]) -> List[dict]:
    """
    Get rate coefficients for a list of reactions.

    Args:
        reaction_list (List[dict]): The list of reactions.

    Returns:
        List[dict]: The list of reactions with rate coefficients.
    """
    print('Loading RMG database...')
    rmgdb = load_rmg_database()
    for i in range(len(reaction_list)):
        rxn = Reaction(reactants=[Species().from_adjacency_list(adjlist) for adjlist in reaction_list[i]['reactants']],
                       products=[Species().from_adjacency_list(adjlist) for adjlist in reaction_list[i]['products']])
        reaction_list[i]['kinetics'] = determine_rmg_kinetics(rmgdb=rmgdb, reaction=rxn, dh_rxn298=reaction_list[i]['dh_rxn298'],
                                                              family=reaction_list[i]['family'] if 'family' in reaction_list[i] else None)
    return reaction_list


def determine_rmg_kinetics(rmgdb: RMGDatabase,
                           reaction: Reaction,
                           dh_rxn298: Optional[float] = None,
                           family: Optional[str] = None,
                           ) -> List[dict]:
    """
    Determine kinetics for `reaction` (an RMG Reaction object) from RMG's database, if possible.
    Assigns a list of all matching entries from both libraries and families.

    Args:
        rmgdb (RMGDatabase): The RMG database instance.
        reaction (Reaction): The RMG Reaction object.
        dh_rxn298 (float, optional): The heat of reaction at 298 K in J/mol.
        family (str, optional): The RMG family label.

    Returns: List[dict]
        All matching RMG reactions kinetics (both libraries and families) as a dict of parameters.
    """
    rmg_reactions = list()
    # Libraries:
    for library in rmgdb.kinetics.libraries.values():
        library_reactions = library.get_library_reactions()
        for library_reaction in library_reactions:
            if reaction.is_isomorphic(library_reaction):
                library_reaction.comment = f'Library: {library.label}'
                rmg_reactions.append(library_reaction)
                break
    # # Families:
    A_units = "cm^3/(mol*s)" if len(reaction.reactants) == 2 else "s^-1"
    fam_list = loop_families(rmgdb, reaction)
    dh_rxn298 = dh_rxn298 or get_dh_rxn298(rmgdb=rmgdb, reaction=reaction)  # J/mol
    for family, degenerate_reactions in fam_list:
        for deg_rxn in degenerate_reactions:
            kinetics_list = family.get_kinetics(reaction=deg_rxn, template_labels=deg_rxn.template, degeneracy=deg_rxn.degeneracy)
            for kinetics_detailes in kinetics_list:
                kinetics = kinetics_detailes[0]
                kinetics.change_rate(deg_rxn.degeneracy)
                if hasattr(kinetics, 'to_arrhenius'):
                    kinetics = kinetics.to_arrhenius(dh_rxn298)  # Convert ArrheniusEP to Arrhenius
                kinetics.A.value_si = kinetics.A.value_si * (1e6 if A_units == "cm^3/(mol*s)" else 1)
                deg_rxn.kinetics = kinetics
                deg_rxn.comment = f'Family: {deg_rxn.family}'
                if 'training' in deg_rxn.kinetics.comment:
                    deg_rxn.comment += ' (training)'
                deg_rxn.reactants = reaction.reactants
                deg_rxn.products = reaction.products
                rxn_copy = deg_rxn.copy()
                rxn_copy.comment = deg_rxn.comment
                if 'training' not in rxn_copy.comment:
                    rmg_reactions.append(rxn_copy)
    return get_kinetics_from_reactions(rmg_reactions)


def get_dh_rxn298(rmgdb: RMGDatabase,
                  reaction: Reaction,
                  ) -> float:
    """
    Get the heat of reaction at 298 K in J/mol.

    Args:
        rmgdb (RMGDatabase): The RMG database instance.
        reaction (Reaction): The RMG Reaction object.

    Returns:
        float: The heat of reaction at 298 K in J/mol.
    """
    dh_rxn = 0
    for spc in reaction.products:
        spc.thermo = rmgdb.thermo.get_thermo_data(spc)
        dh_rxn += spc.thermo.get_enthalpy(298)
    for spc in reaction.reactants:
        spc.thermo = rmgdb.thermo.get_thermo_data(spc)
        dh_rxn -= spc.thermo.get_enthalpy(298)
    return dh_rxn


def get_kinetics_from_reactions(reactions: List[Reaction]) -> List[dict]:
    """
    Get kinetics from a list of RMG Reaction objects.

    Args:
        reactions (List[Reaction]): The RMG Reaction objects.

    Returns: List[dict]
        The kinetics as a dict of parameters.
    """
    kinetics_list = list()
    for rxn in reactions:
        print(f'rxn: {rxn}, kinetics: {rxn.kinetics}, comment: {rxn.comment}')
        kinetics_list.append({
            'kinetics': rxn.kinetics.__repr__(),
            'comment': rxn.comment,
            'A': rxn.kinetics.A.value if hasattr(rxn.kinetics, 'A') else None,
            'n': rxn.kinetics.n.value if hasattr(rxn.kinetics, 'n') else None,
            'Ea': rxn.kinetics.Ea.value_si * 0.001 if hasattr(rxn.kinetics, 'Ea') else None,  # kJ/mol
            'T_min': rxn.kinetics.Tmin.value_si if hasattr(rxn.kinetics, 'Tmin') and rxn.kinetics.Tmin is not None else None,
            'T_max': rxn.kinetics.Tmax.value_si if hasattr(rxn.kinetics, 'Tmax') and rxn.kinetics.Tmax is not None else None,
        })
    return kinetics_list


def loop_families(rmgdb: RMGDatabase,
                  reaction: Reaction,
                  ) -> List[Tuple[KineticsFamily, list]]:
    """
    Loop through kinetic families and return a list of tuples of (family, degenerate_reactions)
    corresponding to ``reaction``.

    Args:
        rmgdb (RMGDatabase): The RMG database instance.
        reaction (Reaction): The RMG Reaction object instance.

    Returns: List[Tuple['KineticsFamily', list]]
        Entries are tuples of a corresponding RMG KineticsFamily instance and a list of degenerate reactions.
    """
    reaction = reaction.copy()  # Use a copy to avoid changing atom order in the molecules by RMG.
    for spc in reaction.reactants + reaction.products:
        spc.generate_resonance_structures()
    fam_list = list()
    for family in rmgdb.kinetics.families.values():
        family.save_order = True
        degenerate_reactions = list()
        family_reactions_by_r = list()  # Family reactions for the specified reactants.
        family_reactions_by_rnp = list()  # Family reactions for the specified reactants and products.

        if len(reaction.reactants) == 1:
            for reactant0 in reaction.reactants[0].molecule:
                fam_rxn = family.generate_reactions(reactants=[reactant0],
                                                    products=reaction.products,
                                                    delete_labels=False,
                                                    )
                if fam_rxn:
                    family_reactions_by_r.extend(fam_rxn)
        elif len(reaction.reactants) == 2:
            for reactant0 in reaction.reactants[0].molecule:
                for reactant1 in reaction.reactants[1].molecule:
                    fam_rxn = family.generate_reactions(reactants=[reactant0, reactant1],
                                                        products=reaction.products,
                                                        delete_labels=False,
                                                        )
                    if fam_rxn:
                        family_reactions_by_r.extend(fam_rxn)
        elif len(reaction.reactants) == 3:
            for reactant0 in reaction.reactants[0].molecule:
                for reactant1 in reaction.reactants[1].molecule:
                    for reactant2 in reaction.reactants[2].molecule:
                        fam_rxn = family.generate_reactions(reactants=[reactant0, reactant1, reactant2],
                                                            products=reaction.products,
                                                            delete_labels=False,
                                                            )
                        if fam_rxn:
                            family_reactions_by_r.extend(fam_rxn)

        if len(reaction.products) == 1:
            for fam_rxn in family_reactions_by_r:
                for product0 in reaction.products[0].molecule:
                    if same_species_lists([product0], fam_rxn.products, save_order=True):
                        family_reactions_by_rnp.append(fam_rxn)
            degenerate_reactions = find_degenerate_reactions(rxn_list=family_reactions_by_rnp,
                                                             same_reactants=False,
                                                             kinetics_database=rmgdb.kinetics,
                                                             save_order=True
                                                             )
        elif len(reaction.products) == 2:
            for fam_rxn in family_reactions_by_r:
                for product0 in reaction.products[0].molecule:
                    for product1 in reaction.products[1].molecule:
                        if same_species_lists([product0, product1], fam_rxn.products, save_order=True):
                            family_reactions_by_rnp.append(fam_rxn)
            degenerate_reactions = find_degenerate_reactions(rxn_list=family_reactions_by_rnp,
                                                             same_reactants=False,
                                                             kinetics_database=rmgdb.kinetics,
                                                             save_order=True
                                                             )
        elif len(reaction.products) == 3:
            for fam_rxn in family_reactions_by_r:
                for product0 in reaction.products[0].molecule:
                    for product1 in reaction.products[1].molecule:
                        for product2 in reaction.products[2].molecule:
                            if same_species_lists([product0, product1, product2], fam_rxn.products, save_order=True):
                                family_reactions_by_rnp.append(fam_rxn)
            degenerate_reactions = find_degenerate_reactions(rxn_list=family_reactions_by_rnp,
                                                             same_reactants=False,
                                                             kinetics_database=rmgdb.kinetics,
                                                             save_order=True
                                                             )
        if degenerate_reactions:
            fam_list.append((family, degenerate_reactions))
    return fam_list


def load_rmg_database() -> RMGDatabase:
    """
    Load the RMG database.

    Returns:
        RMGDatabase: The loaded RMG database.
    """
    rmgdb = RMGDatabase()
    kinetics_libraries = read_yaml_file(path=os.path.join(os.path.dirname(__file__), 'libraries.yaml'))['kinetics']
    thermo_libraries = read_yaml_file(path=os.path.join(os.path.dirname(__file__), 'libraries.yaml'))['thermo']
    rmgdb.load_thermo(path=os.path.join(DB_PATH, 'thermo'), thermo_libraries=thermo_libraries, depository=True, surface=False)
    rmgdb.load_kinetics(path=os.path.join(DB_PATH, 'kinetics'),
                        reaction_libraries=kinetics_libraries,
                        kinetics_families='default',
                        kinetics_depositories=['training'])
    return rmgdb


if __name__ == '__main__':
    main()
