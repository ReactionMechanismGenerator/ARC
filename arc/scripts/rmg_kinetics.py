#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to run RMG and get kinetic rate coefficients for reactions.

Output units (per entry returned by ``get_kinetics_from_reactions``):
    - ``A``:    cm^3/(mol*s) for bimolecular reactions, s^-1 for unimolecular
                (3-body: cm^6/(mol^2*s)). Reported in the units stored on the
                Arrhenius object after the SI->cm conversion below.
    - ``n``:    dimensionless temperature exponent.
    - ``Ea``:   kJ/mol (converted from SI J/mol).
    - ``T_min``, ``T_max``: K.
"""

import copy
import os
import sys
from typing import Dict, List, Optional, Tuple

# Make ``from common import ...`` work no matter how this script is invoked
# (e.g. ``python /abs/path/to/rmg_kinetics.py``, ``cd elsewhere && python ...``).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import parse_command_line_arguments, read_yaml_file, save_yaml_file

from rmgpy.data.kinetics.common import find_degenerate_reactions
from rmgpy.data.kinetics.family import KineticsFamily
from rmgpy.data.rmg import RMGDatabase
from rmgpy import settings as rmg_settings
from rmgpy.kinetics import Arrhenius, ArrheniusEP, KineticsModel
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
    output_file = args.output or input_file
    reaction_list = read_yaml_file(path=input_file)
    if not isinstance(reaction_list, list):
        raise ValueError(f'The content of {input_file} must be a list, got {reaction_list} which is a {type(reaction_list)}')
    result = get_rate_coefficients(reaction_list)
    save_yaml_file(path=output_file, content=result)


def get_rate_coefficients(reaction_list: List[Dict]) -> List[Dict]:
    """
    Get rate coefficients for a list of reactions.

    Args:
        reaction_list (list[dict]): The list of reactions.

    Returns:
        list[dict]: The list of reactions with rate coefficients.
    """
    print('Loading RMG database...')
    rmgdb = load_rmg_database()
    for i in range(len(reaction_list)):
        rxn = Reaction(reactants=[Species().from_adjacency_list(adjlist) for adjlist in reaction_list[i]['reactants']],
                       products=[Species().from_adjacency_list(adjlist) for adjlist in reaction_list[i]['products']])
        reaction_list[i]['kinetics'] = determine_rmg_kinetics(rmgdb=rmgdb, reaction=rxn, dh_rxn298=reaction_list[i]['dh_rxn298'],
                                                              family=reaction_list[i]['family'] if 'family' in reaction_list[i] else None)
    return reaction_list


def get_a_factor_si_to_cm(num_reactants: int) -> float:
    """
    Get the factor that converts an Arrhenius A-factor from SI (m-based) units to
    cm-based units, based on the reaction molecularity.

    Args:
        num_reactants (int): The number of reactants.

    Returns:
        float: 1.0 for unimolecular (s^-1), 1e6 for bimolecular (m^3->cm^3),
               1e12 for termolecular (m^6->cm^6). Defaults to 1.0 otherwise.
    """
    return {1: 1.0, 2: 1e6, 3: 1e12}.get(num_reactants, 1.0)


def scale_kinetics_by_degeneracy(kinetics: KineticsModel,
                                 degeneracy: float,
                                 ) -> KineticsModel:
    """
    Scale Arrhenius-type kinetics in place by the reaction-path degeneracy.

    Non-Arrhenius forms (e.g. Chebyshev, PDepArrhenius) are returned unchanged,
    since ``change_rate`` would otherwise corrupt their parameters.

    Args:
        kinetics (KineticsModel): An RMG kinetics object.
        degeneracy (float): The reaction-path degeneracy to scale by.

    Returns:
        KineticsModel: The (possibly scaled) kinetics object.
    """
    if isinstance(kinetics, (Arrhenius, ArrheniusEP)):
        kinetics.change_rate(degeneracy)
    return kinetics


def determine_rmg_kinetics(rmgdb: RMGDatabase,
                           reaction: Reaction,
                           dh_rxn298: Optional[float] = None,
                           family: Optional[str] = None,
                           ) -> List[Dict]:
    """
    Determine kinetics for `reaction` (an RMG Reaction object) from RMG's database, if possible.
    Assigns a list of all matching entries from both libraries and families.

    Note:
        Family entries originating from the training set are intentionally filtered out
        (an empty returned list therefore means "no matching libraries and only training-set
        family hits", not necessarily "no match at all"). Database kinetics are deep-copied
        before any in-place mutation (degeneracy scaling, unit conversion) so the loaded
        ``rmgdb`` instance remains unchanged across calls.

    Args:
        rmgdb (RMGDatabase): The RMG database instance.
        reaction (Reaction): The RMG Reaction object.
        dh_rxn298 (float, optional): The heat of reaction at 298 K in J/mol.
        family (str, optional): The RMG family label.

    Returns: list[dict]
        All matching RMG reactions kinetics (both libraries and families) as a dict of parameters.
        Empty list if nothing matched (or only training-set entries matched).
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
    # Families:
    a_factor = get_a_factor_si_to_cm(len(reaction.reactants))
    fam_list = loop_families(rmgdb, reaction)
    dh_rxn298 = dh_rxn298 or get_dh_rxn298(rmgdb=rmgdb, reaction=reaction)  # J/mol
    for family, degenerate_reactions in fam_list:
        for deg_rxn in degenerate_reactions:
            kinetics_list = family.get_kinetics(reaction=deg_rxn, template_labels=deg_rxn.template, degeneracy=deg_rxn.degeneracy)
            for kinetics_detailes in kinetics_list:
                # Deep-copy before mutating so the database object isn't double-scaled
                # if the same family rule is queried again for another reaction.
                kinetics = copy.deepcopy(kinetics_detailes[0])
                kinetics = scale_kinetics_by_degeneracy(kinetics, deg_rxn.degeneracy)
                if hasattr(kinetics, 'to_arrhenius'):
                    kinetics = kinetics.to_arrhenius(dh_rxn298)  # Convert ArrheniusEP to Arrhenius
                if a_factor != 1.0 and isinstance(kinetics, Arrhenius):
                    kinetics.A.value_si = kinetics.A.value_si * a_factor
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


def get_kinetics_from_reactions(reactions: List[Reaction]) -> List[Dict]:
    """
    Get kinetics from a list of RMG Reaction objects.

    Args:
        reactions (list[Reaction]): The RMG Reaction objects.

    Returns: list[dict]
        The kinetics as a dict of parameters.
    """
    kinetics_list = list()
    for rxn in reactions:
        try:
            rxn_repr = str(rxn)
        except (TypeError, AttributeError):
            rxn_repr = '<reaction without reactants/products labels>'
        print(f'rxn: {rxn_repr}, kinetics: {rxn.kinetics}, comment: {rxn.comment}', file=sys.stderr)
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

    Returns: list[tuple['KineticsFamily', list]]
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
