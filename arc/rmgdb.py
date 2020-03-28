#!/usr/bin/env python3
# encoding: utf-8

"""
A module for working with the RMG database.
"""

import os

from rmgpy import settings
from rmgpy.data.kinetics.common import find_degenerate_reactions
from rmgpy.data.rmg import RMGDatabase
from rmgpy.exceptions import KineticsError
from rmgpy.reaction import same_species_lists, Reaction

from arc.common import get_logger
from arc.exceptions import InputError


logger = get_logger()

db_path = settings['database.directory']


def make_rmg_database_object():
    """
    Make a clean RMGDatabase object.

    Returns:
        RMGDatabase: A clean RMG database object.
    """
    rmgdb = RMGDatabase()
    return rmgdb


def load_families_only(rmgdb, kinetics_families='default'):
    """
    A helper function for loading kinetic families from RMG's database.

    Args:
        rmgdb (RMGDatabase): The RMG database instance.
        kinetics_families (str, optional): Specific kinetics families to load.
    """
    if kinetics_families not in ('default', 'all', 'none') and not isinstance(kinetics_families, list):
        raise InputError(f"kinetics families should be either 'default', 'all', 'none', or a list of names, e.g., "
                         f"['H_Abstraction', 'R_Recombination'] or ['!Intra_Disproportionation']. "
                         f"Got:\n{kinetics_families}")
    logger.debug('\n\nLoading only kinetic families from the RMG database...')
    rmgdb.load(
        path=db_path,
        thermo_libraries=list(),
        transport_libraries='none',
        reaction_libraries=list(),
        seed_mechanisms=list(),
        kinetics_families=kinetics_families,
        kinetics_depositories=['training'],
        depository=False,
    )


def load_rmg_database(rmgdb, thermo_libraries=None, reaction_libraries=None, kinetics_families='default',
                      load_thermo_libs=True, load_kinetic_libs=True, include_nist=False):
    """
    A helper function for loading the RMG database.

    Args:
        rmgdb (RMGDatabase): The RMG database instance.
        thermo_libraries (list, optional): Specific thermodynamic libraries to load (``None`` will load all).
        reaction_libraries (list, optional): Specific kinetics libraries to load (``None`` will load all).
        kinetics_families (list, str, optional): Specific kinetics families to load
                                                 (either a list or 'default', 'all', 'none')
        load_thermo_libs (bool, optional): Whether to load thermodynamic libraries, ``True`` to load.
        load_kinetic_libs (bool, optional): Whether to load kinetics libraries, ``True`` to load.
        include_nist (bool, optional): Whether to include the NIST kinetics libraries,
                                      ``True`` to include, default is ``False``
    """
    thermo_libraries = thermo_libraries if thermo_libraries is not None else list()
    reaction_libraries = reaction_libraries if reaction_libraries is not None else list()
    if isinstance(thermo_libraries, str):
        thermo_libraries.replace(' ', '')
        thermo_libraries = [lib for lib in thermo_libraries.split(',')]
    if isinstance(reaction_libraries, str):
        reaction_libraries.replace(' ', '')
        reaction_libraries = [lib for lib in reaction_libraries.split(',')]
        reaction_libraries = [reaction_libraries]
    if kinetics_families not in ('default', 'all', 'none'):
        if not isinstance(kinetics_families, list):
            raise InputError("kinetics_families should be either 'default', 'all', 'none', or a list of names, e.g.,"
                             " ['H_Abstraction','R_Recombination'] or ['!Intra_Disproportionation'].")
    if not thermo_libraries:
        thermo_path = os.path.join(db_path, 'thermo', 'libraries')
        for thermo_library_path in os.listdir(thermo_path):
            # Avoid reading .DS_store files for compatibility with Mac OS
            if not thermo_library_path.startswith('.'):
                thermo_library, _ = os.path.splitext(os.path.basename(thermo_library_path))
                thermo_libraries.append(thermo_library)
        # prioritize libraries
        thermo_priority = ['BurkeH2O2', 'thermo_DFT_CCSDTF12_BAC', 'DFT_QCI_thermo', 'Klippenstein_Glarborg2016',
                           'primaryThermoLibrary', 'primaryNS', 'NitrogenCurran', 'NOx2018', 'FFCM1(-)',
                           'SulfurLibrary', 'SulfurGlarborgH2S']
        indices_to_pop = []
        for i, lib in enumerate(thermo_libraries):
            if lib in thermo_priority:
                indices_to_pop.append(i)
        for i in reversed(range(len(thermo_libraries))):  # pop starting from the end, so other indices won't change
            if i in indices_to_pop:
                thermo_libraries.pop(i)
        thermo_libraries = thermo_priority + thermo_libraries

    if not reaction_libraries:
        kinetics_path = os.path.join(db_path, 'kinetics', 'libraries')
        # Avoid reading .DS_store files for compatibility with Mac OS
        reaction_libraries = [library for library in os.listdir(kinetics_path) if not library.startswith('.')]
        indices_to_pop = list()
        second_level_libraries = list()
        for i, library in enumerate(reaction_libraries):
            if not os.path.isfile(os.path.join(kinetics_path, library, 'reactions.py')) and library != 'Surface':
                indices_to_pop.append(i)
                # Avoid reading .DS_store files for compatibility with Mac OS (`if not second_level.startswith('.')`)
                second_level_libraries.extend([library + '/' + second_level
                                               for second_level in os.listdir(os.path.join(kinetics_path, library))
                                               if not second_level.startswith('.')])
        for i in reversed(range(len(reaction_libraries))):  # pop starting from the end, so other indices won't change
            if i in indices_to_pop or reaction_libraries[i] == 'Surface':
                reaction_libraries.pop(i)
        reaction_libraries.extend(second_level_libraries)

    if not load_kinetic_libs:
        reaction_libraries = list()
    if not load_thermo_libs:
        thermo_libraries = list()
    # reaction_libraries = list()  # empty library list for debugging
    logger.info('\n\nLoading the RMG database...')

    kinetics_depositories = ['training', 'NIST'] if include_nist else ['training']
    rmgdb.load(
        path=db_path,
        thermo_libraries=thermo_libraries,
        transport_libraries=['PrimaryTransportLibrary', 'NOx2018', 'GRI-Mech'],
        reaction_libraries=reaction_libraries,
        seed_mechanisms=list(),
        kinetics_families=kinetics_families,
        kinetics_depositories=kinetics_depositories,
        depository=False,
    )
    for family in rmgdb.kinetics.families.values():
        try:
            family.add_rules_from_training(thermo_database=rmgdb.thermo)
        except KineticsError:
            logger.info('Could not train family {0}'.format(family))
        else:
            family.fill_rules_by_averaging_up(verbose=False)
    logger.info('\n\n')


def determine_reaction_family(rmgdb, reaction):
    """
    Determine the RMG kinetic family for a given ARCReaction object.
    Returns None if no family found or more than one family found.

    Args:
        rmgdb (RMGDatabase): The RMG database instance.
        reaction (Reaction): The RMG Reaction object.

    Returns:
        str: The corresponding RMG reaction's family. None if no family was found or more than one family were found.
    Returns:
         bool: Whether the family is its own reverse.
    """
    fam_list = loop_families(rmgdb=rmgdb, reaction=reaction)
    families = [fam_l[0] for fam_l in fam_list]
    if len(set(families)) == 1:
        return families[0], families[0].own_reverse
    else:
        return None, False


def loop_families(rmgdb, reaction):
    """
    Loop through kinetic families and return a list of tuples of (family, degenerate_reactions)
    `reaction` is an RMG Reaction object.
    Returns a list of (family, degenerate_reactions) tuples.

    Args:
        rmgdb (RMGDatabase): The RMG database instance.
        reaction (Reaction): The RMG Reaction object.

    Returns:
        list: Entries are corresponding RMG reaction families.
    """
    reaction = reaction.copy()  # use a copy to avoid changing atom order in the molecules by RMG
    fam_list = list()
    for family in rmgdb.kinetics.families.values():
        degenerate_reactions = list()
        family_reactions_by_r = list()  # family reactions for the specified reactants
        family_reactions_by_rnp = list()  # family reactions for the specified reactants and products

        if len(reaction.reactants) == 1:
            for reactant0 in reaction.reactants[0].molecule:
                fam_rxn = family.generate_reactions(reactants=[reactant0],
                                                    products=reaction.products)
                if fam_rxn:
                    family_reactions_by_r.extend(fam_rxn)
        elif len(reaction.reactants) == 2:
            for reactant0 in reaction.reactants[0].molecule:
                for reactant1 in reaction.reactants[1].molecule:
                    fam_rxn = family.generate_reactions(reactants=[reactant0, reactant1],
                                                        products=reaction.products)
                    if fam_rxn:
                        family_reactions_by_r.extend(fam_rxn)
        elif len(reaction.reactants) == 3:
            for reactant0 in reaction.reactants[0].molecule:
                for reactant1 in reaction.reactants[1].molecule:
                    for reactant2 in reaction.reactants[2].molecule:
                        fam_rxn = family.generate_reactions(reactants=[reactant0, reactant1, reactant2],
                                                            products=reaction.products)
                        if fam_rxn:
                            family_reactions_by_r.extend(fam_rxn)

        if len(reaction.products) == 1:
            for fam_rxn in family_reactions_by_r:
                for product0 in reaction.products[0].molecule:
                    if same_species_lists([product0], fam_rxn.products):
                        family_reactions_by_rnp.append(fam_rxn)
            degenerate_reactions = find_degenerate_reactions(rxn_list=family_reactions_by_rnp,
                                                             same_reactants=False,
                                                             kinetics_database=rmgdb.kinetics)
        elif len(reaction.products) == 2:
            for fam_rxn in family_reactions_by_r:
                for product0 in reaction.products[0].molecule:
                    for product1 in reaction.products[1].molecule:
                        if same_species_lists([product0, product1], fam_rxn.products):
                            family_reactions_by_rnp.append(fam_rxn)
            degenerate_reactions = find_degenerate_reactions(rxn_list=family_reactions_by_rnp,
                                                             same_reactants=False,
                                                             kinetics_database=rmgdb.kinetics)
        elif len(reaction.products) == 3:
            for fam_rxn in family_reactions_by_r:
                for product0 in reaction.products[0].molecule:
                    for product1 in reaction.products[1].molecule:
                        for product2 in reaction.products[2].molecule:
                            if same_species_lists([product0, product1, product2], fam_rxn.products):
                                family_reactions_by_rnp.append(fam_rxn)
            degenerate_reactions = find_degenerate_reactions(rxn_list=family_reactions_by_rnp,
                                                             same_reactants=False,
                                                             kinetics_database=rmgdb.kinetics)
        if degenerate_reactions:
            fam_list.append((family, degenerate_reactions))
    return fam_list


def determine_rmg_kinetics(rmgdb, reaction, dh_rxn298):
    """
    Determine kinetics for `reaction` (an RMG Reaction object) from RMG's database, if possible.
    Assigns a list of all matching entries from both libraries and families.
    Returns a list of all matching RMG reactions (both libraries and families) with a populated .kinetics attribute
    `rmgdb` is the RMG database instance
    `reaction` is an RMG Reaction object
    `dh_rxn298` is the heat of reaction at 298 K in J / mol
    """
    reaction = reaction.copy()  # use a copy to avoid changing atom order in the molecules by RMG
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
    fam_list = loop_families(rmgdb, reaction)
    for family, degenerate_reactions in fam_list:
        for deg_rxn in degenerate_reactions:
            template = family.retrieve_template(deg_rxn.template)
            kinetics = family.estimate_kinetics_using_rate_rules(template)[0]
            kinetics.change_rate(deg_rxn.degeneracy)
            kinetics = kinetics.to_arrhenius(dh_rxn298)  # Convert ArrheniusEP to Arrhenius using the dHrxn at 298K
            deg_rxn.kinetics = kinetics
            deg_rxn.comment = f'Family: {deg_rxn.family}'
            deg_rxn.reactants = reaction.reactants
            deg_rxn.products = reaction.products
        rmg_reactions.extend(degenerate_reactions)
    worked_through_nist_fams = []
    # NIST:
    for family, degenerate_reactions in fam_list:
        if family not in worked_through_nist_fams:
            worked_through_nist_fams.append(family)
            for depo in family.depositories:
                if 'NIST' in depo.label:
                    for entry in depo.entries.values():
                        rxn = entry.item
                        rxn.kinetics = entry.data
                        rxn.comment = f'NIST: {entry.index}'
                        if entry.reference is not None:
                            rxn.comment += f'{entry.reference.authors[0]} {entry.reference.year}'
                        rmg_reactions.append(rxn)
    return rmg_reactions
