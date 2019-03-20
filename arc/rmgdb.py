#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import logging

from rmgpy import settings
from rmgpy.data.rmg import RMGDatabase
from rmgpy.reaction import isomorphic_species_lists
from rmgpy.data.kinetics.common import find_degenerate_reactions
from rmgpy.exceptions import KineticsError

from arc.arc_exceptions import InputError

##################################################################

# A module for loading the RMG-database

db_path = settings['database.directory']


def make_rmg_database_object():
    rmgdb = RMGDatabase()
    return rmgdb


def load_families_only(rmgdb, kinetics_families='default'):
    """
    A helper function for loading kinetic families from RMG's database
    """
    if kinetics_families not in ('default', 'all', 'none'):
        if not isinstance(kinetics_families, list):
            raise InputError("kineticsFamilies should be either 'default', 'all', 'none', or a list of names, e.g.,"
                             " ['H_Abstraction','R_Recombination'] or ['!Intra_Disproportionation'].")
    logging.debug('\n\nLoading only kinetic families from the RMG database...')
    rmgdb.load(
        path=db_path,
        thermoLibraries=list(),
        transportLibraries='none',
        reactionLibraries=list(),
        seedMechanisms=list(),
        kineticsFamilies=kinetics_families,
        kineticsDepositories=['training'],
        depository=False,
    )


def load_rmg_database(rmgdb, thermo_libraries=None, reaction_libraries=None, kinetics_families='default',
                      load_thermo_libs=True, load_kinetic_libs=True):
    """
    A helper function for loading the RMG database
    """
    thermo_libraries = thermo_libraries if thermo_libraries is not None else list()
    reaction_libraries = reaction_libraries if reaction_libraries is not None else list()
    if isinstance(thermo_libraries, str):
        thermo_libraries.replace(' ', '')
        thermo_libraries = [lib for lib in thermo_libraries.split(',')]
    if isinstance(reaction_libraries, (str, unicode)):
        reaction_libraries.replace(' ', '')
        reaction_libraries = [lib for lib in reaction_libraries.split(',')]
        reaction_libraries = [reaction_libraries]
    if kinetics_families not in ('default', 'all', 'none'):
        if not isinstance(kinetics_families, list):
            raise InputError("kineticsFamilies should be either 'default', 'all', 'none', or a list of names, e.g.,"
                             " ['H_Abstraction','R_Recombination'] or ['!Intra_Disproportionation'].")
    if not thermo_libraries:
        thermo_path = os.path.join(db_path, 'thermo', 'libraries')
        for thermo_library_path in os.listdir(thermo_path):
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
        reaction_libraries = os.listdir(kinetics_path)
        indices_to_pop = list()
        second_level_libraries = list()
        for i, library in enumerate(reaction_libraries):
            if not os.path.isfile(os.path.join(kinetics_path, library, 'reactions.py')):
                indices_to_pop.append(i)
                second_level_libraries.extend([library + '/' + second_level\
                                               for second_level in os.listdir(os.path.join(kinetics_path, library))])
        for i in reversed(range(len(reaction_libraries))):  # pop starting from the end, so other indices won't change
            if i in indices_to_pop:
                reaction_libraries.pop(i)
        reaction_libraries.extend(second_level_libraries)

    # set library to be represented by a string rather than a unicode,
    # this might not be needed after a full migration to Py3
    thermo_libraries = [str(lib) for lib in thermo_libraries]
    reaction_libraries = [str(lib) for lib in reaction_libraries]
    if not load_kinetic_libs:
        reaction_libraries = list()
    if not load_thermo_libs:
        thermo_libraries = list()
    # reaction_libraries = list()  # empty library list for debugging
    logging.info('\n\nLoading the RMG database...')

    rmgdb.load(
        path=db_path,
        thermoLibraries=thermo_libraries,
        transportLibraries=['PrimaryTransportLibrary', 'NOx2018', 'GRI-Mech'],
        reactionLibraries=reaction_libraries,
        seedMechanisms=list(),
        kineticsFamilies=kinetics_families,
        kineticsDepositories=['training'],
        depository=False,
    )
    for family in rmgdb.kinetics.families.values():
        try:
            family.addKineticsRulesFromTrainingSet(thermoDatabase=rmgdb.thermo)
        except KineticsError:
            logging.info('Could not train family {0}'.format(family))
        else:
            family.fillKineticsRulesByAveragingUp(verbose=False)
    logging.info('\n\n')


def determine_reaction_family(rmgdb, reaction):
    """
    Determine the RMG kinetic family for a given ARCReaction object
    Returns None if no family found or more than one family found
    """
    fam_list = loop_families(rmgdb=rmgdb, reaction=reaction)
    families = [fam_l[0] for fam_l in fam_list]
    if len(set(families)) == 1:
        return families[0], families[0].ownReverse
    else:
        return None, False


def loop_families(rmgdb, reaction):
    """
    Loop through kinetic families and return a list of tuples of (family, degenerate_reactions)
    `reaction` is an RMG Reaction object
    Returns a list of (family, degenerate_reactions) tuples
    """
    fam_list = list()
    for family in rmgdb.kinetics.families.values():
        degenerate_reactions = list()
        family_reactions_by_r = list()  # family reactions for the specified reactants
        family_reactions_by_rnp = list()  # family reactions for the specified reactants and products

        if len(reaction.reactants) == 1:
            for reactant0 in reaction.reactants[0].molecule:
                fam_rxn = family.generateReactions(reactants=[reactant0],
                                                   products=reaction.products)
                if fam_rxn:
                    family_reactions_by_r.extend(fam_rxn)
        elif len(reaction.reactants) == 2:
            for reactant0 in reaction.reactants[0].molecule:
                for reactant1 in reaction.reactants[1].molecule:
                    fam_rxn = family.generateReactions(reactants=[reactant0, reactant1],
                                                       products=reaction.products)
                    if fam_rxn:
                        family_reactions_by_r.extend(fam_rxn)
        elif len(reaction.reactants) == 3:
            for reactant0 in reaction.reactants[0].molecule:
                for reactant1 in reaction.reactants[1].molecule:
                    for reactant2 in reaction.reactants[2].molecule:
                        fam_rxn = family.generateReactions(reactants=[reactant0, reactant1, reactant2],
                                                           products=reaction.products)
                        if fam_rxn:
                            family_reactions_by_r.extend(fam_rxn)

        if len(reaction.products) == 1:
            for fam_rxn in family_reactions_by_r:
                for product0 in reaction.products[0].molecule:
                    if isomorphic_species_lists([product0], fam_rxn.products):
                        family_reactions_by_rnp.append(fam_rxn)
            degenerate_reactions = find_degenerate_reactions(rxn_list=family_reactions_by_rnp,
                                                             same_reactants=False,
                                                             kinetics_database=rmgdb.kinetics)
        elif len(reaction.products) == 2:
            for fam_rxn in family_reactions_by_r:
                for product0 in reaction.products[0].molecule:
                    for product1 in reaction.products[1].molecule:
                        if isomorphic_species_lists([product0, product1], fam_rxn.products):
                            family_reactions_by_rnp.append(fam_rxn)
            degenerate_reactions = find_degenerate_reactions(rxn_list=family_reactions_by_rnp,
                                                             same_reactants=False,
                                                             kinetics_database=rmgdb.kinetics)
        elif len(reaction.products) == 3:
            for fam_rxn in family_reactions_by_r:
                for product0 in reaction.products[0].molecule:
                    for product1 in reaction.products[1].molecule:
                        for product2 in reaction.products[2].molecule:
                            if isomorphic_species_lists([product0, product1, product2], fam_rxn.products):
                                family_reactions_by_rnp.append(fam_rxn)
            degenerate_reactions = find_degenerate_reactions(rxn_list=family_reactions_by_rnp,
                                                             same_reactants=False,
                                                             kinetics_database=rmgdb.kinetics)
        if degenerate_reactions:
            fam_list.append((family, degenerate_reactions))
    return fam_list


def determine_rmg_kinetics(rmgdb, reaction, dh_rxn298):
    """
    Determine kinetics for `reaction` (an RMG Reaction object) from RMG's database, if possible
    Assigns a list of all matching entries from both libraries and families
    Returns a list of all matching RMG reactions (both libraries and families) with a populated .kinetics attribute
    `rmgdb` is the RMG database instance
    `reaction` is an RMG Reaction object
    `dh_rxn298` is the heat of reaction at 298 K in J / mol
    """
    rmg_reactions = list()
    # Libraries:
    for library in rmgdb.kinetics.libraries.values():
        library_reactions = library.getLibraryReactions()
        for library_reaction in library_reactions:
            if reaction.isIsomorphic(library_reaction):
                library_reaction.comment = str('Library: {0}'.format(library.label))
                rmg_reactions.append(library_reaction)
                break
    # Families:
    fam_list = loop_families(rmgdb, reaction)
    for family, degenerate_reactions in fam_list:
        for deg_rxn in degenerate_reactions:
            template = family.retrieveTemplate(deg_rxn.template)
            kinetics = family.estimateKineticsUsingRateRules(template)[0]
            kinetics.changeRate(deg_rxn.degeneracy)
            kinetics = kinetics.toArrhenius(dh_rxn298)  # Convert ArrheniusEP to Arrhenius using the dHrxn at 298K
            deg_rxn.kinetics = kinetics
            deg_rxn.comment = str('Family: {0}'.format(deg_rxn.family))
            deg_rxn.reactants = reaction.reactants
            deg_rxn.products = reaction.products
        rmg_reactions.extend(degenerate_reactions)
    return rmg_reactions
