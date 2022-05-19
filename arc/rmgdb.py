"""
A module for working with the RMG database.
"""

import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from rmgpy import settings as rmg_settings
from rmgpy.data.kinetics.common import find_degenerate_reactions
from rmgpy.data.rmg import RMGDatabase
from rmgpy.exceptions import KineticsError
from rmgpy.reaction import same_species_lists, Reaction

from arc.common import get_logger
from arc.exceptions import InputError

if TYPE_CHECKING:
    from rmgpy.data.kinetics.family import KineticsFamily
    from arc.reaction import ARCReaction


logger = get_logger()

db_path = rmg_settings['database.directory']

rmg_database_instance = None


def make_rmg_database_object() -> RMGDatabase:
    """
    Make a clean RMGDatabase object.

    Returns: RMGDatabase
        A clean RMG database object.
    """
    global rmg_database_instance
    rmg_database_instance = rmg_database_instance or RMGDatabase()
    return rmg_database_instance


def load_families_only(rmgdb: RMGDatabase,
                       kinetics_families: Union[str, list] = 'default'):
    """
    A helper function for loading kinetic families from RMG's database.

    Args:
        rmgdb (RMGDatabase): The RMG database instance.
        kinetics_families (Union[str, list], optional): Specific kinetics families to load.
    """
    if kinetics_families not in ('default', 'all', 'none') and not isinstance(kinetics_families, list):
        raise InputError(f"kinetics families should be either 'default', 'all', 'none', or a list of names, e.g.,"
                         f" ['H_Abstraction','R_Recombination'] or ['!Intra_Disproportionation']. "
                         f"Got:\n{kinetics_families}")
    if rmgdb.kinetics is None or rmgdb.kinetics.families is None:
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


def load_rmg_database(rmgdb: RMGDatabase,
                      thermo_libraries: list = None,
                      reaction_libraries: list = None,
                      kinetics_families: Union[str, list] = 'default',
                      load_thermo_libs: bool = True,
                      load_kinetic_libs: bool = True,
                      include_nist: bool = False,
                      ) -> None:
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
    thermo_libraries = thermo_libraries or list()
    reaction_libraries = reaction_libraries or list()
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
            # Avoid reading .DS_store files for compatibility with macOS
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
        libraries_path = os.path.join(db_path, 'kinetics', 'libraries')
        for path, folders, _ in os.walk(libraries_path):
            if 'Surface' in path:
                continue
            for folder in folders:
                if os.path.isfile(os.path.join(path, folder, 'reactions.py')):
                    prefix = path.split(libraries_path)[1]
                    library = os.path.join(prefix, folder)
                    library = library[1:] if library[0] == '/' else library
                    reaction_libraries.append(library)

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
            logger.info(f'Could not train family {family}')
        else:
            family.fill_rules_by_averaging_up(verbose=False)
        family.save_order = True
    logger.info('\n\n')


def determine_family(reaction: 'ARCReaction',
                     db: Optional[RMGDatabase] = None,
                     ):
    """
    Determine the RMG reaction family for an ARC reaction.
    A wrapper for ARCReaction.determine_family().
    This wrapper is useful because it makes a new instance of the rmgdb if needed.

    Args:
        reaction ('ARCReaction'): An ARCReaction object instance.
        db (RMGDatabase, optional): The RMG database instance.
    """
    global rmg_database_instance
    if reaction.family is None:
        db = db or rmg_database_instance
        if db is None:
            db = make_rmg_database_object()
            load_families_only(db)
        if reaction.rmg_reaction is None:
            reaction.rmg_reaction_from_arc_species()
        reaction.determine_family(db)


def determine_reaction_family(rmgdb: RMGDatabase,
                              reaction: Reaction,
                              save_order: bool = True,
                              ) -> Tuple[Optional['KineticsFamily'], bool]:
    """
    Determine the RMG kinetic family for a given ``ARCReaction`` object instance.
    Returns ``None`` if no family found or more than one family found.

    Args:
        rmgdb (RMGDatabase): The RMG database instance.
        reaction (Reaction): The RMG Reaction object.
        save_order (bool, optional): Whether to retain atomic order of the RMG ``reaction`` object instance.

    Returns: Tuple[Optional[KineticsFamily], bool]
        - The corresponding RMG reaction's family. ``None`` if no family was found or more than one family were found.
        - Whether the family is its own reverse. ``True`` if it is.
    """
    for rmg_spc in reaction.reactants + reaction.products:
        rmg_spc.generate_resonance_structures(keep_isomorphic=False, filter_structures=True, save_order=True)
    fam_list = loop_families(rmgdb=rmgdb, reaction=reaction)
    families = [fam_l[0] for fam_l in fam_list]
    if len(set(families)) == 1:
        family = families[0]
        if save_order:
            family.save_order = True
        return family, family.own_reverse
    else:
        return None, False


def loop_families(rmgdb: RMGDatabase,
                  reaction: Reaction,
                  ) -> List[Tuple['KineticsFamily', list]]:
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
        spc.generate_resonance_structures(save_order=True)
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


def determine_rmg_kinetics(rmgdb: RMGDatabase,
                           reaction: Reaction,
                           dh_rxn298: float,
                           ) -> List[Reaction]:
    """
    Determine kinetics for `reaction` (an RMG Reaction object) from RMG's database, if possible.
    Assigns a list of all matching entries from both libraries and families.

    Args:
        rmgdb (RMGDatabase): The RMG database instance.
        reaction (Reaction): The RMG Reaction object.
        dh_rxn298 (float): The heat of reaction at 298 K in J/mol.

    Returns: List[Reaction]
        All matching RMG reactions (both libraries and families) with a populated ``.kinetics`` attribute.
    """
    reaction = reaction.copy()
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


def get_family(rmgdb: RMGDatabase,
               label: Optional[str],
               ) -> Optional['KineticsFamily']:
    """
    Get a KineticsFamily object instance corresponding to the input family ``label``.

    Args:
        rmgdb (RMGDatabase): The RMG database instance.
        label (str): The label of the desired family.

    Returns:
        Optional['KineticsFamily']: The desired family object instance.
    """
    if label is None:
        return None
    label = label.lower()
    for family_label, family in rmgdb.kinetics.families.items():
        if family_label.lower() == label:
            family.save_order = True
            return family
    return None
