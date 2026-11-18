"""
Heuristic TS guesses for the QA (quaternary ammonium + OH_d1) reaction families:
qa_sn2, qa_ring_opening, and qa_hofmann.

The functions here are called by the heuristics TS search adapter (see heuristics.py)
via QA_TS_GUESS_FUNCTIONS.
"""

import copy
import functools
import itertools
import json
import math
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from arc.common import determine_top_group_indices, get_angle_in_180_range, get_logger, is_angle_linear
from arc.family import get_reaction_family_products
from arc.imports import settings
from arc.job.adapters.ts.heuristics import (generate_dihedral_variants, get_matching_dihedrals,
                                            get_multiple_bond_pairs, get_neighbors_by_electronegativity,
                                            load_family_parameters, setup_zmat_indices, stretch_ab_bond)
from arc.molecule.molecule import Molecule
from arc.settings.settings import UMA_LATEST_MODEL
from arc.species.conformers import get_number_of_chiral_centers
from arc.species.converter import (add_atom_to_xyz_using_internal_coords, compare_zmats, rdkit_conf_from_mol,
                                   sort_xyz_using_indices, sorted_distances_of_atom, to_rdkit_mol,
                                   xyz_to_xyz_file_format, zmat_to_xyz)
from arc.species.species import ARCSpecies, colliding_atoms
from arc.species.vectors import (apply_rodrigues_rotation, calculate_angle, calculate_dihedral_angle,
                                 calculate_distance, unit_vector)
from arc.species.zmat import xyz_to_zmat

if TYPE_CHECKING:
    from arc.reaction import ARCReaction


logger = get_logger()


def qa_ts_guesses(reaction: 'ARCReaction',
                  family: str,
                  parameters_file: str,
                  no_products_message: str,
                  extract_indices: Callable[['ARCReaction', dict], tuple['ARCSpecies', 'ARCSpecies', dict, dict]],
                  build_guesses: Callable[..., tuple[list[dict], list[dict]]],
                  site_filter: Callable[['ARCReaction', list[dict]], list[dict]] | None = None,
                  ) -> tuple[list[dict], list[str]]:
    """
    Generate TS guesses for a reaction of a QA (quaternary ammonium + OH_d1) TS-guess family.

    The family products of the reaction are generated, each product dictionary (a reactive site) is converted
    into atom indices, and the family builds the TS guesses of each site in turn.
    The QA reactant coordinates are made consistent with the stereochemistry of its SMILES first
    (see get_stereo_consistent_xyz). The Z-matrices of all accepted guesses are shared across the sites,
    so a guess duplicating an earlier one (of any site) is rejected. The guesses are finally pre-filtered by a UMA single point.

    Args:
        reaction (ARCReaction): The reaction to process.
        family (str): The ARC family label.
        parameters_file (str): The family parameters YAML file name in ARC's data directory.
        no_products_message (str): The ValueError message raised if the reaction has no products of the family.
        extract_indices (Callable): ``(reaction, product_dict) -> (main_reactant, oh_d1, initial_xyz, xyz_indices)``,
                                    the reactants and the atom indices of one reactive site.
        build_guesses (Callable): ``(parameters, main_reactant, oh_d1, initial_xyz, xyz_indices, zmats_total,
                                  threshold, multiple_bond_pairs) -> (xyz_guesses, zmats_total)``,
                                  the TS guesses of one reactive site.
        site_filter (Callable, optional): ``(reaction, product_dicts) -> product_dicts``,
                                          removes equivalent reactive sites before the guesses are built.

    Returns:
        Tuple containing:
            - list[dict]: Cartesian coordinates of TS guesses.
            - list[str]: Reaction family label for each guess.
    """
    xyz_guesses_total, zmats_total, reaction_families = [], [], []
    user_reaction, reaction = reaction, get_qa_first_reaction(reaction)
    # Reuse the reaction's product dictionaries if they were already generated for this family
    # (they are not carried over to the QA-first copy, whose atom indices may differ)
    product_dicts = reaction._product_dicts
    if not product_dicts or any(pd.get('family') != family for pd in product_dicts):
        product_dicts = get_reaction_family_products(
            rxn=reaction,
            rmg_family_set="default",
            consider_rmg_families=False,
            consider_arc_families=True,
        )
    product_dicts = [pd for pd in product_dicts if pd.get('family') == family]
    if not product_dicts:
        raise ValueError(no_products_message)
    if all(pd.get('discovered_in_reverse', False) for pd in product_dicts):
        raise ValueError(f'Reaction {reaction.label} matches the {family} family only in reverse. QA TS guesses are '
                         f'generated for reactions written in the degradation direction '
                         f'(QA reactant + OH_d1 <=> products); please write the reaction in that direction.')

    parameters = load_family_parameters(parameters_file)
    if site_filter is not None:
        product_dicts = site_filter(reaction, product_dicts)

    # The species copies returned by extract_indices do not keep the SMILES, so it is taken from the user's reaction
    qa_smiles = next((spc.mol.smiles for spc in user_reaction.r_species if spc.mol.get_net_charge() >= 0), None)
    stereo_xyz = dict()  # the stereo-consistent QA reactant coordinates, computed once
    for product_dict in product_dicts:
        main_reactant, oh_d1, initial_xyz, xyz_indices = extract_indices(reaction, product_dict)
        if 'xyz' not in stereo_xyz:
            stereo_xyz['xyz'] = get_stereo_consistent_xyz(qa_smiles, main_reactant.mol, initial_xyz,
                                                          label=main_reactant.label)
        initial_xyz = stereo_xyz['xyz']
        threshold = 0.8
        multiple_bond_pairs = get_multiple_bond_pairs(main_reactant)
        xyz_guesses, zmats_total = build_guesses(
            parameters, main_reactant, oh_d1, initial_xyz, xyz_indices, zmats_total,
            threshold, multiple_bond_pairs,
        )
        if xyz_guesses:
            xyz_guesses = [order_qa_guess_as_reactants(xyz, user_reaction, len(main_reactant.mol.atoms),
                                                       xyz_indices['atom_order'])
                           for xyz in xyz_guesses]
            xyz_guesses_total.extend(xyz_guesses)
            reaction_families.extend([family] * len(xyz_guesses))

    xyz_guesses_total, reaction_families = filter_ts_guesses_with_uma_sp(
        xyz_guesses_total, reaction_families,
        charge=reaction.charge, multiplicity=reaction.multiplicity,
    )

    return xyz_guesses_total, reaction_families


def get_stereo_consistent_xyz(smiles: str | None,
                              mol: Molecule,
                              xyz: dict,
                              label: str = '',
                              num_conformers: int = 10,
                              seed: int = 0xf00d,
                              ) -> dict:
    """
    Get coordinates of a species consistent with the stereochemistry (e.g., cis/trans ring substituents) given in
    its SMILES. The species' 3D coordinates may be generated from its graph (``mol``) alone, which does not keep
    the SMILES stereochemistry.

    If the SMILES has stereo marks and the stereochemistry perceived from ``xyz`` differs from them (only the
    stereocentres and double bonds specified in the SMILES are compared), the species is re-embedded from the
    SMILES with RDKit (ETKDGv3, then MMFF94 optimization), and the lowest-energy conformer with the SMILES
    stereochemistry is returned in the atom order of the species.

    Args:
        smiles (str | None): The SMILES of the species, possibly with stereo marks.
        mol (Molecule): The species molecule, with the same atom order as ``xyz``.
        xyz (dict): The Cartesian coordinates of the species.
        label (str, optional): The species label, used in log messages.
        num_conformers (int, optional): The number of conformers to embed.
        seed (int, optional): The random seed of the embedding.

    Returns:
        dict: The stereo-consistent coordinates, ``xyz`` itself if it is consistent, or if the SMILES has no stereo marks
              or could not be used.
    """
    reference = Chem.MolFromSmiles(smiles) if smiles and any(c in smiles for c in '@/\\') else None
    if reference is None:
        return xyz
    has_stereo_atoms = any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in reference.GetAtoms())
    has_stereo_bonds = any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in reference.GetBonds())
    if not has_stereo_atoms and not has_stereo_bonds:
        return xyz
    reference = Chem.AddHs(reference)
    match = get_topological_match(reference, mol)
    if match is None:
        logger.warning(f'Could not map the SMILES {smiles} of species {label} onto its atoms; '
                       f'its stereochemistry was not checked.')
        return xyz
    reference_stereo = get_specified_stereo_smiles(reference, reference)
    if get_specified_stereo_smiles(reference, reference, coords=[xyz['coords'][i] for i in match]) == reference_stereo:
        return xyz
    try:
        embedded = Chem.Mol(reference)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 1
        conf_ids = list(AllChem.EmbedMultipleConfs(embedded, numConfs=num_conformers, params=params))
        results = AllChem.MMFFOptimizeMoleculeConfs(embedded, numThreads=1, maxIters=2000)
    except Exception as e:
        conf_ids, results = list(), list()
        logger.debug(f'Could not embed species {label} from its SMILES: {e}')
    candidates = list()
    for conf_id, (_, energy) in zip(conf_ids, results):
        positions = embedded.GetConformer(conf_id).GetPositions()
        if get_specified_stereo_smiles(reference, reference, coords=positions) == reference_stereo:
            candidates.append((energy, conf_id))
    if not candidates:
        logger.warning(f'The coordinates of species {label} do not have the stereochemistry of its SMILES '
                       f'{smiles}, and a conformer with it could not be embedded; using the original coordinates.')
        return xyz
    positions = embedded.GetConformer(min(candidates)[1]).GetPositions()
    coords = [None] * len(match)
    for reference_index, species_index in enumerate(match):
        coords[species_index] = tuple(float(v) for v in positions[reference_index])
    logger.warning(f'The coordinates of species {label} do not have the stereochemistry of its SMILES {smiles}; '
                   f'using a conformer embedded from the SMILES instead.')
    return {'symbols': xyz['symbols'], 'isotopes': xyz['isotopes'], 'coords': tuple(coords)}


def get_topological_match(rd_mol: Chem.Mol,
                          mol: Molecule,
                          ) -> tuple[int, ...] | None:
    """
    Map the atoms of an RDKit molecule onto the atoms of a Molecule with the same constitution,
    comparing elements and connectivity only (not bond orders, charges, or aromaticity).

    Args:
        rd_mol (Chem.Mol): The RDKit molecule, with explicit hydrogens.
        mol (Molecule): The Molecule.

    Returns:
        tuple[int, ...] | None: The index in ``mol`` of each atom of ``rd_mol``, ``None`` if they do not match.
    """
    def topology(molecule: Chem.Mol) -> Chem.Mol:
        """A copy of the molecule with neutral, non-aromatic atoms and single bonds only."""
        rw_mol = Chem.RWMol(molecule)
        for atom in rw_mol.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetIsAromatic(False)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for bond in rw_mol.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        rw_mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(rw_mol)
        return rw_mol.GetMol()

    try:
        species_rd_mol = to_rdkit_mol(mol=mol, remove_h=False, sanitize=False)
        if species_rd_mol.GetNumAtoms() != rd_mol.GetNumAtoms():
            return None
        match = topology(species_rd_mol).GetSubstructMatch(topology(rd_mol))
    except Exception:
        return None
    return match if len(match) == rd_mol.GetNumAtoms() else None


def get_specified_stereo_smiles(rd_mol: Chem.Mol,
                                reference: Chem.Mol,
                                coords: list | np.ndarray | None = None,
                                ) -> str:
    """
    Get the canonical isomeric SMILES of a molecule, keeping only the stereocentres and stereo double bonds
    specified in a reference molecule with the same atom order (e.g., those given in a SMILES).

    Args:
        rd_mol (Chem.Mol): The RDKit molecule.
        reference (Chem.Mol): The reference molecule, whose specified stereo elements are kept.
        coords (list | np.ndarray, optional): Coordinates in the atom order of ``rd_mol``. If given, the stereochemistry
                                              is perceived from them instead of taken from ``rd_mol``.

    Returns:
        str: The canonical isomeric SMILES, without explicit hydrogens.
    """
    rd_mol = Chem.Mol(rd_mol)
    if coords is not None:
        conformer = Chem.Conformer(rd_mol.GetNumAtoms())
        conformer.Set3D(True)
        for i, coord in enumerate(coords):
            conformer.SetAtomPosition(i, [float(v) for v in coord])
        rd_mol.RemoveAllConformers()
        rd_mol.AddConformer(conformer, assignId=True)
        Chem.AssignStereochemistryFrom3D(rd_mol, replaceExistingTags=True)
    for atom, reference_atom in zip(rd_mol.GetAtoms(), reference.GetAtoms()):
        if reference_atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond, reference_bond in zip(rd_mol.GetBonds(), reference.GetBonds()):
        if reference_bond.GetStereo() == Chem.BondStereo.STEREONONE:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    Chem.AssignStereochemistry(rd_mol, cleanIt=True, force=True)
    return get_rdkit_stereo_smiles(rd_mol)


def get_qa_first_reaction(reaction: 'ARCReaction') -> 'ARCReaction':
    """
    Get a QA reaction with the QA reactant listed before the OH_d1 species.

    The QA TS guesses are built from the QA reactant's atom indices, so the family products (and their atom
    labels) are generated with the QA reactant first. The guesses are ordered back as the user's reaction
    by ``order_qa_guess_as_reactants()``.

    Args:
        reaction (ARCReaction): The QA reaction, with its reactants in any order.

    Returns:
        ARCReaction: The reaction itself if the QA reactant is already listed first, otherwise a copy of it with
                     the reactants listed QA first.
    """
    if len(reaction.r_species) != 2 or reaction.r_species[0].mol.get_net_charge() >= 0:
        return reaction
    qa_first = reaction.copy()
    qa_first.r_species = qa_first.r_species[::-1]
    qa_first.reactants = [spc.label for spc in qa_first.r_species]
    return qa_first


def order_qa_guess_as_reactants(xyz: dict,
                                reaction: 'ARCReaction',
                                n_qa: int,
                                atom_order: list[int],
                                ) -> dict:
    """
    Reorder a QA TS guess so its atoms follow the concatenated reactant order of the reaction,
    and give each atom the isotope of its reactant atom.

    A QA TS guess is built as the QA atoms (in the QA reactant's order) followed by the OH_d1 atoms
    in their placement order (``atom_order``). ARC expresses the reaction's bond changes as indices
    into the concatenated reactants (``reaction.r_species`` order), so the TS checks that use them
    (e.g., the normal mode displacement check) require the TS to list its atoms in that same order.
    The guess is built with the most common isotopes, so the reactant isotopes are restored here.

    Args:
        xyz (dict): The TS guess: QA atoms, then the OH_d1 atoms in placement order.
        reaction (ARCReaction): The QA reaction.
        n_qa (int): The number of atoms in the QA reactant.
        atom_order (list[int]): The OH_d1 atom index placed at each position after the QA atoms.

    Returns:
        dict: The TS guess with its atoms (and their isotopes) in the concatenated reactant order.
    """
    reactants, _ = reaction.get_reactants_and_products(return_copies=False)
    indices = list()
    for spc in reactants:
        if spc.mol.get_net_charge() < 0:
            indices.extend(n_qa + atom_order.index(i) for i in range(len(atom_order)))
        else:
            indices.extend(range(n_qa))
    if sorted(indices) != list(range(len(xyz['symbols']))):
        logger.warning(f'Could not order the TS guess of reaction {reaction.label} as its reactants; '
                       f'keeping the construction atom order.')
        return xyz
    ordered_xyz = sort_xyz_using_indices(xyz, indices)
    reactants_xyz = reaction.get_reactants_xyz(return_format='dict')
    if tuple(ordered_xyz['symbols']) == tuple(reactants_xyz['symbols']):
        ordered_xyz = {'symbols': ordered_xyz['symbols'],
                       'isotopes': tuple(reactants_xyz['isotopes']),
                       'coords': ordered_xyz['coords']}
    return ordered_xyz


def qa_sn2(reaction: 'ARCReaction',
           family: str = 'qa_sn2',
           ) -> tuple[list[dict], list[str]]:
    """
    Generate TS guesses for reactions of the ARC "qa_sn2" family,
    or of the "qa_ring_opening" family (a QA SN2 reaction with the Ca-N bond in a ring, with the same parameters).

    Args:
        reaction (ARCReaction): The reaction to process.
        family (str, optional): The ARC family label, 'qa_sn2' or 'qa_ring_opening'.

    Returns:
        Tuple containing:
            - list[dict]: Cartesian coordinates of TS guesses.
            - list[str]: Reaction family label for each guess.
    """
    return qa_ts_guesses(reaction,
                         family=family,
                         parameters_file='qa_sn2_parameters.yml',
                         no_products_message=f'No {family} family products found for this reaction.',
                         extract_indices=extract_sn2_reactant_and_indices,
                         build_guesses=build_sn2_site_guesses,
                         )


def build_sn2_site_guesses(sn2_parameters: dict,
                           main_reactant: 'ARCSpecies',
                           oh_d1: 'ARCSpecies',
                           initial_xyz: dict,
                           xyz_indices: dict,
                           zmats_total: list[dict],
                           threshold: float,
                           multiple_bond_pairs: set[tuple[int, int]],
                           ) -> tuple[list[dict], list[dict]]:
    """
    Build the QA SN2 TS guesses of one reactive site (see process_sn2_chosen_d_indices).

    Args:
        sn2_parameters (dict): QA SN2 parameters loaded from YAML.
        main_reactant (ARCSpecies): The QA reactant.
        oh_d1 (ARCSpecies): The OH_d1 species.
        initial_xyz (dict): Cartesian coordinates of the QA reactant.
        xyz_indices (dict): Atom indices of the site.
        zmats_total (list[dict]): Accumulated Z-matrices (duplicate filter).
        threshold (float): Collision detection threshold.
        multiple_bond_pairs (set[tuple[int, int]]): QA reactant atom pairs bonded with an order above single.

    Returns:
        tuple[list[dict], list[dict]]: The TS guess XYZ dicts and the updated Z-matrices.
    """
    _, xyz_guesses, zmats_total = process_sn2_chosen_d_indices(
        initial_xyz, xyz_indices, sn2_parameters, oh_d1, zmats_total, threshold=threshold,
        multiple_bond_pairs=multiple_bond_pairs, mol=main_reactant.mol,
    )
    return xyz_guesses, zmats_total


def get_qa_and_oh_d1(reaction: 'ARCReaction') -> tuple['ARCSpecies', 'ARCSpecies']:
    """
    Get the QA reactant and OH_d1 species from a QA reaction (of any QA TS-guess family).

    Args:
        reaction: An ARCReaction instance.

    Returns:
        Tuple containing:
            - ARCSpecies: The QA reactant
            - ARCSpecies: The OH_d1 species (contains a negatively charged O)

    Raises:
        ValueError: If the expected species cannot be identified.
    """
    arc_reactants, _ = reaction.get_reactants_and_products()
    qa_reactant, oh_d1 = None, None
    for spc in arc_reactants:
        # Use the net charge: a charge-separated resonance structure (e.g., [O-]C=[O+] of an ester) of the QA
        # reactant also has an atom with a -1 formal charge.
        if spc.mol.get_net_charge() < 0:
            oh_d1 = spc
        else:
            qa_reactant = spc
    if qa_reactant is None or oh_d1 is None:
        raise ValueError("A QA reaction must include a QA reactant and an OH_d1 species (with charge -1 on O).")
    return qa_reactant, oh_d1


def extract_sn2_reactant_and_indices(reaction: 'ARCReaction',
                                      product_dict: dict,
                                      ) -> tuple['ARCSpecies', 'ARCSpecies', dict, dict]:
    """
    Extract the reactant molecules and relevant atomic indices for the QA SN2 reaction.

    Atom scheme for the TS structure::

                R1
                |
         d  —  Ca — — N
                |
                O_r — H_r ··· O_w(H1_w, H2_w)

    Description:
        - Ca (*1): Electrophilic carbon, receives nucleophilic attack from O_r.
        - N (*2): Quaternary ammonium nitrogen, leaving group.
        - O_r (*3): Hydroxide oxygen, nucleophile.
        - H_r: Hydrogen on O_r.
        - O_w, H1_w, H2_w: Water molecule from OH_d1.
        - R1: Most electronegative neighbor of Ca (geometry reference).

    Args:
        reaction: An ARCReaction instance.
        product_dict: Dictionary containing reaction product information and atom mappings.

    Returns:
        Tuple containing:
            - ARCSpecies: The QA reactant
            - ARCSpecies: The OH_d1 species
            - dict: Initial XYZ coordinates of the QA reactant
            - dict: xyz_indices with keys Ca, N, R1, d, O_r, H_r, O_w, H1_w, n_qa, atom_order
    """
    main_reactant, oh_d1 = get_qa_and_oh_d1(reaction)
    Ca_xyz = product_dict['r_label_map']['*1']
    N_xyz = product_dict['r_label_map']['*2']
    O_r_xyz = product_dict['r_label_map']['*3']

    R1_xyz, _ = get_neighbors_by_electronegativity(main_reactant, Ca_xyz, N_xyz, two_neighbors=False)

    n_qa = len(main_reactant.mol.atoms)
    O_r_seq  = n_qa
    H_r_seq  = n_qa + 1
    H1_w_seq = n_qa + 2
    O_w_seq  = n_qa + 3

    xyz_indices = {
        'Ca':   Ca_xyz,
        'N':    N_xyz,
        'R1':   R1_xyz,
        'O_r':  O_r_seq,
        'H_r':  H_r_seq,
        'H1_w': H1_w_seq,
        'O_w':  O_w_seq,
        'atom_order': get_oh_d1_atom_order(oh_d1, O_r_xyz - n_qa),
    }
    return main_reactant, oh_d1, main_reactant.get_xyz(), xyz_indices


def get_oh_d1_atom_order(oh_d1: 'ARCSpecies', local_O_r_idx: int) -> list[int]:
    """
    Get the OH_d1 atom indices in the order the atoms are placed in a QA TS guess: O_r, H_r, H1_w, O_w, H2_w.

    Args:
        oh_d1 (ARCSpecies): The OH_d1 species.
        local_O_r_idx (int): The 0-index of the hydroxide oxygen (O_r) within the OH_d1 species.

    Returns:
        list[int]: The OH_d1 atom indices of O_r, H_r, H1_w, O_w, and H2_w.
    """
    O_r_atom = oh_d1.mol.atoms[local_O_r_idx]
    H_r_local = oh_d1.mol.atoms.index(list(O_r_atom.edges.keys())[0])
    O_w_local = next(
        oh_d1.mol.atoms.index(atom)
        for atom in oh_d1.mol.atoms
        if atom.element.symbol == 'O' and oh_d1.mol.atoms.index(atom) != local_O_r_idx
    )
    O_w_atom = oh_d1.mol.atoms[O_w_local]
    H_w_neighbors = [oh_d1.mol.atoms.index(n) for n in O_w_atom.edges.keys()]
    H1_w_local, H2_w_local = H_w_neighbors[0], H_w_neighbors[1]
    return [local_O_r_idx, H_r_local, H1_w_local, O_w_local, H2_w_local]


def process_sn2_chosen_d_indices(initial_xyz: dict,
                                  xyz_indices: dict,
                                  sn2_parameters: dict,
                                  oh_d1: 'ARCSpecies',
                                  zmats_total: list[dict],
                                  threshold: float = 0.8,
                                  multiple_bond_pairs: set[tuple[int, int]] | None = None,
                                  mol: Molecule | None = None,
                                  ) -> tuple[dict, list[dict], list[dict]]:
    """
    Stretch the Ca-N bond and place the five OH_d1 atoms to generate QA SN2 TS guesses.

    Strategy:
        1. Try the original (unmodified) Z-matrix after bond stretching.
        2. If no valid guesses are found, search the QA reactant Z-matrix for backbone
           dihedral angles that involve all three of N, Ca, and R1.  Generate variants
           of those dihedrals using adjustment factors of ±15°, ±25°, ±35°, ±45°, ±55°
           (via generate_dihedral_variants) and retry placement on each variant.
        3. If still no valid guesses, retry with the same dihedrals flipped by 180°.
        4. If still no valid guesses, the backside of Ca (anti to the leaving N) may be blocked by a
           substituent on a beta atom (Cb) in the given reactant conformer (beta-branching, e.g., neopentyl).
           Generate the staggered Ca-Cb rotamers of the QA reactant (via get_ca_cb_staggered_rotamers),
           and retry Step 1 on each of them. Guesses from all rotamers are collected.

    A ring Ca-N bond (ring-opening SN2) is stretched in Cartesian coordinates (see get_stretched_sn2_zmat),
    and Steps 2 and 3 are skipped since they would change ring dihedrals.

    Args:
        initial_xyz (dict): Initial Cartesian coordinates of the QA reactant.
        xyz_indices (dict): Atom indices: Ca, N, R1, O_r, H_r, H1_w, O_w, n_qa, atom_order.
        sn2_parameters (dict): QA SN2 parameters loaded from YAML.
        oh_d1 (ARCSpecies): The OH_d1 species.
        zmats_total (list[dict]): Accumulated Z-matrices to avoid duplicates.
        multiple_bond_pairs (set[tuple[int, int]], optional): QA reactant atom pairs bonded with an order above single.
        mol (Molecule, optional): The QA reactant molecule, used for the Z-matrix connectivity.

    Returns:
        tuple: Chosen indices dict, list of TS guess XYZ dicts, and updated Z-matrices.
    """
    chosen_xyz_indices = {k: xyz_indices[k] for k in ('Ca', 'N', 'R1', 'O_r', 'H_r', 'H1_w', 'O_w')}
    infra_indices = {
        'a': xyz_indices['Ca'],
        'b': xyz_indices['N'],
        'e': xyz_indices['R1'],
        'd': None,
    }
    ring_ca_n = is_ring_bond(mol, xyz_indices['Ca'], xyz_indices['N'])
    current_zmat, zmat_indices = get_stretched_sn2_zmat(initial_xyz, infra_indices, sn2_parameters, mol, ring_ca_n)

    # Step 1: try the original (unmodified) zmat
    ts_guesses, zmats_total = process_sn2_adjustments(
        sn2_parameters, current_zmat, oh_d1, xyz_indices, zmats_total, threshold=threshold,
        multiple_bond_pairs=multiple_bond_pairs,
    )
    if ts_guesses:
        return chosen_xyz_indices, ts_guesses, zmats_total

    # Step 2: find backbone dihedrals involving N, Ca, and R1, then try adjusted variants.
    # Skipped for a ring Ca-N bond: N, Ca, and R1 are then ring atoms, and changing a ring dihedral
    # in the Z-matrix distorts the ring closure.
    matches = [] if ring_ca_n else get_matching_dihedrals(current_zmat, zmat_indices['a'], zmat_indices['b'],
                                                          zmat_indices['e'], zmat_indices['d'])
    adjustment_factors = [15, 25, 35, 45, 55]
    adjusted_zmats = []
    for indices in matches:
        adjusted_zmats.extend(generate_dihedral_variants(current_zmat, indices, adjustment_factors))

    for zmat_to_process in adjusted_zmats:
        ts_guesses, zmats_total = process_sn2_adjustments(
            sn2_parameters, zmat_to_process, oh_d1, xyz_indices, zmats_total, threshold=threshold,
            multiple_bond_pairs=multiple_bond_pairs,
        )
        if ts_guesses:
            return chosen_xyz_indices, ts_guesses, zmats_total

    # Step 3: retry with each matching dihedral flipped by 180°
    flipped_zmats = []
    for indices in matches:
        flipped_zmats.extend(generate_dihedral_variants(current_zmat, indices, adjustment_factors, flip=True))

    for zmat_to_process in flipped_zmats:
        ts_guesses, zmats_total = process_sn2_adjustments(
            sn2_parameters, zmat_to_process, oh_d1, xyz_indices, zmats_total, threshold=threshold,
            multiple_bond_pairs=multiple_bond_pairs,
        )
        if ts_guesses:
            return chosen_xyz_indices, ts_guesses, zmats_total

    # Step 4: retry Step 1 on staggered Ca-Cb rotamers, in case a Cb substituent blocks the backside of Ca
    # (for a ring Ca-N bond, also ring conformers placing an exocyclic substituent of a ring Cb anti to N)
    if mol is None:
        return {}, [], zmats_total
    rotamer_ts_guesses = []
    for rotamer_xyz in get_ca_cb_staggered_rotamers(xyz=initial_xyz, mol=mol, ca=xyz_indices['Ca'],
                                                    n=xyz_indices['N'], threshold=threshold,
                                                    multiple_bond_pairs=multiple_bond_pairs,
                                                    include_ring_cb=ring_ca_n):
        rotamer_zmat, _ = get_stretched_sn2_zmat(rotamer_xyz, infra_indices, sn2_parameters, mol, ring_ca_n)
        ts_guesses, zmats_total = process_sn2_adjustments(
            sn2_parameters, rotamer_zmat, oh_d1, xyz_indices, zmats_total, threshold=threshold,
            multiple_bond_pairs=multiple_bond_pairs,
        )
        rotamer_ts_guesses.extend(ts_guesses)
    if rotamer_ts_guesses:
        return chosen_xyz_indices, rotamer_ts_guesses, zmats_total

    return {}, [], zmats_total


def is_ring_bond(mol: Molecule | None, a: int, b: int) -> bool:
    """
    Whether atoms a and b are bonded within a ring.

    Args:
        mol (Molecule, optional): The molecule. ``None`` is treated as no ring.
        a (int): The 0-index of the first atom.
        b (int): The 0-index of the second atom.

    Returns:
        bool: Whether the a-b bond is in a ring.
    """
    if mol is None:
        return False
    atom_a, atom_b = mol.atoms[a], mol.atoms[b]
    return mol.has_bond(atom_a, atom_b) and mol.is_bond_in_cycle(mol.get_bond(atom_a, atom_b))


# In an SN2 TS the Ca substituents are roughly perpendicular to the N...Ca...O axis (Ca is planar);
# DFT ring-opening TSs show N-Ca-X angles of about 89-104 degrees.
SN2_TS_N_CA_X_ANGLE_RANGE = (90.0, 100.0)


def get_stretched_sn2_zmat(xyz: dict,
                           infra_indices: dict,
                           sn2_parameters: dict,
                           mol: Molecule | None,
                           ring_ca_n: bool,
                           ) -> tuple[dict, dict]:
    """
    Get the Z-matrix of a QA reactant with the Ca-N bond stretched towards the QA SN2 TS.

    An open-chain Ca-N bond is stretched in the Z-matrix. A ring Ca-N bond (ring-opening SN2) is stretched in Cartesian
    coordinates before the Z-matrix is built, moving only Ca and its hydrogen atoms along the N-Ca axis:
    a Z-matrix has no ring-closure bond, so stretching Ca-N in it would stretch another ring bond instead.
    Moving Ca alone compresses or stretches its other bonds, which keep their usual lengths in an SN2 TS,
    so the stretched reactant is then relaxed with MMFF94 while the Ca-N distance is held fixed and Ca is kept
    planar, as in an SN2 TS, by holding each N-Ca-X angle within SN2_TS_N_CA_X_ANGLE_RANGE
    (see relax_with_fixed_distance).

    Args:
        xyz (dict): Cartesian coordinates of the QA reactant.
        infra_indices (dict): Atom indices: 'a' = Ca, 'b' = N, 'e' = R1, 'd' = None.
        sn2_parameters (dict): QA SN2 parameters loaded from YAML.
        mol (Molecule, optional): The QA reactant molecule, used for the Z-matrix connectivity.
        ring_ca_n (bool): Whether the Ca-N bond is in a ring.

    Returns:
        tuple[dict, dict]: The stretched Z-matrix and its atom indices.
    """
    if ring_ca_n:
        stretch = sn2_parameters['family_parameters']['qa_sn2']['stretch']
        xyz = stretch_xyz_bond(xyz, mol, infra_indices['b'], infra_indices['a'], stretch)
        xyz = relax_with_fixed_distance(xyz, mol, infra_indices['b'], infra_indices['a'],
                                        angle_range=SN2_TS_N_CA_X_ANGLE_RANGE)
        return setup_zmat_indices(xyz, infra_indices, mol=mol)
    zmat, zmat_indices = setup_zmat_indices(xyz, infra_indices, mol=mol)
    stretch_ab_bond(zmat, infra_indices, zmat_indices, sn2_parameters, 'qa_sn2')
    return zmat, zmat_indices


def relax_with_fixed_distance(xyz: dict,
                              mol: Molecule,
                              a: int,
                              b: int,
                              angle_range: tuple[float, float] | None = None,
                              ) -> dict:
    """
    Relax a geometry with the MMFF94 force field while holding the a-b distance fixed.

    Used after a bond has been stretched by moving one of its atoms, so that the other bonds of that atom
    return to their usual lengths and the surrounding atoms move out of the way.
    If MMFF94 fails, the given geometry is returned unchanged.

    Args:
        xyz (dict): Cartesian coordinates.
        mol (Molecule): The molecule, with atoms ordered as in ``xyz``.
        a (int): The 0-index of the first atom.
        b (int): The 0-index of the second atom.
        angle_range (tuple[float, float], optional): If given, each a-b-X angle (X a neighbour of b not bonded to a)
                                                     is held within this range, in degrees.

    Returns:
        dict: The relaxed Cartesian coordinates.
    """
    distance = calculate_distance(coords=xyz, atoms=[a, b], index=0)

    def add_constraints(force_field):
        force_field.MMFFAddDistanceConstraint(a, b, False, distance, distance, 1.0e5)
        if angle_range is not None:
            for neighbor in mol.atoms[b].edges.keys():
                x = mol.atoms.index(neighbor)
                # A neighbour also bonded to a (three-membered ring) cannot be perpendicular to the a-b axis
                if x != a and not mol.has_bond(neighbor, mol.atoms[a]):
                    force_field.MMFFAddAngleConstraint(a, b, x, False, angle_range[0], angle_range[1], 1.0e3)

    try:
        return _constrained_mmff_relax(xyz, mol, add_constraints)
    except Exception as e:
        logger.debug(f'Could not relax the stretched geometry with MMFF94, keeping it as is: {e}')
        return xyz


def _constrained_mmff_relax(xyz: dict,
                            mol: Molecule,
                            add_constraints: Callable[[Any], None],
                            ) -> dict:
    """
    Relax a geometry with the MMFF94 force field under constraints, keeping the atom order.

    Args:
        xyz (dict): Cartesian coordinates.
        mol (Molecule): The molecule, with atoms ordered as in ``xyz``.
        add_constraints (Callable[[Any], None]): Adds the constraints to the RDKit MMFF94 force field it is given.

    Raises:
        ValueError: If MMFF94 atom typing fails or the optimization does not converge within 10 x 1000 iterations.

    Returns:
        dict: The relaxed Cartesian coordinates.
    """
    conf, rd_mol = rdkit_conf_from_mol(mol=mol, xyz=xyz)
    mmff_properties = AllChem.MMFFGetMoleculeProperties(rd_mol)
    if mmff_properties is None:
        raise ValueError('MMFF94 atom typing failed')
    force_field = AllChem.MMFFGetMoleculeForceField(rd_mol, mmff_properties, confId=conf.GetId())
    add_constraints(force_field)
    force_field.Initialize()
    not_converged = 1
    for _ in range(10):
        not_converged = force_field.Minimize(maxIts=1000)
        if not not_converged:
            break
    if not_converged:
        raise ValueError('MMFF94 optimization did not converge')
    positions = force_field.Positions()
    return {'symbols': xyz['symbols'], 'isotopes': xyz['isotopes'],
            'coords': tuple(tuple(positions[3 * i: 3 * i + 3]) for i in range(len(xyz['symbols'])))}


def get_ca_cb_staggered_rotamers(xyz: dict,
                                 mol: Molecule,
                                 ca: int,
                                 n: int,
                                 threshold: float = 0.8,
                                 multiple_bond_pairs: set[tuple[int, int]] | None = None,
                                 tolerance: float = 5.0,
                                 include_ring_cb: bool = False,
                                 screen_after_relax: bool = False,
                                 ) -> list[dict]:
    """
    Generate staggered rotamers about the Ca-Cb bond(s) of a QA reactant, each placing one Cb substituent anti to N.

    An SN2 attack requires the backside of Ca (anti to the leaving N) to be open. The reactant conformer is
    chosen without regard to the reaction, so a substituent on a beta atom (Cb) may occupy the attack path
    (the beta-branching / neopentyl effect). In a staggered Ca-Cb rotamer, the Cb substituent anti to N
    (dihedral N-Ca-Cb-X = 180°) is the one closest to the backside of Ca. Enumerating the staggered rotamers,
    each placing a different Cb substituent X anti to N, therefore includes the rotamer in which the attack path
    is flanked by the smallest Cb substituent (e.g., H), if Cb has one.

    Each Cb is a heavy-atom neighbor of Ca other than N. A Cb is skipped if the Ca-Cb bond is in a ring
    (not freely rotatable) or if Cb has no substituents other than Ca; a substituent X collinear with Ca-Cb
    (e.g., an alkyne) is not considered since rotating it is meaningless. The Cb side of each Ca-Cb bond is rotated
    rigidly about the Ca-Cb axis, the rest of the molecule is kept in place, and the atom order is preserved.
    If Ca has several such Cb atoms, all combinations of their rotamers are generated. Combinations in which
    every chosen X is already anti to N (within ``tolerance``) reproduce the input conformer and are skipped.
    Rotamers that fail the collision screen (colliding_atoms) are discarded.

    A rigid rotation can bring the rotated groups into close nonbonded contacts that the rest of the molecule
    would relax away. Each rotamer that passes the collision screen is therefore relaxed with the MMFF94 force field
    while every chosen N-Ca-Cb-X dihedral is constrained at 180°, keeping the atom order. If the force field
    cannot be set up or the optimization fails or does not converge, the rigid rotamer is kept.

    With ``include_ring_cb``, a Cb bonded to Ca within a ring is also considered, with its exocyclic substituents
    as X (e.g., the hydrogen of a ring CH(CH3) group). A ring Ca-Cb bond cannot be rotated rigidly, so the
    conformer is obtained by the constrained MMFF94 relaxation alone (a ring flip, e.g., equatorial to axial).
    Such a conformer is discarded if the relaxation fails, since without it the input conformer is unchanged.

    With ``screen_after_relax``, the collision screen is applied to the relaxed rotamer instead of the rigid one.
    When both Ca and Cb carry bulky groups (e.g., a tert-butyl Cb next to an NMe3 leaving group), every rigid rotamer
    can bring hydrogens of the two groups nearly on top of each other, although the relaxed rotamer is unstrained.

    Args:
        xyz (dict): The Cartesian coordinates of the QA reactant.
        mol (Molecule): The QA reactant molecule, with the same atom order as ``xyz``.
        ca (int): The 0-index of Ca, the carbon attacked by the nucleophile.
        n (int): The 0-index of the leaving N.
        threshold (float, optional): The collision threshold passed to colliding_atoms.
        multiple_bond_pairs (set[tuple[int, int]], optional): Atom pairs bonded with an order above single.
        tolerance (float, optional): The deviation in degrees from 180° within which X is considered already anti to N.
        include_ring_cb (bool, optional): Whether to also consider a Cb bonded to Ca within a ring.
        screen_after_relax (bool, optional): Whether to apply the collision screen after the MMFF94 relaxation.

    Returns:
        list[dict]: The Cartesian coordinates of the non-colliding (relaxed) rotamers, ordered by the Cb and X atom indices.
    """
    ca_atom = mol.atoms[ca]
    cb_options = list()
    for cb_atom in sorted(ca_atom.edges.keys(), key=lambda atom: mol.atoms.index(atom)):
        cb = mol.atoms.index(cb_atom)
        ring_cb = mol.is_bond_in_cycle(mol.get_bond(ca_atom, cb_atom))
        if cb == n or cb_atom.is_hydrogen() or (ring_cb and not include_ring_cb):
            continue
        x_indices = sorted(mol.atoms.index(atom) for atom in cb_atom.edges.keys() if atom is not ca_atom
                           and not (ring_cb and mol.is_bond_in_cycle(mol.get_bond(cb_atom, atom))))
        x_indices = [x for x in x_indices if not is_angle_linear(calculate_angle(xyz, [ca, cb, x], index=0))]
        if not x_indices:
            continue
        # A ring Ca-Cb bond is not rotated rigidly, so it has no Cb side to rotate
        cb_side = None if ring_cb else determine_top_group_indices(mol=mol, atom1=ca_atom, atom2=cb_atom, index=0)[0]
        options = list()
        for x in x_indices:
            rotation = (180.0 - calculate_dihedral_angle(xyz, [n, ca, cb, x], index=0)) % 360.0
            if min(rotation, 360.0 - rotation) < tolerance:
                rotation = 0.0
            options.append((cb, x, cb_side, rotation))
        cb_options.append(options)

    rotamers = list()
    for combination in itertools.product(*cb_options):
        if all(rotation == 0.0 for _, _, _, rotation in combination):
            continue
        coords = [tuple(coord) for coord in xyz['coords']]
        for cb, _, cb_side, rotation in combination:
            if rotation != 0.0 and cb_side is not None:
                axis = unit_vector([coords[cb][i] - coords[ca][i] for i in range(3)])
                apply_rodrigues_rotation(coords=coords, axis_origin=coords[ca], axis_unit=axis,
                                         angle_rad=math.radians(rotation), indices=cb_side)
        rotamer = {'symbols': xyz['symbols'], 'isotopes': xyz['isotopes'], 'coords': tuple(coords)}
        if not screen_after_relax and colliding_atoms(rotamer, threshold=threshold,
                                                      multiple_bond_pairs=multiple_bond_pairs):
            continue

        def add_constraints(force_field):
            for cb, x, _, _ in combination:
                force_field.MMFFAddTorsionConstraint(n, ca, cb, x, False, 180.0, 180.0, 1.0e4)

        try:
            rotamer = _constrained_mmff_relax(rotamer, mol, add_constraints)
        except Exception as e:
            if any(cb_side is None and rotation != 0.0 for _, _, cb_side, rotation in combination):
                logger.debug(f'Could not relax a ring Ca-Cb conformer with MMFF94, discarding it: {e}')
                continue
            logger.debug(f'Could not relax a Ca-Cb rotamer with MMFF94, keeping the rigid rotamer: {e}')
        if screen_after_relax and colliding_atoms(rotamer, threshold=threshold,
                                                  multiple_bond_pairs=multiple_bond_pairs):
            continue
        rotamers.append(rotamer)
    return rotamers


def process_sn2_adjustments(sn2_parameters: dict,
                             initial_zmat: dict,
                             oh_d1: 'ARCSpecies',
                             xyz_indices: dict,
                             zmats_total: list[dict],
                             threshold: float = 0.8,
                             multiple_bond_pairs: set[tuple[int, int]] | None = None,
                             ) -> tuple[list[dict], list[dict]]:
    """
    Build internal coordinate arrays and invoke TS guess generation for QA SN2.

    The five OH_d1 atoms (O_r, H_r, O_w, H1_w, H2_w) are placed sequentially
    using internal coordinates derived from the QA reactant geometry and YAML parameters.

    Internal coordinate definitions (L-M-N-X convention, X = new atom):
        - O_r:  r(Ca-O_r),    a(N-Ca-O_r),      d(R1-N-Ca-O_r)
        - H_r:  r(O_r-H_r),   a(Ca-O_r-H_r),    d(N-Ca-O_r-H_r)   [scanned over hr_dihedral_scan_values]
        - H1_w: r(O_r-H1_w),  a(H_r-O_r-H1_w),  d(Ca-H_r-O_r-H1_w)   [H-bond bridge: O_r accepts H1_w]
        - O_w:  r(H1_w-O_w),  a(O_r-H1_w-O_w),  d(H_r-O_r-H1_w-O_w)
        - H2_w: r(O_w-H2_w),  a(H1_w-O_w-H2_w), d(O_r-H1_w-O_w-H2_w)

    Args:
        sn2_parameters (dict): QA SN2 parameters loaded from YAML.
        initial_zmat (dict): Z-matrix of the QA reactant after Ca-N bond stretching.
        oh_d1 (ARCSpecies): The OH_d1 species.
        xyz_indices (dict): Dictionary of atom indices.
        zmats_total (list[dict]): Accumulated Z-matrices.
        multiple_bond_pairs (set[tuple[int, int]], optional): QA reactant atom pairs bonded with an order above single.

    Returns:
        tuple[list[dict], list[dict]]: Generated TS guess XYZ dicts and updated Z-matrices.
    """
    Ca_xyz   = xyz_indices['Ca']
    N_xyz    = xyz_indices['N']
    R1_xyz   = xyz_indices['R1']
    O_r_seq  = xyz_indices['O_r']
    H_r_seq  = xyz_indices['H_r']
    H1_w_seq = xyz_indices['H1_w']
    O_w_seq  = xyz_indices['O_w']

    r_atoms = [Ca_xyz, O_r_seq, O_r_seq,  H1_w_seq, O_w_seq]
    a_atoms = [
        [N_xyz,   Ca_xyz ],
        [Ca_xyz,  O_r_seq],
        [H_r_seq, O_r_seq],
        [O_r_seq, H1_w_seq],
        [H1_w_seq, O_w_seq],
    ]
    d_atoms = [
        [R1_xyz,  N_xyz,    Ca_xyz  ],
        [N_xyz,   Ca_xyz,   O_r_seq ],
        [Ca_xyz,  H_r_seq,  O_r_seq ],
        [H_r_seq, O_r_seq,  H1_w_seq],
        [O_r_seq, H1_w_seq, O_w_seq ],
    ]

    params = sn2_parameters['family_parameters']['qa_sn2']
    r_value = params['r_value']
    a_value = params['a_value']
    d_values = params['d_values']

    return place_oh_d1_cluster(zmat_to_xyz(initial_zmat), oh_d1, r_atoms, a_atoms, d_atoms,
                               r_value, a_value, d_values, zmats_total, xyz_indices, threshold=threshold,
                               multiple_bond_pairs=multiple_bond_pairs,
                               scan_values={1: params['hr_dihedral_scan_values']})


def place_oh_d1_cluster(initial_xyz: dict,
                        oh_d1: 'ARCSpecies',
                        r_atoms: list[int],
                        a_atoms: list[list[int]],
                        d_atoms: list[list[int]],
                        r_value: list[float],
                        a_value: list[float],
                        d_values: list[list],
                        zmats_total: list[dict],
                        xyz_indices: dict,
                        threshold: float = 0.8,
                        multiple_bond_pairs: set[tuple[int, int]] | None = None,
                        scan_values: dict[int, list[float]] | None = None,
                        check_ts_bonds_func: Callable[[dict], bool] | None = None,
                        ) -> tuple[list[dict], list[dict]]:
    """
    Generate Cartesian TS guesses by sequentially placing the five OH_d1 atoms onto a QA reactant.
    Shared by the QA SN2 and the QA Hofmann elimination families.

    Each 'scan' entry of a d_values row is expanded by substituting each value from the respective
    ``scan_values`` list, so each YAML row produces a candidate geometry per combination of scanned values.

    Args:
        initial_xyz (dict): XYZ of the QA reactant after Ca-N bond stretching.
        oh_d1 (ARCSpecies): Provides element symbols for each atom to place.
        r_atoms (list[int]): Reference atom index for each of the 5 placements.
        a_atoms (list[list[int]]): Angle-defining atom pairs for each placement.
        d_atoms (list[list[int]]): Dihedral-defining atom triplets for each placement.
        r_value (list[float]): Bond distances for each placement.
        a_value (list[float]): Bond angles for each placement.
        d_values (list[list]): Dihedral rows from YAML; index 1 may be 'scan'.
        zmats_total (list[dict]): Existing Z-matrices (duplicate filter).
        xyz_indices (dict): Must contain O_r, H_r, H1_w, O_w, Ca, atom_order keys.
        threshold (float, optional): Collision detection threshold. Defaults to 0.8.
        multiple_bond_pairs (set[tuple[int, int]], optional): QA reactant atom pairs bonded with an order above single.
        scan_values (dict[int, list[float]], optional): Values to scan per 'scan' column of d_values,
                                                         keyed by the column index. Required if d_values has
                                                         'scan' entries.
        check_ts_bonds_func (Callable[[dict], bool], optional): Validates the bonding of a TS guess.
                                                                 Defaults to the QA SN2 bond check.

    Returns:
        tuple[list[dict], list[dict]]: Accepted TS guess XYZ dicts and updated Z-matrices.
    """
    xyz_guesses = []
    O_r_seq  = xyz_indices['O_r']
    H_r_seq  = xyz_indices['H_r']
    H1_w_seq = xyz_indices['H1_w']
    O_w_seq  = xyz_indices['O_w']
    Ca_xyz   = xyz_indices['Ca']
    atom_order = xyz_indices['atom_order']
    scan_values = scan_values or dict()
    if check_ts_bonds_func is None:
        check_ts_bonds_func = lambda xyz: check_sn2_ts_bonds(xyz, [O_r_seq, H_r_seq, Ca_xyz, H1_w_seq, O_w_seq])

    for d_value_row in d_values:
        scan_columns = [i for i, d_val in enumerate(d_value_row) if d_val == 'scan']

        for scanned_dihedrals in itertools.product(*[scan_values[i] for i in scan_columns]):
            expanded_row = list(d_value_row)
            for i, dihedral in zip(scan_columns, scanned_dihedrals):
                expanded_row[i] = dihedral

            xyz_guess = copy.deepcopy(initial_xyz)
            failed = False
            for i in range(5):
                xyz_guess = add_atom_to_xyz_using_internal_coords(
                    xyz=xyz_guess,
                    element=oh_d1.mol.atoms[atom_order[i]].element.symbol,
                    r_index=r_atoms[i],
                    a_indices=a_atoms[i],
                    d_indices=d_atoms[i],
                    r_value=r_value[i],
                    a_value=a_value[i],
                    d_value=expanded_row[i],
                )
                if xyz_guess is None:
                    failed = True
                    break
            if failed:
                continue

            are_valid_bonds = check_ts_bonds_func(xyz_guess)
            colliding = colliding_atoms(xyz_guess, threshold=threshold, multiple_bond_pairs=multiple_bond_pairs)
            zmat_guess = xyz_to_zmat(xyz_guess)
            duplicate = any(compare_zmats(existing, zmat_guess) for existing in zmats_total)

            if not colliding and not duplicate and are_valid_bonds:
                xyz_guesses.append(xyz_guess)
                zmats_total.append(zmat_guess)

            else:
                logger.debug(f"Rejected TS guess with scanned dihedrals {scanned_dihedrals}°: "
                             f"{xyz_guess if colliding else ''}; ")

    return xyz_guesses, zmats_total


def check_sn2_ts_bonds(xyz_guess: dict, atom_indices: list[int]) -> bool:
    """
    Validate the bonding geometry of a QA SN2 TS guess.

    Checks:
        1. O_r's nearest heavy atom is Ca (forming C-O bond dominates).
        2. H_r's nearest atom is O_r (O-H bond intact).
        3. H1_w's nearest atom is O_w (covalent O_w-H1_w ~1.0 Å) and nearest heavy atom after O_w is O_r (H-bond ···O_r).
        4. H2_w's nearest atom is O_w (covalent O_w-H2_w bond ~0.96 Å).
        5. O_w's two nearest atoms are H1_w and H2_w (both covalent bonds intact).

    Args:
        xyz_guess (dict): The TS guess coordinates.
        atom_indices (list[int]): [O_r_seq, H_r_seq, Ca_xyz, H1_w_seq, O_w_seq].

    Returns:
        bool: True if all checks pass.
    """
    O_r_seq, H_r_seq, Ca_xyz, H1_w_seq, O_w_seq = atom_indices
    H2_w_seq = O_w_seq + 1

    o_r_bonds = sorted_distances_of_atom(xyz_guess, O_r_seq)
    for bond_idx, _ in o_r_bonds:
        if bond_idx == Ca_xyz:
            break
        if xyz_guess['symbols'][bond_idx] != 'H':
            return False

    h_r_bonds = sorted_distances_of_atom(xyz_guess, H_r_seq)
    if h_r_bonds[0][0] != O_r_seq:
        return False

    h1_w_bonds = sorted_distances_of_atom(xyz_guess, H1_w_seq)
    if h1_w_bonds[0][0] != O_w_seq:
        return False
    h1_w_heavy = [idx for idx, _ in h1_w_bonds if xyz_guess['symbols'][idx] != 'H']
    if len(h1_w_heavy) < 2 or h1_w_heavy[1] != O_r_seq:
        return False

    h2_w_bonds = sorted_distances_of_atom(xyz_guess, H2_w_seq)
    if h2_w_bonds[0][0] != O_w_seq:
        return False

    o_w_bonds = sorted_distances_of_atom(xyz_guess, O_w_seq)
    o_w_nearest_two = {o_w_bonds[0][0], o_w_bonds[1][0]}
    if o_w_nearest_two != {H1_w_seq, H2_w_seq}:
        return False

    return True


def qa_hofmann(reaction: 'ARCReaction') -> tuple[list[dict], list[str]]:
    """
    Generate TS guesses for reactions of the ARC "qa_hofmann" (QA Hofmann E2 elimination) family.

    Each product dictionary of the family corresponds to a single beta hydrogen (Hb) abstracted by the hydroxide.
    For each Hb, a QA reactant conformer with Hb anti-periplanar to the leaving N is used,
    the reaction-centre bonds are adjusted, and the OH_d1 atoms are placed onto Hb.

    Args:
        reaction (ARCReaction): The reaction to process.

    Returns:
        Tuple containing:
            - list[dict]: Cartesian coordinates of TS guesses.
            - list[str]: Reaction family label per guess.
    """
    return qa_ts_guesses(reaction,
                         family='qa_hofmann',
                         parameters_file='qa_hofmann_parameters.yml',
                         no_products_message='No qa_hofmann family products found for the reaction.',
                         extract_indices=extract_hofmann_reactant_and_indices,
                         build_guesses=build_hofmann_site_guesses,
                         site_filter=filter_unique_hofmann_sites,
                         )


def filter_unique_hofmann_sites(reaction: 'ARCReaction', product_dicts: list[dict]) -> list[dict]:
    """
    Keep the QA Hofmann elimination product dictionaries that abstract a unique beta hydrogen
    (see get_unique_hofmann_product_dicts).

    Args:
        reaction (ARCReaction): The reaction.
        product_dicts (list[dict]): The qa_hofmann family product dictionaries.

    Returns:
        list[dict]: The product dictionaries with a unique abstracted Hb.
    """
    main_reactant, _ = get_qa_and_oh_d1(reaction)
    return get_unique_hofmann_product_dicts(product_dicts, main_reactant)


def build_hofmann_site_guesses(hofmann_parameters: dict,
                               main_reactant: 'ARCSpecies',
                               oh_d1: 'ARCSpecies',
                               initial_xyz: dict,
                               xyz_indices: dict,
                               zmats_total: list[dict],
                               threshold: float,
                               multiple_bond_pairs: set[tuple[int, int]],
                               ) -> tuple[list[dict], list[dict]]:
    """
    Build the QA Hofmann elimination TS guesses of one abstracted beta hydrogen (Hb):
    the reaction centre of each QA reactant conformer with Hb anti-periplanar to N is adjusted
    and the OH_d1 atoms are placed onto Hb (see process_hofmann_adjustments).

    Args:
        hofmann_parameters (dict): QA Hofmann elimination parameters loaded from YAML.
        main_reactant (ARCSpecies): The QA reactant.
        oh_d1 (ARCSpecies): The OH_d1 species.
        initial_xyz (dict): Cartesian coordinates of the QA reactant.
        xyz_indices (dict): Atom indices of the site.
        zmats_total (list[dict]): Accumulated Z-matrices (duplicate filter).
        threshold (float): Collision detection threshold.
        multiple_bond_pairs (set[tuple[int, int]]): QA reactant atom pairs bonded with an order above single.

    Returns:
        tuple[list[dict], list[dict]]: The TS guess XYZ dicts and the updated Z-matrices.
    """
    site_guesses = list()
    pi_activated = is_pi_activated_beta_carbon(main_reactant.mol, xyz_indices['Cb'], xyz_indices['Ca'],
                                               multiple_bond_pairs)
    for conformer_xyz in get_hofmann_anti_periplanar_conformers(initial_xyz, main_reactant.mol, xyz_indices,
                                                                threshold=threshold,
                                                                multiple_bond_pairs=multiple_bond_pairs):
        xyz_guesses, zmats_total = process_hofmann_adjustments(
            hofmann_parameters, conformer_xyz, main_reactant.mol, oh_d1, xyz_indices, zmats_total,
            pi_activated=pi_activated, threshold=threshold, multiple_bond_pairs=multiple_bond_pairs,
        )
        site_guesses.extend(xyz_guesses)
    return site_guesses, zmats_total


def get_unique_hofmann_product_dicts(product_dicts: list[dict], spc: 'ARCSpecies') -> list[dict]:
    """
    Remove QA Hofmann elimination product dictionaries that abstract an equivalent beta hydrogen.

    Two (Ca, Cb) pairs are equivalent if their atoms are constitutionally equivalent (same canonical rank
    without tie breaking), and only the first pair of each equivalence class is kept.
    If the QA reactant has a chiral centre, constitutionally equivalent (Ca, Cb) pairs may be diastereotopic
    (giving diastereomeric TSs), so all pairs are kept.
    The hydrogens of a methyl Cb are interchanged by the methyl rotation, so a single Hb is kept for a methyl Cb.
    The hydrogens of a methylene or methine Cb are all kept, since they may be diastereotopic (giving E/Z alkenes).

    Args:
        product_dicts (list[dict]): The qa_hofmann family product dictionaries.
        spc (ARCSpecies): The QA reactant (its atom indices are those of the r_label_map).

    Returns:
        list[dict]: The product dictionaries with a unique abstracted Hb.
    """
    mol = spc.mol
    chiral = get_number_of_chiral_centers(spc.label, mol, xyz=spc.get_xyz(), just_get_the_number=True) > 0
    ranks = list(Chem.CanonicalRankAtoms(to_rdkit_mol(mol=mol, remove_h=False), breakTies=False))
    kept_pairs, kept_methyl_cbs, unique_product_dicts = dict(), set(), list()
    for product_dict in product_dicts:
        ca, cb = product_dict['r_label_map']['*1'], product_dict['r_label_map']['*3']
        pair_class = (ca, cb) if chiral else (ranks[ca], ranks[cb])
        if kept_pairs.setdefault(pair_class, (ca, cb)) != (ca, cb):
            continue
        if sum(atom.is_hydrogen() for atom in mol.atoms[cb].edges.keys()) == 3:
            if cb in kept_methyl_cbs:
                continue
            kept_methyl_cbs.add(cb)
        unique_product_dicts.append(product_dict)
    return unique_product_dicts


def extract_hofmann_reactant_and_indices(reaction: 'ARCReaction',
                                         product_dict: dict,
                                         ) -> tuple['ARCSpecies', 'ARCSpecies', dict, dict]:
    """
    Extract reactant molecules and relevant atomic indices for a QA Hofmann elimination reaction.

    Atom map of the TS structure::

        N — Ca — Cb — Hb ··· O_r — H_r
                 |           ·
                 R1          H1_w — O_w — H2_w

    Description:
        - Ca (*1): Alpha carbon, loses the leaving N.
        - N (*2): Quaternary ammonium nitrogen, leaving group.
        - Cb (*3): Beta carbon, loses Hb and forms the C=C bond with Ca.
        - Hb (*4): Beta hydrogen, abstracted by O_r (anti-periplanar to N).
        - O_r (*5): Hydroxide oxygen, the base.
        - H_r: Hydrogen on O_r.
        - O_w, H1_w, H2_w: Water molecule of OH_d1, H1_w is H-bonded to O_r.
        - R1: Most electronegative neighbor of Cb other than Hb (geometry reference).

    Args:
        reaction: An ARCReaction instance.
        product_dict: Dictionary of reaction product information with atom mappings.

    Returns:
        Tuple containing:
            - ARCSpecies: The QA reactant
            - ARCSpecies: The OH_d1 species
            - dict: Initial XYZ coordinates of the QA reactant
            - dict: xyz_indices with keys Ca, N, Cb, Hb, R1, O_r, H_r, H1_w, O_w, atom_order
    """
    main_reactant, oh_d1 = get_qa_and_oh_d1(reaction)
    Ca_xyz = product_dict['r_label_map']['*1']
    N_xyz = product_dict['r_label_map']['*2']
    Cb_xyz = product_dict['r_label_map']['*3']
    Hb_xyz = product_dict['r_label_map']['*4']
    O_r_xyz = product_dict['r_label_map']['*5']

    R1_xyz, _ = get_neighbors_by_electronegativity(main_reactant, Cb_xyz, Hb_xyz, two_neighbors=False)

    n_qa = len(main_reactant.mol.atoms)
    xyz_indices = {
        'Ca':   Ca_xyz,
        'N':    N_xyz,
        'Cb':   Cb_xyz,
        'Hb':   Hb_xyz,
        'R1':   R1_xyz,
        'O_r':  n_qa,
        'H_r':  n_qa + 1,
        'H1_w': n_qa + 2,
        'O_w':  n_qa + 3,
        'atom_order': get_oh_d1_atom_order(oh_d1, O_r_xyz - n_qa),
    }
    return main_reactant, oh_d1, main_reactant.get_xyz(), xyz_indices


def is_pi_activated_beta_carbon(mol: Molecule,
                                cb: int,
                                ca: int,
                                multiple_bond_pairs: set[tuple[int, int]],
                                ) -> bool:
    """
    Check whether a beta carbon is conjugated to a pi system, i.e., whether a neighbor of Cb (other than Ca)
    is in a multiple bond (e.g., an aryl, C=O, C=C, or C#N group).
    A pi system stabilizes the developing carbanion at Cb, shifting the E2 TS toward E1cb
    (earlier proton transfer and a less stretched Ca-N bond).

    Args:
        mol (Molecule): The QA reactant molecule.
        cb (int): The 0-index of the beta carbon.
        ca (int): The 0-index of the alpha carbon.
        multiple_bond_pairs (set[tuple[int, int]]): Atom pairs bonded with an order above single.

    Returns:
        bool: Whether a neighbor of Cb other than Ca is in a multiple bond.
    """
    pi_atoms = {index for pair in multiple_bond_pairs for index in pair}
    return any(mol.atoms.index(neighbor) in pi_atoms
               for neighbor in mol.atoms[cb].edges.keys() if mol.atoms.index(neighbor) != ca)


def get_hofmann_anti_periplanar_conformers(xyz: dict,
                                           mol: Molecule,
                                           xyz_indices: dict,
                                           threshold: float = 0.8,
                                           multiple_bond_pairs: set[tuple[int, int]] | None = None,
                                           tolerance: float = 30.0,
                                           ) -> list[dict]:
    """
    Get QA reactant conformers with the beta hydrogen (Hb) anti-periplanar to the leaving N
    (Hb-Cb-Ca-N dihedral = 180°), the geometry required for an E2 elimination.

    If Hb is already anti to N in the input conformer (within ``tolerance``), only the input conformer is returned.
    Otherwise, the staggered Ca-Cb rotamers (via get_ca_cb_staggered_rotamers) that place Hb anti to N are returned.
    Each rotamer is relaxed before the collision screen, since an anti Hb may put bulky Ca and Cb groups gauche
    to each other, which a rigid rotation cannot accommodate.
    If no such rotamer exists and the Ca-Cb bond is in a ring (which cannot be rotated), the ring conformer
    with Hb closest to anti (e.g., the chair with N axial) is returned instead, if it improves on the input
    conformer (see get_ring_anti_periplanar_conformer). Otherwise, the input conformer is returned as the best available.

    Args:
        xyz (dict): Cartesian coordinates of the QA reactant.
        mol (Molecule): The QA reactant molecule, with the same atom order as ``xyz``.
        xyz_indices (dict): Atom indices, must contain Ca, N, Cb, and Hb.
        threshold (float, optional): The collision threshold passed to get_ca_cb_staggered_rotamers.
        multiple_bond_pairs (set[tuple[int, int]], optional): Atom pairs bonded with an order above single.
        tolerance (float, optional): The deviation in degrees from 180° within which Hb is considered anti to N.

    Returns:
        list[dict]: Cartesian coordinates of the QA reactant conformers.
    """
    dihedral_indices = [xyz_indices['N'], xyz_indices['Ca'], xyz_indices['Cb'], xyz_indices['Hb']]

    def is_anti(conformer_xyz: dict) -> bool:
        """Whether Hb is anti-periplanar to N in the conformer."""
        dihedral = calculate_dihedral_angle(conformer_xyz, dihedral_indices, index=0)
        return abs(get_angle_in_180_range(dihedral)) > 180.0 - tolerance

    if is_anti(xyz):
        return [xyz]
    rotamers = [rotamer for rotamer in get_ca_cb_staggered_rotamers(xyz=xyz, mol=mol, ca=xyz_indices['Ca'],
                                                                    n=xyz_indices['N'], threshold=threshold,
                                                                    multiple_bond_pairs=multiple_bond_pairs,
                                                                    screen_after_relax=True)
                if is_anti(rotamer)]
    if rotamers:
        return rotamers
    ca_atom, cb_atom = mol.atoms[xyz_indices['Ca']], mol.atoms[xyz_indices['Cb']]
    if mol.is_bond_in_cycle(mol.get_bond(ca_atom, cb_atom)):
        ring_conformer = get_ring_anti_periplanar_conformer(xyz, mol, dihedral_indices)
        best_xyz = ring_conformer if ring_conformer is not None else xyz
        if not is_anti(best_xyz):
            dihedral = abs(get_angle_in_180_range(calculate_dihedral_angle(best_xyz, dihedral_indices, index=0)))
            logger.warning(f'No anti-periplanar beta hydrogen (atom {xyz_indices["Hb"]}) is reachable for the ring '
                           f'Ca-Cb bond (atoms {xyz_indices["Ca"]}, {xyz_indices["Cb"]}) of the QA reactant: '
                           f'the best conformer has an N-Ca-Cb-Hb dihedral of {dihedral:.1f} degrees.')
        return [best_xyz]
    return [xyz]


def get_ring_anti_periplanar_conformer(xyz: dict,
                                       mol: Molecule,
                                       dihedral_indices: list[int],
                                       num_conformers: int = 50,
                                       seed: int = 0xf00d,
                                       ) -> dict | None:
    """
    Get the QA reactant ring conformer with the beta hydrogen (Hb) closest to anti-periplanar to the leaving N,
    for a Ca-Cb bond in a ring.

    A ring Ca-Cb bond cannot be rotated, but a ring conformer change (e.g., a chair flip that turns N from
    equatorial to axial) may bring Hb anti to N. Reactant conformers are embedded with RDKit (ETKDGv3) and
    optimized with MMFF94, keeping the stereochemistry of the input conformer (the configuration of each
    stereocentre and double bond, and the local configuration of each tetrahedral atom, so, e.g., Hb is not
    swapped with another hydrogen on Cb) and the atom order. The conformer with the largest |N-Ca-Cb-Hb| dihedral is returned
    if it improves on the input conformer.

    Args:
        xyz (dict): Cartesian coordinates of the QA reactant.
        mol (Molecule): The QA reactant molecule, with the same atom order as ``xyz``.
        dihedral_indices (list[int]): The 0-indices of N, Ca, Cb, and Hb.
        num_conformers (int, optional): The number of conformers to embed.
        seed (int, optional): The random seed of the embedding.

    Returns:
        dict | None: The Cartesian coordinates of the ring conformer, ``None`` if no conformer improves on the input.
    """
    input_dihedral = abs(get_angle_in_180_range(calculate_dihedral_angle(xyz, dihedral_indices, index=0)))
    ca, cb = dihedral_indices[1], dihedral_indices[2]
    try:
        _, rd_mol = rdkit_conf_from_mol(mol=mol, xyz=xyz)
        Chem.AssignStereochemistryFrom3D(rd_mol)
        reference_stereo = get_rdkit_stereo_smiles(rd_mol)
        reference_chirality = [get_local_chirality(xyz['coords'], mol, atom) for atom in (ca, cb)]
        # Tag every tetrahedral atom (not only the stereocentres) so the embedding keeps the local configuration
        # of each atom as in the input conformer, e.g., which hydrogen on Cb is Hb
        Chem.AssignAtomChiralTagsFromStructure(rd_mol, replaceExistingTags=True)
        embedded = Chem.Mol(rd_mol)
        embedded.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 1
        conf_ids = list(AllChem.EmbedMultipleConfs(embedded, numConfs=num_conformers, params=params))
        results = AllChem.MMFFOptimizeMoleculeConfs(embedded, numThreads=1, maxIters=2000)
    except Exception as e:
        logger.debug(f'Could not embed ring conformers of the QA reactant: {e}')
        return None
    best, best_key = None, None
    for conf_id, (_, energy) in zip(conf_ids, results):
        coords = [tuple(float(v) for v in position) for position in embedded.GetConformer(conf_id).GetPositions()]
        conformer_xyz = {'symbols': xyz['symbols'], 'isotopes': xyz['isotopes'], 'coords': tuple(coords)}
        if [get_local_chirality(coords, mol, atom) for atom in (ca, cb)] != reference_chirality:
            continue
        stereo_mol = Chem.Mol(embedded, confId=conf_id)
        Chem.AssignStereochemistryFrom3D(stereo_mol)
        if get_rdkit_stereo_smiles(stereo_mol) != reference_stereo:
            continue
        dihedral = abs(get_angle_in_180_range(calculate_dihedral_angle(conformer_xyz, dihedral_indices, index=0)))
        key = (round(dihedral, 1), -energy)
        if best_key is None or key > best_key:
            best, best_key = conformer_xyz, key
    if best is None or best_key[0] <= round(input_dihedral, 1):
        return None
    return best


def get_rdkit_stereo_smiles(rd_mol: Chem.Mol) -> str:
    """
    Get the canonical isomeric SMILES (without explicit hydrogens) of an RDKit molecule, encoding its assigned stereochemistry.

    Args:
        rd_mol (Chem.Mol): The RDKit molecule.

    Returns:
        str: The canonical isomeric SMILES.
    """
    return Chem.MolToSmiles(Chem.RemoveHs(rd_mol, sanitize=False))


def get_local_chirality(coords: list | tuple,
                        mol: Molecule,
                        atom_index: int,
                        ) -> int:
    """
    Get the sign of the signed volume spanned by the first three neighbors (by atom index) of an atom,
    which distinguishes the two mirror-image arrangements of its neighbors (e.g., which of two hydrogens is which).

    Args:
        coords (list | tuple): The Cartesian coordinates.
        mol (Molecule): The molecule, with the same atom order as ``coords``.
        atom_index (int): The 0-index of the atom.

    Returns:
        int: 1 or -1, 0 if the atom has fewer than three neighbors.
    """
    neighbors = sorted(mol.atoms.index(atom) for atom in mol.atoms[atom_index].edges.keys())
    if len(neighbors) < 3:
        return 0
    center = np.array(coords[atom_index], dtype=float)
    vectors = [np.array(coords[neighbor], dtype=float) - center for neighbor in neighbors[:3]]
    return int(np.sign(np.dot(vectors[0], np.cross(vectors[1], vectors[2]))))


def stretch_xyz_bond(xyz: dict,
                     mol: Molecule,
                     a: int,
                     b: int,
                     stretch: float,
                     ) -> dict:
    """
    Stretch (or shrink) the a-b bond in Cartesian coordinates by translating the atoms on the b side
    along the a-b bond vector, keeping the a side in place.
    If the a-b bond is in a ring, only b and its hydrogen atoms are translated.

    Args:
        xyz (dict): The Cartesian coordinates.
        mol (Molecule): The molecule, with the same atom order as ``xyz``.
        a (int): The 0-index of the atom kept in place.
        b (int): The 0-index of the atom translated.
        stretch (float): The factor by which to multiply the a-b bond length.

    Returns:
        dict: The modified Cartesian coordinates.
    """
    atom_a, atom_b = mol.atoms[a], mol.atoms[b]
    if mol.is_bond_in_cycle(mol.get_bond(atom_a, atom_b)):
        b_side = [b] + [mol.atoms.index(atom) for atom in atom_b.edges.keys() if atom.is_hydrogen()]
    else:
        b_side, _ = determine_top_group_indices(mol=mol, atom1=atom_a, atom2=atom_b, index=0)
    coords = [list(coord) for coord in xyz['coords']]
    shift = [(stretch - 1.0) * (coords[b][i] - coords[a][i]) for i in range(3)]
    for idx in b_side:
        coords[idx] = [coords[idx][i] + shift[i] for i in range(3)]
    return {'symbols': xyz['symbols'], 'isotopes': xyz['isotopes'], 'coords': tuple(tuple(c) for c in coords)}


def process_hofmann_adjustments(hofmann_parameters: dict,
                                initial_xyz: dict,
                                mol: Molecule,
                                oh_d1: 'ARCSpecies',
                                xyz_indices: dict,
                                zmats_total: list[dict],
                                pi_activated: bool = False,
                                threshold: float = 0.8,
                                multiple_bond_pairs: set[tuple[int, int]] | None = None,
                                ) -> tuple[list[dict], list[dict]]:
    """
    Adjust the reaction centre of the QA reactant and invoke TS guess generation for QA Hofmann elimination.

    The Ca-N, Ca-Cb, and Cb-Hb bonds are scaled by the YAML stretch factors (the pi-activated set if ``pi_activated``),
    then the five OH_d1 atoms (O_r, H_r, H1_w, O_w, H2_w) are placed sequentially.

    Internal coordinate definitions (L-M-N-X convention, X = new atom):
        - O_r:  r(Hb-O_r),    a(Cb-Hb-O_r),     d(R1-Cb-Hb-O_r)    [scanned over r1_dihedral_scan_values]
        - H_r:  r(O_r-H_r),   a(Hb-O_r-H_r),    d(Cb-Hb-O_r-H_r)   [scanned over hr_dihedral_scan_values]
        - H1_w: r(O_r-H1_w),  a(H_r-O_r-H1_w),  d(Hb-H_r-O_r-H1_w)   [H-bond bridge: O_r accepts H1_w]
        - O_w:  r(H1_w-O_w),  a(O_r-H1_w-O_w),  d(H_r-O_r-H1_w-O_w)
        - H2_w: r(O_w-H2_w),  a(H1_w-O_w-H2_w), d(O_r-H1_w-O_w-H2_w)

    Args:
        hofmann_parameters (dict): QA Hofmann elimination parameters loaded from YAML.
        initial_xyz (dict): Cartesian coordinates of the QA reactant conformer, Hb anti-periplanar to N.
        mol (Molecule): The QA reactant molecule, with the same atom order as ``initial_xyz``.
        oh_d1 (ARCSpecies): The OH_d1 species.
        xyz_indices (dict): Dictionary of atom indices.
        zmats_total (list[dict]): Accumulated Z-matrices.
        pi_activated (bool, optional): Whether Cb is conjugated to a pi system (activated beta hydrogen).
        threshold (float, optional): Collision detection threshold. Defaults to 0.8.
        multiple_bond_pairs (set[tuple[int, int]], optional): QA reactant atom pairs bonded with an order above single.

    Returns:
        tuple[list[dict], list[dict]]: Generated TS guess XYZ dicts and updated Z-matrices.
    """
    Ca_xyz   = xyz_indices['Ca']
    N_xyz    = xyz_indices['N']
    Cb_xyz   = xyz_indices['Cb']
    Hb_xyz   = xyz_indices['Hb']
    R1_xyz   = xyz_indices['R1']
    O_r_seq  = xyz_indices['O_r']
    H_r_seq  = xyz_indices['H_r']
    H1_w_seq = xyz_indices['H1_w']
    O_w_seq  = xyz_indices['O_w']

    params = hofmann_parameters['family_parameters']['qa_hofmann']
    r_value = list(params['r_value'])
    stretch = params['stretch']
    if pi_activated:
        r_value[0] = params['pi_activated']['r_Hb-Or']
        stretch = params['pi_activated']['stretch']

    stretched_xyz = stretch_xyz_bond(initial_xyz, mol, Ca_xyz, N_xyz, stretch['Ca-N'])
    stretched_xyz = stretch_xyz_bond(stretched_xyz, mol, Ca_xyz, Cb_xyz, stretch['Ca-Cb'])
    stretched_xyz = stretch_xyz_bond(stretched_xyz, mol, Cb_xyz, Hb_xyz, stretch['Cb-Hb'])

    r_atoms = [Hb_xyz, O_r_seq, O_r_seq,  H1_w_seq, O_w_seq]
    a_atoms = [
        [Cb_xyz,   Hb_xyz ],
        [Hb_xyz,   O_r_seq],
        [H_r_seq,  O_r_seq],
        [O_r_seq,  H1_w_seq],
        [H1_w_seq, O_w_seq],
    ]
    d_atoms = [
        [R1_xyz,  Cb_xyz,   Hb_xyz  ],
        [Cb_xyz,  Hb_xyz,   O_r_seq ],
        [Hb_xyz,  H_r_seq,  O_r_seq ],
        [H_r_seq, O_r_seq,  H1_w_seq],
        [O_r_seq, H1_w_seq, O_w_seq ],
    ]
    scan_values = {0: params['r1_dihedral_scan_values'], 1: params['hr_dihedral_scan_values']}

    return place_oh_d1_cluster(
        stretched_xyz, oh_d1, r_atoms, a_atoms, d_atoms, r_value, params['a_value'], params['d_values'],
        zmats_total, xyz_indices, threshold=threshold, multiple_bond_pairs=multiple_bond_pairs,
        scan_values=scan_values,
        check_ts_bonds_func=lambda xyz: check_hofmann_ts_bonds(xyz, [O_r_seq, H_r_seq, Cb_xyz, H1_w_seq,
                                                                     O_w_seq, Hb_xyz]),
    )


def check_hofmann_ts_bonds(xyz_guess: dict, atom_indices: list[int]) -> bool:
    """
    Validate the bonding geometry of a QA Hofmann elimination TS guess.

    Checks:
        1. The QA SN2 checks (check_sn2_ts_bonds) with Cb in place of Ca: O_r's nearest heavy atom is Cb,
           the O_r-H_r bond is intact, and the water is intact and H-bonded to O_r through H1_w.
        2. Hb's two nearest atoms are Cb and O_r (Hb is being transferred from Cb to O_r).

    Args:
        xyz_guess (dict): The TS guess coordinates.
        atom_indices (list[int]): [O_r_seq, H_r_seq, Cb_xyz, H1_w_seq, O_w_seq, Hb_xyz].

    Returns:
        bool: True if all checks pass.
    """
    O_r_seq, H_r_seq, Cb_xyz, H1_w_seq, O_w_seq, Hb_xyz = atom_indices
    if not check_sn2_ts_bonds(xyz_guess, [O_r_seq, H_r_seq, Cb_xyz, H1_w_seq, O_w_seq]):
        return False
    hb_bonds = sorted_distances_of_atom(xyz_guess, Hb_xyz)
    return {hb_bonds[0][0], hb_bonds[1][0]} == {Cb_xyz, O_r_seq}


# The QA TS-guess function of each QA family label, with a family description used in log messages.
QA_TS_GUESS_FUNCTIONS = {
    'qa_sn2': (qa_sn2, 'QA SN2'),
    'qa_ring_opening': (functools.partial(qa_sn2, family='qa_ring_opening'), 'QA SN2 ring opening'),
    'qa_hofmann': (qa_hofmann, 'QA Hofmann elimination'),
}


def remove_duplicate_ts_guesses(xyzs: list[dict],
                                families: list[str],
                                tol: float = 0.01,
                                ) -> tuple[list[dict], list[str]]:
    """
    Remove TS guesses with the same geometry as an earlier guess, regardless of atom labels.
    Two guesses are the same if their sorted interatomic distances agree within ``tol`` for each element pair,
    as for guesses built at symmetry-equivalent reactive sites of a symmetric reactant. A mirror image is also
    removed, since it has the same energy and leads to the mirror-image TS.

    Args:
        xyzs (list[dict]): The candidate TS-guess geometries.
        families (list[str]): The reaction-family label per guess (kept aligned with ``xyzs``).
        tol (float, optional): The distance tolerance in Angstrom.

    Returns:
        tuple[list[dict], list[str]]: The unique ``(xyzs, families)``, in their original order.
    """
    def fingerprint(xyz: dict) -> dict[tuple[str, str], np.ndarray]:
        coords, symbols = np.array(xyz['coords'], dtype=float), xyz['symbols']
        distances = dict()
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pair = tuple(sorted((symbols[i], symbols[j])))
                distances.setdefault(pair, list()).append(float(np.linalg.norm(coords[i] - coords[j])))
        return {pair: np.sort(np.array(values)) for pair, values in distances.items()}

    unique_xyzs, unique_families, unique_fingerprints = list(), list(), list()
    for xyz, family in zip(xyzs, families):
        fp = fingerprint(xyz)
        if any(fp.keys() == other.keys() and all(np.allclose(fp[pair], other[pair], atol=tol) for pair in fp)
               for other in unique_fingerprints):
            continue
        unique_xyzs.append(xyz)
        unique_families.append(family)
        unique_fingerprints.append(fp)
    if len(unique_xyzs) < len(xyzs):
        logger.info(f'Removed {len(xyzs) - len(unique_xyzs)} duplicate TS guesses '
                    f'(same geometry up to atom labels), {len(unique_xyzs)} remain.')
    return unique_xyzs, unique_families


def filter_ts_guesses_with_uma_sp(xyzs: list[dict],
                                  families: list[str],
                                  charge: int = 0,
                                  multiplicity: int = 1,
                                  top_k: int = 5,
                                  device: str = 'cpu',
                                  timeout: int = 1800,
                                  ) -> tuple[list[dict], list[str]]:
    """
    Remove duplicate TS guesses, then rank them by a UMA (fairchem-core) gas-phase single point and keep the top
    ``top_k``. The ranking only runs if the ``UMA_TS_GUESS_PREFILTER`` setting is ``True``.

    Args:
        xyzs (list[dict]): The candidate TS-guess geometries.
        families (list[str]): The reaction-family label per guess (kept aligned with ``xyzs``).
        charge (int, optional): Total charge, conditioned on the UMA (omol) calculation.
        multiplicity (int, optional): Spin multiplicity, conditioned on the UMA calculation.
        top_k (int, optional): The number of best-ranked guesses to keep.
        device (str, optional): The device the UMA worker runs on ('cpu' or 'cuda').
        timeout (int, optional): Seconds to wait for the UMA worker before forwarding all guesses.

    Returns:
        tuple[list[dict], list[str]]: The kept ``(xyzs, families)``, in their original order.
    """
    xyzs, families = remove_duplicate_ts_guesses(xyzs, families)
    if len(xyzs) <= top_k:
        return xyzs, families  # nothing to cut
    if not settings['UMA_TS_GUESS_PREFILTER']:
        logger.warning(f'UMA SP pre-filter is off. Forwarding all {len(xyzs)} TS guesses unranked; set '
                       f'UMA_TS_GUESS_PREFILTER = True in ~/.arc/settings.py to keep only the {top_k} best-ranked ones.')
        return xyzs, families
    uma_python = settings.get('UMA_PYTHON')
    if uma_python is None:
        logger.warning(f"UMA SP pre-filter skipped: 'uma_env' not found. Forwarding all {len(xyzs)} TS guesses "
                       f"unranked; install UMA (devtools/install_uma.sh) to keep only the {top_k} best-ranked ones.")
        return xyzs, families

    worker = os.path.join(os.path.dirname(__file__), 'uma_sp_worker.py')
    try:
        with tempfile.TemporaryDirectory(prefix='uma_sp_filter_') as tmp_dir:
            for i, xyz in enumerate(xyzs):
                with open(os.path.join(tmp_dir, f'guess_{i}.xyz'), 'w') as fh:
                    fh.write(xyz_to_xyz_file_format(xyz))
            proc = subprocess.run(
                [uma_python, worker, tmp_dir, str(charge), str(multiplicity),
                 UMA_LATEST_MODEL, device],
                capture_output=True, text=True, timeout=timeout,
            )
        if proc.returncode != 0:
            logger.warning(f'UMA SP pre-filter failed (exit {proc.returncode}); forwarding all '
                           f'TS guesses.\n{proc.stderr.strip()}')
            return xyzs, families
        energies = json.loads(proc.stdout.strip().splitlines()[-1])
        # energies maps 'guess_<i>' -> eV; require an energy for every guess before cutting.
        idx_energy = {i: energies[f'guess_{i}'] for i in range(len(xyzs)) if f'guess_{i}' in energies}
        if len(idx_energy) != len(xyzs):
            logger.warning('UMA SP pre-filter did not score every TS guess; forwarding all.')
            return xyzs, families
    except Exception as e:
        logger.warning(f'UMA SP pre-filter errored ({e.__class__.__name__}: {e}); forwarding all '
                       f'TS guesses.')
        return xyzs, families

    keep = sorted(sorted(idx_energy, key=idx_energy.get)[:top_k])
    logger.info(f'UMA SP pre-filter: kept {len(keep)}/{len(xyzs)} TS guesses '
                f'(indices {keep}) for the DFT pipeline.')
    return [xyzs[i] for i in keep], [families[i] for i in keep]
