"""
Perceive 3D Cartesian coordinates to 2D graph connectivity
Adapted from https://github.com/jensengroup/xyz2mol, DOI: 10.1002/bkcs.10334
"""

import copy
import itertools
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import networkx as nx

from rdkit.Chem import rdchem
from rdkit.Chem import rdEHTTools
from rdkit import Chem

from arc.common import logger


ATOM_LIST = \
    ['h', 'he',
     'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne',
     'na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar',
     'k', 'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu',
     'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
     'rb', 'sr', 'y', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
     'cd', 'in', 'sn', 'sb', 'te', 'i', 'xe',
     'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',
     'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w', 're', 'os', 'ir', 'pt',
     'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
     'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu',
     ]

atomic_valence = defaultdict(list)
atomic_valence[1] = [1]
atomic_valence[5] = [3, 4]
atomic_valence[6] = [4]
atomic_valence[7] = [3, 4]
atomic_valence[8] = [2, 1, 3]
atomic_valence[9] = [1]
atomic_valence[14] = [4]
atomic_valence[15] = [5, 3]  # [5,4,3]
atomic_valence[16] = [6, 3, 2]  # [6,4,2]
atomic_valence[17] = [1]
atomic_valence[32] = [4]
atomic_valence[35] = [1]
atomic_valence[53] = [1]

atomic_valence_electrons = dict()
atomic_valence_electrons[1] = 1
atomic_valence_electrons[5] = 3
atomic_valence_electrons[6] = 4
atomic_valence_electrons[7] = 5
atomic_valence_electrons[8] = 6
atomic_valence_electrons[9] = 7
atomic_valence_electrons[14] = 4
atomic_valence_electrons[15] = 5
atomic_valence_electrons[16] = 6
atomic_valence_electrons[17] = 7
atomic_valence_electrons[32] = 4
atomic_valence_electrons[35] = 7
atomic_valence_electrons[53] = 7


def xyz_to_smiles(xyz: Union[dict, str],
                  charge: int = 0,
                  use_huckel: bool = True,
                  quick: bool = True,
                  embed_chiral: bool = False,
                  ) -> Optional[List[str]]:
    """
    Convert xyz to 2D SMILES.

    Args:
        xyz (Union[dict, str]): The xyz representation.
        charge (int, optional): The species electronic charge.
        use_huckel (bool, optional): Whether to use Huckel bond orders to locate bonds.
                                     Otherwise, van der Waals radii are used.
        quick (bool, optional): Whether to use networkx to process large systems faster.
        embed_chiral (bool, optional): Whether to embed chirality information into the output.

    Returns:
        List[str]: Entries are respective SMILES representation.
    """
    global ATOM_LIST
    atoms = [ATOM_LIST.index(symbol.lower()) + 1 for symbol in xyz['symbols']]
    mols = xyz2mol(atoms=atoms,
                   coordinates=xyz['coords'],
                   charge=charge,
                   use_graph=quick,
                   allow_charged_fragments=False,
                   embed_chiral=embed_chiral,
                   use_huckel=use_huckel,
                   )
    if mols is None:
        return None
    smiles_list = list()
    for mol in mols:
        try:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=embed_chiral)
            m = Chem.MolFromSmiles(smiles)
            smiles_list.append(Chem.MolToSmiles(m, isomericSmiles=embed_chiral))
        except Exception:
            pass
    return smiles_list or None


def get_ua(max_valence_list: list,
           valence_list: list,
           ) -> Tuple[list, list]:
    """
    Get unsaturated atoms from knowing the valence and maximum valence for each atom.

    Args:
        max_valence_list (list): The maximum valence list.
        valence_list (list): The actual valence list.

    Returns:
        Tuple[list, list]:
            - Entries represent unsaturation level of atoms (ua).
            - Entries represent the degree of unsaturation per atom (du).
    """
    ua, du = list(), list()
    for i, (max_valence, valence) in enumerate(zip(max_valence_list, valence_list)):
        if not max_valence - valence > 0:
            continue
        ua.append(i)
        du.append(max_valence - valence)
    return ua, du


def get_bo(ac: np.ndarray,
           du: list,
           valences: list,
           ua_pairs: list,
           use_graph: bool = True,
           ) -> np.ndarray:
    """
    Get bond orders.

    Args:
        ac (np.ndarray): Atom connectivity.
        du (list): degree of unsaturation per atom.
        valences (list): Atom valences.
        ua_pairs (list): List of unsaturated atom pairs.
        use_graph (bool, optional): Whether to use the graph representation of the molecule.

    Returns:
        np.ndarray: Bond orders.
    """
    bo = ac.copy()
    du_save = list()
    while du_save != du:
        for i, j in ua_pairs:
            bo[i, j] += 1
            bo[j, i] += 1
        bo_valence = list(bo.sum(axis=1))
        du_save = copy.copy(du)
        ua, du = get_ua(valences, bo_valence)
        ua_pairs = get_ua_pairs(ua, ac, use_graph=use_graph)[0]
    return bo


def valences_not_too_large(bo: np.ndarray,
                           valences: list,
                           ) -> bool:
    """
    Check that atomic valences are not too large.

    Args:
        bo (np.ndarray): Bond orders.
        valences (list): atomic valences.

    Returns:
        bool: ``True`` if valences are not too large, ``False`` otherwise.
    """
    number_of_bonds_list = bo.sum(axis=1)
    for valence, number_of_bonds in zip(valences, number_of_bonds_list):
        if number_of_bonds > valence:
            return False
    return True


def is_charge_ok(bo: np.ndarray,
                 charge: int,
                 atomic_valence_electrons_: list,
                 atoms: List[int],
                 allow_charged_fragments: bool = True,
                 ) -> bool:
    """
    Check that the overall charge is in agreement with the formal charges of all atoms in the perceived molecule.

    Args:
        bo (np.ndarray): Bond orders.
        charge (int): The overall molecule charge.
        atomic_valence_electrons_ (list): The number of valence electrons per atom.
        atoms (List[int]): Atoms.
        allow_charged_fragments (bool, optional): Whether to allow molecule fragments to be charged.

    Returns:
        bool: ``True`` if the overall charge makes sense, ``False`` otherwise.
    """
    total_charge = 0
    q_list = list()  # Charge fragment list.
    if allow_charged_fragments:
        bo_valences = list(bo.sum(axis=1))
        for i, atom in enumerate(atoms):
            q = get_atomic_charge(atom, atomic_valence_electrons_[atom], bo_valences[i])
            total_charge += q
            if atom == 6:
                number_of_single_bonds_to_c = list(bo[i, :]).count(1)
                if number_of_single_bonds_to_c == 2 and bo_valences[i] == 2:
                    total_charge += 1
                    q = 2
                if number_of_single_bonds_to_c == 3 and total_charge + 1 < charge:
                    total_charge += 2
                    q = 1
            if q != 0:
                q_list.append(q)
    return charge == total_charge


def bo_is_ok(bo: np.ndarray,
             ac: np.ndarray,
             charge: int,
             du: list,
             atomic_valence_electrons_: list,
             atoms: List[int],
             valences: list,
             allow_charged_fragments: bool = True,
             ) -> bool:
    """
    Sanity check for perceived bond-orders.

    Args:
        bo (np.ndarray): Bond orders.
        ac (np.ndarray): Atom connectivity.
        charge (int): Overall charge.
        du (list): degree of unsaturation per atom.
        atomic_valence_electrons_ (list): The number of valence electrons per atom.
        atoms (List[int]): Atoms.
        valences (list): Atom valences. 
        allow_charged_fragments (bool): Whether to allow charged fragments.

    Returns:
        bool: ``True`` if perceived bond-orders make sense, ``False`` otherwise.
    """
    if not valences_not_too_large(bo, valences):
        return False
    check_sum = (bo - ac).sum() == sum(du)
    check_charge = is_charge_ok(bo,
                                charge,
                                atomic_valence_electrons_,
                                atoms,
                                allow_charged_fragments,
                                )
    if check_charge and check_sum:
        return True
    return False


def get_atomic_charge(atom: int,
                      atomic_valence_electrons_: int,
                      bo_valence: int,
                      ) -> int:
    """
    Get formal charge for an atom.

    Args:
        atom (int): The atom index. 
        atomic_valence_electrons_ (int): The number of valence electrons for this atom.
        bo_valence (int): Bond order valence.

    Returns:
        int: The respective formal charge.
    """
    if atom == 1:
        return 1 - bo_valence
    if atom == 5:
        return 3 - bo_valence
    if atom == 15 and bo_valence == 5:
        return 0
    if atom == 16 and bo_valence == 6:
        return 0
    return atomic_valence_electrons_ - 8 + bo_valence


def bo2mol(mol,
           bo_matrix: np.ndarray,
           atoms: List[int],
           atomic_valence_electrons_: dict,
           mol_charge: int,
           allow_charged_fragments: bool = True,
           ):
    """
    From bond order, atoms, valence structure and total charge, generate an rdkit molecule.
    Based on code written by Paolo Toscani.

    Args:
        mol (RDMol) An rdkit molecule object instance.
        bo_matrix (np.ndarray): bond order matrix of molecule.
        atoms (List[int]): Entries are integer atomic symbols.
        atomic_valence_electrons_ (dict): The number of valence electrons per atom.
        mol_charge (int) The total charge of molecule.
        allow_charged_fragments (bool, optional): Whether to allow charged fragments.

    Returns:
        Optional[RDMol]: An updated rdkit molecule with bond connectivity.
    """
    l1 = len(bo_matrix)
    l2 = len(atoms)
    bo_valences = list(bo_matrix.sum(axis=1))
    if l1 != l2:
        logger.warning(f'sizes of adjMat ({l1:d}) and Atoms {l2:d} differ.')
        return None
    rw_mol = Chem.RWMol(mol)
    bond_type_dict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    for i in range(l1):
        for j in range(i + 1, l1):
            bo = int(round(bo_matrix[i, j]))
            if bo == 0:
                continue
            bt = bond_type_dict.get(bo, Chem.BondType.SINGLE)
            rw_mol.AddBond(i, j, bt)
    mol = rw_mol.GetMol()
    if allow_charged_fragments:
        mol = set_atomic_charges(
            mol,
            atoms,
            atomic_valence_electrons_,
            bo_valences,
            bo_matrix,
            mol_charge,
        )
    else:
        mol = set_atomic_radicals(mol, atoms, atomic_valence_electrons_, bo_valences)
    return mol


def set_atomic_charges(mol,
                       atoms: List[int],
                       atomic_valence_electrons_: dict,
                       bo_valences: list,
                       bo_matrix: np.ndarray,
                       mol_charge: int,
                       ):
    """
    Set format charges for all atoms.

    Args:
        mol (RDMol) An rdkit molecule object instance.
        atoms (List[int]): Entries are integer atomic symbols.
        atomic_valence_electrons_ (dict): The number of valence electrons per atom.
        bo_valences (list): Bond order valences.
        bo_matrix (np.ndarray): Bond order matrix.
        mol_charge (int): The overall molecular charge.

    Returns:
        RDMol: An updated rdkit molecule with formal charges.
    """
    q = 0
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom, atomic_valence_electrons_[atom], bo_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_c = list(bo_matrix[i, :]).count(1)
            if number_of_single_bonds_to_c == 2 and bo_valences[i] == 2:
                q += 1
                charge = 0
            if number_of_single_bonds_to_c == 3 and q + 1 < mol_charge:
                q += 2
                charge = 1
        if abs(charge) > 0:
            a.SetFormalCharge(int(charge))
    return mol


def set_atomic_radicals(mol,
                        atoms: List[int],
                        atomic_valence_electrons_: dict,
                        bo_valences: list,
                        ):
    """
    Set the radical appropriately. The number of radical electrons is set as equal to the absolute atomic charge.

    Args:
        mol (RDMol) An rdkit molecule object instance.
        atoms (List[int]): Entries are integer atomic symbols.
        atomic_valence_electrons_ (dict): The number of valence electrons per atom.
        bo_valences (list): Bond order valences.

    Returns:
        RDMol: An updated rdkit molecule with radical electrons.
    """
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(
            atom,
            atomic_valence_electrons_[atom],
            bo_valences[i])
        if abs(charge) > 0:
            a.SetNumRadicalElectrons(abs(int(charge)))
    return mol


def get_bonds(ua: list,
              ac: np.ndarray,
              ) -> list:
    """
    Get the molecule bonds.

    Args:
        ua (list): Unsaturated atoms.
        ac (np.ndarray): Atom connectivity.

    Returns:
        list: The molecule's bonds.
    """
    bonds = list()
    for k, i in enumerate(ua):
        for j in ua[k + 1:]:
            if ac[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))
    return bonds


def get_ua_pairs(ua: list,
                 ac: np.ndarray,
                 use_graph: bool = True,
                 ) -> list:
    """

    Args:
        ua (list): Unsaturated atoms.
        ac (np.ndarray): Atom connectivity.
        use_graph (bool, optional): Whether to use the graph representation of the molecule.

    Returns:
        list: Pairs of unsaturated atoms.
    """
    bonds = get_bonds(ua, ac)
    if len(bonds) == 0:
        return [()]
    if use_graph:
        g = nx.Graph()
        g.add_edges_from(bonds)
        ua_pairs = [list(nx.max_weight_matching(g))]
        return ua_pairs
    max_atoms_in_combo = 0
    ua_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(ua) / 2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            ua_pairs = [combo]
        elif atoms_in_combo == max_atoms_in_combo:
            ua_pairs.append(combo)
    return ua_pairs


def ac2bo(ac: np.ndarray,
          atoms: list,
          charge: int,
          allow_charged_fragments: bool = True,
          use_graph: bool = True,
          ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Atom connectivity to bond order.
    Implementation of the bond order assignment algorithm shown in Figure 2 of 10.1002/bkcs.10334.
    Acronyms: ua = unsaturated atoms, du = degree of unsaturation, best_bo: B^curr in the figure.

    Args:
        ac (np.ndarray): Atom connectivity.
        atoms (List[int]): Entries are integer atomic symbols.
        charge (int): The molecular charge.
        allow_charged_fragments (bool, optional): Whether to allow charged fragments.
        use_graph (bool, optional): Whether to use the graph representation of the molecule.

    Returns:
        Tuple[np.ndarray, dict]:
            -  Best Bond orders.
            - Atomic valence electrons.
    """
    global atomic_valence
    global atomic_valence_electrons
    # Make a list of valences, e.g. for CO: [[4],[2,1]].
    valences_list_of_lists = []
    ac_valence = list(ac.sum(axis=1))
    for i, (atomic_num, valence) in enumerate(zip(atoms, ac_valence)):
        # The valence cannot be smaller than the number of neighbours.
        possible_valence = [x for x in atomic_valence[atomic_num] if x >= valence]
        if not possible_valence:
            logger.warning(f'Valence of atom {i} is {valence}, which bigger than the allowed max '
                           f'{max(atomic_valence[atomic_num])}. Stopping')
            return None, None
        valences_list_of_lists.append(possible_valence)
    # Convert [[4],[2,1]] to [[4,2],[4,1]].
    valences_list = itertools.product(*valences_list_of_lists)
    best_bo = ac.copy()
    for valences in valences_list:
        ua, du_from_ac = get_ua(valences, ac_valence)
        check_len = (len(ua) == 0)
        if check_len:
            check_bo = bo_is_ok(ac, ac, charge, du_from_ac, atomic_valence_electrons, atoms, valences,
                                allow_charged_fragments=allow_charged_fragments)
        else:
            check_bo = None
        if check_len and check_bo:
            return ac, atomic_valence_electrons
        ua_pairs_list = get_ua_pairs(ua, ac, use_graph=use_graph)
        for ua_pairs in ua_pairs_list:
            bo = get_bo(ac, du_from_ac, valences, ua_pairs, use_graph=use_graph)
            status = bo_is_ok(bo, ac, charge, du_from_ac, atomic_valence_electrons, atoms, valences,
                              allow_charged_fragments=allow_charged_fragments)
            charge_ok = is_charge_ok(bo, charge, atomic_valence_electrons, atoms,
                                     allow_charged_fragments=allow_charged_fragments)
            if status:
                return bo, atomic_valence_electrons
            elif bo.sum() >= best_bo.sum() and valences_not_too_large(bo, valences) and charge_ok:
                best_bo = bo.copy()
    return best_bo, atomic_valence_electrons


def ac2mol(mol,
           ac,
           atoms: List[int],
           charge: int,
           allow_charged_fragments: bool = True,
           use_graph: bool = True,
           ) -> Optional[list]:
    """

    Args:
        mol (RDMol) An rdkit molecule object instance.
        ac (np.ndarray): Atom connectivity.
        atoms (List[int]): Entries are integer atomic symbols.
        charge (int): The molecular charge.
        allow_charged_fragments (bool, optional): Whether to allow charged fragments.
        use_graph (bool, optional): Whether to use the graph representation of the molecule.

    Returns:
        List[RDMol]: Respective RDKit Molecule object instances.
    """
    # Convert ac matrix to bond order (bo) matrix.
    bo, atomic_valence_electrons_ = ac2bo(
        ac,
        atoms,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph,
    )
    if bo is None or atomic_valence_electrons_ is None:
        return None
    # Add bo connectivity and charge info to mol object.
    mol = bo2mol(
        mol,
        bo,
        atoms,
        atomic_valence_electrons_,
        charge,
        allow_charged_fragments=allow_charged_fragments,
    )
    if mol is None:
        return None
    # If charge is not correct don't return mol.
    if Chem.GetFormalCharge(mol) != charge:
        return []
    # bd2mol() returns an arbitrary resonance form. Let's make the rest.
    mols = rdchem.ResonanceMolSupplier(mol, Chem.UNCONSTRAINED_CATIONS, Chem.UNCONSTRAINED_ANIONS)
    mols = [mol for mol in mols]
    return mols


def get_proto_mol(atoms: List[int]):
    """
    Get a template of an RDKit Molecule object instance.

    Args:
        atoms (List[int]): Entries are integer atomic symbols.

    Returns:
        RDMol: The RDKit Molecule object instance.
    """
    mol = Chem.MolFromSmarts("[#" + str(atoms[0]) + "]")
    rw_mol = Chem.RWMol(mol)
    for i in range(1, len(atoms)):
        a = Chem.Atom(atoms[i])
        rw_mol.AddAtom(a)
    mol = rw_mol.GetMol()
    return mol


def xyz2ac(atoms,
           xyz: Union[List[List[float]], Tuple[Tuple[float, float, float], ...]],
           charge: int,
           use_huckel: bool = False,
           ) -> tuple:
    """
    Atoms and coordinates to atom connectivity (ac).

    Args:
        atoms (List[int]): Entries are integer representations of atom types.
        xyz (Union[List[List[float]], Tuple[Tuple[float, float, float], ...]]): The coordinates.
        charge (int): The molecular charge.
        use_huckel (bool, optional): Whether to use Huckel bond orders to locate bonds.
                                     Otherwise, van der Waals radii are used.

    Returns:
        Tuple[np.ndarray, RDMol]:
            - The atom connectivity matrix.
            - RDMol.
    """
    if use_huckel:
        return xyz2ac_huckel(atoms, xyz, charge)
    else:
        return xyz2ac_vdw(atoms, xyz)


def xyz2ac_huckel(atomic_num_list,
                  xyz,
                  charge,
                  ) -> tuple:
    """
    Generate an adjacency matrix from atoms and coordinates using the Huckle method.

    Args:
        atomic_num_list (List[int]): Entries are integer representations of atom types.
        xyz (Union[List[List[float]], Tuple[Tuple[float, float, float], ...]]): The coordinates.
        charge (int): The molecular charge.

    Returns:
        Tuple[np.ndarray, RDMol]:
            - The atom connectivity matrix.
            - RDMol.
    """
    mol = get_proto_mol(atomic_num_list)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)
    num_atoms = len(atomic_num_list)
    ac = np.zeros((num_atoms, num_atoms)).astype(int)
    mol_huckel = Chem.Mol(mol)
    mol_huckel.GetAtomWithIdx(0).SetFormalCharge(charge)  # Mol charge arbitrarily added to 1st atom.
    passed, result = rdEHTTools.RunMol(mol_huckel)
    opop = result.GetReducedOverlapPopulationMatrix()
    tri = np.zeros((num_atoms, num_atoms))
    tri[np.tril(np.ones((num_atoms, num_atoms), dtype=bool))] = opop  # Lower triangular to square matrix.
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            pair_pop = abs(tri[j, i])
            if pair_pop >= 0.15:  # Arbitrary cutoff for bond. May need adjustment.
                ac[i, j] = 1
                ac[j, i] = 1
    return ac, mol


def xyz2ac_vdw(atoms,
               xyz: Union[List[List[float]], Tuple[Tuple[float, float, float], ...]],
               ) -> tuple:
    """
    Generate an adjacency matrix from atoms and coordinates using the Van der Waals method.

    Args:
        atoms (List[int]): Entries are integer representations of atom types.
        xyz (Union[List[List[float]], Tuple[Tuple[float, float, float], ...]]): The coordinates.

    Returns:
        Tuple[np.ndarray, RDMol]:
            - The atom connectivity matrix.
            - RDMol.
    """
    mol = get_proto_mol(atoms)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)
    ac = get_ac(mol)
    return ac, mol


def get_ac(mol,
           covalent_factor: float = 1.3,
           ):
    """
    Generate an adjacency matrix from atoms and coordinates.
    ``ac`` is a (num_atoms, num_atoms) matrix with 1 being a covalent bond and 0 represents no bond.

    Args:
        mol (RDMol): An RDKit Molecule object instance with a 3D conformer.
        covalent_factor (float): A factor to multiply covalent bond lengths by.

    Returns:
        np.ndarray: The adjacency matrix.
    """
    dmat = Chem.Get3DDistanceMatrix(mol)
    pt = Chem.GetPeriodicTable()
    num_atoms = mol.GetNumAtoms()
    ac = np.zeros((num_atoms, num_atoms), dtype=int)
    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        r_cov_i = pt.GetRcovalent(a_i.GetAtomicNum()) * covalent_factor
        for j in range(i + 1, num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            r_cov_j = pt.GetRcovalent(a_j.GetAtomicNum()) * covalent_factor
            if dmat[i, j] <= r_cov_i + r_cov_j:
                ac[i, j] = 1
                ac[j, i] = 1
    return ac


def chiral_stereo_check(mol):
    """
    Find and embed chiral information into the model based on the coordinates.

    Args:
        mol (RDMol): The RDKit molecule object instance with an embedded 3D conformer.
    """
    Chem.SanitizeMol(mol)
    Chem.DetectBondStereochemistry(mol, -1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)


def xyz2mol(atoms: List[int],
            coordinates: Union[List[List[float]], Tuple[Tuple[float, float, float], ...]],
            charge: int = 0,
            allow_charged_fragments: bool = True,
            use_graph: bool = True,
            use_huckel: bool = False,
            embed_chiral: bool = True,
            ) -> Optional[list]:
    """
    Generate an RDKit Molecule object instance from atoms, coordinates, and an overall charge.

    Args:
        atoms (List[int]): list of atom types.
        coordinates (Union[List[List[float]], Tuple[Tuple[float, float, float], ...]]): A 3xN Cartesian coordinates.
        charge (int, optional): The total charge of the system (default: 0).
        allow_charged_fragments (bool, optional): Alternatively, radicals are made.
        use_graph (bool, optional): Use graph (networkx).
        use_huckel (bool, optional): Use Huckel method for atom connectivity perception.
        embed_chiral (bool, optional): Embed chiral information into the molecule.

    Returns:
        list[RDMol]: A list of corresponding RDKit Molecule object instances.
    """
    # Get the atom connectivity (ac) matrix, list of atomic numbers, molecular charge,
    # and mol object with no connectivity information.
    ac, mol = xyz2ac(atoms, coordinates, charge, use_huckel=use_huckel)
    # Convert ac to bond order matrix and add connectivity and charge info to a mol object.
    new_mols = ac2mol(mol, ac, atoms, charge,
                      allow_charged_fragments=allow_charged_fragments,
                      use_graph=use_graph)
    if new_mols is None:
        return None
    # Check for stereocenters and chiral centers.
    if embed_chiral:
        for new_mol in new_mols:
            chiral_stereo_check(new_mol)
    return new_mols
