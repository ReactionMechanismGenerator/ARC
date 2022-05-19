"""
A module for representing and manipulating `Z matrices <https://en.wikipedia.org/wiki/Z-matrix_(chemistry)>`_
(internal coordinates).

An example for a (consolidated) zmat representation for methane::

    {'symbols': ('C', 'H', 'H', 'H', 'H'),
     'coords': ((None, None, None),
                ('R_1_0', None, None),
                ('R_2|3|4_1|2|3', 'A_2|3|4_1|2|3_0|0|0', None),
                ('R_2|3|4_1|2|3', 'A_2|3|4_1|2|3_0|0|0', 'D_3_2_0_1'),
                ('R_2|3|4_1|2|3', 'A_2|3|4_1|2|3_0|0|0', 'D_4_3_0_2')),
     'vars': {'R_1_0': 1.09125,
              'D_3_2_0_1': 120.0,
              'D_4_3_0_2': 240.0,
              'R_2|3|4_1|2|3': 1.78200,
              'A_2|3|4_1|2|3_0|0|0': 35.26439},
     'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
     }

Isotope information is not saved in the zmat, it exists in the xyz dict,
and can be attained using the zmat map if needed.

Note that a 180 (or 0) degree for an angle in a Z matrix is not allowed, since a GIC (Generalized Internal Coordinates)
optimization has no defined derivative at 180 degrees. Instead, a dummy atom 'X' is used.
Dihedral angles may have any value.
"""

import math
import numpy as np
import operator
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from rmgpy.molecule.molecule import Molecule

from arc.common import get_logger, key_by_val, determine_top_group_indices
from arc.exceptions import ZMatError
from arc.species.vectors import calculate_distance, calculate_angle, calculate_dihedral_angle, get_vector_length

if TYPE_CHECKING:
    from rmgpy.molecule.molecule import Atom


logger = get_logger()

DEFAULT_CONSOLIDATION_R_TOL = 1e-4
DEFAULT_CONSOLIDATION_A_TOL = 1e-3
DEFAULT_CONSOLIDATION_D_TOL = 1e-3
DEFAULT_COMPARISON_R_TOL = 0.01  # Angstrom
DEFAULT_COMPARISON_A_TOL = 2.0  # degrees
DEFAULT_COMPARISON_D_TOL = 2.0  # degrees
TOL_180 = 0.9  # degrees
KEY_FROM_LEN = {2: 'R', 3: 'A', 4: 'D'}


def xyz_to_zmat(xyz: Dict[str, tuple],
                mol: Optional[Molecule] = None,
                constraints: Optional[Dict[str, List[Tuple[int, ...]]]] = None,
                consolidate: bool = True,
                consolidation_tols: Dict[str, float] = None,
                fragments: Optional[List[List[int]]] = None,
                ) -> Dict[str, tuple]:
    """
    Generate a z-matrix from cartesian coordinates.
    The zmat is a dictionary with the following keys:
    - 'symbols': a tuple of strings representing atomic symbols; defined as a list and converted to a tuple
    - 'coords': a tuple of tuples representing internal coordinates; defined as a list and converted to a tuple
    - 'vars': a dictionary of all variables defined for the coordinates
    - 'map': a dictionary connecting atom indices in the zmat (keys) to atom indices in the mol/coords (values)
    This function assumes ``xyz`` has no dummy atoms.
    This function does not attempt to resolve constrain locks, and assumes only few non-circular constraints were given.

    Args:
        xyz (dict): The xyz coordinates.
        mol (Molecule, optional): The corresponding RMG Molecule. If given, the bonding information will be used
                                  to generate a more meaningful zmat.
        constraints (dict, optional): Accepted keys are:
                                      'R_atom', 'R_group',
                                      'A_atom', 'A_group',
                                      'D_atom', 'D_group'.
                                      'R', 'A', and 'D' are constrain distances, angles, and dihedrals, respectively.
                                      Values are lists of atom index tuples (0-indexed). The atom indices order matters.
                                      Specifying '_atom' will cause only the first atom in the specified list values
                                      to translate/rotate if the corresponding zmat parameter is changed.
                                      Specifying '_group' will cause the entire group connected to the first atom
                                      to translate/rotate if the corresponding zmat parameter is changed.
                                      Specifying '_groups' (only valid for D) will cause the groups connected to
                                      the first two atoms to translate/rotate if the corresponding parameter is changed.
                                      Note that 'D_groups' should not be passed directly to this function, but to
                                      arc.species.converter.modify_coords() instead (it is a composite constraint).
        consolidate (bool, optional): Whether to consolidate the zmat after generation, ``True`` to consolidate.
        consolidation_tols (dict, optional): Keys are 'R', 'A', 'D', values are floats representing absolute tolerance
                                             for consolidating almost equal internal coordinates.
        fragments (List[List[int]], optional):
            Fragments represented by the species, i.e., as in a VdW well or a TS.
            Entries are atom index lists of all atoms in a fragment, each list represents a different fragment.
            indices are 0-indexed.

    Raises:
        ZMatError: If the zmat could not be generated.

    Returns: Dict[str, tuple]
        The z-matrix.
    """
    fragments = fragments or [list(range(len(xyz['symbols'])))]
    constraints = constraints or dict()
    if mol is None and any('group' in constraint_key for constraint_key in constraints.keys()):
        raise ZMatError(f'Cannot generate a constrained zmat without mol. Got mol=None and constraints=\n{constraints}')
    for constraint_list in constraints.values():
        for constraint_tuple in constraint_list:
            for index in constraint_tuple:
                if mol is not None and index >= len(mol.atoms):
                    raise ZMatError(f'The following constraints (containing atom index {index}) are invalid for '
                                    f'a molecule with only {len(mol.atoms)} atoms:\n{constraints}')
                if index >= len(xyz['symbols']):
                    raise ZMatError(f'The following constraints (containing atom index {index}) are invalid for '
                                    f'coordinates with only {len(xyz["symbols"])} atoms:\n{constraints}')
    xyz = xyz.copy()
    zmat = {'symbols': list(), 'coords': list(), 'vars': dict(), 'map': dict()}
    atom_order = get_atom_order(xyz=xyz, mol=mol, constraints_dict=constraints, fragments=fragments)
    connectivity = get_connectivity(mol=mol) if mol is not None else None
    skipped_atoms = list()  # atoms for which constrains are applied
    for atom_index in atom_order:
        zmat, xyz, skipped = _add_nth_atom_to_zmat(
            zmat=zmat,
            xyz=xyz,
            connectivity=connectivity,
            n=len(list(zmat['symbols'])),
            atom_index=atom_index,
            constraints=constraints,
            fragments=fragments,
        )
        skipped_atoms.extend(skipped)

    while len(skipped_atoms):
        num_of_skipped_atoms = len(skipped_atoms)
        indices_to_pop = list()
        for i, atom_index in enumerate(skipped_atoms):
            zmat, xyz, skipped = _add_nth_atom_to_zmat(
                zmat=zmat,
                xyz=xyz,
                connectivity=connectivity,
                n=len(list(zmat['symbols'])),
                atom_index=atom_index,
                constraints=constraints,
                fragments=fragments,
            )
            if not len(skipped):
                # This atom was not skipped this time, remove it from the skipped atoms list....
                indices_to_pop.append(i)
        for i in reversed(indices_to_pop):
            skipped_atoms.pop(i)
        if num_of_skipped_atoms == len(skipped_atoms):
            # No atoms were popped from the skipped atoms list when iterating through all skipped atoms.
            raise ZMatError(f"Could not generate the zmat, skipped atoms could not be assigned, there's probably "
                            f"a constraint lock. The partial zmat is:\n{zmat}\n\nskipped atoms are:\n{skipped_atoms}.")

    if consolidate and not constraints:
        try:
            zmat = consolidate_zmat(zmat, mol, consolidation_tols)
        except (KeyError, ZMatError) as e:
            logger.error(f'Could not consolidate zmat, got:\n{e.__class__}: {str(e)}')
            logger.error(f'Generating zmat without consolidation.')

    zmat['symbols'] = tuple(zmat['symbols'])
    zmat['coords'] = tuple(zmat['coords'])
    return zmat


def determine_r_atoms(zmat: Dict[str, Union[dict, tuple]],
                      xyz: Dict[str, tuple],
                      connectivity: Dict[int, List[int]],
                      n: int,
                      atom_index: int,
                      r_constraint: Optional[Tuple[int]] = None,
                      a_constraint: Optional[Tuple[int, int]] = None,
                      d_constraint: Optional[Tuple[int, int, int]] = None,
                      trivial_assignment: bool = False,
                      fragments: Optional[List[List[int]]] = None,
                      ) -> Optional[List[int]]:
    """
    Determine the atoms for defining the distance R.
    This should be in the form: [n, <some other atom already in the zmat>]

    Args:
        zmat (dict): The zmat.
        xyz (dict): The xyz dict.
        connectivity (dict): The atoms connectivity (keys are indices in the mol/xyz).
        n (int): The 0-index of the atom in the zmat to be added.
        atom_index (int): The 0-index of the atom in the molecule or cartesian coordinates to be added.
                          (``n`` and ``atom_index`` refer to the same atom, but it might have different indices
                          in the zmat and the molecule/xyz)
        r_constraint (tuple, optional): R-type constraints. The atom index to which the atom being checked is
                                        constrained. ``None`` if it is not constrained.
        a_constraint (tuple, optional): A-type constraints. The atom indices to which the atom being checked is
                                        constrained. ``None`` if it is not constrained.
        d_constraint (tuple, optional): D-type constraints. The atom indices to which the atom being checked is
                                        constrained. ``None`` if it is not constrained.
        trivial_assignment (bool, optional): Whether to attempt assigning atoms without considering connectivity
                                             if the connectivity assignment fails.
        fragments (List[List[int]], optional):
            Fragments represented by the species, i.e., as in a VdW well or a TS.
            Entries are atom index lists of all atoms in a fragment, each list represents a different fragment.
            indices are 0-indexed.

    Raises:
        ZMatError: If the R atoms could not be determined.

    Returns: Optional[List[int]]
        The 0-indexed z-mat R atoms.
    """
    if is_atom_in_new_fragment(atom_index=atom_index, zmat=zmat, fragments=fragments):
        connectivity = None
    if len(zmat['coords']) == 0:
        # This is the 1st atom added to the zmat, there's no distance definition here.
        r_atoms = None
    elif any(constraint is not None for constraint in [r_constraint, a_constraint, d_constraint]):
        # 1. Always use the constraint if given.
        if r_constraint is not None:
            r_atoms = [n] + [key_by_val(zmat['map'], r_constraint[1])]
        elif a_constraint is not None:
            r_atoms = [n] + [key_by_val(zmat['map'], a_constraint[1])]
        elif d_constraint is not None:
            r_atoms = [n] + [key_by_val(zmat['map'], d_constraint[1])]
    elif connectivity is not None:
        # 2. Use connectivity if the atom is not constrained.
        r_atoms = [n]
        atom_dict = dict()  # Keys are neighbor atom indices, values are tuples of depth and linearity.
        for atom_c in connectivity[atom_index]:
            # Code adopted from arc.common.determine_top_group_indices():.
            linear = True  # Assume the chain from atom_index towards atom_c is linear unless otherwise proven.
            explored_atom_list, atom_list_to_explore1, atom_list_to_explore2, top = [atom_index], [atom_c], [], []
            while len(atom_list_to_explore1 + atom_list_to_explore2):
                for atom3 in atom_list_to_explore1:
                    top.append(atom3)
                    if atom3 in connectivity:
                        for atom4 in connectivity[atom3]:
                            if atom4 not in explored_atom_list and atom4 not in atom_list_to_explore2:
                                if xyz['symbols'][atom4] in ['H', 'F', 'Cl', 'Br', 'I', 'X']:
                                    # Append w/o further exploration.
                                    top.append(atom4)
                                else:
                                    atom_list_to_explore2.append(atom4)  # Explore it further.
                    explored_atom_list.append(atom3)  # Mark as explored.
                atom_list_to_explore1, atom_list_to_explore2 = atom_list_to_explore2, []
                if len(top) >= 2:
                    # Calculate the angle formed with the index_atom.
                    angle = calculate_angle(coords=xyz['coords'], atoms=[atom_index] + top[-2:], index=0, units='degs')
                    if not is_angle_linear(angle):
                        linear = False
                        if len(top) >= 3:
                            # If it's not linear and there are 3 or more atoms, we have all the info that we need.
                            break
            atom_dict[atom_c] = (len(top), linear)
        long_non_linear, long_linear, two_non_linear, two_linear, one = list(), list(), list(), list(), list()
        for atom_c, val in atom_dict.items():
            if atom_c in list(zmat['map'].values()):
                zmat_index_c = key_by_val(zmat['map'], atom_c)
                if not is_dummy(zmat, zmat_index_c):
                    if val[0] >= 3 and not val[1]:
                        long_non_linear.append(zmat_index_c)
                    elif val[0] >= 3 and val[1]:
                        long_linear.append(zmat_index_c)
                    elif val[0] == 2 and not val[1]:
                        two_non_linear.append(zmat_index_c)
                    elif val[0] == 2 and val[1]:
                        two_linear.append(zmat_index_c)
                    elif val[0] == 1:
                        one.append(zmat_index_c)
        if len(long_non_linear):
            r_atoms.append(long_non_linear[0])
        elif len(two_non_linear):
            r_atoms.append(two_non_linear[0])
        elif len(long_linear):
            r_atoms.append(long_linear[0])
        elif len(two_linear):
            r_atoms.append(two_linear[0])
        elif len(one):
            r_atoms.append(one[0])
        if len(set(r_atoms)) != 2:
            if not trivial_assignment:
                raise ZMatError(f'Could not come up with two unique r_atoms from connectivity (r_atoms = {r_atoms}).')
    else:
        trivial_assignment = True
        r_atoms = list()
    if trivial_assignment and isinstance(r_atoms, list) and len(r_atoms) != 2:
        # 3. Use trivial atom assigment if constraint and connectivity were not given.
        r_atoms = [n]
        if len(zmat['coords']) in [1, 2]:
            r_atoms.append(len(zmat['coords']) - 1)
        else:
            for i in reversed(range(n)):
                if not is_dummy(zmat, i):
                    r_atoms.append(i)
                    break
        if len(set(r_atoms)) != 2:
            raise ZMatError(f'Could not come up with two unique non-dummy r_atoms (r_atoms = {r_atoms}).')
    if r_atoms is not None and r_atoms[-1] not in list(zmat['map'].keys()):
        raise ZMatError(f'The reference R atom {r_atoms[-1]} for the index atom {atom_index} has not been '
                        f'added to the zmat yet. Added atoms are (zmat index: xyz index): {zmat["map"]}.')
    if r_atoms is not None and len(set(r_atoms)) != 2:
        raise ZMatError(f'Could not come up with two unique r_atoms (r_atoms = {r_atoms}).')
    return r_atoms


def determine_a_atoms(zmat: Dict[str, Union[dict, tuple]],
                      coords: Union[list, tuple],
                      connectivity: Dict[int, List[int]],
                      r_atoms: Optional[List[int]],
                      n: int,
                      atom_index: int,
                      a_constraint: Optional[Tuple[int, int]] = None,
                      d_constraint: Optional[Tuple[int, int, int]] = None,
                      a_constraint_type: Optional[str] = None,
                      trivial_assignment: bool = False,
                      fragments: Optional[List[List[int]]] = None,
                      ) -> Optional[List[int]]:
    """
    Determine the atoms for defining the angle A.
    This should be in the form: [n, r_atoms[1], <some other atom already in the zmat>]

    Args:
        zmat (dict): The zmat.
        coords (list, tuple): Just the 'coords' part of the xyz dict.
        connectivity (dict): The atoms connectivity (keys are indices in the mol/xyz).
        r_atoms (list): The determined r_atoms.
        n (int): The 0-index of the atom in the zmat to be added.
        atom_index (int): The 0-index of the atom in the molecule or cartesian coordinates to be added.
                          (``n`` and ``atom_index`` refer to the same atom, but it might have different indices
                          in the zmat and the molecule/xyz)
        a_constraint (tuple, optional): A-type constraints. The atom indices to which the atom being checked is
                                        constrained. ``None`` if it is not constrained.
        d_constraint (tuple, optional): D-type constraints. The atom indices to which the atom being checked is
                                        constrained. ``None`` if it is not constrained.
        a_constraint_type (str, optional): The A constraint type ('A_atom', or 'A_group').
        trivial_assignment (bool, optional): Whether to attempt assigning atoms without considering connectivity
                                             if the connectivity assignment fails.
        fragments (List[List[int]], optional):
            Fragments represented by the species, i.e., as in a VdW well or a TS.
            Entries are atom index lists of all atoms in a fragment, each list represents a different fragment.

    Raises:
        ZMatError: If the A atoms could not be determined.
            indices are 0-indexed.

    Returns: Optional[List[int]]
        The 0-indexed z-mat A atoms.
    """
    if r_atoms is not None and is_atom_in_new_fragment(atom_index=atom_index, zmat=zmat,
                                                       fragments=fragments, skip_atoms=r_atoms):
        connectivity = None
    if r_atoms is not None and len(r_atoms) != 2:
        raise ZMatError(f'r_atoms must be a list of length 2, got {r_atoms}')
    if len(zmat['coords']) <= 1:
        # This is the 1st or 2nd atom added to the zmat, there's no angle definition here.
        a_atoms = None
    elif a_constraint is not None:
        # always use the constraint if given
        if a_constraint_type not in ['A_atom', 'A_group', None]:
            raise ZMatError(f'Got an invalid A constraint type "{a_constraint_type}" for {a_constraint}')
        a_atoms = [n] + [key_by_val(zmat['map'], atom) for atom in a_constraint[1:]]
    elif d_constraint is not None:
        # consider d_constraint for consistency if a_constraint is None
        a_atoms = [n] + [key_by_val(zmat['map'], atom) for atom in d_constraint[1:3]]
    elif connectivity is not None:
        a_atoms = [atom for atom in r_atoms]
        for atom in connectivity[zmat['map'][a_atoms[-1]]] + connectivity[atom_index]:
            # An angle should be defined between an atom, its neighbor, and its neighbor's neighbor.
            if atom in list(zmat['map'].values()):
                zmat_index = key_by_val(zmat['map'], atom)
                if atom != atom_index and zmat_index not in a_atoms and not is_dummy(zmat, zmat_index):
                    # Check whether this atom (B) is part of a linear chain. If it is, try to correctly determine
                    # dihedrals in this molecule w/o this atom, otherwise it's meaningless, and the zmat looses info.
                    #
                    #                     D (atom_index, r_atoms[0])
                    #                    /
                    #    E -- A -- B -- C  (r_atoms[1])
                    #   /
                    #  F          (B is the atom considered here, corresponding to 'atom' / 'zmat_index')
                    i = 0
                    atom_b, atom_c = atom, zmat['map'][r_atoms[1]]
                    while i < len(list(connectivity.keys())):
                        num_of_neighbors = len(list(connectivity[atom_b]))
                        if num_of_neighbors == 1:
                            # Atom B is only connected to C, no need to intervene.
                            break
                        elif num_of_neighbors == 2:
                            # Atom B might be in a linear chain, determine the A -- B -- C angle.
                            b_neighbors = connectivity[atom_b]
                            atom_a = b_neighbors[0] if b_neighbors[0] != atom_c else b_neighbors[1]
                            angle = calculate_angle(coords=coords, atoms=[atom_a, atom_b, atom_c])
                            if is_angle_linear(angle):
                                # A -- B -- C is linear, change indices and test angle E -- A -- B instead.
                                atom_c = atom_b
                                atom_b = atom_a
                            elif key_by_val(zmat['map'], atom_b) not in a_atoms:
                                # A -- B -- C is not linear, use atom B.
                                zmat_index = key_by_val(zmat['map'], atom_b)
                        elif num_of_neighbors > 2:
                            # Atom B does not necessarily lead to a linear A -- B -- C chain, no need to intervene.
                            zmat_index = key_by_val(zmat['map'], atom_b)
                            break
                        i += 1  # Don't loop forever.
                    a_atoms.append(zmat_index)
                    break
        if len(set(a_atoms)) != 3:
            if not trivial_assignment:
                raise ZMatError(f'Could not come up with three unique a_atoms from connectivity (a_atoms = {a_atoms}).')
    else:
        trivial_assignment = True
        a_atoms = list()
    if trivial_assignment and isinstance(a_atoms, list) and len(a_atoms) != 3:
        a_atoms = [atom for atom in r_atoms]
        for i in reversed(range(n)):
            # Check whether this atom (B) is part of a linear chain. If it is, try to correctly determine
            # dihedrals in this molecule w/o this atom, otherwise it's meaningless, and the zmat looses info.
            #
            #                     D (atom_index, r_atoms[0])
            #                    /
            #    E -- A -- B -- C  (r_atoms[1])
            #   /
            #  F          (B is the atom considered here)
            zmat_index = i
            if i not in a_atoms and i in list(zmat['map'].keys()) and not is_dummy(zmat, i):
                zmat_index, j = i, n - 1
                atom_b, atom_c = zmat['map'][i], zmat['map'][r_atoms[1]]
                while j > 0:
                    atom_a = zmat['map'][j]
                    if j != i and atom_a not in [atom_b, atom_c] \
                            and (j in list(zmat['map'].keys()) and not is_dummy(zmat, j)
                                 or j not in list(zmat['map'].keys())):
                        angle = calculate_angle(coords=coords, atoms=[atom_a, atom_b, atom_c])
                        if is_angle_linear(angle):
                            # A -- B -- C is linear, change indices and test angle E -- A -- B.
                            atom_b = atom_a
                        elif zmat_index not in a_atoms:
                            # A -- B -- C is not linear, use atom B.
                            zmat_index = key_by_val(zmat['map'], atom_b)
                            a_atoms.append(zmat_index)
                            break
                    j -= 1  # Don't loop forever.
            if len(a_atoms) == 3:
                break
        if len(a_atoms) == 2 and zmat_index not in a_atoms:
            a_atoms.append(zmat_index)
    if a_atoms is not None and any([a_atom not in list(zmat['map'].keys()) for a_atom in a_atoms[1:]]):
        raise ZMatError(f'The reference A atom in {a_atoms} for the index atom {atom_index} has not been '
                        f'added to the zmat yet. Added atoms are (zmat index: xyz index): {zmat["map"]}.')
    if a_atoms is not None and len(set(a_atoms)) != 3:
        raise ZMatError(f'Could not come up with three unique a_atoms (a_atoms = {a_atoms}).')
    return a_atoms


def determine_d_atoms(zmat: Dict[str, Union[dict, tuple]],
                      xyz: Dict[str, tuple],
                      coords: Union[list, tuple],
                      connectivity: Dict[int, List[int]],
                      a_atoms: Optional[List[int]],
                      n: int,
                      atom_index: int,
                      d_constraint: Optional[Tuple[int, int, int]] = None,
                      d_constraint_type: Optional[str] = None,
                      specific_atom: Optional[int] = None,
                      dummy: bool = False,
                      fragments: Optional[List[List[int]]] = None,
                      ) -> Optional[List[int]]:
    """
    Determine the atoms for defining the dihedral angle D.
    This should be in the form: [n, a_atoms[1], a_atoms[2], <some other atom already in the zmat>]

    Args:
        zmat (dict): The zmat.
        xyz (dict): The xyz dict.
        coords (list, tuple): Just the 'coords' part of the xyz dict.
        connectivity (dict): The atoms connectivity (keys are indices in the mol/xyz).
        a_atoms (list): The determined a_atoms.
        n (int): The 0-index of the atom in the zmat to be added.
        atom_index (int): The 0-index of the atom in the molecule or cartesian coordinates to be added.
                          (``n`` and ``atom_index`` refer to the same atom, but it might have different indices
                          in the zmat and the molecule/xyz)
        d_constraint (tuple, optional): D-type constraints. The atom indices to which the atom being checked is
                                        constrained. ``None`` if it is not constrained.
        d_constraint_type (str, optional): The D constraint type ('D_atom', or 'D_group').
        specific_atom (int, optional): A 0-index of the zmat atom to be added to a_atoms to create d_atoms.
        dummy (bool, optional): Whether the atom being added (n) represents a dummy atom. ``True`` if it does.
        fragments (List[List[int]], optional):
            Fragments represented by the species, i.e., as in a VdW well or a TS.
            Entries are atom index lists of all atoms in a fragment, each list represents a different fragment.

    Raises:
        ZMatError: If the A atoms could not be determined.
            indices are 0-indexed.

    Returns:
        Optional[List[int]]: The 0-indexed z-mat D atoms.
    """
    if a_atoms is not None and is_atom_in_new_fragment(atom_index=atom_index, zmat=zmat,
                                                       fragments=fragments, skip_atoms=a_atoms):
        connectivity = None
    if a_atoms is not None and len(a_atoms) != 3:
        raise ZMatError(f'a_atoms must be a list of length 3, got {a_atoms}')
    if len(zmat['coords']) <= 2:
        # This is the 1st, 2nd, or 3rd atom added to the zmat, there's no dihedral angle definition here.
        d_atoms = None
    elif d_constraint is not None:
        if d_constraint_type not in ['D_atom', 'D_group']:
            raise ZMatError(f'Got an invalid D constraint type "{d_constraint_type}" for {d_constraint}')
        d_atoms = [n] + [key_by_val(zmat['map'], atom) for atom in d_constraint[1:]]
    elif specific_atom is not None and a_atoms is not None:
        # A specific atom was specified (e.g., a dummy atom was added to assist defining this atom), consider it.
        if not isinstance(specific_atom, int):
            raise ZMatError(f'specific atom must be of type int, got {type(specific_atom)}')
        d_atoms = a_atoms + [specific_atom]
    elif connectivity is not None:
        d_atoms = determine_d_atoms_from_connectivity(zmat, xyz, coords, connectivity, a_atoms, atom_index,
                                                      dummy=dummy, allow_a_to_be_dummy=False)
        if len(d_atoms) < 4:
            d_atoms = determine_d_atoms_without_connectivity(zmat, coords, a_atoms, n)
            if len(d_atoms) < 4:
                d_atoms = determine_d_atoms_from_connectivity(zmat, xyz, coords, connectivity, a_atoms, atom_index,
                                                              dummy=dummy, allow_a_to_be_dummy=True)
    else:
        d_atoms = determine_d_atoms_without_connectivity(zmat, coords, a_atoms, n)
    if d_atoms is not None:
        if len(d_atoms) < 4:
            for i in reversed(range(len(xyz['symbols']))):
                if i not in d_atoms and i in list(zmat['map'].keys()):
                    angle = calculate_angle(coords=coords, atoms=[zmat['map'][z_index]
                                                                  for z_index in d_atoms[1:] + [i]])
                    if not is_angle_linear(angle):
                        d_atoms.append(i)
                        break
        if len(set(d_atoms)) != 4:
            logger.error(f'Could not come up with four unique d_atoms (d_atoms = {d_atoms}). '
                         f'Setting d_atoms to [{n}, 2, 1, 0]')
            d_atoms = [n, 2, 1, 0]
        if any([d_atom not in list(zmat['map'].keys()) for d_atom in d_atoms[1:]]):
            raise ZMatError(f'A reference D atom in {d_atoms} for the index atom {atom_index} has not been '
                            f'added to the zmat yet. Added atoms are (zmat index: xyz index): {zmat["map"]}.')
    return d_atoms


def determine_d_atoms_without_connectivity(zmat: dict,
                                           coords: Union[list, tuple],
                                           a_atoms: list,
                                           n: int,
                                           ) -> list:
    """
    A helper function to determine d_atoms without connectivity information.

    Args:
        zmat (dict): The zmat.
        coords (Union[list, tuple]): Just the 'coords' part of the xyz dict.
        a_atoms (list): The determined a_atoms.
        n (int): The 0-index of the atom in the zmat to be added.

    Returns:
        list: The d_atoms.
    """
    d_atoms = [atom for atom in a_atoms]
    for i in reversed(range(n)):
        if i not in d_atoms and i in list(zmat['map'].keys()) and (i >= len(zmat['symbols']) or not is_dummy(zmat, i)):
            angle = calculate_angle(coords=coords, atoms=[zmat['map'][z_index] for z_index in d_atoms[1:] + [i]])
            if not is_angle_linear(angle):
                d_atoms.append(i)
                break
    if len(d_atoms) < 4:
        # Try again and consider dummies.
        for i in reversed(range(n)):
            if i not in d_atoms and i in list(zmat['map'].keys()):
                angle = calculate_angle(coords=coords, atoms=[zmat['map'][z_index] for z_index in d_atoms[1:] + [i]])
                if not is_angle_linear(angle):
                    d_atoms.append(i)
                    break
    return d_atoms


def determine_d_atoms_from_connectivity(zmat: dict,
                                        xyz: dict,
                                        coords: Union[list, tuple],
                                        connectivity: dict,
                                        a_atoms: list,
                                        atom_index: int,
                                        dummy: bool = False,
                                        allow_a_to_be_dummy: bool = False,
                                        ) -> list:
    """
    A helper function to determine d_atoms from the connectivity information.

    Args:
        zmat (dict): The zmat.
        xyz (dict): The xyz dict.
        coords (Union[list, tuple]): Just the 'coords' part of the xyz dict.
        connectivity (dict): The atoms connectivity (keys are indices in the mol/xyz).
        a_atoms (list): The determined a_atoms.
        atom_index (int): The 0-index of the atom in the molecule or cartesian coordinates to be added.
                          (``n`` and ``atom_index`` refer to the same atom, but it might have different indices
                          in the zmat and the molecule/xyz)
        dummy (bool, optional): Whether the atom being added (n) represents a dummy atom. ``True`` if it does.
        allow_a_to_be_dummy (bool, optional): Whether the last atom ('A') in d_atoms is allowed to by a dummy atom.
                                              ``True`` if it is, ``False`` by default.

    Returns:
        list: The d_atoms.
    """
    d_atoms = [atom for atom in a_atoms]
    for atom in connectivity[zmat['map'][d_atoms[-1]]] + connectivity[zmat['map'][d_atoms[-2]]] \
            + connectivity[atom_index]:
        if atom != atom_index and atom in list(zmat['map'].values()) \
                and (not is_dummy(zmat, key_by_val(zmat['map'], atom)) or (not dummy and allow_a_to_be_dummy)):
            # Atom A is allowed to be a dummy atom only if the atom represented by n is not.
            zmat_index = None
            if atom not in list([zmat['map'][d_atom] for d_atom in d_atoms[1:]]):
                # Check whether this atom (A) is part of a linear chain. If it is, try to correctly determine
                # dihedrals in this molecule w/o this atom, otherwise it's meaningless, and the zmat looses info.
                #
                #             X       D (atom_index)
                #              \     /
                #    E -- A -- B -- C  (r_atoms[1])
                #   /
                #  F     (A is the atom considered here, corresponding to 'atom' / 'zmat_index')
                i = 0
                atom_a, atom_b, atom_c = atom, zmat['map'][d_atoms[2]], zmat['map'][d_atoms[1]]
                while i < len(list(connectivity.keys())):
                    angle = calculate_angle(coords=coords, atoms=[atom_a, atom_b, atom_c])
                    if is_angle_linear(angle):
                        num_of_neighbors = len(list(connectivity[atom_a]))
                        if num_of_neighbors == 1:
                            # Atom A is only connected to B, use the dummy atom on atom B as atom A.
                            b_neighbors = connectivity[atom_b]
                            x_neighbor = [neighbor for neighbor in b_neighbors
                                          if xyz['symbols'][neighbor] == 'X'][0]
                            if key_by_val(zmat['map'], f'X{x_neighbor}') not in d_atoms:
                                zmat_index = key_by_val(zmat['map'], f'X{x_neighbor}')
                                break
                        elif num_of_neighbors == 2:
                            # atom A is only connected to B and E, check the E -- B -- C angle.
                            a_neighbors = connectivity[atom_a]
                            atom_e = a_neighbors[0] if a_neighbors[0] != atom_b else a_neighbors[1]
                            if atom_e in list(zmat['map'].values()):
                                angle = calculate_angle(coords=coords, atoms=[atom_e, atom_b, atom_c])
                                if is_angle_linear(angle):
                                    # E -- B -- C is linear, change indices and test angle F -- B -- C.
                                    atom_a = atom_e
                                elif key_by_val(zmat['map'], atom_e) not in d_atoms:
                                    # E -- B -- C is not linear, use atom E.
                                    zmat_index = key_by_val(zmat['map'], atom_e)
                                    break
                        elif num_of_neighbors > 2:
                            # Atom A is connected to at least one other atom not in this linear chain.
                            for a_neighbor in connectivity[atom_a]:
                                if a_neighbor != atom_b:
                                    angle = calculate_angle(coords=coords, atoms=[a_neighbor, atom_b, atom_c])
                                    if not is_angle_linear(angle) \
                                            and a_neighbor in list(zmat['map'].values()) \
                                            and key_by_val(zmat['map'], a_neighbor) not in d_atoms:
                                        # E -- B -- C is not linear, use atom E (a_neighbor).
                                        zmat_index = key_by_val(zmat['map'], a_neighbor)
                                        break
                    elif atom_a in list(zmat['map'].values()):
                        zmat_index = key_by_val(zmat['map'], atom_a)
                        break
                    i += 1  # Don't loop forever.
            if zmat_index is None and len(d_atoms) == 3 and 'X' in zmat['symbols'] \
                    and not dummy and allow_a_to_be_dummy:
                # Still could not find a non-linear forth d atom, look for dummy atoms.
                dummies = [(key, int(val[1:])) for key, val in zmat['map'].items() if re.match(r'X\d', str(val))]
                for dummy in dummies:
                    zmat_index = dummy[0]
            if zmat_index is not None:
                d_atoms.append(zmat_index)
            break
    if len(d_atoms) == 3 and len(connectivity[atom_index]) > 2 \
            and connectivity[atom_index][2] in list(zmat['map'].values()) \
            and connectivity[atom_index][2] not in [zmat['map'][d_atom] for d_atom in d_atoms[1:]]:
        angle = calculate_angle(coords=coords, atoms=[zmat['map'][d_atom] for d_atom in d_atoms[1:]]
                                                     + [connectivity[atom_index][2]])
        if not is_angle_linear(angle) \
                and connectivity[atom_index][2] in list(zmat['map'].values()) \
                and key_by_val(zmat['map'], connectivity[atom_index][2]) not in d_atoms:
            d_atoms.append(key_by_val(zmat['map'], connectivity[atom_index][2]))
    return d_atoms


def _add_nth_atom_to_zmat(zmat: Dict[str, Union[dict, tuple]],
                          xyz: Dict[str, tuple],
                          connectivity: Dict[int, List[int]],
                          n: int,
                          atom_index: int,
                          constraints: Dict[str, List[Tuple[int]]],
                          fragments: List[List[int]],
                          ) -> Tuple[Dict[str, tuple], Dict[str, tuple], List[int]]:
    """
    Add the n-th atom to the zmat (n >= 0).
    Also considers the special cases where ``n`` is the first, second, or third atom to be added to the zmat.
    Adds a dummy atom if an angle (not a dihedral angle) is 180 (or 0) degrees.

    Args:
        zmat (dict): The zmat.
        xyz (dict): The coordinates.
        connectivity (dict): The atoms connectivity (keys are indices in the mol/xyz).
        n (int): The 0-index of the atom in the zmat to be added.
        atom_index (int): The 0-index of the atom in the molecule or cartesian coordinates to be added.
                          (``n`` and ``atom_index`` refer to the same atom, but it might have different indices
                          in the zmat and the molecule/xyz)
        constraints (dict): Constraints to consider.
                            Accepted keys are:
                           'R_atom', 'R_group',
                           'A_atom', 'A_group',
                           'D_atom', 'D_group'.
        fragments (List[List[int]]):
            Fragments represented by the species, i.e., as in a VdW well or a TS.
            Entries are atom index lists of all atoms in a fragment, each list represents a different fragment.
            indices are 0-indexed.

    Raises:
        ZMatError: If the zmat could not be generated.

    Returns:
        Tuple[Dict[str, tuple], Dict[str, tuple], List[int]]:
          - The updated zmat.
          - The xyz coordinates updated with dummy atoms.
          - A 0- or 1-length list with the skipped atom index.
    """
    coords = xyz['coords']
    skipped_atoms = list()
    specific_last_d_atom = None
    r_constraint, r_constraint_type = check_atom_r_constraints(atom_index, constraints)
    a_constraint, a_constraint_type = check_atom_a_constraints(atom_index, constraints)
    d_constraint, d_constraint_type = check_atom_d_constraints(atom_index, constraints)
    if sum([constraint is not None for constraint in [r_constraint, a_constraint, d_constraint]]) > 1:
        raise ZMatError(f'A single atom cannot be constrained by more than one constraint type, got:\n'
                        f'R {r_constraint_type}: {r_constraint}\n'
                        f'A {a_constraint_type}: {a_constraint}\n'
                        f'D {d_constraint_type}: {d_constraint}')
    r_constraints_passed, a_constraints_passed, d_constraints_passed = \
        [constraint is None or all([entry in list(zmat['map'].values()) for entry in constraint[1:]])
         for constraint in [r_constraint, a_constraint, d_constraint]]

    if all([passed for passed in [r_constraints_passed, a_constraints_passed, d_constraints_passed]]):
        # Add the n-th atom to the zmat.

        # If an '_atom' constraint was specified, only consider this atom if n is the last atom to consider.
        if (r_constraint_type == 'R_atom' or a_constraint_type == 'A_atom' or d_constraint_type == 'D_atom') \
                and n != len([symbol for symbol in xyz['symbols'] if 'X' not in symbol]) - 1:
            logger.debug(f'Skipping atom index {atom_index} when creating a zmat due to a specified _atom constraint.')
            skipped_atoms.append(atom_index)
            return zmat, xyz, skipped_atoms

        # Determine the atoms for defining the distance, R; this should be [n, <some other atom already in the zmat>].
        r_atoms = determine_r_atoms(
            zmat,
            xyz,
            connectivity,
            n,
            atom_index,
            r_constraint,
            a_constraint,
            d_constraint,
            trivial_assignment=any('_atom' in constraint_key for constraint_key in constraints.keys()),
            fragments=fragments,
        )
        # Determine the atoms for defining the angle, A.
        if a_constraint is None and d_constraint is not None:
            # If a D constraint is given, the A constraint must obey it as well.
            a_constraint = d_constraint[:-1]
        a_atoms = determine_a_atoms(
            zmat,
            coords,
            connectivity,
            r_atoms,
            n,
            atom_index,
            a_constraint,
            d_constraint,
            a_constraint_type,
            trivial_assignment=any('_atom' in constraint_key for constraint_key in constraints.keys()),
            fragments=fragments,
        )

        # Calculate the angle, add a dummy atom if needed.
        added_dummy = False
        if a_atoms is not None and all([not re.match(r'X\d', str(zmat['map'][atom])) for atom in a_atoms[1:]]):
            angle = calculate_angle(coords=coords, atoms=[atom_index] + [zmat['map'][atom] for atom in a_atoms[1:]])
            if is_angle_linear(angle):
                # The angle is too close to 180 (or 0) degrees, add a dummy atom.
                zmat, coords, n, r_atoms, a_atoms, specific_last_d_atom = \
                    add_dummy_atom(zmat, xyz, coords, connectivity, r_atoms, a_atoms, n, atom_index)
                added_dummy = True

        # Determine the atoms for defining the dihedral angle, D.
        d_atoms = determine_d_atoms(
            zmat,
            xyz,
            coords,
            connectivity,
            a_atoms,
            n,
            atom_index,
            d_constraint,
            d_constraint_type, specific_atom=specific_last_d_atom,
            fragments=fragments,
        )

        # Update the zmat.
        zmat = update_zmat_with_new_atom(zmat, xyz, coords, n, atom_index, r_atoms, a_atoms, d_atoms, added_dummy)

    else:
        # Some constraints did not "pass": some atoms were not added to the zmat yet; skip this atom until they are.
        skipped_atoms.append(atom_index)

    xyz['coords'] = coords  # Update xyz with the updated coords.
    return zmat, xyz, skipped_atoms


def update_zmat_with_new_atom(zmat: dict,
                              xyz: dict,
                              coords: Union[list, tuple],
                              n: int,
                              atom_index: int,
                              r_atoms: list,
                              a_atoms: list,
                              d_atoms: list,
                              added_dummy: bool = False,
                              ) -> dict:
    """
    Update the zmat with a new atom.

    Args:
        zmat (dict): The zmat.
        xyz (dict): The xyz dict.
        coords (Union[list, tuple]): Just the 'coords' part of the xyz dict.
        n (int): The 0-index of the atom in the zmat to be added.
        atom_index (int): The 0-index of the atom in the molecule or cartesian coordinates to be added.
                          (``n`` and ``atom_index`` refer to the same atom, but it might have different indices
                          in the zmat and the molecule/xyz)
        r_atoms (list): The R atom index descriptors.
        a_atoms (list): The A atom index descriptors.
        d_atoms (list): The D atom index descriptors.
        added_dummy (bool, optional): Whether a dummy atom was added as the last index of a_atoms.

    Returns:
        dict: The updated zmat.
    """
    zmat['symbols'].append(xyz['symbols'][atom_index])
    zmat['map'][n] = atom_index
    if r_atoms is None:
        r_str = None
    else:
        x = 'X' if any(['X' in list([symbol for symbol in zmat['symbols'][atom]]) for atom in r_atoms]) else ''
        r_str = f'R{x}_{r_atoms[0]}_{r_atoms[1]}'
    if a_atoms is None:
        a_str = None
    else:
        x = 'X' if any(['X' in list([symbol for symbol in zmat['symbols'][atom]]) for atom in a_atoms]) else ''
        a_str = f'A{x}_{a_atoms[0]}_{a_atoms[1]}_{a_atoms[2]}'
    if d_atoms is None:
        d_str = None
    else:
        x = 'X' if any(['X' in list([symbol for symbol in zmat['symbols'][atom]]) for atom in d_atoms]) else ''
        d_str = f'D{x}_{d_atoms[0]}_{d_atoms[1]}_{d_atoms[2]}_{d_atoms[3]}'

    for string in [r_str, a_str, d_str]:
        if string is not None and string in zmat['vars']:
            raise ZMatError(f'{string} is already in vars: {zmat["vars"]}')
    zmat['coords'].append((r_str, a_str, d_str))
    if any([atoms is not None and any([atoms.count(atom) > 1 for atom in atoms])
            for atoms in [r_atoms, a_atoms, d_atoms]]):
        raise ZMatError(f'zmat atom specifications must not have repetitions, got:\n'
                        f'r_atoms = {r_atoms}, a_atoms = {a_atoms}, d_atoms ={d_atoms}')
    if r_atoms is not None:
        zmat['vars'][r_str] = calculate_distance(coords=coords, atoms=[zmat['map'][atom] for atom in r_atoms])
    if added_dummy:
        zmat['vars'][a_str] = 90.0
        # The dihedral angle could be either 0 or 180 degrees, depends on the relative position of atom D and B, C
        # d_atoms represent the zmat indices of atoms D, C, X, and B.
        bcd_angle = calculate_angle(coords=coords, atoms=[zmat['map'][d_atoms[3]], zmat['map'][d_atoms[1]],
                                                          zmat['map'][d_atoms[0]]], index=0, units='degs')
        if 180 - TOL_180 < bcd_angle <= 180:
            zmat['vars'][d_str] = 180.0
        elif 0 <= bcd_angle < TOL_180:
            zmat['vars'][d_str] = 0.0
        else:
            raise ZMatError(f'Atoms {d_atoms} for a non-linear sequence with an angle of {bcd_angle}. '
                            f'Expected a linear sequence when using a dummy atom.')
    else:
        if a_atoms is not None:
            zmat['vars'][a_str] = calculate_angle(coords=coords, atoms=[zmat['map'][atom] for atom in a_atoms])
        if d_atoms is not None:
            zmat['vars'][d_str] = calculate_dihedral_angle(coords=coords, torsion=[zmat['map'][atom]
                                                                                   for atom in d_atoms])
    return zmat


def add_dummy_atom(zmat: dict,
                   xyz: dict,
                   coords: Union[list, tuple],
                   connectivity: dict,
                   r_atoms: list,
                   a_atoms: list,
                   n: int,
                   atom_index: int,
                   ) -> Tuple[dict, list, int, list, list, int]:
    """
    Add a dummy atom 'X' to the zmat.
    Also updates the r_atoms and a_atoms lists for the original (non-dummy) atom.

    Args:
        zmat (dict): The zmat.
        xyz (dict): The xyz dict.
        coords (Union[list, tuple]): Just the 'coords' part of the xyz dict.
        connectivity (dict): The atoms connectivity (keys are indices in the mol/xyz).
        r_atoms (list): The determined r_atoms.
        a_atoms (list): The determined a_atoms.
        n (int): The 0-index of the atom in the zmat to be added.
        atom_index (int): The 0-index of the atom in the molecule or cartesian coordinates to be added.
                          (``n`` and ``atom_index`` refer to the same atom, but it might have different indices
                          in the zmat and the molecule/xyz)

    Returns:
        Tuple[dict, list, int, list, list, int]:
            - The zmat.
            - The coordinates (list of tuples).
            - The updated atom index in the zmat.
            - The R atom indices.
            - The A atom indices.
            - A specific atom index to be used as the last entry of the D atom indices.
    """
    zmat['symbols'].append('X')
    zmat['map'][n] = f'X{len(xyz["symbols"])}'
    xyz['symbols'] += ('X',)
    xyz['isotopes'] += ('None',)

    # Determine the atoms for defining the dihedral angle, D, **for the dummy atom, X**.
    d_atoms = determine_d_atoms(zmat, xyz, coords, connectivity, a_atoms, n, atom_index, dummy=True)
    r_str = f'RX_{r_atoms[0]}_{r_atoms[1]}'
    a_str = f'AX_{a_atoms[0]}_{a_atoms[1]}_{a_atoms[2]}'
    d_str = f'DX_{d_atoms[0]}_{d_atoms[1]}_{d_atoms[2]}_{d_atoms[3]}' if d_atoms is not None else None
    zmat['coords'].append((r_str, a_str, d_str))  # the coords of the dummy atom
    zmat['vars'][r_str] = 1.0
    zmat['vars'][a_str] = 90.0
    if d_str is not None:
        zmat['vars'][d_str] = 180
    # Update xyz with the dummy atom (useful when this atom is used to define dihedrals of other atoms).
    coords = _add_nth_atom_to_coords(zmat=zmat, coords=list(coords), i=n)
    if connectivity is not None:
        # Update the connectivity dict to reflect that X is connected to the respective atom (r_atoms[1]),
        # this will help later in avoiding linear angles in the last three indices of a dihedral.
        connectivity[zmat['map'][r_atoms[1]]].append(int(zmat['map'][n][1:]))  # Take from 'X15'.
        connectivity[int(zmat['map'][n][1:])] = [zmat['map'][r_atoms[1]]]
    # Before adding the original (non-dummy) atom, increase n due to the increased number of atoms.
    n += 1
    # Store atom B's index for the dihedral of atom D.
    specific_last_d_atom = a_atoms[-1]
    # Update the r_atoms and a_atoms for the original (non-dummy) atom (d_atoms is set below).
    #
    #          X (dummy atom)
    #          |
    #     B -- C -- D (original atom)
    #   /
    #  A (optional)
    r_atoms = [n, r_atoms[1]]  # make this (D, C)
    a_atoms = r_atoms + [n - 1]  # make this (D, C, X)
    return zmat, coords, n, r_atoms, a_atoms, specific_last_d_atom


def zmat_to_coords(zmat: dict,
                   keep_dummy: bool = False,
                   skip_undefined: bool = False,
                   ) -> Tuple[List[dict], List[str]]:
    """
    Generate the cartesian coordinates from a zmat dict.
    Considers the zmat atomic map so the returned coordinates is ordered correctly.
    Most common isotopes assumed, if this is not the case, then isotopes should be reassigned to the xyz.
    This function assumes that all zmat variables relate to already defined atoms with a lower index in the zmat.

    This function implements the SN-NeRF algorithm as described in:
    J. Parsons, J.B. Holmes, J.M Rojas, J. Tsai, C.E.M. Strauss, "Practical Conversion from Torsion Space to Cartesian
    Space for In Silico Protein Synthesis", Journal of Computational Chemistry 2005, 26 (10), 1063-1068,
    https://doi.org/10.1002/jcc.20237

    Tested in converterTest.py rather than zmatTest

    Args:
        zmat (dict): The zmat.
        keep_dummy (bool): Whether to keep dummy atoms ('X'), ``True`` to keep, default is ``False``.
        skip_undefined (bool): Whether to skip atoms with undefined variables, instead of raising an error.
                               ``True`` to skip, default is ``False``.

    Raises:
        ZMatError: If zmat is of wrong type or does not contain all keys.

    Returns: Tuple[List[dict], List[str]]
        - The cartesian coordinates.
        - The atomic symbols corresponding to the coordinates.
    """
    if not isinstance(zmat, dict):
        raise ZMatError(f'zmat has to be a dictionary, got {type(zmat)}')
    if 'symbols' not in zmat or 'coords' not in zmat or 'vars' not in zmat or 'map' not in zmat:
        raise ZMatError(f'Expected to find symbols, coords, vars, and map in zmat, got instead: {list(zmat.keys())}.')
    if not len(zmat['symbols']) == len(zmat['coords']) == len(zmat['map']):
        raise ZMatError(f'zmat sections symbols, coords, and map have different lengths: {len(zmat["symbols"])}, '
                        f'{len(zmat["coords"])}, and {len(zmat["map"])}, respectively.')
    for key, value in zmat['vars'].items():
        if value is None:
            raise ZMatError(f'Got ``None`` for var {key} in zmat:\n{zmat}')
    var_list = list(zmat['vars'].keys())
    coords_to_skip = list()
    for i, coords in enumerate(zmat['coords']):
        for coord in coords:
            if coord is not None and coord not in var_list:
                if skip_undefined:
                    coords_to_skip.append(i)
                else:
                    raise ZMatError(f'The parameter {coord} was not found in the "vars" section of '
                                    f'the zmat:\n{zmat["vars"]}')

    coords = list()
    for i in range(len(zmat['symbols'])):
        coords = _add_nth_atom_to_coords(zmat=zmat, coords=coords, i=i, coords_to_skip=coords_to_skip)

    # Reorder the xyz according to the zmat map and remove dummy atoms if requested.
    ordered_coords, ordered_symbols = list(), list()
    for i in range(len(zmat['symbols'])):
        zmat_index = key_by_val(zmat['map'], i)
        if zmat_index < len(coords) and i not in coords_to_skip and (zmat['symbols'][zmat_index] != 'X' or keep_dummy):
            ordered_coords.append(coords[zmat_index])
            ordered_symbols.append(zmat['symbols'][zmat_index])

    return ordered_coords, ordered_symbols


def _add_nth_atom_to_coords(zmat: dict,
                            coords: list,
                            i: int,
                            coords_to_skip: Optional[list] = None,
                            ) -> list:
    """
    Add the n-th atom to the coords (n >= 0).

    Args:
        zmat (dict): The zmat.
        coords (list): The coordinates to be updated (not the entire xyz dict).
        i (int): The atom number in the zmat to be added to the coords (0-indexed)
        coords_to_skip (list, optional): Entries are indices to skip.

    Returns:
        list: The updated coords.
    """
    coords_to_skip = coords_to_skip or list()
    if i == 0:
        # Add the 1st atom.
        coords.append((0.0, 0.0, 0.0))  # atom A is placed at the origin
    elif i == 1:
        # Add the 2nd atom.
        r_key = zmat['coords'][i][0]
        coords.append((0.0, 0.0, zmat['vars'][r_key]))  # atom B is placed on axis Z, distant by the AB bond length
    elif i == 2:
        # Add the 3rd atom (atom "C").
        r_key, a_key = zmat['coords'][i][0], zmat['coords'][i][1]
        bc_length = zmat['vars'][r_key]
        alpha = zmat['vars'][a_key]
        alpha = math.radians(alpha if alpha < 180 else 360 - alpha)
        b_index = [indices for indices in get_atom_indices_from_zmat_parameter(r_key) if indices[0] == i][0][1]
        b_z = coords[b_index][2]
        c_y = bc_length * math.sin(alpha)
        # We differentiate between two cases for c_z:
        # Either atom A is at the origin (case 1), or atom B is at the origin (case 2).
        # One of them has to be at the origin (0, 0, 0), since we're adding the 3rd atom (so either A or B were 1st).
        #
        #  y
        #  ^                    C                         C
        #  |           (1)       \        or     (2)     /
        #  L__ > z           A -- B                    B -- A
        #
        # In case 1, we need to deduct len(B-C) from the z coordinate of atom B,
        # but in case 2 we need to take the positive value of len(B-C).
        # The above is also true if alpha(A-B-C) is > 90 degrees.
        c_z = b_z - bc_length * math.cos(alpha) if b_z else bc_length * math.cos(alpha)
        coords.append((0.0, c_y, c_z))
    elif i not in coords_to_skip:
        d_indices = [indices for indices in get_atom_indices_from_zmat_parameter(zmat['coords'][i][2])
                     if indices[0] == i][0]
        a_index, b_index, c_index = d_indices[3], d_indices[2], d_indices[1]
        # Atoms B and C aren't necessarily connected in the zmat, calculate from coords.
        bc_length = get_vector_length([coords[c_index][0] - coords[b_index][0],
                                       coords[c_index][1] - coords[b_index][1],
                                       coords[c_index][2] - coords[b_index][2]])
        cd_length = zmat['vars'][zmat['coords'][i][0]]
        bcd_angle = math.radians(zmat['vars'][zmat['coords'][i][1]])
        abcd_dihedral = math.radians(zmat['vars'][zmat['coords'][i][2]])
        # A vector pointing from atom A to atom B:
        ab = [(coords[b_index][0] - coords[a_index][0]),
              (coords[b_index][1] - coords[a_index][1]),
              (coords[b_index][2] - coords[a_index][2])]
        # A normalized vector pointing from atom B to atom C:
        ubc = [(coords[c_index][0] - coords[b_index][0]) / bc_length,
               (coords[c_index][1] - coords[b_index][1]) / bc_length,
               (coords[c_index][2] - coords[b_index][2]) / bc_length]
        n = np.cross(ab, ubc)
        un = n / get_vector_length(n)
        un_cross_ubc = np.cross(un, ubc)

        # The transformation matrix:
        m = np.array([[ubc[0], un_cross_ubc[0], un[0]],
                      [ubc[1], un_cross_ubc[1], un[1]],
                      [ubc[2], un_cross_ubc[2], un[2]]], np.float64)

        # Place atom D in a default coordinate system.
        d = np.array([- cd_length * math.cos(bcd_angle),
                      cd_length * math.sin(bcd_angle) * math.cos(abcd_dihedral),
                      cd_length * math.sin(bcd_angle) * math.sin(abcd_dihedral)])
        d = m.dot(d)  # Rotate the coordinate system into the reference frame of orientation defined by A, B, C.
        # Add the coordinates of atom C to the resulting atom D:
        coords.append((d[0] + coords[c_index][0], d[1] + coords[c_index][1], d[2] + coords[c_index][2]))
    return coords


def check_atom_r_constraints(atom_index: int,
                             constraints: dict,
                             ) -> Tuple[Optional[tuple], Optional[str]]:
    """
    Check distance constraints for an atom.
    'R' constraints are a list of tuples with length 2.
    The first atom in an R constraint is considered "constraint" for zmat generation:
    its distance will be relative to the second atom.

    Args:
        atom_index (int): The 0-indexed atom index to check.
        constraints (dict): The 'R', 'A', 'D' constraints dict. Values are lists of constraints.

    Raises:
        ZMatError: If the R constraint lengths do not equal two, or if the atom is constrained more than once.

    Returns:
        Tuple[Optional[tuple], Optional[str]]:
            - The atom index to which the atom being checked is constrained. ``None`` if it is not constrained.
            - The constraint type ('R_atom', or 'R_group').
    """
    if not any(['R' in key for key in constraints.keys()]):
        return None, None
    r_constraints = {key: val for key, val in constraints.items() if 'R' in key}
    for r_constraint_list in r_constraints.values():
        if any([len(r_constraint_tuple) != 2 for r_constraint_tuple in r_constraint_list]):
            raise ZMatError(f'"R" constraints must contain only tuples of length two, got: {r_constraints}.')
    if any([r_constraint_type not in ['R_atom', 'R_group'] for r_constraint_type in r_constraints.keys()]):
        raise ZMatError(f'"R" constraints must be either "R_atom" or "R_group", got: {r_constraints}.')
    occurrences = 0
    for r_constraint_list in r_constraints.values():
        occurrences += sum([r_constraint[0] == atom_index for r_constraint in r_constraint_list])
    if not occurrences:
        return None, None
    if occurrences > 1:
        raise ZMatError(f'A single atom cannot be constrained more than once. Atom {atom_index} is constrained '
                        f'{occurrences} times in "R" constraints.')
    # At this point there's only one occurrence of this constraint, find it and report the constraining atom.
    for constraint_type, r_constraint_list in r_constraints.items():
        for r_constraint in r_constraint_list:
            if r_constraint[0] == atom_index:
                return r_constraint, constraint_type


def check_atom_a_constraints(atom_index: int,
                             constraints: dict,
                             ) -> Tuple[Optional[tuple], Optional[str]]:
    """
    Check angle constraints for an atom.
    'A' constraints are a list of tuples with length 3.
    The first atom in an A constraint is considered "constraint" for zmat generation:
    its angle will be relative to the second and third atoms.

    Args:
        atom_index (int): The 0-indexed atom index to check.
        constraints (dict): The 'R', 'A', 'D' constraints dict. Values are lists of constraints.

    Raises:
        ZMatError: If the A constraint lengths do not equal three, or if the atom is constrained more than once.

    Returns:
        Tuple[Optional[tuple], Optional[str]]:
            - The atom indices to which the atom being checked is constrained. ``None`` if it is not constrained.
            - The constraint type ('A_atom', or 'A_group'). ``None`` if it is not constrained.
    """
    if not any(['A' in key for key in constraints.keys()]):
        return None, None
    a_constraints = {key: val for key, val in constraints.items() if 'A' in key}
    for a_constraint_list in a_constraints.values():
        if any([len(a_constraint_tuple) != 3 for a_constraint_tuple in a_constraint_list]):
            raise ZMatError(f'"A" constraints must contain only tuples of length three, got: {a_constraints}.')
    if any([a_constraint_type not in ['A_atom', 'A_group'] for a_constraint_type in a_constraints.keys()]):
        raise ZMatError(f'"A" constraints must be either "A_atom" or "A_group", got: {a_constraints}.')
    occurrences = 0
    for a_constraint_list in a_constraints.values():
        occurrences += sum([a_constraint[0] == atom_index for a_constraint in a_constraint_list])
    if not occurrences:
        return None, None
    if occurrences > 1:
        raise ZMatError(f'A single atom cannot be constrained more than once. Atom {atom_index} is constrained '
                        f'{occurrences} times in "A" constraints.')
    # At this point there's only one occurrence of this constraint, find it and report the constraining atoms.
    for constraint_type, a_constraint_list in a_constraints.items():
        for a_constraint in a_constraint_list:
            if a_constraint[0] == atom_index:
                return a_constraint, constraint_type


def check_atom_d_constraints(atom_index: int,
                             constraints: dict,
                             ) -> Tuple[Optional[tuple], Optional[str]]:
    """
    Check dihedral angle constraints for an atom.
    'D' constraints are a list of tuples with length 4.
    The first atom in a D constraint is considered "constraint" for zmat generation:
    its dihedral angle will be relative to the second, third, and forth atoms.

    Args:
        atom_index (int): The 0-indexed atom index to check.
        constraints (dict): The 'R', 'A', 'D' constraints dict. Values are lists of constraints.

    Raises:
        ZMatError: If the A constraint lengths do not equal three, or if the atom is constrained more than once.

    Returns:
        Tuple[Optional[tuple], Optional[str]]:
            - The atom indices to which the atom being checked is constrained. ``None`` if it is not constrained.
            - The constraint type ('D_atom', 'D_group').
    """
    if not any(['D' in key for key in constraints.keys()]):
        return None, None
    d_constraints = {key: val for key, val in constraints.items() if 'D' in key}
    for d_constraint_list in d_constraints.values():
        if any([len(d_constraint_tuple) != 4 for d_constraint_tuple in d_constraint_list]):
            raise ZMatError(f'"D" constraints must contain only tuples of length four, got: {d_constraints}.')
    if any([d_constraint_type not in ['D_atom', 'D_group'] for d_constraint_type in d_constraints.keys()]):
        raise ZMatError(f'"D" constraints must be either "D_atom", or "D_group", got: {d_constraints}.')
    occurrences = 0
    for d_constraint_list in d_constraints.values():
        occurrences += sum([d_constraint[0] == atom_index for d_constraint in d_constraint_list])
    if not occurrences:
        return None, None
    if occurrences > 1:
        raise ZMatError(f'A single atom cannot be constrained more than once. Atom {atom_index} is constrained '
                        f'{occurrences} times in "D" constraints.')
    # At this point there's only one occurrence of this constraint, find it and report the constraining atoms.
    for constraint_type, d_constraint_list in d_constraints.items():
        for d_constraint in d_constraint_list:
            if d_constraint[0] == atom_index:
                return d_constraint, constraint_type


def is_dummy(zmat: dict,
             zmat_index: int,
             ) -> bool:
    """
    Determine whether an atom in a zmat is a dummy atom by its zmat index.

    Args:
        zmat (dict): The zmat with symbol and map information.
        zmat_index (int): The atom index in the zmat.

    Raises:
        ZMatError: If the index is invalid.

    Returns:
        bool: Whether the atom represents a dummy atom 'X'. ``True`` if it does.
    """
    if len(zmat['symbols']) <= zmat_index:
        raise ZMatError(f'index {zmat_index} is invalid for a zmat with only {len(zmat["symbols"])} atoms')
    return zmat['symbols'][zmat_index] == 'X'


def is_angle_linear(angle: float,
                    tolerance: Optional[float] = None,
                    ) -> bool:
    """
    Check whether an angle is close to 180 or 0 degrees.

    Args:
        angle (float): The angle in degrees.
        tolerance (float): The tolerance to consider.

    Returns:
        bool: Whether the angle is close to 180 or 0 degrees, ``True`` if it is.
    """
    tol = tolerance or TOL_180
    if 180 - tol < angle <= 180 or 0 <= angle < tol:
        return True
    return False


def get_atom_connectivity_from_mol(mol: Molecule,
                                   atom1: 'Atom',
                                   ) -> List[int]:
    """
    Get the connectivity of ``atom`` in ``mol``.
    Returns heavy (non-H) atoms first.

    Args:
        mol (Molecule): The molecule with connectivity information.
        atom1 (Atom): The atom to check connectivity for.

    Returns:
        List[int]: 0-indices of atoms in ``mol`` connected to ``atom``.
    """
    return [mol.atoms.index(atom2) for atom2 in list(atom1.edges.keys()) if atom2.is_non_hydrogen()] \
        + [mol.atoms.index(atom2) for atom2 in list(atom1.edges.keys()) if atom2.is_hydrogen()]


def get_connectivity(mol: Molecule) -> Dict[int, List[int]]:
    """
    Get the connectivity information from the molecule object.

    Args:
        mol (Molecule): The Molecule object.

    Returns:
        Dict[int, List[int]]: The connectivity information.
              Keys are atom indices, values are tuples of respective edges, ordered with heavy atoms first.
              All indices are 0-indexed, corresponding to atom indices in ``mol`` (not in the zmat).
              ``None`` if ``xyz`` is given.
    """
    connectivity = dict()
    for atom in mol.atoms:
        connectivity[mol.atoms.index(atom)] = get_atom_connectivity_from_mol(mol, atom)
    return connectivity


def order_fragments_by_constraints(fragments: List[List[int]],
                                   constraints_dict: Optional[Dict[str, List[tuple]]] = None,
                                   ) -> List[List[int]]:
    """
    Get the order in which atoms should be added to the zmat from a 2D or a 3D representation.

    Args:
        fragments (List[List[int]]):
            Fragments represented by the species, i.e., as in a VdW well or a TS.
            Entries are atom index lists of all atoms in a fragment, each list represents a different fragment.
        constraints_dict (dict, optional):
            A dictionary of atom constraints. The function will try to find an atom order in which all constrained atoms
            are after the atoms they are constraint to.

    Returns:
        List[List[int]]: The ordered fragments list.
    """
    if constraints_dict is None or not len(fragments):
        return fragments
    constraints_in_fragments = list()
    for i in range(len(fragments)):
        # Initialize with general constraint types and the original index.
        constraints_in_fragments.append({'R': 0, 'A': 0, 'D': 0, 'i': i})
    for constraint_type, constraint_list in constraints_dict.items():
        for constraint in constraint_list:
            for i, fragment in enumerate(fragments):
                if all([c in fragment for c in constraint]):
                    constraints_in_fragments[i][constraint_type[0]] += 1
    constraints_in_fragments.sort(key=operator.itemgetter('R', 'A', 'D'), reverse=False)
    new_order = [constraint['i'] for constraint in constraints_in_fragments]
    new_fragments = [[]] * len(fragments)
    for fragment, i in zip(fragments, new_order):
        new_fragments[i] = fragment
    return new_fragments


def get_atom_order(xyz: Optional[Dict[str, tuple]] = None,
                   mol: Optional[Molecule] = None,
                   fragments: Optional[List[List[int]]] = None,
                   constraints_dict: Optional[Dict[str, List[tuple]]] = None,
                   ) -> List[int]:
    """
    Get the order in which atoms should be added to the zmat from a 2D or a 3D representation.

    Args:
        xyz (dict, optional): The 3D coordinates.
        mol (Molecule, optional): The Molecule object.
        fragments (List[List[int]], optional):
            Fragments represented by the species, i.e., as in a VdW well or a TS.
            Entries are atom index lists of all atoms in a fragment, each list represents a different fragment.
        constraints_dict (dict, optional):
            A dictionary of atom constraints. The function will try to find an atom order in which all constrained atoms
            are after the atoms they are constraint to.

    Returns:
        List[int]: The atom order, 0-indexed.
    """
    if mol is None and xyz is None:
        raise ValueError('Either mol or xyz must be given.')
    if fragments is None or not len(fragments):
        if mol is not None:
            fragments = [list(range(len(mol.atoms)))]
        if xyz is not None:
            fragments = [list(range(len(xyz['symbols'])))]
    else:
        fragments = order_fragments_by_constraints(fragments=fragments, constraints_dict=constraints_dict)

    atom_order = list()
    if mol is not None:
        for fragment in fragments:
            atom_order.extend(get_atom_order_from_mol(mol=mol, fragment=fragment, constraints_dict=constraints_dict))
    elif xyz is not None:
        for fragment in fragments:
            atom_order.extend(get_atom_order_from_xyz(xyz, fragment=fragment))
    return atom_order


def get_atom_order_from_mol(mol: Molecule,
                            fragment: List[int] = None,
                            constraints_dict: Optional[Dict[str, List[tuple]]] = None,
                            ) -> List[int]:
    """
    Get the order in which atoms should be added to the zmat from the 2D graph representation of the molecule.

    Args:
        mol (Molecule): The Molecule object.
        fragment (List[int], optional): Entries are 0-indexed atom indices to consider in the molecule.
                                        Only atoms within the fragment are considered.
        constraints_dict (dict, optional): A dictionary of atom constraints.
                                           The function will try to find an atom order in which all constrained atoms
                                           are after the atoms they are constraint to.

    Returns:
        List[int]: The atom order, 0-indexed.
    """
    fragment = fragment or list(range(len(mol.atoms)))
    atoms_to_explore, constraints, constraint_atoms, unsuccessful, top_d = list(), list(), list(), list(), list()
    constraints_dict = constraints_dict or dict()
    number_of_heavy_atoms = len([atom for atom in mol.atoms if atom.is_non_hydrogen()])

    for constraint_type, constraint_list in constraints_dict.items():
        constraints.extend(constraint_list)  # A list of all constraint tuples.
        for constraint in constraint_list:
            # A list of the atoms being constraint to other atoms.
            constraint_atoms.append(constraint[0])  # Only the first atom in the constraint tuple is really constrained.
        if constraint_type == 'D_group':
            for constraint_indices in constraint_list:
                if len(top_d):
                    raise ZMatError(f'zmats can only handle one D_group constraint at a time, got:\n{constraints_dict}')
                # Determine the "top" of the *relevant branch* of this torsion
                # since the first atom (number 0) is the constrained one being changed,
                # this top is defined from the second atom in the torsion (number 1) to it.
                top_d = determine_top_group_indices(mol=mol,
                                                    atom1=mol.atoms[constraint_indices[1]],
                                                    atom2=mol.atoms[constraint_indices[0]],
                                                    index=0)[0]

    for i in range(len(mol.atoms)):
        # Iterate through the atoms until a successful atom_order is reached.
        atom_order, start = list(), None
        # Try determining a starting point as a heavy atom connected to no more than one heavy atom neighbor.
        for atom1 in mol.atoms:
            # Find a tail, e.g.: CH3-C-...
            atom1_index = mol.atoms.index(atom1)
            if atom1.is_non_hydrogen() \
                    and sum([atom2.is_non_hydrogen() for atom2 in list(atom1.edges.keys())]) <= 1 \
                    and atom1_index not in constraint_atoms \
                    and atom1_index not in unsuccessful \
                    and atom1_index not in top_d \
                    and atom1_index in fragment:
                start = atom1_index
                break
        else:
            # If a tail could not be found (e.g., cyclohexane), just start from the first non-constrained heavy atom.
            for atom1 in mol.atoms:
                atom1_index = mol.atoms.index(atom1)
                if atom1.is_non_hydrogen() \
                        and mol.atoms.index(atom1) not in constraint_atoms \
                        and atom1_index not in unsuccessful \
                        and atom1_index not in top_d \
                        and atom1_index in fragment:
                    start = atom1_index
                    break
            else:
                # Try hydrogens (an atom might be constraint to H, in which case H should come before that atom).
                for atom1 in mol.atoms:
                    atom1_index = mol.atoms.index(atom1)
                    if atom1.is_hydrogen() \
                            and atom1_index not in constraint_atoms \
                            and atom1_index not in unsuccessful \
                            and atom1_index not in top_d \
                            and atom1_index in fragment:
                        start = atom1_index
                        break
        if start is None:
            if i == len(mol.atoms) - 1:
                raise ZMatError('Could not determine a starting atom for the zmat')
            continue
        atoms_to_explore = [start]

        if number_of_heavy_atoms == 2:
            # To have meaningful dihedrals for H's, add one H from each heavy atom before adding the heavy atoms.
            heavy_atoms = [atom for atom in mol.atoms if atom.is_non_hydrogen()]
            hydrogens_0 = [atom for atom in heavy_atoms[0].edges.keys() if atom.is_hydrogen()]
            hydrogens_1 = [atom for atom in heavy_atoms[1].edges.keys() if atom.is_hydrogen()]
            if len(hydrogens_0) and len(hydrogens_1):
                if not constraint_atoms:
                    hydrogen_0, hydrogen_1 = hydrogens_0[0], hydrogens_1[0]
                    for atom_index in [mol.atoms.index(heavy_atoms[0]), mol.atoms.index(heavy_atoms[1]),
                                       mol.atoms.index(hydrogen_0), mol.atoms.index(hydrogen_1)]:
                        if atom_index in fragment:
                            atom_order.append(atom_index)
                else:
                    for constraint in constraints:
                        for atom_index in constraint[::-1]:
                            if atom_index in fragment and atom_index not in atom_order:
                                atom_order.append(atom_index)
                    for atom1 in mol.atoms:
                        atom1_index = mol.atoms.index(atom1)
                        if atom1_index not in atom_order and atom1_index in fragment:
                            atom_order.append(atom1_index)
                atoms_to_explore = list()

        unexplored = list()  # Atoms purposely not added to atom_order due to a D_group constraint.
        while len(atoms_to_explore):
            # Add all heavy atoms, consider branching and rings.
            atom1_index = atoms_to_explore[0]
            atoms_to_explore.pop(0)
            atom1 = mol.atoms[atom1_index]
            if atom1_index not in top_d and atom1_index in fragment and atom1_index not in atom_order:
                atom_order.append(atom1_index)
            else:
                unexplored.append(atom1_index)
            for atom2 in list(atom1.edges.keys()):
                index2 = mol.atoms.index(atom2)
                if index2 not in atom_order and index2 not in atoms_to_explore and index2 not in unexplored \
                        and atom2.is_non_hydrogen():
                    atoms_to_explore.append(index2)

        for atom1 in mol.atoms:
            # Add all hydrogen atoms.
            if atom1.is_hydrogen():
                index = mol.atoms.index(atom1)
                if index not in atom_order and index not in top_d and index in fragment:
                    atom_order.append(index)

        # Now add top_d.
        for top_d_atom in top_d:
            if top_d_atom not in atom_order and top_d_atom in fragment:
                atom_order.append(top_d_atom)

        if len(atom_order) != len(fragment):
            continue

        if not len(constraints):
            break

        success = True

        for constraint in constraints:
            # Loop through all constraint tuples, verify that the atoms are ordered in accordance with them.
            if any([c in fragment for c in constraint]):
                # Consider this constraint.
                constraint_atom_order = [atom_order.index(constraint_atom) for constraint_atom in constraint
                                         if constraint_atom in fragment]
                diff = [constraint_atom_order[j+1] - constraint_atom_order[j]
                        for j in range(len(constraint_atom_order) - 1)]
                if any(entry > 0 for entry in diff):
                    # Diff should only have negative entries.
                    unsuccessful.append(start)
                    success = False

        if success:
            # The atom order list answers all constraints criteria.
            break

    else:
        # The outer for loop exhausted all possibilities and was unsuccessful.
        raise ZMatError(f'Could not derive an atom order from connectivity that answers all '
                        f'constraint criteria:\n{constraints_dict}')

    if len(set(atom_order)) < len(atom_order):
        raise ZMatError(f'Could not determine a unique atom order!\n({atom_order})')

    return atom_order


def get_atom_order_from_xyz(xyz: Dict[str, tuple],
                            fragment: Optional[List[int]] = None,
                            ) -> List[int]:
    """
    Get the order in which atoms should be added to the zmat from the 3D geometry.

    Args:
        xyz (dict): The 3D coordinates.
        fragment (List[int], optional): Entries are 0-indexed atom indices to consider in the molecule.
                              Only atoms within the fragment are considered.

    Returns:
        List[int]: The atom order, 0-indexed.
    """
    fragment = fragment or list(range(len(xyz['symbols'])))
    atom_order, hydrogens = list(), list()
    for i, symbol in enumerate(xyz['symbols']):
        if i in fragment:
            if symbol == 'H':
                hydrogens.append(i)
            else:
                atom_order.append(i)
    atom_order.extend(hydrogens)
    return atom_order


def consolidate_zmat(zmat: dict,
                     mol: Optional[Molecule] = None,
                     consolidation_tols: Optional[dict] = None,
                     ) -> dict:
    """
    Consolidate (almost) identical vars in the zmat.

    Args:
        zmat (dict): The zmat.
        mol (Molecule, optional): The RMG molecule, used for atom types if not None,
                                  otherwise atom symbols from the zmat are used.
        consolidation_tols (dict, optional): Keys are 'R', 'A', 'D', values are floats representing absolute tolerance
                                             for consolidating almost equal internal coordinates.

    Returns:
        dict: The consolidated zmat.
    """
    consolidation_tols = consolidation_tols or dict()

    # Assign defaults if needed:
    if 'R' not in consolidation_tols:
        consolidation_tols['R'] = DEFAULT_CONSOLIDATION_R_TOL
    if 'A' not in consolidation_tols:
        consolidation_tols['A'] = DEFAULT_CONSOLIDATION_A_TOL
    if 'D' not in consolidation_tols:
        consolidation_tols['D'] = DEFAULT_CONSOLIDATION_D_TOL

    zmat['coords'] = list(zmat['coords'])  # Make sure it is mutable.

    keys_to_consolidate1, keys_to_consolidate2 = {'R': list(), 'A': list(), 'D': list()}, \
                                                 {'R': list(), 'A': list(), 'D': list()}

    # Identify potential keys to consolidate, save in keys_to_consolidate1.
    for i, key1 in enumerate(zmat['vars'].keys()):
        if key1 is not None and not any([key1 in keys for keys in keys_to_consolidate1[key1[0]]]):
            dup_keys = list()
            for j, key2 in enumerate(zmat['vars'].keys()):
                if j > i and key2 is not None and key1[0] == key2[0] \
                        and abs(zmat['vars'][key1] - zmat['vars'][key2]) < consolidation_tols[key1[0]]:
                    for key in [key1, key2]:
                        if key not in dup_keys:
                            dup_keys.append(key)
            if len(dup_keys):
                appended = False
                for dup_key in dup_keys:
                    for j in range(len(keys_to_consolidate1[key1[0]])):
                        if dup_key in keys_to_consolidate1[key1[0]][j]:
                            # There are two lists that must be combined during consolidation,
                            # but weren't identified as such (probably ony some variables are within the tolerance).
                            keys_to_consolidate1[key1[0]][j] = \
                                sorted(list(set(keys_to_consolidate1[key1[0]][j] + dup_keys)))
                            appended = True
                if not appended:
                    keys_to_consolidate1[key1[0]].append(dup_keys)

    # Check atom types before consolidating, save in keys_to_consolidate2.
    for key_type in ['R', 'A', 'D']:
        for keys in keys_to_consolidate1[key_type]:
            atoms_dict, indices_to_pop = dict(), list()
            for key in keys:
                indices = [int(index) for index in key.split('_')[1:]]
                if any([zmat['symbols'][index] == 'X' for index in indices]):
                    # This is a dummy atom, don't check atoms, always consolidate.
                    atoms_dict[key] = key[:2]
                else:
                    if mol is not None and all([zmat['map'][index] != 'X' for index in indices]):
                        # Use atom types.
                        atoms_dict[key] = tuple(mol.atoms[zmat['map'][index]].atomtype.label for index in indices)
                    else:
                        # Use symbols.
                        atoms_dict[key] = tuple(zmat['symbols'][index] for index in indices)
            atoms_sets = list(set(list(atoms_dict.values())))
            for i, atoms_set in enumerate(atoms_sets):
                # Check if the reverse is there as well, e.g., ('C', 'H') is the same as ('H', 'C')),
                # but if popping the first, don't pop the second...
                # also, avid popping symmetric tuples such as ('O', 'O'), or ('H', 'C', 'H').
                if tuple(reversed(atoms_set)) in atoms_sets \
                        and tuple(reversed(atoms_set)) not in [atom_set for j, atom_set in enumerate(atoms_sets)
                                                               if j in indices_to_pop] \
                        and atoms_set != tuple(reversed(atoms_set)):
                    indices_to_pop.append(i)
            for i in reversed(range(len(atoms_sets))):
                if i in indices_to_pop:
                    atoms_sets.pop(i)
            for atoms_tuple in atoms_sets:
                keys_to_consolidate2[key_type].append([key for key, val in atoms_dict.items()
                                                       if val == atoms_tuple or tuple(reversed(val)) == atoms_tuple])

    # Consolidate the zmat.
    for key_type in ['R', 'A', 'D']:
        for keys in keys_to_consolidate2[key_type]:
            # Generate a new key name.
            indices, new_indices = list(), list()
            for i in range(len(keys[0].split('_')[1:])):
                indices.append([key.split('_')[1:][i] for key in keys])
            for i in range(len(keys[0].split('_')[1:])):
                new_indices.append('|'.join(str(index) for index in indices[i]))
            if any(['X' in list([zmat['symbols'][int(index)] for index in entry]) for entry in indices]):
                new_key = '_'.join([key_type + 'X'] + new_indices)
            else:
                new_key = '_'.join([key_type] + new_indices)

            # Replace all occurrences in zmat['coords'].
            for i in range(len(zmat['coords'])):
                if any([coord in keys for coord in zmat['coords'][i]]):
                    new_coord = list()
                    for coord in zmat['coords'][i]:
                        if coord not in keys:
                            new_coord.append(coord)
                        else:
                            new_coord.append(new_key)
                    zmat['coords'][i] = tuple(new_coord)

            # Replace all occurrences in zmat['vars'] with the new key and an average value.
            if all([key in list(zmat['vars'].keys()) for key in keys]):
                values = [zmat['vars'][key] for key in keys]
                new_value = sum(values) / len(values)
                for key in keys:
                    del zmat['vars'][key]
            else:
                # Adding a new value to an existing key.
                found_key = False
                keys_as_indices = [get_atom_indices_from_zmat_parameter(key)[0] for key in keys]
                for variable in zmat['vars'].keys():
                    if variable[0] == key_type:
                        var_indices = get_atom_indices_from_zmat_parameter(variable)
                        if any([key_indices in var_indices for key_indices in keys_as_indices]):
                            found_key = True
                            for key_indices in keys_as_indices:
                                if key_indices not in var_indices:
                                    var_indices += (key_indices,)
                            new_consolidated_key = key_type
                            for i in range(len(var_indices[0])):
                                new_consolidated_key += '_' + \
                                                        '|'.join([str(var_index[i])
                                                                  for var_index in var_indices])
                            zmat['vars'][new_consolidated_key] = zmat['vars'][variable]
                            break
                if found_key:
                    del zmat['vars'][variable]
                    for key in keys:
                        if key in list(zmat['vars'].keys()):
                            del zmat['vars'][key]
                else:
                    raise ZMatError('Could not consolidate zmat')
            zmat['vars'][new_key] = new_value
    zmat['coords'] = tuple(zmat['coords'])
    return zmat


def get_atom_indices_from_zmat_parameter(param: str) -> tuple:
    """
    Get the atom indices from a zmat parameter.

    Examples:
        'R_0_2' --> ((0, 2),)
        'A_0_1_2' --> ((0, 1, 2),) corresponding to angle 0-1-2
        'D_0_1_2_4' --> ((0, 1, 2, 4),)
        'R_0|0_3|4' --> ((0, 3), (0, 4))
        'A_0|0|0_1|1|1_2|3|4' --> ((0, 1, 2), (0, 1, 3), (0, 1, 4)) corresponding to angles 0-1-2, 0-1-3, and 0-1-4
        'D_0|0|0_1|1|1_2|3|4_5|6|9' --> ((0, 1, 2, 5), (0, 1, 3, 6), (0, 1, 4, 9))
        'RX_0_2' --> ((0, 2),)

    Args:
        param (str): The zmat parameter.

    Returns:
        tuple: Entries are tuples of indices, each describing R, A, or D parameters.
               The tuple entries for R, A, and D types are of lengths 2, 3, and 4, respectively.
               The number of tuple entries depends on the number of consolidated parameters.
    """
    result, index_groups = list(), list()
    splits = param.split('_')[1:]  # exclude the type char ('R', 'A', or 'D')
    for split in splits:
        index_groups.append(split.split('|'))
    for i in range(len(index_groups[0])):
        result.append(tuple(int(index_group[i]) for index_group in index_groups))
    return tuple(result)


def get_parameter_from_atom_indices(zmat: dict,
                                    indices: Union[list, tuple],
                                    xyz_indexed: bool = True,
                                    ) -> Union[str, tuple, list]:
    """
    Get the zmat parameter from the atom indices.
    If indices are of length two, three, or four, an R, A, or D parameter is returned, respectively.

    If a requested parameter represents an angle split by a dummy atom,
    combine the two dummy angles to get the original angle.
    In this case, a list of the two corresponding parameters will be returned.

    Examples:
        [0, 2] --> 'R_0_2', or 'R_0|1_2_5', etc.
        [0, 2, 4] --> 'A_0_2_4', or a respective consolidated key
        [0, 2, 4, 9] --> 'D_0_2_4_9', or a respective consolidated key

    Args:
        zmat (dict): The zmat.
        indices (Union[list, tuple]): Entries are 0-indices of atoms, list is of length 2, 3, or 4.
        xyz_indexed (bool, optional): Whether the atom indices relate to the xyz (and the zmat map will be used)
                                      or they already relate to the zmat. Default is ``True`` (relate to xyz).

    Raises:
        TypeError: If ``indices`` are of wrong type.
        ZMatError: If ``indices`` has a wrong length, or not all indices are in the zmat map.

    Returns: Union[str, tuple, list]
        The corresponding zmat parameter.
    """
    if not isinstance(indices, (list, tuple)):
        raise TypeError(f'indices must be a list, got {indices} which is a {type(indices)}')
    if len(indices) not in [2, 3, 4]:
        raise ZMatError(f'indices must be of length 2, 3, or 4, got {indices} (length {len(indices)}.')
    if xyz_indexed:
        if any([index not in list(zmat['map'].values()) for index in indices]):
            raise ZMatError(f'Not all indices ({indices}) are in the zmat map values ({list(zmat["map"].values())}).')
        indices = [key_by_val(zmat['map'], index) for index in indices]
    if any([index not in list(zmat['map'].keys()) for index in indices]):
        raise ZMatError(f'Not all indices ({indices}) are in the zmat map keys ({list(zmat["map"].keys())}).')
    key = '_'.join([KEY_FROM_LEN[len(indices)]] + [str(index) for index in indices])
    if key in list(zmat['vars'].keys()):
        # It's a non-consolidated key.
        return key
    # It's a consolidated key.
    key = KEY_FROM_LEN[len(indices)]
    for var in zmat['vars'].keys():
        if var[0] == key and tuple(indices) in list(get_atom_indices_from_zmat_parameter(var)):
            return var
    # If no value found, check whether this is an angle split by a dummy atom.
    var1, var2 = None, None
    if len(indices) == 3:
        # 180 degree angles aren't given explicitly in the zmat,
        # they are separated in to two angles using a dummy atom, check if this is the case here.
        dummy_indices = [str(key) for key, val in zmat['map'].items() if isinstance(val, str) and 'X' in val]
        param1 = 'AX_{0}_{1}_{2}'
        all_parameters = list(zmat['vars'].keys())
        for dummy_str_index in dummy_indices:
            var1 = param1.format(indices[0], indices[1], dummy_str_index) \
                if param1.format(indices[0], indices[1], dummy_str_index) in all_parameters and var1 is None else var1
            var1 = param1.format(dummy_str_index, indices[1], indices[0]) \
                if param1.format(dummy_str_index, indices[1], indices[0]) in all_parameters and var1 is None else var1
            if var1 is not None:
                param2a = f'AX_{dummy_str_index}_{indices[1]}_{indices[2]}'
                param2b = f'AX_{indices[2]}_{indices[1]}_{dummy_str_index}'
                var2 = param2a if param2a in all_parameters and var2 is None else var2
                var2 = param2b if param2b in all_parameters and var2 is None else var2
                if var2 is not None:
                    break
                var1 = None
    if var1 is not None and var2 is not None:
        return [var1, var2]
    raise ZMatError(f'Could not find a key corresponding to {key} {indices}.')


def _compare_zmats(zmat1: dict,
                   zmat2: dict,
                   r_tol: Optional[float] = None,
                   a_tol: Optional[float] = None,
                   d_tol: Optional[float] = None,
                   symmetric_torsions: Optional[dict] = None,
                   verbose: bool = False,
                   ) -> bool:
    """
    Compare two zmats. The zmats must have identical variables (i.e., derived from the same connectivity or ordered xyz,
    using the same constraints).
    This function does not make use of the zmat map, but does check that it is identical.

    Args:
        zmat1 (dict): zmat1.
        zmat2 (dict): zmat2.
        r_tol (float, optional): A tolerance for comparing distances.
        a_tol (float, optional): A tolerance for comparing angles.
        d_tol (float, optional): A tolerance for comparing dihedral angles.
        symmetric_torsions (dict, optional): Keys are tuples of 0-indexed scan indices, values are internal rotation
                                             symmetry numbers (sigma). Conformers which only differ by an integer number
                                             times 360 degrees / sigma are considered identical.
        verbose (bool, optional): Whether to print a reason for determining the zmats are different if they are,
                                  ``True`` to print.

    Raises:
        ZMatError: If the zmats are of wrong type or don't have all attributes.

    Returns:
        bool: Whether the two zmats represent the same conformation to the desired tolerance. ``True`` if they do.
    """
    if not isinstance(zmat1, dict) or not isinstance(zmat2, dict):
        raise ZMatError(f'zmats must be dictionaries, got {type(zmat1)} and {type(zmat2)}')
    if not len(list(zmat1.keys())):
        raise ZMatError(f'zmat1 is empty! Got {zmat1}')
    if not len(list(zmat2.keys())):
        raise ZMatError(f'zmat2 is empty! Got {zmat2}')
    if 'symbols' not in zmat1 or 'coords' not in zmat1 or 'vars' not in zmat1 \
            or 'symbols' not in zmat2 or 'coords' not in zmat2 or 'vars' not in zmat2:
        raise ZMatError(f'zmats must contain the "symbols", "coords", and "vars" keys, got: '
                        f'{list(zmat1.keys())} and {list(zmat2.keys())}.')
    tol = {'R': r_tol or DEFAULT_COMPARISON_R_TOL,
           'A': a_tol or DEFAULT_COMPARISON_A_TOL,
           'D': d_tol or DEFAULT_COMPARISON_D_TOL}
    if zmat1['map'] != zmat2['map']:
        if verbose:
            logger.info(f'zmats have different maps:\n{zmat1["map"]}\n{zmat2["map"]}')
        return False
    if len(zmat1['symbols']) != len(zmat2['symbols']):
        if verbose:
            logger.info(f'zmats have different symbols:\n{zmat1["symbols"]}\n{zmat2["symbols"]}')
        return False
    for coord in zmat1['coords']:
        if coord not in zmat2['coords']:
            if verbose:
                logger.info(f'zmats differ since the coordinates {coord} are missing from zmat2.')
            return False
        if coord is None:
            continue
        for var in coord:
            if var is not None:
                val1, val2 = zmat1['vars'][var], zmat2['vars'][var]
                key_type = var[0]
                if abs(val1 - val2) > tol[key_type]:
                    # The dihedrals are not the same, but check whether they disagree by an integer times 360 / sigma.
                    sigma = None
                    if symmetric_torsions is not None and 'D' in var:
                        vars_ = get_atom_indices_from_zmat_parameter(var)
                        # vars is generated from var as 'D_0|0_1|1_2|3|_5|6' --> ((0, 1, 2, 5), (0, 1, 3, 6)).
                        for var_ in vars_:
                            for symmetric_torsion in symmetric_torsions.keys():
                                # Check the pivots only (not the entire four torsion indices).
                                if all([var_[i + 1] in symmetric_torsion[1:3] for i in range(2)]):
                                    sigma = symmetric_torsions[symmetric_torsion]
                        if sigma is not None:
                            diff = abs(val1 - val2)
                            rotation_symmetry = 360 / sigma  # The rotation symmetry in degrees.
                            if not any([diff - rotation_symmetry * i <= tol[key_type] for i in range(sigma)]):
                                sigma = None  # Mark this test as unsuccessful to return False below.
                    if sigma is None:
                        if verbose:
                            logger.info(f'zmats differ since the respective values for var {var} in zmat1 and zmat2 '
                                        f'({val1:.4f}, {val2:.4f}) differ by {abs(val1 - val2):.4f}, which is greater '
                                        f'than the required tolerance of {tol[key_type]}')
                        return False
    return True


def get_all_neighbors(mol: Molecule,
                      atom_index: int,
                      ) -> List[int]:
    """
    Get atom indices of all neighbors of an atom in a molecule.

    Args:
        mol (Molecule): The RMG molecule with connectivity information.
        atom_index (int): The index of the atom whose neighbors are requested.

    Returns:
        List[int]: Atom indices of all neighbors of the requested atom.
    """
    neighbors = list()
    for atom in mol.atoms[atom_index].edges.keys():
        neighbors.append(mol.atoms.index(atom))
    return neighbors


def is_atom_in_new_fragment(atom_index: int,
                            zmat: Dict[str, Union[dict, tuple]],
                            fragments: Optional[List[List[int]]] = None,
                            skip_atoms: Optional[List[int]] = None,
                            ) -> bool:
    """
    Whether an atom is present in a new fragment that hasn't been added to the zmat yet,
    and therefore atom assignment should not be done based on connectivity.

    Args:
        atom_index (int): The 0-index of the atom in the molecule or cartesian coordinates to be added.
                          (``n`` and ``atom_index`` refer to the same atom, but it might have different indices
                          in the zmat/molecule/xyz/fragments)
        zmat (dict): The zmat.
        skip_atoms (list): Atoms in the zmat map to ignore when checking fragments.
        fragments (List[List[int]], optional):
            Fragments represented by the species, i.e., as in a VdW well or a TS.
            Entries are atom index lists of all atoms in a fragment, each list represents a different fragment.
            indices are 0-indexed.

    Returns:
        bool: Whether to consider connectivity for assigning atoms in z-mat variables.
    """
    skip_atoms = skip_atoms or list()
    if fragments is not None and len(fragments) > 1:
        for fragment in fragments:
            if atom_index in fragment:
                if all([z_index in skip_atoms or frag_index not in fragment
                        for z_index, frag_index in zmat['map'].items()]):
                    # All atoms considered thus far are not in the current fragment, connectivity is meaningless.
                    return True
                break
    return False


def up_param(param: str,
             increment: Optional[int] = None,
             increment_list: Optional[List[int]] = None,
             ) -> str:
    """
    Increase the indices represented by a zmat parameter.

    Args:
        param (str): The zmat parameter.
        increment (int, optional): The increment to increase by.
        increment_list (list, optional): Entries are individual indices to use when incrementing the ``param`` indices.

    Raises:
        ZMatError: If neither ``increment`` nor ``increment_list`` were specified,
                   or if the increase resulted in a negative number.

    Returns: str
        The new parameter with increased indices.
    """
    if increment is None and increment_list is None:
        raise ZMatError('Either increment or increment_list must be specified.')
    indices = get_atom_indices_from_zmat_parameter(param)[0]
    if increment is not None:
        new_indices = [0 if not index and increment < 0 else index + increment for index in indices]
    else:
        if len(increment_list) != len(indices):
            raise ZMatError(f'The number of increments in {increment_list} ({len(increment_list)} is different than '
                            f'the number of indices to increment {indices} ({len(indices)})')
        new_indices = [index + inc for index, inc in zip(indices, increment_list)]
    if any(index < 0 for index in new_indices):
        raise ZMatError(f'Got a negative zmat index when bumping {param} by {increment}')
    new_indices = [str(index) for index in new_indices]
    new_param = '_'.join([param.split('_')[0]] + new_indices)
    return new_param


def remove_1st_atom(zmat: dict) -> dict:
    """
    Remove the first atom of a zmat.
    Note: The first atom in 'symbols' with map key 0 is removed,
    it is not necessarily the first atom in the corresponding xyz with map value 0.

    Args:
        zmat (dict): The zmat to process.

    Returns:
        dict: The updated zmat.
    """
    new_symbols = tuple(zmat['symbols'][1:])
    new_coords, removed_vars = list(), list()
    for i, coords in enumerate(zmat['coords']):
        if i == 0:
            continue
        removed_vars.extend([coord for j, coord in enumerate(coords) if coord is not None and j > i - 2])
        new_coords.append((up_param(coords[0], increment=-1) if i >= 2 else None,
                           up_param(coords[1], increment=-1) if i >= 3 else None,
                           up_param(coords[2], increment=-1) if i >= 4 else None))
    new_coords = tuple(new_coords)
    new_vars = {up_param(key, increment=-1): val for key, val in zmat['vars'].items() if key not in removed_vars}
    val_0 = zmat['map'][0]
    new_map = {key - 1: val - 1 if val > val_0 else val for key, val in zmat['map'].items() if key != 0}
    return {'symbols': new_symbols, 'coords': new_coords, 'vars': new_vars, 'map': new_map}
