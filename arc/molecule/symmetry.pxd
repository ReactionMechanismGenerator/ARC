
from arc.molecule.molecule cimport Atom, Bond, Molecule


cpdef float calculate_atom_symmetry_number(Molecule molecule, Atom atom) except -1

cpdef float calculate_bond_symmetry_number(Molecule molecule, Atom atom1, Atom atom2) except -1

cpdef float calculate_axis_symmetry_number(Molecule molecule) except -1

cpdef float calculate_cyclic_symmetry_number(Molecule molecule) except -1

cpdef bint _indistinguishable(Atom atom1, Atom atom2) except -2

cpdef float calculate_symmetry_number(Molecule molecule) except -1
