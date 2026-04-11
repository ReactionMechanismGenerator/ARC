cimport arc.molecule.molecule as mm
cimport arc.molecule.element as elements


cpdef to_rdkit_mol(mm.Molecule mol, bint remove_h=*, bint return_mapping=*, bint sanitize=*, bint save_order=?)

cpdef mm.Molecule from_rdkit_mol(mm.Molecule mol, object rdkitmol, bint raise_atomtype_exception=?)
