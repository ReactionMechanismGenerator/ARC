
cimport arc.molecule.molecule as mm


cdef list BACKENDS
cdef dict INCHI_LOOKUPS
cdef dict SMILES_LOOKUPS

cdef dict MOLECULE_LOOKUPS
cdef dict RADICAL_LOOKUPS

cpdef str to_inchi(mm.Molecule mol, str backend=?, int aug_level=?)

cpdef str to_inchi_key(mm.Molecule mol, str backend=?, int aug_level=?)

cpdef str to_smarts(mm.Molecule mol, backend=?)

cpdef str to_smiles(mm.Molecule mol, backend=?)

cpdef mm.Molecule from_inchi(mm.Molecule mol, str inchistr, backend=?, bint raise_atomtype_exception=?)

cpdef mm.Molecule from_smiles(mm.Molecule mol, str smilesstr, str backend=?, bint raise_atomtype_exception=?)

cpdef mm.Molecule from_smarts(mm.Molecule mol, str smartsstr, str backend=?, bint raise_atomtype_exception=?)

cpdef mm.Molecule from_augmented_inchi(mm.Molecule mol, aug_inchi, bint raise_atomtype_exception=?)

cpdef object _rdkit_translator(object input_object, str identifier_type, mm.Molecule mol=?)

cpdef object _openbabel_translator(object input_object, str identifier_type, mm.Molecule mol=?, bint raise_atomtype_exception=?)

cdef mm.Molecule _lookup(mm.Molecule mol, str identifier, str identifier_type)

cpdef _check_output(mm.Molecule mol, str identifier)

cdef mm.Molecule _read(mm.Molecule mol, str identifier, str identifier_type, str backend, bint raise_atomtype_exception=?)

cdef str _write(mm.Molecule mol, str identifier_type, str backend)

cdef _get_backend_list(str backend)
