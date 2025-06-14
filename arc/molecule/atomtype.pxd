
cdef class AtomType:

    cdef public str label
    cdef public list generic
    cdef public list specific

    cdef public list increment_bond
    cdef public list decrement_bond
    cdef public list form_bond
    cdef public list break_bond
    cdef public list increment_radical
    cdef public list decrement_radical
    cdef public list increment_lone_pair
    cdef public list decrement_lone_pair
    cdef public list increment_charge
    cdef public list decrement_charge

    cdef public list single
    cdef public list all_double
    cdef public list r_double
    cdef public list o_double
    cdef public list s_double
    cdef public list triple
    cdef public list quadruple
    cdef public list benzene
    cdef public list lone_pairs
    cdef public list charge

    cpdef bint is_specific_case_of(self, AtomType other)

    cpdef bint equivalent(self, AtomType other)

    cpdef list get_features(self)

cpdef list get_features(atom, dict bonds)

cpdef AtomType get_atomtype(atom, dict bonds)
