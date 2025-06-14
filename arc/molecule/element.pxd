cdef class Element:

    cdef public int number
    cdef public str name
    cdef public str symbol
    cdef public float mass
    cdef public float cov_radius
    cdef public int isotope
    cdef public str chemkin_name

cpdef Element get_element(value, int isotope=?)
