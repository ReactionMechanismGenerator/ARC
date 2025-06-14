
from arc.molecule.graph cimport Vertex, Edge, Graph

cdef class VF2:

    cdef Graph graph1, graph2

    cdef Graph graphA, graphB
    
    cdef dict initial_mapping
    cdef bint subgraph
    cdef bint find_all
    cdef bint strict
    
    cdef bint is_match
    cdef list mapping_list
    
    cpdef bint is_isomorphic(self, Graph graph1, Graph graph2, dict initial_mapping, bint save_order=?, bint strict=?) except -2
        
    cpdef list find_isomorphism(self, Graph graph1, Graph graph2, dict initial_mapping, bint save_order=?, bint strict=?)

    cpdef bint is_subgraph_isomorphic(self, Graph graph1, Graph graph2, dict initial_mapping, bint save_order=?) except -2

    cpdef list find_subgraph_isomorphisms(self, Graph graph1, Graph graph2, dict initial_mapping, bint save_order=?)
    
    cdef isomorphism(self, Graph graph1, Graph graph2, dict initial_mapping, bint subgraph, bint find_all, bint save_order=?, bint strict=?)

    cdef bint match(self, int call_depth) except -2
        
    cpdef bint feasible(self, Vertex vertex1, Vertex vertex2) except -2
    
    cdef add_to_mapping(self, Vertex vertex1, Vertex vertex2)
        
    cdef remove_from_mapping(self, Vertex vertex1, Vertex vertex2)
