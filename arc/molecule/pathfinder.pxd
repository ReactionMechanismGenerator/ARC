
from .graph cimport Vertex, Edge, Graph

cpdef list find_butadiene(Vertex start, Vertex end)

cpdef list find_butadiene_end_with_charge(Vertex start)

cpdef list find_allyl_end_with_charge(Vertex start)

cpdef list find_shortest_path(Vertex start, Vertex end, list path=*)

cpdef list add_unsaturated_bonds(list path)

cpdef list add_allyls(list path)

cpdef list add_inverse_allyls(list path)

cpdef dict compute_atom_distance(list atom_indices, Graph mol)

cpdef list find_allyl_delocalization_paths(Vertex atom1)

cpdef list find_lone_pair_multiple_bond_paths(Vertex atom1)

cpdef list find_adj_lone_pair_radical_delocalization_paths(Vertex atom1)

cpdef list find_adj_lone_pair_multiple_bond_delocalization_paths(Vertex atom1)

cpdef list find_adj_lone_pair_radical_multiple_bond_delocalization_paths(Vertex atom1)

cpdef list find_N5dc_radical_delocalization_paths(Vertex atom1)

cpdef bint is_atom_able_to_gain_lone_pair(Vertex atom)

cpdef bint is_atom_able_to_lose_lone_pair(Vertex atom)

cpdef list find_adsorbate_delocalization_paths(Vertex atom1)

cpdef list find_adsorbate_conjugate_delocalization_paths(Vertex atom1)