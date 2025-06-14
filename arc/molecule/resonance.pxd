
from arc.molecule.graph cimport Vertex, Edge, Graph
from arc.molecule.molecule cimport Atom, Bond, Molecule

cpdef list populate_resonance_algorithms(dict features=?)

cpdef dict analyze_molecule(Graph mol, bint save_order=?)

cpdef list generate_resonance_structures_safely(Graph mol, bint clar_structures=?, bint keep_isomorphic=?, bint filter_structures=?, bint save_order=?)

cpdef list generate_resonance_structures(Graph mol, bint clar_structures=?, bint keep_isomorphic=?, bint filter_structures=?, bint save_order=?)

cpdef list _generate_resonance_structures(list mol_list, list method_list, bint keep_isomorphic=?, bint copy=?, bint save_order=?)

cpdef list generate_allyl_delocalization_resonance_structures(Graph mol)

cpdef list generate_lone_pair_multiple_bond_resonance_structures(Graph mol)

cpdef list generate_adj_lone_pair_radical_resonance_structures(Graph mol)

cpdef list generate_adj_lone_pair_multiple_bond_resonance_structures(Graph mol)

cpdef list generate_adj_lone_pair_radical_multiple_bond_resonance_structures(Graph mol)

cpdef list generate_N5dc_radical_resonance_structures(Graph mol)

cpdef list generate_isomorphic_resonance_structures(Graph mol, bint saturate_h=?)

cpdef list generate_optimal_aromatic_resonance_structures(Graph mol, dict features=?, bint save_order=?)

cpdef list generate_aromatic_resonance_structure(Graph mol, list aromatic_bonds=?, bint copy=?, bint save_order=?)

cpdef list generate_aryne_resonance_structures(Graph mol)

cpdef list generate_kekule_structure(Graph mol)

cpdef list generate_clar_structures(Graph mol, bint save_order=?)

cpdef list _clar_transformation(Graph mol, list aromatic_ring)

cpdef list generate_adsorbate_shift_down_resonance_structures(Graph mol)

cpdef list generate_adsorbate_shift_up_resonance_structures(Graph mol)

cpdef list generate_adsorbate_conjugate_resonance_structures(Graph mol)