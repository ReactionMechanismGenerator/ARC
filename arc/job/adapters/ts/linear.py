"""
An adapter for generating transition state (TS) guess geometries by linear interpolation
of internal coordinate (Z-matrix) values between R and P.

Overview
--------
This adapter is **incore-only** and **geometry-generation-only**: it produces TS guess
Cartesian coordinates without submitting any external quantum chemistry calculation.
It supports:

- **Isomerization** (A <=> B): Z-matrix interpolation between reactant and product
  geometries via ``interpolate_isomerization``.
- **Addition / dissociation** (A <=> B + C, or B + C <=> A): fragment-based
  stretching of the unimolecular species via ``interpolate_addition``.  Applies
  to any reaction with exactly one unimolecular side.  The algorithm starts from
  that side, identifies fragment boundaries (either from the RMG reaction template
  or by combinatorial bond cutting), and rigidly translates fragments apart to
  approximate the TS geometry.

Pipeline
--------
For each reaction path (product_dict) the adapter runs the following steps:

1.  **Atom mapping** — ``map_rxn`` finds the atom-to-atom correspondence between the
    reactant and product for the specific reaction path (e.g., which of several
    equivalent H's is migrating).  A global fallback map is used if the per-path call
    fails.

2.  **Near-attack conformation** — ``get_near_attack_xyz`` pre-rotates the reactant (and
    product) geometry so that the reactive atoms are physically close to each other before
    any Z-matrix is built.  Without this step, the equilibrium conformations of reactant
    and product are often far from the TS ring geometry, and interpolation produces a poor
    guess.  See *Near-attack geometry* below for details.

3.  **Z-matrix chimeras** — Two independent TS guesses are built per path:

    * **Type R** (reactant topology): A Z-matrix is constructed from the near-attack
      reactant geometry.  The near-attack product geometry (in reactant atom ordering) is
      projected onto the same topology via ``update_zmat_by_xyz``.  The two internal-
      coordinate sets are blended at ``weight`` (default 0.5) by ``average_zmat_params``.
      Only coordinates that reference (in their variable name) at least one atom
      participating in a forming or breaking bond are interpolated; all other
      coordinates are taken unchanged from the reactant Z-matrix.

    * **Type P** (product topology): Symmetric to Type R — built from the near-attack
      product geometry and blended at ``1 − weight``.

    Anchor atoms for both Z-matrices are chosen by ``find_smart_anchors``, which prefers
    spectator atoms adjacent to the reactive core so the coordinate frame is stable across
    interpolation.

4.  **Weight grid** — ``get_weight_grid`` returns the set of interpolation weights to try
    (default ``(0.35, 0.50, 0.65)``).  If reactant and/or product energies (E0 or
    e_elect) are available, the grid is biased toward the Hammond/Marcus-predicted TS
    position (early TS for exothermic reactions, late TS for endothermic), with ±0.10
    spread around the prediction.

5.  **Geometry post-processing and validation** — Each candidate XYZ that emerges from
    ``zmat_to_xyz`` is corrected and validated before being added to the output list:

    * **H-transfer distance correction** (``fix_forming_bond_distances``): for each
      forming bond involving a hydrogen, the migrating H is placed by triangulation —
      at the intersection of two spheres centred on the donor and acceptor with
      Pauling-estimated TS radii (d0 + 0.42 Å), preserving the approach direction
      from the interpolation.

    * **Donor terminal-group staggering** (``stagger_donor_terminal_h``): non-
      migrating H atoms on the donor heavy atom are rotated around the donor–backbone
      bond axis to ±120° from the migrating H, preventing near-collisions caused by
      the H reconstruction.

    * **Non-reactive H distance correction** (``fix_nonreactive_h_distances``): any
      non-migrating hydrogen displaced beyond 105 % of its equilibrium single-bond
      length from its parent heavy atom (per the reactant topology) is projected back
      to the equilibrium distance.

    * **Migrating-group umbrella flip** (``fix_migrating_group_umbrella``): for
      group migrations (e.g. 1,2-shift of CH₃), the non-reactive H atoms on the
      migrating heavy atom may point toward the backbone after interpolation.
      This step detects the wrong orientation and reflects each H through the
      equatorial plane perpendicular to the umbrella axis, so 90° + A° becomes
      90° − A° and the H's point outward.

    * **Atom-collision filter** (``colliding_atoms``): atoms closer than ~60 % of
      their single-bond length are considered overlapping → geometry discarded.

    * **H close-contact filter** (``_has_h_close_contact``): any pair involving at
      least one hydrogen closer than 85 % of their single-bond length is discarded.
      This catches H–H and H–heavy contacts that slip past the 60 % collision filter.

    * **Detached-hydrogen filter** (``_has_detached_hydrogen``): any hydrogen more
      than 3.0 Å from every heavy atom → geometry discarded.

    * **Misoriented migrating-H filter** (``_has_misoriented_migrating_h``): rejects
      guesses where the migrating H is closer to an H on the acceptor atom than to
      the acceptor itself (bad rotor orientation).

    * **TS-motif distance filter** (``_has_bad_ts_motif``): rejects guesses where
      donor–H, acceptor–H, or donor–acceptor distances fall outside physically
      reasonable TS windows.

6.  **Deduplication** — Near-identical surviving guesses (compared pairwise by
    ``almost_equal_coords``) are collapsed to a single entry, preventing symmetry-
    equivalent paths from flooding the output.

Near-attack geometry (``get_near_attack_xyz``)
----------------------------------------------
The key challenge for intramolecular H-migrations and similar cyclic-TS reactions is that
the donor and acceptor atoms may be several Ångströms apart in the equilibrium geometry,
and simple Z-matrix interpolation cannot bridge that gap.  The near-attack step adjusts
the backbone dihedral angles so the molecule folds into the ring-like TS geometry before
the Z-matrix is built.

**Ring size** — the number of atoms in the cyclic TS (including the migrating atom and
the acceptor) equals the length of the shortest BFS path from the migrating atom to the
acceptor: N = len(path).  Example: for CCCOO intra-H migration the ring is
[H, C, C, C, O, O] → N = 6.

**Per-position dihedral targets** — ``_TS_RING_DIHEDRALS`` maps each ring size N to a
list of (N − 3) target dihedral magnitudes, one for each rotatable interior bond (ordered
from the donor end inward).  The values obey the empirical sum rule:

    sum of interior backbone dihedrals ≈ (N − 4) × 95°

Confirmed values for N = 6 from CCCOO TS data: HCCC = 50°, CCCO = 60°, CCOO = 80°
(sum = 190° = 2 × 95°).  For ring sizes outside the table a uniform fallback of 60° per
bond is used.

    +---------+---------------------------+--------------------+
    | Ring N  | Interior dihedrals (°)    | Sum (≈ (N−4)×95°) |
    +=========+===========================+====================+
    |    4    | [0.0]                     |    0°              |
    |    5    | [45.0, 50.0]              |   95°              |
    |    6    | [50.0, 60.0, 80.0]        |  190°              |
    |    7    | [55.0, 65.0, 75.0, 90.0]  |  285°              |
    |    8    | [55.0, 65.0, 75.0, 85.0,  |  380°              |
    |         |  100.0]                   |                    |
    +---------+---------------------------+--------------------+

**Sign selection** — for each interior bond the function trials both +T and −T rotations
(where T is the target magnitude) and applies the one that brings the donor atom and
acceptor atom closer together.  This is done by a fast Rodrigues in-place rotation on a
copy of the coordinate array; the actual geometry is updated only once the better
direction is known.  Non-rotatable bonds (double, triple, aromatic, in-ring) are skipped.

**Terminal-group orientation** (H-migration only) — after backbone rotation, the migrating
hydrogen has not yet been moved (it is upstream of all interior bonds).  A second pass
scans ±60° (in 5° steps) around the bond connecting the donor carbon to the next backbone
atom and selects the angle that minimises the H–acceptor distance.  Only the H atoms and
other substituents on the donor carbon are moved; the backbone remains unchanged.  This
step is restricted to reactions where the migrating atom is hydrogen, to avoid disturbing
heavy-atom rearrangements (e.g., NO₂ ↔ ONO conversions) where the same rotation would
push oxygen atoms into collision.

Implementation detail
---------------------
Heavy-lifting utilities live in the ``linear_utils`` subpackage:

- ``math_zmat``     — Z-matrix interpolation, weight/grid helpers, math utilities
- ``postprocess``   — validation filters, geometry fixers, family dispatch
- ``isomerization`` — near-attack conformations, ring-closure, Z-matrix branch generation
- ``addition``      — fragment-based stretching, insertion-ring, H migration

This module re-exports every public symbol from those subpackages so that
existing imports (``from arc.job.adapters.ts.linear import …``) continue
to work without modification.
"""

import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from arc.common import almost_equal_coords, get_logger, get_single_bond_length
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter, ts_adapters_by_rmg_family
from arc.job.factory import register_job_adapter
from arc.mapping.driver import map_rxn
from arc.plotter import save_geo
from arc.species.converter import order_mol_by_atom_map, order_xyz_by_atom_map
from arc.species.species import ARCSpecies, TSGuess, colliding_atoms
from arc.species.zmat import find_smart_anchors

if TYPE_CHECKING:
    from arc.level import Level
    from arc.molecule import Molecule
    from arc.reaction import ARCReaction

from arc.job.adapters.ts.linear_utils.math_zmat import get_r_constraints, get_weight_grid
from arc.job.adapters.ts.linear_utils.postprocess import (
    adjust_reactive_bond_distances,
    has_excessive_backbone_drift,
    orient_h_on_reactive_centers,
    postprocess_ts_guess,
    validate_ts_guess,
)
from arc.job.adapters.ts.linear_utils.isomerization import (
    RING_CLOSURE_THRESHOLD,
    backbone_atom_map,
    build_4center_interchange_ts,
    generate_zmat_branch,
    get_near_attack_xyz,
    get_path_length,
    path_has_cumulated_bonds,
    ring_closure_xyz,
)
from arc.job.adapters.ts.linear_utils.addition import (
    apply_intra_frag_contraction,
    build_concerted_ts,
    find_split_bonds_by_fragmentation,
    map_and_verify_fragments,
    migrate_h_between_fragments,
    migrate_verified_atoms,
    stretch_bond,
    stretch_core_from_large,
)
from arc.job.adapters.ts.linear_utils.families import build_xy_elimination_ts


logger = get_logger()


@dataclass
class _PathContext:
    """Bundle of per-path state for TS-building strategies."""
    r_xyz: dict
    r_mol: 'Molecule'
    op_xyz: dict
    mapped_p_mol: 'Molecule'
    bb: List[Tuple[int, int]]
    fb: List[Tuple[int, int]]
    family: Optional[str]
    r_label_map: Optional[dict]
    weight: float
    ring_sets: List[Set[int]]
    label: str
    discovered_in_reverse: bool = False
    reactive_xyz_indices: Set[int] = field(default_factory=set)
    anchors: Optional[list] = None
    constraints: Optional[dict] = None
    r_xyz_na: Optional[dict] = None
    op_xyz_na: Optional[dict] = None


@dataclass
class _StrategyResult:
    """Result of a TS-building strategy."""
    guesses: List[dict] = field(default_factory=list)
    halt: bool = False


class LinearAdapter(JobAdapter):
    """
    A class for executing TS guess jobs based on linear interpolation of internal coordinate values.

    This adapter is incore-only and geometry-generation-only: it produces TS guess Cartesian
    coordinates without submitting any external quantum chemistry calculation.

    Args:
        project (str): The project's name. Used for setting the remote path.
        project_directory (str): The path to the local project directory.
        job_type (list, str): The job's type, validated against ``JobTypeEnum``. If it's a list, pipe.py will be called.
        args (dict, optional): Methods (including troubleshooting) to be used in input files.
                               Keys are either 'keyword', 'block', or 'trsh', values are dictionaries with values
                               to be used either as keywords or as blocks in the respective software input file.
                               If 'trsh' is specified, an action might be taken instead of appending a keyword or a
                               block to the input file (e.g., change server or change scan resolution).
        bath_gas (str, optional): A bath gas. Currently only used in OneDMin to calculate L-J parameters.
        checkfile (str, optional): The path to a previous Gaussian checkfile to be used in the current job.
        conformer (int, optional): Conformer number if optimizing conformers.
        constraints (list, optional): A list of constraints to use during an optimization or scan.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
        dihedral_increment (float, optional): The degrees increment to use when scanning dihedrals of TS guesses.
        dihedrals (List[float], optional): The dihedral angels corresponding to self.torsions.
        directed_scan_type (str, optional): The type of the directed scan.
        ess_settings (dict, optional): A dictionary of available ESS and a corresponding server list.
        ess_trsh_methods (List[str], optional): A list of troubleshooting methods already tried out.
        execution_type (str, optional): The execution type, 'incore', 'queue', or 'pipe'.
        fine (bool, optional): Whether to use fine geometry optimization parameters. Default: ``False``.
        initial_time (datetime.datetime or str, optional): The time at which this job was initiated.
        irc_direction (str, optional): The direction of the IRC job (`forward` or `reverse`).
        job_id (int, optional): The job's ID determined by the server.
        job_memory_gb (int, optional): The total job allocated memory in GB (14 by default).
        job_name (str, optional): The job's name (e.g., 'opt_a103').
        job_num (int, optional): Used as the entry number in the database, as well as in ``job_name``.
        job_server_name (str, optional): Job's name on the server (e.g., 'a103').
        job_status (list, optional): The job's server and ESS statuses.
        level (Level, optional): The level of theory to use.
        max_job_time (float, optional): The maximal allowed job time on the server in hours (can be fractional).
        reactions (List[ARCReaction], optional): Entries are ARCReaction instances, used for TS search methods.
        rotor_index (int, optional): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        server (str): The server to run on.
        server_nodes (list, optional): The nodes this job was previously submitted to.
        species (List[ARCSpecies], optional): Entries are ARCSpecies instances.
                                              Either ``reactions`` or ``species`` must be given.
        testing (bool, optional): Whether the object is generated for testing purposes, ``True`` if it is.
        times_rerun (int, optional): Number of times this job was re-run with the same arguments (no trsh methods).
        torsions (List[List[int]], optional): The 0-indexed atom indices of the torsion(s).
        tsg (int, optional): TSGuess number if optimizing TS guesses.
        xyz (dict, optional): The 3D coordinates to use. If not give, species.get_xyz() will be used.
    """

    def __init__(self,
                 project: str,
                 project_directory: str,
                 job_type: Union[List[str], str],
                 args: Optional[dict] = None,
                 bath_gas: Optional[str] = None,
                 checkfile: Optional[str] = None,
                 conformer: Optional[int] = None,
                 constraints: Optional[List[Tuple[List[int], float]]] = None,
                 cpu_cores: Optional[str] = None,
                 dihedral_increment: Optional[float] = None,
                 dihedrals: Optional[List[float]] = None,
                 directed_scan_type: Optional[str] = None,
                 ess_settings: Optional[dict] = None,
                 ess_trsh_methods: Optional[List[str]] = None,
                 execution_type: Optional[str] = None,
                 fine: bool = False,
                 initial_time: Optional[Union['datetime.datetime', str]] = None,
                 irc_direction: Optional[str] = None,
                 job_id: Optional[int] = None,
                 job_memory_gb: float = 14.0,
                 job_name: Optional[str] = None,
                 job_num: Optional[int] = None,
                 job_server_name: Optional[str] = None,
                 job_status: Optional[List[Union[dict, str]]] = None,
                 level: Optional['Level'] = None,
                 max_job_time: Optional[float] = None,
                 reactions: Optional[List['ARCReaction']] = None,
                 rotor_index: Optional[int] = None,
                 server: Optional[str] = None,
                 server_nodes: Optional[list] = None,
                 species: Optional[List[ARCSpecies]] = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: Optional[List[List[int]]] = None,
                 tsg: Optional[int] = None,
                 xyz: Optional[dict] = None,
                 ):

        self.incore_capacity = 50
        self.job_adapter = 'linear'
        self.command = None
        self.execution_type = execution_type or 'incore'

        if reactions is None:
            raise ValueError('Cannot execute TS Linear without ARCReaction object(s).')

        _initialize_adapter(obj=self,
                            is_ts=True,
                            project=project,
                            project_directory=project_directory,
                            job_type=job_type,
                            args=args,
                            bath_gas=bath_gas,
                            checkfile=checkfile,
                            conformer=conformer,
                            constraints=constraints,
                            cpu_cores=cpu_cores,
                            dihedral_increment=dihedral_increment,
                            dihedrals=dihedrals,
                            directed_scan_type=directed_scan_type,
                            ess_settings=ess_settings,
                            ess_trsh_methods=ess_trsh_methods,
                            fine=fine,
                            initial_time=initial_time,
                            irc_direction=irc_direction,
                            job_id=job_id,
                            job_memory_gb=job_memory_gb,
                            job_name=job_name,
                            job_num=job_num,
                            job_server_name=job_server_name,
                            job_status=job_status,
                            level=level,
                            max_job_time=max_job_time,
                            reactions=reactions,
                            rotor_index=rotor_index,
                            server=server,
                            server_nodes=server_nodes,
                            species=species,
                            testing=testing,
                            times_rerun=times_rerun,
                            torsions=torsions,
                            tsg=tsg,
                            xyz=xyz,
                            )

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.
        """
        pass

    def set_files(self) -> None:
        """
        Set files to be uploaded and downloaded. Writes the files if needed.
        Modifies the self.files_to_upload and self.files_to_download attributes.

        self.files_to_download is a list of remote paths.

        self.files_to_upload is a list of dictionaries, each with the following keys:
        ``'name'``, ``'source'``, ``'make_x'``, ``'local'``, and ``'remote'``.
        If ``'source'`` = ``'path'``, then the value in ``'local'`` is treated as a file path.
        Else if ``'source'`` = ``'input_files'``, then the value in ``'local'`` will be taken
        from the respective entry in inputs.py
        If ``'make_x'`` is ``True``, the file will be made executable.
        """
        pass

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        pass

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        pass

    def execute_incore(self):
        """
        Execute a job incore.
        """
        self._log_job_execution()
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()

        supported_families = [key for key, val in ts_adapters_by_rmg_family.items() if 'linear' in val]

        self.reactions = [self.reactions] if not isinstance(self.reactions, list) else self.reactions
        for rxn in self.reactions:
            if rxn.family not in supported_families or not rxn.is_unimolecular():
                logger.warning(f'The linear TS search adapter does not support the {rxn.family} reaction family.')
                continue
            rxn.ts_species = rxn.ts_species or ARCSpecies(label='TS',
                                                          is_ts=True,
                                                          charge=rxn.charge,
                                                          multiplicity=rxn.multiplicity,
                                                          )
            weights = get_weight_grid(rxn)
            all_xyzs_so_far: List[dict] = []
            for w_i, w in enumerate(weights):
                t0 = datetime.datetime.now()
                xyzs = interpolate(rxn=rxn, weight=w, existing_xyzs=all_xyzs_so_far)
                t_ex = datetime.datetime.now() - t0
                if not xyzs:
                    continue
                all_xyzs_so_far.extend(xyzs)

                for xyz_i, xyz in enumerate(xyzs):
                    if colliding_atoms(xyz):
                        continue
                    unique = True
                    for other_tsg in rxn.ts_species.ts_guesses:
                        if almost_equal_coords(xyz, other_tsg.initial_xyz):
                            if 'linear' not in other_tsg.method.lower():
                                other_tsg.method += f' and Linear (w={w:.2f}, {xyz_i})'
                            unique = False
                            break

                    if unique:
                        method = f'Linear (w={w:.2f}, {xyz_i})'
                        ts_guess = TSGuess(method=method,
                                           index=len(rxn.ts_species.ts_guesses),
                                           method_index=w_i,
                                           t0=t0,
                                           execution_time=t_ex,
                                           success=True,
                                           family=rxn.family,
                                           xyz=xyz,
                                           )
                        rxn.ts_species.ts_guesses.append(ts_guess)

                        save_geo(xyz=xyz,
                                 path=self.local_path,
                                 filename=f'Linear w={w:.2f}, {xyz_i}',
                                 format_='xyz',
                                 comment=f'Linear w={w:.2f}, {xyz_i}, family: {rxn.family}',
                                 )

            if len(self.reactions) < 5:
                successes = len([tsg for tsg in rxn.ts_species.ts_guesses if tsg.success and 'linear' in tsg.method])
                if successes:
                    logger.info(f'Linear successfully found {successes} TS guesses for {rxn.label}.')
                else:
                    logger.info(f'Linear did not find any successful TS guesses for {rxn.label}.')
        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        """
        (Execute a job to the server's queue.)
        A single Heuristics job will always be executed incore.
        """
        self.execute_incore()


# ---------------------------------------------------------------------------
# Top-level dispatchers
# ---------------------------------------------------------------------------

def _frag_composition_key(uni_mol: 'Molecule',
                          atom_to_idx: dict,
                          uni_symbols: tuple,
                          cut: List[Tuple[int, int]],
                          ) -> Tuple[Tuple[Tuple[str, int], ...], ...]:
    """
    Compute a hashable composition key for the fragments produced by *cut*.

    The key is a sorted tuple of per-fragment element formulas, each formula
    being a sorted tuple of (element_symbol, count) pairs.  Two cuts that
    produce fragments with identical element compositions will have the same
    key, regardless of which specific atoms are in each fragment.
    """
    _local_adj: Dict[int, Set[int]] = {k: set() for k in range(len(uni_mol.atoms))}
    for atom in uni_mol.atoms:
        _ia = atom_to_idx[atom]
        for nbr in atom.edges:
            _ib = atom_to_idx[nbr]
            _local_adj[_ia].add(_ib)
    for _a, _b in cut:
        _local_adj[_a].discard(_b)
        _local_adj[_b].discard(_a)
    _visited: Set[int] = set()
    _frags: List[Set[int]] = []
    for _start in range(len(uni_mol.atoms)):
        if _start in _visited:
            continue
        _comp: Set[int] = set()
        _q: deque = deque([_start])
        while _q:
            _n = _q.popleft()
            if _n in _visited:
                continue
            _visited.add(_n)
            _comp.add(_n)
            _q.extend(_local_adj[_n] - _visited)
        _frags.append(_comp)
    frag_formulas = []
    for frag in _frags:
        formula: Dict[str, int] = {}
        for idx in frag:
            sym = uni_symbols[idx]
            formula[sym] = formula.get(sym, 0) + 1
        frag_formulas.append(tuple(sorted(formula.items())))
    return tuple(sorted(frag_formulas))


def _build_ring_scission_ts(product_xyz: dict,
                            breaking_bonds: List[Tuple[int, int]],
                            weight: float = 0.5,
                            stretch_factor: float = 1.8,
                            ) -> Optional[dict]:
    """Build a TS guess for ring-scission reactions from the product geometry.

    For reactions where the product has a ring that the reactant doesn't
    (ring-scission in reverse = ring-formation in forward), the TS is
    product-like with the breaking bond stretched.

    The breaking bond is elongated by ``stretch_factor`` (default 1.8×),
    adjusted by ``weight`` (0.5 = halfway between product and fully stretched).
    Only the first atom of each breaking bond is displaced; the rest of the
    molecule stays fixed.

    Args:
        product_xyz: Atom-mapped product XYZ dictionary.
        breaking_bonds: List of (i, j) pairs for bonds that break.
        weight: Interpolation weight (0 = product-like, 1 = reactant-like).
        stretch_factor: Maximum stretch multiplier for the breaking bond.

    Returns:
        XYZ dictionary or ``None`` if no breaking bonds.
    """
    if not breaking_bonds:
        return None
    coords = np.array(product_xyz['coords'], dtype=float)
    for i, j in breaking_bonds:
        vec = coords[i] - coords[j]
        d_orig = float(np.linalg.norm(vec))
        if d_orig < 1e-6:
            continue
        direction = vec / d_orig
        d_target = d_orig * (1.0 + (stretch_factor - 1.0) * weight)
        displacement = direction * (d_target - d_orig)
        # Identify the fragment connected to i that doesn't include j
        # (BFS from i, not crossing j).  Move the entire fragment.
        from collections import deque
        fragment: Set[int] = set()
        queue: deque = deque([i])
        n_atoms = len(product_xyz['symbols'])
        # Build adjacency from distances (bonded if < 1.8 Å for heavy, 1.3 for H).
        adj: Dict[int, Set[int]] = {k: set() for k in range(n_atoms)}
        for a in range(n_atoms):
            for b in range(a + 1, n_atoms):
                d_ab = float(np.linalg.norm(coords[a] - coords[b]))
                thresh = 1.3 if 'H' in (product_xyz['symbols'][a], product_xyz['symbols'][b]) else 1.8
                if d_ab < thresh:
                    adj[a].add(b)
                    adj[b].add(a)
        while queue:
            node = queue.popleft()
            if node in fragment or node == j:
                continue
            fragment.add(node)
            queue.extend(adj[node] - fragment - {j})
        for k in fragment:
            coords[k] += displacement
    return {'symbols': product_xyz['symbols'],
            'isotopes': product_xyz['isotopes'],
            'coords': tuple(tuple(row) for row in coords)}


def _reposition_migrating_atom(xyz: dict,
                               coords: np.ndarray,
                               mig_idx: int,
                               don_idx: int,
                               acc_idx: int,
                               ) -> dict:
    """Reposition a non-H migrating atom symmetrically between donor and acceptor.

    Places the atom at the donor-acceptor midpoint, then offsets it along the
    original migration direction so that d(mig, donor) ≈ d(mig, acceptor) ≈
    the average of the original donor distance and a stretched TS estimate
    (1.5× the original bond length).

    Args:
        xyz: XYZ dictionary (used for symbols/isotopes).
        coords: Coordinate array (modified in-place).
        mig_idx: Index of the migrating atom.
        don_idx: Index of the donor heavy atom.
        acc_idx: Index of the acceptor heavy atom.

    Returns:
        Updated XYZ dictionary.
    """
    midpoint = (coords[don_idx] + coords[acc_idx]) / 2.0
    half_da = float(np.linalg.norm(coords[don_idx] - coords[acc_idx])) / 2.0
    # Original bond length to donor; target TS distance ≈ 1.5× (stretched).
    d_orig = float(np.linalg.norm(coords[mig_idx] - coords[don_idx]))
    d_target = max(d_orig * 1.5, half_da + 0.3)
    # Direction perpendicular to the donor-acceptor axis, in the plane
    # defined by donor, acceptor, and the original migrating atom position.
    da_vec = coords[acc_idx] - coords[don_idx]
    da_unit = da_vec / max(np.linalg.norm(da_vec), 1e-8)
    mig_to_mid = coords[mig_idx] - midpoint
    # Remove the component along the donor-acceptor axis.
    perp = mig_to_mid - np.dot(mig_to_mid, da_unit) * da_unit
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm > 1e-6:
        offset_dir = perp / perp_norm
    else:
        # Degenerate: pick an arbitrary perpendicular direction.
        cross = np.cross(da_vec, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(cross) < 1e-6:
            cross = np.cross(da_vec, np.array([0.0, 1.0, 0.0]))
        offset_dir = cross / np.linalg.norm(cross)
    # Height above midpoint so that d(mig, donor) = d_target.
    height_sq = d_target ** 2 - half_da ** 2
    height = np.sqrt(max(height_sq, 0.1))
    new_pos = midpoint + offset_dir * height
    displacement = new_pos - coords[mig_idx]
    # Move the entire fragment bonded to mig_idx (e.g. CH3 group),
    # excluding donor and acceptor, so H's travel with the migrating atom.
    from collections import deque as _deque
    n = len(xyz['symbols'])
    adj: Dict[int, Set[int]] = {k: set() for k in range(n)}
    for a in range(n):
        for b in range(a + 1, n):
            d_ab = float(np.linalg.norm(coords[a] - coords[b]))
            thresh = 1.3 if 'H' in (xyz['symbols'][a], xyz['symbols'][b]) else 1.8
            if d_ab < thresh:
                adj[a].add(b)
                adj[b].add(a)
    fragment: Set[int] = set()
    queue = _deque([mig_idx])
    while queue:
        node = queue.popleft()
        if node in fragment or node in (don_idx, acc_idx):
            continue
        fragment.add(node)
        queue.extend(adj[node] - fragment - {don_idx, acc_idx})
    for k in fragment:
        coords[k] += displacement
    return {'symbols': xyz['symbols'],
            'isotopes': xyz['isotopes'],
            'coords': tuple(tuple(row) for row in coords)}


def _get_ring_preservation_bonds(mol: 'Molecule',
                                 reactive_core: Set[int],
                                 changing_bonds: Set[Tuple[int, int]],
                                 ring_sets: List[Set[int]],
                                 ) -> Tuple[List[Tuple[int, int]], Set[int]]:
    """
    Identify spectator ring bonds adjacent to reactive atoms for Z-matrix preservation.

    When a reactive atom sits in a ring, the Z-matrix may place the ring atoms through
    a chain that does not pass through the reactive atom.  If the reactive atom moves
    during interpolation, its ring-bonded spectator neighbors stay anchored elsewhere
    in the Z-matrix tree, and the ring breaks.

    Only ring bonds that are NOT themselves breaking or forming are considered.  Only
    rings with exactly one reactive atom are eligible (rings with multiple reactive atoms
    are part of the reactive event itself).

    Args:
        mol: RMG Molecule object (reactant topology).
        reactive_core: Atom indices participating in forming/breaking bonds.
        changing_bonds: ``{(min_idx, max_idx), ...}`` for all breaking/forming bonds.
        ring_sets: Pre-computed list of ring atom index sets (from SSSR on a mol copy).

    Returns:
        ``(ring_bond_pairs, expanded_reactive_set)``
    """
    ring_bond_pairs: List[Tuple[int, int]] = []
    expanded = set(reactive_core)
    for ring_set in ring_sets:
        reactive_in_ring = ring_set & reactive_core
        if len(reactive_in_ring) != 1:
            continue
        reactive_idx = next(iter(reactive_in_ring))
        atom = mol.atoms[reactive_idx]
        for bonded_atom in atom.edges:
            bonded_idx = mol.atoms.index(bonded_atom)
            if bonded_idx in ring_set and bonded_idx not in reactive_core:
                bond_pair = (min(reactive_idx, bonded_idx), max(reactive_idx, bonded_idx))
                if bond_pair in changing_bonds:
                    continue
                ring_bond_pairs.append((bonded_idx, reactive_idx))
                expanded.add(bonded_idx)
    return ring_bond_pairs, expanded


def _fix_broken_ring_bonds(xyz: dict,
                           r_mol: 'Molecule',
                           ring_sets: List[Set[int]],
                           reactive_xyz_indices: Set[int],
                           changing_bonds: Set[Tuple[int, int]],
                           max_ring_bond: float = 2.0,
                           ) -> dict:
    """
    Repair ring bonds that were broken during Z-matrix interpolation.

    When the Z-matrix topology opens a ring between a reactive atom and its spectator
    neighbor, the interpolated TS geometry may have ring bond distances of 2+ Angstrom.
    This function detects such bonds and slides the spectator atom (and its non-reactive
    subtree) along the broken bond vector to restore a reasonable distance.

    Args:
        xyz: TS guess XYZ dictionary.
        r_mol: Reactant molecule topology.
        ring_sets: Pre-computed SSSR ring index sets.
        reactive_xyz_indices: Atom indices in the reactive event.
        changing_bonds: Breaking/forming bond pairs ``{(min_i, max_i), ...}``.
        max_ring_bond: Threshold above which a ring bond is considered broken.

    Returns:
        Corrected XYZ dictionary (modified in-place coords).
    """
    import numpy as np
    coords = np.array(xyz['coords'], dtype=float)
    for ring_set in ring_sets:
        # Only repair rings whose bonds are NOT themselves breaking or forming.
        # If a ring bond is in changing_bonds, the ring is part of the reactive
        # event and should not be rigidly repaired.
        ring_has_changing_bond = False
        for ai in ring_set:
            for ba in r_mol.atoms[ai].edges:
                bi = r_mol.atoms.index(ba)
                if bi in ring_set and (min(ai, bi), max(ai, bi)) in changing_bonds:
                    ring_has_changing_bond = True
                    break
            if ring_has_changing_bond:
                break
        if ring_has_changing_bond:
            continue
        for atom_idx in ring_set:
            atom = r_mol.atoms[atom_idx]
            for bonded_atom in atom.edges:
                bonded_idx = r_mol.atoms.index(bonded_atom)
                if bonded_idx > atom_idx and bonded_idx in ring_set:
                    pair = (min(atom_idx, bonded_idx), max(atom_idx, bonded_idx))
                    if pair in changing_bonds:
                        continue
                    dist = float(np.linalg.norm(coords[atom_idx] - coords[bonded_idx]))
                    if dist <= max_ring_bond:
                        continue
                    # Determine which atom is the spectator (to be moved)
                    if atom_idx in reactive_xyz_indices and bonded_idx not in reactive_xyz_indices:
                        mobile, anchor = bonded_idx, atom_idx
                    elif bonded_idx in reactive_xyz_indices and atom_idx not in reactive_xyz_indices:
                        mobile, anchor = atom_idx, bonded_idx
                    elif atom_idx not in reactive_xyz_indices and bonded_idx not in reactive_xyz_indices:
                        # Both spectator — move the one with fewer non-ring bonds
                        mobile, anchor = atom_idx, bonded_idx
                    else:
                        continue  # both reactive — skip
                    # Compute target distance from the reactant geometry
                    r_coords = np.array(xyz['coords'], dtype=float)  # reuse pre-interpolation
                    # Use a standard single-bond length as target (ring bonds are 1.3–1.5 Å)
                    target = 1.45
                    vec = coords[mobile] - coords[anchor]
                    vec_norm = np.linalg.norm(vec)
                    if vec_norm < 0.01:
                        continue
                    shift = vec / vec_norm * (target - vec_norm)
                    # Move the mobile atom and its bonded H atoms
                    moved = {mobile}
                    for sub_bonded in r_mol.atoms[mobile].edges:
                        sub_idx = r_mol.atoms.index(sub_bonded)
                        if sub_idx not in ring_set and sub_idx not in reactive_xyz_indices:
                            moved.add(sub_idx)
                    for idx in moved:
                        coords[idx] += shift
    xyz['coords'] = tuple(tuple(row) for row in coords)
    return xyz


def _fix_rh_add_motif(xyz: dict,
                      r_mol: 'Molecule',
                      forming_bonds: List[Tuple[int, int]],
                      breaking_bonds: List[Tuple[int, int]],
                      changed_bonds: List[Tuple[int, int]],
                      ) -> dict:
    """
    Build the 4-membered ring TS motif for Intra_RH_Add reactions.

    In these reactions an X-H bond breaks, X attacks one end (Z) of a Y=Z
    multiple bond while H attacks the other end (Y), forming a 4-membered
    ring TS: ``X--H···Y==Z--X``.  This function:

    1. Stretches the X-H bond by moving H partway toward Y.
    2. Stretches the Y=Z multiple bond.

    The pattern is detected from the bond lists: a breaking X-H bond, a
    forming Y-H bond, and a changed Y=Z bond where Z is also involved in
    a forming X-Z bond.

    Args:
        xyz: TS guess XYZ dictionary (modified in-place).
        r_mol: Reactant molecule topology.
        forming_bonds: Forming-bond pairs.
        breaking_bonds: Breaking-bond pairs.
        changed_bonds: Bond-order-changed pairs (e.g. C=C → C-C).

    Returns:
        The corrected XYZ dictionary.
    """
    import numpy as np
    if not breaking_bonds or not forming_bonds or not changed_bonds:
        return xyz
    coords = np.array(xyz['coords'], dtype=float)
    symbols = xyz['symbols']

    # Find the pattern: breaking X-H, forming Y-H, changed Y=Z, forming X-Z
    for bx, bh in breaking_bonds:
        if symbols[bh] != 'H':
            bx, bh = bh, bx
        if symbols[bh] != 'H':
            continue
        h_idx, x_idx = bh, bx
        # Find forming bond Y-H
        y_idx = None
        for fa, fb in forming_bonds:
            if fa == h_idx and fb != x_idx:
                y_idx = fb
            elif fb == h_idx and fa != x_idx:
                y_idx = fa
        if y_idx is None:
            continue
        # Find changed bond involving Y → that's the Y=Z double bond
        z_idx = None
        for ca, cb in changed_bonds:
            if ca == y_idx:
                z_idx = cb
            elif cb == y_idx:
                z_idx = ca
        if z_idx is None:
            continue
        # Verify X-Z is a forming bond (ring closure)
        xz = (min(x_idx, z_idx), max(x_idx, z_idx))
        if not any((min(a, b), max(a, b)) == xz for a, b in forming_bonds):
            continue

        # Found the 4-membered ring motif: X--H--Y==Z--X
        # Only apply if the ring-closure distance X-Z is reasonable;
        # if X and Z are too far apart, the ring isn't closing and the
        # motif would produce wrong geometry.
        xz_dist = float(np.linalg.norm(coords[x_idx] - coords[z_idx]))
        if xz_dist > 3.0:
            continue

        # 1. Place H via triangulation between X and Y
        xy_vec = coords[y_idx] - coords[x_idx]
        xy_len = float(np.linalg.norm(xy_vec))
        if xy_len < 0.01:
            continue
        # Pauling TS estimates
        if symbols[x_idx] == 'O':
            d_xh = 0.97 + 0.15   # stretched O-H
        elif symbols[x_idx] == 'C':
            d_xh = 1.09 + 0.38   # stretched C-H
        else:
            d_xh = 1.09 + 0.15
        if symbols[y_idx] == 'C':
            d_hy = 1.09 + 0.35   # forming C-H bond in TS
        elif symbols[y_idx] == 'O':
            d_hy = 0.97 + 0.35   # forming O-H bond in TS
        else:
            d_hy = 1.09 + 0.35
        # Triangulation: find point at distance d_xh from X and d_hy from Y
        # along the X-Y axis, offset perpendicular to stay near original H
        xy_dir = xy_vec / xy_len
        # Project H onto X-Y axis
        xh_vec = coords[h_idx] - coords[x_idx]
        proj = float(np.dot(xh_vec, xy_dir))
        # Target projection from X along X→Y
        if d_xh + d_hy <= xy_len:
            # Circles don't overlap — place linearly
            t_proj = d_xh
        else:
            # Intersection of two spheres: solve for projection
            t_proj = (xy_len**2 + d_xh**2 - d_hy**2) / (2.0 * xy_len)
        perp = xh_vec - proj * xy_dir
        perp_len = float(np.linalg.norm(perp))
        # Perpendicular distance from X-Y axis
        r_sq = d_xh**2 - t_proj**2
        if r_sq < 0.01:
            r_perp = 0.0
        else:
            r_perp = float(np.sqrt(r_sq))
        if perp_len > 0.01:
            perp_dir = perp / perp_len
        else:
            # H is on the X-Y axis — pick an arbitrary perpendicular
            arb = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(xy_dir, arb)) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            perp_dir = np.cross(xy_dir, arb)
            perp_dir /= np.linalg.norm(perp_dir)
        coords[h_idx] = coords[x_idx] + t_proj * xy_dir + r_perp * perp_dir

        # 2. Stretch the Y=Z double bond by ~5%
        yz_vec = coords[z_idx] - coords[y_idx]
        yz_len = float(np.linalg.norm(yz_vec))
        if yz_len > 0.01:
            stretch = 0.05 * yz_len
            yz_dir = yz_vec / yz_len
            coords[z_idx] += 0.5 * stretch * yz_dir
            coords[y_idx] -= 0.5 * stretch * yz_dir

    xyz['coords'] = tuple(tuple(row) for row in coords)
    return xyz


def _clear_forming_bond_path(xyz: dict,
                              r_mol: 'Molecule',
                              forming_bonds: List[Tuple[int, int]],
                              ) -> dict:
    """
    Reflect H atoms that project into the forming-bond path during ring closure.

    When a new bond forms between two atoms, H atoms bonded to either endpoint may
    point inward (toward the partner) rather than outward.  This physically blocks
    ring closure and produces TS guesses with unrealistically short H-to-partner
    distances.

    For each H on a forming-bond atom, if the H-parent-partner angle is acute
    (< 90 deg), the H is reflected through the plane perpendicular to the forming
    bond that passes through its parent atom.

    Args:
        xyz: TS guess XYZ dictionary (modified in-place).
        r_mol: Reactant molecule topology.
        forming_bonds: List of forming-bond ``(atom_i, atom_j)`` pairs.

    Returns:
        The corrected XYZ dictionary.
    """
    import numpy as np
    coords = np.array(xyz['coords'], dtype=float)
    for a, b in forming_bonds:
        # Only clear for heavy-atom forming bonds that close a ring across a
        # substantial chain (topological distance >= 4 in the reactant graph).
        # Skip H-migration bonds (endpoint is H) and short-range bonds.
        if xyz['symbols'][a] == 'H' or xyz['symbols'][b] == 'H':
            continue
        # BFS for topological distance between a and b in the reactant
        from collections import deque
        visited = {a}
        queue = deque([(a, 0)])
        topo_dist = 0
        while queue:
            node, depth = queue.popleft()
            if node == b:
                topo_dist = depth
                break
            for nb in r_mol.atoms[node].edges:
                nb_idx = r_mol.atoms.index(nb)
                if nb_idx not in visited:
                    visited.add(nb_idx)
                    queue.append((nb_idx, depth + 1))
        if topo_dist < 4:
            continue
        for endpoint, partner in [(a, b), (b, a)]:
            bond_vec = coords[partner] - coords[endpoint]
            bond_len = np.linalg.norm(bond_vec)
            if bond_len < 0.01:
                continue
            bond_dir = bond_vec / bond_len
            atom = r_mol.atoms[endpoint]
            for bonded_atom in atom.edges:
                h_idx = r_mol.atoms.index(bonded_atom)
                if xyz['symbols'][h_idx] != 'H':
                    continue
                h_vec = coords[h_idx] - coords[endpoint]
                proj = float(np.dot(h_vec, bond_dir))
                if proj > 0:
                    # H projects toward the partner — reflect it away
                    coords[h_idx] -= 2.0 * proj * bond_dir
    xyz['coords'] = tuple(tuple(row) for row in coords)
    return xyz


def interpolate(rxn: 'ARCReaction',
                weight: float = 0.5,
                existing_xyzs: Optional[List[dict]] = None,
                ) -> Optional[List[dict]]:
    """
    Search for a TS by interpolating internal coords.

    Args:
        rxn (ARCReaction): The reaction to process.
        weight (float): Interpolation weight (0=reactant-like, 1=product-like).
        existing_xyzs (List[dict], optional): Previously generated XYZ guesses
            (e.g. from earlier weight iterations) to deduplicate against.

    Returns:
        Optional[List[dict]]: XYZ coordinate guesses in reactant atom ordering.
    """
    if rxn.is_isomerization():
        # A <=> B
        return interpolate_isomerization(rxn=rxn, weight=weight, existing_xyzs=existing_xyzs)
    elif rxn.is_unimolecular():
        # A <=> B + C or A + B <=> C
        return interpolate_addition(rxn=rxn, weight=weight, existing_xyzs=existing_xyzs)
    return None


def interpolate_addition(rxn: 'ARCReaction',
                         weight: float = 0.5,
                         existing_xyzs: Optional[List[dict]] = None,
                         ) -> Optional[List[dict]]:
    """
    Search for a TS of a non-isomerization unimolecular reaction where one side
    has a single species and the other has multiple species (A <=> B + C, or
    B + C <=> A).

    The algorithm starts from the unimolecular species and stretches the
    bonds that fragment it into the multi-species products.  Two strategies
    are tried in order:

    1.  **Template-guided** — uses ``product_dicts`` and the RMG reaction
        recipe to identify reactive bonds.  Bond classification (split vs.
        cross) is direction-agnostic: bonds that exist in the unimolecular
        species' molecular graph are "split bonds" (stretched), and bonds
        that don't are "cross bonds" (used for insertion-ring detection).
        This handles ``discovered_in_reverse`` correctly without swapping.

    2.  **Fragmentation fallback** — when no ``product_dicts`` are available
        or the template gives unreliable bonds, the function enumerates
        single- and two-bond cuts in the unimolecular species' graph and
        selects cuts whose fragments match the product species' element
        compositions.

    In both strategies the actual TS geometry is built by ``stretch_bond``,
    which rigidly translates the smaller fragment(s) so that the split bonds
    reach their Pauling TS estimates (single-bond length + 0.42 Å).  For 3-fragment
    insertion/elimination patterns, ``try_insertion_ring`` positions the
    fragments in a 3-membered TS ring using triangle geometry.

    Args:
        rxn (ARCReaction): The reaction to process.
        weight (float): Interpolation weight (0 = reactant-like, 1 = product-like).
        existing_xyzs (List[dict], optional): Previously generated XYZ guesses
            (e.g. from earlier weight iterations) to deduplicate against.

    Returns:
        Optional[List[dict]]: Validated XYZ coordinate guesses in the unimolecular-species
            atom ordering, or an empty list if no valid guess could be produced.
    """
    if not (0.0 <= weight <= 1.0):
        return None

    n_r = len(rxn.r_species)
    n_p = len(rxn.p_species)
    if n_r == 1 and n_p > 1:
        uni_species = rxn.r_species[0]
        multi_species = rxn.p_species
        uni_is_product = False
    elif n_p == 1 and n_r > 1:
        uni_species = rxn.p_species[0]
        multi_species = rxn.r_species
        uni_is_product = True
    else:
        logger.debug(f'Linear addition (rxn={rxn.label}): not an addition/dissociation reaction.')
        return []

    uni_xyz = uni_species.get_xyz()
    uni_mol = uni_species.mol

    ts_xyzs: List[dict] = []
    seen_split_sets: Set[frozenset] = set()
    verified_guess_count: int = 0

    # --- Dedicated family builders (fire first, return immediately if matched) ---
    # Dispatch by family name when known, fall back to structural detection
    # for unrecognized families.
    _family_name = rxn.family or ''
    for pd in rxn.product_dicts:
        if pd.get('family'):
            _family_name = pd['family']
            break

    # XY_elimination_hydroxyl: concerted 6-membered ring elimination.
    if _family_name.startswith('XY_elimination') or (
            not _family_name and len(multi_species) >= 3):
        ts_family = build_xy_elimination_ts(uni_xyz, uni_mol)
        if ts_family is not None:
            logger.debug(f'Linear addition (rxn={rxn.label}): used XY_elimination dedicated builder.')
            return [ts_family]

    # Build a set of bonds present in the unimolecular species' graph (used
    # by both the template-guided path and the fragmentation fallback).
    atom_to_idx = {atom: idx for idx, atom in enumerate(uni_mol.atoms)}
    uni_bond_set: Set[Tuple[int, int]] = set()
    for atom in uni_mol.atoms:
        idx_a = atom_to_idx[atom]
        for neighbor in atom.edges:
            idx_b = atom_to_idx[neighbor]
            uni_bond_set.add((min(idx_a, idx_b), max(idx_a, idx_b)))

    # ----- Strategy 1: template-guided (product_dicts) -----
    for i, product_dict in enumerate(rxn.product_dicts):
        r_label_map = product_dict['r_label_map']
        if r_label_map is None:
            continue

        bb, fb = rxn.get_expected_changing_bonds(r_label_dict=r_label_map)
        all_reactive_r: List[Tuple[int, int]] = list(bb or []) + list(fb or [])
        if not all_reactive_r:
            continue

        # Convert to uni-species ordering.
        if uni_is_product:
            try:
                atom_map = map_rxn(rxn=rxn, product_dict_index_to_try=i)
            except Exception:
                atom_map = None
            if atom_map is None:
                continue
            all_reactive_uni = [(min(atom_map[a], atom_map[b]),
                                 max(atom_map[a], atom_map[b])) for a, b in all_reactive_r]
        else:
            all_reactive_uni = [(min(a, b), max(a, b)) for a, b in all_reactive_r]

        # Direction-agnostic classification.
        split_bonds = [b for b in all_reactive_uni if b in uni_bond_set]
        cross_bonds = [b for b in all_reactive_uni if b not in uni_bond_set]
        if not split_bonds:
            continue

        sb_key = frozenset(split_bonds)
        if sb_key in seen_split_sets:
            continue
        seen_split_sets.add(sb_key)

        # ---- Verify fragments via subgraph isomorphism ----
        frag_map = map_and_verify_fragments(
            uni_mol=uni_mol,
            split_bonds=split_bonds,
            multi_species=multi_species,
            cross_bonds=cross_bonds,
            product_dict=product_dict,
            uni_is_reactant=not uni_is_product,
        )

        if frag_map is not None:
            # Use the verified mapping to determine product-guided fragments.
            product_groups: Dict[int, Set[int]] = {}
            for r_idx, (sp_idx, _) in frag_map.items():
                product_groups.setdefault(sp_idx, set()).add(r_idx)
            sorted_groups = sorted(product_groups.values(), key=len)
            small_prod_atoms = sorted_groups[0]
            large_prod_atoms = sorted_groups[-1] if len(sorted_groups) > 1 else set()

            # Identify migrating atoms: atoms in the small product that are
            # only bonded (in the reactant) to atoms in the large product.
            adj_full: Dict[int, Set[int]] = {k: set() for k in range(len(uni_mol.atoms))}
            for atom in uni_mol.atoms:
                ia = atom_to_idx[atom]
                for nbr in atom.edges:
                    ib = atom_to_idx[nbr]
                    adj_full[ia].add(ib)
            heavy_in_small = [idx for idx in small_prod_atoms
                              if uni_xyz['symbols'][idx] != 'H']
            if heavy_in_small:
                core_seed = heavy_in_small[0]
            else:
                core_seed = min(small_prod_atoms) if small_prod_atoms else None
            if core_seed is not None:
                core: Set[int] = set()
                queue: deque = deque([core_seed])
                while queue:
                    node = queue.popleft()
                    if node in core:
                        continue
                    if node not in small_prod_atoms:
                        continue
                    core.add(node)
                    queue.extend(adj_full[node] - core)
                migrating_atoms = small_prod_atoms - core
            else:
                core = small_prod_atoms
                migrating_atoms = set()

            # Apply intra-fragment ring contraction BEFORE stretching.
            # Contraction moves atoms (e.g. S toward C in cyclic thioether
            # formation) that affect the stretch direction.  Contracting
            # first ensures the break-bond stretch operates from the
            # correct geometry.  When no contraction applies, bases is
            # just [uni_xyz] (identity).
            bases = apply_intra_frag_contraction(
                uni_xyz, uni_mol, split_bonds, cross_bonds, multi_species,
                weight, label=f'rxn={rxn.label}, path={i}')
            # When contraction produced modified geometries, also keep
            # the uncontracted original as a fallback candidate.
            if bases[0] is not uni_xyz:
                bases.append(uni_xyz)

            for base_xyz in bases:
                ts_xyz = stretch_bond(base_xyz, uni_mol, split_bonds,
                                       cross_bonds, weight,
                                       label=f'rxn={rxn.label}, path={i}')
                if ts_xyz is not None and migrating_atoms:
                    # stretch_bond only moves the smallest fragment (often
                    # just the migrating atom).  When there are 3+ fragments,
                    # the core of the small product may still need to be
                    # stretched away from the large product (e.g. C-O in
                    # HO2 elimination).
                    ts_xyz = stretch_core_from_large(
                        ts_xyz, uni_mol, split_bonds, core, large_prod_atoms,
                        small_prod_atoms, weight)
                    # Reset migrating atoms to their pre-stretch positions
                    # before calling migrate_verified_atoms.  stretch_bond
                    # may have moved them in the wrong direction.
                    ts_coords_arr = np.array(ts_xyz['coords'], dtype=float)
                    base_coords = np.array(base_xyz['coords'], dtype=float)
                    for m_idx in migrating_atoms:
                        ts_coords_arr[m_idx] = base_coords[m_idx]
                    ts_xyz = {
                        'symbols': ts_xyz['symbols'],
                        'isotopes': ts_xyz.get('isotopes',
                                               tuple(0 for _ in range(len(ts_xyz['symbols'])))),
                        'coords': tuple(tuple(float(x) for x in row)
                                        for row in ts_coords_arr),
                    }
                    # Place migrating atoms at TS-like positions between
                    # donor (in large_prod_atoms) and acceptor (in core).
                    ts_xyz = migrate_verified_atoms(
                        ts_xyz, uni_mol, migrating_atoms, core,
                        large_prod_atoms, weight, cross_bonds=cross_bonds)
                    is_valid, reason = validate_ts_guess(
                        ts_xyz, set(), split_bonds, uni_mol,
                        label=f'rxn={rxn.label}, path={i}-post-migrate')
                    if not is_valid:
                        logger.debug(
                            f'Linear addition (rxn={rxn.label}, path={i}): '
                            f'post-migration rejected — {reason}.')
                        continue
                if ts_xyz is not None:
                    ts_xyzs.append(ts_xyz)
                    verified_guess_count += 1
        else:
            # Fragment verification failed — fall back to direct stretch.
            logger.debug(f'Linear addition (rxn={rxn.label}, path={i}): '
                         f'fragment verification failed, falling back to direct stretch.')
            # Try direct stretch first (works for Diels–Alder and other
            # simple additions where no intra-fragment contraction is needed).
            ts_xyz = stretch_bond(uni_xyz, uni_mol, split_bonds,
                                   cross_bonds, weight,
                                   label=f'rxn={rxn.label}, path={i}-fallback')
            if ts_xyz is not None:
                ts_xyzs.append(ts_xyz)
            # Also try ring contraction then stretch for reactions that need
            # it (e.g. cyclic ether formation).
            ring_xyzs = apply_intra_frag_contraction(
                uni_xyz, uni_mol, split_bonds, cross_bonds, multi_species,
                weight, label=f'rxn={rxn.label}, path={i}-fallback')
            for ring_xyz in ring_xyzs:
                ts_xyz = stretch_bond(ring_xyz, uni_mol, split_bonds,
                                       cross_bonds, weight,
                                       label=f'rxn={rxn.label}, path={i}-fallback')
                if ts_xyz is not None:
                    is_valid, reason = validate_ts_guess(
                        ts_xyz, set(), split_bonds, uni_mol,
                        label=f'rxn={rxn.label}, path={i}-fallback-contracted')
                    if is_valid:
                        ts_xyzs.append(ts_xyz)

    # ----- Concerted multi-bond supplement -----
    # For reactions with 3+ products and multiple split/cross bonds (e.g.
    # concerted eliminations, retro-cycloadditions), simultaneously stretch
    # breaking bonds and contract forming bonds from the unimolecular geometry.
    # Only fires for 3+ products to avoid interfering with simple dissociations.
    if len(multi_species) >= 3:
        for product_dict in rxn.product_dicts:
            r_label_map = product_dict.get('r_label_map')
            if r_label_map is None:
                continue
            bb_conc, fb_conc = rxn.get_expected_changing_bonds(r_label_dict=r_label_map)
            if not bb_conc or not fb_conc:
                continue
            if uni_is_product:
                try:
                    am = map_rxn(rxn=rxn,
                                 product_dict_index_to_try=rxn.product_dicts.index(product_dict))
                except Exception:
                    am = None
                if am is None:
                    continue
                sb_conc = [(min(am[a], am[b]), max(am[a], am[b])) for a, b in bb_conc]
                cb_conc = [(min(am[a], am[b]), max(am[a], am[b])) for a, b in fb_conc]
            else:
                sb_conc = [(min(a, b), max(a, b)) for a, b in bb_conc]
                cb_conc = [(min(a, b), max(a, b)) for a, b in fb_conc]
            split_in_uni = [b for b in sb_conc if b in uni_bond_set]
            cross_in_uni = [b for b in cb_conc if b not in uni_bond_set]
            if len(split_in_uni) >= 2 and cross_in_uni:
                ts_conc = build_concerted_ts(uni_xyz, uni_mol, split_in_uni, cross_in_uni, weight)
                if ts_conc is not None:
                    ts_xyzs.append(ts_conc)
                    logger.debug(f'Linear addition (rxn={rxn.label}): used concerted-TS builder '
                                 f'(split={split_in_uni}, cross={cross_in_uni}).')

    # ----- SN2-like supplement: ring-close then stretch leaving group -----
    # For reactions where a bond breaks and a bond forms at the same pivot
    # atom (e.g. intra_substitutionCS_cyclization), build the TS by:
    #   1. Ring-closing the forming bond (bringing the attacking atom close)
    #   2. Stretching the breaking bond (moving the leaving group away)
    #   3. Slightly stretching the forming bond to TS distance
    for product_dict in rxn.product_dicts:
        r_label_map = product_dict.get('r_label_map')
        if r_label_map is None:
            continue
        bb_sn2, fb_sn2 = rxn.get_expected_changing_bonds(r_label_dict=r_label_map)
        if not bb_sn2 or not fb_sn2 or len(bb_sn2) != 1 or len(fb_sn2) != 1:
            continue
        bb_atoms_sn2 = {a for bond in bb_sn2 for a in bond}
        fb_atoms_sn2 = {a for bond in fb_sn2 for a in bond}
        shared = bb_atoms_sn2 & fb_atoms_sn2
        if len(shared) != 1:
            continue
        pivot = next(iter(shared))
        if uni_xyz['symbols'][pivot] == 'H':
            continue
        bb_other = bb_sn2[0][0] if bb_sn2[0][1] == pivot else bb_sn2[0][1]
        fb_other = fb_sn2[0][0] if fb_sn2[0][1] == pivot else fb_sn2[0][1]

        # Step 1: ring-close the forming bond.
        rc_sn2 = ring_closure_xyz(uni_xyz, uni_mol, forming_bond=(fb_other, pivot))
        if rc_sn2 is None:
            continue
        sn2_coords = np.array(rc_sn2['coords'], dtype=float)

        # Build molecular-graph adjacency (not distance-based) for
        # fragment detection — avoids false bonds from close contacts.
        mol_adj: Dict[int, Set[int]] = {k: set() for k in range(len(uni_xyz['symbols']))}
        atom_to_idx_sn2 = {a: idx for idx, a in enumerate(uni_mol.atoms)}
        for atom in uni_mol.atoms:
            i = atom_to_idx_sn2[atom]
            for nbr in atom.bonds:
                mol_adj[i].add(atom_to_idx_sn2[nbr])

        # Step 1.5: orient the leaving group away from the attacking atom.
        # After ring closure, the leaving group (bb_other and its fragment)
        # may have folded inward toward the ring.  Check the angle
        # fb_other-pivot-bb_other: if < 90° the leaving group is on the
        # same side as the attacker.  Reflect it through the pivot to the
        # opposite side.
        vec_attack = sn2_coords[fb_other] - sn2_coords[pivot]
        vec_leave_cur = sn2_coords[bb_other] - sn2_coords[pivot]
        d_att = float(np.linalg.norm(vec_attack))
        d_lea = float(np.linalg.norm(vec_leave_cur))
        if d_att > 1e-6 and d_lea > 1e-6:
            cos_angle = np.dot(vec_attack, vec_leave_cur) / (d_att * d_lea)
            if cos_angle > 0:  # angle < 90° → leaving group on same side
                # Reflect the leaving-group fragment through the pivot.
                # New position: pivot + reflected vector.
                leave_frag: Set[int] = set()
                q_leave = deque([bb_other])
                while q_leave:
                    node = q_leave.popleft()
                    if node in leave_frag or node == pivot:
                        continue
                    leave_frag.add(node)
                    q_leave.extend(mol_adj[node] - leave_frag - {pivot})
                # Reflect each atom: new_pos = 2*pivot - old_pos
                for k in leave_frag:
                    sn2_coords[k] = 2.0 * sn2_coords[pivot] - sn2_coords[k]

        # Step 2: stretch the breaking bond (move leaving-group fragment).
        sbl_break = get_single_bond_length(
            uni_xyz['symbols'][pivot], uni_xyz['symbols'][bb_other]) or 1.8
        d_break_target = sbl_break * 1.5  # ~2.7 Å for C-S
        vec_leave = sn2_coords[bb_other] - sn2_coords[pivot]
        d_break_cur = float(np.linalg.norm(vec_leave))
        if d_break_cur > 1e-6 and d_break_target > d_break_cur:
            direction = vec_leave / d_break_cur
            displacement = direction * (d_break_target - d_break_cur)
            # Move the leaving-group fragment (BFS from bb_other, not crossing pivot).
            frag: Set[int] = set()
            queue_sn2 = deque([bb_other])
            while queue_sn2:
                node = queue_sn2.popleft()
                if node in frag or node == pivot:
                    continue
                frag.add(node)
                queue_sn2.extend(mol_adj[node] - frag - {pivot})
            for k in frag:
                sn2_coords[k] += displacement

        # Step 3: slightly stretch the forming bond.
        sbl_form = get_single_bond_length(
            uni_xyz['symbols'][pivot], uni_xyz['symbols'][fb_other]) or 1.5
        d_form_target = sbl_form * 1.5  # ~2.3 Å for C-C
        vec_form = sn2_coords[fb_other] - sn2_coords[pivot]
        d_form_cur = float(np.linalg.norm(vec_form))
        if d_form_cur > 1e-6 and d_form_target > d_form_cur:
            direction_form = vec_form / d_form_cur
            disp_form = direction_form * (d_form_target - d_form_cur)
            # Move the attacking-atom fragment.
            frag_form: Set[int] = set()
            queue_form = deque([fb_other])
            while queue_form:
                node = queue_form.popleft()
                if node in frag_form or node == pivot:
                    continue
                frag_form.add(node)
                queue_form.extend(mol_adj[node] - frag_form - {pivot})
            for k in frag_form:
                sn2_coords[k] += disp_form

        ts_sn2 = {'symbols': uni_xyz['symbols'], 'isotopes': uni_xyz['isotopes'],
                   'coords': tuple(tuple(row) for row in sn2_coords)}
        if not colliding_atoms(ts_sn2):
            ts_xyzs.append(ts_sn2)
            logger.debug(f'Linear (rxn={rxn.label}): used SN2-like builder '
                         f'(pivot={pivot}, leaving={bb_other}, attacking={fb_other}).')
            break

    # ----- Strategy 2: fragmentation supplement -----
    # Always run fragmentation-based search in addition to the template path.
    # The template's r_label_map can mis-identify the primary fragmentation
    # (e.g. marking a C-H bond as the split instead of the C-O bond in
    # 1,3_Insertion_ROR), so fragmentation supplements those results.
    #
    # Two-tier filtering of fragmentation cuts:
    # 1. Subgraph isomorphism (via map_and_verify_fragments) is the gold
    #    standard: it checks that each fragment is topologically isomorphic
    #    to one of the multi_species.  This catches wrong-reaction cuts
    #    (e.g. H removal from the wrong C atom producing a different radical).
    #    However, it requires that the cut produces exactly len(multi_species)
    #    fragments, which fails for multi-bond cuts in insertion/elimination
    #    reactions where cross_bonds are needed to merge fragments.
    # 2. Composition dedup: when a cut fails the subgraph check, fall back
    #    to keeping one representative per unique set of fragment element
    #    compositions.  This is coarser but works for all cut types.
    cut_lists = find_split_bonds_by_fragmentation(uni_mol, multi_species)
    isomorphism_verified: List[List[Tuple[int, int]]] = []
    isomorphism_unverified: List[List[Tuple[int, int]]] = []
    for cut in cut_lists:
        frag_map = map_and_verify_fragments(
            uni_mol=uni_mol,
            split_bonds=cut,
            multi_species=multi_species,
        )
        if frag_map is not None:
            isomorphism_verified.append(cut)
        else:
            isomorphism_unverified.append(cut)

    # For unverified cuts, skip only those whose fragment composition is
    # already covered by a *verified* cut (those are topologically confirmed).
    # Do NOT dedup among unverified cuts themselves — they may represent
    # geometrically distinct TS guesses (e.g. H₂ loss from different
    # positions on a PAH), and only some may pass downstream validation.
    verified_compositions: Set[Tuple[Tuple[Tuple[str, int], ...], ...]] = set()
    uni_symbols = tuple(atom.element.symbol for atom in uni_mol.atoms)
    for cut in isomorphism_verified:
        comp_key = _frag_composition_key(uni_mol, atom_to_idx, uni_symbols, cut)
        verified_compositions.add(comp_key)
    composition_filtered: List[List[Tuple[int, int]]] = []
    for cut in isomorphism_unverified:
        comp_key = _frag_composition_key(uni_mol, atom_to_idx, uni_symbols, cut)
        if comp_key not in verified_compositions:
            composition_filtered.append(cut)
    # Drop unverified fragmentation cuts when Strategy 1 already produced
    # verified guesses or Strategy 2 has its own verified cuts.  Verified
    # cuts use subgraph isomorphism against product species; unverified
    # cuts only match by element composition, which allows wrong
    # reactive-site assignments (e.g. migrating a methyl H instead of a
    # hydroxyl H in HO2 elimination when only one of the two split bonds
    # is identified).  When Strategy 1 produced no usable guesses (e.g.
    # retroene where all verified geometries fail validation), unverified
    # cuts remain as a necessary fallback.
    if verified_guess_count > 0 or isomorphism_verified:
        cut_lists = isomorphism_verified
    else:
        cut_lists = isomorphism_verified + composition_filtered

    # For 3+-product reactions, synthesise 3-bond cuts by merging pairs of
    # 2-bond cuts that share exactly one bond.  This handles concerted
    # eliminations (e.g. XY_elimination_hydroxyl) where two H atoms must
    # detach simultaneously and merge into H₂.
    if len(multi_species) >= 3:
        merged_cuts: List[List[Tuple[int, int]]] = []
        seen_parent_pairs: Set[Tuple[int, int]] = set()
        all_2bond = [c for c in cut_lists if len(c) == 2]
        for ci in range(len(all_2bond)):
            for cj in range(ci + 1, len(all_2bond)):
                shared = set(map(tuple, all_2bond[ci])) & set(map(tuple, all_2bond[cj]))
                if len(shared) == 1:
                    merged = list(set(map(tuple, all_2bond[ci])) | set(map(tuple, all_2bond[cj])))
                    if frozenset(merged) in seen_split_sets:
                        continue
                    # Deduplicate by parent-atom pair: only keep one merged
                    # cut per unique pair of heavy-atom parents of the
                    # detached H atoms.  This avoids producing 5 nearly
                    # identical concerted guesses for (H5-H10, H6-H10, H7-H10, ...).
                    h_parents = []
                    for a, b in merged:
                        for idx in (a, b):
                            if uni_xyz['symbols'][idx] == 'H':
                                for nbr in uni_mol.atoms[idx].bonds:
                                    pi = atom_to_idx[nbr]
                                    if uni_xyz['symbols'][pi] != 'H':
                                        h_parents.append(pi)
                                        break
                    parent_key = tuple(sorted(set(h_parents)))
                    if parent_key in seen_parent_pairs:
                        continue
                    seen_parent_pairs.add(parent_key)
                    merged_cuts.append(merged)
        # Put merged cuts FIRST so concerted guesses get priority.
        cut_lists = merged_cuts + cut_lists

    for cut in cut_lists:
        sb_key = frozenset(cut)
        if sb_key in seen_split_sets:
            continue
        seen_split_sets.add(sb_key)

        # --- Detect cross bonds for concerted builder ---
        # After cutting split bonds, find fragments. If two single-atom
        # fragments can merge to match a product (e.g. two H's → H₂),
        # the bond between them is a cross bond.
        _adj_cut: Dict[int, Set[int]] = {k: set() for k in range(len(uni_mol.atoms))}
        for atom in uni_mol.atoms:
            ia = atom_to_idx[atom]
            for nbr in atom.edges:
                ib = atom_to_idx[nbr]
                _adj_cut[ia].add(ib)
        for a, b in cut:
            _adj_cut[a].discard(b)
            _adj_cut[b].discard(a)
        _vis: Set[int] = set()
        _frags: List[Set[int]] = []
        for s in range(len(uni_mol.atoms)):
            if s in _vis:
                continue
            comp: Set[int] = set()
            stk = [s]
            while stk:
                n = stk.pop()
                if n in _vis:
                    continue
                _vis.add(n)
                comp.add(n)
                stk.extend(_adj_cut[n] - _vis)
            _frags.append(comp)
        # Identify cross bonds: single-atom H fragments that should merge
        # into H₂.  Only pair H atoms from DIFFERENT parent heavy atoms —
        # two H's from the same CH₃ group would not form H₂ in a concerted TS.
        h_singletons = [f for f in _frags if len(f) == 1
                        and uni_xyz['symbols'][next(iter(f))] == 'H']
        cross_bonds_frag: List[Tuple[int, int]] = []
        if len(h_singletons) >= 2:
            has_h2_product = any(
                len(sp.mol.atoms) == 2
                and all(a.element.symbol == 'H' for a in sp.mol.atoms)
                for sp in multi_species)
            if has_h2_product:
                # Find parent heavy atom for each singleton H.
                _h_parent: Dict[int, int] = {}
                for sf in h_singletons:
                    h_idx = next(iter(sf))
                    for nbr in uni_mol.atoms[h_idx].bonds:
                        pi = atom_to_idx[nbr]
                        if uni_xyz['symbols'][pi] != 'H':
                            _h_parent[h_idx] = pi
                            break
                for si in range(len(h_singletons)):
                    for sj in range(si + 1, len(h_singletons)):
                        ai = next(iter(h_singletons[si]))
                        aj = next(iter(h_singletons[sj]))
                        if _h_parent.get(ai) != _h_parent.get(aj):
                            cross_bonds_frag.append((min(ai, aj), max(ai, aj)))
                            break  # one cross bond per cut is enough
                    if cross_bonds_frag:
                        break

        # Try concerted builder when cross bonds are detected.
        # When it succeeds, skip the stretch_bond fallback for this cut —
        # the concerted geometry is strictly better than fragment-stretching
        # which produces "pull one H off" artifacts.
        used_concerted = False
        if cross_bonds_frag and len(cut) >= 2:
            ts_conc = build_concerted_ts(uni_xyz, uni_mol, cut, cross_bonds_frag, weight)
            if ts_conc is not None and not colliding_atoms(ts_conc):
                ts_xyzs.append(ts_conc)
                used_concerted = True
                logger.debug(f'Linear addition (rxn={rxn.label}, frag-fallback): '
                             f'used concerted-TS builder (split={cut}, '
                             f'cross={cross_bonds_frag}).')

        if used_concerted:
            continue  # skip stretch_bond fallback for this cut

        # Apply ring contraction on original geometry first, then stretch.
        ring_xyzs = apply_intra_frag_contraction(
            uni_xyz, uni_mol, cut, None, multi_species,
            weight, label=f'rxn={rxn.label}, frag-fallback')
        for ring_xyz in ring_xyzs:
            ts_xyz = stretch_bond(ring_xyz, uni_mol, cut, cross_bonds=None,
                                   weight=weight,
                                   label=f'rxn={rxn.label}, frag-fallback')
            if ts_xyz is not None:
                ts_xyz = migrate_h_between_fragments(
                    ts_xyz, uni_mol, cut, multi_species, weight)
                is_valid, reason = validate_ts_guess(
                    ts_xyz, set(), cut, uni_mol,
                    label=f'rxn={rxn.label}, frag-fallback-post-migrate')
                if not is_valid:
                    logger.debug(f'Linear (rxn={rxn.label}, frag-fallback): '
                                 f'post-migration guess rejected — {reason}.')
                    continue
                ts_xyzs.append(ts_xyz)

    # Deduplicate near-identical guesses (including against existing_xyzs
    # from prior weight iterations to avoid wasting downstream DFT resources).
    prior: List[dict] = list(existing_xyzs or [])
    unique: List[dict] = []
    for xyz_candidate in ts_xyzs:
        if colliding_atoms(xyz_candidate):
            continue
        if not any(almost_equal_coords(xyz_candidate, u)
                   for u in unique + prior):
            unique.append(xyz_candidate)

    # Cap: keep at most 5 guesses to avoid flooding downstream DFT.
    if len(unique) > 5:
        logger.debug(f'Linear addition (rxn={rxn.label}): capping {len(unique)} guesses to 5.')
        unique = unique[:5]

    return unique if unique else []


# ---------------------------------------------------------------------------
# Per-path context builder and TS-building strategies
# ---------------------------------------------------------------------------

def _reactive_heavy_atoms_share_ring(bonds_to_check: list,
                                     mol: 'Molecule',
                                     symbols: tuple,
                                     ring_sets: List[Set[int]],
                                     ) -> bool:
    """Check whether the reactive heavy atoms in any bond share a ring."""
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    for a, b in bonds_to_check:
        ha, hb = a, b
        if symbols[ha] == 'H':
            for nbr in mol.atoms[ha].bonds:
                ni = atom_to_idx[nbr]
                if symbols[ni] != 'H':
                    ha = ni
                    break
        if symbols[hb] == 'H':
            for nbr in mol.atoms[hb].bonds:
                ni = atom_to_idx[nbr]
                if symbols[ni] != 'H':
                    hb = ni
                    break
        for ring_set in ring_sets:
            if ha in ring_set and hb in ring_set:
                return True
    return False


def _build_path_context(r_xyz: dict, r_mol: 'Molecule', op_xyz: dict,
                        mapped_p_mol: 'Molecule',
                        bb: list, fb: list,
                        family: Optional[str], r_label_map: Optional[dict],
                        weight: float, ring_sets: List[Set[int]],
                        label: str, discovered_in_reverse: bool,
                        ) -> _PathContext:
    """Build a fully-populated _PathContext for the strategy pipeline."""
    # Smart anchor selection.
    anchors = find_smart_anchors(r_mol, breaking_bonds=bb, forming_bonds=fb)

    # Reactive indices: atoms involved in forming/breaking bonds.
    reactive_xyz_indices: Set[int] = set()
    for bond in bb + fb:
        reactive_xyz_indices.update(bond)

    # Drop ring heavy atoms with large R→P displacement (atom-map artifact).
    r_coords = np.array(r_xyz['coords'], dtype=float)
    op_coords = np.array(op_xyz['coords'], dtype=float)
    ring_atom_indices = set().union(*ring_sets) if ring_sets else set()
    heavy_bond_atoms: Set[int] = set()
    for a, b in bb + fb:
        if r_xyz['symbols'][a] != 'H' and r_xyz['symbols'][b] != 'H':
            heavy_bond_atoms.add(a)
            heavy_bond_atoms.add(b)
    drop = set()
    for idx in reactive_xyz_indices:
        if r_xyz['symbols'][idx] != 'H' and idx in ring_atom_indices \
                and idx not in heavy_bond_atoms:
            disp = float(np.linalg.norm(r_coords[idx] - op_coords[idx]))
            if disp > 1.5:
                drop.add(idx)
    if drop:
        reactive_xyz_indices -= drop
        logger.debug(f'Linear ({label}): dropped ring heavy atoms {drop} '
                     f'from reactive set (large R→P displacement).')

    # Ring-preservation bonds.
    changing_bonds = {(min(a, b), max(a, b)) for a, b in bb + fb}
    ring_bonds, reactive_xyz_indices = _get_ring_preservation_bonds(
        r_mol, reactive_xyz_indices, changing_bonds, ring_sets)

    # Constraints.
    constraints = get_r_constraints(expected_breaking_bonds=bb, expected_forming_bonds=fb)
    if ring_bonds:
        reactive_bond_atoms = {a for bond in bb + fb for a in bond}
        constrained_atoms = reactive_bond_atoms | {a for pair in constraints.get('R_atom', []) for a in pair}
        added = False
        for rb in ring_bonds:
            if not added and rb[0] not in constrained_atoms and rb[1] not in constrained_atoms:
                constraints['R_atom'].append(rb)
                constrained_atoms.add(rb[0])
                constrained_atoms.add(rb[1])
                added = True
            else:
                reactive_xyz_indices.discard(rb[0])

    # Near-attack rotation.
    fb_in_ring = _reactive_heavy_atoms_share_ring(fb, r_mol, r_xyz['symbols'], ring_sets)
    bb_in_ring = _reactive_heavy_atoms_share_ring(bb, r_mol, r_xyz['symbols'], ring_sets)
    r_xyz_na = r_xyz if fb_in_ring else get_near_attack_xyz(r_xyz, r_mol, bonds=fb)
    op_xyz_na = op_xyz if bb_in_ring else get_near_attack_xyz(op_xyz, mapped_p_mol, bonds=bb)

    return _PathContext(
        r_xyz=r_xyz, r_mol=r_mol, op_xyz=op_xyz, mapped_p_mol=mapped_p_mol,
        bb=bb, fb=fb, family=family, r_label_map=r_label_map,
        weight=weight, ring_sets=ring_sets, label=label,
        discovered_in_reverse=discovered_in_reverse,
        reactive_xyz_indices=reactive_xyz_indices,
        anchors=anchors, constraints=constraints,
        r_xyz_na=r_xyz_na, op_xyz_na=op_xyz_na,
    )

def _strategy_ring_scission(ctx: _PathContext) -> _StrategyResult:
    """Build TS for ring-scission reactions discovered in reverse.

    Folds the reactant chain into a ring via torsion rotation, then
    stretches the breaking bond.  Dominant: halts the pipeline on success.
    """
    if not (ctx.discovered_in_reverse and ctx.bb and not ctx.fb):
        return _StrategyResult()
    rc = ring_closure_xyz(ctx.r_xyz, ctx.r_mol, forming_bond=ctx.bb[0])
    if rc is None:
        return _StrategyResult()
    ts = _build_ring_scission_ts(rc, ctx.bb, ctx.weight)
    if ts is not None and not colliding_atoms(ts):
        logger.debug(f'Linear ({ctx.label}): used ring-scission builder.')
        return _StrategyResult(guesses=[ts], halt=True)
    return _StrategyResult()


def _strategy_ring_closure(ctx: _PathContext) -> _StrategyResult:
    """Build TS via torsion-driven ring closure for forming bonds.

    For each forming bond whose atoms are far apart or connected by a long
    molecular-graph path, rotate torsions to bring them close.  For non-H
    atom migrations (e.g. halogen shift), also reposition the migrating atom
    symmetrically.  Dominant: halts the pipeline on success.
    """
    if not ctx.fb:
        return _StrategyResult()
    guesses = []
    used = False
    for bond_pair in ctx.fb:
        r_coords = np.array(ctx.r_xyz_na['coords'] if ctx.r_xyz_na else ctx.r_xyz['coords'], dtype=float)
        site_dist = float(np.linalg.norm(r_coords[bond_pair[0]] - r_coords[bond_pair[1]]))
        both_h = (ctx.r_xyz['symbols'][bond_pair[0]] == 'H'
                  or ctx.r_xyz['symbols'][bond_pair[1]] == 'H')
        path_len = get_path_length(ctx.r_mol, bond_pair[0], bond_pair[1])
        use_rc = (site_dist > RING_CLOSURE_THRESHOLD
                  or path_has_cumulated_bonds(ctx.r_mol, bond_pair)
                  or (not both_h and path_len is not None and path_len >= 3
                      and site_dist > 3.0))
        if not use_rc:
            continue
        if abs(ctx.weight - 0.5) > 0.01:
            continue
        src_xyz = ctx.r_xyz_na if ctx.r_xyz_na is not None else ctx.r_xyz
        rc_xyz = ring_closure_xyz(src_xyz, ctx.r_mol, forming_bond=bond_pair)
        if rc_xyz is None:
            continue
        # Reposition migrating atom only for true atom migration:
        # the atom must be non-H, monovalent in the reactant (F, Cl, Br —
        # exactly 1 bond), and present in both a breaking and forming bond.
        # Carbon atoms forming ring closures are never repositioned.
        mig_idx = bond_pair[0] if ctx.r_xyz['symbols'][bond_pair[0]] != 'H' else bond_pair[1]
        atom_in_bb = any(mig_idx in b for b in ctx.bb)
        n_bonds = len(ctx.r_mol.atoms[mig_idx].bonds)
        if ctx.r_xyz['symbols'][mig_idx] != 'H' and atom_in_bb and n_bonds == 1:
            acc_idx = bond_pair[1] if mig_idx == bond_pair[0] else bond_pair[0]
            atom_to_idx = {a: idx for idx, a in enumerate(ctx.r_mol.atoms)}
            don_idx = None
            for nbr in ctx.r_mol.atoms[mig_idx].bonds:
                ni = atom_to_idx[nbr]
                if ni != acc_idx and ctx.r_mol.atoms[ni].symbol != 'H':
                    don_idx = ni
                    break
            if don_idx is not None:
                rc_coords = np.array(rc_xyz['coords'], dtype=float)
                rc_xyz = _reposition_migrating_atom(
                    rc_xyz, rc_coords, mig_idx, don_idx, acc_idx)
        # Postprocess and validate.
        rc_xyz, migrating_hs = postprocess_ts_guess(
            rc_xyz, ctx.r_mol, list(ctx.fb), list(ctx.bb),
            family=ctx.family, r_label_map=ctx.r_label_map)
        is_valid, _ = validate_ts_guess(
            rc_xyz, migrating_hs, list(ctx.fb), ctx.r_mol,
            label=f'{ctx.label}, ring-closure', family=ctx.family)
        if is_valid:
            guesses.append(rc_xyz)
            used = True
    return _StrategyResult(guesses=guesses, halt=used)


def _strategy_zmat_interpolation(ctx: _PathContext) -> _StrategyResult:
    """Build TS by Z-matrix chimera interpolation (Type R and Type P).

    Builds two Z-matrix branches: one anchored on the reactant topology,
    one on the product topology.  Always runs (no early exit).
    """
    forming_bonds_list = list(ctx.fb)
    breaking_bonds_list = list(ctx.bb)
    guesses = []

    # Type R: reactant-topology Z-matrix chimera
    ts_r = generate_zmat_branch(
        anchor_xyz=ctx.r_xyz_na, anchor_mol=ctx.r_mol, target_xyz=ctx.op_xyz_na,
        weight=ctx.weight, reactive_xyz_indices=ctx.reactive_xyz_indices,
        anchors=ctx.anchors, constraints=ctx.constraints, r_mol=ctx.r_mol,
        forming_bonds=forming_bonds_list, breaking_bonds=breaking_bonds_list,
        label=f'{ctx.label}, type=R', family=ctx.family, r_label_map=ctx.r_label_map)
    if ts_r is not None:
        guesses.append(ts_r)

    # Type P: product-topology Z-matrix chimera
    ts_p = generate_zmat_branch(
        anchor_xyz=ctx.op_xyz_na, anchor_mol=ctx.mapped_p_mol, target_xyz=ctx.r_xyz_na,
        weight=1.0 - ctx.weight, reactive_xyz_indices=ctx.reactive_xyz_indices,
        anchors=ctx.anchors, constraints=ctx.constraints, r_mol=ctx.r_mol,
        forming_bonds=forming_bonds_list, breaking_bonds=breaking_bonds_list,
        label=f'{ctx.label}, type=P', family=ctx.family, r_label_map=ctx.r_label_map)
    if ts_p is not None:
        guesses.append(ts_p)

    # 4-center interchange fallback: when both Z-mat branches fail.
    if ts_r is None and ts_p is None:
        ts_4c = build_4center_interchange_ts(
            r_xyz=ctx.r_xyz, r_mol=ctx.r_mol,
            bb=breaking_bonds_list, fb=forming_bonds_list,
            weight=ctx.weight, label=ctx.label)
        if ts_4c is not None:
            guesses.append(ts_4c)
            logger.debug(f'Linear ({ctx.label}): used 4-center interchange builder.')

    return _StrategyResult(guesses=guesses, halt=False)


def _strategy_direct_contraction(ctx: _PathContext) -> _StrategyResult:
    """Build TS by contracting short forming bonds from the reactant geometry.

    When forming-bond atoms are already close (< 3.0 Å) and there are no
    breaking bonds, the TS is essentially the reactant with the terminal
    atom moved partway toward its partner.  This preserves the existing
    ring/backbone and avoids Z-mat interpolation artifacts (bivalent H's).

    Fires before Z-mat interpolation.  Dominant: halts the pipeline on
    success to prevent the Z-mat path from producing artifacts.
    """
    if ctx.bb or not ctx.fb:
        return _StrategyResult()
    guesses = []
    for bond_pair in ctx.fb:
        r_coords_arr = np.array(ctx.r_xyz['coords'], dtype=float)
        site_dist = float(np.linalg.norm(r_coords_arr[bond_pair[0]] - r_coords_arr[bond_pair[1]]))
        if site_dist > 3.0 or site_dist < 1e-6:
            continue
        # Identify the terminal atom (fewer heavy neighbors) to move.
        atom_to_idx = {a: idx for idx, a in enumerate(ctx.r_mol.atoms)}
        n_heavy_0 = sum(1 for nbr in ctx.r_mol.atoms[bond_pair[0]].bonds
                        if ctx.r_mol.atoms[atom_to_idx[nbr]].symbol != 'H')
        n_heavy_1 = sum(1 for nbr in ctx.r_mol.atoms[bond_pair[1]].bonds
                        if ctx.r_mol.atoms[atom_to_idx[nbr]].symbol != 'H')
        if n_heavy_0 <= n_heavy_1:
            mover, target = bond_pair[0], bond_pair[1]
        else:
            mover, target = bond_pair[1], bond_pair[0]
        # Target distance: ~1.5× the single bond length.
        sbl = get_single_bond_length(
            ctx.r_xyz['symbols'][mover], ctx.r_xyz['symbols'][target]) or 1.5
        d_target = sbl * 1.35
        if d_target >= site_dist:
            continue  # already close enough
        # Move only the mover atom and its H-only substituents.
        # Don't move backbone/ring neighbors — they should stay in place.
        vec = r_coords_arr[target] - r_coords_arr[mover]
        direction = vec / site_dist
        shift = direction * (site_dist - d_target) * ctx.weight / 0.5
        frag: Set[int] = {mover}
        for nbr in ctx.r_mol.atoms[mover].bonds:
            ni = atom_to_idx[nbr]
            if ctx.r_mol.atoms[ni].symbol == 'H':
                frag.add(ni)
        coords_dc = r_coords_arr.copy()
        for k in frag:
            coords_dc[k] += shift
        # Breathing compensation: if the contraction compressed any mover–neighbor
        # bond below single-bond length, push the neighbor (+ its H's) outward.
        for nbr in ctx.r_mol.atoms[mover].bonds:
            ni = atom_to_idx[nbr]
            if ni == target or ni in frag:
                continue
            new_dist = float(np.linalg.norm(coords_dc[mover] - coords_dc[ni]))
            sbl_mn = get_single_bond_length(
                ctx.r_xyz['symbols'][mover], ctx.r_xyz['symbols'][ni]) or 1.5
            ts_floor = sbl_mn * 1.10  # TS bonds are ~10% longer than equilibrium
            if new_dist < ts_floor:
                vec_mn = coords_dc[ni] - coords_dc[mover]
                dir_mn = vec_mn / new_dist
                push_dist = ts_floor - new_dist
                nbr_frag: Set[int] = {ni}
                for nbr2 in ctx.r_mol.atoms[ni].bonds:
                    ni2 = atom_to_idx[nbr2]
                    if ctx.r_xyz['symbols'][ni2] == 'H':
                        nbr_frag.add(ni2)
                for k in nbr_frag:
                    coords_dc[k] += dir_mn * push_dist
        ts_dc = {'symbols': ctx.r_xyz['symbols'], 'isotopes': ctx.r_xyz['isotopes'],
                  'coords': tuple(tuple(row) for row in coords_dc)}
        if not colliding_atoms(ts_dc):
            guesses.append(ts_dc)
            logger.debug(f'Linear ({ctx.label}): used direct-contraction builder '
                         f'(mover={mover}, target={target}, d={site_dist:.2f}→{site_dist-float(np.linalg.norm(shift)):.2f}).')
    return _StrategyResult(guesses=guesses, halt=False)


def _strategy_3center_shift(ctx: _PathContext) -> _StrategyResult:
    """Build TS for 3-center atom shifts (e.g. 1,2_shiftC).

    When breaking and forming bonds share a non-H atom (the pivot),
    form a 3-membered ring TS.

    Two cases:
    - **Pivot has ≤2 heavy neighbors** (terminal/chain): reposition the
      pivot between the other two atoms (e.g. halogen migration).
    - **Pivot has ≥3 heavy neighbors** (ring junction): keep the pivot
      in place and reposition the migrating group (bb_other) toward
      the forming-bond partner (fb_other).  This preserves the ring.

    Supplementary: never halts.
    """
    if len(ctx.bb) != 1 or len(ctx.fb) != 1:
        return _StrategyResult()
    bb_atoms = {a for bond in ctx.bb for a in bond}
    fb_atoms = {a for bond in ctx.fb for a in bond}
    shared = bb_atoms & fb_atoms
    if len(shared) != 1:
        return _StrategyResult()
    pivot = next(iter(shared))
    if ctx.r_xyz['symbols'][pivot] == 'H':
        return _StrategyResult()
    bb_other = ctx.bb[0][0] if ctx.bb[0][1] == pivot else ctx.bb[0][1]
    fb_other = ctx.fb[0][0] if ctx.fb[0][1] == pivot else ctx.fb[0][1]

    # Count heavy-atom neighbors of the pivot.
    atom_to_idx = {a: idx for idx, a in enumerate(ctx.r_mol.atoms)}
    n_heavy_pivot = sum(1 for nbr in ctx.r_mol.atoms[pivot].bonds
                        if ctx.r_mol.atoms[atom_to_idx[nbr]].symbol != 'H')

    if n_heavy_pivot >= 3:
        # Pivot is a ring/junction atom — keep it fixed.
        # Stretch the bb_other-pivot bond: move bb_other (and its H's)
        # AWAY from pivot along the pivot→bb_other direction.  This
        # keeps the pivot's ring intact while partially detaching the
        # migrating group, which stays bonded to fb_other.
        ts_3c = _build_ring_scission_ts(
            ctx.r_xyz, breaking_bonds=[(bb_other, pivot)],
            weight=ctx.weight, stretch_factor=1.6)
    else:
        # Pivot is terminal/chain — reposition it between the other two.
        ts_3c = _reposition_migrating_atom(
            dict(ctx.r_xyz), np.array(ctx.r_xyz['coords'], dtype=float),
            mig_idx=pivot, don_idx=bb_other, acc_idx=fb_other)

    if ts_3c is not None and not colliding_atoms(ts_3c):
        logger.debug(f'Linear ({ctx.label}): used 3-center shift builder '
                     f'(pivot={pivot}, n_heavy={n_heavy_pivot}).')
        return _StrategyResult(guesses=[ts_3c], halt=False)
    return _StrategyResult()


def interpolate_isomerization(rxn: 'ARCReaction',
                              weight: float = 0.5,
                              existing_xyzs: Optional[List[dict]] = None,
                              ) -> Optional[List[dict]]:
    """
    Search for a TS of an A <=> B (1 to 1) isomerization reaction by interpolating internal coords.

    For each reaction-family product_dict (i.e., each distinct reaction path) two
    structurally independent TS guesses are generated as Z-matrix chimeras:

    * **Type R** (reactant-topology): A Z-matrix is built from the reactant geometry
      using :func:`~arc.species.zmat.xyz_to_zmat`.  The atom-mapped product geometry
      is projected onto that topology with :func:`~arc.species.zmat.update_zmat_by_xyz`,
      and the two internal-coordinate sets are blended at ``weight`` via
      :func:`average_zmat_params`.  Only reactive coordinates (those that reference
      at least one atom participating in a forming or breaking bond) are interpolated;
      spectator coordinates are preserved from the reactant anchor to avoid washing out
      good TS geometry.

    * **Type P** (product-topology): Symmetric to Type R.  A Z-matrix is built from the
      atom-mapped product geometry (in reactant atom ordering), the reactant geometry is
      projected onto that topology, and the blend uses weight ``1 - weight``.  This
      explores a genuinely product-like region of the TS surface.

    The anchor atoms for both Z-matrices are selected by :func:`~arc.species.zmat.find_smart_anchors`,
    which prefers spectator atoms adjacent to the reactive core so that the coordinate
    frame is stable across the interpolation.

    Generated geometries pass through a shared postprocessing pipeline
    (:func:`postprocess_ts_guess`) and validation pipeline (:func:`validate_ts_guess`)
    before being added to the output.  Near-identical surviving guesses are deduplicated.

    When species E0 or e_elect energies are pre-populated, :func:`get_weight_grid`
    automatically biases the weight toward the Hammond/Marcus prediction; no extra
    configuration is needed.

    The ordered-product XYZ is obtained via a per-path atom map
    (``map_rxn(product_dict_index_to_try=i)``).  If that call returns ``None``,
    the reaction's global atom map (``rxn.get_products_xyz``) is used as a fallback
    (degraded mode) and logged at DEBUG level.  If ``map_rxn`` raises an exception
    the path is skipped entirely.

    Args:
        rxn (ARCReaction): The reaction to process.  Must have exactly one reactant
            species and exactly one product species.
        weight (float): Interpolation weight on a 0 (reactant) → 1 (product) scale.
        existing_xyzs (List[dict], optional): Previously generated XYZ guesses
            (e.g. from earlier weight iterations).  New guesses that are
            near-identical to any entry in this list are suppressed, preventing
            duplicate TS guesses across multiple calls with different weights.

    Returns:
        Optional[List[dict]]: Validated, deduplicated XYZ coordinate guesses in reactant atom ordering.
    """
    if not (0.0 <= weight <= 1.0):
        return None
    if len(rxn.r_species) != 1 or len(rxn.p_species) != 1:
        logger.debug(f'Linear (rxn={rxn.label}): skipping — requires exactly 1 reactant and '
                     f'1 product species (got {len(rxn.r_species)} and {len(rxn.p_species)}).')
        return []

    ts_xyzs: List[dict] = list()
    r_xyz = rxn.r_species[0].get_xyz()
    r_mol = rxn.r_species[0].mol
    # Defer the expensive global atom map computation: only compute when
    # a per-path map_rxn returns None (fallback) or at the end for the
    # trivial-map condition.  Use a sentinel to distinguish "not computed" from None.
    _SENTINEL = object()
    initial_atom_map = _SENTINEL
    op_xyz_fallback: Optional[dict] = None

    def _ensure_global_atom_map():
        """Lazily compute the global atom map and ordered-product fallback XYZ."""
        nonlocal initial_atom_map, op_xyz_fallback
        if initial_atom_map is not _SENTINEL:
            return
        initial_atom_map = rxn.atom_map
        try:
            op_xyz_fallback = rxn.get_products_xyz(return_format='dict') if initial_atom_map is not None else None
        except Exception as e:
            logger.debug(f'Linear (rxn={rxn.label}): get_products_xyz raised {type(e).__name__}: {e}; fallback unavailable.')
            op_xyz_fallback = None

    # Pre-compute SSSR on a copy to avoid setting ring flags on r_mol (which
    # would interfere with subsequent RMG template matching / atom mapping).
    _r_mol_copy = r_mol.copy(deep=True)
    _ring_sets: List[Set[int]] = [
        set(_r_mol_copy.atoms.index(a) for a in ring)
        for ring in _r_mol_copy.get_smallest_set_of_smallest_rings()
    ]
    # Stash fallback bond lists for end-of-function post-processing.
    _fallback_fb: Optional[List[Tuple[int, int]]] = None
    _fallback_bb: Optional[List[Tuple[int, int]]] = None
    _fallback_changed: Optional[List[Tuple[int, int]]] = None

    seen_bond_signatures: Set[tuple] = set()
    for i, product_dict in enumerate(rxn.product_dicts):
        r_label_dict = product_dict['r_label_map']
        path_family = product_dict.get('family') or rxn.family
        if r_label_dict is None:
            logger.debug(f'Linear (rxn={rxn.label}, path={i}): r_label_dict is None; skipping path.')
            continue
        bb, fb = rxn.get_expected_changing_bonds(r_label_dict=r_label_dict)
        if bb is None or fb is None:
            logger.debug(f'Linear (rxn={rxn.label}, path={i}): get_expected_changing_bonds returned None; skipping path.')
            continue
        # Deduplicate paths whose breaking/forming bonds are identical:
        # they differ only in non-reactive label assignments and produce
        # near-identical TS guesses, while each map_rxn call is expensive.
        bond_sig = (tuple(sorted(tuple(sorted(b)) for b in bb)),
                    tuple(sorted(tuple(sorted(b)) for b in fb)))
        if bond_sig in seen_bond_signatures:
            logger.debug(f'Linear (rxn={rxn.label}, path={i}): duplicate bond signature {bond_sig}; skipping path.')
            continue
        seen_bond_signatures.add(bond_sig)

        # Per-path atom map: different paths may involve different equivalent atoms
        # (e.g., distinct H's in intra_H_migration), so we must not reuse the
        # global rxn.atom_map for all paths.
        try:
            atom_map = map_rxn(rxn=rxn, product_dict_index_to_try=i)
        except Exception as e:
            logger.debug(f'Linear (rxn={rxn.label}, path={i}): map_rxn raised {type(e).__name__}: {e}; skipping path.')
            continue
        if atom_map is not None:
            if initial_atom_map is _SENTINEL:
                initial_atom_map = atom_map
            op_xyz = order_xyz_by_atom_map(xyz=rxn.p_species[0].get_xyz(), atom_map=atom_map)
            # Reorder p_mol to match reactant atom indexing so the Type-P Z-matrix is
            # built from the product bond topology, not the reactant's.
            try:
                mapped_p_mol = order_mol_by_atom_map(rxn.p_species[0].mol, atom_map)
            except Exception as e:
                logger.debug(f'Linear (rxn={rxn.label}, path={i}): order_mol_by_atom_map failed '
                             f'({e}); falling back to r_mol for Type-P.')
                mapped_p_mol = r_mol
        else:
            # Per-path map failed; lazily compute the global atom map for fallback.
            _ensure_global_atom_map()
            if op_xyz_fallback is not None:
                op_xyz = op_xyz_fallback
                # Global fallback: use rxn.atom_map if available to reorder p_mol,
                # otherwise use r_mol (degraded connectivity for Type-P).
                if rxn.atom_map is not None:
                    try:
                        mapped_p_mol = order_mol_by_atom_map(rxn.p_species[0].mol, rxn.atom_map)
                    except Exception:
                        mapped_p_mol = r_mol
                else:
                    mapped_p_mol = r_mol
                logger.debug(f'Linear (rxn={rxn.label}, path={i}): per-path map_rxn returned None; '
                             f'using global fallback atom map (degraded mode).')
            else:
                logger.debug(f'Linear (rxn={rxn.label}, path={i}): no atom map available, skipping path.')
                continue

        # Sanity check: the mapped product XYZ must have the same atoms as the reactant.
        if op_xyz['symbols'] != r_xyz['symbols']:
            logger.warning(f'Linear (rxn={rxn.label}, path={i}): atom-mapped product symbols '
                           f'{op_xyz["symbols"]} do not match reactant symbols {r_xyz["symbols"]}; '
                           f'skipping path.')
            continue

        # --- Build the path context and run the strategy pipeline ---
        ctx = _build_path_context(
            r_xyz=r_xyz, r_mol=r_mol, op_xyz=op_xyz,
            mapped_p_mol=mapped_p_mol, bb=list(bb), fb=list(fb),
            family=path_family, r_label_map=r_label_dict,
            weight=weight, ring_sets=_ring_sets,
            label=f'rxn={rxn.label}, path={i}',
            discovered_in_reverse=bool(product_dict.get('discovered_in_reverse')),
        )

        # Strategy pipeline: dominant strategies halt on success.
        for strategy in [_strategy_ring_scission,
                         _strategy_direct_contraction,
                         _strategy_ring_closure,
                         _strategy_zmat_interpolation,
                         _strategy_3center_shift]:
            result = strategy(ctx)
            if result.guesses:
                ts_xyzs.extend(result.guesses)
            if result.halt:
                break

    # Direct-contraction supplement: when the per-path pipeline produced no
    # guesses (e.g. atom map failed) but the family recipe has fb-only with
    # short forming bonds, contract them directly from the reactant geometry.
    if not ts_xyzs:
        for product_dict in rxn.product_dicts:
            rl = product_dict.get('r_label_map')
            if rl is None:
                continue
            bb_dc, fb_dc = rxn.get_expected_changing_bonds(r_label_dict=rl)
            if bb_dc or not fb_dc:
                continue
            for bond_pair_dc in fb_dc:
                r_coords_dc = np.array(r_xyz['coords'], dtype=float)
                site_d = float(np.linalg.norm(r_coords_dc[bond_pair_dc[0]] - r_coords_dc[bond_pair_dc[1]]))
                if site_d > 3.0 or site_d < 1e-6:
                    continue
                atom_to_idx_dc = {a: idx for idx, a in enumerate(r_mol.atoms)}
                nh0 = sum(1 for nbr in r_mol.atoms[bond_pair_dc[0]].bonds
                          if r_mol.atoms[atom_to_idx_dc[nbr]].symbol != 'H')
                nh1 = sum(1 for nbr in r_mol.atoms[bond_pair_dc[1]].bonds
                          if r_mol.atoms[atom_to_idx_dc[nbr]].symbol != 'H')
                mover = bond_pair_dc[0] if nh0 <= nh1 else bond_pair_dc[1]
                target_dc = bond_pair_dc[1] if mover == bond_pair_dc[0] else bond_pair_dc[0]
                sbl_dc = get_single_bond_length(
                    r_xyz['symbols'][mover], r_xyz['symbols'][target_dc]) or 1.5
                d_tgt = sbl_dc * 1.35
                if d_tgt >= site_d:
                    continue
                vec_dc = r_coords_dc[target_dc] - r_coords_dc[mover]
                shift_dc = (vec_dc / site_d) * (site_d - d_tgt) * weight / 0.5
                # Only move the mover and its H-only substituents.
                frag_dc: Set[int] = {mover}
                for nbr_dc in r_mol.atoms[mover].bonds:
                    ni_dc = atom_to_idx_dc[nbr_dc]
                    if r_mol.atoms[ni_dc].symbol == 'H':
                        frag_dc.add(ni_dc)
                for k in frag_dc:
                    r_coords_dc[k] += shift_dc
                # Breathing compensation: push compressed neighbors outward.
                for nbr_bc in r_mol.atoms[mover].bonds:
                    ni_bc = atom_to_idx_dc[nbr_bc]
                    if ni_bc == target_dc or ni_bc in frag_dc:
                        continue
                    new_d = float(np.linalg.norm(r_coords_dc[mover] - r_coords_dc[ni_bc]))
                    sbl_bc = get_single_bond_length(
                        r_xyz['symbols'][mover], r_xyz['symbols'][ni_bc]) or 1.5
                    ts_floor_bc = sbl_bc * 1.10
                    if new_d < ts_floor_bc:
                        vec_bc = r_coords_dc[ni_bc] - r_coords_dc[mover]
                        dir_bc = vec_bc / new_d
                        push_d = ts_floor_bc - new_d
                        bc_frag: Set[int] = {ni_bc}
                        for nbr2_bc in r_mol.atoms[ni_bc].bonds:
                            ni2_bc = atom_to_idx_dc[nbr2_bc]
                            if r_xyz['symbols'][ni2_bc] == 'H':
                                bc_frag.add(ni2_bc)
                        for k in bc_frag:
                            r_coords_dc[k] += dir_bc * push_d
                ts_dc = {'symbols': r_xyz['symbols'], 'isotopes': r_xyz['isotopes'],
                          'coords': tuple(tuple(row) for row in r_coords_dc)}
                if not colliding_atoms(ts_dc):
                    ts_xyzs.append(ts_dc)
                    logger.debug(f'Linear (rxn={rxn.label}): direct-contraction from recipe '
                                 f'(mover={mover}, target={target_dc}).')
            if ts_xyzs:
                break

    # Trivial atom-map fallback: when no product_dicts could be determined (e.g., the
    # reaction family is unknown) fall back to an identity atom map and determine
    # breaking/forming bonds via the Reaction's bond-comparison methods.
    # Also run when the initial atom map was None — any non-trivial guesses
    # produced in that case used degraded-mode atom maps and may be unreliable.
    _ensure_global_atom_map()
    if not ts_xyzs or initial_atom_map is None:
        n_atoms = len(r_xyz['symbols'])
        p_xyz = rxn.p_species[0].get_xyz()
        p_mol = rxn.p_species[0].mol

        bb_map_from_reorder = None
        if r_xyz['symbols'] != p_xyz['symbols']:
            # Symbols differ (different atom ordering).  Try backbone_atom_map
            # which now handles ring-forming reactions (P has one extra edge).
            bb_map_from_reorder = backbone_atom_map(r_mol, p_mol)
            if bb_map_from_reorder is not None:
                p_xyz = order_xyz_by_atom_map(xyz=p_xyz, atom_map=bb_map_from_reorder)
                try:
                    p_mol = order_mol_by_atom_map(p_mol, bb_map_from_reorder)
                except Exception:
                    pass
                if rxn.atom_map is None:
                    rxn.atom_map = bb_map_from_reorder
                logger.debug(f'Linear (rxn={rxn.label}): trivial-map fallback — '
                             f'reordered P atoms via backbone atom map.')
            else:
                logger.debug(f'Linear (rxn={rxn.label}): trivial-map fallback skipped — '
                             f'R and P symbols differ and backbone matching failed.')
        if r_xyz['symbols'] == p_xyz['symbols']:
            # Try backbone graph matching first — it correctly handles
            # reactions where RMG family detection fails (e.g. singlet
            # biradicals) by matching heavy-atom connectivity ignoring
            # bond orders.  Fall back to the identity atom map only when
            # backbone matching fails.
            original_atom_map = rxn.atom_map
            # Try backbone graph matching when no family was detected AND
            # no atom map was set by the constructor — it correctly handles
            # reactions where RMG family detection fails (e.g. singlet
            # biradicals) by matching heavy-atom connectivity ignoring
            # bond orders.  Fall back to the identity atom map only when
            # backbone matching fails.
            bb_map = bb_map_from_reorder or (backbone_atom_map(r_mol, p_mol) if rxn.atom_map is None else None)
            if bb_map is not None and bb_map_from_reorder is None:
                rxn.atom_map = bb_map
                p_xyz = order_xyz_by_atom_map(xyz=p_xyz, atom_map=bb_map)
                try:
                    p_mol = order_mol_by_atom_map(p_mol, bb_map)
                except Exception:
                    pass  # keep original p_mol
                logger.debug(f'Linear (rxn={rxn.label}): trivial-map fallback — '
                             f'using backbone atom map.')
            elif rxn.atom_map is None:
                rxn.atom_map = list(range(n_atoms))

            try:
                fb, bb = rxn.get_formed_and_broken_bonds()
            except Exception as e:
                logger.debug(f'Linear (rxn={rxn.label}): trivial-map fallback — '
                             f'get_formed_and_broken_bonds raised {type(e).__name__}: {e}.')
                fb, bb = [], []

            try:
                changed = rxn.get_changed_bonds()
            except Exception as e:
                logger.debug(f'Linear (rxn={rxn.label}): trivial-map fallback — '
                             f'get_changed_bonds raised {type(e).__name__}: {e}.')
                changed = []

            # Restore original atom_map to avoid side effects.
            rxn.atom_map = original_atom_map
            # Stash for end-of-function RH_Add motif post-processing.
            _fallback_fb = list(fb)
            _fallback_bb = list(bb)
            _fallback_changed = list(changed)

            if bb_map is not None:
                # Backbone map produces correct atom correspondences, so
                # H-bond changes represent real reactive bonds (e.g. H
                # migration) — keep them.  Also include bond-order changes
                # (e.g. C=C → C-C) as forming bonds so the multiple bond
                # gets stretched in the TS.
                fb = fb + changed
            else:
                # The identity atom map can misassign H atoms (an H bonded
                # to C0 in R may be bonded to C1 in P even though both H's
                # sit at atom index 6).  Spurious H-bond changes inflate
                # the reactive set to nearly all atoms, drowning the real
                # skeletal rearrangement in noise.  Filter to heavy-atom
                # pairs only; H positions are handled by the spectator-
                # preservation in average_zmat_params (they inherit R-side
                # Z-mat values).
                symbols = r_xyz['symbols']
                fb = [(i, j) for i, j in fb
                      if symbols[i] != 'H' and symbols[j] != 'H']
                bb = [(i, j) for i, j in bb
                      if symbols[i] != 'H' and symbols[j] != 'H']

            if not bb and not fb:
                logger.debug(f'Linear (rxn={rxn.label}): trivial-map fallback — '
                             f'no changing bonds found between R and P.')
            else:
                logger.debug(f'Linear (rxn={rxn.label}): trivial-map fallback — '
                             f'breaking={bb}, forming={fb}.')
                reactive_xyz_indices: Set[int] = set()
                for bond in bb + fb:
                    reactive_xyz_indices.update(bond)
                # Also mark direct neighbors of reactive atoms: they shift
                # when a reactive atom moves and should not trigger backbone
                # drift rejection (e.g. F's on a C that participates in a
                # halogen migration).
                atom_to_idx_fb = {a: idx for idx, a in enumerate(r_mol.atoms)}
                for idx in list(reactive_xyz_indices):
                    for nbr in r_mol.atoms[idx].bonds:
                        reactive_xyz_indices.add(atom_to_idx_fb[nbr])
                changing_bonds_fb = {(min(a, b), max(a, b)) for a, b in bb + fb}
                ring_bonds_fb, reactive_xyz_indices = _get_ring_preservation_bonds(
                    r_mol, reactive_xyz_indices, changing_bonds_fb, _ring_sets)

                anchors = find_smart_anchors(r_mol, breaking_bonds=list(bb), forming_bonds=list(fb))
                constraints = get_r_constraints(expected_breaking_bonds=list(bb),
                                                expected_forming_bonds=list(fb))
                if ring_bonds_fb:
                    reactive_bond_atoms_fb = {a for bond in bb + fb for a in bond}
                    constrained_atoms_fb = reactive_bond_atoms_fb | {a for pair in constraints.get('R_atom', []) for a in pair}
                    added_fb = False
                    for rb in ring_bonds_fb:
                        if not added_fb and rb[0] not in constrained_atoms_fb and rb[1] not in constrained_atoms_fb:
                            constraints['R_atom'].append(rb)
                            constrained_atoms_fb.add(rb[0])
                            constrained_atoms_fb.add(rb[1])
                            added_fb = True
                        else:
                            reactive_xyz_indices.discard(rb[0])

                # Skip near-attack for backbone-map cases where the
                # reaction occurs within a pre-existing ring — rotating
                # the molecule to bring the forming-bond atoms closer
                # just distorts the backbone.  For ring-FORMING reactions
                # (open chain → ring), NAC is essential to fold the chain.
                if bb_map is not None and bb_map_from_reorder is None:
                    r_xyz_na = r_xyz
                    op_xyz_na = p_xyz
                else:
                    r_xyz_na = get_near_attack_xyz(r_xyz, r_mol, bonds=list(fb))
                    op_xyz_na = get_near_attack_xyz(p_xyz, p_mol, bonds=list(bb))

                # When the backbone map detects an H migration but RMG
                # couldn't assign a family, override the family so the
                # H-migration postprocessor fires (it places the migrating
                # H via triangulation, which Z-mat interpolation can't do
                # when the forming bond isn't a direct Z-mat variable).
                effective_family = rxn.family
                if bb_map is not None and effective_family is None:
                    symbols = r_xyz['symbols']
                    has_h_transfer = any(
                        symbols[b[0]] == 'H' or symbols[b[1]] == 'H'
                        for b in fb)
                    if has_h_transfer:
                        effective_family = 'intra_H_migration'

                # Ring-closure fast path (mirrors the per-product_dict check):
                # when a forming-bond distance exceeds the threshold, use the
                # dedicated ring-closure algorithm instead of Z-matrix interpolation.
                # Use the original r_xyz (not the near-attack r_xyz_na) for both
                # the distance check and ring closure: the near-attack transform
                # can distort the geometry before ring closure, causing artifacts.
                # Ring closure is weight-independent (purely geometric).
                # Only produce it at the canonical weight (0.5) to avoid
                # duplicating identical guesses across weight iterations.
                # When a forming-bond distance exceeds the threshold, also skip
                # the zmat branch at non-0.5 weights (zmat interpolation is
                # unreliable for such distant sites).
                used_ring_closure = False
                needs_ring_closure = False
                for bond_pair in fb:
                    r_coords = np.array(r_xyz['coords'], dtype=float)
                    site_dist = float(np.linalg.norm(
                        r_coords[bond_pair[0]] - r_coords[bond_pair[1]]))
                    # Use ring closure when the forming-bond distance exceeds
                    # the threshold, when cumulated bonds are present, or when
                    # the forming-bond atoms are connected by a path in the
                    # reactant (intramolecular ring-forming TS).
                    both_h_fb = (r_xyz['symbols'][bond_pair[0]] == 'H'
                                 or r_xyz['symbols'][bond_pair[1]] == 'H')
                    path_len = get_path_length(r_mol, bond_pair[0], bond_pair[1])
                    use_rc = (site_dist > RING_CLOSURE_THRESHOLD
                              or path_has_cumulated_bonds(r_mol, bond_pair)
                              or (not both_h_fb and path_len is not None and path_len >= 3
                                  and site_dist > 3.0))
                    if use_rc:
                        needs_ring_closure = True
                        if abs(weight - 0.5) <= 0.01:
                            rc_xyz = ring_closure_xyz(r_xyz, r_mol,
                                                       forming_bond=bond_pair)
                            if rc_xyz is not None:
                                # Reposition only for true monovalent atom migration
                                # (F, Cl — 1 bond in reactant), not ring closure.
                                mig_idx = bond_pair[0] if r_xyz['symbols'][bond_pair[0]] != 'H' else bond_pair[1]
                                atom_in_bb = any(mig_idx in b for b in bb)
                                n_bonds_fb = len(r_mol.atoms[mig_idx].bonds)
                                if r_xyz['symbols'][mig_idx] != 'H' and atom_in_bb and n_bonds_fb == 1:
                                    acc_idx = bond_pair[1] if mig_idx == bond_pair[0] else bond_pair[0]
                                    atom_to_idx_rc = {a: idx for idx, a in enumerate(r_mol.atoms)}
                                    don_idx = None
                                    for nbr in r_mol.atoms[mig_idx].bonds:
                                        ni = atom_to_idx_rc[nbr]
                                        if ni != acc_idx and r_mol.atoms[ni].symbol != 'H':
                                            don_idx = ni
                                            break
                                    if don_idx is not None:
                                        rc_coords = np.array(rc_xyz['coords'], dtype=float)
                                        rc_xyz = _reposition_migrating_atom(
                                            rc_xyz, rc_coords, mig_idx, don_idx, acc_idx)
                                # Try H-migration postprocessing first (places
                                # migrating H via triangulation), fall back to
                                # generic if H-migration validation rejects it.
                                import copy
                                for fam in ([effective_family, rxn.family]
                                            if effective_family != rxn.family else [rxn.family]):
                                    rc_try = copy.deepcopy(rc_xyz)
                                    rc_try, mhs = postprocess_ts_guess(
                                        rc_try, r_mol, list(fb), list(bb), family=fam)
                                    ok, _ = validate_ts_guess(
                                        rc_try, mhs, fb, r_mol,
                                        label=f'rxn={rxn.label}, trivial, ring-closure',
                                        family=fam)
                                    if ok:
                                        ts_xyzs.append(rc_try)
                                        used_ring_closure = True
                                        break

                # Direct contraction: when the forming bond is short (< 3.0 Å)
                # and there are no breaking bonds, just contract the forming
                # bond from the reactant geometry.  This avoids Z-mat artifacts.
                if not bb and fb:
                    for bond_pair_dc in fb:
                        r_coords_dc = np.array(r_xyz['coords'], dtype=float)
                        site_d = float(np.linalg.norm(r_coords_dc[bond_pair_dc[0]] - r_coords_dc[bond_pair_dc[1]]))
                        if site_d > 3.0 or site_d < 1e-6:
                            continue
                        atom_to_idx_dc = {a: idx for idx, a in enumerate(r_mol.atoms)}
                        nh0 = sum(1 for nbr in r_mol.atoms[bond_pair_dc[0]].bonds
                                  if r_mol.atoms[atom_to_idx_dc[nbr]].symbol != 'H')
                        nh1 = sum(1 for nbr in r_mol.atoms[bond_pair_dc[1]].bonds
                                  if r_mol.atoms[atom_to_idx_dc[nbr]].symbol != 'H')
                        mover = bond_pair_dc[0] if nh0 <= nh1 else bond_pair_dc[1]
                        target = bond_pair_dc[1] if mover == bond_pair_dc[0] else bond_pair_dc[0]
                        sbl_dc = get_single_bond_length(
                            r_xyz['symbols'][mover], r_xyz['symbols'][target]) or 1.5
                        d_tgt = sbl_dc * 1.35
                        if d_tgt >= site_d:
                            continue
                        vec_dc = r_coords_dc[target] - r_coords_dc[mover]
                        shift_dc = (vec_dc / site_d) * (site_d - d_tgt) * weight / 0.5
                        mol_adj_dc2: Dict[int, Set[int]] = {k: set() for k in range(n_atoms)}
                        for atom_dc in r_mol.atoms:
                            ii = atom_to_idx_dc[atom_dc]
                            for nbr_dc in atom_dc.bonds:
                                mol_adj_dc2[ii].add(atom_to_idx_dc[nbr_dc])
                        frag_dc: Set[int] = set()
                        q_dc = deque([mover])
                        while q_dc:
                            nd = q_dc.popleft()
                            if nd in frag_dc or nd == target:
                                continue
                            frag_dc.add(nd)
                            q_dc.extend(mol_adj_dc2[nd] - frag_dc - {target})
                        for k in frag_dc:
                            r_coords_dc[k] += shift_dc
                        ts_dc = {'symbols': r_xyz['symbols'], 'isotopes': r_xyz['isotopes'],
                                  'coords': tuple(tuple(row) for row in r_coords_dc)}
                        if not colliding_atoms(ts_dc):
                            ts_xyzs.append(ts_dc)
                            logger.debug(f'Linear (rxn={rxn.label}): trivial-fallback direct-contraction '
                                         f'(mover={mover}, target={target}).')

                if not used_ring_closure and not needs_ring_closure:
                    # With the backbone atom map the reactive set is small and
                    # accurate, so postprocessing (H distance fix, etc.) is safe
                    # and needed.  With the identity map the reactive set is
                    # inflated to nearly all atoms and postprocessing would
                    # perturb the geometry — skip it.
                    skip_pp = bb_map is None
                    ts_r = generate_zmat_branch(
                        anchor_xyz=r_xyz_na, anchor_mol=r_mol, target_xyz=op_xyz_na,
                        weight=weight, reactive_xyz_indices=reactive_xyz_indices,
                        anchors=anchors, constraints=constraints, r_mol=r_mol,
                        forming_bonds=list(fb), breaking_bonds=list(bb),
                        label=f'rxn={rxn.label}, trivial, w={weight}, type=R',
                        skip_postprocess=skip_pp, family=effective_family,
                        redistribute_ch2=bb_map is not None)
                    if ts_r is not None and not has_excessive_backbone_drift(
                            ts_r, r_xyz_na, max_mean_heavy_disp=3.0,
                            reactive_indices=reactive_xyz_indices):
                        ts_xyzs.append(ts_r)
                    elif ts_r is not None:
                        logger.debug(f'Linear (rxn={rxn.label}, trivial, w={weight}, type=R): '
                                     f'discarded — excessive backbone drift from anchor.')

                    ts_p = generate_zmat_branch(
                        anchor_xyz=op_xyz_na, anchor_mol=p_mol, target_xyz=r_xyz_na,
                        weight=1.0 - weight, reactive_xyz_indices=reactive_xyz_indices,
                        anchors=anchors, constraints=constraints, r_mol=r_mol,
                        forming_bonds=list(fb), breaking_bonds=list(bb),
                        label=f'rxn={rxn.label}, trivial, w={weight}, type=P',
                        skip_postprocess=skip_pp, family=effective_family,
                        redistribute_ch2=bb_map is not None)
                    if ts_p is not None and not has_excessive_backbone_drift(
                            ts_p, op_xyz_na, max_mean_heavy_disp=3.0,
                            reactive_indices=reactive_xyz_indices):
                        ts_xyzs.append(ts_p)
                    elif ts_p is not None:
                        logger.debug(f'Linear (rxn={rxn.label}, trivial, w={weight}, type=P): '
                                     f'discarded — excessive backbone drift from anchor.')

    # Collect all forming/breaking bonds across paths for post-processing.
    all_forming: List[Tuple[int, int]] = []
    all_breaking: List[Tuple[int, int]] = []
    changing_all: Set[Tuple[int, int]] = set()
    for pd in rxn.product_dicts:
        rl = pd.get('r_label_map')
        if rl:
            bb_i, fb_i = rxn.get_expected_changing_bonds(r_label_dict=rl)
            for b in list(bb_i or []) + list(fb_i or []):
                changing_all.add((min(b), max(b)))
            all_forming.extend(fb_i or [])
            all_breaking.extend(bb_i or [])
    reactive_all: Set[int] = set()
    for b in changing_all:
        reactive_all.update(b)

    # Deduplicate: collapse near-identical guesses from different paths, from
    # Type R ≈ Type P (common for symmetric reactions at weight=0.5), or from
    # weight-insensitive algorithms (e.g. ring-closure) across repeated calls.
    # Two-tier dedup:
    #   1. Exact match (almost_equal_coords) — catches identical guesses.
    #   2. Heavy-atom match — catches guesses that differ only in H placement
    #      (e.g. two chirality-preserving H-migration paths that produce the
    #      same backbone with H on opposite sides).
    prior: List[dict] = list(existing_xyzs or [])
    unique: List[dict] = []

    def _heavy_atoms_match(xyz1: dict, xyz2: dict, tol: float = 0.05) -> bool:
        """Return True if all heavy-atom coordinates match within *tol* Å."""
        c1 = np.array(xyz1['coords'], dtype=float)
        c2 = np.array(xyz2['coords'], dtype=float)
        for i, sym in enumerate(xyz1['symbols']):
            if sym != 'H' and float(np.linalg.norm(c1[i] - c2[i])) > tol:
                return False
        return True

    for xyz in ts_xyzs:
        if colliding_atoms(xyz):
            continue
        if any(almost_equal_coords(xyz, other) or _heavy_atoms_match(xyz, other)
               for other in unique + prior):
            continue
        unique.append(xyz)

    if unique:
        # Repair ring bonds broken by Z-matrix interpolation.
        if _ring_sets:
            unique = [_fix_broken_ring_bonds(xyz, r_mol, _ring_sets, reactive_all, changing_all)
                      for xyz in unique]
        # Reflect H atoms that project into forming-bond paths (blocking ring closure).
        if all_forming:
            unique = [_clear_forming_bond_path(xyz, r_mol, all_forming)
                      for xyz in unique]
        # Build the 4-membered ring TS motif for Intra_RH_Add patterns
        # (stretch X-H, stretch Y=Z, move H toward Y).
        if _fallback_fb is not None and _fallback_bb is not None and _fallback_changed is not None:
            unique = [_fix_rh_add_motif(xyz, r_mol, _fallback_fb, _fallback_bb, _fallback_changed)
                      for xyz in unique]
        # Adjust reactive-bond distances toward TS-like values using the recipe.
        if all_breaking or all_forming:
            unique = [adjust_reactive_bond_distances(xyz, r_mol, all_breaking, all_forming)
                      for xyz in unique]
        # Orient H atoms on reactive centres away from forming/breaking bonds.
        if all_breaking or all_forming:
            unique = [orient_h_on_reactive_centers(xyz, r_mol, all_breaking, all_forming)
                      for xyz in unique]
        # Final collision and bivalent-H filter after all post-processing.
        def _has_bivalent_h(xyz_dict: dict) -> bool:
            """Return True if any H is within tight bonding distance of 2+ heavy atoms."""
            coords_check = np.array(xyz_dict['coords'], dtype=float)
            for h, sym in enumerate(xyz_dict['symbols']):
                if sym != 'H':
                    continue
                n_close = sum(1 for j, sj in enumerate(xyz_dict['symbols'])
                              if sj != 'H' and j != h
                              and float(np.linalg.norm(coords_check[h] - coords_check[j])) < 1.15)
                if n_close >= 2:
                    return True
            return False
        unique = [xyz for xyz in unique
                  if not colliding_atoms(xyz) and not _has_bivalent_h(xyz)]

    # Cap: keep at most 5 guesses to avoid flooding downstream DFT.
    if len(unique) > 5:
        logger.debug(f'Linear (rxn={rxn.label}): capping {len(unique)} guesses to 5.')
        unique = unique[:5]

    return unique


register_job_adapter('linear', LinearAdapter)
