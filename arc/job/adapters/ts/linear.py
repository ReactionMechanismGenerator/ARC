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
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from arc.common import almost_equal_coords, get_logger
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

# ---------------------------------------------------------------------------
# Re-exports from linear_utils subpackage
# ---------------------------------------------------------------------------
from arc.job.adapters.ts.linear_utils.math_zmat import (  # noqa: F401
    BASE_WEIGHT_GRID,
    HAMMOND_DELTA,
    WEIGHT_ROUND,
    _ANGLE_MAX,
    _ANGLE_MIN,
    _BOND_LENGTH_MIN,
    _clip01,
    _get_all_referenced_atoms,
    _get_all_zmat_rows,
    average_zmat_params,
    get_r_constraints,
    get_rxn_weight,
    get_weight_grid,
    interp_dihedral_deg,
)
from arc.job.adapters.ts.linear_utils.postprocess import (  # noqa: F401
    FamilyPostprocessor,
    FamilyValidator,
    _FAMILY_POSTPROCESSORS,
    _FAMILY_VALIDATORS,
    _PAULING_DELTA,
    _postprocess_generic,
    _postprocess_h_migration,
    _postprocess_ts_guess,
    _validate_h_migration,
    _has_excessive_backbone_drift,
    _validate_ts_guess,
    has_inward_migrating_group_h,
)
from arc.job.adapters.ts.linear_utils.isomerization import (  # noqa: F401
    _RING_CLOSURE_THRESHOLD,
    _build_4center_interchange_ts,
    _generate_zmat_branch,
    _ring_closure_xyz,
    backbone_atom_map,
    get_near_attack_xyz,
    path_has_cumulated_bonds,
)
from arc.job.adapters.ts.linear_utils.addition import (  # noqa: F401
    _apply_intra_frag_contraction,
    _detect_intra_frag_ring_bonds,
    _find_split_bonds_by_fragmentation,
    _map_and_verify_fragments,
    _migrate_h_between_fragments,
    _migrate_verified_atoms,
    _stretch_bond,
    _stretch_core_from_large,
    _try_insertion_ring,
)


logger = get_logger()


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
                           max_ring_bond: float = 1.75,
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

    In both strategies the actual TS geometry is built by ``_stretch_bond``,
    which rigidly translates the smaller fragment(s) so that the split bonds
    reach their Pauling TS estimates (single-bond length + 0.42 Å).  For 3-fragment
    insertion/elimination patterns, ``_try_insertion_ring`` positions the
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
        frag_map = _map_and_verify_fragments(
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
            bases = _apply_intra_frag_contraction(
                uni_xyz, uni_mol, split_bonds, cross_bonds, multi_species,
                weight, label=f'rxn={rxn.label}, path={i}')
            # When contraction produced modified geometries, also keep
            # the uncontracted original as a fallback candidate.
            if bases[0] is not uni_xyz:
                bases.append(uni_xyz)

            for base_xyz in bases:
                ts_xyz = _stretch_bond(base_xyz, uni_mol, split_bonds,
                                       cross_bonds, weight,
                                       label=f'rxn={rxn.label}, path={i}')
                if ts_xyz is not None and migrating_atoms:
                    # _stretch_bond only moves the smallest fragment (often
                    # just the migrating atom).  When there are 3+ fragments,
                    # the core of the small product may still need to be
                    # stretched away from the large product (e.g. C-O in
                    # HO2 elimination).
                    ts_xyz = _stretch_core_from_large(
                        ts_xyz, uni_mol, split_bonds, core, large_prod_atoms,
                        small_prod_atoms, weight)
                    # Reset migrating atoms to their pre-stretch positions
                    # before calling _migrate_verified_atoms.  _stretch_bond
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
                    ts_xyz = _migrate_verified_atoms(
                        ts_xyz, uni_mol, migrating_atoms, core,
                        large_prod_atoms, weight, cross_bonds=cross_bonds)
                    is_valid, reason = _validate_ts_guess(
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
            ts_xyz = _stretch_bond(uni_xyz, uni_mol, split_bonds,
                                   cross_bonds, weight,
                                   label=f'rxn={rxn.label}, path={i}-fallback')
            if ts_xyz is not None:
                ts_xyzs.append(ts_xyz)
            # Also try ring contraction then stretch for reactions that need
            # it (e.g. cyclic ether formation).
            ring_xyzs = _apply_intra_frag_contraction(
                uni_xyz, uni_mol, split_bonds, cross_bonds, multi_species,
                weight, label=f'rxn={rxn.label}, path={i}-fallback')
            for ring_xyz in ring_xyzs:
                ts_xyz = _stretch_bond(ring_xyz, uni_mol, split_bonds,
                                       cross_bonds, weight,
                                       label=f'rxn={rxn.label}, path={i}-fallback')
                if ts_xyz is not None:
                    is_valid, reason = _validate_ts_guess(
                        ts_xyz, set(), split_bonds, uni_mol,
                        label=f'rxn={rxn.label}, path={i}-fallback-contracted')
                    if is_valid:
                        ts_xyzs.append(ts_xyz)

    # ----- Strategy 2: fragmentation supplement -----
    # Always run fragmentation-based search in addition to the template path.
    # The template's r_label_map can mis-identify the primary fragmentation
    # (e.g. marking a C-H bond as the split instead of the C-O bond in
    # 1,3_Insertion_ROR), so fragmentation supplements those results.
    #
    # Two-tier filtering of fragmentation cuts:
    # 1. Subgraph isomorphism (via _map_and_verify_fragments) is the gold
    #    standard: it checks that each fragment is topologically isomorphic
    #    to one of the multi_species.  This catches wrong-reaction cuts
    #    (e.g. H removal from the wrong C atom producing a different radical).
    #    However, it requires that the cut produces exactly len(multi_species)
    #    fragments, which fails for multi-bond cuts in insertion/elimination
    #    reactions where cross_bonds are needed to merge fragments.
    # 2. Composition dedup: when a cut fails the subgraph check, fall back
    #    to keeping one representative per unique set of fragment element
    #    compositions.  This is coarser but works for all cut types.
    cut_lists = _find_split_bonds_by_fragmentation(uni_mol, multi_species)
    isomorphism_verified: List[List[Tuple[int, int]]] = []
    isomorphism_unverified: List[List[Tuple[int, int]]] = []
    for cut in cut_lists:
        frag_map = _map_and_verify_fragments(
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
    for cut in cut_lists:
        sb_key = frozenset(cut)
        if sb_key in seen_split_sets:
            continue
        seen_split_sets.add(sb_key)
        # Apply ring contraction on original geometry first, then stretch.
        ring_xyzs = _apply_intra_frag_contraction(
            uni_xyz, uni_mol, cut, None, multi_species,
            weight, label=f'rxn={rxn.label}, frag-fallback')
        for ring_xyz in ring_xyzs:
            ts_xyz = _stretch_bond(ring_xyz, uni_mol, cut, cross_bonds=None,
                                   weight=weight,
                                   label=f'rxn={rxn.label}, frag-fallback')
            if ts_xyz is not None:
                ts_xyz = _migrate_h_between_fragments(
                    ts_xyz, uni_mol, cut, multi_species, weight)
                is_valid, reason = _validate_ts_guess(
                    ts_xyz, set(), cut, uni_mol,
                    label=f'rxn={rxn.label}, frag-fallback-post-migrate')
                if not is_valid:
                    logger.debug(f'Linear (rxn={rxn.label}, frag-fallback): '
                                 f'post-migration guess rejected — {reason}.')
                    continue
                ts_xyzs.append(ts_xyz)

    # Deduplicate near-identical guesses (including against existing_xyzs
    # from prior weight iterations to avoid wasting downstream DFT resources).
    # Repair ring bonds broken by Z-matrix interpolation before deduplication.
    if _ring_sets:
        changing_all = set()
        for pd in rxn.product_dicts:
            rl = pd.get('r_label_map')
            if rl:
                bb_i, fb_i = rxn.get_expected_changing_bonds(r_label_dict=rl)
                for b in list(bb_i or []) + list(fb_i or []):
                    changing_all.add((min(b), max(b)))
        reactive_all: Set[int] = set()
        for b in changing_all:
            reactive_all.update(b)
        ts_xyzs = [_fix_broken_ring_bonds(xyz, r_mol, _ring_sets, reactive_all, changing_all)
                    for xyz in ts_xyzs]

    prior: List[dict] = list(existing_xyzs or [])
    unique: List[dict] = []
    for xyz_candidate in ts_xyzs:
        if not any(almost_equal_coords(xyz_candidate, u)
                   for u in unique + prior):
            unique.append(xyz_candidate)
    return unique if unique else []


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
    (:func:`_postprocess_ts_guess`) and validation pipeline (:func:`_validate_ts_guess`)
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

        # Smart anchor selection: prefer spectator atoms adjacent to the reactive core
        # so the Z-matrix coordinate frame is stable across the interpolation.
        anchors = find_smart_anchors(r_mol, breaking_bonds=list(bb), forming_bonds=list(fb))

        # XYZ indices of atoms involved in the reactive event (forming + breaking bonds).
        # Only these coordinates will be interpolated; spectator coordinates are preserved
        # from the anchor to avoid washing out good TS geometry.
        reactive_xyz_indices: Set[int] = set()
        for bond in list(bb) + list(fb):
            reactive_xyz_indices.update(bond)
        # When a reactive atom sits in a ring, include its ring-bonded spectator
        # neighbors so they can track the reactive atom's motion during interpolation.
        changing_bonds = {(min(a, b), max(a, b)) for a, b in list(bb) + list(fb)}
        ring_bonds, reactive_xyz_indices = _get_ring_preservation_bonds(
            r_mol, reactive_xyz_indices, changing_bonds, _ring_sets)

        # Force the Z-matrix to measure every reactive bond distance explicitly,
        # regardless of whether that bond exists in the current molecule graph.
        # Type R misses forming bonds (not in r_mol); Type P misses breaking bonds
        # (not in mapped_p_mol).  A shared 'R_atom' constraint set covers both:
        # xyz_to_zmat will insert R_atom rows for each (i, j) pair before normal
        # atom ordering, so the interpolated Z-matrix captures the full reaction path.
        constraints = get_r_constraints(expected_breaking_bonds=list(bb),
                                        expected_forming_bonds=list(fb))
        # Also constrain ring bonds adjacent to reactive atoms so the Z-mat tree
        # directly connects the reactive atom to its ring neighbors, keeping
        # ring bond distances as explicit (interpolatable) Z-mat variables.
        # Only add when the reactive atom is already in the constraint tree.
        if ring_bonds:
            constrained_atoms = {a for pair in constraints.get('R_atom', []) for a in pair}
            existing = set(map(tuple, constraints.get('R_atom', [])))
            for rb in ring_bonds:
                if (rb[0] in constrained_atoms
                        and rb not in existing and (rb[1], rb[0]) not in existing):
                    constraints['R_atom'].append(rb)
                else:
                    reactive_xyz_indices.discard(rb[1])

        # Pre-rotate to near-attack conformation: rotate along the donor→acceptor
        # path so forming-bond atoms are geometrically close before Z-matrix
        # construction.  Without this, conformational changes required to bring
        # reactive atoms into proximity (e.g. O–O rotation in intra-H migrations)
        # are absent from both R and P equilibrium geometries and would never be
        # captured by interpolation alone.
        r_xyz_na = get_near_attack_xyz(r_xyz, r_mol, bonds=list(fb))
        # For the product frame, approach-relevant torsions involve the breaking
        # bonds (atoms that are bonded in R but separate in P).
        op_xyz_na = get_near_attack_xyz(op_xyz, mapped_p_mol, bonds=list(bb))

        # Ring-closure fast path: when the forming-bond distance exceeds the threshold,
        # the reactive sites are too far apart for Z-matrix interpolation to produce a
        # connected TS geometry.  Use a dedicated ring-closure algorithm instead.
        # Ring closure is weight-independent (purely geometric), so only produce it
        # at the canonical weight (0.5) to avoid duplicates across weight iterations.
        # At non-0.5 weights, the zmat branch still runs — it has proper reactive
        # bond info and can produce valid weight-dependent guesses.
        used_ring_closure = False
        needs_ring_closure = False
        for bond_pair in list(fb):
            r_coords = np.array(r_xyz_na['coords'], dtype=float)
            site_dist = float(np.linalg.norm(r_coords[bond_pair[0]] - r_coords[bond_pair[1]]))
            use_rc = site_dist > _RING_CLOSURE_THRESHOLD or path_has_cumulated_bonds(r_mol, bond_pair)
            if use_rc:
                needs_ring_closure = True
                if abs(weight - 0.5) <= 0.01:
                    rc_xyz = _ring_closure_xyz(r_xyz_na, r_mol, forming_bond=bond_pair)
                    if rc_xyz is not None:
                        # Ring closure only handles the ring-forming bond;
                        # apply the full postprocessing pipeline so that
                        # other reactive atoms (e.g. a migrating H in
                        # Concerted_Intra_Diels_alder_monocyclic_1,2_shiftH)
                        # are also repositioned to their TS positions.
                        rc_xyz, migrating_hs_rc = _postprocess_ts_guess(
                            rc_xyz, r_mol, list(fb), list(bb),
                            family=path_family, r_label_map=r_label_dict)
                        is_valid, _ = _validate_ts_guess(
                            rc_xyz, migrating_hs_rc, list(fb), r_mol,
                            label=f'rxn={rxn.label}, path={i}, ring-closure',
                            family=path_family)
                        if is_valid:
                            ts_xyzs.append(rc_xyz)
                            used_ring_closure = True
                            logger.debug(f'Linear (rxn={rxn.label}, path={i}): used ring-closure algorithm '
                                         f'(forming-bond distance {site_dist:.2f} Å, cumulated={path_has_cumulated_bonds(r_mol, bond_pair)}).')
        if used_ring_closure:
            continue

        forming_bonds_list = list(fb)
        breaking_bonds_list = list(bb)

        # Type R: reactant-topology Z-matrix chimera
        ts_xyz_r = _generate_zmat_branch(
            anchor_xyz=r_xyz_na, anchor_mol=r_mol, target_xyz=op_xyz_na,
            weight=weight, reactive_xyz_indices=reactive_xyz_indices,
            anchors=anchors, constraints=constraints, r_mol=r_mol,
            forming_bonds=forming_bonds_list, breaking_bonds=breaking_bonds_list,
            label=f'rxn={rxn.label}, path={i}, type=R',
            family=path_family, r_label_map=r_label_dict)
        if ts_xyz_r is not None:
            ts_xyzs.append(ts_xyz_r)

        # Type P: product-topology Z-matrix chimera (symmetric to Type R).
        # mapped_p_mol has product bond topology but with atoms reindexed to match op_xyz
        # (reactant atom ordering).
        ts_xyz_p = _generate_zmat_branch(
            anchor_xyz=op_xyz_na, anchor_mol=mapped_p_mol, target_xyz=r_xyz_na,
            weight=1.0 - weight, reactive_xyz_indices=reactive_xyz_indices,
            anchors=anchors, constraints=constraints, r_mol=r_mol,
            forming_bonds=forming_bonds_list, breaking_bonds=breaking_bonds_list,
            label=f'rxn={rxn.label}, path={i}, type=P',
            family=path_family, r_label_map=r_label_dict)
        if ts_xyz_p is not None:
            ts_xyzs.append(ts_xyz_p)

        # 4-center interchange fallback: when Z-mat chimeras produce nothing for
        # this path, try direct geometry construction for 4-center swap reactions
        # (e.g. 1,2_XY_interchange: X-C-C-Y → Y-C-C-X).
        if ts_xyz_r is None and ts_xyz_p is None:
            ts_4c = _build_4center_interchange_ts(
                r_xyz=r_xyz, r_mol=r_mol,
                bb=breaking_bonds_list, fb=forming_bonds_list,
                weight=weight, label=f'rxn={rxn.label}, path={i}')
            if ts_4c is not None:
                ts_xyzs.append(ts_4c)
                logger.debug(f'Linear (rxn={rxn.label}, path={i}): used 4-center interchange builder.')

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

        if r_xyz['symbols'] != p_xyz['symbols']:
            logger.debug(f'Linear (rxn={rxn.label}): trivial-map fallback skipped — '
                         f'R and P symbols do not match (different atom ordering).')
        else:
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
            bb_map = backbone_atom_map(r_mol, p_mol) if rxn.atom_map is None else None
            if bb_map is not None:
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

            if bb_map is not None:
                # Backbone map produces correct atom correspondences, so
                # H-bond changes represent real reactive bonds (e.g. H
                # migration) — keep them.  Bond-order changes in the
                # backbone are a *consequence* of the reactive event, not
                # independent reactive sites; including them would inflate
                # the reactive set and distort the bicyclic backbone.
                pass
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
                changing_bonds_fb = {(min(a, b), max(a, b)) for a, b in bb + fb}
                ring_bonds_fb, reactive_xyz_indices = _get_ring_preservation_bonds(
                    r_mol, reactive_xyz_indices, changing_bonds_fb, _ring_sets)

                anchors = find_smart_anchors(r_mol, breaking_bonds=list(bb), forming_bonds=list(fb))
                constraints = get_r_constraints(expected_breaking_bonds=list(bb),
                                                expected_forming_bonds=list(fb))
                if ring_bonds_fb:
                    constrained_atoms_fb = {a for pair in constraints.get('R_atom', []) for a in pair}
                    existing_fb = set(map(tuple, constraints.get('R_atom', [])))
                    for rb in ring_bonds_fb:
                        if (rb[0] in constrained_atoms_fb
                                and rb not in existing_fb and (rb[1], rb[0]) not in existing_fb):
                            constraints['R_atom'].append(rb)
                        else:
                            reactive_xyz_indices.discard(rb[1])

                # Skip near-attack for backbone-map cases: the forming
                # bond is across a rigid ring, so rotating the molecule to
                # bring the atoms closer just distorts the backbone.
                if bb_map is not None:
                    r_xyz_na = r_xyz
                    op_xyz_na = p_xyz
                else:
                    r_xyz_na = get_near_attack_xyz(r_xyz, r_mol, bonds=list(fb))
                    op_xyz_na = get_near_attack_xyz(p_xyz, p_mol, bonds=list(bb))

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
                    use_rc = site_dist > _RING_CLOSURE_THRESHOLD or path_has_cumulated_bonds(r_mol, bond_pair)
                    if use_rc:
                        needs_ring_closure = True
                        if abs(weight - 0.5) <= 0.01:
                            rc_xyz = _ring_closure_xyz(r_xyz, r_mol,
                                                       forming_bond=bond_pair)
                            if rc_xyz is not None:
                                rc_xyz, migrating_hs_rc = _postprocess_ts_guess(
                                    rc_xyz, r_mol, list(fb), list(bb),
                                    family=rxn.family)
                                is_valid, _ = _validate_ts_guess(
                                    rc_xyz, migrating_hs_rc, fb, r_mol,
                                    label=f'rxn={rxn.label}, trivial, ring-closure',
                                    family=rxn.family)
                                if is_valid:
                                    ts_xyzs.append(rc_xyz)
                                    used_ring_closure = True

                if not used_ring_closure and not needs_ring_closure:
                    # With the backbone atom map the reactive set is small and
                    # accurate, so postprocessing (H distance fix, etc.) is safe
                    # and needed.  With the identity map the reactive set is
                    # inflated to nearly all atoms and postprocessing would
                    # perturb the geometry — skip it.
                    skip_pp = bb_map is None
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
                    ts_r = _generate_zmat_branch(
                        anchor_xyz=r_xyz_na, anchor_mol=r_mol, target_xyz=op_xyz_na,
                        weight=weight, reactive_xyz_indices=reactive_xyz_indices,
                        anchors=anchors, constraints=constraints, r_mol=r_mol,
                        forming_bonds=list(fb), breaking_bonds=list(bb),
                        label=f'rxn={rxn.label}, trivial, w={weight}, type=R',
                        skip_postprocess=skip_pp, family=effective_family,
                        redistribute_ch2=bb_map is not None)
                    if ts_r is not None and not _has_excessive_backbone_drift(
                            ts_r, r_xyz_na, max_mean_heavy_disp=3.0,
                            reactive_indices=reactive_xyz_indices):
                        ts_xyzs.append(ts_r)
                    elif ts_r is not None:
                        logger.debug(f'Linear (rxn={rxn.label}, trivial, w={weight}, type=R): '
                                     f'discarded — excessive backbone drift from anchor.')

                    ts_p = _generate_zmat_branch(
                        anchor_xyz=op_xyz_na, anchor_mol=p_mol, target_xyz=r_xyz_na,
                        weight=1.0 - weight, reactive_xyz_indices=reactive_xyz_indices,
                        anchors=anchors, constraints=constraints, r_mol=r_mol,
                        forming_bonds=list(fb), breaking_bonds=list(bb),
                        label=f'rxn={rxn.label}, trivial, w={weight}, type=P',
                        skip_postprocess=skip_pp, family=effective_family,
                        redistribute_ch2=bb_map is not None)
                    if ts_p is not None and not _has_excessive_backbone_drift(
                            ts_p, op_xyz_na, max_mean_heavy_disp=3.0,
                            reactive_indices=reactive_xyz_indices):
                        ts_xyzs.append(ts_p)
                    elif ts_p is not None:
                        logger.debug(f'Linear (rxn={rxn.label}, trivial, w={weight}, type=P): '
                                     f'discarded — excessive backbone drift from anchor.')

    # Deduplicate: collapse near-identical guesses from different paths, from
    # Type R ≈ Type P (common for symmetric reactions at weight=0.5), or from
    # weight-insensitive algorithms (e.g. ring-closure) across repeated calls.
    prior: List[dict] = list(existing_xyzs or [])
    unique: List[dict] = []
    for xyz in ts_xyzs:
        if not any(almost_equal_coords(xyz, other)
                   for other in unique + prior):
            unique.append(xyz)
    return unique


register_job_adapter('linear', LinearAdapter)
