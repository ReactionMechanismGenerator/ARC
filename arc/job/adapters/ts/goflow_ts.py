"""
An adapter for executing GoFlow TS-guess jobs.

GoFlow is a flow-matching, E(3)-equivariant ML model that predicts 3D
transition-state geometries directly from atom-mapped reactant + product
SMILES (with RDKit-derived atom features), without requiring an initial
guess. Like RitS it complements ARC's existing TS-search stack with a
fast ML method; unlike GCN it is not restricted to isomerizations and
unlike RitS it is conditioned on 2D reaction graphs rather than mapped
3D structures.

References
----------
- Galustian et al., *Digital Discovery* 2025: 10.1039/D5DD00283D
- Preprint: doi.org/10.26434/chemrxiv-2025-bk2rh
- Code (practical fork)  : https://github.com/heid-lab/goflow_lean
- Code (research fork)   : https://github.com/heid-lab/goflow

Implementation notes
--------------------
* The heavy ML stack (torch + torch-geometric + lightning + torchdiffeq)
  lives in its own conda env (``goflow_env``). This adapter never imports
  it directly — it shells out to ``arc/job/adapters/scripts/goflow_script.py``
  via ``subprocess.run``, which loads the pretrained checkpoint and runs
  the flow-matching ODE sampler.
* GoFlow's input is **atom-mapped reactant + product SMILES** (every H must
  be an explicit, mapped atom). ARC builds them in :func:`build_atom_mapped_smiles`
  from each ARCSpecies' RDKit Mol, using ``rxn.atom_map`` to keep
  reactant→product correspondences consistent.
* GoFlow was trained on RDB7 (small organic; H/C/N/O/F). Reactions outside
  this domain are skipped cleanly with a one-line warning by
  :func:`_within_goflow_supported_domain`.
* The shipped checkpoint in goflow_lean@main is a 45-byte LFS pointer
  rather than a real Lightning ckpt. We validate by file size at adapter
  init time, defer ``torch.load``-level validation to ``goflow_script.py``,
  and skip cleanly if no real checkpoint is available — the rest of ARC's
  TS-search pipeline (heuristics, GCN, AutoTST, …) keeps running.
* ``incore_capacity = 1`` so the scheduler serializes GoFlow jobs and a
  single GPU is not asked to load multiple checkpoints in parallel.
"""

import datetime
import os
import subprocess
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem

from arc.common import ARC_PATH, get_logger, save_yaml_file, read_yaml_file
from arc.imports import settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter
from arc.job.factory import register_job_adapter
from arc.plotter import save_geo
from arc.species.converter import str_to_xyz, to_rdkit_mol, xyz_to_dmat, xyz_to_str
from arc.species.species import ARCSpecies, TSGuess, colliding_atoms

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction


GOFLOW_PYTHON = settings.get('GOFLOW_PYTHON')
GOFLOW_REPO_PATH = settings.get('GOFLOW_REPO_PATH')
GOFLOW_CKPT_PATH = settings.get('GOFLOW_CKPT_PATH')
GOFLOW_FEAT_DICT_PATH = settings.get('GOFLOW_FEAT_DICT_PATH')

GOFLOW_SCRIPT_PATH = os.path.join(ARC_PATH, 'arc', 'job', 'adapters', 'scripts', 'goflow_script.py')
DEFAULT_N_SAMPLES = 10
DEFAULT_NUM_STEPS = 25

# Domain guard — GoFlow was trained on RDB7 (small organic).
SUPPORTED_GOFLOW_ELEMENTS = frozenset({'H', 'C', 'N', 'O', 'F'})
MAX_GOFLOW_ATOMS = 100
GOFLOW_DEDUP_DMAT_RMSD = 0.15

# File-size thresholds (mirror settings.py, used by _goflow_environment_ready).
_GOFLOW_CKPT_MIN_SIZE = 1_000_000
_GOFLOW_FEAT_DICT_MIN_SIZE = 100

logger = get_logger()


class GoFlowAdapter(JobAdapter):
    """
    A class for executing GoFlow TS-guess jobs.

    Args:
        project (str): The project's name.
        project_directory (str): The path to the local project directory.
        job_type (list, str): The job's type, validated against ``JobTypeEnum``.
        args (dict, optional): Methods/troubleshooting; honored keys are
            ``args['keyword']['n_samples']`` and ``args['keyword']['num_steps']``.
        bath_gas (str, optional): A bath gas. Currently only used in OneDMin.
        checkfile (str, optional): The path to a previous Gaussian checkfile.
        conformer (int, optional): Conformer number if optimizing conformers.
        constraints (list, optional): A list of constraints.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
        dihedral_increment (float, optional): Unused for GoFlow.
        dihedrals (List[float], optional): The dihedral angles corresponding to self.torsions.
        directed_scan_type (str, optional): The type of the directed scan.
        ess_settings (dict, optional): A dictionary of available ESS.
        ess_trsh_methods (List[str], optional): A list of troubleshooting methods.
        execution_type (str, optional): The execution type, 'incore', 'queue', or 'pipe'.
        fine (bool, optional): Whether to use fine geometry optimization parameters.
        initial_time (datetime.datetime or str, optional): The time at which this job was initiated.
        irc_direction (str, optional): The direction of the IRC job.
        job_id (int, optional): The job's ID determined by the server.
        job_memory_gb (int, optional): The total job allocated memory in GB.
        job_name (str, optional): The job's name.
        job_num (int, optional): Used as the entry number in the database.
        job_server_name (str, optional): Job's name on the server.
        job_status (list, optional): The job's server and ESS statuses.
        level (Level, optional): The level of theory to use.
        max_job_time (float, optional): The maximal allowed job time on the server in hours.
        run_multi_species (bool, optional): Whether to run a job for multiple species in the same input file.
        reactions (List[ARCReaction], optional): Entries are ARCReaction instances.
        rotor_index (int, optional): The 0-indexed rotor number.
        server (str): The server to run on.
        server_nodes (list, optional): The nodes this job was previously submitted to.
        species (List[ARCSpecies], optional): Entries are ARCSpecies instances.
        testing (bool, optional): Whether the object is generated for testing purposes.
        times_rerun (int, optional): Number of times this job was re-run.
        torsions (List[List[int]], optional): The 0-indexed atom indices of the torsion(s).
        tsg (int, optional): TSGuess number if optimizing TS guesses.
        xyz (dict, optional): The 3D coordinates to use.
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
                 run_multi_species: bool = False,
                 reactions: Optional[List['ARCReaction']] = None,
                 rotor_index: Optional[int] = None,
                 server: Optional[str] = None,
                 server_nodes: Optional[list] = None,
                 queue: Optional[str] = None,
                 attempted_queues: Optional[List[str]] = None,
                 species: Optional[List['ARCSpecies']] = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: Optional[List[List[int]]] = None,
                 tsg: Optional[int] = None,
                 xyz: Optional[dict] = None,
                 ):

        self.incore_capacity = 1
        self.job_adapter = 'goflow'
        self.execution_type = execution_type or 'incore'
        self.command = 'goflow_script.py'
        self.url = 'https://github.com/heid-lab/goflow_lean'

        if reactions is None:
            raise ValueError('Cannot execute GoFlow without ARCReaction object(s).')

        self.n_samples = DEFAULT_N_SAMPLES
        self.num_steps = DEFAULT_NUM_STEPS
        if args and isinstance(args, dict):
            kw = args.get('keyword') or dict()
            if 'n_samples' in kw:
                try:
                    self.n_samples = int(kw['n_samples'])
                except (TypeError, ValueError):
                    logger.warning(f"GoFlow adapter: could not parse args['keyword']['n_samples']="
                                   f"{kw['n_samples']!r} as an int; falling back to "
                                   f"DEFAULT_N_SAMPLES={DEFAULT_N_SAMPLES}.")
            if 'num_steps' in kw:
                try:
                    self.num_steps = int(kw['num_steps'])
                except (TypeError, ValueError):
                    logger.warning(f"GoFlow adapter: could not parse args['keyword']['num_steps']="
                                   f"{kw['num_steps']!r} as an int; falling back to "
                                   f"DEFAULT_NUM_STEPS={DEFAULT_NUM_STEPS}.")

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
                            run_multi_species=run_multi_species,
                            reactions=reactions,
                            rotor_index=rotor_index,
                            server=server,
                            server_nodes=server_nodes,
                            queue=queue,
                            attempted_queues=attempted_queues,
                            species=species,
                            testing=testing,
                            times_rerun=times_rerun,
                            torsions=torsions,
                            tsg=tsg,
                            xyz=xyz,
                            )

    def write_input_file(self) -> None:
        """No standalone input file — see set_files() (writes input.yml)."""
        pass

    def set_files(self) -> None:
        """
        Set files to be uploaded and downloaded for queue execution.

        ``self.files_to_upload`` is a list of dictionaries, each with the keys
        ``'name'``, ``'source'``, ``'make_x'``, ``'local'``, and ``'remote'``.
        """
        # 1. Upload
        if self.execution_type != 'incore':
            self.write_submit_script()
            from arc.imports import settings as _s
            self.files_to_upload.append(self.get_file_property_dictionary(
                file_name=_s['submit_filenames'][_s['servers'][self.server]['cluster_soft']]))
        if os.path.isfile(self.yml_in_path):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='input.yml'))
        if os.path.isfile(self.reactant_xyz_path):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='reactant.xyz'))
        if os.path.isfile(self.product_xyz_path):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='product.xyz'))
        # 2. Download
        self.files_to_download.append(self.get_file_property_dictionary(file_name='output.yml'))
        self.files_to_download.append(self.get_file_property_dictionary(file_name='goflow_ts.xyz'))

    def set_additional_file_paths(self) -> None:
        """Set the local file paths used by GoFlow at job time."""
        self.reactant_xyz_path = os.path.join(self.local_path, 'reactant.xyz')
        self.product_xyz_path = os.path.join(self.local_path, 'product.xyz')
        self.ts_out_xyz_path = os.path.join(self.local_path, 'goflow_ts.xyz')
        self.yml_in_path = os.path.join(self.local_path, 'input.yml')
        self.yml_out_path = os.path.join(self.local_path, 'output.yml')

    def set_input_file_memory(self) -> None:
        """Set the input file memory attribute."""
        self.cpu_cores, self.job_memory_gb = 1, 1

    def execute_incore(self):
        """Execute the GoFlow job locally (in-process subprocess)."""
        self._log_job_execution()
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()
        self.execute_goflow()
        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        """Execute the GoFlow job to the server's queue."""
        self.execute_goflow(exe_type='queue')

    def execute_goflow(self, exe_type: str = 'incore'):
        """
        Drive the GoFlow subprocess and stitch its output back into ARC.

        Iterates over every reaction in ``self.reactions`` (per the JobAdapter
        contract; multiple reactions can share one adapter instance). The
        per-rxn input.yml / reactant.xyz / product.xyz / output.yml files
        share the adapter's ``self.local_path`` and are overwritten between
        reactions; results are consumed into ``rxn.ts_species.ts_guesses``
        before the next reaction runs (same idiom as AutoTST).

        Args:
            exe_type (str, optional): Either ``'incore'`` (run locally now) or
                ``'queue'`` (just stage the input.yml + submit script).
        """
        if not _goflow_environment_ready():
            return
        self.reactions = [self.reactions] if not isinstance(self.reactions, list) else self.reactions
        timeout_s = getattr(self, 'goflow_subprocess_timeout', 600)

        for rxn in self.reactions:
            ok, reason = _within_goflow_supported_domain(rxn)
            if not ok:
                logger.warning(f'GoFlow: skipping {rxn.label} — outside validated domain ({reason}).')
                continue

            if rxn.ts_species is None:
                rxn.ts_species = ARCSpecies(label=self.species_label,
                                            is_ts=True,
                                            charge=rxn.charge,
                                            multiplicity=rxn.multiplicity,
                                            )

            # Build atom-aligned reactant + product XYZ files. ARC's get_reactants_xyz /
            # get_products_xyz already use rxn.atom_map to align orderings.
            try:
                r_xyz_dict = rxn.get_reactants_xyz(return_format='dict')
                p_xyz_dict = rxn.get_products_xyz(return_format='dict')
            except Exception as e:
                logger.warning(f'GoFlow: could not build mapped XYZs for {rxn.label}: {e}')
                continue
            if r_xyz_dict is None or p_xyz_dict is None:
                logger.warning(f'GoFlow: empty mapped XYZs for {rxn.label}')
                continue
            if len(r_xyz_dict['symbols']) != len(p_xyz_dict['symbols']):
                logger.warning(f'GoFlow: atom count mismatch for {rxn.label} '
                               f'(R has {len(r_xyz_dict["symbols"])}, P has {len(p_xyz_dict["symbols"])}). Skipping.')
                continue

            r_smiles = build_atom_mapped_smiles(rxn, side='reactants')
            p_smiles = build_atom_mapped_smiles(rxn, side='products')
            if r_smiles is None or p_smiles is None:
                logger.warning(f'GoFlow: could not build atom-mapped SMILES for {rxn.label} — skipping.')
                continue

            write_xyz_file(r_xyz_dict, self.reactant_xyz_path, comment=f'{rxn.label} reactant')
            write_xyz_file(p_xyz_dict, self.product_xyz_path, comment=f'{rxn.label} product')

            input_dict = {'reactant_xyz_path': self.reactant_xyz_path,
                          'product_xyz_path': self.product_xyz_path,
                          'reactant_smiles': r_smiles,
                          'product_smiles': p_smiles,
                          'goflow_repo_path': GOFLOW_REPO_PATH,
                          'ckpt_path': GOFLOW_CKPT_PATH,
                          'feat_dict_path': GOFLOW_FEAT_DICT_PATH,
                          'output_xyz_path': self.ts_out_xyz_path,
                          'yml_out_path': self.yml_out_path,
                          'n_samples': self.n_samples,
                          'num_steps': self.num_steps,
                          'device': 'auto'}
            save_yaml_file(path=self.yml_in_path, content=input_dict)

            if exe_type == 'queue':
                self.legacy_queue_execution()
                continue

            cmd = [GOFLOW_PYTHON, GOFLOW_SCRIPT_PATH, '--yml_in_path', self.yml_in_path]
            try:
                result = subprocess.run(cmd, check=False, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                logger.warning(f'GoFlow subprocess timed out after {timeout_s}s for {rxn.label}; '
                               f'skipping. Increase adapter.goflow_subprocess_timeout to extend.')
                continue
            if result.returncode != 0:
                logger.warning(f'GoFlow subprocess returned non-zero exit code {result.returncode} '
                               f'for {rxn.label}.')
                continue

            if not os.path.isfile(self.yml_out_path):
                logger.warning(f'GoFlow produced no output YAML at {self.yml_out_path} for {rxn.label}.')
                continue

            tsg_dicts = read_yaml_file(self.yml_out_path) or list()
            n_added = 0
            for tsg_dict in tsg_dicts:
                if process_goflow_tsg(tsg_dict=tsg_dict,
                                      local_path=self.local_path,
                                      ts_species=rxn.ts_species):
                    n_added += 1

            if len(self.reactions) < 5:
                if n_added:
                    logger.info(f'GoFlow successfully found {n_added} TS guesses for {rxn.label}.')
                else:
                    logger.info(f'GoFlow did not find any successful TS guesses for {rxn.label}.')


def write_xyz_file(xyz_dict: dict, path: str, comment: str = '') -> None:
    """Write an ARC xyz dict to a plain XYZ file (``<n>\\n<comment>\\n<body>``)."""
    body = xyz_to_str(xyz_dict)
    n_atoms = len(xyz_dict['symbols'])
    safe_comment = comment.replace('\n', ' ').strip()
    with open(path, 'w') as f:
        f.write(f'{n_atoms}\n{safe_comment}\n{body}\n')


def build_atom_mapped_smiles(rxn: 'ARCReaction', side: str) -> Optional[str]:
    """
    Build an atom-mapped SMILES (every H an explicit, mapped atom) for the
    reactant or product side of ``rxn``, with map numbers consistent across
    the two sides via ``rxn.atom_map``.

    Args:
        rxn: The ARCReaction.
        side: ``'reactants'`` or ``'products'``.

    Returns:
        A SMILES string with every atom carrying an atom-map number 1..N,
        or ``None`` if any precondition fails (mol unavailable, atom_map
        missing, post-roundtrip validation fails). Returning ``None`` lets
        the caller skip cleanly without raising.
    """
    if side not in ('reactants', 'products'):
        return None
    try:
        expanded_r, expanded_p = rxn.get_reactants_and_products(return_copies=False)
    except Exception:
        return None
    species = expanded_r if side == 'reactants' else expanded_p
    if not species:
        return None
    try:
        atom_map = rxn.atom_map
    except Exception:
        return None
    if atom_map is None and side == 'products':
        return None

    # Combine each species's RDKit mol (with explicit Hs) into one editable mol.
    combined = Chem.RWMol()
    running = 0
    for spc in species:
        if spc.mol is None:
            return None
        try:
            rd_mol = to_rdkit_mol(spc.mol, remove_h=False, sanitize=True)
        except Exception:
            return None
        # Defensive: ensure every H is a real atom (not a bracket H count).
        rd_mol = Chem.AddHs(rd_mol)
        for atom in rd_mol.GetAtoms():
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons())
            combined.AddAtom(new_atom)
        for bond in rd_mol.GetBonds():
            combined.AddBond(running + bond.GetBeginAtomIdx(),
                             running + bond.GetEndAtomIdx(),
                             bond.GetBondType())
        running += rd_mol.GetNumAtoms()

    n_atoms_combined = combined.GetNumAtoms()

    # Apply atom-map numbers.
    if side == 'reactants':
        for i in range(n_atoms_combined):
            combined.GetAtomWithIdx(i).SetAtomMapNum(i + 1)
    else:
        # Product atom at combined-position p was reactant atom r where
        # atom_map[r] == p. Build the inverse map once (O(N)) instead of
        # calling list.index per atom (O(N²)).
        product_to_reactant = [-1] * n_atoms_combined
        for r_idx, p_idx in enumerate(atom_map):
            if 0 <= p_idx < n_atoms_combined:
                product_to_reactant[p_idx] = r_idx
        if any(r == -1 for r in product_to_reactant):
            return None  # mapping incomplete or out of range
        for p_idx, r_idx in enumerate(product_to_reactant):
            combined.GetAtomWithIdx(p_idx).SetAtomMapNum(r_idx + 1)

    try:
        smiles = Chem.MolToSmiles(combined.GetMol(), canonical=False)
    except Exception:
        return None

    # Hard validation: round-trip the SMILES and confirm atom count + map set.
    params = Chem.SmilesParserParams()
    params.removeHs = False
    check = Chem.MolFromSmiles(smiles, params)
    if check is None:
        return None
    if check.GetNumAtoms() != n_atoms_combined:
        return None
    maps = sorted(a.GetAtomMapNum() for a in check.GetAtoms())
    if maps != list(range(1, n_atoms_combined + 1)):
        return None
    return smiles


def _heavy_atom_pair_distances(xyz: dict) -> Optional[np.ndarray]:
    """
    Flat array of pairwise atom-atom distances over the HEAVY-atom subset
    of ``xyz`` (skipping every H). Returns ``None`` if there are <2 heavy
    atoms — caller should fall back to all-atom dmat.
    """
    symbols = xyz['symbols']
    heavy = [i for i, s in enumerate(symbols) if s != 'H']
    if len(heavy) < 2:
        return None
    coords = np.asarray(xyz['coords'], dtype=float)[heavy]
    iu = np.triu_indices(len(heavy), k=1)
    return np.linalg.norm(coords[iu[0]] - coords[iu[1]], axis=1)


def _heavy_atom_dmat_rmsd(xyz1: dict, xyz2: dict,
                          dmat1: Optional[np.ndarray] = None,
                          dmat2: Optional[np.ndarray] = None) -> float:
    """
    Aggregate root-mean-square deviation of pairwise distance matrices
    computed over the HEAVY-atom subset only. Translation- and rotation-
    invariant (uses internal distances) AND torsion-invariant (rotor H
    motions don't enter the comparison).

    Why heavy-atom only: torsional sampling of non-reactive rotors (e.g.
    a terminal CH2 swinging around its single bond) moves only the H atoms;
    the heavy-atom skeleton stays put. ML samples drawn from a flat
    torsional PES therefore look "different" by all-atom dmat (rotor H's
    at different angles) but are chemically the same TS — they collapse
    to the same optimized geometry under any QM saddle-point search.

    ``dmat1`` / ``dmat2`` accept pre-computed heavy-atom distance vectors
    (from :func:`_heavy_atom_pair_distances`); pass them when comparing
    one xyz against many to avoid recomputation in the dedup loop.

    Falls back to all-atom dmat for purely-hydrogen systems (defensive;
    real TS guesses always have ≥2 heavy atoms). Returns ``inf`` if the
    two xyzs have differing symbol sequences.
    """
    if xyz1.get('symbols') != xyz2.get('symbols'):
        return float('inf')
    if dmat1 is None:
        dmat1 = _heavy_atom_pair_distances(xyz1)
    if dmat2 is None:
        dmat2 = _heavy_atom_pair_distances(xyz2)
    if dmat1 is None or dmat2 is None:
        d1 = xyz_to_dmat(xyz1)
        d2 = xyz_to_dmat(xyz2)
        if d1 is None or d2 is None or d1.shape != d2.shape:
            return float('inf')
        diff = (d1 - d2).flatten()
        return float(np.sqrt(np.mean(diff * diff)))
    diff = dmat1 - dmat2
    return float(np.sqrt(np.mean(diff * diff)))


def process_goflow_tsg(tsg_dict: dict,
                       local_path: str,
                       ts_species: ARCSpecies,
                       dmat_rmsd_tol: float = GOFLOW_DEDUP_DMAT_RMSD,
                       ) -> bool:
    """
    Convert a single TSGuess-shaped dict from ``goflow_script.py`` into an
    ARC ``TSGuess`` object, consolidate against existing similar guesses,
    and append if unique.

    Dedup uses an *aggregate* distance-matrix RMSD threshold
    (``dmat_rmsd_tol``, default ``GOFLOW_DEDUP_DMAT_RMSD`` Å) computed over
    HEAVY atoms only. This is translation/rotation-invariant (important for
    ML samples in random orientations) AND torsion-invariant (so terminal
    rotor wells of the same TS collapse to a single guess instead of
    leaking 5+ near-identical structures into the downstream queue).

    If a near-duplicate is found, the existing guess's ``method`` is
    annotated with ``" and GoFlow"`` (mirroring the heuristics adapter's
    consolidation pattern) so the consumer knows GoFlow also produced it.

    Returns:
        ``True`` if a new (unique, non-colliding) TS guess was appended,
        ``False`` otherwise.
    """
    if not tsg_dict.get('success') or not tsg_dict.get('initial_xyz'):
        return False
    try:
        ts_xyz = str_to_xyz(tsg_dict['initial_xyz'])
    except Exception as e:
        logger.warning(f'GoFlow: could not parse TS xyz: {e}')
        return False
    if colliding_atoms(ts_xyz):
        return False

    # Pre-compute the new candidate's heavy-atom dmat once; cache existing
    # guesses' dmats on the TSGuess object on first comparison so successive
    # calls amortize to O(1) per pair (the dedup loop itself is O(N) per
    # call → O(N²) over a batch of N samples; without caching it would be O(N^3)).
    ts_dmat = _heavy_atom_pair_distances(ts_xyz)
    for other_tsg in ts_species.ts_guesses:
        if not (other_tsg.success and other_tsg.initial_xyz is not None):
            continue
        other_dmat = getattr(other_tsg, '_goflow_heavy_dmat', None)
        if other_dmat is None:
            other_dmat = _heavy_atom_pair_distances(other_tsg.initial_xyz)
            try:
                other_tsg._goflow_heavy_dmat = other_dmat
            except (AttributeError, TypeError):
                pass  # accept the recompute cost
        if _heavy_atom_dmat_rmsd(ts_xyz, other_tsg.initial_xyz,
                                 dmat1=ts_dmat, dmat2=other_dmat) < dmat_rmsd_tol:
            if 'goflow' not in other_tsg.method.lower():
                other_tsg.method += ' and GoFlow'
            return False

    method_index = int(tsg_dict.get('method_index', 0))
    tsg = TSGuess(method='GoFlow',
                  method_direction=tsg_dict.get('method_direction', 'F'),
                  method_index=method_index,
                  index=len(ts_species.ts_guesses),
                  success=True,
                  )
    tsg.process_xyz(ts_xyz)
    ts_species.ts_guesses.append(tsg)
    save_geo(xyz=ts_xyz,
             path=local_path,
             filename=f'GoFlow {method_index}',
             format_='xyz',
             comment=f'GoFlow sample {method_index}',
             )
    return True


def _within_goflow_supported_domain(rxn: 'ARCReaction') -> Tuple[bool, str]:
    """
    Check that ``rxn`` falls within GoFlow's validated training domain
    (RDB7-like: H/C/N/O/F, modest atom count).

    Returns:
        (True, '') if the reaction is supported; (False, reason) otherwise.
    """
    elements_seen = set()
    n_atoms = 0
    for spc in (rxn.r_species or []):
        try:
            symbols = spc.get_xyz()['symbols'] if spc.get_xyz() else \
                tuple(a.element.symbol for a in spc.mol.atoms) if spc.mol is not None else tuple()
        except Exception:
            symbols = tuple(a.element.symbol for a in spc.mol.atoms) if spc.mol is not None else tuple()
        elements_seen.update(symbols)
        n_atoms += len(symbols)
    unsupported = elements_seen - SUPPORTED_GOFLOW_ELEMENTS
    if unsupported:
        return False, f'unsupported element(s): {sorted(unsupported)}'
    if n_atoms > MAX_GOFLOW_ATOMS:
        return False, f'reaction has {n_atoms} atoms, above MAX_GOFLOW_ATOMS={MAX_GOFLOW_ATOMS}'
    if n_atoms == 0:
        return False, 'no atoms found in reactants'
    return True, ''


def _goflow_environment_ready() -> bool:
    """
    Verify that everything GoFlow needs at runtime is in place. Logs a clear
    one-line warning per missing piece and returns ``False`` so the adapter
    can skip cleanly without raising.

    Note: ``torch.load``-level checkpoint validation is deferred to
    ``goflow_script.py`` (which runs in goflow_env where torch is available).
    """
    ok = True
    if not GOFLOW_PYTHON or not os.path.isfile(GOFLOW_PYTHON):
        logger.warning('GoFlow adapter: goflow_env python not found '
                       '(set GOFLOW_PYTHON or run `make install-goflow`). Skipping GoFlow TS guesses.')
        ok = False
    if not GOFLOW_REPO_PATH or not os.path.isdir(GOFLOW_REPO_PATH):
        logger.warning('GoFlow adapter: goflow_lean source checkout not found '
                       '(set ARC_GOFLOW_REPO or run `make install-goflow`). Skipping GoFlow TS guesses.')
        ok = False
    if not GOFLOW_CKPT_PATH or not os.path.isfile(GOFLOW_CKPT_PATH) \
            or os.path.getsize(GOFLOW_CKPT_PATH) < _GOFLOW_CKPT_MIN_SIZE:
        logger.warning('GoFlow adapter: pretrained checkpoint not found or too small '
                       '(the in-repo file may be a 45-byte LFS pointer; set ARC_GOFLOW_CKPT to a real ckpt). '
                       'Skipping GoFlow TS guesses.')
        ok = False
    if not GOFLOW_FEAT_DICT_PATH or not os.path.isfile(GOFLOW_FEAT_DICT_PATH) \
            or os.path.getsize(GOFLOW_FEAT_DICT_PATH) < _GOFLOW_FEAT_DICT_MIN_SIZE:
        logger.warning('GoFlow adapter: atom-feature dictionary not found or too small '
                       '(set ARC_GOFLOW_FEAT_DICT to a real pickle). Skipping GoFlow TS guesses.')
        ok = False
    return ok


register_job_adapter('goflow', GoFlowAdapter)
