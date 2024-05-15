"""Migration-inference helpers for the linear TS-guess adapter.

This module owns the *graph-aware* helpers that recover migration
topology from a molecular bond graph plus one or two coordinate
snapshots — i.e. the helpers that answer "which H atom migrated, and
between which heavy atoms?" without re-running the migration itself.

These helpers were previously co-located with the local-shell geometry
helpers in :mod:`arc.job.adapters.ts.linear_utils.local_geometry`, but
they have a fundamentally different scope:

* :mod:`local_geometry` operates on the *immediate first shell* of a
  named heavy center (terminal CH₂/CH₃ regularization, internal
  reactive CH₂ repair, the orchestration helper that composes those).
* This module operates on the *whole bond graph* and on
  *displacement signals* between two coordinate snapshots — the
  donor/acceptor logic that mirrors what
  :func:`arc.job.adapters.ts.linear_utils.addition.migrate_verified_atoms`
  does internally, exposed as standalone helpers for the orchestrator.

Public API:

* :func:`identify_h_migration_pairs` — recover (donor, acceptor) for
  every migrating H given the small product's core, the large product's
  atoms, and (optionally) verified cross bonds from the template.
* :func:`infer_frag_fallback_h_migration` — strict S1–S5 deterministic
  inference of the single-H migration triple for a fragmentation
  fallback addition guess.

The helpers are pure: they accept dicts/molecules and return new dicts
or ``None``; they never mutate inputs.
"""

from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from arc.common import get_logger, get_single_bond_length
from arc.job.adapters.ts.linear_utils.geom_utils import split_mol_at_bonds

if TYPE_CHECKING:
    from arc.molecule import Molecule
    from arc.species.species import ARCSpecies


logger = get_logger()


# ---------------------------------------------------------------------------
# identify_h_migration_pairs — donor/acceptor recovery from bond graph
# ---------------------------------------------------------------------------


def identify_h_migration_pairs(xyz: dict,
                                 mol: 'Molecule',
                                 migrating_atoms: Set[int],
                                 core: Set[int],
                                 large_prod_atoms: Set[int],
                                 cross_bonds: Optional[List[Tuple[int, int]]] = None,
                                 ) -> List[Dict]:
    """Determine the (donor, acceptor) heavy atoms for each migrating H.

    This mirrors the donor/acceptor logic inside
    :func:`arc.job.adapters.ts.linear_utils.addition.migrate_verified_atoms`,
    but exposes it as a standalone function so the orchestrator
    (``interpolate_addition``) can recover the migration topology
    *without* re-running the migration.  The output is a deterministic
    list of dicts so callers can both feed local-cleanup helpers and
    drive  topology gates from the same source.

    Args:
        xyz: TS guess XYZ dict (used for the nearest-core fallback only).
        mol: RMG Molecule of the unimolecular species.
        migrating_atoms: Atom indices that are migrating.
        core: Atom indices in the small product's core.
        large_prod_atoms: Atom indices in the large product.
        cross_bonds: Forming bonds (verified, from the template).

    Returns:
        A deterministic list ordered by ``h_idx``.  Each entry is::

            {
                'h_idx': int,
                'donor': int,
                'acceptor': int,
                'source': 'cross_bond' | 'nearest_core',
            }

        Migrating atoms with no identifiable donor are dropped.
    """
    if xyz is None or not migrating_atoms:
        return []
    coords = np.asarray(xyz['coords'], dtype=float)
    symbols = xyz['symbols']
    atom_to_idx = {atom: i for i, atom in enumerate(mol.atoms)}

    cross_acceptor: Dict[int, int] = {}
    for a, b in (cross_bonds or []):
        if a in migrating_atoms and b in core:
            cross_acceptor[a] = b
        elif b in migrating_atoms and a in core:
            cross_acceptor[b] = a

    core_heavy = sorted(idx for idx in core if symbols[idx] != 'H')

    out: List[Dict] = []
    for h_idx in sorted(migrating_atoms):
        if h_idx >= len(coords):
            continue
        # Donor: heavy neighbor of h_idx in large_prod_atoms.
        donor: Optional[int] = None
        for nbr in mol.atoms[h_idx].bonds.keys():
            ni = atom_to_idx[nbr]
            if symbols[ni] != 'H' and ni in large_prod_atoms:
                donor = ni
                break
        if donor is None:
            continue
        # Acceptor: cross-bond partner if available, else nearest core heavy atom.
        acceptor = cross_acceptor.get(h_idx)
        source = 'cross_bond'
        if acceptor is None:
            if not core_heavy:
                continue
            dists = np.linalg.norm(
                coords[core_heavy] - coords[h_idx], axis=1)
            acceptor = core_heavy[int(dists.argmin())]
            source = 'nearest_core'
        out.append({
            'h_idx': int(h_idx),
            'donor': int(donor),
            'acceptor': int(acceptor),
            'source': source,
        })
    return out


# ---------------------------------------------------------------------------
# Fragmentation-fallback single-H migration inference
# ---------------------------------------------------------------------------


def _split_into_fragments(uni_mol: 'Molecule',
                            split_bonds: Sequence[Tuple[int, int]],
                            ) -> List[Set[int]]:
    """Return the connected components of ``uni_mol`` after removing
    every bond in ``split_bonds``.

    Delegates to :func:`geom_utils.split_mol_at_bonds`.
    """
    return split_mol_at_bonds(uni_mol, list(split_bonds or []))


def _heavy_formula_of_fragment(symbols: Sequence[str],
                                 fragment: Set[int]) -> Tuple[Tuple[str, int], ...]:
    """Return a hashable heavy-atom composition for *fragment*."""
    counts: Dict[str, int] = {}
    for idx in fragment:
        sym = symbols[idx]
        if sym == 'H':
            continue
        counts[sym] = counts.get(sym, 0) + 1
    return tuple(sorted(counts.items()))


def _h_count_of_fragment(symbols: Sequence[str], fragment: Set[int]) -> int:
    return sum(1 for idx in fragment if symbols[idx] == 'H')


def infer_frag_fallback_h_migration(pre_xyz: dict,
                                     post_xyz: dict,
                                     uni_mol: 'Molecule',
                                     split_bonds: Sequence[Tuple[int, int]],
                                     multi_species: Optional[Sequence['ARCSpecies']],
                                     label: str = '',
                                     displacement_threshold: float = 0.05,
                                     ) -> Optional[Dict]:
    """Deterministically infer the single-H migration triple for a
    fragmentation-fallback addition guess.

    The helper combines five strict signals (S1–S5) and returns either
    one trustworthy ``(h_idx, donor, acceptor, source)`` record or
    ``None`` whenever any ambiguity remains.  It NEVER returns multiple
    candidates and NEVER partially enriches.

    The contract:

    * **S1 — pre/post displacement**: exactly one H atom must have moved
      by more than ``displacement_threshold`` Å between ``pre_xyz`` and
      ``post_xyz``.  Multi-H or zero-H displacement → ``None``.
    * **S2 — reactant-graph adjacency**: the displaced H must have
      *exactly one* heavy neighbor in the reactant graph (the donor).
      Two heavy neighbors of the same H → ``None``.
    * **S3 — split-bond fragment membership**: after cutting
      ``split_bonds``, the donor and the acceptor must lie in
      *different* connected components.  Same-fragment placement is
      chemically incompatible with fragment-to-fragment H transfer →
      ``None``.
    * **S4 — product-composition consistency** (when ``multi_species``
      is supplied): the donor's fragment must show an H surplus
      relative to its matched product, and the acceptor's fragment an
      H deficit.  This mirrors the same logic
      :func:`migrate_h_between_fragments` itself uses.  Skipped when
      ``multi_species`` is ``None`` or fragment-to-product matching is
      ambiguous.
    * **S5 — local donor/acceptor geometry consistency**: in the
      post-migration geometry, the migrated H must satisfy
      ``d(D,H) ≤ 1.6 × sbl(D,H)`` AND ``d(A,H) ≤ 1.6 × sbl(A,H)``
      AND no other heavy atom in the acceptor's fragment can be closer
      to the H than ``0.95 × sbl(rival,H)``.

    Args:
        pre_xyz: Pre-migration TS XYZ dict (output of ``stretch_bond``).
        post_xyz: Post-migration TS XYZ dict (output of
            ``migrate_h_between_fragments``).
        uni_mol: Unimolecular reactant Molecule (defines the bond graph).
        split_bonds: The fragmentation cut.
        multi_species: Sequence of product :class:`ARCSpecies` for the
            optional composition consistency check (S4).  May be
            ``None`` to skip S4.
        label: Optional logging label.
        displacement_threshold: Å threshold for "moved" in S1.

    Returns:
        A single migration record dict
        ``{'h_idx': int, 'donor': int, 'acceptor': int, 'source': 'frag_inferred'}``
        on full success, otherwise ``None``.
    """
    if pre_xyz is None or post_xyz is None or uni_mol is None:
        return None
    pre_arr = np.asarray(pre_xyz['coords'], dtype=float)
    post_arr = np.asarray(post_xyz['coords'], dtype=float)
    if pre_arr.shape != post_arr.shape:
        return None
    symbols = post_xyz['symbols']
    n_atoms = len(symbols)

    # ---- S1: exactly one H moved ----
    moved_h: List[int] = []
    for h in range(n_atoms):
        if symbols[h] != 'H':
            continue
        if float(np.linalg.norm(post_arr[h] - pre_arr[h])) > displacement_threshold:
            moved_h.append(h)
    if len(moved_h) != 1:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S1: {len(moved_h)} H atoms moved).')
        return None
    h_idx = moved_h[0]

    # ---- S2: exactly one heavy neighbor of the migrated H ----
    atom_to_idx = {atom: i for i, atom in enumerate(uni_mol.atoms)}
    heavy_nbrs: List[int] = []
    for nbr in uni_mol.atoms[h_idx].bonds.keys():
        ni = atom_to_idx[nbr]
        if symbols[ni] != 'H':
            heavy_nbrs.append(ni)
    if len(heavy_nbrs) != 1:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S2: H{h_idx} has {len(heavy_nbrs)} '
                         f'heavy neighbors).')
        return None
    donor = heavy_nbrs[0]

    # ---- S3: donor and acceptor on different fragments after the cut ----
    fragments = _split_into_fragments(uni_mol, split_bonds)
    if len(fragments) < 2:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S3: only {len(fragments)} fragment(s) '
                         f'after cut).')
        return None

    donor_frag_idx: Optional[int] = None
    for fi, frag in enumerate(fragments):
        if donor in frag:
            donor_frag_idx = fi
            break
    if donor_frag_idx is None:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S3: donor {donor} not in any fragment).')
        return None
    other_frag_atoms: Set[int] = set()
    for fi, frag in enumerate(fragments):
        if fi != donor_frag_idx:
            other_frag_atoms.update(frag)
    if not other_frag_atoms:
        return None

    # The acceptor candidate set: heavy atoms in the *other* fragments.
    acceptor_candidates = [i for i in other_frag_atoms if symbols[i] != 'H']
    if not acceptor_candidates:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S3: no heavy atom in any non-donor fragment).')
        return None

    # ---- S4: composition consistency (donor fragment has H surplus,
    # acceptor fragment has H deficit), when products available.
    acceptor_frag_idx: Optional[int] = None
    if multi_species:
        try:
            target_h: List[int] = []
            target_heavy: List[Tuple[Tuple[str, int], ...]] = []
            for sp in multi_species:
                sp_symbols = sp.get_xyz()['symbols']
                target_h.append(sum(1 for s in sp_symbols if s == 'H'))
                heavy_counts: Dict[str, int] = {}
                for s in sp_symbols:
                    if s == 'H':
                        continue
                    heavy_counts[s] = heavy_counts.get(s, 0) + 1
                target_heavy.append(tuple(sorted(heavy_counts.items())))
        except Exception:  # pragma: no cover - defensive
            target_h = []
            target_heavy = []

        if target_heavy and len(target_heavy) == len(fragments):
            # Match each fragment to a target by unique heavy formula.
            frag_heavy = [_heavy_formula_of_fragment(symbols, f) for f in fragments]
            # Build a deterministic 1-to-1 matching: each target consumed once.
            frag_to_target: Dict[int, int] = {}
            used_targets: Set[int] = set()
            ok_match = True
            for fi, fh in enumerate(frag_heavy):
                match = None
                for ti, th in enumerate(target_heavy):
                    if ti in used_targets:
                        continue
                    if th == fh:
                        match = ti
                        break
                if match is None:
                    ok_match = False
                    break
                frag_to_target[fi] = match
                used_targets.add(match)

            if ok_match:
                # Compute H surplus/deficit per fragment.
                surplus: Dict[int, int] = {}
                for fi, ti in frag_to_target.items():
                    diff = _h_count_of_fragment(symbols, fragments[fi]) - target_h[ti]
                    surplus[fi] = diff
                # Donor's fragment must have surplus > 0; acceptor's must have deficit < 0.
                if surplus.get(donor_frag_idx, 0) <= 0:
                    if logger is not None:
                        logger.debug(
                            f'Linear addition ({label}): frag-fallback '
                            f'inference skipped (S4: donor fragment surplus '
                            f'{surplus.get(donor_frag_idx, 0)} ≤ 0).')
                    return None
                deficit_frags = [fi for fi, d in surplus.items() if d < 0]
                if len(deficit_frags) != 1:
                    if logger is not None:
                        logger.debug(
                            f'Linear addition ({label}): frag-fallback '
                            f'inference skipped (S4: {len(deficit_frags)} '
                            f'deficit fragments).')
                    return None
                acceptor_frag_idx = deficit_frags[0]

    # If S4 narrowed the acceptor fragment, restrict candidates to it.
    if acceptor_frag_idx is not None:
        acceptor_candidates = [i for i in fragments[acceptor_frag_idx]
                                if symbols[i] != 'H']
        if not acceptor_candidates:
            if logger is not None:
                logger.debug(f'Linear addition ({label}): frag-fallback '
                             f'inference skipped (S4: deficit fragment has '
                             f'no heavy atoms).')
            return None

    # ---- S5: local donor/acceptor geometry consistency ----
    sbl_dh = get_single_bond_length(symbols[donor], 'H')
    if not sbl_dh:
        return None
    sbl_dh = float(sbl_dh)
    d_dh = float(np.linalg.norm(post_arr[donor] - post_arr[h_idx]))
    if d_dh > 1.60 * sbl_dh:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S5: d(D,H)={d_dh:.3f} > {1.60 * sbl_dh:.3f}).')
        return None

    # The acceptor is the heavy atom in the acceptor candidate set that
    # is closest to the migrated H — and uniquely so within its fragment.
    cand_arr = np.asarray([post_arr[i] for i in acceptor_candidates], dtype=float)
    h_pos = post_arr[h_idx]
    cand_dists = np.linalg.norm(cand_arr - h_pos, axis=1)
    nearest_idx = int(cand_dists.argmin())
    acceptor = int(acceptor_candidates[nearest_idx])
    nearest_d = float(cand_dists[nearest_idx])

    sbl_ah = get_single_bond_length(symbols[acceptor], 'H')
    if not sbl_ah:
        return None
    sbl_ah = float(sbl_ah)
    if nearest_d > 1.60 * sbl_ah:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S5: d(A,H)={nearest_d:.3f} > '
                         f'{1.60 * sbl_ah:.3f}).')
        return None

    # No rival heavy atom inside the acceptor's fragment may be too
    # close to the migrated H either. This is the "uniquely within
    # its fragment" half of S5.
    if acceptor_frag_idx is not None:
        rival_atoms = [i for i in fragments[acceptor_frag_idx]
                       if symbols[i] != 'H' and i != acceptor]
    else:
        # No S4 narrowing — restrict the rival check to the same
        # fragment the chosen acceptor lives in.
        chosen_frag: Optional[Set[int]] = None
        for frag in fragments:
            if acceptor in frag:
                chosen_frag = frag
                break
        rival_atoms = ([i for i in chosen_frag
                        if symbols[i] != 'H' and i != acceptor]
                       if chosen_frag is not None else [])
    for rival in rival_atoms:
        sbl_rh = get_single_bond_length(symbols[rival], 'H')
        if not sbl_rh:
            continue
        d_rh = float(np.linalg.norm(post_arr[rival] - h_pos))
        if d_rh < 0.95 * float(sbl_rh):
            if logger is not None:
                logger.debug(f'Linear addition ({label}): frag-fallback '
                             f'inference skipped (S5: rival '
                             f'{symbols[rival]}{rival} at d={d_rh:.3f} < '
                             f'{0.95 * float(sbl_rh):.3f}).')
            return None

    return {
        'h_idx': int(h_idx),
        'donor': int(donor),
        'acceptor': int(acceptor),
        'source': 'frag_inferred',
    }
