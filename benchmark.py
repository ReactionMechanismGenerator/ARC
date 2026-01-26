#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark old vs new RDKit force-field optimization paths in ARC.

- Embeds conformers once (shared for both paths).
- Times rdkit_force_field (new batch path) vs old_rdkit_force_field (legacy loop).
- Reports mean/median/min/max, per-conformer cost, and simple sanity checks.

Usage examples are at the end of this file or run:  python bench_ff.py -h
"""

import argparse
import gc
import statistics as stats
import time
from typing import Tuple

from rdkit import Chem

# ARC imports
import arc.species.conformers as conformers
from arc.species.species import ARCSpecies


def _time_once(fn, *args, **kwargs) -> Tuple[float, Tuple[list, list]]:
    """Time a single call and return (elapsed_seconds, (xyzs, energies))."""
    gcold = gc.isenabled()
    try:
        gc.disable()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
    finally:
        if gcold:
            gc.enable()
    return dt, out


def _fmt_s(x: float) -> str:
    return f"{x:.6f}s"


def _fmt_ms(x: float) -> str:
    return f"{x*1e3:.3f} ms"


def run_benchmark(
    smiles: str,
    nconfs: int,
    repeats: int,
    force_field: str,
    optimize: bool,
    try_ob: bool,
) -> None:
    label = "BENCH"
    spc = ARCSpecies(label=label, smiles=smiles)
    # Embed once to ensure both paths see identical starting coordinates
    rd_mol = conformers.embed_rdkit(label=label, mol=spc.mol, num_confs=nconfs, xyz=None)

    # Warm-up (JIT, import caches, allocator warm, etc.)
    _time_once(conformers.rdkit_force_field, label, rd_mol, spc.mol, None, force_field, True, optimize, try_ob)
    _time_once(conformers.old_rdkit_force_field, label, rd_mol, spc.mol, None, force_field, True, optimize, try_ob)

    # Measure
    new_times, old_times = [], []
    new_last, old_last = None, None

    for _ in range(repeats):
        dt_new, new_last = _time_once(
            conformers.rdkit_force_field,
            label, rd_mol, spc.mol, None, force_field, True, optimize, try_ob
        )
        new_times.append(dt_new)

        dt_old, old_last = _time_once(
            conformers.old_rdkit_force_field,
            label, rd_mol, spc.mol, None, force_field, True, optimize, try_ob
        )
        old_times.append(dt_old)

    # Unpack last results
    new_xyzs, new_energies = new_last
    old_xyzs, old_energies = old_last

    # Reporting
    print("\n=== ARC RDKit FF Benchmark ===")
    print(f"SMILES:         {smiles}")
    print(f"Conformers:     {nconfs}")
    print(f"Repeats:        {repeats}")
    print(f"Force field:    {force_field}")
    print(f"Optimize:       {optimize}")
    print(f"Try OpenBabel:  {try_ob}")
    print()

    def summary(times):
        return {
            "mean": stats.mean(times),
            "median": stats.median(times),
            "min": min(times),
            "max": max(times),
            "per_conf_mean": stats.mean(times) / max(1, nconfs),
        }

    s_new = summary(new_times)
    s_old = summary(old_times)

    print("New path: rdkit_force_field (batch MMFF)")
    print(f"  mean:   {_fmt_s(s_new['mean'])}   ({_fmt_ms(s_new['per_conf_mean'])}/conf)")
    print(f"  median: {_fmt_s(s_new['median'])}")
    print(f"  min:    {_fmt_s(s_new['min'])}")
    print(f"  max:    {_fmt_s(s_new['max'])}")

    print("\nOld path: old_rdkit_force_field (per-conf loop)")
    print(f"  mean:   {_fmt_s(s_old['mean'])}   ({_fmt_ms(s_old['per_conf_mean'])}/conf)")
    print(f"  median: {_fmt_s(s_old['median'])}")
    print(f"  min:    {_fmt_s(s_old['min'])}")
    print(f"  max:    {_fmt_s(s_old['max'])}")

    # Basic sanity checks
    print("\n--- Sanity checks ---")
    print(f"New returned: {len(new_xyzs)} xyzs, {len(new_energies)} energies")
    print(f"Old returned: {len(old_xyzs)} xyzs, {len(old_energies)} energies")

    if not optimize:
        print("Expect energies empty with optimize=False:")
        print(f"  new energies empty? {len(new_energies) == 0}")
        print(f"  old energies empty? {len(old_energies) == 0}")
    else:
        # Compare energy arrays shape, not values (values may differ due to tiny numerical drift/order)
        same_len = len(new_energies) == len(old_energies) == nconfs
        print(f"Energies length match nconfs? {same_len}")
        if same_len and len(new_energies) and len(old_energies):
            # Report a quick aggregate difference to spot gross issues
            try:
                import numpy as np
                diff = np.nanmean(np.abs(np.array(new_energies) - np.array(old_energies)))
                print(f"Mean |ΔE| (new-old): {diff:.6g}")
            except Exception:
                pass

    # Simple speed ratio
    if s_old["mean"] > 0:
        ratio = s_old["mean"] / s_new["mean"]
        print(f"\nSpeedup (old/new mean): {ratio:.2f}×")


def main():
    p = argparse.ArgumentParser(description="Benchmark ARC RDKit force-field paths (old vs new).")
    p.add_argument("--smiles", required=True, help="Input molecule as SMILES.")
    p.add_argument("--nconfs", type=int, default=50, help="Number of conformers to embed.")
    p.add_argument("--repeats", type=int, default=3, help="Number of timing repeats per path.")
    p.add_argument("--ff", default="MMFF94s", help="Force field variant (MMFF94 or MMFF94s).")
    p.add_argument("--no-opt", action="store_true", help="Disable optimization (just read coords).")
    p.add_argument("--try-ob", action="store_true", help="Allow OpenBabel fallback if RDKit fails.")
    args = p.parse_args()

    run_benchmark(
        smiles=args.smiles,
        nconfs=args.nconfs,
        repeats=args.repeats,
        force_field=args.ff,
        optimize=not args.no_opt,
        try_ob=args.try_ob,
    )


if __name__ == "__main__":
    main()
