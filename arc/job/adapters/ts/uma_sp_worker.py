#!/usr/bin/env python
"""
UMA single-point energies for a directory of xyz files. Runs INSIDE ``uma_env``.

This is a standalone worker invoked by the qa_sn2 heuristics TS-guess pre-filter
(``arc/job/adapters/ts/heuristics.py``) via the ``UMA_PYTHON`` interpreter, because
``fairchem`` cannot be imported in ARC's own environment. It mirrors the UMA
single-point path of ``arc/job/adapters/scripts/ase_script.py`` (fairchem-core
``uma-s-1p1``, ``omol`` task, gas phase; total charge and spin conditioned on the
``ase.Atoms`` via ``atoms.info``).

Usage:
    UMA_PYTHON uma_sp_worker.py <xyz_dir> <charge> <multiplicity> [model] [device]

Reads every ``*.xyz`` in ``<xyz_dir>``, runs a UMA single point at the given
charge/spin, and prints a JSON map ``{filename_stem: energy_eV}`` to stdout.
"""
import glob
import json
import os
import sys

from ase.io import read
from fairchem.core import FAIRChemCalculator, pretrained_mlip


def main():
    xyz_dir = sys.argv[1]
    charge = int(sys.argv[2])
    multiplicity = int(sys.argv[3])
    model = sys.argv[4] if len(sys.argv) > 4 else 'uma-s-1p1'
    device = sys.argv[5] if len(sys.argv) > 5 else 'cpu'
    task = 'omol'

    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    calc = FAIRChemCalculator(predictor, task_name=task)

    energies = {}
    for f in sorted(glob.glob(os.path.join(xyz_dir, '*.xyz'))):
        stem = os.path.splitext(os.path.basename(f))[0]
        atoms = read(f)
        atoms.info.update({'charge': charge, 'spin': multiplicity})  # UMA (omol) conditions on these
        atoms.calc = calc
        energies[stem] = float(atoms.get_potential_energy())

    print(json.dumps(energies))


if __name__ == '__main__':
    main()
