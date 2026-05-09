#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to run GoFlow inference on a single reaction and emit
TS guesses as a YAML file consumable by ARC's GoFlowAdapter.

This script must be invoked from inside the ``goflow_env`` conda environment
(it imports ``goflow``, ``hydra``, ``torch``, ``torch_geometric`` and
``torchdiffeq``). The parent ARC process shells out to it via
``subprocess.run`` so that ARC's main env does not have to carry the heavy
ML dependency stack.

Architecture note
-----------------
GoFlow Lean ships no single-reaction inference CLI (the closest is ``test_save_all_samples_rdb7.sh``,
which runs the entire RDB7 test split via Hydra). This script:

  1. Loads ``feat_dict_organic.pkl`` and derives the model's atom-feature
     dimension (``n_atom_rdkit_feats = sum(len(v) for v in feat_dict.values())``).
     This is necessary because ``configs/model/flow.yaml`` defaults to 27 but
     the lean repo's own training script overrides to 36 — using the wrong
     value causes a silent ``state_dict`` shape mismatch on load.
  2. Composes Hydra config ``train.yaml`` programmatically with overrides
     ``model=flow``, ``data=rdb7``, ``model.representation.n_atom_rdkit_feats=<derived>``,
     ``model.num_samples=<n>``, ``model.num_steps=<k>``, ``model.sample_method=gaussian``.
  3. Instantiates the FlowModule via Hydra and loads the checkpoint's raw
     ``state_dict`` with ``strict=True``. Validates the checkpoint is a real
     Lightning ckpt (not the 45-byte LFS-pointer placeholder).
  4. Builds a single-reaction PyG ``Data`` via ``goflow.preprocessing.generate_graph_data``,
     using the atom-mapped reactant + product SMILES that ARC produced.
     Sets ``pos_gt`` to the reactant geometry only as a length-N placeholder
     (goflow's ``CountNodesPerGraph`` reads ``len(data.pos)``); since we
     never call ``test_step``, the GT-alignment branch is never triggered.
  5. Runs a custom in-script ODE sampling loop (mirroring only the sampling
     part of ``FlowModule.test_step``, NOT the substruct-match/Kabsch align
     of samples to GT). Yields one geometry per sample.
  6. Writes a multi-frame XYZ + a list-of-TSGuess-dicts YAML.

If GoFlow fails to produce any usable output, the script writes a list with
a single failed-guess entry instead of raising — the parent adapter then
logs the failure but continues running other TS methods.

Input file (``input.yml``) — required keys
-------------------------------------------
    reactant_xyz_path : str   absolute path to a plain XYZ file
    product_xyz_path  : str   absolute path to the matching product XYZ
    reactant_smiles   : str   atom-mapped SMILES (every H explicit)
    product_smiles    : str   ditto, map numbers consistent with reactant
    goflow_repo_path  : str   absolute path to the goflow_lean source checkout
    ckpt_path         : str   absolute path to the pretrained ckpt
    feat_dict_path    : str   absolute path to feat_dict_organic.pkl
    output_xyz_path   : str   absolute path for the multi-frame XYZ output
    yml_out_path      : str   absolute path for the parsed TSGuess list

Optional keys (with defaults):
    n_samples: int  default 10
    num_steps: int  default 25
    device   : str  default 'auto'

Output (``yml_out_path``)
-------------------------
A YAML *list* of TSGuess dictionaries. Each entry has:
    method           : 'GoFlow'
    method_direction : 'F'
    method_index     : int      (0-based sample index)
    initial_xyz      : str      (XYZ-format coordinate block, no header lines)
    success          : bool
    execution_time   : str      (str(datetime.timedelta))
"""

import argparse
import datetime
import os
import pickle
import sys
import traceback
from typing import List, Optional

import yaml


def read_xyz_positions(xyz_path: str):
    """
    Parse a single-frame plain XYZ file into an (N, 3) coordinate array.

    The file is expected to start with an atom-count line, then a comment
    line, then N coordinate lines of the form ``<symbol> <x> <y> <z>``.
    Leading blank lines are tolerated; trailing rows are not required to
    match the count exactly (extra rows are ignored).

    Args:
        xyz_path (str): Path to a plain XYZ file.

    Returns:
        numpy.ndarray: An ``(N, 3)`` float32 array of Cartesian coordinates (atomic symbols are dropped).

    Raises:
        ValueError: If the file is empty, the header declares more atoms than are present, or any row is malformed.
    """
    import numpy as np
    with open(xyz_path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f]
    # Skip leading blank lines.
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        raise ValueError(f'XYZ file is empty: {xyz_path}')
    n_atoms = int(lines[i].strip())
    i += 2  # skip count + comment line
    coords: List[List[float]] = []
    for _ in range(n_atoms):
        if i >= len(lines):
            raise ValueError(f'XYZ file {xyz_path} is truncated: header declares {n_atoms} '
                             f'atoms but only {len(coords)} coordinate rows are present.')
        parts = lines[i].split()
        if len(parts) < 4:
            raise ValueError(f'Malformed XYZ row in {xyz_path} at line {i + 1}: '
                             f'expected `<symbol> <x> <y> <z>`, got {lines[i]!r}')
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        i += 1
    return np.asarray(coords, dtype=np.float32)


def format_xyz_block(symbols, pos) -> str:
    """
    Return a body-only XYZ coordinate block, one atom per line.

    No leading count or comment header is emitted — just N rows of ``<symbol> <x> <y> <z>``.
    Coordinates are formatted to 6 decimal places.

    Args:
        symbols (Iterable[str]): Atomic-symbol strings, length N.
        pos (Iterable[Sequence[float]]): N triples of Cartesian coordinates (each a 3-element ``(x, y, z)`` sequence).

    Returns:
        str: Multi-line XYZ block, no trailing newline.
    """
    rows = []
    for sym, (x, y, z) in zip(symbols, pos):
        rows.append(f'{sym} {float(x):.6f} {float(y):.6f} {float(z):.6f}')
    return '\n'.join(rows)


def write_multi_frame_xyz(path: str, symbols, pos_S_N_3) -> None:
    """
    Write a multi-frame XYZ file (one frame per GoFlow sample).

    Each frame is laid out as::

        <n_atoms>
        GoFlow sample <i>
        <symbol> <x> <y> <z>
        ...

    Args:
        path (str): Output file path. Overwritten if it exists.
        symbols (Sequence[str]): Atomic-symbol strings, length N.
        pos_S_N_3 (Iterable[Sequence[Sequence[float]]]): S frames, each an N×3 coordinate iterable.
    """
    n_atoms = len(symbols)
    with open(path, 'w') as f:
        for i, frame in enumerate(pos_S_N_3):
            f.write(f'{n_atoms}\n')
            f.write(f'GoFlow sample {i}\n')
            f.write(format_xyz_block(symbols, frame))
            f.write('\n')


def _failed_guess(elapsed: datetime.timedelta, index: int = 0) -> dict:
    """
    Build the standard failed-TSGuess sentinel dict.

    Returned to the parent ARC adapter when GoFlow inference raises so the
    adapter can mark the attempt as unsuccessful without losing track of
    the elapsed time.

    Args:
        elapsed (datetime.timedelta): Wall-clock time spent before failure.
        index (int): Sample index to record. Defaults to 0.

    Returns:
        dict: A TSGuess-shaped dict with ``success=False`` and ``initial_xyz=None``.
    """
    return {'method': 'GoFlow',
            'method_direction': 'F',
            'method_index': index,
            'initial_xyz': None,
            'success': False,
            'execution_time': str(elapsed)}


def string_representer(dumper, data):
    """
    Represent a Python ``str`` as a YAML scalar.

    Multi-line strings (e.g. an XYZ coordinate block stored under ``initial_xyz``)
    get the literal-block ``|`` style so they round-trip cleanly; single-line strings get the default style.

    Args:
        dumper (yaml.Dumper): The YAML dumper invoking the representer.
        data (str): The string being serialized.

    Returns:
        yaml.ScalarNode: The representer node.
    """
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)


def save_yaml_file_local(path: str, content) -> None:
    """
    Serialize ``content`` to ``path`` as YAML using the safe dumper.

    Multi-line strings are written using the literal-block ``|`` style (see :func:`string_representer`).
    The representer is registered on ``yaml.SafeDumper`` only, leaving the global default dumper untouched.

    Args:
        path (str): Output file path. Overwritten if it exists.
        content: Any YAML-serializable Python object (typically a list/dict).

    Returns:
        None
    """
    yaml.add_representer(str, string_representer, Dumper=yaml.SafeDumper)
    with open(path, 'w') as f:
        f.write(yaml.safe_dump(data=content))


def read_yaml_file_local(path: str) -> dict:
    """
    Read a YAML file using the safe loader.

    Args:
        path (str): Path to a YAML file.

    Returns:
        dict: The loaded mapping (or whatever top-level type ``yaml.safe_load`` returned,
              typically a ``dict`` for our input.yml schema).
    """
    with open(path, 'r') as f:
        return yaml.safe_load(stream=f)


def _resolve_device(requested: str) -> str:
    """
    Pick a concrete torch device string.

    ``'auto'`` defers to ``torch.cuda.is_available()`` (returns ``'cuda'`` if available, ``'cpu'`` otherwise).
    Any explicit value (``'cpu'``, ``'cuda'``, ``'cuda:1'``, …) is honored as-is.

    Args:
        requested (str): Either ``'auto'`` or a literal torch device string.

    Returns:
        str: The resolved torch device string.
    """
    if requested != 'auto':
        return requested
    try:
        import torch
    except ImportError:
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _validate_ckpt(ckpt_path: str) -> None:
    """
    Verify the checkpoint at ``ckpt_path`` is plausibly real.

    Three guards: file exists, size ≥ 1 MB (rejects the 45-byte LFS-pointer placeholder shipped in goflow_lean@main),
    and ``torch.load`` returns a dict containing a ``'state_dict'`` key (rejects malformed pickles or
    non-Lightning checkpoints).

    Args:
        ckpt_path (str): Path to the checkpoint file.

    Raises:
        FileNotFoundError: If ``ckpt_path`` does not exist.
        ValueError: If the file is too small or not a Lightning-style ckpt.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'GoFlow checkpoint not found: {ckpt_path}')
    if os.path.getsize(ckpt_path) < 1_000_000:
        raise ValueError(
            f'GoFlow checkpoint is suspiciously small ({os.path.getsize(ckpt_path)} bytes) '
            f'at {ckpt_path}. The 45-byte file shipped in goflow_lean@main is an LFS '
            f'pointer; set ARC_GOFLOW_CKPT to a real Lightning ckpt.'
        )
    import torch
    # weights_only=False: Lightning ckpts embed an omegaconf.DictConfig in
    # 'hyper_parameters' which PyTorch 2.6+'s safe-by-default unpickler refuses.
    # We trust the source (a user-supplied or self-trained ckpt that already
    # passed the size check above).
    obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if not isinstance(obj, dict) or 'state_dict' not in obj:
        raise ValueError(f'GoFlow checkpoint at {ckpt_path} is not a Lightning ckpt '
                         f'(missing "state_dict" key). Got type={type(obj).__name__}.')


def run_goflow_inference(input_dict: dict) -> List[dict]:
    """
    Run flow-matching ODE sampling on a single reaction.

    Loads the pretrained GoFlow model (Hydra-composed FlowModule + ckpt
    state_dict), builds a single-reaction PyG ``Data`` from the atom-mapped
    SMILES, and runs ``num_steps`` Euler ODE steps to draw ``n_samples``
    TS-geometry samples. Never propagates exceptions to the caller — any
    failure produces a single sentinel entry with ``success=False`` so the
    parent adapter can log and continue.

    Args:
        input_dict (dict): Parsed ``input.yml`` payload.
                           Required keys:
                           ``reactant_xyz_path``, ``product_xyz_path``, ``reactant_smiles``, ``product_smiles``,
                           ``goflow_repo_path``, ``ckpt_path``, ``feat_dict_path``, ``output_xyz_path``, ``yml_out_path``.
                           Optional keys: ``n_samples`` (default 10), ``num_steps`` (default 25), ``device`` (default ``'auto'``).

    Returns:
        list[dict]: One TSGuess-shaped dict per sample (or a single failure sentinel if the pipeline raised).
                    Each entry has keys ``method``,  ``method_direction``, ``method_index``, ``initial_xyz``,
                    ``success``, ``execution_time``.
    """
    t0 = datetime.datetime.now()
    try:
        # Late imports: this function only runs inside goflow_env.
        import torch
        import numpy as np
        from torch_geometric.data import Batch
        from torchdiffeq import odeint
        from hydra import initialize_config_dir, compose
        from hydra.utils import instantiate
        from ase.data import chemical_symbols

        from goflow.preprocessing import generate_graph_data
        from goflow.gotennet.data.components.utils import CountNodesPerGraph

        _validate_ckpt(input_dict['ckpt_path'])
        with open(input_dict['feat_dict_path'], 'rb') as f:
            feat_dict = pickle.load(f)
        feat_dim = sum(len(v) for v in feat_dict.values())

        n_samples = int(input_dict.get('n_samples', 10))
        num_steps = int(input_dict.get('num_steps', 25))
        device = _resolve_device(input_dict.get('device', 'auto'))

        cfg_dir = os.path.join(input_dict['goflow_repo_path'], 'src', 'goflow', 'configs')
        with initialize_config_dir(config_dir=cfg_dir, version_base='1.3'):
            cfg = compose(config_name='train',
                          overrides=['model=flow',
                                     'data=rdb7',
                                     f'model.representation.n_atom_rdkit_feats={feat_dim}',
                                     f'model.num_samples={n_samples}',
                                     f'model.num_steps={num_steps}',
                                     'model.sample_method=gaussian'],
                          )
        flow_module = instantiate(cfg.model)

        ckpt = torch.load(input_dict['ckpt_path'], map_location='cpu', weights_only=False)
        flow_module.load_state_dict(ckpt['state_dict'], strict=True)
        flow_module = flow_module.to(device).eval()

        pos_r = read_xyz_positions(input_dict['reactant_xyz_path'])
        # goflow's CountNodesPerGraph transform reads len(data.pos) to set
        # num_nodes, so pos_gt must be a tensor of shape (N, 3) — None breaks it.
        # We pass pos_r purely as a placeholder of the right shape; FlowModule's
        # test_step (which would Kabsch-align samples to data.pos) is never
        # invoked because our _ode_sample helper drives the ODE directly.
        data = generate_graph_data(r_smiles=input_dict['reactant_smiles'],
                                   p_smiles=input_dict['product_smiles'],
                                   pos_guess=pos_r,
                                   pos_gt=pos_r,
                                   feat_dict=feat_dict,
                                   )
        data = CountNodesPerGraph()(data)
        batch = Batch.from_data_list([data]).to(device)

        n_nodes = batch.num_nodes
        seed_base = getattr(flow_module, 'seed', 1) or 1

        t_T = torch.linspace(0, 1, steps=num_steps, device=device)

        def ode_func(t, x_t_N_3):
            t_G = torch.tensor([t] * batch.num_graphs, device=device)
            return flow_module.model_output(x_t_N_3, batch, t_G)

        out_S_N_3 = torch.zeros((n_samples, n_nodes, 3), device=device)
        with torch.no_grad():
            for i in range(n_samples):
                torch.manual_seed(seed_base + i)
                x0 = torch.randn(n_nodes, 3, device=device)
                out_S_N_3[i] = odeint(ode_func, x0, t_T, method='euler')[-1]

        pos_S_N_3 = out_S_N_3.cpu().numpy()
        symbols = [chemical_symbols[int(z)] for z in batch.atom_type.cpu().numpy()]
        write_multi_frame_xyz(input_dict['output_xyz_path'], symbols, pos_S_N_3)

        elapsed = datetime.datetime.now() - t0
        return [{'method': 'GoFlow',
                 'method_direction': 'F',
                 'method_index': i,
                 'initial_xyz': format_xyz_block(symbols, pos_S_N_3[i]),
                 'success': True,
                 'execution_time': str(elapsed)}
                for i in range(n_samples)]
    except Exception:
        traceback.print_exc()
        elapsed = datetime.datetime.now() - t0
        return [_failed_guess(elapsed, index=0)]


def parse_command_line_arguments(command_line_args: Optional[list] = None):
    """
    Parse the script's command-line arguments.

    Args:
        command_line_args (list, optional): Override sys.argv (used by tests).
                                            Defaults to ``None`` which reads from ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed flags. Currently exposes only ``yml_in_path`` (path to ``input.yml``).
    """
    parser = argparse.ArgumentParser(description='Run GoFlow to generate TS guesses for an ARC reaction.')
    parser.add_argument('--yml_in_path', metavar='input', type=str, default='input.yml',
                        help='Path to the input YAML file (default: ./input.yml).')
    return parser.parse_args(command_line_args)


def main():
    """
    Script entry point.

    Reads ``input.yml`` (path from ``--yml_in_path``), runs :func:`run_goflow_inference`, and writes the TSGuess list
    to the ``yml_out_path`` declared in the input. Prints a one-line summary to stdout.
    Exits with code 1 if the input file is missing.
    """
    args = parse_command_line_arguments()
    yml_in_path = str(args.yml_in_path)
    if not os.path.isfile(yml_in_path):
        print(f'[goflow_script] input file not found: {yml_in_path}', file=sys.stderr)
        sys.exit(1)
    input_dict = read_yaml_file_local(yml_in_path)

    tsgs = run_goflow_inference(input_dict)
    save_yaml_file_local(path=input_dict['yml_out_path'], content=tsgs)
    n_ok = sum(1 for tsg in tsgs if tsg.get('success'))
    print(f'[goflow_script] wrote {len(tsgs)} TSGuess entries ({n_ok} successful) to {input_dict["yml_out_path"]}',
          flush=True)


if __name__ == '__main__':
    main()
