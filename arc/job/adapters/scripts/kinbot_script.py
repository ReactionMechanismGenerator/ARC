#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to generate TS guess geometries with KinBot,
should be run under the kinbot_env environment.

This script must only import KinBot, PyYAML, and the standard library
(it runs in an environment where ARC is not installed).

Input (YAML file passed via ``--yml_in_path``), keys:
    charge (int): The well/reaction charge.
    multiplicity (int): The well/reaction multiplicity.
    families (list[str]): The KinBot reaction families to search.
    wells (list[dict]): Entries are dicts with keys:
        direction (str): 'F' or 'R', the direction this well corresponds to.
        smiles (str): A SMILES representation of the well (resonance structure specific).
        structure (list): The cartesian coordinates in the KinBot flat list format,
                          [symbol_1, x_1, y_1, z_1, symbol_2, ...].
    uma (dict, optional): UMA (FAIRChem) refinement options with keys:
        refine (bool): Whether to refine successful TS guesses with a Sella
                       saddle-point search on the UMA potential (default: False).
        model_path (str): A path to a local UMA checkpoint file (e.g., uma-s-1p2.pt).
        model_name (str): A pretrained UMA model name to fetch from HuggingFace
                          (license-gated) if no model_path is given (default: uma-m-1p1).
        task_name (str): The UMA task name (default: omol).
        device (str): The torch device to use (default: cpu).
        fmax (float): The Sella force convergence criterion (default: 0.005).
        steps (int): The maximal number of Sella steps (default: 250).
    yml_out_path (str): The path to which the results are saved.

Output (YAML file written to ``yml_out_path``): a list of TS guess dicts with keys:
    direction (str): 'F' or 'R' (copied from the corresponding well).
    success (bool): Whether KinBot successfully generated a template-modified geometry.
    coords (list[list[float]] or None): The TS guess cartesian coordinates (N x 3).
    execution_time (str): The time it took to generate this guess.
    uma_refined (bool): Whether the coordinates were refined on the UMA potential.
    uma_error (str, optional): Why UMA refinement fell back to the unrefined guess.
    error (str, optional): An error message if the well processing failed.
"""

import argparse
import datetime
import json
import os
import sys
import tempfile
import traceback

import yaml

from kinbot.modify_geom import modify_coordinates
from kinbot.parameters import Parameters
from kinbot.qc import QuantumChemistry
from kinbot.reaction_finder import ReactionFinder
from kinbot.reaction_generator import ReactionGenerator
from kinbot.stationary_pt import StationaryPoint


def set_up_kinbot(smiles: str,
                  structure: list,
                  families: list,
                  multiplicity: int,
                  charge: int,
                  ) -> 'ReactionGenerator':
    """
    Set up KinBot to run for a unimolecular reaction starting from the single well side.

    Args:
        smiles (str): The SMILES representation of the unimolecular well to react.
        structure (list): The cartesian coordinates of the well in the KinBot list format.
        families (list): The specific KinBot families to try.
        multiplicity (int): The well/reaction multiplicity.
        charge (int): The well/reaction charge.

    Returns:
        ReactionGenerator: The KinBot ReactionGenerator instance.
    """
    par_dict = {'title': 'ARC',
                # molecule information
                'smiles': smiles,
                'structure': structure,
                'charge': charge,
                'mult': multiplicity,
                # steps
                'reaction_search': 1,
                'families': families,
                'pes': 0,
                'high_level': 0,
                'conformer_search': 0,
                'me': 0,
                'ringrange': [3, 9],
                # Not used in ARC's flow (no KinBot QC jobs are ever spawned), but
                # Parameters refuses to initialize without a barrier threshold (kcal/mol).
                'barrier_threshold': 200.0,
                }
    # Parameters validates its values at construction time (and calls sys.exit() on
    # invalid ones), so the overrides must be supplied via a JSON input file rather
    # than by mutating params.par after the fact.
    fd, par_path = tempfile.mkstemp(suffix='.json', prefix='kinbot_params_', dir=os.getcwd(), text=True)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(par_dict, f)
        params = Parameters(par_path)
    finally:
        os.remove(par_path)

    well = StationaryPoint(name='well0',
                           charge=charge,
                           mult=multiplicity,
                           structure=structure,
                           )

    well.calc_chemid()
    well.bond_mx()
    well.find_cycle()
    well.find_atom_eqv()
    well.find_conf_dihedral()

    qc = QuantumChemistry(params.par)
    rxn_finder = ReactionFinder(well, params.par, qc)
    rxn_finder.find_reactions()

    reaction_generator = ReactionGenerator(species=well,
                                           par=params.par,
                                           qc=qc,
                                           input_file=None,
                                           )

    return reaction_generator


def apply_kinbot_template(kinbot_rxn) -> tuple:
    """
    Drive a KinBot reaction instance's constraint template over its full step sequence,
    applying the geometry ``change`` constraints of every step to the evolving geometry.

    This mirrors what KinBot's ``reac_family.carry_out_reaction()`` does between the
    constrained-optimization jobs it submits (including the short-instance ``skip``
    shortcut, the possible step bump returned by ``get_constraints``, and the ``'L'``
    change-entry prefix), just without running any QC relaxation in between.
    Steps ``0 .. max_step - 1`` are the template modification steps; at
    ``step == max_step`` KinBot runs a saddle-point search that does not modify the
    template geometry, so the loop stops there.

    Args:
        kinbot_rxn: The KinBot reaction instance (a GeneralReac subclass instance).

    Returns:
        tuple: (success (bool), geom (N x 3 array-like)).
    """
    geom = kinbot_rxn.species.geom
    success = True
    step = 0
    if kinbot_rxn.skip and len(kinbot_rxn.instance) < 4:
        step = 12
    while step < kinbot_rxn.max_step:
        step, fix, change, release = kinbot_rxn.get_constraints(step, geom)

        change_starting_zero = list()
        for c in change:
            if c[0] == 'L':
                c_new = ['L'] + [ci - 1 for ci in c[1:-1]]
            else:
                c_new = [ci - 1 for ci in c[:-1]]
            c_new.append(c[-1])
            change_starting_zero.append(c_new)

        if change_starting_zero and 'frozen' not in kinbot_rxn.instance_name:
            step_success, geom = modify_coordinates(species=kinbot_rxn.species,
                                                    name=kinbot_rxn.instance_name,
                                                    geom=geom,
                                                    changes=change_starting_zero,
                                                    bond=kinbot_rxn.species.bond,
                                                    )
            success = success and bool(step_success)
        step += 1
    return success, geom


def get_ts_guesses_for_well(well: dict,
                            families: list,
                            multiplicity: int,
                            charge: int,
                            ) -> list:
    """
    Generate TS guesses for a single well.

    Args:
        well (dict): The well dictionary with 'direction', 'smiles', and 'structure' keys.
        families (list): The specific KinBot families to try.
        multiplicity (int): The well/reaction multiplicity.
        charge (int): The well/reaction charge.

    Returns:
        list: TS guess dictionaries.
    """
    results = list()
    reaction_generator = set_up_kinbot(smiles=well['smiles'],
                                       structure=well['structure'],
                                       families=families,
                                       multiplicity=multiplicity,
                                       charge=charge,
                                       )
    for kinbot_rxn in reaction_generator.species.reac_obj:
        t0 = datetime.datetime.now()
        success, coords = apply_kinbot_template(kinbot_rxn)
        results.append({'direction': well['direction'],
                        'success': bool(success),
                        'coords': [[float(ci) for ci in coord] for coord in coords] if success else None,
                        'execution_time': str(datetime.datetime.now() - t0),
                        'uma_refined': False,
                        })
    return results


_UMA_PREDICT_UNIT = None


def _get_uma_calculator(uma: dict):
    """
    Lazily import fairchem and build a FAIRChemCalculator on the UMA potential.
    The (expensive to load) predict unit is cached at the module level, so multiple
    guesses within one worker run reuse it.

    Args:
        uma (dict): The UMA options (see the module docstring).

    Returns:
        FAIRChemCalculator: The ASE calculator.
    """
    global _UMA_PREDICT_UNIT
    from fairchem.core import FAIRChemCalculator
    if _UMA_PREDICT_UNIT is None:
        device = uma.get('device') or 'cpu'
        model_path = uma.get('model_path') or ''
        if model_path:
            from fairchem.core.units.mlip_unit import load_predict_unit
            _UMA_PREDICT_UNIT = load_predict_unit(model_path, device=device)
        else:
            # Fetch a pretrained checkpoint from HuggingFace. UMA checkpoints are
            # license-gated (require a HuggingFace login + accepting Meta's license),
            # any failure here is caught by the caller and refinement is skipped.
            from fairchem.core import pretrained_mlip
            _UMA_PREDICT_UNIT = pretrained_mlip.get_predict_unit(uma.get('model_name') or 'uma-m-1p1',
                                                                 device=device)
    return FAIRChemCalculator(_UMA_PREDICT_UNIT, task_name=uma.get('task_name') or 'omol')


def refine_ts_guess_with_uma(symbols: list,
                             coords: list,
                             charge: int,
                             multiplicity: int,
                             uma: dict,
                             label: str,
                             ) -> tuple:
    """
    Refine a TS guess with a Sella saddle-point search (order=1) on the UMA
    (FAIRChem) machine-learned potential, mirroring KinBot 2.3.0's
    ``tpl/ase_fc_ts_end.tpl.py`` template (without KinBot's frequency check —
    ARC validates TS guesses downstream with its own frequency and IRC checks).

    fairchem is imported lazily in here only: the worker must keep functioning in a
    kinbot_env without the fairchem extra installed.

    Args:
        symbols (list): The chemical element symbols.
        coords (list): The initial (template-modified) coordinates, N x 3.
        charge (int): The species charge.
        multiplicity (int): The species multiplicity.
        uma (dict): The UMA options (see the module docstring).
        label (str): A label used for the Sella log file name.

    Returns:
        tuple: (refined coords (N x 3 list) or None, error message (str) or None).
    """
    try:
        from ase import Atoms
        from sella import Sella
        calc = _get_uma_calculator(uma)
    except (ImportError, ModuleNotFoundError) as e:
        return None, (f'UMA refinement was requested, but fairchem is not available in kinbot_env '
                      f'({e}). Install it with "devtools/install_kinbot.sh --uma".')
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return None, f'Could not load the UMA model: {e.__class__.__name__}: {e}'
    try:
        mol = Atoms(symbols=symbols, positions=coords)
        mol.info.update({'charge': charge, 'spin': multiplicity})
        mol.calc = calc
        opt = Sella(mol,
                    order=1,
                    internal=len(symbols) >= 5,  # mirrors KinBot's fc TS templates
                    logfile=f'{label}_uma_sella.log',
                    )
        converged = opt.run(fmax=float(uma.get('fmax') or 0.005),
                            steps=int(uma.get('steps') or 250))
        if not converged:
            return None, 'The Sella saddle-point search on the UMA potential did not converge.'
        return [[float(ci) for ci in coord] for coord in mol.positions], None
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return None, f'UMA refinement failed: {e.__class__.__name__}: {e}'


def run_kinbot(input_dict: dict) -> None:
    """
    Run KinBot for all wells in the input dictionary and save the results.

    Args:
        input_dict (dict): The input dictionary.
    """
    results = list()
    uma = input_dict.get('uma') or dict()
    for well in input_dict['wells']:
        try:
            well_results = get_ts_guesses_for_well(well=well,
                                                   families=input_dict['families'],
                                                   multiplicity=input_dict['multiplicity'],
                                                   charge=input_dict['charge'],
                                                   )
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            results.append({'direction': well.get('direction'),
                            'success': False,
                            'coords': None,
                            'execution_time': None,
                            'uma_refined': False,
                            'error': f'{e.__class__.__name__}: {e}',
                            })
            continue
        if uma.get('refine'):
            symbols = well['structure'][0::4]
            for i, entry in enumerate(well_results):
                if not entry['success']:
                    continue
                refined_coords, uma_error = refine_ts_guess_with_uma(
                    symbols=symbols,
                    coords=entry['coords'],
                    charge=input_dict['charge'],
                    multiplicity=input_dict['multiplicity'],
                    uma=uma,
                    label=f"{well['direction']}_{i}",
                )
                if refined_coords is not None:
                    entry['coords'] = refined_coords
                    entry['uma_refined'] = True
                else:
                    # Fall back to the unrefined template guess, never fail the task.
                    entry['uma_error'] = uma_error
                    print(f"UMA refinement fell back to the unrefined guess for "
                          f"{well['direction']} {i}: {uma_error}", file=sys.stderr)
        results.extend(well_results)
    save_yaml_file(path=input_dict['yml_out_path'], content=results)


def read_yaml_file(path: str):
    """
    Read a YAML file and return its content.

    Args:
        path (str): The YAML file path to read.

    Returns: Union[dict, list]
        The content read from the file.
    """
    with open(path, 'r') as f:
        content = yaml.load(stream=f, Loader=yaml.FullLoader)
    return content


def save_yaml_file(path: str,
                   content: list,
                   ) -> None:
    """
    Save a YAML file.

    Args:
        path (str): The YAML file path to save.
        content (list): The content to save.
    """
    yaml.add_representer(str, string_representer)
    yaml_str = yaml.dump(data=content)
    with open(path, 'w') as f:
        f.write(yaml_str)


def string_representer(dumper, data):
    """
    Add a custom string representer to use block literals for multiline strings.
    """
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.
    This uses the :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """
    parser = argparse.ArgumentParser(description='KinBot')
    parser.add_argument('--yml_in_path', metavar='input', type=str, default='input.yml',
                        help='A path to the YAML input file')
    args = parser.parse_args(command_line_args)
    return args


def main():
    """
    Run KinBot to generate TS guesses.
    """
    args = parse_command_line_arguments()
    yml_in_path = str(args.yml_in_path)
    input_dict = read_yaml_file(yml_in_path)
    # KinBot's QuantumChemistry writes a 'kinbot.db' file (and possibly other
    # scratch files) to the current working directory, direct these to the job folder.
    os.chdir(os.path.dirname(os.path.abspath(input_dict['yml_out_path'])))
    run_kinbot(input_dict=input_dict)


if __name__ == '__main__':
    main()
