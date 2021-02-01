"""
A module for calling TS-GEN

Notes for possible future developments:
- Currently, ARC will only use the network for unimolecular reactions (A --> B). The network was trained on both A --> B
  and A --> B + C reactions. However, the products must be oriented as if they just reacted. They can't be arbitrarily
  placed in 3D space. ARC would have to automatically orient the two molecules correctly before using the network for
  more reaction families.

- The network was trained on geometries optimized at ωB97X-D3 with a def2-TZVP basis set. For now, ARC runs the GCN
  using force field optimized geometries from RDKit. The TS guesses have looked reasonable, though in the future, it may
  be beneficial to use the final QM optimized geometries as input to the network.

- This network is not symmetric so it's possible that reversing the reaction direction could impact the convergence of
  the TS guess. For example, it might be beneficial to call the network on A --> B and on B --> A to generate two TS
  guesses, though the impact on convergence has not been studied.

"""

import os
import subprocess

from arc.common import get_logger
from arc.common import TS_GCN_PATH, TS_GCN_PYTHON
from arc.species.converter import str_to_xyz

logger = get_logger()


def gcn(ts_path,
        ):
    """
    Generates a TS guess using the updated graph convolutional network originally published by Pattanaik et al.
    https://chemrxiv.org/articles/Genereting_Transition_States_of_Isomerization_Reactions_with_Deep_Learning/12302084
    This function writes the TS guess as an xyz file to the corresponding TS directory.

    Args:
        ts_path (str): Path to the inputs for the network.

    Returns:
         ts_xyz_dict (dict): ARC xyz dictionary of TS guess
    """

    # run the GCN as a subprocess
    p = subprocess.run(f'bash -l {{0}} && {TS_GCN_PYTHON} {os.path.join(TS_GCN_PATH, "inference.py")} '
                       f'--r_sdf_path {os.path.join(ts_path, "reactant.sdf")} '
                       f'--p_sdf_path {os.path.join(ts_path, "product.sdf")} '
                       f'--ts_xyz_path {os.path.join(ts_path, "TS.xyz")} ',
                       shell=True)
    # if subprocess ran successfully, read the TS structure into ARC's dictionary format
    if p.returncode == 0:
        ts_xyz_dict = str_to_xyz(os.path.join(ts_path, "TS.xyz"))
    # otherwise, log the error
    else:
        logger.error(f'GCN subprocess did not give a successful return code:\n'
                     f'Got return code: {p.returncode}\n'
                     f'stdout: {p.stdout}\n'
                     f'stderr: {p.stderr}')
        ts_xyz_dict = None

    return ts_xyz_dict
