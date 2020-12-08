"""
A GNN to predict TS guesses for isomerization reactions.
The original network was published using TensorFlow 1.4 and Python 2.7
Citation: Pattanaik, L.; Ingraham, J.; Grambow, C.; Green, W. H. 2020.
This file uses `ts_gen_v2`, which uses the same architecture as the original network
translated into PyTorch Geometric and Python 3.7

To use this network
1) clone `ts_gen_v2` repo into the same level as ARC via
`git clone https://github.com/kspieks/ts_gen_v2.git`

2) Add `ts_gen_v2` to your PYTHONPATH

3) Install PyTorch Geometric in the `arc_env` via
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
Alternatively, run `bash devtools/create_env.sh` from ARC's home directory
to install PyTorch Geometric.

"""

import copy
import os
import torch
from torch_geometric.data import DataLoader
import yaml

from arc.exceptions import TSError

from model.G2C import G2C
from model.common import ts_gen_v2_path
from features.arc_featurization import featurization


def gcn_isomerization(reactions):
    """
    Generates TS guesses using the updated graph convolutional network originally published by Pattanaik et al.

    Args:
        reactions:

    Returns:
         ts_xyz_dict_list (List[dict]): entries are ARC xyz dictionaries of TS guesses
    """

    # create torch data loader
    data_list = list()
    ts_xyz_dict_list = list()
    for reaction in reactions:
        # check that this is an isomerization reaction i.e. only one reactant and one product
        num_reactants = len(reaction.r_species)
        num_products = len(reaction.p_species)
        if num_reactants > 1:
            raise TSError(f'Error while using GCN with reaction: {reaction.label}'
                          f'Isomerization reactions must have only 1 reactant.')
        if num_products > 1:
            raise TSError(f'Error while using GCN with reaction: {reaction.label}'
                          f'Isomerization reactions must have only 1 product.')

        #todo: ensure that product atoms are mapped onto reactant atoms before featurizing the reaction
        # might use: https://github.com/ReactionMechanismGenerator/ARC/blob/job_gsm/arc/reaction.py#L707

        data = featurization(reaction)
        data_list.append(data)

        # copy the reactant xyz dictionary since the atom symbols and isotopes will be identical in the TS
        # the TS xyz coordinates will be updated updated with those output by the GCN
        ts_xyz_dict = copy.deepcopy(reaction.r_species[0].get_xyz())
        ts_xyz_dict_list.append(ts_xyz_dict)

    loader = DataLoader(data_list, batch_size=16)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define paths to model parameters and state dictionary
    yaml_file_name = os.path.join(ts_gen_v2_path, 'best_model', 'model_parameters.yml')
    state_dict = os.path.join(ts_gen_v2_path, 'best_model', 'best_model')

    # create the network with the best architecture from hyperopt and load the corresponding best weights
    with open(yaml_file_name, 'r') as f:
        content = yaml.load(stream=f, Loader=yaml.FullLoader)
    model = G2C(**content).to(device)
    model.load_state_dict(torch.load(state_dict, map_location=device))
    model.eval()

    for i, data in enumerate(loader):
        data = data.to(device)
        out, mask = model(data)  # out is distance matrix. mask is matrix of 1s with 0s along diagonal
        # the model modifies the data object by adding the coords attribute, which stores the xyz coordinates
        # shape of data.coords is (batch_size, n_atoms, 3). Each batch entry stores 1 TS guess
        for batch in data.coords:
            coords = batch.double().cpu().detach().numpy().tolist()
            ts_guess_coords = tuple()
            for atom in coords:
                # unpack coordinates from list and convert to tuple
                x, y, z = atom
                coord = (x, y, z)
                ts_guess_coords += (coord,)
            ts_xyz_dict_list[i]['coords'] = ts_guess_coords

    return ts_xyz_dict_list

