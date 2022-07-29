# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda create -n tani_env python=3.8 -y
conda activate tani_env
conda install -c conda-forge torchani -y
conda install -c conda-forge qcelemental -y
conda install -c anaconda yaml -y
conda deactivate
