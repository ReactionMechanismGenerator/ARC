# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda create -n xtb_env python=3.8 -y
conda activate xtb_env
conda install -c conda-forge xtb -y
conda install -c anaconda pyyaml -y
conda deactivate
