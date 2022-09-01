# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda create -n psi4_env python=3.7 -y
conda activate psi4_env
conda install -c psi4 psi4 -y
