# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# clone the repo in the parent directory
echo "Creating the Sella environment..."
source ~/.bashrc
conda create -n sella_env python=3.8 -y
conda activate sella_env
conda install -c conda-forge xtb-python -y
conda install -c conda-forge pyyaml -y
conda install -c anaconda pandas -y
conda install -c conda-forge ase -y
#conda install -c anaconda networkx -y
#conda install -c conda-forge rmsd -y
conda install -c anaconda pip -y
$CONDA_BASE/envs/sella_env/bin/pip install sella
source ~/.bashrc
echo "Done installing Sella."
