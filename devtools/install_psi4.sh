# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# clone the repo in the parent directory
echo "Creating the psi4 environment..."
source ~/.bashrc
conda create -n p4env psi4 -c psi4 -y
conda update psi4 -c psi4 -y

source ~/.bashrc
echo "Done installing psi4."
