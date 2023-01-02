# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

#Check if mamba/conda is installed
if [ -x "$(command -v mamba)" ]; then
	echo "mamba is installed."
	COMMAND_PKG=mamba
elif [ -x "$(command -v conda)" ]; then
	echo "conda is installed."
	COMMAND_PKG=conda
else
    echo "mamba and conda are not installed. Please download and install mamba or conda - we strongly recommend mamba"
fi

# clone the repo in the parent directory
echo "Creating the Sella environment..."
source ~/.bashrc
$COMMAND_PKG create -n sella_env python=3.7 -y
conda activate sella_env
$COMMAND_PKG  install -c conda-forge xtb-python -y
$COMMAND_PKG  install -c conda-forge pyyaml -y
$COMMAND_PKG  install -c anaconda pandas -y
$COMMAND_PKG  install -c conda-forge ase -y
$COMMAND_PKG  install -c conda-forge ncurses
$COMMAND_PKG  install -c anaconda pip -y
$CONDA_BASE/envs/sella_env/bin/pip install sella
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'"$CONDA_BASE"'/envs/sella_env/lib' >> ~/.bashrc
source ~/.bashrc
echo "Done installing Sella."
