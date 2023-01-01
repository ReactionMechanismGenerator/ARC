# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

if [ $(mamba) ] > /dev/null 2>&1; then 
	COMMAND_PKG=conda
	echo "conda found"
	else
	COMMAND_PKG=mamba
	echo "mamba found"
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
