# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Check if mamba or conda is installed
if command -v mamba > /dev/null; then
  COMMAND_PKG=mamba
elif command -v conda > /dev/null; then
  COMMAND_PKG=conda
else
  echo "mamba and conda are not installed. Please download and install mamba or conda - we strongly recommend mamba"
  exit 1
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

cd $CONDA_BASE/envs/sella_env
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

#This sets up the LD_LIBRARY_PATH to include Sella_Env, but only when the environment is active
echo 'export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}' >> $CONDA_BASE/envs/sella_env/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:'"$CONDA_BASE"'/envs/sella_env/lib' >> $CONDA_BASE/envs/sella_env/etc/conda/activate.d/env_vars.sh
#This will reset the LD_LIBRARY_PATH back to the original LD_LIBRARY_PATH when the environment is deactivated
echo 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}' >> $CONDA_BASE/envs/sella_env/etc/conda/deactivate.d/env_vars.sh
echo 'unset OLD_LD_LIBRARY_PATH' >> $CONDA_BASE/envs/sella_env/etc/conda/deactivate.d/env_vars.sh

source ~/.bashrc
echo "Done installing Sella."
