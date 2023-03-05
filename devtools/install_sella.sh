#!/bin/bash -l

# Check if Micromamba is installed
if [ -x "$(command -v micromamba)" ]; then
    echo "Micromamba is installed."
    COMMAND_PKG=micromamba
# Check if Mamba is installed
elif [ -x "$(command -v mamba)" ]; then
    echo "Mamba is installed."
    COMMAND_PKG=mamba
# Check if Conda is installed
elif [ -x "$(command -v conda)" ]; then
    echo "Conda is installed."
    COMMAND_PKG=conda
else
    echo "Micromamba, Mamba, and Conda are not installed. Please download and install one of them - we strongly recommend Micromamba or Mamba."
    exit 1
fi

# Set up Conda/Micromamba environment
if [ "$COMMAND_PKG" == "micromamba" ]; then
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate base
    BASE=$MAMBA_ROOT_PREFIX
    # shellcheck source=/dev/null
    source "$BASE/etc/profile.d/micromamba.sh"
else
    BASE=$(conda info --base)
    # shellcheck source=/dev/null
    source "$BASE/etc/profile.d/conda.sh"
fi

# clone the repo in the parent directory
echo "Creating the Sella environment..."
$COMMAND_PKG create -n sella_env python=3.7 -c conda-forge -y
# Activate the environment
if [ "$COMMAND_PKG" == "micromamba" ]; then
    micromamba activate sella_env
else
    conda activate sella_env
fi

$COMMAND_PKG  install -c conda-forge xtb-python -y
$COMMAND_PKG  install -c conda-forge pyyaml -y
$COMMAND_PKG  install -c anaconda pandas -y
$COMMAND_PKG  install -c conda-forge ase -y
$COMMAND_PKG  install -c conda-forge ncurses
$COMMAND_PKG  install -c anaconda pip -y
$BASE/envs/sella_env/bin/pip install sella

cd $BASE/envs/sella_env || exit
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

#This sets up the LD_LIBRARY_PATH to include Sella_Env, but only when the environment is active
echo 'export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}' >> $BASE/envs/sella_env/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:'"$BASE"'/envs/sella_env/lib' >> $BASE/envs/sella_env/etc/conda/activate.d/env_vars.sh
#This will reset the LD_LIBRARY_PATH back to the original LD_LIBRARY_PATH when the environment is deactivated
echo 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}' >> $BASE/envs/sella_env/etc/conda/deactivate.d/env_vars.sh
echo 'unset OLD_LD_LIBRARY_PATH' >> $BASE/envs/sella_env/etc/conda/deactivate.d/env_vars.sh

source ~/.bashrc
echo "Done installing Sella."
