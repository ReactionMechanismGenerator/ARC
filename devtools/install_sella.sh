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

if [ "$COMMAND_PKG" = "micromamba" ]; then
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate base
    BASE=$MAMBA_ROOT_PREFIX
    # Verify if the micromamba profile script exists
    if [ -f "$BASE/etc/profile.d/micromamba.sh" ]; then
        . "$BASE/etc/profile.d/micromamba.sh"
    else
        echo "File not found: $BASE/etc/profile.d/micromamba.sh"
        exit 1
    fi
else
    BASE=$(conda info --base)
    echo "Conda base directory: $BASE"
    # Verify if the conda profile script exists
    if [ -f "$BASE/etc/profile.d/conda.sh" ]; then
        . "$BASE/etc/profile.d/conda.sh"
    else
        echo "File not found: $BASE/etc/profile.d/conda.sh"
        exit 1
    fi
fi

# clone the repo in the parent directory
echo "Creating the Sella environment..."
$COMMAND_PKG env create -f devtools/sella_environment.yml
# Activate the environment
if [ "$COMMAND_PKG" = "micromamba" ]; then
    micromamba activate sella_env
else
    conda activate sella_env
fi

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

. ~/.bashrc
echo "Done installing Sella."
