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
    CONDA_BASE=$(conda info --base)
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

$COMMAND_PKG create -n xtb_env python=3.7  -c conda-forge -y
# Activate the environment
if [ "$COMMAND_PKG" == "micromamba" ]; then
    micromamba activate xtb_env
else
    conda activate xtb_env
fi

# Install xtb
$COMMAND_PKG install -n xtb_env -c conda-forge  xtb=6.3.3 -y

# Install pyyaml
$COMMAND_PKG install -c anaconda pyyaml -y

$COMMAND_PKG deactivate
