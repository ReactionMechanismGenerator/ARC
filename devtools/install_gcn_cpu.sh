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

# temporarily change directory to install software
pushd .
cd ..

# clone the repo in the parent directory and update it
echo "Cloning/Updating GCN..."
if [ -d "./TS-GCN" ]; then
    cd TS-GCN
else
git clone https://github.com/ReactionMechanismGenerator/TS-GCN
    cd TS-GCN || exit
fi
git fetch origin
git checkout main
git pull origin main

# Add to PYTHONPATH
echo "Adding GCN to PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo 'export PYTHONPATH=$PYTHONPATH:'"$(pwd)" >> ~/.bashrc
echo $PYTHONPATH

# create the environment
echo "Creating the GCN-cpu environment..."
source ~/.bashrc

bash devtools/create_env_cpu.sh

# Activate the environment
if [ "$COMMAND_PKG" == "micromamba" ]; then
    micromamba activate ts_gcn
else
    conda activate ts_gcn
fi

$COMMAND_PKG env update -f devtools/cpu_environment.yml

# Restore the original directory
cd ../ARC || exit
echo "Done installing GCN-cpu."
popd || exit
