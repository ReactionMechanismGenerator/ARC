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

# temporarily change directory to install software, and move one directory up in the tree
pushd .
cd ..

# clone the repo in the parent directory
echo "Cloning/Updating AutoTST..."
if [ -d "./AutoTST" ]; then
    cd AutoTST
else
    git clone https://github.com/ReactionMechanismGenerator/AutoTST
    cd AutoTST || exit
fi
git fetch origin
git checkout main
git pull origin main

# Add to PYTHONPATH
echo "Adding AutoTST to PYTHONPATH..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo 'export PYTHONPATH=$PYTHONPATH:'"$(pwd)" >> ~/.bashrc
echo $PYTHONPATH

# create the environment
echo "Creating the AutoTST environment..."
$COMMAND_PKG env create -f environment.yml
# Activate the environment
if [ "$COMMAND_PKG" == "micromamba" ]; then
    micromamba activate tst_env
else
    conda activate tst_env
fi

$COMMAND_PKG install -c anaconda yaml -y

# Restore the original directory
cd ../ARC || exit
echo "Done installing AutoTST."
popd || exit
