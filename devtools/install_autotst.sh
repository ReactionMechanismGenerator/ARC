# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Check for package manager
if command -v mamba 2>/dev/null; then
    COMMAND_PKG=mamba
elif command -v conda >/dev/null 2>&1; then
    COMMAND_PKG=conda
else
    echo "Error: mamba or conda is not installed. Please download and install mamba or conda - we strongly recommend mamba"
    exit 1
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
conda activate tst_env
$COMMAND_PKG install -c anaconda yaml -y

# Restore the original directory
cd ../ARC || exit
echo "Done installing AutoTST."
popd || exit
