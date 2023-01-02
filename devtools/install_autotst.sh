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

# temporarily change directory to install software, and move one directory up in the tree
pushd .
cd ..

# clone the repo in the parent directory
echo "Cloning/Updating AutoTST..."
git clone https://github.com/ReactionMechanismGenerator/AutoTST
cd AutoTST || exit
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
source ~/.bashrc
$COMMAND_PKG env create -f environment.yml
conda activate tst_env
$COMMAND_PKG install -c anaconda yaml -y
source ~/.bashrc

# Restore the original directory
cd ../ARC || exit
echo "Done installing AutoTST."
popd || exit
